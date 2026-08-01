"""Aligned finalist residual-mass diagnostics for challenger activation."""

from collections.abc import Mapping

import numpy as np
import pandas as pd

from baram.contracts.hashing import sha256_dataframe
from baram.evaluation.slices import attach_diagnostic_bins
from baram.exceptions import ContractError

_KEYS = ["forecast_id", "forecast_kst_dtm", "group_id"]
_PREDICTION_COLUMNS = [*_KEYS, "actual_kwh", "prediction_kwh"]
_SLICE_COLUMNS = (
    "operating_season",
    "target_hour",
    "lead_bin",
    "nwp_missing_state",
    "wind_speed_bin",
    "actual_generation_bin",
    "ramp_flag",
)


def _canonical_prediction(frame: pd.DataFrame, candidate_id: str) -> pd.DataFrame:
    missing = set(_PREDICTION_COLUMNS) - set(frame)
    if missing:
        raise ContractError(f"finalist {candidate_id} is missing: {sorted(missing)}")
    result = frame[_PREDICTION_COLUMNS].copy()
    result["forecast_kst_dtm"] = pd.to_datetime(result["forecast_kst_dtm"], errors="raise")
    result = result.sort_values(_KEYS, kind="stable").reset_index(drop=True)
    if result.empty or result.duplicated(_KEYS).any():
        raise ContractError(f"finalist {candidate_id} requires nonempty unique keys")
    numeric = result[["actual_kwh", "prediction_kwh"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ContractError(f"finalist {candidate_id} contains non-finite values")
    return result


def shared_failure_slices(
    predictions: Mapping[str, pd.DataFrame],
    context: pd.DataFrame,
    capacities: Mapping[int, float],
    *,
    threshold: float = 0.25,
) -> dict[str, object]:
    """Find identical diagnostic buckets carrying threshold error mass for every finalist."""
    if len(predictions) < 2:
        raise ContractError("shared failure analysis requires at least two finalists")
    if not np.isfinite(threshold) or threshold <= 0.0 or threshold > 1.0:
        raise ContractError("failure-slice threshold must be in (0, 1]")
    required_context = {*_KEYS, "lead_hour", "wind_speed", "nwp_missing_count"}
    if not required_context.issubset(context):
        raise ContractError(
            f"failure context is missing: {sorted(required_context - set(context))}"
        )
    aligned_context = context[[*required_context]].copy()
    aligned_context["forecast_kst_dtm"] = pd.to_datetime(
        aligned_context["forecast_kst_dtm"], errors="raise"
    )
    aligned_context = aligned_context.sort_values(_KEYS, kind="stable").reset_index(drop=True)
    if aligned_context.empty or aligned_context.duplicated(_KEYS).any():
        raise ContractError("failure context requires nonempty unique keys")

    canonical = {
        candidate_id: _canonical_prediction(frame, candidate_id)
        for candidate_id, frame in sorted(predictions.items())
    }
    first = next(iter(canonical.values()))
    if not first[_KEYS].equals(aligned_context[_KEYS]):
        raise ContractError("finalist and failure-context keys are not aligned")
    for candidate_id, frame in canonical.items():
        if not frame[_KEYS].equals(first[_KEYS]) or not frame[[*_KEYS, "actual_kwh"]].equals(
            first[[*_KEYS, "actual_kwh"]]
        ):
            raise ContractError(f"finalist {candidate_id} keys/labels are not aligned")

    per_candidate: dict[str, dict[str, dict[str, float]]] = {}
    for candidate_id, frame in canonical.items():
        diagnostic = frame.merge(
            aligned_context,
            on=_KEYS,
            how="left",
            validate="one_to_one",
        )
        diagnostic = attach_diagnostic_bins(diagnostic, capacities)
        diagnostic["absolute_error"] = (
            diagnostic["actual_kwh"] - diagnostic["prediction_kwh"]
        ).abs()
        total_error = float(diagnostic["absolute_error"].sum())
        if not np.isfinite(total_error) or total_error <= 0.0:
            raise ContractError(f"finalist {candidate_id} has no positive finite error mass")
        slices: dict[str, dict[str, float]] = {}
        for column in _SLICE_COLUMNS:
            values = diagnostic[column].astype(str)
            for value in sorted(values.unique()):
                mask = values.eq(value)
                slice_id = f"{column}={value}"
                mass = float(diagnostic.loc[mask, "absolute_error"].sum() / total_error)
                slices[slice_id] = {
                    "error_mass_fraction": mass,
                    "row_fraction": float(mask.mean()),
                }
        per_candidate[candidate_id] = slices

    common_ids = set.intersection(*(set(items) for items in per_candidate.values()))
    records: list[dict[str, object]] = []
    for slice_id in sorted(common_ids):
        masses = {
            candidate_id: per_candidate[candidate_id][slice_id]["error_mass_fraction"]
            for candidate_id in per_candidate
        }
        rows = {
            candidate_id: per_candidate[candidate_id][slice_id]["row_fraction"]
            for candidate_id in per_candidate
        }
        minimum_mass = min(masses.values())
        maximum_rows = max(rows.values())
        records.append(
            {
                "slice_id": slice_id,
                "minimum_error_mass_fraction": minimum_mass,
                "mean_error_mass_fraction": float(np.mean(list(masses.values()))),
                "maximum_row_fraction": maximum_rows,
                "minimum_concentration_lift": float(
                    min(masses[candidate_id] / rows[candidate_id] for candidate_id in masses)
                ),
                "error_mass_fraction_by_candidate": masses,
                "row_fraction_by_candidate": rows,
                "passes_threshold": minimum_mass >= threshold,
            }
        )
    records.sort(
        key=lambda item: (
            -float(item["minimum_error_mass_fraction"]),
            str(item["slice_id"]),
        )
    )
    passing = [item for item in records if bool(item["passes_threshold"])]
    concentrated = [
        item
        for item in passing
        if float(item["maximum_row_fraction"]) < 0.95
        and float(item["minimum_concentration_lift"]) > 1.0
    ]
    selected = max(
        concentrated,
        key=lambda item: (
            float(item["minimum_concentration_lift"]),
            float(item["minimum_error_mass_fraction"]),
            str(item["slice_id"]),
        ),
        default=(passing[0] if passing else None),
    )
    return {
        "threshold": threshold,
        "candidate_ids": list(canonical),
        "aligned_row_count": len(first),
        "aligned_keys_sha256": sha256_dataframe(first[_KEYS]),
        "slice_columns": list(_SLICE_COLUMNS),
        "passing_slices": passing,
        "selected_shared_failure_slice": selected,
        "all_slices": records,
    }
