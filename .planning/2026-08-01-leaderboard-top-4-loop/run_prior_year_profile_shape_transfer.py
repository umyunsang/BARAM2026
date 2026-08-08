"""Screen one same-calendar prior-year profile-shape transfer on exact M107.

Groups 1/2 retrieve their 2022 target profile at the same month/day/hour as
each complete 2023 issuance.  The historical normalized profile is rescaled to
the exact M107 daily mean, then blended with M107.  Q2 and Q3 jointly select a
small per-group mass; that policy transfers once to Q4.  Group 3 and incomplete
or boundary issuances remain exact M107.  The pre-2024 development surface is
the only target source materialized by this runner.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from build_v2_transfer_sequence_challenger import (
    CAPACITIES,
    METRIC_COLUMNS,
    _group_score,
    _paired_issuance_bootstrap,
    _score,
)
from run_conditional_daily_analog_profile import BASELINE, BASELINE_SHA, OPEN, OPEN_SHA
from run_strict_parent_analog_transfer import FOLD_MAP, _pooled, _strict_parents
from run_temporal_reconciliation import _verified_m107_receipt
from strict_dev_surface import DEV_CUTOFF, development_surface

from baram.contracts.hashing import canonical_sha256, sha256_file

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "artifacts" / "backtests" / "metric-aligned-probe"
MODEL_ID = "M262_PRIOR_YEAR_PROFILE_SHAPE_TRANSFER"
ELIGIBLE_GROUPS = (1, 2)
HISTORY_YEAR = 2022
QUERY_YEAR = 2023
WEIGHTS = (0.0, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20)
LOWER_BOUND = 0.0
UPPER_BOUND = 1.075
KEYS = ["forecast_id", "forecast_kst_dtm", "group_id"]


def _history_lookup(surface: pd.DataFrame) -> pd.DataFrame:
    history = surface.loc[
        surface["forecast_kst_dtm"].dt.year.eq(HISTORY_YEAR)
        & surface["group_id"].isin(ELIGIBLE_GROUPS),
        ["forecast_kst_dtm", "group_id", "actual_kwh"],
    ].copy()
    history["month"] = history["forecast_kst_dtm"].dt.month
    history["day"] = history["forecast_kst_dtm"].dt.day
    history["hour"] = history["forecast_kst_dtm"].dt.hour
    history["prior_normalized"] = history["actual_kwh"] / history["group_id"].map(
        CAPACITIES
    )
    history = history.loc[np.isfinite(history["prior_normalized"].to_numpy(dtype=float))]
    calendar_keys = ["group_id", "month", "day", "hour"]
    if history[calendar_keys].duplicated().any():
        raise RuntimeError("M262 prior-year calendar lookup is not one-to-one")
    if history.empty or set(history["group_id"].astype(int)) != set(ELIGIBLE_GROUPS):
        raise RuntimeError("M262 prior-year history coverage changed")
    return history[[*calendar_keys, "prior_normalized"]]


def _raw_fold_profile(
    parent: pd.DataFrame,
    history: pd.DataFrame,
    fold: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    output = parent.reset_index(drop=True).copy()
    if not output["forecast_kst_dtm"].dt.year.isin((QUERY_YEAR, 2024)).all():
        raise RuntimeError(f"M262 {fold} query-year contract changed")
    output["month"] = output["forecast_kst_dtm"].dt.month
    output["day"] = output["forecast_kst_dtm"].dt.day
    output["hour"] = output["forecast_kst_dtm"].dt.hour
    output = output.merge(
        history,
        on=["group_id", "month", "day", "hour"],
        how="left",
        validate="many_to_one",
        sort=False,
    )
    output["profile_prediction_kwh"] = output["prediction_kwh"].to_numpy(dtype=float)
    output["profile_eligible"] = False
    output["profile_scale"] = np.nan

    diagnostics: dict[str, object] = {"groups": {}}
    for group_id in CAPACITIES:
        group_mask = output["group_id"].eq(group_id)
        eligible_rows = 0
        eligible_issuances = 0
        fallback_rows = int(group_mask.sum())
        scales: list[float] = []
        if group_id in ELIGIBLE_GROUPS:
            group = output.loc[group_mask]
            for _, indices in group.groupby(
                "data_available_kst_dtm", sort=False
            ).indices.items():
                positions = group.index.to_numpy()[np.asarray(indices, dtype=int)]
                block = output.loc[positions]
                complete = (
                    len(block) == 24
                    and block["forecast_kst_dtm"].dt.year.eq(QUERY_YEAR).all()
                    and block["prior_normalized"].notna().all()
                )
                if not complete:
                    continue
                parent_normalized = (
                    block["prediction_kwh"].to_numpy(dtype=float)
                    / CAPACITIES[group_id]
                )
                prior = block["prior_normalized"].to_numpy(dtype=float)
                prior_mean = float(prior.mean())
                if not np.isfinite(prior_mean) or prior_mean <= 1e-4:
                    continue
                scale = float(parent_normalized.mean() / prior_mean)
                profile = np.clip(prior * scale, LOWER_BOUND, UPPER_BOUND)
                if not np.isfinite(profile).all():
                    raise RuntimeError(f"M262 {fold} group {group_id} profile is invalid")
                output.loc[positions, "profile_prediction_kwh"] = (
                    profile * CAPACITIES[group_id]
                )
                output.loc[positions, "profile_eligible"] = True
                output.loc[positions, "profile_scale"] = scale
                eligible_rows += len(positions)
                eligible_issuances += 1
                scales.append(scale)
            fallback_rows -= eligible_rows
        diagnostics["groups"][str(group_id)] = {
            "eligible_rows": eligible_rows,
            "eligible_complete_issuances": eligible_issuances,
            "parent_fallback_rows": fallback_rows,
            "scale_mean": float(np.mean(scales)) if scales else None,
            "scale_min": float(np.min(scales)) if scales else None,
            "scale_max": float(np.max(scales)) if scales else None,
        }

    diagnostics["eligible_rows"] = int(output["profile_eligible"].sum())
    diagnostics["parent_fallback_rows"] = int((~output["profile_eligible"]).sum())
    return output, diagnostics


def _apply_weight(
    raw: pd.DataFrame,
    group_id: int,
    weight: float,
) -> pd.DataFrame:
    output = raw[[*METRIC_COLUMNS, "data_available_kst_dtm"]].copy()
    group = output["group_id"].eq(group_id).to_numpy()
    eligible = raw["profile_eligible"].to_numpy(dtype=bool)
    affected = group & eligible
    parent = raw.loc[affected, "prediction_kwh"].to_numpy(dtype=float)
    profile = raw.loc[affected, "profile_prediction_kwh"].to_numpy(dtype=float)
    output.loc[affected, "prediction_kwh"] = (1.0 - weight) * parent + weight * profile
    return output


def _select_weights(
    raw: dict[str, pd.DataFrame],
) -> tuple[dict[int, float], dict[str, object]]:
    selections = {group_id: 0.0 for group_id in CAPACITIES}
    diagnostics: dict[str, object] = {}
    for group_id in CAPACITIES:
        q2_parent = _group_score(raw["q2"], group_id)
        q3_parent = _group_score(raw["q3"], group_id)
        candidates: list[dict[str, object]] = []
        for weight in WEIGHTS:
            q2 = _group_score(_apply_weight(raw["q2"], group_id, weight), group_id)
            q3 = _group_score(_apply_weight(raw["q3"], group_id, weight), group_id)
            candidates.append(
                {
                    "weight": weight,
                    "q2_total": q2["total"],
                    "q3_total": q3["total"],
                    "q2_delta": q2["total"] - q2_parent["total"],
                    "q3_delta": q3["total"] - q3_parent["total"],
                }
            )
        stable = [
            item
            for item in candidates
            if item["weight"] > 0.0
            and item["q2_delta"] > 0.0
            and item["q3_delta"] > 0.0
        ]
        if group_id in ELIGIBLE_GROUPS and stable:
            selected = max(
                stable,
                key=lambda item: (
                    min(float(item["q2_delta"]), float(item["q3_delta"])),
                    0.5 * (float(item["q2_delta"]) + float(item["q3_delta"])),
                    -float(item["weight"]),
                ),
            )
            selections[group_id] = float(selected["weight"])
        diagnostics[str(group_id)] = {
            "q2_parent": q2_parent,
            "q3_parent": q3_parent,
            "candidate_scores": candidates,
            "stable_positive_candidates": len(stable),
            "selected_weight": selections[group_id],
        }
    if selections[3] != 0.0:
        raise RuntimeError("M262 group 3 must remain exact M107")
    return selections, diagnostics


def _apply_policy(
    raw: pd.DataFrame,
    selections: dict[int, float],
) -> pd.DataFrame:
    output = raw[[*METRIC_COLUMNS, "data_available_kst_dtm"]].copy()
    for group_id, weight in selections.items():
        if weight == 0.0:
            continue
        group = output["group_id"].eq(group_id).to_numpy()
        eligible = raw["profile_eligible"].to_numpy(dtype=bool)
        affected = group & eligible
        output.loc[affected, "prediction_kwh"] = (
            (1.0 - weight) * raw.loc[affected, "prediction_kwh"].to_numpy(dtype=float)
            + weight
            * raw.loc[affected, "profile_prediction_kwh"].to_numpy(dtype=float)
        )
    return output


def main() -> None:
    if sha256_file(OPEN) != OPEN_SHA or sha256_file(BASELINE) != BASELINE_SHA:
        raise RuntimeError("immutable competition input hash mismatch")
    started = time.perf_counter()
    m107_receipt = _verified_m107_receipt()
    surface, _ = development_surface()
    if surface["forecast_kst_dtm"].ge(DEV_CUTOFF).any():
        raise RuntimeError("lockbox row reached M262 runner")
    history = _history_lookup(surface)
    parents, boundary_fallback = _strict_parents(surface)

    raw: dict[str, pd.DataFrame] = {}
    coverage: dict[str, object] = {}
    for fold in FOLD_MAP:
        raw[fold], coverage[fold] = _raw_fold_profile(parents[fold], history, fold)
    selections, selection_diagnostics = _select_weights(raw)
    outputs = {fold: _apply_policy(raw[fold], selections) for fold in FOLD_MAP}

    fold_scores: dict[str, object] = {}
    fold_deltas: dict[str, float] = {}
    for fold in FOLD_MAP:
        parent_score = _score(parents[fold])
        selected_score = _score(outputs[fold])
        expected = m107_receipt["fold_scores"][FOLD_MAP[fold]]
        for metric in ("total", "one_minus_nmae", "ficr"):
            if abs(parent_score[metric] - expected[metric]) > 1e-12:
                raise RuntimeError(f"M107 {fold} {metric} reproduction changed")
        deltas = {
            metric: selected_score[metric] - parent_score[metric]
            for metric in ("total", "one_minus_nmae", "ficr")
        }
        fold_deltas[fold] = deltas["total"]
        fold_scores[fold] = {
            "parent": parent_score,
            "selected": selected_score,
            "deltas": deltas,
        }

    pooled_parent_frame = _pooled(parents)
    pooled_selected_frame = _pooled(outputs)
    pooled_parent = _score(pooled_parent_frame)
    pooled_selected = _score(pooled_selected_frame)
    for metric in ("total", "one_minus_nmae", "ficr"):
        if abs(pooled_parent[metric] - m107_receipt["pooled"][metric]) > 1e-12:
            raise RuntimeError(f"M107 pooled {metric} reproduction changed")
    pooled_deltas = {
        metric: pooled_selected[metric] - pooled_parent[metric]
        for metric in ("total", "one_minus_nmae", "ficr")
    }
    q4_bootstrap = _paired_issuance_bootstrap(parents["q4"], outputs["q4"])
    all_folds_positive = all(delta > 0.0 for delta in fold_deltas.values())
    pooled_positive = pooled_deltas["total"] > 0.0
    bootstrap_positive = q4_bootstrap["positive_fraction"] > 0.50
    promoted = all_folds_positive and pooled_positive and bootstrap_positive

    prediction_path = OUTPUT / f"{MODEL_ID}-oof.parquet"
    prediction_output = pooled_selected_frame.copy()
    prediction_output["model_id"] = MODEL_ID
    prediction_output[
        [*METRIC_COLUMNS, "fold_id", "model_id", "data_available_kst_dtm"]
    ].to_parquet(prediction_path, index=False)
    policy = {
        "architecture": "same_calendar_prior_year_daily_profile_shape_transfer",
        "parent": "M107_STRICT_TEMPORAL_TOP100",
        "eligible_groups": list(ELIGIBLE_GROUPS),
        "history_year": HISTORY_YEAR,
        "query_year": QUERY_YEAR,
        "calendar_key": ["month", "day", "hour"],
        "profile_transform": "prior_normalized_times_parent_daily_mean_over_prior_daily_mean",
        "bounds": [LOWER_BOUND, UPPER_BOUND],
        "selection_folds": [FOLD_MAP["q2"], FOLD_MAP["q3"]],
        "frozen_transfer_fold": FOLD_MAP["q4"],
        "weight_grid": list(WEIGHTS),
        "selection_rule": (
            "require positive Q2 and Q3 group-total deltas; maximize worst delta, "
            "then mean delta, then prefer smaller mass"
        ),
        "fallback": "exact_M107_for_group3_incomplete_profiles_and_boundary",
        "alternate_calendar_search": False,
        "transform_search": False,
        "weather_gate_search": False,
        "group_or_fold_exception_search": False,
        "test_build": False,
    }
    receipt = {
        "schema_version": 1,
        "candidate_id": MODEL_ID,
        "state": (
            "LOCAL_PRIOR_YEAR_PROFILE_PROMOTED_NO_TEST_BUILD"
            if promoted
            else "LOCAL_PRIOR_YEAR_PROFILE_REJECTED_NO_TEST_BUILD"
        ),
        "policy": policy,
        "policy_sha256": canonical_sha256(policy),
        "selected_weights": {str(group): weight for group, weight in selections.items()},
        "selection_diagnostics": selection_diagnostics,
        "coverage": coverage,
        "boundary_fallback": boundary_fallback,
        "fold_scores": fold_scores,
        "pooled": {
            "parent": pooled_parent,
            "selected": pooled_selected,
            "deltas": pooled_deltas,
        },
        "q4_paired_bootstrap": q4_bootstrap,
        "promotion": {
            "promoted": promoted,
            "all_q2_q3_q4_total_deltas_positive": all_folds_positive,
            "pooled_total_delta_positive": pooled_positive,
            "q4_bootstrap_positive_fraction_above_half": bootstrap_positive,
            "rule": (
                "Q2, Q3, Q4, and pooled Total deltas are positive and Q4 paired "
                "issuance bootstrap positive fraction exceeds 0.50"
            ),
        },
        "source_receipts": {
            "open_zip_sha256": OPEN_SHA,
            "baseline_ipynb_sha256": BASELINE_SHA,
            "m107_prediction_sha256": m107_receipt["prediction_sha256"],
        },
        "prediction_path": str(prediction_path.relative_to(ROOT)),
        "prediction_sha256": sha256_file(prediction_path),
        "completion_target": {
            "metric": "Dacon Total",
            "strict_threshold": 0.66000,
            "status": "UNVERIFIED_REQUIRES_EXTERNAL_DACON_RESULT",
        },
        "online_score": None,
        "no_external_upload": True,
        "new_2024_evaluation": False,
        "lockbox_reopened": False,
        "external_actions": [],
        "runtime_seconds": round(time.perf_counter() - started, 2),
    }
    receipt_path = OUTPUT / f"{MODEL_ID}.json"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    print(json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
