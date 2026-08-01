"""Exact-score, two-parent convex blending on keyed OOF predictions."""

from collections.abc import Mapping

import numpy as np
import pandas as pd

from baram.contracts.hashing import canonical_sha256
from baram.contracts.types import BlendPolicy, GroupId
from baram.evaluation.official import evaluate_group_component
from baram.exceptions import ContractError

_KEYS = ["forecast_id", "forecast_kst_dtm", "group_id"]


def _canonical_predictions(frame: pd.DataFrame, model_id: str) -> pd.DataFrame:
    required = {*_KEYS, "prediction_kwh"}
    missing = required - set(frame.columns)
    if missing:
        raise ContractError(f"prediction {model_id} is missing: {sorted(missing)}")
    result = frame[[*_KEYS, "prediction_kwh"]].copy()
    result["forecast_kst_dtm"] = pd.to_datetime(result["forecast_kst_dtm"], errors="raise")
    if result.empty or result.duplicated(_KEYS).any():
        raise ContractError(f"prediction {model_id} must have nonempty unique keys")
    if not np.isfinite(result["prediction_kwh"].to_numpy(dtype=float)).all():
        raise ContractError(f"prediction {model_id} contains non-finite values")
    return result.sort_values(_KEYS, kind="stable").reset_index(drop=True)


def _canonical_labels(labels: pd.DataFrame) -> pd.DataFrame:
    required = {*_KEYS, "actual_kwh"}
    missing = required - set(labels.columns)
    if missing:
        raise ContractError(f"blend labels are missing: {sorted(missing)}")
    result = labels[[*_KEYS, "actual_kwh"]].copy()
    result["forecast_kst_dtm"] = pd.to_datetime(result["forecast_kst_dtm"], errors="raise")
    if result.empty or result.duplicated(_KEYS).any():
        raise ContractError("blend labels must have nonempty unique keys")
    if not np.isfinite(result["actual_kwh"].to_numpy(dtype=float)).all():
        raise ContractError("blend labels contain non-finite values")
    return result.sort_values(_KEYS, kind="stable").reset_index(drop=True)


def _hash_prediction(frame: pd.DataFrame) -> str:
    serializable = frame.copy()
    serializable["forecast_kst_dtm"] = serializable["forecast_kst_dtm"].dt.strftime(
        "%Y-%m-%dT%H:%M:%S.%f"
    )
    return canonical_sha256(serializable.to_dict(orient="records"))


def _validated_inputs(
    predictions: Mapping[str, pd.DataFrame],
) -> tuple[tuple[str, str], dict[str, pd.DataFrame]]:
    if len(predictions) != 2:
        raise ContractError("blend requires exactly two parent predictions")
    model_ids = tuple(sorted(predictions))
    canonical = {
        model_id: _canonical_predictions(predictions[model_id], model_id) for model_id in model_ids
    }
    if not canonical[model_ids[0]][_KEYS].equals(canonical[model_ids[1]][_KEYS]):
        raise ContractError("blend parent keys do not match")
    return (model_ids[0], model_ids[1]), canonical


def _component_total(frame: pd.DataFrame, group_id: GroupId, capacity: float) -> float:
    score = evaluate_group_component(frame, group_id, capacity)
    return 0.5 * (1.0 - score.nmae) + 0.5 * score.ficr


def fit_two_model_blend(
    predictions: Mapping[str, pd.DataFrame],
    labels: pd.DataFrame,
    capacities: Mapping[int, float],
    training_rows_sha256: str,
    metric_sha256: str,
    *,
    weight_step: float = 0.05,
) -> BlendPolicy:
    """Fit independent group weights on a deterministic convex grid."""
    model_ids, canonical = _validated_inputs(predictions)
    truth = _canonical_labels(labels)
    if not canonical[model_ids[0]][_KEYS].equals(truth[_KEYS]):
        raise ContractError("blend prediction keys do not match label keys")
    if set(truth["group_id"].unique()) != set(capacities):
        raise ContractError("blend capacities do not match label groups")
    if not np.isfinite(weight_step) or weight_step <= 0.0 or weight_step > 1.0:
        raise ContractError("blend weight step must be in (0, 1]")
    intervals = round(1.0 / weight_step)
    if not np.isclose(intervals * weight_step, 1.0):
        raise ContractError("blend weight step must divide one exactly")

    weights_by_group: dict[GroupId, dict[str, float]] = {}
    first, second = model_ids
    for raw_group_id, capacity in sorted(capacities.items()):
        if raw_group_id not in (1, 2, 3) or not np.isfinite(capacity) or capacity <= 0.0:
            raise ContractError("blend capacities must be finite and positive for valid groups")
        group_id: GroupId = raw_group_id  # type: ignore[assignment]
        mask = truth["group_id"].eq(group_id)
        candidates: list[tuple[float, int, float, float]] = []
        for first_weight in np.linspace(0.0, 1.0, intervals + 1):
            first_weight = float(round(first_weight, 12))
            second_weight = float(round(1.0 - first_weight, 12))
            metric_frame = truth.loc[mask, [*_KEYS, "actual_kwh"]].copy()
            metric_frame["prediction_kwh"] = first_weight * canonical[first].loc[
                mask, "prediction_kwh"
            ].to_numpy(dtype=float) + second_weight * canonical[second].loc[
                mask, "prediction_kwh"
            ].to_numpy(dtype=float)
            score = _component_total(metric_frame, group_id, capacity)
            active_parents = int(first_weight > 0.0) + int(second_weight > 0.0)
            candidates.append((-score, active_parents, -first_weight, first_weight))
        _, _, _, first_weight = min(candidates)
        weights_by_group[group_id] = {
            first: first_weight,
            second: float(round(1.0 - first_weight, 12)),
        }

    return BlendPolicy(
        weights_by_group=weights_by_group,
        training_rows_sha256=training_rows_sha256,
        input_prediction_hashes={
            model_id: _hash_prediction(canonical[model_id]) for model_id in model_ids
        },
        metric_sha256=metric_sha256,
    )


def apply_blend(
    policy: BlendPolicy,
    predictions: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """Apply a frozen groupwise convex blend to exactly matching parent keys."""
    model_ids, canonical = _validated_inputs(predictions)
    if set(model_ids) != set(policy.input_prediction_hashes):
        raise ContractError("blend policy parents do not match supplied predictions")

    result = canonical[model_ids[0]][_KEYS].copy()
    result["prediction_kwh"] = np.nan
    for raw_group_id in sorted(result["group_id"].unique()):
        if raw_group_id not in policy.weights_by_group:
            raise ContractError(f"blend policy has no weights for group {raw_group_id}")
        weights = policy.weights_by_group[raw_group_id]
        if set(weights) != set(model_ids) or any(
            not np.isfinite(value) or value < 0.0 for value in weights.values()
        ):
            raise ContractError("blend policy contains invalid parent weights")
        if not np.isclose(sum(weights.values()), 1.0):
            raise ContractError("blend policy weights must sum to one")
        mask = result["group_id"].eq(raw_group_id)
        result.loc[mask, "prediction_kwh"] = sum(
            weights[model_id]
            * canonical[model_id].loc[mask, "prediction_kwh"].to_numpy(dtype=float)
            for model_id in model_ids
        )
    if result["prediction_kwh"].isna().any():
        raise ContractError("blend left predictions unresolved")
    return result
