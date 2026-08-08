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


def _validated_convex_inputs(
    predictions: Mapping[str, pd.DataFrame],
) -> tuple[tuple[str, ...], dict[str, pd.DataFrame]]:
    if len(predictions) not in {2, 3}:
        raise ContractError("convex blend requires two or three parent predictions")
    model_ids = tuple(sorted(predictions))
    canonical = {
        model_id: _canonical_predictions(predictions[model_id], model_id) for model_id in model_ids
    }
    reference = canonical[model_ids[0]][_KEYS]
    if any(not reference.equals(canonical[model_id][_KEYS]) for model_id in model_ids[1:]):
        raise ContractError("blend parent keys do not match")
    return model_ids, canonical


def _validated_inputs(
    predictions: Mapping[str, pd.DataFrame],
) -> tuple[tuple[str, str], dict[str, pd.DataFrame]]:
    if len(predictions) != 2:
        raise ContractError("blend requires exactly two parent predictions")
    model_ids, canonical = _validated_convex_inputs(predictions)
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
    return fit_convex_blend(
        predictions,
        labels,
        capacities,
        training_rows_sha256,
        metric_sha256,
        weight_step=weight_step,
    )


def _simplex_weights(parent_count: int, intervals: int) -> list[tuple[float, ...]]:
    if parent_count == 2:
        integer_weights = [(first, intervals - first) for first in range(intervals + 1)]
    elif parent_count == 3:
        integer_weights = [
            (first, second, intervals - first - second)
            for first in range(intervals + 1)
            for second in range(intervals - first + 1)
        ]
    else:
        raise ContractError("simplex enumeration supports two or three parents")
    return [tuple(round(value / intervals, 12) for value in row) for row in integer_weights]


def fit_convex_blend(
    predictions: Mapping[str, pd.DataFrame],
    labels: pd.DataFrame,
    capacities: Mapping[int, float],
    training_rows_sha256: str,
    metric_sha256: str,
    *,
    weight_step: float = 0.05,
) -> BlendPolicy:
    """Fit groupwise exact-score weights for two or three residual-diverse parents."""
    model_ids, canonical = _validated_convex_inputs(predictions)
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

    simplex = _simplex_weights(len(model_ids), intervals)
    weights_by_group: dict[GroupId, dict[str, float]] = {}
    for raw_group_id, capacity in sorted(capacities.items()):
        if raw_group_id not in (1, 2, 3) or not np.isfinite(capacity) or capacity <= 0.0:
            raise ContractError("blend capacities must be finite and positive for valid groups")
        group_id: GroupId = raw_group_id  # type: ignore[assignment]
        mask = truth["group_id"].eq(group_id)
        candidates: list[tuple[float, int, tuple[float, ...], tuple[float, ...]]] = []
        for weights in simplex:
            metric_frame = truth.loc[mask, [*_KEYS, "actual_kwh"]].copy()
            metric_frame["prediction_kwh"] = sum(
                weight
                * canonical[model_id].loc[mask, "prediction_kwh"].to_numpy(dtype=float)
                for model_id, weight in zip(model_ids, weights, strict=True)
            )
            score = _component_total(metric_frame, group_id, capacity)
            active_parents = sum(weight > 0.0 for weight in weights)
            favor_earlier = tuple(-weight for weight in weights)
            candidates.append((-score, active_parents, favor_earlier, weights))
        _, _, _, selected = min(candidates)
        weights_by_group[group_id] = dict(zip(model_ids, selected, strict=True))

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


def apply_convex_blend(
    policy: BlendPolicy,
    predictions: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """Apply a frozen two/three-parent convex policy to matching future keys."""
    model_ids, canonical = _validated_convex_inputs(predictions)
    if set(model_ids) != set(policy.input_prediction_hashes):
        raise ContractError("blend policy parents do not match supplied predictions")
    result = canonical[model_ids[0]][_KEYS].copy()
    result["prediction_kwh"] = np.nan
    for raw_group_id in sorted(result["group_id"].unique()):
        weights = policy.weights_by_group.get(raw_group_id)
        if weights is None or set(weights) != set(model_ids):
            raise ContractError(f"blend policy has invalid weights for group {raw_group_id}")
        invalid_values = any(
            not np.isfinite(value) or value < 0.0 for value in weights.values()
        )
        if invalid_values or not np.isclose(sum(weights.values()), 1.0):
            raise ContractError("blend policy contains invalid convex weights")
        mask = result["group_id"].eq(raw_group_id)
        result.loc[mask, "prediction_kwh"] = sum(
            weights[model_id]
            * canonical[model_id].loc[mask, "prediction_kwh"].to_numpy(dtype=float)
            for model_id in model_ids
        )
    if result["prediction_kwh"].isna().any():
        raise ContractError("convex blend left predictions unresolved")
    return result
