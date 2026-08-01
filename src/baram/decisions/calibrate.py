"""Leakage-safe, OOF-only group calibration policies."""

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from baram.contracts.hashing import canonical_sha256
from baram.contracts.types import CalibrationPolicy, GroupId
from baram.evaluation.official import evaluate_group_component
from baram.exceptions import ContractError, LeakageError

_KEYS = ["forecast_id", "forecast_kst_dtm", "group_id"]
_REQUIRED = {*_KEYS, "actual_kwh", "prediction_kwh", "model_id"}
_SCALES = (0.96, 0.98, 1.0, 1.02, 1.04)
_OFFSETS = (-0.02, -0.01, 0.0, 0.01, 0.02)
_CAP_MODES = ("nonnegative_only", "1.01_capacity", "capacity")


def _normalized_records(frame: pd.DataFrame, columns: Sequence[str]) -> list[dict[str, object]]:
    normalized = frame[list(columns)].copy()
    if "forecast_kst_dtm" in normalized:
        normalized["forecast_kst_dtm"] = pd.to_datetime(
            normalized["forecast_kst_dtm"], errors="raise"
        ).dt.strftime("%Y-%m-%dT%H:%M:%S.%f")
    normalized = normalized.sort_values(list(columns), kind="stable").reset_index(drop=True)
    return normalized.to_dict(orient="records")


def _validate_group_frame(frame: pd.DataFrame, group_id: int, capacity: float) -> pd.DataFrame:
    missing = _REQUIRED - set(frame.columns)
    if missing:
        raise ContractError(f"calibration frame is missing: {sorted(missing)}")
    if group_id not in (1, 2, 3):
        raise ContractError("calibration group must be 1, 2, or 3")
    if not np.isfinite(capacity) or capacity <= 0.0:
        raise ContractError("calibration capacity must be finite and positive")
    result = frame.copy()
    result["forecast_kst_dtm"] = pd.to_datetime(result["forecast_kst_dtm"], errors="raise")
    if set(result["group_id"].unique()) != {group_id}:
        raise ContractError("calibration frame must contain exactly the requested group")
    if result.empty or result.duplicated(_KEYS).any():
        raise ContractError("calibration frame must be nonempty with unique keys")
    values = result[["actual_kwh", "prediction_kwh"]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ContractError("calibration values must be finite")
    return result


def apply_calibration(
    prediction: np.ndarray,
    group_id: int,
    capacity: float,
    policy: CalibrationPolicy,
) -> np.ndarray:
    """Apply one frozen group policy with physical nonnegative/cap bounds."""
    if policy.group_id != group_id:
        raise ContractError("cannot apply a calibration policy belonging to another group")
    if not np.isfinite(capacity) or capacity <= 0.0:
        raise ContractError("calibration capacity must be finite and positive")
    values = np.asarray(prediction, dtype=float)
    if not np.isfinite(values).all():
        raise ContractError("calibration predictions must be finite")
    adjusted = values * policy.scale + policy.offset_capacity * capacity
    if policy.cap_mode == "capacity":
        return np.clip(adjusted, 0.0, capacity)
    if policy.cap_mode == "1.01_capacity":
        return np.clip(adjusted, 0.0, 1.01 * capacity)
    if policy.cap_mode == "nonnegative_only":
        return np.maximum(adjusted, 0.0)
    raise ContractError(f"unsupported calibration cap mode: {policy.cap_mode}")


def _component_total(frame: pd.DataFrame, group_id: GroupId, capacity: float) -> float:
    score = evaluate_group_component(
        frame[[*_KEYS, "actual_kwh", "prediction_kwh"]], group_id, capacity
    )
    return 0.5 * (1.0 - score.nmae) + 0.5 * score.ficr


def fit_group_calibration(
    frame: pd.DataFrame,
    group_id: GroupId,
    capacity: float,
    training_rows_sha256: str,
    metric_sha256: str,
) -> CalibrationPolicy:
    """Select the fixed-grid policy maximizing the exact group score component."""
    training = _validate_group_frame(frame, group_id, capacity)
    parent_ids = tuple(sorted(str(item) for item in training["model_id"].unique()))
    input_hash = canonical_sha256(
        _normalized_records(training, [*_KEYS, "prediction_kwh", "model_id"])
    )

    candidates: list[tuple[float, float, int, float, float, str]] = []
    base_prediction = training["prediction_kwh"].to_numpy(dtype=float)
    cap_preference = {"nonnegative_only": 0, "1.01_capacity": 1, "capacity": 2}
    for scale in _SCALES:
        for offset in _OFFSETS:
            for cap_mode in _CAP_MODES:
                provisional = CalibrationPolicy(
                    group_id=group_id,
                    scale=scale,
                    offset_capacity=offset,
                    cap_mode=cap_mode,  # type: ignore[arg-type]
                    training_rows_sha256=training_rows_sha256,
                    parent_model_ids=parent_ids,
                    input_prediction_sha256=input_hash,
                    metric_sha256=metric_sha256,
                )
                scored = training[[*_KEYS, "actual_kwh"]].copy()
                scored["prediction_kwh"] = apply_calibration(
                    base_prediction, group_id, capacity, provisional
                )
                total = _component_total(scored, group_id, capacity)
                candidates.append(
                    (
                        -total,
                        abs(scale - 1.0) + abs(offset),
                        cap_preference[cap_mode],
                        scale,
                        offset,
                        cap_mode,
                    )
                )
    _, _, _, scale, offset, cap_mode = min(candidates)
    return CalibrationPolicy(
        group_id=group_id,
        scale=scale,
        offset_capacity=offset,
        cap_mode=cap_mode,  # type: ignore[arg-type]
        training_rows_sha256=training_rows_sha256,
        parent_model_ids=parent_ids,
        input_prediction_sha256=input_hash,
        metric_sha256=metric_sha256,
    )


def cross_fit_calibration(
    oof: pd.DataFrame,
    fold_order: Sequence[str],
    capacities: Mapping[int, float],
    metric_sha256: str,
) -> pd.DataFrame:
    """Fit on preceding OOF folds and apply only to the next unseen fold."""
    required = {*_REQUIRED, "fold_id"}
    missing = required - set(oof.columns)
    if missing:
        raise ContractError(f"cross-fit frame is missing: {sorted(missing)}")
    ordered_folds = tuple(fold_order)
    if len(ordered_folds) < 2 or len(set(ordered_folds)) != len(ordered_folds):
        raise ContractError("cross-fit fold order must contain at least two unique folds")
    observed = set(str(item) for item in oof["fold_id"].unique())
    if observed != set(ordered_folds):
        raise ContractError("cross-fit fold order does not match observed folds")
    if not capacities:
        raise ContractError("cross-fit capacities cannot be empty")

    source = oof.copy()
    source["forecast_kst_dtm"] = pd.to_datetime(source["forecast_kst_dtm"], errors="raise")
    outputs: list[pd.DataFrame] = []
    for position, validation_fold in enumerate(ordered_folds[1:], start=1):
        training = source.loc[source["fold_id"].isin(ordered_folds[:position])]
        validation = source.loc[source["fold_id"].eq(validation_fold)].copy()
        train_keys = set(map(tuple, training[_KEYS].itertuples(index=False, name=None)))
        validation_keys = set(map(tuple, validation[_KEYS].itertuples(index=False, name=None)))
        if train_keys & validation_keys:
            raise LeakageError("cross-fit calibration rows overlap their policy-training rows")
        for group_id, capacity in sorted(capacities.items()):
            train_group = training.loc[training["group_id"].eq(group_id)]
            validation_mask = validation["group_id"].eq(group_id)
            if not validation_mask.any():
                continue
            if train_group.empty:
                raise ContractError(f"group {group_id} has no preceding calibration rows")
            training_hash = canonical_sha256(_normalized_records(train_group, [*_KEYS, "fold_id"]))
            policy = fit_group_calibration(
                train_group,
                group_id,  # type: ignore[arg-type]
                capacity,
                training_hash,
                metric_sha256,
            )
            validation.loc[validation_mask, "prediction_kwh"] = apply_calibration(
                validation.loc[validation_mask, "prediction_kwh"].to_numpy(dtype=float),
                group_id,
                capacity,
                policy,
            )
        outputs.append(validation)
    return pd.concat(outputs, ignore_index=True)
