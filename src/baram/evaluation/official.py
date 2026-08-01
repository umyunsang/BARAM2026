"""Exact local implementation of the official 1-NMAE/FICR score."""

from collections.abc import Mapping

import numpy as np
import pandas as pd

from baram.constants import METRIC_COLUMNS
from baram.contracts.types import GroupId, GroupScore, OfficialScore
from baram.exceptions import MetricError


def settlement_unit(error_rate: np.ndarray) -> np.ndarray:
    """Return settlement units for capacity-relative absolute error."""
    error = np.asarray(error_rate, dtype=float)
    if not np.isfinite(error).all() or (error < 0).any():
        raise MetricError("error rate must be finite and nonnegative")
    return np.select([error <= 0.06, error <= 0.08], [4.0, 3.0], default=0.0)


def _validate_columns_and_values(frame: pd.DataFrame) -> None:
    if len(frame.columns) != len(METRIC_COLUMNS) or set(frame.columns) != METRIC_COLUMNS:
        raise MetricError(f"metric columns must equal {sorted(METRIC_COLUMNS)}")
    if frame[["actual_kwh", "prediction_kwh"]].isna().any().any():
        raise MetricError("metric inputs contain NaN")
    if not np.isfinite(frame[["actual_kwh", "prediction_kwh"]].to_numpy(dtype=float)).all():
        raise MetricError("metric inputs contain non-finite values")
    if frame.duplicated(["forecast_id", "forecast_kst_dtm", "group_id"]).any():
        raise MetricError("metric inputs contain duplicate keys")


def _validate_capacity(capacity: float) -> float:
    value = float(capacity)
    if not np.isfinite(value) or value <= 0.0:
        raise MetricError("capacity must be finite and positive")
    return value


def _score_group(part: pd.DataFrame, capacity: float) -> GroupScore:
    valid = part.loc[part["actual_kwh"].ge(0.1 * capacity)]
    if valid.empty:
        raise MetricError("group has zero valid rows")
    actual = valid["actual_kwh"].to_numpy(dtype=float)
    error = (valid["prediction_kwh"] - valid["actual_kwh"]).abs().to_numpy(dtype=float) / capacity
    units = settlement_unit(error)
    denominator = float((actual * 4.0).sum())
    return GroupScore(
        nmae=float(error.mean()),
        ficr=float((actual * units).sum() / denominator),
        valid_rows=len(valid),
        settlement_tier_counts={
            "unit_4": int(np.count_nonzero(units == 4.0)),
            "unit_3": int(np.count_nonzero(units == 3.0)),
            "unit_0": int(np.count_nonzero(units == 0.0)),
        },
    )


def evaluate_group_component(
    frame: pd.DataFrame,
    group: GroupId,
    capacity: float,
) -> GroupScore:
    """Score one group for explicitly diagnostic use without constructing a total."""
    _validate_columns_and_values(frame)
    if group not in (1, 2, 3) or set(frame["group_id"].unique()) != {group}:
        raise MetricError("group component inputs must contain exactly the requested group")
    return _score_group(frame, _validate_capacity(capacity))


def evaluate_official(
    frame: pd.DataFrame,
    capacities: Mapping[int, float],
) -> OfficialScore:
    """Compute the exact equal-group official total without intermediate rounding."""
    _validate_columns_and_values(frame)
    if set(capacities) != {1, 2, 3} or not all(
        np.isfinite(value) and value > 0.0 for value in capacities.values()
    ):
        raise MetricError("capacities must be finite positive values for groups 1, 2, and 3")
    normalized_capacities = {group: _validate_capacity(capacities[group]) for group in (1, 2, 3)}
    if set(frame["group_id"].unique()) != {1, 2, 3}:
        raise MetricError("metric inputs must contain groups 1, 2, and 3")

    components: dict[int, GroupScore] = {}
    for group in (1, 2, 3):
        components[group] = _score_group(
            frame.loc[frame["group_id"].eq(group)], normalized_capacities[group]
        )
    group_nmae = {group: component.nmae for group, component in components.items()}
    group_ficr = {group: component.ficr for group, component in components.items()}
    one_minus_nmae = 1.0 - float(np.mean(list(group_nmae.values())))
    ficr = float(np.mean(list(group_ficr.values())))
    return OfficialScore(
        total=0.5 * one_minus_nmae + 0.5 * ficr,
        one_minus_nmae=one_minus_nmae,
        ficr=ficr,
        group_nmae=group_nmae,  # type: ignore[arg-type]
        group_ficr=group_ficr,  # type: ignore[arg-type]
        valid_rows={group: component.valid_rows for group, component in components.items()},  # type: ignore[arg-type]
        settlement_tier_counts={
            group: dict(component.settlement_tier_counts) for group, component in components.items()
        },  # type: ignore[arg-type]
    )
