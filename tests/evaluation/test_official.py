import numpy as np
import pandas as pd
import pytest

from baram.evaluation.official import (
    evaluate_group_component,
    evaluate_official,
    settlement_unit,
)
from baram.exceptions import MetricError

CAPACITIES = {1: 21600.0, 2: 21600.0, 3: 21000.0}
REQUIRED = [
    "forecast_id",
    "forecast_kst_dtm",
    "group_id",
    "actual_kwh",
    "prediction_kwh",
]


def _handcase() -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for group, capacity in CAPACITIES.items():
        for idx, error in enumerate((0.05, 0.07)):
            actual = 0.5 * capacity
            rows.append(
                {
                    "forecast_id": f"{group}-{idx}",
                    "forecast_kst_dtm": f"2024-01-01 0{idx + 1}:00:00",
                    "group_id": group,
                    "actual_kwh": actual,
                    "prediction_kwh": actual + error * capacity,
                }
            )
    return pd.DataFrame(rows, columns=REQUIRED)


def test_handcase_score_is_09075() -> None:
    """Catches divergence from the hand-derived official combined score."""
    score = evaluate_official(_handcase(), CAPACITIES)
    assert score.one_minus_nmae == pytest.approx(0.94)
    assert score.ficr == pytest.approx(0.875)
    assert score.total == pytest.approx(0.9075)
    assert score.settlement_tier_counts[1] == {"unit_4": 1, "unit_3": 1, "unit_0": 0}


def test_settlement_boundaries_are_inclusive() -> None:
    """Catches off-by-one comparison changes at the 6% and 8% tiers."""
    units = settlement_unit(np.array([0.06, np.nextafter(0.06, 1.0), 0.08, 0.0800001]))
    assert units.tolist() == [4.0, 3.0, 3.0, 0.0]


@pytest.mark.parametrize("invalid", [np.array([-0.1]), np.array([np.nan])])
def test_settlement_rejects_invalid_error_rates(invalid: np.ndarray) -> None:
    """Catches invalid settlement tiers being assigned to malformed errors."""
    with pytest.raises(MetricError, match="finite and nonnegative"):
        settlement_unit(invalid)


def test_actual_threshold_includes_exactly_ten_percent() -> None:
    """Catches accidental strict-greater threshold filtering."""
    frame = _handcase()
    frame.loc[0, "actual_kwh"] = 2160.0
    frame.loc[0, "prediction_kwh"] = 2160.0
    frame.loc[1, "actual_kwh"] = np.nextafter(2160.0, 0.0)
    frame.loc[1, "prediction_kwh"] = frame.loc[1, "actual_kwh"]
    score = evaluate_official(frame, CAPACITIES)
    assert score.valid_rows[1] == 1


def test_group_helper_matches_official_component() -> None:
    """Catches diagnostic scoring drifting from the official primitive."""
    frame = _handcase()
    component = evaluate_group_component(frame.loc[frame["group_id"].eq(2)], 2, 21600.0)
    official = evaluate_official(frame, CAPACITIES)
    assert component.nmae == official.group_nmae[2]
    assert component.ficr == official.group_ficr[2]
    assert component.valid_rows == official.valid_rows[2]


def test_group_helper_rejects_wrong_group_or_capacity() -> None:
    """Catches a diagnostic component being labeled as another group."""
    group_two = _handcase().loc[lambda x: x["group_id"].eq(2)]
    with pytest.raises(MetricError, match="requested group"):
        evaluate_group_component(group_two, 1, 21600.0)
    with pytest.raises(MetricError, match="capacity"):
        evaluate_group_component(group_two, 2, 0.0)


@pytest.mark.parametrize("column,value", [("actual_kwh", np.nan), ("prediction_kwh", np.inf)])
def test_metric_rejects_nonfinite_values(column: str, value: float) -> None:
    """Catches silent propagation of invalid targets or predictions."""
    frame = _handcase()
    frame.loc[0, column] = value
    with pytest.raises(MetricError, match=r"NaN|non-finite"):
        evaluate_official(frame, CAPACITIES)


def test_metric_rejects_duplicate_keys() -> None:
    """Catches duplicated evaluation weight."""
    frame = pd.concat([_handcase(), _handcase().iloc[[0]]], ignore_index=True)
    with pytest.raises(MetricError, match="duplicate keys"):
        evaluate_official(frame, CAPACITIES)


def test_metric_requires_exact_columns_and_groups() -> None:
    """Catches accidental index columns and partial official totals."""
    with pytest.raises(MetricError, match="columns"):
        evaluate_official(_handcase().assign(extra=1), CAPACITIES)
    with pytest.raises(MetricError, match="groups"):
        evaluate_official(_handcase().loc[lambda x: x["group_id"].ne(3)], CAPACITIES)


@pytest.mark.parametrize(
    "capacities",
    [{1: 21600.0, 2: 21600.0}, {1: 21600.0, 2: 21600.0, 3: 0.0}],
)
def test_metric_rejects_invalid_capacities(capacities: dict[int, float]) -> None:
    """Catches incomplete or nonpositive capacity normalization."""
    with pytest.raises(MetricError, match="capacities"):
        evaluate_official(_handcase(), capacities)


def test_metric_rejects_group_with_zero_valid_rows() -> None:
    """Catches undefined official group components."""
    frame = _handcase()
    frame.loc[frame["group_id"].eq(3), "actual_kwh"] = 0.0
    with pytest.raises(MetricError, match="zero valid rows"):
        evaluate_official(frame, CAPACITIES)
