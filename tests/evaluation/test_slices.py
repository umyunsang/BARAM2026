import pandas as pd
import pytest

from baram.evaluation.official import evaluate_group_component
from baram.evaluation.slices import attach_diagnostic_bins, score_diagnostic_slices

CAPACITIES = {1: 21600.0, 2: 21600.0, 3: 21000.0}


def _frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group, capacity in CAPACITIES.items():
        for index, timestamp in enumerate(
            pd.to_datetime(["2023-06-30 23:00:00", "2023-07-01 00:00:00"])
        ):
            rows.append(
                {
                    "forecast_id": f"{group}-{index}",
                    "forecast_kst_dtm": timestamp,
                    "group_id": group,
                    "actual_kwh": 0.5 * capacity,
                    "prediction_kwh": (0.5 + 0.05 + 0.02 * index) * capacity,
                    "operating_day": pd.Timestamp("2023-06-30"),
                    "lead_hour": 34 + index,
                    "issuance_batch": "2023062913",
                    "nwp_missing_count": 0,
                    "wind_speed": 10.0 + index,
                }
            )
    return pd.DataFrame(rows)


def test_diagnostic_bins_keep_midnight_in_operating_month() -> None:
    """Catches calendar slices disagreeing with split operating periods."""
    result = attach_diagnostic_bins(_frame(), CAPACITIES)
    midnight = result.loc[result["forecast_kst_dtm"].dt.hour.eq(0)]
    assert midnight["operating_month"].eq(6).all()
    assert midnight["target_hour"].eq(0).all()
    assert set(result["settlement_tier"]) == {"unit_4", "unit_3"}


def test_group_slice_uses_same_official_component() -> None:
    """Catches diagnostic scoring formula drift from the official primitive."""
    frame = attach_diagnostic_bins(_frame(), CAPACITIES)
    report = score_diagnostic_slices(frame, CAPACITIES, ("target_hour",))
    assert report["diagnostic_only"].all()
    group_one = frame.loc[frame["group_id"].eq(1)]
    component = evaluate_group_component(
        group_one[
            [
                "forecast_id",
                "forecast_kst_dtm",
                "group_id",
                "actual_kwh",
                "prediction_kwh",
            ]
        ],
        1,
        21600.0,
    )
    pooled = score_diagnostic_slices(frame, CAPACITIES, ("all",))
    group_row = pooled.loc[pooled["group_id"].eq(1)].iloc[0]
    assert group_row["nmae"] == pytest.approx(component.nmae)
    assert group_row["ficr"] == pytest.approx(component.ficr)
