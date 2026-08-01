import pandas as pd
import pytest

from baram.evaluation.failure_slices import shared_failure_slices
from baram.exceptions import ContractError


def _prediction(scale: float) -> pd.DataFrame:
    rows = []
    for index in range(8):
        timestamp = pd.Timestamp("2023-01-01 01:00") + pd.Timedelta(index, unit="h")
        for group_id in (1, 2, 3):
            actual = 10000.0
            error = (1000.0 if index >= 6 else 100.0) * scale
            rows.append(
                {
                    "forecast_id": f"f{index}",
                    "forecast_kst_dtm": timestamp,
                    "group_id": group_id,
                    "actual_kwh": actual,
                    "prediction_kwh": actual + error,
                }
            )
    return pd.DataFrame(rows)


def test_shared_failure_slice_requires_same_named_bucket_across_finalists() -> None:
    """Catches aggregating different high-error rows into a false shared failure mode."""
    predictions = {"a": _prediction(1.0), "b": _prediction(1.1), "c": _prediction(0.9)}
    context = predictions["a"][["forecast_id", "forecast_kst_dtm", "group_id"]].copy()
    context["lead_hour"] = [12 + index // 3 for index in range(len(context))]
    context["wind_speed"] = 5.0
    context["nwp_missing_count"] = 0
    report = shared_failure_slices(predictions, context, {1: 21600.0, 2: 21600.0, 3: 21000.0})
    passing = {item["slice_id"]: item for item in report["passing_slices"]}
    assert "lead_bin=18-23" in passing
    assert passing["lead_bin=18-23"]["minimum_error_mass_fraction"] >= 0.25
    assert report["aligned_keys_sha256"]


def test_shared_failure_slice_rejects_misaligned_finalist_keys() -> None:
    predictions = {"a": _prediction(1.0), "b": _prediction(1.0).iloc[:-1]}
    context = _prediction(1.0)[["forecast_id", "forecast_kst_dtm", "group_id"]].copy()
    context["lead_hour"] = 12
    context["wind_speed"] = 5.0
    context["nwp_missing_count"] = 0
    with pytest.raises(ContractError, match="aligned"):
        shared_failure_slices(
            predictions,
            context,
            {1: 21600.0, 2: 21600.0, 3: 21000.0},
        )
