import numpy as np
import pandas as pd
import pytest

from baram.contracts.types import CalibrationPolicy
from baram.decisions.calibrate import (
    apply_calibration,
    cross_fit_calibration,
    fit_group_calibration,
)
from baram.exceptions import ContractError


def _group_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "forecast_id": [f"a{idx}" for idx in range(4)],
            "forecast_kst_dtm": pd.date_range("2023-01-01 01:00", periods=4, freq="h"),
            "group_id": [1] * 4,
            "actual_kwh": [10800.0] * 4,
            "prediction_kwh": [10800.0] * 4,
            "model_id": ["m"] * 4,
        }
    )


def test_apply_calibration_obeys_group_and_cap_policy() -> None:
    """Catches wrong-group policy use or missing nonnegative/cap bounds."""
    policy = CalibrationPolicy(1, 1.04, 0.02, "capacity", "a" * 64, ("m",), "b" * 64, "c" * 64)
    adjusted = apply_calibration(np.array([-100.0, 30000.0]), 1, 21600.0, policy)
    assert adjusted.tolist() == [328.0, 21600.0]
    with pytest.raises(ContractError, match="another group"):
        apply_calibration(np.array([1.0]), 2, 21600.0, policy)


def test_perfect_predictions_choose_identity_calibration() -> None:
    """Catches a tie-break selecting needless scale, offset, or clipping."""
    policy = fit_group_calibration(_group_frame(), 1, 21600.0, "1" * 64, "2" * 64)
    assert policy.scale == 1.0
    assert policy.offset_capacity == 0.0
    assert policy.cap_mode == "nonnegative_only"


def test_cross_fit_calibration_excludes_first_fold() -> None:
    """Catches fitting and scoring a decision policy on the same OOF fold."""
    rows: list[pd.DataFrame] = []
    for index, fold in enumerate(("q2", "q3", "q4")):
        part = _group_frame()
        part["forecast_id"] = [f"{fold}-{idx}" for idx in range(len(part))]
        part["forecast_kst_dtm"] += np.timedelta64(index * 10, "D")
        part["fold_id"] = fold
        rows.append(part)
    result = cross_fit_calibration(
        pd.concat(rows, ignore_index=True),
        ("q2", "q3", "q4"),
        {1: 21600.0},
        "2" * 64,
    )
    assert set(result["fold_id"]) == {"q3", "q4"}
    assert not result["forecast_id"].str.startswith("q2-").any()
