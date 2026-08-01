import pandas as pd

from baram.decisions.utility import fit_residual_utility


def test_residual_utility_does_not_activate_without_500_rows_per_group() -> None:
    """Catches sparse residual cells creating an unstable point shift."""
    frame = pd.DataFrame(
        {
            "group_id": [1] * 10 + [2] * 10 + [3] * 10,
            "actual_kwh": [1000.0] * 30,
            "prediction_kwh": [900.0] * 30,
            "lead_bin": ["12-17"] * 30,
            "wind_speed_bin": ["medium"] * 30,
        }
    )
    decision = fit_residual_utility(frame, {1: 21600.0, 2: 21600.0, 3: 21000.0}, "1" * 64)
    assert decision.status == "NOT_ACTIVATED"
    assert decision.policy is None
