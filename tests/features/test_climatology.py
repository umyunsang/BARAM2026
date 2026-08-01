import pandas as pd
import pytest

from baram.exceptions import LeakageError
from baram.features.climatology import apply_climatology, fit_climatology


def test_climatology_uses_train_rows_only() -> None:
    """Catches validation targets entering fold-fitted target summaries."""
    train = pd.DataFrame(
        {
            "forecast_id": ["a", "b", "c"],
            "group_id": [1, 1, 1],
            "month": [1, 1, 2],
            "hour": [1, 1, 2],
            "actual_kwh": [100.0, 300.0, 500.0],
        }
    )
    valid = pd.DataFrame(
        {
            "forecast_id": ["v"],
            "group_id": [1],
            "month": [1],
            "hour": [1],
            "actual_kwh": [9999.0],
        }
    )
    state = fit_climatology(train, "fold-a")
    transformed = apply_climatology(state, valid, "fold-a")
    assert transformed.loc[0, "clim_median"] == 200.0


def test_climatology_backoff_and_fold_identity() -> None:
    """Catches missing-key NaNs and cross-fold target-summary reuse."""
    train = pd.DataFrame(
        {
            "forecast_id": ["a", "b"],
            "group_id": [1, 1],
            "month": [1, 2],
            "hour": [1, 2],
            "actual_kwh": [100.0, 300.0],
        }
    )
    valid = pd.DataFrame({"forecast_id": ["v"], "group_id": [1], "month": [3], "hour": [4]})
    state = fit_climatology(train, "fold-a")
    assert apply_climatology(state, valid, "fold-a").loc[0, "clim_median"] == 200.0
    with pytest.raises(LeakageError, match="different fold"):
        apply_climatology(state, valid, "fold-b")
