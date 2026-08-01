import pandas as pd
import pytest

from baram.exceptions import LeakageError
from baram.validation.leakage import (
    assert_artifact_fold,
    assert_batch_isolation,
    assert_feature_availability,
    assert_inference_schema,
    assert_label_cutoff,
)


def test_batch_fragmentation_fails() -> None:
    """Catches a single issuance batch appearing in train and validation."""
    batch = pd.Timestamp("2023-01-01 13:00:00")
    train = pd.DataFrame({"data_available_kst_dtm": [batch] * 12})
    valid = pd.DataFrame({"data_available_kst_dtm": [batch] * 12})
    with pytest.raises(LeakageError, match="batch appears in both"):
        assert_batch_isolation(train, valid)


def test_label_cutoff_rejects_target_not_yet_observable() -> None:
    """Catches training on a target finalized at or after validation issuance."""
    labels = pd.DataFrame({"forecast_kst_dtm": pd.to_datetime(["2023-04-01 13:00:00"])})
    valid = pd.DataFrame({"data_available_kst_dtm": pd.to_datetime(["2023-04-01 13:00:00"])})
    with pytest.raises(LeakageError, match="not available"):
        assert_label_cutoff(labels, valid)


def test_feature_availability_rejects_late_publication() -> None:
    """Catches forecast data published at or after its target time."""
    frame = pd.DataFrame(
        {
            "forecast_kst_dtm": pd.to_datetime(["2023-01-01 01:00:00"]),
            "data_available_kst_dtm": pd.to_datetime(["2023-01-01 01:00:00"]),
        }
    )
    with pytest.raises(LeakageError, match="before forecast"):
        assert_feature_availability(frame)


@pytest.mark.parametrize("column", ["actual_kwh_lag1", "scada_wind", "target_rolling"])
def test_inference_schema_rejects_forbidden_columns(column: str) -> None:
    """Catches target-history or train-only SCADA leakage at inference."""
    with pytest.raises(LeakageError, match="forbidden"):
        assert_inference_schema(["lead_hour", column])


def test_artifact_fold_must_match_consumer() -> None:
    """Catches reuse of a train-fitted object across fold identities."""
    with pytest.raises(LeakageError, match="different fold"):
        assert_artifact_fold("dev-2023-Q2", "dev-2023-Q3")
