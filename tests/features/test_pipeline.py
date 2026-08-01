import pandas as pd
import pytest

from baram.exceptions import LeakageError, ModelError
from baram.features.pipeline import fit_feature_pipeline, transform_features


def test_imputer_uses_training_rows_only() -> None:
    """Catches validation values influencing train-fitted missing-value state."""
    train = pd.DataFrame({"a": [1.0, 3.0, None], "b": [10.0, 20.0, 30.0]})
    valid = pd.DataFrame({"a": [None, 1000.0], "b": [None, 1000.0]})
    state = fit_feature_pipeline(train, ("a", "b"), "fold-a")
    result = transform_features(state, valid, "fold-a")
    assert result.loc[0, "a"] == 2.0
    assert result.loc[0, "b"] == 20.0


def test_feature_pipeline_rejects_fold_or_order_skew() -> None:
    """Catches cross-fold state reuse and final feature-order mismatch."""
    train = pd.DataFrame({"a": [1.0], "b": [2.0]})
    state = fit_feature_pipeline(train, ("a", "b"), "fold-a")
    with pytest.raises(LeakageError, match="different fold"):
        transform_features(state, train, "fold-b")
    with pytest.raises(ModelError, match="feature columns"):
        transform_features(state, train[["b", "a"]], "fold-a")
