import numpy as np
import pandas as pd
import pytest

from baram.exceptions import ModelError
from baram.models.baselines import (
    fit_supplied_rf_bundle,
    make_supplied_rf,
    predict_bundle,
)


def test_random_forest_baseline_configuration() -> None:
    """Catches drift from the supplied notebook's declared estimator."""
    model = make_supplied_rf(seed=42, n_jobs=1)
    assert model.n_estimators == 120
    assert model.max_depth == 14
    assert model.min_samples_leaf == 8
    assert model.max_features == "sqrt"
    assert model.random_state == 42
    assert model.n_jobs == 1


def test_supplied_rf_bundle_is_deterministic_and_train_imputed() -> None:
    """Catches nondeterminism or validation-fitted imputation in the control."""
    features = pd.DataFrame(
        {
            "f1": np.linspace(0.0, 1.0, 40),
            "f2": [np.nan if idx % 7 == 0 else float(idx) for idx in range(40)],
        }
    )
    target = pd.Series(100.0 + 500.0 * features["f1"], name="actual_kwh")
    first = fit_supplied_rf_bundle(
        features,
        target,
        ("f1", "f2"),
        fold_id="fold-a",
        group_id=1,
        capacity=21600.0,
        seed=42,
        n_jobs=1,
    )
    second = fit_supplied_rf_bundle(
        features,
        target,
        ("f1", "f2"),
        fold_id="fold-a",
        group_id=1,
        capacity=21600.0,
        seed=42,
        n_jobs=1,
    )
    valid = pd.DataFrame({"f1": [0.5, 0.9], "f2": [np.nan, 10000.0]})
    assert first.feature_state.medians["f2"] == pytest.approx(features["f2"].median())
    assert np.array_equal(
        predict_bundle(first, valid, "fold-a"),
        predict_bundle(second, valid, "fold-a"),
    )
    assert (predict_bundle(first, valid, "fold-a") >= 0.0).all()


def test_model_bundle_rejects_feature_order_skew() -> None:
    """Catches inference with reordered columns."""
    features = pd.DataFrame({"a": [0.0, 1.0] * 10, "b": np.arange(20.0)})
    target = pd.Series(np.arange(20.0) + 1.0)
    bundle = fit_supplied_rf_bundle(
        features,
        target,
        ("a", "b"),
        "fold-a",
        1,
        21600.0,
        42,
        1,
    )
    with pytest.raises(ModelError, match="feature columns"):
        predict_bundle(bundle, features[["b", "a"]], "fold-a")
