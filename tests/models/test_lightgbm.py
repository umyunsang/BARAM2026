from pathlib import Path

import numpy as np
import pandas as pd

from baram.models.baselines import predict_bundle
from baram.models.lightgbm import (
    expand_lgbm_grid,
    fit_lgbm_bundle,
    make_lgbm,
)


def _params() -> dict[str, object]:
    return {
        "objective": "l1",
        "n_estimators": 30,
        "learning_rate": 0.05,
        "num_leaves": 7,
        "min_child_samples": 3,
        "subsample": 0.9,
        "colsample_bytree": 1.0,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "early_stopping_rounds": 5,
    }


def test_declared_lightgbm_grid_has_sixteen_unique_configs() -> None:
    """Catches silent expansion or contraction of the approved search budget."""
    configs = expand_lgbm_grid(Path("configs/models/lightgbm.yaml"))
    assert len(configs) == 16
    assert len({tuple(sorted(config.items())) for config in configs}) == 16


def test_lightgbm_factory_binds_determinism_and_worker_cap() -> None:
    """Catches nondeterministic or over-budget estimator construction."""
    model = make_lgbm(_params(), seed=20260801, n_jobs=8)
    assert model.random_state == 20260801
    assert model.n_jobs == 6
    assert model.deterministic is True
    assert model.force_col_wise is True


def test_lightgbm_bundle_smoke_fit_normalizes_and_inverts_target() -> None:
    """Catches capacity normalization or inner-stop/refit plumbing failures."""
    rows = 60
    features = pd.DataFrame(
        {
            "x": np.linspace(0.0, 1.0, rows),
            "z": np.sin(np.linspace(0.0, 3.0, rows)),
        }
    )
    target = pd.Series(21600.0 * (0.1 + 0.5 * features["x"]))
    batches = pd.Series([f"b{idx // 6:02d}" for idx in range(rows)])
    bundle = fit_lgbm_bundle(
        features,
        target,
        batches,
        ("x", "z"),
        "fold-a",
        1,
        21600.0,
        _params(),
        20260801,
        1,
    )
    prediction = predict_bundle(bundle, features.iloc[:5], "fold-a")
    assert prediction.shape == (5,)
    assert np.isfinite(prediction).all()
    assert (prediction >= 0.0).all()
    assert prediction.max() <= 21600.0
