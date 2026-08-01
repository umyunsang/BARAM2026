from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from baram.contracts.hashing import canonical_sha256
from baram.contracts.types import FoldSpec
from baram.exceptions import ModelError
from baram.models.baselines import predict_bundle
from baram.models.challengers import (
    expand_challenger_grid,
    fit_challenger_bundle,
    make_catboost,
    make_xgb,
    split_inner_stopping_batches,
)
from baram.models.oof import generate_oof


def test_declared_challenger_grid_has_four_unique_configs_per_family() -> None:
    """Catches silent expansion beyond the eight authorized configurations."""
    grids = expand_challenger_grid(Path("configs/models/challengers.yaml"))
    assert set(grids) == {"catboost", "xgboost"}
    for configs in grids.values():
        assert len(configs) == 4
        assert len({canonical_sha256(config) for config in configs}) == 4


def test_factories_bind_cpu_seed_worker_cap_and_no_catboost_side_files() -> None:
    """Catches nondeterministic, GPU, over-budget, or side-effecting factories."""
    grids = expand_challenger_grid(Path("configs/models/challengers.yaml"))
    xgb = make_xgb(grids["xgboost"][0], seed=20260801, n_jobs=8)
    assert xgb.get_params()["random_state"] == 20260801
    assert xgb.get_params()["n_jobs"] == 6
    assert xgb.get_params()["device"] == "cpu"
    cat = make_catboost(grids["catboost"][0], seed=20260801, n_jobs=8)
    assert cat.get_params()["random_seed"] == 20260801
    assert cat.get_params()["thread_count"] == 6
    assert cat.get_params()["task_type"] == "CPU"
    assert cat.get_params()["allow_writing_files"] is False


def test_inner_stopping_split_keeps_whole_issuance_batches() -> None:
    """Catches row-wise inner early stopping or overlap with outer-training refit."""
    batches = pd.Series([f"b{index // 3}" for index in range(30)])
    fit_mask, stop_mask = split_inner_stopping_batches(batches)
    assert not (fit_mask & stop_mask).any()
    assert (fit_mask | stop_mask).all()
    fit_batches = set(batches.loc[fit_mask])
    stop_batches = set(batches.loc[stop_mask])
    assert fit_batches.isdisjoint(stop_batches)
    assert stop_batches == {"b8", "b9"}


@pytest.mark.parametrize("family", ["xgboost", "catboost"])
def test_challenger_bundle_normalizes_inverts_and_rejects_feature_order(
    family: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catches normalization, inner-stop/refit, feature skew, and CatBoost side files."""
    monkeypatch.chdir(tmp_path)
    rows = 60
    features = pd.DataFrame(
        {
            "x": np.linspace(0.0, 1.0, rows),
            "z": np.sin(np.linspace(0.0, 3.0, rows)),
        }
    )
    target = pd.Series(21600.0 * (0.1 + 0.5 * features["x"]))
    batches = pd.Series([f"b{index // 6:02d}" for index in range(rows)])
    params = expand_challenger_grid(
        Path("/Users/um-yunsang/BARAM2026/configs/models/challengers.yaml")
    )[family][0]
    params = {
        **params,
        "n_estimators" if family == "xgboost" else "iterations": 30,
        "early_stopping_rounds": 5,
    }
    bundle = fit_challenger_bundle(
        family,
        features,
        target,
        batches,
        ("x", "z"),
        "fold-a",
        1,
        21600.0,
        params,
        20260801,
        1,
    )
    prediction = predict_bundle(bundle, features.iloc[:5], "fold-a")
    assert prediction.shape == (5,)
    assert np.isfinite(prediction).all()
    assert (prediction >= 0.0).all()
    assert not (tmp_path / "catboost_info").exists()
    with pytest.raises(ModelError, match="feature columns/order"):
        predict_bundle(bundle, features.iloc[:5][["z", "x"]], "fold-a")


def test_xgboost_shared_oof_uses_disjoint_outer_batches() -> None:
    """Catches a challenger bypassing the genuine OOF engine or capacity inversion."""
    feature_rows: list[dict[str, object]] = []
    label_rows: list[dict[str, object]] = []
    for batch_index in range(6):
        for slot in range(2):
            timestamp = pd.Timestamp("2023-01-01 01:00") + pd.Timedelta(
                batch_index * 2 + slot, unit="h"
            )
            for group_id, capacity in ((1, 21600.0), (2, 21600.0), (3, 21000.0)):
                key = {
                    "forecast_id": f"f-{timestamp:%Y%m%d%H}",
                    "forecast_kst_dtm": timestamp,
                    "group_id": group_id,
                }
                feature_rows.append(
                    {
                        **key,
                        "issuance_batch": f"b{batch_index}",
                        "capacity_kwh": capacity,
                        "x": float(batch_index + group_id / 10),
                    }
                )
                label_rows.append({**key, "actual_kwh": capacity * (0.15 + 0.03 * batch_index)})
    fold = FoldSpec(
        fold_id="dev-2023-Q2",
        train_batches=("b0", "b1", "b2", "b3"),
        validation_batches=("b4", "b5"),
        eligible_groups=(1, 2, 3),
        official_total_eligible=True,
    )
    params = expand_challenger_grid(
        Path("/Users/um-yunsang/BARAM2026/configs/models/challengers.yaml")
    )["xgboost"][0]
    params = {**params, "n_estimators": 30, "early_stopping_rounds": 5}
    result = generate_oof(
        pd.DataFrame(feature_rows),
        pd.DataFrame(label_rows),
        (fold,),
        ("x",),
        family="xgboost",  # type: ignore[arg-type]
        architecture="shared",
        params=params,
        seed=20260801,
        n_jobs=1,
    )
    assert len(result.predictions) == 12
    assert set(result.predictions["forecast_id"]).isdisjoint(
        result.training_forecast_ids[fold.fold_id]
    )
    assert np.isfinite(result.predictions["prediction_kwh"]).all()
