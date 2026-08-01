import numpy as np
import pandas as pd

from baram.contracts.types import FoldSpec
from baram.models.oof import generate_oof


def _synthetic_frames() -> tuple[pd.DataFrame, pd.DataFrame, FoldSpec]:
    feature_rows: list[dict[str, object]] = []
    label_rows: list[dict[str, object]] = []
    batches = [f"b{idx}" for idx in range(6)]
    for batch_index, batch in enumerate(batches):
        for slot in range(3):
            timestamp = pd.Timestamp("2023-01-01") + np.timedelta64(batch_index * 3 + slot + 1, "h")
            forecast_id = f"train-{timestamp:%Y%m%d%H}"
            for group, capacity in ((1, 21600.0), (2, 21600.0), (3, 21000.0)):
                feature_rows.append(
                    {
                        "forecast_id": forecast_id,
                        "forecast_kst_dtm": timestamp,
                        "issuance_batch": batch,
                        "group_id": group,
                        "capacity_kwh": capacity,
                        "x": float(batch_index + slot / 10 + group / 100),
                        "z": float(group),
                    }
                )
                actual = capacity * (0.1 + 0.03 * batch_index + 0.01 * slot)
                if group == 3 and batch_index == 0:
                    actual = np.nan
                label_rows.append(
                    {
                        "forecast_id": forecast_id,
                        "forecast_kst_dtm": timestamp,
                        "group_id": group,
                        "actual_kwh": actual,
                    }
                )
    fold = FoldSpec(
        fold_id="dev-2023-Q2",
        train_batches=("b0", "b1", "b2", "b3"),
        validation_batches=("b4", "b5"),
        eligible_groups=(1, 2, 3),
        official_total_eligible=True,
    )
    return pd.DataFrame(feature_rows), pd.DataFrame(label_rows), fold


def test_group_model_one_fold_smoke_has_genuine_oof_keys() -> None:
    """Catches OOF rows that were included in their estimator fit set."""
    features, labels, fold = _synthetic_frames()
    result = generate_oof(
        features,
        labels,
        (fold,),
        ("x", "z"),
        family="random_forest",
        architecture="group_specific",
        params={},
        seed=42,
        n_jobs=1,
    )
    predictions = result.predictions
    assert len(predictions) == 18
    assert not predictions.duplicated(["forecast_id", "group_id", "model_id"]).any()
    assert set(predictions["forecast_id"]).isdisjoint(result.training_forecast_ids[fold.fold_id])
    assert predictions["actual_kwh"].notna().all()


def test_shared_model_one_fold_smoke_preserves_all_groups() -> None:
    """Catches shared-model group metadata or capacity inversion loss."""
    features, labels, fold = _synthetic_frames()
    params = {
        "objective": "l1",
        "n_estimators": 20,
        "learning_rate": 0.05,
        "num_leaves": 7,
        "min_child_samples": 3,
        "subsample": 0.9,
        "colsample_bytree": 1.0,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "early_stopping_rounds": 5,
    }
    result = generate_oof(
        features,
        labels,
        (fold,),
        ("x", "z"),
        family="lightgbm",
        architecture="shared",
        params=params,
        seed=20260801,
        n_jobs=1,
    )
    assert set(result.predictions["group_id"]) == {1, 2, 3}
    assert np.isfinite(result.predictions["prediction_kwh"]).all()
    assert (result.predictions["prediction_kwh"] >= 0.0).all()
