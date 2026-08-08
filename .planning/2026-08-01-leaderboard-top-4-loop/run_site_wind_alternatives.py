"""Screen alternative official-data-only NWP-to-site-wind regressors."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from run_sequence_classifier import (
    BASELINE,
    BASELINE_SHA,
    OPEN,
    OPEN_SHA,
    OUTPUT,
    _sha256,
    _surface,
)
from run_site_wind_teacher import _all_weather_columns, _metrics, _validation_mask
from xgboost import XGBRegressor

FOLDS = ("dev-2023-Q2", "dev-2023-Q3", "dev-2023-Q4")


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _split_masks(
    surface: pd.DataFrame,
    preceding: np.ndarray,
    group_id: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    group = surface["group_id"].eq(group_id).to_numpy()
    training = preceding & group & surface["scada_ws"].notna().to_numpy()
    batches = (
        surface.loc[training, "data_available_kst_dtm"].drop_duplicates().sort_values()
    )
    cutoff = batches.iloc[int(len(batches) * 0.80)]
    stop = training & surface["data_available_kst_dtm"].ge(cutoff).to_numpy()
    return training, training & ~stop, stop


def _fit_catboost(
    matrix: pd.DataFrame,
    surface: pd.DataFrame,
    preceding: np.ndarray,
    validation: np.ndarray,
) -> tuple[np.ndarray, dict[int, int]]:
    prediction = np.full(len(surface), np.nan, dtype="float32")
    iterations: dict[int, int] = {}
    for group_id in (1, 2, 3):
        training, fit, stop = _split_masks(surface, preceding, group_id)
        probe = CatBoostRegressor(
            loss_function="RMSE",
            eval_metric="MAE",
            iterations=1600,
            learning_rate=0.03,
            depth=8,
            l2_leaf_reg=5.0,
            random_strength=0.3,
            random_seed=20260802,
            thread_count=6,
            allow_writing_files=False,
            verbose=False,
        )
        probe.fit(
            matrix.loc[fit],
            surface.loc[fit, "scada_ws"],
            eval_set=(matrix.loc[stop], surface.loc[stop, "scada_ws"]),
            early_stopping_rounds=100,
            verbose=False,
        )
        selected = max(1, int(probe.get_best_iteration()) + 1)
        model = CatBoostRegressor(
            loss_function="RMSE",
            iterations=selected,
            learning_rate=0.03,
            depth=8,
            l2_leaf_reg=5.0,
            random_strength=0.3,
            random_seed=20260802,
            thread_count=6,
            allow_writing_files=False,
            verbose=False,
        )
        model.fit(matrix.loc[training], surface.loc[training, "scada_ws"], verbose=False)
        apply = validation & surface["group_id"].eq(group_id).to_numpy()
        prediction[apply] = model.predict(matrix.loc[apply])
        iterations[group_id] = selected
    return prediction, iterations


def _fit_xgboost(
    matrix: pd.DataFrame,
    surface: pd.DataFrame,
    preceding: np.ndarray,
    validation: np.ndarray,
) -> tuple[np.ndarray, dict[int, int]]:
    prediction = np.full(len(surface), np.nan, dtype="float32")
    iterations: dict[int, int] = {}
    for group_id in (1, 2, 3):
        training, fit, stop = _split_masks(surface, preceding, group_id)
        probe = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=1800,
            learning_rate=0.025,
            max_depth=7,
            min_child_weight=20.0,
            subsample=0.9,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=5.0,
            max_bin=256,
            tree_method="hist",
            early_stopping_rounds=100,
            random_state=20260802,
            n_jobs=6,
        )
        probe.fit(
            matrix.loc[fit],
            surface.loc[fit, "scada_ws"],
            eval_set=[(matrix.loc[stop], surface.loc[stop, "scada_ws"])],
            verbose=False,
        )
        selected = max(1, int(probe.best_iteration) + 1)
        model = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=selected,
            learning_rate=0.025,
            max_depth=7,
            min_child_weight=20.0,
            subsample=0.9,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=5.0,
            max_bin=256,
            tree_method="hist",
            random_state=20260802,
            n_jobs=6,
        )
        model.fit(matrix.loc[training], surface.loc[training, "scada_ws"], verbose=False)
        apply = validation & surface["group_id"].eq(group_id).to_numpy()
        prediction[apply] = model.predict(matrix.loc[apply])
        iterations[group_id] = selected
    return prediction, iterations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--fold", choices=FOLDS, default=FOLDS[-1])
    parser.add_argument(
        "--families", nargs="+", choices=("catboost", "xgboost"), required=True
    )
    args = parser.parse_args()
    if _sha256(OPEN) != OPEN_SHA or _sha256(BASELINE) != BASELINE_SHA:
        raise RuntimeError("immutable competition input hash mismatch")
    started = time.perf_counter()
    surface, _, _ = _surface()
    validation = _validation_mask(surface, args.fold)
    start = surface.loc[validation, "forecast_kst_dtm"].min()
    preceding = surface["forecast_kst_dtm"].lt(start).to_numpy()
    columns = _all_weather_columns(surface)
    matrix = surface[columns].astype("float32")
    output = surface.loc[
        validation,
        ["forecast_id", "forecast_kst_dtm", "group_id", "scada_ws"],
    ].copy()
    receipts: dict[str, object] = {}
    for family in args.families:
        family_started = time.perf_counter()
        if family == "catboost":
            prediction, iterations = _fit_catboost(
                matrix, surface, preceding, validation
            )
        else:
            prediction, iterations = _fit_xgboost(
                matrix, surface, preceding, validation
            )
        values = prediction[validation]
        output[family] = values
        metrics = {
            "pooled": _metrics(output["scada_ws"].to_numpy(), values),
            "groups": {
                str(group_id): _metrics(
                    output.loc[output["group_id"].eq(group_id), "scada_ws"].to_numpy(),
                    output.loc[output["group_id"].eq(group_id), family].to_numpy(),
                )
                for group_id in (1, 2, 3)
            },
        }
        receipts[family] = {
            "feature_count": len(columns),
            "selected_iterations": iterations,
            "metrics": metrics,
            "runtime_seconds": round(time.perf_counter() - family_started, 2),
        }
        print(json.dumps({"family": family, **receipts[family]}), flush=True)

    output_path = OUTPUT / f"{args.candidate_id}-{args.fold}-site-wind.parquet"
    output.to_parquet(output_path, index=False)
    receipt = {
        "schema_version": 1,
        "candidate_id": args.candidate_id,
        "fold_id": args.fold,
        "scope": "2023 development-fold alternative site-wind diagnostic only",
        "families": receipts,
        "runtime_seconds": round(time.perf_counter() - started, 2),
        "prediction_path": str(output_path.relative_to(Path.cwd())),
        "prediction_sha256": _file_sha(output_path),
        "observed_validation_scada_used_for_power_prediction": False,
        "new_2024_evaluation": False,
        "lockbox_reopened": False,
        "external_actions": [],
    }
    receipt_path = OUTPUT / f"{args.candidate_id}-{args.fold}-site-wind.json"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
