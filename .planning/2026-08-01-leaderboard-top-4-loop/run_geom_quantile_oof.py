"""Complete the geometric quantile candidate on frozen 2023 OOF folds."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import lightgbm
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from run_sequence_classifier import (
    BASELINE,
    BASELINE_SHA,
    CAPACITIES,
    METRIC_COLUMNS,
    OPEN,
    OPEN_SHA,
    OUTPUT,
    _sha256,
    _surface,
)
from sklearn.model_selection import KFold

from baram.evaluation.official import evaluate_official

LEVELS = np.asarray([0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90])
MIDPOINTS = (np.arange(39, dtype=float) + 0.5) / 39.0
WIDTH_SCALES = (0.50, 0.75, 1.00, 1.25, 1.50)
GAMMAS = (0.25, 0.50, 0.75, 1.00, 1.25)
FOLDS = ("dev-2023-Q2", "dev-2023-Q3", "dev-2023-Q4")


def _auxiliary_prediction(
    surface: pd.DataFrame,
    auxiliary_columns: list[str],
    preceding: pd.Series,
    validation: np.ndarray,
) -> np.ndarray:
    matrix = surface[auxiliary_columns].astype("float32")
    training = preceding & surface["scada_ws"].notna()
    positions = np.flatnonzero(training.to_numpy())
    prediction = np.full(len(surface), np.nan, dtype="float32")
    params = {
        "objective": "l2",
        "n_estimators": 400,
        "learning_rate": 0.04,
        "num_leaves": 31,
        "min_child_samples": 60,
        "subsample": 0.9,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 2.0,
        "random_state": 20260801,
        "n_jobs": 6,
        "deterministic": True,
        "force_col_wise": True,
        "verbosity": -1,
    }
    splitter = KFold(3, shuffle=True, random_state=20260801)
    for fit_index, holdout_index in splitter.split(positions):
        model = LGBMRegressor(**params)
        model.fit(
            matrix.iloc[positions[fit_index]],
            surface["scada_ws"].iloc[positions[fit_index]],
        )
        prediction[positions[holdout_index]] = model.predict(
            matrix.iloc[positions[holdout_index]]
        )
    model = LGBMRegressor(**params)
    model.fit(matrix.loc[training], surface.loc[training, "scada_ws"])
    prediction[validation] = model.predict(matrix.loc[validation])
    return prediction


def _fit_quantiles(
    matrix: pd.DataFrame,
    normalized_target: pd.Series,
    weights: pd.Series,
    training: pd.Series,
    data_available: pd.Series,
    validation: np.ndarray,
) -> tuple[np.ndarray, dict[str, int]]:
    batches = data_available.loc[training].drop_duplicates().sort_values()
    cutoff = batches.iloc[int(len(batches) * 0.80)]
    inner_fit = training & data_available.lt(cutoff)
    inner_stop = training & ~inner_fit
    params = {
        "objective": "quantile",
        "n_estimators": 1600,
        "learning_rate": 0.04,
        "num_leaves": 15,
        "min_child_samples": 80,
        "subsample": 0.9,
        "subsample_freq": 1,
        "colsample_bytree": 1.0,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 20260801,
        "n_jobs": 6,
        "deterministic": True,
        "force_col_wise": True,
        "verbosity": -1,
    }
    predictions: list[np.ndarray] = []
    iterations: dict[str, int] = {}
    for level in LEVELS:
        started = time.perf_counter()
        model = LGBMRegressor(**{**params, "alpha": float(level)})
        model.fit(
            matrix.loc[inner_fit],
            normalized_target.loc[inner_fit],
            sample_weight=weights.loc[inner_fit],
            eval_set=[(matrix.loc[inner_stop], normalized_target.loc[inner_stop])],
            eval_sample_weight=[weights.loc[inner_stop]],
            callbacks=[lightgbm.early_stopping(100, verbose=False)],
        )
        selected = max(1, int(model.best_iteration_ or params["n_estimators"]))
        model = LGBMRegressor(
            **{**params, "alpha": float(level), "n_estimators": selected}
        )
        model.fit(
            matrix.loc[training],
            normalized_target.loc[training],
            sample_weight=weights.loc[training],
        )
        predictions.append(model.predict(matrix.loc[validation]))
        iterations[str(level)] = selected
        print(
            json.dumps(
                {
                    "quantile": level,
                    "iteration": selected,
                    "seconds": round(time.perf_counter() - started, 2),
                }
            ),
            flush=True,
        )
    return np.maximum.accumulate(np.column_stack(predictions), axis=1), iterations


def _policy_predictions(
    quantiles: np.ndarray,
    group_ids: np.ndarray,
    mean_generation: dict[int, float],
) -> dict[str, np.ndarray]:
    samples = np.vstack([np.interp(MIDPOINTS, LEVELS, row) for row in quantiles])
    median = quantiles[:, 3]
    offsets = np.arange(-0.15, 0.151, 0.0025)
    results: dict[str, np.ndarray] = {}
    for width_scale in WIDTH_SCALES:
        scaled = np.clip(
            median[:, None] + width_scale * (samples - median[:, None]), 0.075, 1.075
        )
        for gamma in GAMMAS:
            chosen = np.empty(len(quantiles), dtype=float)
            for group_id in CAPACITIES:
                positions = np.flatnonzero(group_ids == group_id)
                for lower in range(0, len(positions), 256):
                    index = positions[lower : lower + 256]
                    actions = np.clip(
                        median[index, None] + offsets[None, :], 0.075, 1.075
                    )
                    error = np.abs(actions[:, :, None] - scaled[index, None, :])
                    units = np.select(
                        [error <= 0.06, error <= 0.08], [4.0, 3.0], default=0.0
                    )
                    utility = -error.mean(axis=2) + gamma * (
                        scaled[index, None, :] * units
                    ).mean(axis=2) / (4.0 * mean_generation[group_id])
                    chosen[index] = actions[np.arange(len(index)), np.argmax(utility, axis=1)]
            results[f"W{width_scale:g}_G{gamma:g}"] = chosen
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", nargs="+", choices=FOLDS, required=True)
    args = parser.parse_args()
    assert _sha256(OPEN) == OPEN_SHA and _sha256(BASELINE) == BASELINE_SHA
    surface, base_columns, auxiliary_columns = _surface()
    normalized_target = surface["actual_kwh"] / surface["group_id"].map(CAPACITIES)
    weights = normalized_target.clip(lower=1e-8)
    reference = pd.read_parquet(OUTPUT / "M50_GEOM_CLASS_TEMP075_G05-oof.parquet")
    for fold_id in args.folds:
        started = time.perf_counter()
        fold_reference = reference.loc[reference["fold_id"].eq(fold_id)]
        keys = set(zip(fold_reference["forecast_id"], fold_reference["group_id"], strict=True))
        validation = np.asarray(
            [
                (forecast_id, group_id) in keys
                for forecast_id, group_id in zip(
                    surface["forecast_id"], surface["group_id"], strict=True
                )
            ]
        )
        preceding = surface["forecast_kst_dtm"].lt(fold_reference["forecast_kst_dtm"].min())
        auxiliary = _auxiliary_prediction(
            surface, auxiliary_columns, preceding, validation
        )
        matrix = surface[base_columns].astype("float32")
        matrix["aux_scada_ws"] = auxiliary
        matrix["aux_scada_ws2"] = auxiliary**2
        matrix["aux_scada_ws3"] = auxiliary**3
        training = preceding & surface["actual_kwh"].notna() & normalized_target.ge(0.10)
        quantiles, iterations = _fit_quantiles(
            matrix,
            normalized_target,
            weights,
            training,
            surface["data_available_kst_dtm"],
            validation,
        )
        base = surface.loc[
            validation, ["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh"]
        ].copy()
        group_ids = base["group_id"].to_numpy(dtype=int)
        mean_generation = {
            group_id: float(
                normalized_target.loc[training & surface["group_id"].eq(group_id)].mean()
            )
            for group_id in CAPACITIES
        }
        policies = _policy_predictions(quantiles, group_ids, mean_generation)
        scores: dict[str, dict[str, float]] = {}
        for policy_id, normalized_prediction in policies.items():
            candidate = base.copy()
            candidate["prediction_kwh"] = (
                normalized_prediction
                * candidate["group_id"].map(CAPACITIES).to_numpy(dtype=float)
            )
            score = evaluate_official(candidate[METRIC_COLUMNS], CAPACITIES)
            scores[policy_id] = {
                "total": score.total,
                "one_minus_nmae": score.one_minus_nmae,
                "ficr": score.ficr,
            }
        path = OUTPUT / f"M61_GEOM_QUANTILES-{fold_id}.npz"
        np.savez_compressed(
            path,
            quantiles=quantiles,
            levels=LEVELS,
            policy_ids=np.asarray(list(policies)),
            policy_predictions=np.column_stack(list(policies.values())),
        )
        print(
            json.dumps(
                {
                    "fold_id": fold_id,
                    "feature_count": matrix.shape[1],
                    "iterations": iterations,
                    "best_policy": max(scores, key=lambda name: scores[name]["total"]),
                    "scores": scores,
                    "artifact": str(path.relative_to(Path.cwd())),
                    "artifact_sha256": _sha256(path),
                    "seconds": round(time.perf_counter() - started, 2),
                    "new_2024_evaluation": False,
                    "lockbox_reopened": False,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
