"""Screen a metric-native action-value model on chronology-safe NWP features."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from run_sequence_classifier import (
    BASELINE,
    BASELINE_SHA,
    CAPACITIES,
    OPEN,
    OPEN_SHA,
    OUTPUT,
    _score,
    _sha256,
    _surface,
)
from run_site_wind_classifier import FOLDS, _add_site_wind_features
from run_site_wind_teacher import _validation_mask

GAMMAS = (0.0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
ITERATIONS = (120, 200, 300)
BASE_COLUMNS = ["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh"]


def _feature_names(fold: str, count: int) -> list[str]:
    receipt = json.loads((OUTPUT / f"M102_TOP100-{fold}.json").read_text())
    names = receipt["selected_feature_names"][:count]
    if len(names) != count or len(set(names)) != count:
        raise RuntimeError("fixed M102 feature contract changed")
    return names


def _action_features(base: np.ndarray, actions: np.ndarray) -> np.ndarray:
    repeated = np.repeat(base, actions.shape[1], axis=0)
    flat_actions = actions.reshape(-1, 1).astype("float32")
    return np.concatenate(
        [repeated, flat_actions, flat_actions**2, flat_actions**3], axis=1
    )


def _independent_training_actions(row_count: int, count: int) -> np.ndarray:
    # Each row receives one action in every equal-width stratum.  The irrational
    # phase is based only on row position, never on the target, so action sampling
    # cannot disclose the label to the action-value regressors.
    phase = np.mod(np.arange(row_count, dtype=float) * 0.6180339887498949, 1.0)
    unit = (np.arange(count, dtype=float)[None, :] + phase[:, None]) / count
    return (0.075 + unit).astype("float32")


def _fit_models(
    base: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    action_count: int,
) -> tuple[LGBMRegressor, LGBMRegressor]:
    actions = _independent_training_actions(len(target), action_count)
    matrix = _action_features(base, actions)
    repeated_target = np.repeat(target, action_count)
    repeated_groups = np.repeat(groups, action_count)
    flat_actions = actions.reshape(-1)
    error_target = np.abs(repeated_target - flat_actions)
    units = np.select(
        [error_target <= 0.06, error_target <= 0.08],
        [4.0, 3.0],
        default=0.0,
    )
    group_means = {
        group_id: float(target[groups == group_id].mean())
        for group_id in CAPACITIES
    }
    denominators = np.asarray([group_means[int(value)] for value in repeated_groups])
    settlement_target = repeated_target * units / (4.0 * denominators)
    params = {
        "objective": "l2",
        "n_estimators": max(ITERATIONS),
        "learning_rate": 0.035,
        "num_leaves": 31,
        "min_child_samples": 120,
        "max_bin": 127,
        "subsample": 0.9,
        "subsample_freq": 1,
        "colsample_bytree": 0.9,
        "reg_alpha": 0.1,
        "reg_lambda": 4.0,
        "random_state": 20260802,
        "n_jobs": 6,
        "deterministic": True,
        "force_col_wise": True,
        "verbosity": -1,
    }
    error_model = LGBMRegressor(**params)
    settlement_model = LGBMRegressor(**params)
    error_model.fit(matrix, error_target)
    settlement_model.fit(matrix, settlement_target)
    return error_model, settlement_model


def _action_surfaces(
    base: np.ndarray,
    error_model: LGBMRegressor,
    settlement_model: LGBMRegressor,
    iteration: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    actions = np.arange(0.075, 1.0751, 0.005, dtype="float32")
    expected_error = np.empty((len(base), len(actions)), dtype="float32")
    expected_settlement = np.empty_like(expected_error)
    for start in range(0, len(actions), 20):
        stop = min(start + 20, len(actions))
        chunk = np.broadcast_to(actions[None, start:stop], (len(base), stop - start))
        matrix = _action_features(base, chunk)
        expected_error[:, start:stop] = error_model.predict(
            matrix, num_iteration=iteration
        ).reshape(len(base), stop - start)
        expected_settlement[:, start:stop] = settlement_model.predict(
            matrix, num_iteration=iteration
        ).reshape(len(base), stop - start)
    return actions, expected_error, expected_settlement


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--fold", choices=FOLDS, default=FOLDS[-1])
    parser.add_argument("--sitewind-cache-id", required=True)
    parser.add_argument("--feature-count", type=int, choices=(30, 50, 100), default=50)
    parser.add_argument("--training-action-count", type=int, default=20)
    args = parser.parse_args()
    if not 8 <= args.training_action_count <= 40:
        raise ValueError("training-action-count must be between eight and forty")
    if _sha256(OPEN) != OPEN_SHA or _sha256(BASELINE) != BASELINE_SHA:
        raise RuntimeError("immutable competition input hash mismatch")
    started = time.perf_counter()
    surface, base_columns, _ = _surface()
    validation = _validation_mask(surface, args.fold)
    validation_start = surface.loc[validation, "forecast_kst_dtm"].min()
    normalized_target = surface["actual_kwh"] / surface["group_id"].map(CAPACITIES)
    training = (
        surface["forecast_kst_dtm"].lt(validation_start).to_numpy()
        & surface["actual_kwh"].notna().to_numpy()
        & normalized_target.ge(0.10).to_numpy()
    )
    cached = np.load(
        OUTPUT / f"{args.sitewind_cache_id}-{args.fold}-sitewind-features.npz"
    )
    matrix = surface[base_columns].astype("float32")
    sitewind_columns = _add_site_wind_features(
        matrix, cached["legacy"], cached["allweather"]
    )
    selected_features = _feature_names(args.fold, args.feature_count)
    missing = set(selected_features).difference(matrix.columns)
    if missing:
        raise RuntimeError(f"missing fixed features: {sorted(missing)}")
    matrix = matrix[selected_features]
    error_model, settlement_model = _fit_models(
        matrix.loc[training].to_numpy(dtype="float32"),
        normalized_target.loc[training].to_numpy(dtype=float),
        surface.loc[training, "group_id"].to_numpy(dtype=int),
        args.training_action_count,
    )
    validation_matrix = matrix.loc[validation].to_numpy(dtype="float32")
    base = surface.loc[validation, BASE_COLUMNS].copy()
    capacity = base["group_id"].map(CAPACITIES).to_numpy(dtype=float)
    sweep: dict[str, object] = {}
    best: tuple[float, int, float, pd.DataFrame] | None = None
    for iteration in ITERATIONS:
        actions, expected_error, expected_settlement = _action_surfaces(
            validation_matrix, error_model, settlement_model, iteration
        )
        iteration_scores: dict[str, dict[str, float]] = {}
        for gamma in GAMMAS:
            utility = -expected_error + gamma * expected_settlement
            normalized_prediction = actions[np.argmax(utility, axis=1)]
            output = base.copy()
            output["prediction_kwh"] = normalized_prediction * capacity
            score = _score(output)
            iteration_scores[f"G{gamma:g}"] = score
            choice = (score["total"], iteration, gamma, output)
            if best is None or choice[0] > best[0]:
                best = choice
        best_policy = max(
            iteration_scores,
            key=lambda name: iteration_scores[name]["total"],
        )
        sweep[str(iteration)] = {
            "best_policy": best_policy,
            "best_score": iteration_scores[best_policy],
            "scores": iteration_scores,
        }
        print(
            json.dumps(
                {
                    "iteration": iteration,
                    "best_policy": best_policy,
                    "best_score": iteration_scores[best_policy],
                }
            ),
            flush=True,
        )
    assert best is not None
    output = best[3]
    output["fold_id"] = args.fold
    output["model_id"] = args.candidate_id
    output_path = OUTPUT / f"{args.candidate_id}-{args.fold}.parquet"
    output.to_parquet(output_path, index=False)
    receipt = {
        "schema_version": 1,
        "candidate_id": args.candidate_id,
        "fold_id": args.fold,
        "architecture": "metric_native_action_value_regression",
        "scope": "same-fold representation screen; action samples target-independent",
        "feature_count": args.feature_count,
        "selected_feature_names": selected_features,
        "sitewind_feature_count": len(sitewind_columns),
        "training_action_count": args.training_action_count,
        "action_grid_step": 0.005,
        "selected_iteration": best[1],
        "selected_gamma": best[2],
        "selected_score": _score(output),
        "sweep": sweep,
        "runtime_seconds": round(time.perf_counter() - started, 2),
        "prediction_path": str(output_path.relative_to(Path.cwd())),
        "prediction_sha256": _sha256(output_path),
        "observed_scada_feature_count": 0,
        "new_2024_evaluation": False,
        "lockbox_reopened": False,
        "external_actions": [],
    }
    receipt_path = OUTPUT / f"{args.candidate_id}-{args.fold}.json"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
