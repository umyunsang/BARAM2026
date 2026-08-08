"""Strict group-specific boosted-leaf distributions with Bayes actions."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from run_inner_policy_classifier import _group_total
from run_sequence_classifier import (
    BASELINE,
    BASELINE_SHA,
    CAPACITIES,
    OPEN,
    OPEN_SHA,
    OUTPUT,
    _score,
    _sha256,
)
from run_site_wind_teacher import _strict_preceding_mask, _validation_mask
from strict_dev_surface import DEV_CUTOFF, development_surface

BASE_COLUMNS = ["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh"]
CLASS_WIDTH = 0.01
CLASS_COUNT = 98
TREE_FRACTIONS = (0.0, 0.5, 0.75)
TEMPERATURES = (0.50, 0.75, 1.00)
ANCHOR_SIGMAS = (0.0, 0.06, 0.10, 0.16)
UTILITY_GAMMAS = (0.0, 0.50, 0.75, 1.00, 1.25, 1.50)


def _strict_before(surface: pd.DataFrame, cutoff: pd.Timestamp) -> np.ndarray:
    batch_last = surface.groupby("data_available_kst_dtm", sort=False)[
        "forecast_kst_dtm"
    ].transform("max")
    mask = batch_last.lt(cutoff).to_numpy()
    if mask.any() and not surface.loc[mask, "forecast_kst_dtm"].lt(cutoff).all():
        raise RuntimeError("inner training includes an unavailable target")
    return mask


def _centers() -> np.ndarray:
    lower = 0.10 + CLASS_WIDTH * np.arange(CLASS_COUNT, dtype=float)
    upper = np.minimum(lower + CLASS_WIDTH, 1.075)
    return (lower + upper) / 2.0


def _classes(target: np.ndarray) -> np.ndarray:
    clipped = np.clip(target, 0.10, 1.074999)
    classes = np.floor((clipped - 0.10) / CLASS_WIDTH).astype(int)
    return np.clip(classes, 0, CLASS_COUNT - 1)


def _model(iterations: int, seed: int) -> LGBMRegressor:
    return LGBMRegressor(
        objective="l1",
        n_estimators=iterations,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=45,
        max_bin=255,
        subsample=0.90,
        subsample_freq=1,
        colsample_bytree=0.82,
        reg_alpha=0.15,
        reg_lambda=4.0,
        random_state=seed,
        n_jobs=6,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )


def _screen_features(
    matrix: pd.DataFrame,
    target: pd.Series,
    training: np.ndarray,
    top_features: int,
    seed: int,
) -> list[str]:
    screen = _model(140, seed)
    weights = target.loc[training].clip(lower=0.10).to_numpy(dtype=float)
    screen.fit(matrix.loc[training], target.loc[training], sample_weight=weights)
    gains = screen.booster_.feature_importance(importance_type="gain")
    order = np.argsort(gains)[::-1]
    selected = [matrix.columns[position] for position in order[:top_features]]
    if len(selected) != top_features or len(set(selected)) != top_features:
        raise RuntimeError("boosted-leaf feature screen contract changed")
    return selected


def _fit_leaf_model(
    matrix: pd.DataFrame,
    target: pd.Series,
    training: np.ndarray,
    apply: np.ndarray,
    top_features: int,
    iterations: int,
    seed: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[str],
]:
    selected = _screen_features(matrix, target, training, top_features, seed)
    model = _model(iterations, seed + 1)
    weights = target.loc[training].clip(lower=0.10).to_numpy(dtype=float)
    model.fit(
        matrix.loc[training, selected],
        target.loc[training],
        sample_weight=weights,
    )
    train_leaf = np.asarray(
        model.predict(matrix.loc[training, selected], pred_leaf=True), dtype=int
    )
    apply_leaf = np.asarray(
        model.predict(matrix.loc[apply, selected], pred_leaf=True), dtype=int
    )
    if train_leaf.ndim == 1:
        train_leaf = train_leaf[:, None]
        apply_leaf = apply_leaf[:, None]
    if train_leaf.shape[1] != iterations or apply_leaf.shape[1] != iterations:
        raise RuntimeError("boosted-leaf iteration contract changed")
    point = np.clip(
        model.predict(matrix.loc[apply, selected]), 0.075, 1.075
    ).astype(float)
    return (
        train_leaf,
        apply_leaf,
        target.loc[training].to_numpy(dtype=float),
        point,
        selected,
    )


def _leaf_probability(
    train_leaf: np.ndarray,
    apply_leaf: np.ndarray,
    train_target: np.ndarray,
    tree_fraction: float,
) -> np.ndarray:
    first_tree = int(np.floor(train_leaf.shape[1] * tree_fraction))
    if first_tree >= train_leaf.shape[1]:
        raise ValueError("tree fraction removed every tree")
    target_class = _classes(train_target)
    probability = np.zeros((len(apply_leaf), CLASS_COUNT), dtype=float)
    tree_count = 0
    for tree in range(first_tree, train_leaf.shape[1]):
        train_ids = train_leaf[:, tree]
        apply_ids = apply_leaf[:, tree]
        leaf_count = int(max(train_ids.max(initial=0), apply_ids.max(initial=0))) + 1
        histogram = np.zeros((leaf_count, CLASS_COUNT), dtype=float)
        np.add.at(histogram, (train_ids, target_class), 1.0)
        row_sum = histogram.sum(axis=1, keepdims=True)
        if np.any(row_sum[apply_ids] <= 0):
            raise RuntimeError("validation row reached an empty training leaf")
        histogram /= np.maximum(row_sum, 1.0)
        probability += histogram[apply_ids]
        tree_count += 1
    probability /= float(tree_count)
    probability /= probability.sum(axis=1, keepdims=True)
    return probability


def _calibrate_distribution(
    probability: np.ndarray,
    point: np.ndarray,
    temperature: float,
    anchor_sigma: float,
) -> np.ndarray:
    calibrated = np.clip(probability, 1e-12, 1.0) ** (1.0 / temperature)
    if anchor_sigma > 0:
        distance = (_centers()[None, :] - point[:, None]) / anchor_sigma
        calibrated *= np.exp(-0.5 * distance**2)
    calibrated /= calibrated.sum(axis=1, keepdims=True)
    return calibrated


def _bayes_action(
    probability: np.ndarray,
    mean_generation: float,
    gamma: float,
) -> np.ndarray:
    centers = _centers()
    actions = np.arange(0.075, 1.0751, 0.0025)
    error = np.abs(actions[:, None] - centers[None, :])
    units = np.select([error <= 0.06, error <= 0.08], [4.0, 3.0], default=0.0)
    utility = -(probability @ error.T)
    if gamma > 0:
        utility += gamma * (probability @ (centers[None, :] * units).T) / (
            4.0 * mean_generation
        )
    return actions[np.argmax(utility, axis=1)]


def _select_configuration(
    train_leaf: np.ndarray,
    apply_leaf: np.ndarray,
    train_target: np.ndarray,
    point: np.ndarray,
    actual: np.ndarray,
    mean_generation: float,
) -> tuple[dict[str, float], dict[str, object]]:
    best: tuple[float, dict[str, float], dict[str, float]] | None = None
    ranking: list[tuple[float, str, dict[str, float]]] = []
    point_score = _group_total(actual, point)
    point_config = {
        "tree_fraction": -1.0,
        "temperature": 1.0,
        "anchor_sigma": 0.0,
        "utility_gamma": -1.0,
    }
    best = (point_score["total"], point_config, point_score)
    ranking.append((point_score["total"], "POINT", point_score))
    for tree_fraction in TREE_FRACTIONS:
        leaf_probability = _leaf_probability(
            train_leaf, apply_leaf, train_target, tree_fraction
        )
        for temperature in TEMPERATURES:
            for anchor_sigma in ANCHOR_SIGMAS:
                probability = _calibrate_distribution(
                    leaf_probability, point, temperature, anchor_sigma
                )
                for gamma in UTILITY_GAMMAS:
                    action = _bayes_action(probability, mean_generation, gamma)
                    score = _group_total(actual, action)
                    config = {
                        "tree_fraction": tree_fraction,
                        "temperature": temperature,
                        "anchor_sigma": anchor_sigma,
                        "utility_gamma": gamma,
                    }
                    tag = (
                        f"F{tree_fraction:g}_T{temperature:g}_"
                        f"S{anchor_sigma:g}_G{gamma:g}"
                    )
                    ranking.append((score["total"], tag, score))
                    choice = (score["total"], config, score)
                    if best is None or choice[0] > best[0]:
                        best = choice
    assert best is not None
    diagnostics: dict[str, object] = {
        "selected_inner_score": best[2],
        "top_inner_trials": [
            {"tag": tag, "score": score}
            for _, tag, score in sorted(ranking, reverse=True)[:10]
        ],
    }
    return best[1], diagnostics


def _apply_configuration(
    train_leaf: np.ndarray,
    apply_leaf: np.ndarray,
    train_target: np.ndarray,
    point: np.ndarray,
    mean_generation: float,
    configuration: dict[str, float],
) -> np.ndarray:
    if configuration["tree_fraction"] < 0:
        return point.copy()
    probability = _leaf_probability(
        train_leaf,
        apply_leaf,
        train_target,
        configuration["tree_fraction"],
    )
    probability = _calibrate_distribution(
        probability,
        point,
        configuration["temperature"],
        configuration["anchor_sigma"],
    )
    return _bayes_action(
        probability, mean_generation, configuration["utility_gamma"]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument(
        "--fold",
        choices=("dev-2023-Q2", "dev-2023-Q3", "dev-2023-Q4"),
        required=True,
    )
    parser.add_argument("--calibration-days", type=int, default=60)
    parser.add_argument("--top-features", type=int, default=160)
    parser.add_argument("--iterations", type=int, default=220)
    args = parser.parse_args()
    if not 45 <= args.calibration_days <= 120:
        raise ValueError("calibration-days must be between 45 and 120")
    if not 80 <= args.top_features <= 240:
        raise ValueError("top-features must be between 80 and 240")
    if not 120 <= args.iterations <= 320:
        raise ValueError("iterations must be between 120 and 320")
    if _sha256(OPEN) != OPEN_SHA or _sha256(BASELINE) != BASELINE_SHA:
        raise RuntimeError("immutable competition input hash mismatch")

    started = time.perf_counter()
    surface, feature_columns = development_surface()
    if surface["forecast_kst_dtm"].ge(DEV_CUTOFF).any():
        raise RuntimeError("lockbox row reached boosted-leaf runner")
    validation = _validation_mask(surface, args.fold)
    outer_history = _strict_preceding_mask(surface, validation)
    target = surface["actual_kwh"] / surface["group_id"].map(CAPACITIES)
    observed_eligible = surface["actual_kwh"].notna().to_numpy() & target.ge(
        0.10
    ).to_numpy()
    outer_training = outer_history & observed_eligible
    validation_cutoff = pd.Timestamp(
        surface.loc[validation, "data_available_kst_dtm"].min()
    )
    calibration_cutoff = validation_cutoff - np.timedelta64(
        args.calibration_days, "D"
    )
    inner_history = _strict_before(surface, calibration_cutoff)
    inner_training = inner_history & observed_eligible
    calibration = outer_training & ~inner_history
    matrix = surface[feature_columns].astype("float32")

    configurations: dict[str, dict[str, float]] = {}
    diagnostics: dict[str, object] = {}
    inner_predictions = np.full(len(surface), np.nan, dtype=float)
    final_predictions = np.full(len(surface), np.nan, dtype=float)
    selected_features: dict[str, dict[str, list[str]]] = {
        "inner": {},
        "final": {},
    }
    for group_id in CAPACITIES:
        group = surface["group_id"].eq(group_id).to_numpy()
        inner_fit = inner_training & group
        inner_apply = calibration & group
        outer_fit = outer_training & group
        outer_apply = validation & group
        if min(inner_fit.sum(), inner_apply.sum(), outer_fit.sum(), outer_apply.sum()) < 300:
            raise RuntimeError(f"group {group_id} boosted-leaf split is too small")

        inner = _fit_leaf_model(
            matrix,
            target,
            inner_fit,
            inner_apply,
            args.top_features,
            args.iterations,
            20260830 + group_id * 100,
        )
        inner_mean = float(target.loc[inner_fit].mean())
        configuration, group_diagnostics = _select_configuration(
            inner[0],
            inner[1],
            inner[2],
            inner[3],
            target.loc[inner_apply].to_numpy(dtype=float),
            inner_mean,
        )
        inner_action = _apply_configuration(
            inner[0], inner[1], inner[2], inner[3], inner_mean, configuration
        )
        inner_predictions[inner_apply] = inner_action
        selected_features["inner"][str(group_id)] = inner[4]

        final = _fit_leaf_model(
            matrix,
            target,
            outer_fit,
            outer_apply,
            args.top_features,
            args.iterations,
            20260930 + group_id * 100,
        )
        outer_mean = float(target.loc[outer_fit].mean())
        final_action = _apply_configuration(
            final[0], final[1], final[2], final[3], outer_mean, configuration
        )
        final_predictions[outer_apply] = final_action
        selected_features["final"][str(group_id)] = final[4]
        configurations[str(group_id)] = configuration
        diagnostics[str(group_id)] = {
            **group_diagnostics,
            "inner_fit_rows": int(inner_fit.sum()),
            "inner_apply_rows": int(inner_apply.sum()),
            "outer_fit_rows": int(outer_fit.sum()),
            "outer_apply_rows": int(outer_apply.sum()),
            "inner_mean_generation": inner_mean,
            "outer_mean_generation": outer_mean,
        }
        print(
            json.dumps(
                {
                    "group_id": group_id,
                    "configuration": configuration,
                    "inner_score": group_diagnostics["selected_inner_score"],
                }
            ),
            flush=True,
        )

    if not np.isfinite(inner_predictions[calibration]).all():
        raise RuntimeError("inner boosted-leaf prediction is incomplete")
    if not np.isfinite(final_predictions[validation]).all():
        raise RuntimeError("outer boosted-leaf prediction is incomplete")
    inner_base = surface.loc[calibration, BASE_COLUMNS].copy()
    inner_base["prediction_kwh"] = inner_predictions[calibration] * inner_base[
        "group_id"
    ].map(CAPACITIES).to_numpy(dtype=float)
    output = surface.loc[validation, BASE_COLUMNS].copy()
    output["prediction_kwh"] = final_predictions[validation] * output[
        "group_id"
    ].map(CAPACITIES).to_numpy(dtype=float)
    output["fold_id"] = args.fold
    output["model_id"] = args.candidate_id
    output_path = OUTPUT / f"{args.candidate_id}-{args.fold}.parquet"
    output.to_parquet(output_path, index=False)
    receipt = {
        "schema_version": 1,
        "candidate_id": args.candidate_id,
        "fold_id": args.fold,
        "architecture": "strict_group_boosted_leaf_distribution_bayes_action",
        "scope": "outer labels excluded from feature, distribution, and policy selection",
        "physical_development_cutoff": str(DEV_CUTOFF),
        "calibration_days": args.calibration_days,
        "calibration_cutoff": str(calibration_cutoff),
        "class_width": CLASS_WIDTH,
        "class_count": CLASS_COUNT,
        "top_features": args.top_features,
        "iterations": args.iterations,
        "configurations": configurations,
        "diagnostics": diagnostics,
        "inner_score": _score(inner_base),
        "fold_score": _score(output),
        "selected_features": selected_features,
        "runtime_seconds": round(time.perf_counter() - started, 2),
        "prediction_path": str(output_path.relative_to(Path.cwd())),
        "prediction_sha256": _sha256(output_path),
        "new_2024_evaluation": False,
        "lockbox_reopened": False,
        "external_actions": [],
    }
    receipt_path = OUTPUT / f"{args.candidate_id}-{args.fold}.json"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
