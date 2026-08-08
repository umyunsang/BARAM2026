"""Predict the official 6% and 8% settlement events for candidate actions."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from run_action_utility_regressor import _expand
from run_alternative_booster_classifier import _feature_names
from run_consensus_classifier import _screen_blends
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

BASE_COLUMNS = ["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh"]
PARENT_PATH = OUTPUT / "M107_STRICT_TEMPORAL_TOP100-oof.parquet"
TEMPERATURES = (0.70, 1.00, 1.30)
GAMMAS = (0.50, 0.75, 1.00, 1.25, 1.50, 2.00)


def _parameters(iterations: int, num_leaves: int) -> dict[str, object]:
    return {
        "n_estimators": iterations,
        "learning_rate": 0.04,
        "num_leaves": num_leaves,
        "min_child_samples": 160,
        "max_bin": 127,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.2,
        "reg_lambda": 5.0,
        "random_state": 20260803,
        "n_jobs": 6,
        "deterministic": True,
        "force_col_wise": True,
        "verbosity": -1,
    }


def _temperature_probability(probability: np.ndarray, temperature: float) -> np.ndarray:
    clipped = np.clip(probability, 1e-5, 1.0 - 1e-5)
    logit = np.log(clipped / (1.0 - clipped)) / temperature
    return 1.0 / (1.0 + np.exp(-logit))


def _policy_frame(
    base: pd.DataFrame,
    policies: dict[str, np.ndarray],
) -> pd.DataFrame:
    capacity = base["group_id"].map(CAPACITIES).to_numpy(dtype=float)
    values = {
        name: np.clip(normalized, 0.075, 1.075) * capacity
        for name, normalized in policies.items()
    }
    return pd.concat([base.reset_index(drop=True), pd.DataFrame(values)], axis=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--fold", choices=FOLDS, required=True)
    parser.add_argument("--sitewind-cache-id", required=True)
    parser.add_argument("--feature-count", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=110)
    parser.add_argument("--training-action-step", type=float, default=0.05)
    parser.add_argument("--inference-action-step", type=float, default=0.025)
    parser.add_argument("--num-leaves", type=int, default=31)
    args = parser.parse_args()
    if not 20 <= args.feature_count <= 100:
        raise ValueError("feature-count must be between 20 and 100")
    if not 50 <= args.iterations <= 400:
        raise ValueError("iterations must be between 50 and 400")
    if args.training_action_step not in {0.025, 0.05}:
        raise ValueError("training action step must be 0.025 or 0.05")
    if args.inference_action_step not in {0.0125, 0.025, 0.05}:
        raise ValueError("inference action step is unsupported")
    if args.num_leaves not in {15, 31, 63}:
        raise ValueError("num-leaves must be 15, 31, or 63")
    if _sha256(OPEN) != OPEN_SHA or _sha256(BASELINE) != BASELINE_SHA:
        raise RuntimeError("immutable competition input hash mismatch")

    started = time.perf_counter()
    surface, base_columns, _ = _surface()
    validation = _validation_mask(surface, args.fold)
    validation_start = surface.loc[validation, "forecast_kst_dtm"].min()
    target = surface["actual_kwh"] / surface["group_id"].map(CAPACITIES)
    training = (
        surface["forecast_kst_dtm"].lt(validation_start).to_numpy()
        & target.ge(0.10).to_numpy()
    )
    cached = np.load(
        OUTPUT / f"{args.sitewind_cache_id}-{args.fold}-sitewind-features.npz"
    )
    matrix = surface[base_columns].astype("float32")
    sitewind_columns = _add_site_wind_features(
        matrix, cached["legacy"], cached["allweather"]
    )
    selected_features = _feature_names(args.fold)[: args.feature_count]
    raw_matrix = matrix[selected_features].to_numpy(dtype="float32")
    sitewind_curve = matrix["sitewind__mean_powercurve"].to_numpy(dtype="float32")
    means = {
        group_id: float(
            target.loc[training & surface["group_id"].eq(group_id).to_numpy()].mean()
        )
        for group_id in CAPACITIES
    }
    group_mean = surface["group_id"].map(means).to_numpy(dtype="float32")
    train_actions = np.arange(
        0.075, 1.0751, args.training_action_step, dtype="float32"
    )
    inference_actions = np.arange(
        0.075, 1.0751, args.inference_action_step, dtype="float32"
    )
    expanded_training = _expand(
        raw_matrix[training],
        train_actions,
        sitewind_curve[training],
        group_mean[training],
    )
    training_target = target.loc[training].to_numpy(dtype="float32")
    error = np.abs(training_target[:, None] - train_actions[None, :])
    within6 = (error <= 0.06).astype("int8").reshape(-1)
    within8 = (error <= 0.08).astype("int8").reshape(-1)
    absolute_error = error.astype("float32").reshape(-1)
    training_groups = surface.loc[training, "group_id"].to_numpy(dtype=int)
    group_counts = {
        group_id: int((training_groups == group_id).sum()) for group_id in CAPACITIES
    }
    group_weight = np.asarray(
        [len(training_groups) / (3.0 * group_counts[int(group)]) for group in training_groups],
        dtype="float32",
    )
    event_weight = np.repeat(group_weight * training_target, len(train_actions))
    error_weight = np.repeat(group_weight, len(train_actions))
    parameters = _parameters(args.iterations, args.num_leaves)
    event6_model = LGBMClassifier(objective="binary", **parameters)
    event8_model = LGBMClassifier(objective="binary", **parameters)
    error_model = LGBMRegressor(objective="l2", **parameters)
    event6_model.fit(expanded_training, within6, sample_weight=event_weight)
    event8_model.fit(expanded_training, within8, sample_weight=event_weight)
    error_model.fit(expanded_training, absolute_error, sample_weight=error_weight)
    point_model = LGBMRegressor(
        objective="l1",
        **{**parameters, "n_estimators": max(80, args.iterations)},
    )
    point_model.fit(
        raw_matrix[training],
        training_target,
        sample_weight=group_weight,
    )

    expanded_validation = _expand(
        raw_matrix[validation],
        inference_actions,
        sitewind_curve[validation],
        group_mean[validation],
    )
    shape = (int(validation.sum()), len(inference_actions))
    probability6 = event6_model.predict_proba(expanded_validation)[:, 1].reshape(shape)
    probability8 = event8_model.predict_proba(expanded_validation)[:, 1].reshape(shape)
    expected_error = np.clip(
        error_model.predict(expanded_validation).reshape(shape), 0.0, None
    )
    point = np.clip(point_model.predict(raw_matrix[validation]), 0.10, 1.075)
    validation_mean = group_mean[validation]
    normalized_policies: dict[str, np.ndarray] = {}
    for temperature in TEMPERATURES:
        p6 = _temperature_probability(probability6, temperature)
        p8 = np.maximum(_temperature_probability(probability8, temperature), p6)
        settlement = (4.0 * p6 + 3.0 * (p8 - p6)) / 4.0
        for gamma in GAMMAS:
            utility = -expected_error + gamma * (
                point[:, None] / validation_mean[:, None]
            ) * settlement
            normalized_policies[f"T{temperature:g}_G{gamma:g}"] = inference_actions[
                np.argmax(utility, axis=1)
            ]
    base = surface.loc[validation, BASE_COLUMNS].copy()
    policies = _policy_frame(base, normalized_policies)
    raw_scores = {
        name: _score(base.assign(prediction_kwh=policies[name].to_numpy(dtype=float)))
        for name in normalized_policies
    }
    raw_best_policy = max(raw_scores, key=lambda name: raw_scores[name]["total"])
    raw_output = base.assign(
        prediction_kwh=policies[raw_best_policy].to_numpy(dtype=float)
    )
    parent = pd.read_parquet(PARENT_PATH)
    parent = parent.loc[parent["fold_id"].eq(args.fold)]
    blended, selections = _screen_blends(base, policies, parent)
    output = blended.assign(fold_id=args.fold, model_id=args.candidate_id)
    output_path = OUTPUT / f"{args.candidate_id}-{args.fold}.parquet"
    policy_path = OUTPUT / f"{args.candidate_id}-{args.fold}-policies.parquet"
    output.to_parquet(output_path, index=False)
    policies.to_parquet(policy_path, index=False)
    receipt = {
        "schema_version": 1,
        "candidate_id": args.candidate_id,
        "fold_id": args.fold,
        "architecture": "action_conditional_6pct_8pct_event_lightgbm",
        "scope": "unseen-fold direct settlement-event classification screen",
        "training_action_count": len(train_actions),
        "inference_action_count": len(inference_actions),
        "training_action_step": args.training_action_step,
        "inference_action_step": args.inference_action_step,
        "iterations": args.iterations,
        "num_leaves": args.num_leaves,
        "feature_count": args.feature_count,
        "expanded_feature_count": expanded_training.shape[1],
        "expanded_training_rows": len(expanded_training),
        "group_training_rows": group_counts,
        "sitewind_feature_count": len(sitewind_columns),
        "raw_best_policy": raw_best_policy,
        "raw_best_score": _score(raw_output),
        "oracle_blend_score": _score(output),
        "oracle_blends": selections,
        "runtime_seconds": round(time.perf_counter() - started, 2),
        "prediction_path": str(output_path.relative_to(Path.cwd())),
        "prediction_sha256": _sha256(output_path),
        "policy_path": str(policy_path.relative_to(Path.cwd())),
        "policy_sha256": _sha256(policy_path),
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
