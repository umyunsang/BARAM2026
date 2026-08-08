"""Learn the official action utility directly on an expanded prediction grid."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
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


def _expand(
    matrix: np.ndarray,
    actions: np.ndarray,
    sitewind_curve: np.ndarray,
    group_mean: np.ndarray,
) -> np.ndarray:
    rows = len(matrix)
    expanded = np.repeat(matrix, len(actions), axis=0)
    action = np.tile(actions, rows)
    curve = np.repeat(sitewind_curve, len(actions))
    mean = np.repeat(group_mean, len(actions))
    derived = np.column_stack(
        [
            action,
            action**2,
            action**3,
            action - curve,
            np.abs(action - curve),
            action * curve,
            action - mean,
            np.abs(action - mean),
        ]
    ).astype("float32")
    return np.column_stack([expanded, derived]).astype("float32", copy=False)


def _utility_targets(
    target: np.ndarray,
    groups: np.ndarray,
    actions: np.ndarray,
    means: dict[int, float],
    settlement_weight: float,
) -> np.ndarray:
    error = np.abs(target[:, None] - actions[None, :])
    units = np.select([error <= 0.06, error <= 0.08], [4.0, 3.0], default=0.0)
    mean = np.asarray([means[int(group_id)] for group_id in groups])
    utility = -error + settlement_weight * (
        target[:, None] * units / (4.0 * mean[:, None])
    )
    return utility.astype("float32").reshape(-1)


def _policy_frame(
    base: pd.DataFrame,
    normalized: np.ndarray,
) -> pd.DataFrame:
    capacity = base["group_id"].map(CAPACITIES).to_numpy(dtype=float)
    policies: dict[str, np.ndarray] = {"DIRECT": normalized * capacity}
    for scale in (0.95, 1.00, 1.05):
        for offset in np.arange(-0.05, 0.0501, 0.005):
            value = np.clip(scale * normalized + offset, 0.075, 1.075)
            policies[f"S{scale:.2f}_O{offset:+.3f}"] = value * capacity
    return pd.concat(
        [base.reset_index(drop=True), pd.DataFrame(policies)], axis=1
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--fold", choices=FOLDS, required=True)
    parser.add_argument("--sitewind-cache-id", required=True)
    parser.add_argument("--feature-count", type=int, default=40)
    parser.add_argument("--iterations", type=int, default=180)
    parser.add_argument("--action-step", type=float, default=0.025)
    parser.add_argument("--settlement-weight", type=float, default=1.0)
    parser.add_argument("--num-leaves", type=int, default=63)
    args = parser.parse_args()
    if not 20 <= args.feature_count <= 100:
        raise ValueError("feature-count must be between 20 and 100")
    if not 50 <= args.iterations <= 500:
        raise ValueError("iterations must be between 50 and 500")
    if args.action_step not in {0.025, 0.05}:
        raise ValueError("action-step must be 0.025 or 0.05")
    if not 0.25 <= args.settlement_weight <= 2.0:
        raise ValueError("settlement-weight must be between 0.25 and two")
    if args.num_leaves not in {31, 63, 127}:
        raise ValueError("num-leaves must be 31, 63, or 127")
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
    actions = np.arange(0.075, 1.0751, args.action_step, dtype="float32")
    training_matrix = _expand(
        raw_matrix[training],
        actions,
        sitewind_curve[training],
        group_mean[training],
    )
    training_target = _utility_targets(
        target.loc[training].to_numpy(dtype=float),
        surface.loc[training, "group_id"].to_numpy(dtype=int),
        actions,
        means,
        args.settlement_weight,
    )
    training_groups = surface.loc[training, "group_id"].to_numpy(dtype=int)
    group_counts = {
        group_id: int((training_groups == group_id).sum()) for group_id in CAPACITIES
    }
    row_weight = np.asarray(
        [len(training_groups) / (3.0 * group_counts[int(group)]) for group in training_groups],
        dtype="float32",
    )
    expanded_weight = np.repeat(row_weight, len(actions))
    model = LGBMRegressor(
        objective="l2",
        n_estimators=args.iterations,
        learning_rate=0.04,
        num_leaves=args.num_leaves,
        min_child_samples=200,
        max_bin=127,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.2,
        reg_lambda=5.0,
        random_state=20260802,
        n_jobs=6,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )
    model.fit(training_matrix, training_target, sample_weight=expanded_weight)
    validation_matrix = _expand(
        raw_matrix[validation],
        actions,
        sitewind_curve[validation],
        group_mean[validation],
    )
    utility = model.predict(validation_matrix).reshape(-1, len(actions))
    normalized = actions[np.argmax(utility, axis=1)]
    base = surface.loc[validation, BASE_COLUMNS].copy()
    policies = _policy_frame(base, normalized)
    raw_scores: dict[str, dict[str, float]] = {}
    for name in sorted(set(policies.columns).difference(BASE_COLUMNS)):
        raw_scores[name] = _score(
            base.assign(prediction_kwh=policies[name].to_numpy(dtype=float))
        )
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
        "architecture": "action_conditional_direct_official_utility_lightgbm",
        "scope": "unseen-fold official-utility action regression screen",
        "action_count": len(actions),
        "action_step": args.action_step,
        "settlement_weight": args.settlement_weight,
        "iterations": args.iterations,
        "num_leaves": args.num_leaves,
        "feature_count": args.feature_count,
        "expanded_feature_count": training_matrix.shape[1],
        "expanded_training_rows": len(training_matrix),
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
