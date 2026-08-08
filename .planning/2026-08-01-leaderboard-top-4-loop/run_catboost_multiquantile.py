"""Fit a CatBoost multi-quantile distribution and optimize official utility."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
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
from run_site_wind_classifier import DECISION_GAMMAS, FOLDS, _add_site_wind_features
from run_site_wind_teacher import _validation_mask

BASE_COLUMNS = ["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh"]
PARENT_PATH = OUTPUT / "M107_STRICT_TEMPORAL_TOP100-oof.parquet"
QUANTILES = np.asarray((0.05, 0.15, 0.30, 0.50, 0.70, 0.85, 0.95))
MIDPOINTS = (np.arange(39, dtype=float) + 0.5) / 39.0
SPREAD_SCALES = (0.75, 1.0, 1.25, 1.5)


def _policies(
    quantiles: np.ndarray,
    groups: np.ndarray,
    means: dict[int, float],
) -> dict[str, np.ndarray]:
    quantiles = np.maximum.accumulate(quantiles, axis=1)
    samples = np.vstack(
        [np.interp(MIDPOINTS, QUANTILES, row) for row in quantiles]
    )
    median = quantiles[:, 3]
    actions = np.arange(0.075, 1.076, 0.0025)
    output: dict[str, np.ndarray] = {"MEDIAN": np.clip(median, 0.075, 1.075)}
    for spread in SPREAD_SCALES:
        distribution = np.clip(
            median[:, None] + spread * (samples - median[:, None]),
            0.075,
            1.075,
        )
        for gamma in DECISION_GAMMAS:
            chosen = np.empty(len(quantiles), dtype=float)
            for group_id in CAPACITIES:
                positions = np.flatnonzero(groups == group_id)
                for lower in range(0, len(positions), 256):
                    index = positions[lower : lower + 256]
                    error = np.abs(
                        actions[None, :, None] - distribution[index, None, :]
                    )
                    units = np.select(
                        [error <= 0.06, error <= 0.08], [4.0, 3.0], default=0.0
                    )
                    utility = -error.mean(axis=2) + gamma * (
                        distribution[index, None, :] * units
                    ).mean(axis=2) / (4.0 * means[group_id])
                    chosen[index] = actions[
                        np.argmax(utility, axis=1)
                    ]
            output[f"S{spread:g}_G{gamma:g}"] = chosen
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--fold", choices=FOLDS, required=True)
    parser.add_argument("--sitewind-cache-id", required=True)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--depth", type=int, choices=(6, 7, 8), default=7)
    args = parser.parse_args()
    if not 100 <= args.iterations <= 1_000:
        raise ValueError("iterations must be between 100 and 1000")
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
    selected_features = _feature_names(args.fold)
    missing = set(selected_features).difference(matrix.columns)
    if missing:
        raise RuntimeError(f"missing fixed features: {sorted(missing)}")
    loss = "MultiQuantile:alpha=" + ",".join(f"{value:g}" for value in QUANTILES)
    model = CatBoostRegressor(
        loss_function=loss,
        iterations=args.iterations,
        learning_rate=0.03,
        depth=args.depth,
        l2_leaf_reg=5.0,
        random_strength=0.3,
        random_seed=20260802,
        thread_count=6,
        allow_writing_files=False,
        verbose=False,
    )
    model.fit(
        matrix.loc[training, selected_features],
        normalized_target.loc[training],
        sample_weight=normalized_target.loc[training].clip(lower=0.10),
    )
    quantile_prediction = model.predict(matrix.loc[validation, selected_features])
    base = surface.loc[validation, BASE_COLUMNS].copy()
    groups = base["group_id"].to_numpy(dtype=int)
    means = {
        group_id: float(
            normalized_target.loc[
                training & surface["group_id"].eq(group_id).to_numpy()
            ].mean()
        )
        for group_id in CAPACITIES
    }
    policy_predictions: dict[str, np.ndarray] = {}
    raw_scores: dict[str, dict[str, float]] = {}
    capacity = base["group_id"].map(CAPACITIES).to_numpy(dtype=float)
    for policy, normalized in _policies(
        quantile_prediction, groups, means
    ).items():
        prediction = normalized * capacity
        policy_predictions[policy] = prediction
        raw_scores[policy] = _score(base.assign(prediction_kwh=prediction))
    policies = pd.concat(
        [base, pd.DataFrame(policy_predictions, index=base.index)], axis=1
    )
    raw_best_policy = max(raw_scores, key=lambda name: raw_scores[name]["total"])
    raw_output = base.assign(prediction_kwh=policy_predictions[raw_best_policy])
    parent = pd.read_parquet(PARENT_PATH)
    parent = parent.loc[parent["fold_id"].eq(args.fold)]
    blended, selections = _screen_blends(base, policies, parent)
    output = blended.assign(fold_id=args.fold, model_id=args.candidate_id)
    output_path = OUTPUT / f"{args.candidate_id}-{args.fold}.parquet"
    policy_path = OUTPUT / f"{args.candidate_id}-{args.fold}-policies.parquet"
    quantile_path = OUTPUT / f"{args.candidate_id}-{args.fold}-quantiles.npz"
    output.to_parquet(output_path, index=False)
    policies.to_parquet(policy_path, index=False)
    np.savez_compressed(
        quantile_path,
        quantiles=quantile_prediction.astype("float32"),
        levels=QUANTILES.astype("float32"),
    )
    receipt = {
        "schema_version": 1,
        "candidate_id": args.candidate_id,
        "fold_id": args.fold,
        "architecture": "catboost_multi_quantile_official_utility",
        "scope": "same-fold official-data-only distribution screen",
        "quantiles": QUANTILES.tolist(),
        "distribution_midpoints": len(MIDPOINTS),
        "spread_scales": list(SPREAD_SCALES),
        "iterations": args.iterations,
        "depth": args.depth,
        "feature_count": len(selected_features),
        "selected_feature_names": selected_features,
        "raw_best_policy": raw_best_policy,
        "raw_best_score": _score(raw_output),
        "oracle_blend_score": _score(output),
        "oracle_blends": selections,
        "sitewind_feature_count": len(sitewind_columns),
        "runtime_seconds": round(time.perf_counter() - started, 2),
        "prediction_path": str(output_path.relative_to(Path.cwd())),
        "prediction_sha256": _sha256(output_path),
        "policy_path": str(policy_path.relative_to(Path.cwd())),
        "policy_sha256": _sha256(policy_path),
        "quantile_path": str(quantile_path.relative_to(Path.cwd())),
        "quantile_sha256": _sha256(quantile_path),
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
