"""Screen XGBoost class targets smoothed only inside preceding history."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from run_alternative_booster_classifier import _feature_names, _fit_model, _probability
from run_consensus_classifier import _screen_blends
from run_inner_policy_classifier import _policy_values
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


def _smooth_preceding_target(
    surface: pd.DataFrame,
    target: pd.Series,
    preceding: np.ndarray,
    window: int,
    alpha: float,
) -> pd.Series:
    smoothed = pd.Series(np.nan, index=surface.index, dtype=float)
    for group_id in CAPACITIES:
        positions = np.flatnonzero(
            preceding & surface["group_id"].eq(group_id).to_numpy()
        )
        ordered = surface.iloc[positions].sort_values("forecast_kst_dtm").index
        values = target.loc[ordered]
        latent = values.rolling(window, center=True, min_periods=1).mean()
        smoothed.loc[ordered] = (1.0 - alpha) * values + alpha * latent
    if smoothed.loc[preceding].isna().any():
        missing = int(smoothed.loc[preceding].isna().sum())
        if missing != int(target.loc[preceding].isna().sum()):
            raise RuntimeError("preceding smoothing introduced unexpected missing values")
    return smoothed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--fold", choices=FOLDS, required=True)
    parser.add_argument("--sitewind-cache-id", required=True)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--window", type=int, choices=(3, 5, 7), default=3)
    parser.add_argument("--alpha", type=float, default=1.0)
    args = parser.parse_args()
    if not 40 <= args.iterations <= 200:
        raise ValueError("iterations must be between 40 and 200")
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("smoothing alpha must be between zero and one")
    if _sha256(OPEN) != OPEN_SHA or _sha256(BASELINE) != BASELINE_SHA:
        raise RuntimeError("immutable competition input hash mismatch")

    started = time.perf_counter()
    surface, base_columns, _ = _surface()
    validation = _validation_mask(surface, args.fold)
    validation_start = surface.loc[validation, "forecast_kst_dtm"].min()
    preceding = surface["forecast_kst_dtm"].lt(validation_start).to_numpy()
    normalized_target = surface["actual_kwh"] / surface["group_id"].map(CAPACITIES)
    training = (
        preceding
        & surface["actual_kwh"].notna().to_numpy()
        & normalized_target.ge(0.10).to_numpy()
    )
    class_target = _smooth_preceding_target(
        surface,
        normalized_target,
        preceding,
        args.window,
        args.alpha,
    )
    raw_bins = np.floor((class_target.clip(0.10, 1.074999) - 0.10) / 0.02).astype(
        "Int64"
    )
    active_bins = [
        int(value)
        for value in sorted(raw_bins.loc[training].dropna().astype(int).unique())
    ]
    bin_to_class = {bin_id: class_id for class_id, bin_id in enumerate(active_bins)}
    classes = raw_bins.map(bin_to_class).astype("Int64")
    training &= classes.notna().to_numpy()
    centers = np.asarray(
        [
            normalized_target.loc[training & classes.eq(class_id)].mean()
            for class_id in range(len(active_bins))
        ]
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
    model = _fit_model(
        "xgboost",
        matrix[selected_features],
        classes,
        training,
        normalized_target.loc[training].clip(lower=0.10).to_numpy(dtype=float),
        len(active_bins),
        args.iterations,
    )
    probability = _probability(
        model,
        "xgboost",
        matrix[selected_features],
        validation,
        args.iterations,
    )
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
    for policy, normalized in _policy_values(
        probability, centers, groups, means
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
    output.to_parquet(output_path, index=False)
    policies.to_parquet(policy_path, index=False)
    receipt = {
        "schema_version": 1,
        "candidate_id": args.candidate_id,
        "fold_id": args.fold,
        "architecture": "preceding_target_smoothed_xgboost_multiclass",
        "scope": "same-fold representation screen; validation labels excluded from smoothing",
        "smoothing_window": args.window,
        "smoothing_alpha": args.alpha,
        "class_count": len(active_bins),
        "iterations": args.iterations,
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
