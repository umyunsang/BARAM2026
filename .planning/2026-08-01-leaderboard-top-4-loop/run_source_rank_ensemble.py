"""Screen probabilistic rank ensembles across supplied GFS and LDAPS sources."""

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
from run_site_wind_teacher import _strict_preceding_mask, _validation_mask
from xgboost import XGBClassifier

BASE_COLUMNS = ["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh"]
PARENT_PATH = OUTPUT / "M107_STRICT_TEMPORAL_TOP100-oof.parquet"
MIXTURES = (
    (1.00, 0.00),
    (0.00, 1.00),
    (0.00, 0.00),
    (0.50, 0.25),
    (0.50, 0.50),
    (0.50, 0.75),
    (0.75, 0.25),
    (0.75, 0.50),
    (0.75, 0.75),
)


def _source_columns(
    columns: list[str],
    source: str,
) -> list[str]:
    common_tokens = (
        "sitewind__",
        "hour",
        "month",
        "lead_hour",
        "cal__",
        "group_",
        "capacity",
        "turbine_count",
        "rotor",
        "latitude_centroid",
        "longitude_centroid",
    )
    selected = [
        name
        for name in columns
        if source in name.lower() or any(token in name for token in common_tokens)
    ]
    if len(selected) < 80:
        raise RuntimeError(f"{source} feature contract resolved only {len(selected)} columns")
    return selected


def _source_probability(
    matrix: pd.DataFrame,
    source_columns: list[str],
    classes: pd.Series,
    training: np.ndarray,
    validation: np.ndarray,
    sample_weight: np.ndarray,
    class_count: int,
    iterations: int,
    top_features: int,
) -> tuple[np.ndarray, list[str]]:
    screen = XGBClassifier(
        objective="multi:softprob",
        num_class=class_count,
        n_estimators=60,
        learning_rate=0.04,
        max_depth=4,
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
    screen.fit(
        matrix.loc[training, source_columns],
        classes.loc[training].astype(int),
        sample_weight=sample_weight,
    )
    order = np.argsort(screen.feature_importances_)[::-1]
    selected = [source_columns[index] for index in order[:top_features]]
    model = _fit_model(
        "xgboost",
        matrix[selected],
        classes,
        training,
        sample_weight,
        class_count,
        iterations,
    )
    return (
        _probability(model, "xgboost", matrix[selected], validation, iterations),
        selected,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--fold", choices=FOLDS, required=True)
    parser.add_argument("--sitewind-cache-id", required=True)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--source-top-features", type=int, default=100)
    args = parser.parse_args()
    if not 40 <= args.iterations <= 200:
        raise ValueError("iterations must be between 40 and 200")
    if not 60 <= args.source_top_features <= 160:
        raise ValueError("source top-feature count must be between 60 and 160")
    if _sha256(OPEN) != OPEN_SHA or _sha256(BASELINE) != BASELINE_SHA:
        raise RuntimeError("immutable competition input hash mismatch")

    started = time.perf_counter()
    surface, base_columns, _ = _surface()
    validation = _validation_mask(surface, args.fold)
    normalized_target = surface["actual_kwh"] / surface["group_id"].map(CAPACITIES)
    training = (
        _strict_preceding_mask(surface, validation)
        & surface["actual_kwh"].notna().to_numpy()
        & normalized_target.ge(0.10).to_numpy()
    )
    raw_bins = np.floor((normalized_target.clip(0.10, 1.074999) - 0.10) / 0.02).astype(
        "Int64"
    )
    active_bins = [
        int(value)
        for value in sorted(raw_bins.loc[training].dropna().astype(int).unique())
    ]
    bin_to_class = {bin_id: class_id for class_id, bin_id in enumerate(active_bins)}
    classes = raw_bins.map(bin_to_class).astype("Int64")
    centers = np.asarray(
        [
            normalized_target.loc[training & classes.eq(class_id)].mean()
            for class_id in range(len(active_bins))
        ]
    )
    sample_weight = normalized_target.loc[training].clip(lower=0.10).to_numpy(float)

    cached = np.load(
        OUTPUT / f"{args.sitewind_cache_id}-{args.fold}-sitewind-features.npz"
    )
    matrix = surface[base_columns].astype("float32")
    sitewind_columns = _add_site_wind_features(
        matrix, cached["legacy"], cached["allweather"]
    )
    global_features = _feature_names(args.fold)
    missing = set(global_features).difference(matrix.columns)
    if missing:
        raise RuntimeError(f"missing fixed features: {sorted(missing)}")
    global_model = _fit_model(
        "xgboost",
        matrix[global_features],
        classes,
        training,
        sample_weight,
        len(active_bins),
        args.iterations,
    )
    global_probability = _probability(
        global_model,
        "xgboost",
        matrix[global_features],
        validation,
        args.iterations,
    )
    gfs_columns = _source_columns(base_columns, "gfs")
    ldaps_columns = _source_columns(base_columns, "ldaps")
    gfs_probability, gfs_selected = _source_probability(
        matrix,
        gfs_columns,
        classes,
        training,
        validation,
        sample_weight,
        len(active_bins),
        args.iterations,
        args.source_top_features,
    )
    ldaps_probability, ldaps_selected = _source_probability(
        matrix,
        ldaps_columns,
        classes,
        training,
        validation,
        sample_weight,
        len(active_bins),
        args.iterations,
        args.source_top_features,
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
    for global_weight, gfs_share in MIXTURES:
        if global_weight == 1.0:
            probability = global_probability.copy()
            mix_tag = "GLOBAL"
        elif global_weight == 0.0 and gfs_share == 1.0:
            probability = gfs_probability.copy()
            mix_tag = "GFS"
        elif global_weight == 0.0 and gfs_share == 0.0:
            probability = ldaps_probability.copy()
            mix_tag = "LDAPS"
        else:
            source_probability = (
                gfs_share * gfs_probability
                + (1.0 - gfs_share) * ldaps_probability
            )
            probability = (
                global_weight * global_probability
                + (1.0 - global_weight) * source_probability
            )
            mix_tag = f"W{global_weight:g}_R{gfs_share:g}"
        probability /= probability.sum(axis=1, keepdims=True)
        for policy, normalized in _policy_values(
            probability, centers, groups, means
        ).items():
            tag = f"{mix_tag}_{policy}"
            prediction = (
                normalized * base["group_id"].map(CAPACITIES).to_numpy(dtype=float)
            )
            trial = base.assign(prediction_kwh=prediction)
            policy_predictions[tag] = prediction
            raw_scores[tag] = _score(trial)
    policies = pd.concat(
        [base, pd.DataFrame(policy_predictions, index=base.index)],
        axis=1,
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
        "architecture": "gfs_ldaps_probabilistic_rank_ensemble",
        "scope": "same-fold representation screen using supplied NWP sources only",
        "mixtures": [
            {"global_weight": global_weight, "gfs_share": gfs_share}
            for global_weight, gfs_share in MIXTURES
        ],
        "gfs_feature_count": len(gfs_selected),
        "gfs_selected_features": gfs_selected,
        "ldaps_feature_count": len(ldaps_selected),
        "ldaps_selected_features": ldaps_selected,
        "raw_best_policy": raw_best_policy,
        "raw_best_score": _score(raw_output),
        "oracle_blend_score": _score(output),
        "oracle_blends": selections,
        "iterations": args.iterations,
        "global_feature_count": len(global_features),
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
