"""Blend a season-weighted group-3 pseudo distribution into source ranks."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from run_alternative_booster_classifier import _feature_names, _fit_model, _probability
from run_consensus_classifier import _screen_blends
from run_inner_policy_classifier import _policy_values
from run_pseudo_group3_classifier import (
    GROUP_ID,
    _pseudo_season_weights,
    _pseudo_targets,
)
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
from run_source_rank_ensemble import MIXTURES, _source_columns, _source_probability

BASE_COLUMNS = ["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh"]
PARENT_PATH = OUTPUT / "M107_STRICT_TEMPORAL_TOP100-oof.parquet"
PSEUDO_MIX_WEIGHTS = (0.0, 0.25, 0.50, 0.75, 1.0)


def _pseudo_probability(
    surface: pd.DataFrame,
    matrix: pd.DataFrame,
    preceding: np.ndarray,
    validation: np.ndarray,
    global_bins: list[int],
    pseudo_weight: float,
    season_bandwidth_days: float,
) -> tuple[np.ndarray, dict[str, object]]:
    normalized_target = surface["actual_kwh"] / surface["group_id"].map(CAPACITIES)
    pseudo, pseudo_confidence, pseudo_diagnostics = _pseudo_targets(
        surface, matrix, preceding, "compact", 1.0
    )
    target = normalized_target.copy()
    target.loc[pseudo.notna()] = pseudo.loc[pseudo.notna()]
    pseudo_mask = pseudo.notna().to_numpy()
    observed_mask = surface["actual_kwh"].notna().to_numpy()
    group = surface["group_id"].eq(GROUP_ID).to_numpy()
    eligible = preceding & group & target.ge(0.10).to_numpy()
    season_weight, season_diagnostics = _pseudo_season_weights(
        surface,
        eligible,
        pseudo_mask,
        validation,
        season_bandwidth_days,
    )
    raw_bins = np.floor((target.clip(0.10, 1.074999) - 0.10) / 0.02).astype(
        "Int64"
    )
    active_bins = [
        int(value)
        for value in sorted(raw_bins.loc[eligible].dropna().astype(int).unique())
    ]
    bin_to_class = {bin_id: class_id for class_id, bin_id in enumerate(active_bins)}
    classes = raw_bins.map(bin_to_class).astype("Int64")
    params = {
        "objective": "multiclass",
        "num_class": len(active_bins),
        "n_estimators": 80,
        "learning_rate": 0.025,
        "num_leaves": 15,
        "min_child_samples": 60,
        "subsample": 0.9,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 2.0,
        "random_state": 20260802,
        "n_jobs": 6,
        "deterministic": True,
        "force_col_wise": True,
        "verbosity": -1,
    }
    screen = LGBMClassifier(**params)
    screen.fit(
        matrix.loc[eligible],
        classes.loc[eligible].astype(int),
        sample_weight=target.loc[eligible].clip(lower=0.10),
    )
    gains = screen.booster_.feature_importance(importance_type="gain")
    positions = np.argsort(gains)[::-1][:100]
    selected_features = [matrix.columns[index] for index in positions]
    weights = target.loc[eligible].clip(lower=0.10).to_numpy(dtype=float)
    weights *= np.where(pseudo_mask[eligible], pseudo_weight, 1.0)
    weights *= pseudo_confidence.loc[eligible].to_numpy(dtype=float)
    weights *= season_weight
    model = LGBMClassifier(**params)
    model.fit(
        matrix.loc[eligible, selected_features],
        classes.loc[eligible].astype(int),
        sample_weight=weights,
    )
    apply = validation & group
    compact_probability = model.predict_proba(
        matrix.loc[apply, selected_features], num_iteration=60
    )
    aligned = np.zeros((int(apply.sum()), len(global_bins)), dtype=float)
    global_positions = {bin_id: index for index, bin_id in enumerate(global_bins)}
    missing_bins = set(active_bins).difference(global_positions)
    if missing_bins:
        raise RuntimeError(f"pseudo classes not present globally: {sorted(missing_bins)}")
    for class_id, bin_id in enumerate(active_bins):
        aligned[:, global_positions[bin_id]] = compact_probability[:, class_id]
    aligned /= aligned.sum(axis=1, keepdims=True)
    diagnostics = {
        "active_bins": active_bins,
        "pseudo_weight": pseudo_weight,
        "season_bandwidth_days": season_bandwidth_days,
        "selected_iteration": 60,
        "selected_feature_names": selected_features,
        "pseudo_diagnostics": pseudo_diagnostics,
        "season_diagnostics": season_diagnostics,
        "observed_training_rows": int((eligible & observed_mask).sum()),
        "eligible_rows": int(eligible.sum()),
    }
    return aligned, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--fold", choices=FOLDS, required=True)
    parser.add_argument("--sitewind-cache-id", required=True)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--source-top-features", type=int, default=100)
    parser.add_argument("--pseudo-weight", type=float, default=0.20)
    parser.add_argument("--pseudo-season-bandwidth-days", type=float, default=45.0)
    args = parser.parse_args()
    if not 40 <= args.iterations <= 200:
        raise ValueError("iterations must be between 40 and 200")
    if not 60 <= args.source_top_features <= 160:
        raise ValueError("source top-feature count must be between 60 and 160")
    if not 0.0 <= args.pseudo_weight <= 2.0:
        raise ValueError("pseudo weight must be between zero and two")
    if not 0.0 <= args.pseudo_season_bandwidth_days <= 183.0:
        raise ValueError("pseudo season bandwidth must be between zero and 183")
    if _sha256(OPEN) != OPEN_SHA or _sha256(BASELINE) != BASELINE_SHA:
        raise RuntimeError("immutable competition input hash mismatch")

    started = time.perf_counter()
    surface, base_columns, _ = _surface()
    validation = _validation_mask(surface, args.fold)
    preceding = _strict_preceding_mask(surface, validation)
    normalized_target = surface["actual_kwh"] / surface["group_id"].map(CAPACITIES)
    training = (
        preceding
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
    pseudo_probability, pseudo_diagnostics = _pseudo_probability(
        surface,
        matrix,
        preceding,
        validation,
        active_bins,
        args.pseudo_weight,
        args.pseudo_season_bandwidth_days,
    )

    base = surface.loc[validation, BASE_COLUMNS].copy()
    groups = base["group_id"].to_numpy(dtype=int)
    group3 = groups == GROUP_ID
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
            source_rank_probability = global_probability.copy()
            mix_tag = "GLOBAL"
        elif global_weight == 0.0 and gfs_share == 1.0:
            source_rank_probability = gfs_probability.copy()
            mix_tag = "GFS"
        elif global_weight == 0.0 and gfs_share == 0.0:
            source_rank_probability = ldaps_probability.copy()
            mix_tag = "LDAPS"
        else:
            source_probability = (
                gfs_share * gfs_probability
                + (1.0 - gfs_share) * ldaps_probability
            )
            source_rank_probability = (
                global_weight * global_probability
                + (1.0 - global_weight) * source_probability
            )
            mix_tag = f"W{global_weight:g}_R{gfs_share:g}"
        source_rank_probability /= source_rank_probability.sum(axis=1, keepdims=True)
        for pseudo_mix in PSEUDO_MIX_WEIGHTS:
            probability = source_rank_probability.copy()
            probability[group3] = (
                (1.0 - pseudo_mix) * probability[group3]
                + pseudo_mix * pseudo_probability
            )
            probability /= probability.sum(axis=1, keepdims=True)
            for policy, normalized in _policy_values(
                probability, centers, groups, means
            ).items():
                tag = f"{mix_tag}_P{pseudo_mix:g}_{policy}"
                prediction = (
                    normalized
                    * base["group_id"].map(CAPACITIES).to_numpy(dtype=float)
                )
                policy_predictions[tag] = prediction
                raw_scores[tag] = _score(base.assign(prediction_kwh=prediction))
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
        "architecture": "seasonal_group3_pseudo_source_rank_probability",
        "scope": "same-fold supplied-data-only probability representation screen",
        "pseudo_mix_weights": list(PSEUDO_MIX_WEIGHTS),
        "pseudo_diagnostics": pseudo_diagnostics,
        "gfs_feature_count": len(gfs_selected),
        "ldaps_feature_count": len(ldaps_selected),
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
