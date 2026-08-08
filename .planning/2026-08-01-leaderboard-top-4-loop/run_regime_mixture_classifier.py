"""Screen NWP-regime experts blended with a global probabilistic classifier."""

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
from run_pseudo_group3_classifier import _pseudo_season_weights, _pseudo_targets
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
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer

BASE_COLUMNS = ["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh"]
PARENT_PATH = OUTPUT / "M107_STRICT_TEMPORAL_TOP100-oof.parquet"
ALPHAS = (0.50, 0.75, 1.00)


def _regime_features(selected: list[str], count: int) -> list[str]:
    tokens = (
        "sitewind__",
        "wind10",
        "wind50",
        "wind80",
        "wind100",
        "coherence",
        "vector_spread",
        "gradient_norm",
        "divergence",
        "vorticity",
        "lead_hour",
        "hour",
        "doy",
    )
    resolved = [name for name in selected if any(token in name for token in tokens)]
    if len(resolved) < count:
        raise RuntimeError(f"regime feature contract resolved only {len(resolved)} columns")
    return resolved[:count]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--fold", choices=FOLDS, required=True)
    parser.add_argument("--sitewind-cache-id", required=True)
    parser.add_argument("--regimes", type=int, choices=(3, 4, 5, 6), default=4)
    parser.add_argument("--regime-features", type=int, default=24)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--group3-pseudo-weight", type=float, default=0.0)
    parser.add_argument("--pseudo-season-bandwidth-days", type=float, default=45.0)
    args = parser.parse_args()
    if not 12 <= args.regime_features <= 40:
        raise ValueError("regime feature count must be between 12 and 40")
    if not 40 <= args.iterations <= 200:
        raise ValueError("iterations must be between 40 and 200")
    if not 0.0 <= args.group3_pseudo_weight <= 1.0:
        raise ValueError("group-3 pseudo weight must be between zero and one")
    if not 1.0 <= args.pseudo_season_bandwidth_days <= 183.0:
        raise ValueError("pseudo season bandwidth must be between one and 183")
    if _sha256(OPEN) != OPEN_SHA or _sha256(BASELINE) != BASELINE_SHA:
        raise RuntimeError("immutable competition input hash mismatch")

    started = time.perf_counter()
    surface, base_columns, _ = _surface()
    validation = _validation_mask(surface, args.fold)
    validation_start = surface.loc[validation, "forecast_kst_dtm"].min()
    preceding = surface["forecast_kst_dtm"].lt(validation_start).to_numpy()
    normalized_target = surface["actual_kwh"] / surface["group_id"].map(CAPACITIES)
    target = normalized_target.copy()
    pseudo_mask = np.zeros(len(surface), dtype=bool)
    pseudo_diagnostics: dict[str, object] = {"active": False}
    if args.group3_pseudo_weight > 0.0:
        pseudo, _, mapper_diagnostics = _pseudo_targets(
            surface,
            pd.DataFrame(index=surface.index),
            preceding,
            "compact",
            1.0,
        )
        pseudo_mask = pseudo.notna().to_numpy()
        target.loc[pseudo_mask] = pseudo.loc[pseudo_mask]
        pseudo_diagnostics = {
            "active": True,
            "weight": args.group3_pseudo_weight,
            "mapper": mapper_diagnostics,
        }
    training = (
        preceding
        & target.notna().to_numpy()
        & target.ge(0.10).to_numpy()
    )
    sample_weight = target.clip(lower=0.10).to_numpy(dtype=float)
    if args.group3_pseudo_weight > 0.0:
        season_weight, season_diagnostics = _pseudo_season_weights(
            surface,
            training,
            pseudo_mask,
            validation,
            args.pseudo_season_bandwidth_days,
        )
        sample_weight[pseudo_mask] *= args.group3_pseudo_weight
        sample_weight[np.flatnonzero(training)] *= season_weight
        pseudo_diagnostics["season"] = season_diagnostics
    raw_bins = np.floor((target.clip(0.10, 1.074999) - 0.10) / 0.02).astype(
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
            target.loc[training & classes.eq(class_id)].mean()
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
    matrix = matrix[selected_features]
    regime_features = _regime_features(selected_features, args.regime_features)

    global_model = _fit_model(
        "xgboost",
        matrix,
        classes,
        training,
        sample_weight[training],
        len(active_bins),
        args.iterations,
    )
    global_probability = _probability(
        global_model, "xgboost", matrix, validation, args.iterations
    )
    expert_probability = np.zeros_like(global_probability)
    group_diagnostics: dict[str, object] = {}
    validation_positions = np.flatnonzero(validation)
    for group_id in CAPACITIES:
        group_training = training & surface["group_id"].eq(group_id).to_numpy()
        group_validation = validation & surface["group_id"].eq(group_id).to_numpy()
        clusterer = make_pipeline(
            SimpleImputer(strategy="median"),
            QuantileTransformer(
                n_quantiles=200,
                output_distribution="normal",
                random_state=20260802,
            ),
            KMeans(
                n_clusters=args.regimes,
                n_init=20,
                random_state=20260802,
            ),
        )
        train_regime = clusterer.fit_predict(matrix.loc[group_training, regime_features])
        valid_regime = clusterer.predict(matrix.loc[group_validation, regime_features])
        group_train_positions = np.flatnonzero(group_training)
        group_valid_outer_positions = np.flatnonzero(
            surface.loc[validation, "group_id"].eq(group_id).to_numpy()
        )
        regime_receipts: dict[str, object] = {}
        for regime_id in range(args.regimes):
            train_positions = group_train_positions[train_regime == regime_id]
            valid_local = valid_regime == regime_id
            valid_outer_positions = group_valid_outer_positions[valid_local]
            local_global_classes = sorted(
                classes.iloc[train_positions].dropna().astype(int).unique()
            )
            local_map = {
                global_class: local_class
                for local_class, global_class in enumerate(local_global_classes)
            }
            local_classes = classes.map(local_map).astype("Int64")
            local_training = np.zeros(len(surface), dtype=bool)
            local_training[train_positions] = True
            if len(local_global_classes) < 2 or len(train_positions) < 300:
                expert_probability[valid_outer_positions] = global_probability[
                    valid_outer_positions
                ]
                regime_receipts[str(regime_id)] = {
                    "training_rows": len(train_positions),
                    "validation_rows": int(valid_local.sum()),
                    "fallback_global": True,
                }
                continue
            expert = _fit_model(
                "xgboost",
                matrix,
                local_classes,
                local_training,
                sample_weight[train_positions],
                len(local_global_classes),
                args.iterations,
            )
            local_validation = np.zeros(len(surface), dtype=bool)
            local_validation[validation_positions[valid_outer_positions]] = True
            local_probability = _probability(
                expert,
                "xgboost",
                matrix,
                local_validation,
                args.iterations,
            )
            mapped = np.zeros((int(valid_local.sum()), len(active_bins)), dtype=float)
            mapped[:, local_global_classes] = local_probability
            expert_probability[valid_outer_positions] = mapped
            regime_receipts[str(regime_id)] = {
                "training_rows": len(train_positions),
                "validation_rows": int(valid_local.sum()),
                "class_count": len(local_global_classes),
                "fallback_global": False,
            }
        group_diagnostics[str(group_id)] = regime_receipts

    if not np.isfinite(expert_probability).all() or np.any(
        expert_probability.sum(axis=1) <= 0.0
    ):
        raise RuntimeError("regime expert probability contract failed")
    expert_probability /= expert_probability.sum(axis=1, keepdims=True)
    base = surface.loc[validation, BASE_COLUMNS].copy()
    groups = base["group_id"].to_numpy(dtype=int)
    means = {
        group_id: float(
            target.loc[
                training & surface["group_id"].eq(group_id).to_numpy()
            ].mean()
        )
        for group_id in CAPACITIES
    }
    policy_predictions: dict[str, np.ndarray] = {}
    raw_scores: dict[str, dict[str, float]] = {}
    for alpha in ALPHAS:
        probability = (
            alpha * expert_probability + (1.0 - alpha) * global_probability
        )
        probability /= probability.sum(axis=1, keepdims=True)
        for policy, normalized in _policy_values(
            probability, centers, groups, means
        ).items():
            tag = f"A{alpha:g}_{policy}"
            trial = base.copy()
            trial["prediction_kwh"] = (
                normalized
                * trial["group_id"].map(CAPACITIES).to_numpy(dtype=float)
            )
            policy_predictions[tag] = trial["prediction_kwh"].to_numpy(dtype=float)
            raw_scores[tag] = _score(trial)
    policies = pd.concat(
        [base, pd.DataFrame(policy_predictions, index=base.index)],
        axis=1,
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
        "architecture": "groupwise_unsupervised_nwp_regime_xgboost_mixture",
        "scope": "same-fold representation screen; regimes use preceding NWP only",
        "regime_count": args.regimes,
        "regime_feature_count": len(regime_features),
        "regime_feature_names": regime_features,
        "group_regimes": group_diagnostics,
        "pseudo_diagnostics": pseudo_diagnostics,
        "alphas": list(ALPHAS),
        "raw_best_policy": raw_best_policy,
        "raw_best_score": _score(raw_output),
        "oracle_blend_score": _score(output),
        "oracle_blends": selections,
        "iterations": args.iterations,
        "feature_count": len(selected_features),
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
