"""Screen label-free covariate-shift weighting for the target NWP period."""

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
from sklearn.metrics import roc_auc_score

BASE_COLUMNS = ["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh"]
PARENT_PATH = OUTPUT / "M107_STRICT_TEMPORAL_TOP100-oof.parquet"


def _density_ratio(
    surface: pd.DataFrame,
    matrix: pd.DataFrame,
    features: list[str],
    preceding: np.ndarray,
    validation: np.ndarray,
    strength: float,
    lower: float,
    upper: float,
) -> tuple[np.ndarray, dict[str, object]]:
    ratios = np.ones(len(surface), dtype=float)
    diagnostics: dict[str, object] = {}
    domain_features = [
        name
        for name in features
        if name not in {"cal__doy_sin", "cal__doy_cos", "month"}
    ]
    for group_id in CAPACITIES:
        group = surface["group_id"].eq(group_id).to_numpy()
        history = preceding & group
        target = validation & group
        domain = history | target
        labels = target[domain].astype(int)
        model = LGBMClassifier(
            objective="binary",
            n_estimators=120,
            learning_rate=0.03,
            num_leaves=7,
            min_child_samples=120,
            max_bin=127,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.7,
            reg_alpha=0.3,
            reg_lambda=5.0,
            class_weight="balanced",
            random_state=20260802 + group_id,
            n_jobs=6,
            deterministic=True,
            force_col_wise=True,
            verbosity=-1,
        )
        model.fit(matrix.loc[domain, domain_features], labels)
        probability = model.predict_proba(matrix.loc[history, domain_features])[:, 1]
        raw = probability / np.clip(1.0 - probability, 1e-6, None)
        ratio = np.clip(raw, lower, upper) ** strength
        ratio /= max(float(ratio.mean()), 1e-12)
        ratios[history] = ratio
        domain_probability = model.predict_proba(
            matrix.loc[domain, domain_features]
        )[:, 1]
        diagnostics[str(group_id)] = {
            "history_rows": int(history.sum()),
            "target_rows": int(target.sum()),
            "domain_auc_in_sample": float(roc_auc_score(labels, domain_probability)),
            "raw_ratio_min": float(raw.min()),
            "raw_ratio_mean": float(raw.mean()),
            "raw_ratio_max": float(raw.max()),
            "applied_ratio_min": float(ratio.min()),
            "applied_ratio_mean": float(ratio.mean()),
            "applied_ratio_max": float(ratio.max()),
        }
    return ratios, {"domain_feature_count": len(domain_features), "groups": diagnostics}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--fold", choices=FOLDS, required=True)
    parser.add_argument("--sitewind-cache-id", required=True)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--strength", type=float, default=0.5)
    parser.add_argument("--ratio-lower", type=float, default=0.20)
    parser.add_argument("--ratio-upper", type=float, default=5.0)
    args = parser.parse_args()
    if not 40 <= args.iterations <= 200:
        raise ValueError("iterations must be between 40 and 200")
    if not 0.0 < args.strength <= 1.0:
        raise ValueError("domain strength must be in (0, 1]")
    if not 0.0 < args.ratio_lower < 1.0 < args.ratio_upper:
        raise ValueError("density-ratio clipping bounds are invalid")
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
    ratio, domain_diagnostics = _density_ratio(
        surface,
        matrix,
        selected_features,
        preceding,
        validation,
        args.strength,
        args.ratio_lower,
        args.ratio_upper,
    )
    sample_weight = normalized_target.loc[training].clip(lower=0.10).to_numpy(
        dtype=float
    )
    sample_weight *= ratio[training]
    model = _fit_model(
        "xgboost",
        matrix[selected_features],
        classes,
        training,
        sample_weight,
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
            np.average(
                normalized_target.loc[
                    training & surface["group_id"].eq(group_id).to_numpy()
                ],
                weights=ratio[
                    training & surface["group_id"].eq(group_id).to_numpy()
                ],
            )
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
        "architecture": "label_free_domain_weighted_xgboost_multiclass",
        "scope": "same-fold transductive NWP-only representation screen",
        "domain_strength": args.strength,
        "ratio_bounds": [args.ratio_lower, args.ratio_upper],
        "domain_diagnostics": domain_diagnostics,
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
