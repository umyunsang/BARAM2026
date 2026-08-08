"""Screen an ExtraTrees conditional distribution on the fixed feature contract."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from run_alternative_booster_classifier import _feature_names
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
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer

BASE_COLUMNS = ["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh"]
PARENT_PATH = OUTPUT / "M107_STRICT_TEMPORAL_TOP100-oof.parquet"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--fold", choices=FOLDS, required=True)
    parser.add_argument("--sitewind-cache-id", required=True)
    parser.add_argument("--family", choices=("extra", "random"), default="extra")
    parser.add_argument("--trees", type=int, default=256)
    parser.add_argument("--min-leaf", type=int, default=20)
    parser.add_argument("--max-features", type=float, default=0.70)
    args = parser.parse_args()
    if not 64 <= args.trees <= 512:
        raise ValueError("tree count must be between 64 and 512")
    if not 5 <= args.min_leaf <= 160:
        raise ValueError("minimum leaf size must be between five and 160")
    if not 0.2 <= args.max_features <= 1.0:
        raise ValueError("max-features must be between 0.2 and 1.0")
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
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    fit_matrix = imputer.fit_transform(matrix.loc[training, selected_features])
    apply_matrix = imputer.transform(matrix.loc[validation, selected_features])
    common = {
        "n_estimators": args.trees,
        "min_samples_leaf": args.min_leaf,
        "max_features": args.max_features,
        "class_weight": None,
        "random_state": 20260802,
        "n_jobs": 6,
    }
    if args.family == "extra":
        model = ExtraTreesClassifier(
            **common,
            criterion="log_loss",
            bootstrap=False,
        )
    else:
        model = RandomForestClassifier(
            **common,
            criterion="log_loss",
            bootstrap=True,
            max_samples=0.85,
        )
    model.fit(
        fit_matrix,
        classes.loc[training].astype(int),
        sample_weight=normalized_target.loc[training].clip(lower=0.10),
    )
    probability = model.predict_proba(apply_matrix)
    if probability.shape[1] != len(active_bins):
        raise RuntimeError("tree probability class contract changed")

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
    for policy, normalized in _policy_values(
        probability, centers, groups, means
    ).items():
        prediction = (
            normalized * base["group_id"].map(CAPACITIES).to_numpy(dtype=float)
        )
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
        "architecture": f"{args.family}_forest_multiclass_distribution",
        "scope": "same-fold official-data-only representation screen",
        "trees": args.trees,
        "min_leaf": args.min_leaf,
        "max_features": args.max_features,
        "class_count": len(active_bins),
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
