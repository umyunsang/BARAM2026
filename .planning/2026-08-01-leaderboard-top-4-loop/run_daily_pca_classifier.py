"""Screen a classifier augmented with preceding-fitted daily NWP embeddings."""

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
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE_COLUMNS = ["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh"]
PARENT_PATH = OUTPUT / "M107_STRICT_TEMPORAL_TOP100-oof.parquet"


def _trajectory_features(selected: list[str], count: int) -> list[str]:
    tokens = (
        "sitewind__",
        "wind10",
        "wind50",
        "wind80",
        "wind100",
        "vector_speed",
        "mean_speed3",
        "layout_along",
        "layout_cross",
        "coherence",
        "vector_spread",
    )
    names = [
        name
        for name in selected
        if any(token in name for token in tokens)
        and "batch_" not in name
        and "fourier" not in name
    ]
    if len(names) < count:
        raise RuntimeError(f"trajectory feature contract resolved only {len(names)}")
    return names[:count]


def _daily_embedding(
    surface: pd.DataFrame,
    matrix: pd.DataFrame,
    value_columns: list[str],
    validation_start: pd.Timestamp,
    components: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    embedding = np.full((len(surface), components), np.nan, dtype="float32")
    diagnostics: dict[str, object] = {}
    for group_id in CAPACITIES:
        group_positions = np.flatnonzero(surface["group_id"].eq(group_id).to_numpy())
        group = surface.iloc[group_positions][
            ["data_available_kst_dtm", "forecast_kst_dtm", "lead_hour"]
        ].join(matrix.iloc[group_positions][value_columns])
        group = group.sort_values(
            ["data_available_kst_dtm", "forecast_kst_dtm"]
        )
        sizes = group.groupby("data_available_kst_dtm", sort=True).size()
        if not sizes.eq(24).all():
            raise RuntimeError("daily embedding requires 24 horizons per issuance")
        issuances = group["data_available_kst_dtm"].drop_duplicates().reset_index(drop=True)
        x = group[value_columns].to_numpy(dtype="float32").reshape(len(issuances), -1)
        day_end = group.groupby("data_available_kst_dtm", sort=True)[
            "forecast_kst_dtm"
        ].max().reset_index(drop=True)
        fit_days = day_end.lt(validation_start).to_numpy()
        component_count = min(components, int(fit_days.sum()) - 1, x.shape[1])
        if component_count != components:
            raise RuntimeError("requested daily PCA component count is not feasible")
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            PCA(
                n_components=components,
                whiten=False,
                random_state=20260802,
            ),
        )
        model.fit(x[fit_days])
        transformed = model.transform(x).astype("float32")
        ordered_positions = group.index.to_numpy(dtype=int)
        embedding[ordered_positions] = np.repeat(transformed, 24, axis=0)
        pca = model.named_steps["pca"]
        diagnostics[str(group_id)] = {
            "fit_days": int(fit_days.sum()),
            "all_days": len(issuances),
            "explained_variance_ratio_sum": float(
                pca.explained_variance_ratio_.sum()
            ),
        }
    if not np.isfinite(embedding).all():
        raise RuntimeError("daily PCA embedding contains non-finite values")
    columns = [f"daily_pca__{index:02d}" for index in range(components)]
    return pd.DataFrame(embedding, index=surface.index, columns=columns), diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--fold", choices=FOLDS, required=True)
    parser.add_argument("--sitewind-cache-id", required=True)
    parser.add_argument("--trajectory-features", type=int, default=20)
    parser.add_argument("--components", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    if not 12 <= args.trajectory_features <= 40:
        raise ValueError("trajectory feature count must be between 12 and 40")
    if not 4 <= args.components <= 32:
        raise ValueError("components must be between four and 32")
    if not 40 <= args.iterations <= 200:
        raise ValueError("iterations must be between 40 and 200")
    if _sha256(OPEN) != OPEN_SHA or _sha256(BASELINE) != BASELINE_SHA:
        raise RuntimeError("immutable competition input hash mismatch")

    started = time.perf_counter()
    surface, base_columns, _ = _surface()
    validation = _validation_mask(surface, args.fold)
    validation_start = pd.Timestamp(
        surface.loc[validation, "forecast_kst_dtm"].min()
    )
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
    trajectory_features = _trajectory_features(
        selected_features, args.trajectory_features
    )
    embedding, embedding_diagnostics = _daily_embedding(
        surface,
        matrix,
        trajectory_features,
        validation_start,
        args.components,
    )
    matrix = pd.concat([matrix[selected_features], embedding], axis=1)
    model = _fit_model(
        "xgboost",
        matrix,
        classes,
        training,
        normalized_target.loc[training].clip(lower=0.10).to_numpy(dtype=float),
        len(active_bins),
        args.iterations,
    )
    probability = _probability(
        model, "xgboost", matrix, validation, args.iterations
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
        "architecture": "daily_nwp_pca_xgboost_multiclass",
        "scope": "same-fold representation screen; PCA fit on preceding NWP only",
        "trajectory_feature_count": len(trajectory_features),
        "trajectory_feature_names": trajectory_features,
        "component_count": args.components,
        "embedding_diagnostics": embedding_diagnostics,
        "raw_best_policy": raw_best_policy,
        "raw_best_score": _score(raw_output),
        "oracle_blend_score": _score(output),
        "oracle_blends": selections,
        "iterations": args.iterations,
        "feature_count": matrix.shape[1],
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
