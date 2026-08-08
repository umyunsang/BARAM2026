"""Screen source-rank classifiers augmented with supervised daily NWP scores."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from run_alternative_booster_classifier import _feature_names, _fit_model, _probability
from run_consensus_classifier import _screen_blends
from run_daily_pca_classifier import _trajectory_features
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
from run_source_rank_ensemble import MIXTURES, _source_columns, _source_probability
from sklearn.cross_decomposition import PLSRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

BASE_COLUMNS = ["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh"]
PARENT_PATH = OUTPUT / "M107_STRICT_TEMPORAL_TOP100-oof.parquet"


def _daily_pls_features(
    surface: pd.DataFrame,
    matrix: pd.DataFrame,
    value_columns: list[str],
    validation_start: pd.Timestamp,
    components: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Fit label-supervised daily projections on preceding complete profiles only."""
    feature_names = [f"daily_pls__score_{index:02d}" for index in range(components)]
    feature_names.extend(
        [
            "daily_pls__profile_current",
            "daily_pls__profile_prev1",
            "daily_pls__profile_next1",
            "daily_pls__profile_prev2",
            "daily_pls__profile_next2",
            "daily_pls__profile_mean",
            "daily_pls__profile_std",
            "daily_pls__profile_max",
            "daily_pls__profile_slope2",
            "daily_pls__profile_curvature",
        ]
    )
    values = np.full((len(surface), len(feature_names)), np.nan, dtype="float32")
    diagnostics: dict[str, object] = {}
    normalized_target = surface["actual_kwh"] / surface["group_id"].map(CAPACITIES)
    for group_id in CAPACITIES:
        group_positions = np.flatnonzero(surface["group_id"].eq(group_id).to_numpy())
        group = surface.iloc[group_positions][
            ["data_available_kst_dtm", "forecast_kst_dtm"]
        ].join(matrix.iloc[group_positions][value_columns])
        group["normalized_target"] = normalized_target.iloc[group_positions]
        group = group.sort_values(["data_available_kst_dtm", "forecast_kst_dtm"])
        sizes = group.groupby("data_available_kst_dtm", sort=True).size()
        if not sizes.eq(24).all():
            raise RuntimeError("daily PLS requires 24 horizons per issuance")
        issuances = group["data_available_kst_dtm"].drop_duplicates().reset_index(drop=True)
        x = group[value_columns].to_numpy(dtype="float32").reshape(len(issuances), -1)
        y = group["normalized_target"].to_numpy(dtype="float32").reshape(
            len(issuances), 24
        )
        day_end = (
            group.groupby("data_available_kst_dtm", sort=True)["forecast_kst_dtm"]
            .max()
            .reset_index(drop=True)
        )
        fit_days = day_end.lt(validation_start).to_numpy() & np.isfinite(y).all(axis=1)
        component_count = min(components, int(fit_days.sum()) - 1, x.shape[1], y.shape[1])
        if component_count != components:
            raise RuntimeError(
                f"group {group_id} cannot support {components} daily PLS components"
            )
        imputer = SimpleImputer(strategy="median")
        x_fit = imputer.fit_transform(x[fit_days])
        x_all = imputer.transform(x)
        scaler = StandardScaler()
        x_fit = scaler.fit_transform(x_fit)
        x_all = scaler.transform(x_all)
        model = PLSRegression(
            n_components=components,
            scale=False,
            max_iter=1_000,
            tol=1e-7,
        )
        model.fit(x_fit, y[fit_days])
        scores = model.transform(x_all).astype("float32")
        profile = np.clip(model.predict(x_all), 0.0, 1.075).astype("float32")
        previous1 = np.concatenate([profile[:, :1], profile[:, :-1]], axis=1)
        next1 = np.concatenate([profile[:, 1:], profile[:, -1:]], axis=1)
        previous2 = np.concatenate([profile[:, :2], profile[:, :-2]], axis=1)
        next2 = np.concatenate([profile[:, 2:], profile[:, -2:]], axis=1)
        repeated_scores = np.repeat(scores, 24, axis=0)
        profile_mean = np.repeat(profile.mean(axis=1), 24)
        profile_std = np.repeat(profile.std(axis=1), 24)
        profile_max = np.repeat(profile.max(axis=1), 24)
        additions = np.column_stack(
            [
                repeated_scores,
                profile.reshape(-1),
                previous1.reshape(-1),
                next1.reshape(-1),
                previous2.reshape(-1),
                next2.reshape(-1),
                profile_mean,
                profile_std,
                profile_max,
                (next1 - previous1).reshape(-1),
                (next1 - 2.0 * profile + previous1).reshape(-1),
            ]
        ).astype("float32")
        ordered_positions = group.index.to_numpy(dtype=int)
        values[ordered_positions] = additions
        fit_prediction = profile[fit_days]
        diagnostics[str(group_id)] = {
            "fit_days": int(fit_days.sum()),
            "all_days": len(issuances),
            "x_columns": int(x.shape[1]),
            "components": components,
            "fit_normalized_mae": float(np.mean(np.abs(y[fit_days] - fit_prediction))),
            "iterations_max": int(np.max(model.n_iter_)),
        }
    if not np.isfinite(values).all():
        raise RuntimeError("daily PLS features contain non-finite values")
    return pd.DataFrame(values, index=surface.index, columns=feature_names), diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--fold", choices=FOLDS, required=True)
    parser.add_argument("--sitewind-cache-id", required=True)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--source-top-features", type=int, default=100)
    parser.add_argument("--trajectory-features", type=int, default=20)
    parser.add_argument("--components", type=int, default=8)
    args = parser.parse_args()
    if not 40 <= args.iterations <= 200:
        raise ValueError("iterations must be between 40 and 200")
    if not 60 <= args.source_top_features <= 160:
        raise ValueError("source top-feature count must be between 60 and 160")
    if not 12 <= args.trajectory_features <= 40:
        raise ValueError("trajectory feature count must be between 12 and 40")
    if not 2 <= args.components <= 16:
        raise ValueError("PLS component count must be between two and 16")
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
    trajectory_features = _trajectory_features(
        global_features, args.trajectory_features
    )
    pls_features, pls_diagnostics = _daily_pls_features(
        surface,
        matrix,
        trajectory_features,
        validation_start,
        args.components,
    )
    matrix = pd.concat([matrix, pls_features], axis=1)
    augmented_global_features = [*global_features, *pls_features.columns]
    global_model = _fit_model(
        "xgboost",
        matrix[augmented_global_features],
        classes,
        training,
        sample_weight,
        len(active_bins),
        args.iterations,
    )
    global_probability = _probability(
        global_model,
        "xgboost",
        matrix[augmented_global_features],
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
        "architecture": "supervised_daily_pls_source_rank_xgboost",
        "scope": (
            "same-fold representation screen; PLS fit on preceding complete "
            "target profiles and supplied NWP only"
        ),
        "trajectory_feature_names": trajectory_features,
        "component_count": args.components,
        "pls_feature_count": len(pls_features.columns),
        "pls_diagnostics": pls_diagnostics,
        "gfs_feature_count": len(gfs_selected),
        "gfs_selected_features": gfs_selected,
        "ldaps_feature_count": len(ldaps_selected),
        "ldaps_selected_features": ldaps_selected,
        "raw_best_policy": raw_best_policy,
        "raw_best_score": _score(raw_output),
        "oracle_blend_score": _score(output),
        "oracle_blends": selections,
        "iterations": args.iterations,
        "global_feature_count": len(augmented_global_features),
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
