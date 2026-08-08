"""Screen alternative multiclass boosters on the fixed M102 feature contract."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
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
from run_site_wind_classifier import FOLDS, _add_site_wind_features, _choose_actions
from run_site_wind_teacher import _validation_mask
from xgboost import XGBClassifier

BASE_COLUMNS = ["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh"]
PARENT_PATH = OUTPUT / "M107_STRICT_TEMPORAL_TOP100-oof.parquet"


def _feature_names(fold: str) -> list[str]:
    receipt_path = OUTPUT / f"M102_TOP100-{fold}.json"
    receipt = json.loads(receipt_path.read_text())
    names = receipt["selected_feature_names"]
    if len(names) != 100 or len(set(names)) != 100:
        raise RuntimeError("M102 top-100 feature contract changed")
    return names


def _fit_model(
    family: str,
    matrix: pd.DataFrame,
    classes: pd.Series,
    training: np.ndarray,
    sample_weight: np.ndarray,
    class_count: int,
    max_iteration: int,
) -> object:
    if family in {"lgbm_gbdt", "lgbm_dart"}:
        model = LGBMClassifier(
            objective="multiclass",
            num_class=class_count,
            n_estimators=max_iteration,
            learning_rate=0.025,
            num_leaves=15,
            min_child_samples=80,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=2.0,
            boosting_type="dart" if family == "lgbm_dart" else "gbdt",
            drop_rate=0.05,
            skip_drop=0.5,
            random_state=20260802,
            n_jobs=6,
            deterministic=True,
            force_col_wise=True,
            verbosity=-1,
        )
    elif family == "xgboost":
        model = XGBClassifier(
            objective="multi:softprob",
            num_class=class_count,
            n_estimators=max_iteration,
            learning_rate=0.03,
            max_depth=5,
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
    elif family == "catboost":
        model = CatBoostClassifier(
            loss_function="MultiClass",
            iterations=max_iteration,
            learning_rate=0.03,
            depth=7,
            l2_leaf_reg=5.0,
            random_strength=0.3,
            random_seed=20260802,
            thread_count=6,
            allow_writing_files=False,
            verbose=False,
        )
    else:
        raise ValueError(f"unknown family: {family}")
    model.fit(
        matrix.loc[training],
        classes.loc[training].astype(int),
        sample_weight=sample_weight,
    )
    return model


def _probability(
    model: object,
    family: str,
    matrix: pd.DataFrame,
    validation: np.ndarray,
    iteration: int,
) -> np.ndarray:
    if family in {"lgbm_gbdt", "lgbm_dart"}:
        return model.predict_proba(matrix.loc[validation], num_iteration=iteration)
    if family == "xgboost":
        return model.predict_proba(
            matrix.loc[validation], iteration_range=(0, iteration)
        )
    return model.predict_proba(matrix.loc[validation], ntree_end=iteration)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--fold", choices=FOLDS, default=FOLDS[-1])
    parser.add_argument("--sitewind-cache-id", required=True)
    parser.add_argument(
        "--family",
        choices=("lgbm_gbdt", "lgbm_dart", "xgboost", "catboost"),
        required=True,
    )
    parser.add_argument("--iterations", nargs="+", type=int, required=True)
    parser.add_argument("--sequence-feature-count", type=int, default=0)
    parser.add_argument("--training-floor", type=float, choices=(0.0, 0.1), default=0.1)
    parser.add_argument("--class-width", type=float, default=0.02)
    parser.add_argument("--season-bandwidth-days", type=float, default=0.0)
    parser.add_argument(
        "--season-groups", nargs="+", type=int, choices=tuple(CAPACITIES), default=[]
    )
    args = parser.parse_args()
    iterations = sorted(set(args.iterations))
    if not iterations or iterations[0] < 1 or iterations[-1] > 500:
        raise ValueError("iterations must be between one and 500")
    if not 0 <= args.sequence_feature_count <= 100:
        raise ValueError("sequence-feature-count must be between zero and 100")
    if not 0.005 <= args.class_width <= 0.05:
        raise ValueError("class-width must be between 0.005 and 0.05")
    if not 0.0 <= args.season_bandwidth_days <= 183.0:
        raise ValueError("season-bandwidth-days must be between zero and 183")
    if args.season_bandwidth_days > 0 and not args.season_groups:
        raise ValueError("season-groups are required when seasonal weighting is active")
    if _sha256(OPEN) != OPEN_SHA or _sha256(BASELINE) != BASELINE_SHA:
        raise RuntimeError("immutable competition input hash mismatch")
    started = time.perf_counter()
    surface, base_columns, _ = _surface()
    validation = _validation_mask(surface, args.fold)
    start = surface.loc[validation, "forecast_kst_dtm"].min()
    preceding = surface["forecast_kst_dtm"].lt(start).to_numpy()
    normalized_target = surface["actual_kwh"] / surface["group_id"].map(CAPACITIES)
    training = (
        preceding
        & surface["actual_kwh"].notna().to_numpy()
        & normalized_target.ge(args.training_floor).to_numpy()
    )
    raw_bins = np.floor(
        (
            normalized_target.clip(args.training_floor, 1.074999)
            - args.training_floor
        )
        / args.class_width
    ).astype("Int64")
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
    missing_features = set(selected_features).difference(matrix.columns)
    if missing_features:
        raise RuntimeError(f"missing M102 features: {sorted(missing_features)}")
    matrix = matrix[selected_features]
    sequence_source_features = selected_features[: args.sequence_feature_count]
    sequence_columns: list[str] = []
    if sequence_source_features:
        group_keys = [surface["group_id"], surface["data_available_kst_dtm"]]
        additions: dict[str, pd.Series] = {}
        for name in sequence_source_features:
            current = matrix[name]
            grouped = current.groupby(group_keys, sort=False)
            neighbors: dict[int, pd.Series] = {}
            for offset in (-2, -1, 1, 2):
                shifted = grouped.shift(-offset).fillna(current)
                feature = f"seq__{name}__h{offset:+d}"
                additions[feature] = shifted
                sequence_columns.append(feature)
                neighbors[offset] = shifted
            local3 = pd.concat([neighbors[-1], current, neighbors[1]], axis=1)
            derived = {
                f"seq__{name}__mean3": local3.mean(axis=1),
                f"seq__{name}__slope2": neighbors[1] - neighbors[-1],
                f"seq__{name}__curvature": (
                    neighbors[1] - 2.0 * current + neighbors[-1]
                ),
            }
            additions.update(derived)
            sequence_columns.extend(derived)
        matrix = pd.concat([matrix, pd.DataFrame(additions, index=matrix.index)], axis=1)
    sample_weight = normalized_target.loc[training].clip(lower=0.10).to_numpy()
    seasonal_diagnostics: dict[str, object] = {"active": False}
    if args.season_bandwidth_days > 0:
        validation_times = surface.loc[validation, "forecast_kst_dtm"].sort_values()
        center = pd.Timestamp(validation_times.iloc[len(validation_times) // 2])
        day = surface.loc[training, "forecast_kst_dtm"].dt.dayofyear.to_numpy(float)
        distance = np.abs(day - float(center.dayofyear))
        distance = np.minimum(distance, 365.25 - distance)
        factor = np.exp(-0.5 * (distance / args.season_bandwidth_days) ** 2)
        training_groups = surface.loc[training, "group_id"].to_numpy(dtype=int)
        apply_season = np.isin(training_groups, args.season_groups)
        factor /= max(float(factor[apply_season].mean()), 1e-6)
        factor = np.clip(factor, 0.10, 4.0)
        sample_weight *= np.where(apply_season, factor, 1.0)
        seasonal_diagnostics = {
            "active": True,
            "center_dayofyear": center.dayofyear,
            "bandwidth_days": args.season_bandwidth_days,
            "groups": args.season_groups,
            "factor_min": float(factor[apply_season].min()),
            "factor_mean": float(factor[apply_season].mean()),
            "factor_max": float(factor[apply_season].max()),
        }
    model = _fit_model(
        args.family,
        matrix,
        classes,
        training,
        sample_weight,
        len(active_bins),
        max(iterations),
    )
    base = surface.loc[validation, BASE_COLUMNS].copy()
    parent = pd.read_parquet(PARENT_PATH)
    parent = parent.loc[parent["fold_id"].eq(args.fold)]
    sweep: dict[str, object] = {}
    best: tuple[
        float,
        int,
        pd.DataFrame,
        pd.DataFrame,
        dict[str, object],
        str,
        dict[str, float],
    ] | None = None
    for iteration in iterations:
        probability = _probability(
            model, args.family, matrix, validation, iteration
        )
        eligible_centers = centers >= 0.10
        probability = probability[:, eligible_centers]
        probability /= probability.sum(axis=1, keepdims=True)
        utility_centers = centers[eligible_centers]
        utility_training = training & normalized_target.ge(0.10).to_numpy()
        raw_output, raw_policy, _, policies = _choose_actions(
            base,
            probability,
            utility_centers,
            normalized_target,
            utility_training,
            surface["group_id"],
        )
        blended, selections = _screen_blends(base, policies, parent)
        raw_score = _score(raw_output)
        blend_score = _score(blended)
        sweep[str(iteration)] = {
            "raw_best_policy": raw_policy,
            "raw_best_score": raw_score,
            "oracle_blend_score": blend_score,
            "oracle_blends": selections,
        }
        choice = (
            blend_score["total"],
            iteration,
            blended,
            policies,
            selections,
            raw_policy,
            raw_score,
        )
        if best is None or choice[0] > best[0]:
            best = choice
        print(
            json.dumps(
                {
                    "iteration": iteration,
                    "raw_policy": raw_policy,
                    "raw_score": raw_score,
                    "oracle_blend_score": blend_score,
                    "oracle_blends": selections,
                }
            ),
            flush=True,
        )
    assert best is not None
    output = best[2]
    output["fold_id"] = args.fold
    output["model_id"] = args.candidate_id
    output_path = OUTPUT / f"{args.candidate_id}-{args.fold}.parquet"
    policy_path = OUTPUT / f"{args.candidate_id}-{args.fold}-policies.parquet"
    output.to_parquet(output_path, index=False)
    best[3].to_parquet(policy_path, index=False)
    receipt = {
        "schema_version": 1,
        "candidate_id": args.candidate_id,
        "fold_id": args.fold,
        "architecture": f"{args.family}_multiclass_fixed_m102_features",
        "scope": "Q4 booster-family representation screen; oracle blends not promoted",
        "selected_iteration": best[1],
        "selected_raw_policy": best[5],
        "selected_raw_score": best[6],
        "selected_oracle_blend_score": _score(output),
        "selected_oracle_blends": best[4],
        "sweep": sweep,
        "feature_count": matrix.shape[1],
        "base_feature_count": len(selected_features),
        "sequence_source_feature_count": len(sequence_source_features),
        "sequence_feature_count": len(sequence_columns),
        "sequence_source_features": sequence_source_features,
        "training_floor": args.training_floor,
        "class_width": args.class_width,
        "seasonal_weighting": seasonal_diagnostics,
        "utility_condition_floor": 0.1,
        "sitewind_feature_count": len(sitewind_columns),
        "selected_feature_names": selected_features,
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
