"""Classify power residuals around a cross-fitted empirical site-wind curve."""

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
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold

BASE_COLUMNS = ["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh"]
PARENT_PATH = OUTPUT / "M107_STRICT_TEMPORAL_TOP100-oof.parquet"
TARGET_WIDTH = 0.02
TARGET_CENTERS = np.arange(0.11, 1.071, TARGET_WIDTH)


def _sitewind_baselines(
    surface: pd.DataFrame,
    sitewind: np.ndarray,
    preceding: np.ndarray,
    validation: np.ndarray,
    target: pd.Series,
) -> tuple[np.ndarray, dict[str, object]]:
    baseline = np.full(len(surface), np.nan, dtype="float32")
    diagnostics: dict[str, object] = {}
    for group_id in CAPACITIES:
        group = surface["group_id"].eq(group_id).to_numpy()
        training = preceding & group & target.ge(0.10).to_numpy()
        positions = np.flatnonzero(training)
        splitter = KFold(5, shuffle=True, random_state=20260802 + group_id)
        for fit_index, holdout_index in splitter.split(positions):
            fit_positions = positions[fit_index]
            holdout_positions = positions[holdout_index]
            mapper = IsotonicRegression(
                y_min=0.075,
                y_max=1.075,
                out_of_bounds="clip",
            )
            mapper.fit(
                sitewind[fit_positions],
                target.iloc[fit_positions],
                sample_weight=target.iloc[fit_positions].clip(lower=0.10),
            )
            baseline[holdout_positions] = mapper.predict(sitewind[holdout_positions])
        mapper = IsotonicRegression(
            y_min=0.075,
            y_max=1.075,
            out_of_bounds="clip",
        )
        mapper.fit(
            sitewind[positions],
            target.iloc[positions],
            sample_weight=target.iloc[positions].clip(lower=0.10),
        )
        apply = validation & group
        baseline[apply] = mapper.predict(sitewind[apply])
        residual = target.iloc[positions] - baseline[positions]
        diagnostics[str(group_id)] = {
            "training_rows": len(positions),
            "baseline_mae": float(residual.abs().mean()),
            "residual_mean": float(residual.mean()),
            "residual_std": float(residual.std()),
        }
    required = preceding & target.ge(0.10).to_numpy() | validation
    if not np.isfinite(baseline[required]).all():
        raise RuntimeError("site-wind baseline is non-finite on required rows")
    return baseline, diagnostics


def _align_target_probability(
    residual_probability: np.ndarray,
    residual_centers: np.ndarray,
    baseline: np.ndarray,
) -> np.ndarray:
    aligned = np.zeros((len(baseline), len(TARGET_CENTERS)), dtype="float32")
    targets = np.clip(
        baseline[:, None] + residual_centers[None, :],
        TARGET_CENTERS[0],
        TARGET_CENTERS[-1],
    )
    bins = np.rint((targets - TARGET_CENTERS[0]) / TARGET_WIDTH).astype(int)
    bins = np.clip(bins, 0, len(TARGET_CENTERS) - 1)
    rows = np.arange(len(baseline))
    for class_id in range(len(residual_centers)):
        np.add.at(
            aligned,
            (rows, bins[:, class_id]),
            residual_probability[:, class_id],
        )
    aligned /= aligned.sum(axis=1, keepdims=True)
    return aligned


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--fold", choices=FOLDS, required=True)
    parser.add_argument("--sitewind-cache-id", required=True)
    parser.add_argument("--iterations", type=int, default=120)
    parser.add_argument("--residual-width", type=float, default=0.02)
    parser.add_argument("--residual-limit", type=float, default=0.50)
    args = parser.parse_args()
    if not 40 <= args.iterations <= 300:
        raise ValueError("iterations must be between 40 and 300")
    if not 0.01 <= args.residual_width <= 0.05:
        raise ValueError("residual width must be between 0.01 and 0.05")
    if not 0.20 <= args.residual_limit <= 1.0:
        raise ValueError("residual limit must be between 0.20 and 1.0")
    if _sha256(OPEN) != OPEN_SHA or _sha256(BASELINE) != BASELINE_SHA:
        raise RuntimeError("immutable competition input hash mismatch")

    started = time.perf_counter()
    surface, base_columns, _ = _surface()
    validation = _validation_mask(surface, args.fold)
    validation_start = surface.loc[validation, "forecast_kst_dtm"].min()
    preceding = surface["forecast_kst_dtm"].lt(validation_start).to_numpy()
    target = surface["actual_kwh"] / surface["group_id"].map(CAPACITIES)
    training = preceding & target.ge(0.10).to_numpy()
    cached = np.load(
        OUTPUT / f"{args.sitewind_cache_id}-{args.fold}-sitewind-features.npz"
    )
    sitewind = (cached["legacy"] + cached["allweather"]) / 2.0
    baseline, baseline_diagnostics = _sitewind_baselines(
        surface,
        sitewind,
        preceding,
        validation,
        target,
    )
    residual = target - baseline
    residual_bins = np.floor(
        (
            residual.clip(-args.residual_limit, args.residual_limit - 1e-9)
            + args.residual_limit
        )
        / args.residual_width
    ).astype("Int64")
    active_bins = [
        int(value)
        for value in sorted(residual_bins.loc[training].dropna().astype(int).unique())
    ]
    bin_to_class = {bin_id: class_id for class_id, bin_id in enumerate(active_bins)}
    classes = residual_bins.map(bin_to_class).astype("Int64")
    residual_centers = np.asarray(
        [
            residual.loc[training & classes.eq(class_id)].mean()
            for class_id in range(len(active_bins))
        ]
    )

    matrix = surface[base_columns].astype("float32")
    sitewind_columns = _add_site_wind_features(
        matrix, cached["legacy"], cached["allweather"]
    )
    selected_features = _feature_names(args.fold)
    matrix["sitewind__isotonic_baseline"] = baseline
    matrix["sitewind__isotonic_baseline2"] = baseline**2
    feature_names = list(
        dict.fromkeys(
            [
                *selected_features,
                "sitewind__isotonic_baseline",
                "sitewind__isotonic_baseline2",
            ]
        )
    )
    model = _fit_model(
        "xgboost",
        matrix[feature_names],
        classes,
        training,
        target.loc[training].clip(lower=0.10).to_numpy(dtype=float),
        len(active_bins),
        args.iterations,
    )
    residual_probability = _probability(
        model,
        "xgboost",
        matrix[feature_names],
        validation,
        args.iterations,
    )
    target_probability = _align_target_probability(
        residual_probability,
        residual_centers,
        baseline[validation],
    )
    base = surface.loc[validation, BASE_COLUMNS].copy()
    validation_groups = base["group_id"].to_numpy(dtype=int)
    means = {
        group_id: float(
            target.loc[training & surface["group_id"].eq(group_id).to_numpy()].mean()
        )
        for group_id in CAPACITIES
    }
    policy_predictions: dict[str, np.ndarray] = {}
    raw_scores: dict[str, dict[str, float]] = {}
    for policy, normalized in _policy_values(
        target_probability,
        TARGET_CENTERS,
        validation_groups,
        means,
    ).items():
        prediction = (
            normalized * base["group_id"].map(CAPACITIES).to_numpy(dtype=float)
        )
        policy_predictions[policy] = prediction
        raw_scores[policy] = _score(base.assign(prediction_kwh=prediction))
    policies = pd.concat(
        [base.reset_index(drop=True), pd.DataFrame(policy_predictions)], axis=1
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
        "architecture": "crossfit_sitewind_curve_residual_xgboost_distribution",
        "scope": "unseen-fold supplied-data residual distribution screen",
        "iterations": args.iterations,
        "residual_width": args.residual_width,
        "residual_limit": args.residual_limit,
        "residual_class_count": len(active_bins),
        "residual_centers": residual_centers.tolist(),
        "baseline_diagnostics": baseline_diagnostics,
        "feature_count": len(feature_names),
        "sitewind_feature_count": len(sitewind_columns),
        "raw_best_policy": raw_best_policy,
        "raw_best_score": _score(raw_output),
        "oracle_blend_score": _score(output),
        "oracle_blends": selections,
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
