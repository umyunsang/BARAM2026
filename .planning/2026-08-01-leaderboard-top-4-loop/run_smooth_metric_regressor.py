"""Screen smooth approximations of the exact NMAE-plus-FICR objective."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from run_alternative_booster_classifier import _feature_names
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
from run_site_wind_classifier import FOLDS, _add_site_wind_features
from run_site_wind_teacher import _validation_mask
from xgboost import XGBRegressor

BASE_COLUMNS = ["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh"]
PARENT_PATH = OUTPUT / "M107_STRICT_TEMPORAL_TOP100-oof.parquet"
SPECS = (
    (0.0, 0.010),
    (0.5, 0.010),
    (1.0, 0.010),
    (1.5, 0.010),
    (1.0, 0.020),
    (1.5, 0.020),
)


def _objective(
    gamma: float,
    tau: float,
    mean_generation: float,
) -> Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    def smooth_metric(
        target: np.ndarray,
        prediction: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        residual = prediction - target
        epsilon = 0.004
        radius = np.sqrt(residual**2 + epsilon**2)
        first_radius = residual / radius
        second_radius = epsilon**2 / radius**3
        settlement_weight = gamma * target / (4.0 * mean_generation)
        band_slope = np.zeros_like(radius)
        band_curvature = np.zeros_like(radius)
        for coefficient, threshold in ((3.0, 0.08), (1.0, 0.06)):
            exponent = np.clip((radius - threshold) / tau, -60.0, 60.0)
            probability = 1.0 / (1.0 + np.exp(exponent))
            local = probability * (1.0 - probability)
            band_slope += coefficient * local
            band_curvature += coefficient * local * (1.0 - 2.0 * probability)
        slope_multiplier = 1.0 + settlement_weight * band_slope / tau
        slope_derivative = -settlement_weight * band_curvature / tau**2
        gradient = first_radius * slope_multiplier
        hessian = (
            second_radius * slope_multiplier
            + first_radius**2 * slope_derivative
        )
        return gradient, np.clip(hessian, 1e-3, 100.0)

    return smooth_metric


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--fold", choices=FOLDS, required=True)
    parser.add_argument("--sitewind-cache-id", required=True)
    parser.add_argument("--iterations", type=int, default=300)
    args = parser.parse_args()
    if not 50 <= args.iterations <= 600:
        raise ValueError("iterations must be between 50 and 600")
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
    base = surface.loc[validation, BASE_COLUMNS].copy()
    policies = base.copy()
    sweep: dict[str, object] = {}
    for gamma, tau in SPECS:
        tag = f"G{gamma:g}_T{tau:g}"
        model = XGBRegressor(
            objective=_objective(
                gamma,
                tau,
                float(normalized_target.loc[training].mean()),
            ),
            n_estimators=args.iterations,
            learning_rate=0.025,
            max_depth=5,
            min_child_weight=20.0,
            subsample=0.9,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=5.0,
            max_bin=256,
            tree_method="hist",
            base_score=float(normalized_target.loc[training].median()),
            random_state=20260802,
            n_jobs=6,
        )
        model.fit(
            matrix.loc[training],
            normalized_target.loc[training].to_numpy(dtype=float),
        )
        normalized = np.clip(model.predict(matrix.loc[validation]), 0.0, 1.075)
        trial = base.copy()
        trial["prediction_kwh"] = (
            normalized * trial["group_id"].map(CAPACITIES).to_numpy(dtype=float)
        )
        policies[tag] = trial["prediction_kwh"].to_numpy(dtype=float)
        sweep[tag] = {"gamma": gamma, "tau": tau, "raw_score": _score(trial)}
        print(json.dumps({"spec": tag, **sweep[tag]}), flush=True)

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
        "architecture": "smooth_official_metric_xgboost_regression",
        "scope": "same-fold objective screen; oracle blends are diagnostic only",
        "specs": [{"gamma": gamma, "tau": tau} for gamma, tau in SPECS],
        "iterations": args.iterations,
        "sweep": sweep,
        "oracle_blend_score": _score(output),
        "oracle_blends": selections,
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
