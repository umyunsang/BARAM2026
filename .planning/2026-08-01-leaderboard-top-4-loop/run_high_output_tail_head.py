"""Screen a strict high-output hurdle/tail head over the frozen parent forecast.

The head targets the measured Q3 compression above 50% capacity.  It learns
binary high-output probabilities and conditional tail quantiles from complete
preceding issuance batches only.  Supplied SCADA contributes only through the
already cross-fitted NWP-to-site-wind cache; observed SCADA is never a feature.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
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
from run_site_wind_teacher import _strict_preceding_mask, _validation_mask
from sklearn.isotonic import IsotonicRegression

BASE_COLUMNS = ["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh"]
PARENT_PATH = OUTPUT / "M107_STRICT_TEMPORAL_TOP100-oof.parquet"
TAIL_THRESHOLDS = (0.50, 0.70, 0.90)
TAIL_QUANTILES = (0.50, 0.75)


def _classifier_parameters(iterations: int, seed: int) -> dict[str, object]:
    return {
        "objective": "binary",
        "n_estimators": iterations,
        "learning_rate": 0.025,
        "num_leaves": 31,
        "min_child_samples": 80,
        "max_bin": 255,
        "subsample": 0.9,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.2,
        "reg_lambda": 4.0,
        "random_state": seed,
        "n_jobs": 6,
        "deterministic": True,
        "force_col_wise": True,
        "verbosity": -1,
    }


def _regressor_parameters(
    iterations: int,
    quantile: float,
    seed: int,
) -> dict[str, object]:
    return {
        "objective": "quantile",
        "alpha": quantile,
        "n_estimators": iterations,
        "learning_rate": 0.025,
        "num_leaves": 31,
        "min_child_samples": 60,
        "max_bin": 255,
        "subsample": 0.9,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.2,
        "reg_lambda": 4.0,
        "random_state": seed,
        "n_jobs": 6,
        "deterministic": True,
        "force_col_wise": True,
        "verbosity": -1,
    }


def _calibrated_probabilities(
    matrix: pd.DataFrame,
    target: pd.Series,
    training: np.ndarray,
    validation: np.ndarray,
    weights: np.ndarray,
    threshold: float,
    iterations: int,
    seed: int,
    issuance: pd.Series,
) -> tuple[np.ndarray, dict[str, object]]:
    batches = issuance.loc[training].drop_duplicates().sort_values()
    cutoff = batches.iloc[int(len(batches) * 0.80)]
    inner_fit = training & issuance.lt(cutoff).to_numpy()
    calibration = training & ~inner_fit
    labels = target.ge(threshold).astype("int8")
    if labels.loc[inner_fit].nunique() != 2 or labels.loc[calibration].nunique() != 2:
        raise RuntimeError(f"tail threshold {threshold} lacks both inner classes")
    inner_model = LGBMClassifier(**_classifier_parameters(iterations, seed))
    inner_model.fit(
        matrix.loc[inner_fit],
        labels.loc[inner_fit],
        sample_weight=weights[inner_fit],
    )
    raw_calibration = inner_model.predict_proba(matrix.loc[calibration])[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(
        raw_calibration,
        labels.loc[calibration].to_numpy(dtype=float),
        sample_weight=weights[calibration],
    )
    final_model = LGBMClassifier(**_classifier_parameters(iterations, seed))
    final_model.fit(
        matrix.loc[training],
        labels.loc[training],
        sample_weight=weights[training],
    )
    raw_validation = final_model.predict_proba(matrix.loc[validation])[:, 1]
    probability = np.asarray(calibrator.predict(raw_validation), dtype=float)
    return probability, {
        "threshold": threshold,
        "inner_fit_rows": int(inner_fit.sum()),
        "calibration_rows": int(calibration.sum()),
        "training_positive_rate": float(labels.loc[training].mean()),
        "calibration_raw_mean": float(np.mean(raw_calibration)),
        "calibration_observed_rate": float(labels.loc[calibration].mean()),
        "validation_probability_mean": float(np.mean(probability)),
    }


def _group_scores(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    scores: dict[str, dict[str, float]] = {}
    for group_id, capacity in CAPACITIES.items():
        part = frame.loc[frame["group_id"].eq(group_id)]
        actual = part["actual_kwh"].to_numpy(dtype=float)
        prediction = part["prediction_kwh"].to_numpy(dtype=float)
        valid = actual >= 0.10 * capacity
        actual = actual[valid]
        error = np.abs(prediction[valid] - actual) / capacity
        units = np.select(
            [error <= 0.06, error <= 0.08], [4.0, 3.0], default=0.0
        )
        one_minus_nmae = 1.0 - float(np.mean(error))
        ficr = float(np.sum(actual * units) / np.sum(actual * 4.0))
        scores[str(group_id)] = {
            "total": 0.5 * (one_minus_nmae + ficr),
            "one_minus_nmae": one_minus_nmae,
            "ficr": ficr,
        }
    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--fold", choices=FOLDS, required=True)
    parser.add_argument("--sitewind-cache-id", required=True)
    parser.add_argument("--iterations", type=int, default=180)
    parser.add_argument("--top-features", type=int, default=140)
    args = parser.parse_args()
    if not 80 <= args.iterations <= 400:
        raise ValueError("iterations must be between 80 and 400")
    if not 60 <= args.top_features <= 240:
        raise ValueError("top feature count must be between 60 and 240")
    if _sha256(OPEN) != OPEN_SHA or _sha256(BASELINE) != BASELINE_SHA:
        raise RuntimeError("immutable competition input hash mismatch")

    started = time.perf_counter()
    surface, feature_columns, _ = _surface()
    validation = _validation_mask(surface, args.fold)
    preceding = _strict_preceding_mask(surface, validation)
    normalized_target = surface["actual_kwh"] / surface["group_id"].map(CAPACITIES)
    training = (
        preceding
        & surface["actual_kwh"].notna().to_numpy()
        & normalized_target.ge(0.10).to_numpy()
    )
    cached = np.load(
        OUTPUT / f"{args.sitewind_cache_id}-{args.fold}-sitewind-features.npz"
    )
    matrix = surface[feature_columns].astype("float32")
    sitewind_columns = _add_site_wind_features(
        matrix, cached["legacy"], cached["allweather"]
    )

    group3 = surface["group_id"].eq(3).to_numpy()
    base_weights = normalized_target.clip(lower=0.10).to_numpy(dtype=float)
    base_weights *= np.where(group3, 2.0, 1.0)
    screening_labels = normalized_target.ge(0.50).astype("int8")
    screening = LGBMClassifier(
        **_classifier_parameters(min(args.iterations, 120), 2026080301)
    )
    screening.fit(
        matrix.loc[training],
        screening_labels.loc[training],
        sample_weight=base_weights[training],
    )
    gains = screening.booster_.feature_importance(importance_type="gain")
    selected_positions = np.argsort(gains)[::-1][: args.top_features]
    selected_features = [matrix.columns[index] for index in selected_positions]
    selected = matrix[selected_features]

    probability: dict[float, np.ndarray] = {}
    probability_diagnostics: dict[str, object] = {}
    for position, threshold in enumerate(TAIL_THRESHOLDS):
        values, diagnostics = _calibrated_probabilities(
            selected,
            normalized_target,
            training,
            validation,
            base_weights,
            threshold,
            args.iterations,
            2026080310 + position,
            surface["data_available_kst_dtm"],
        )
        probability[threshold] = values
        probability_diagnostics[str(threshold)] = diagnostics

    tail_training = training & normalized_target.ge(0.45).to_numpy()
    tail_weights = base_weights[tail_training] * normalized_target.loc[
        tail_training
    ].to_numpy(dtype=float)
    tail_predictions: dict[float, np.ndarray] = {}
    for position, quantile in enumerate(TAIL_QUANTILES):
        model = LGBMRegressor(
            **_regressor_parameters(
                args.iterations + 40,
                quantile,
                2026080320 + position,
            )
        )
        model.fit(
            selected.loc[tail_training],
            normalized_target.loc[tail_training],
            sample_weight=tail_weights,
        )
        tail_predictions[quantile] = np.clip(
            model.predict(selected.loc[validation]), 0.45, 1.075
        )

    base = surface.loc[validation, BASE_COLUMNS].copy()
    parent = pd.read_parquet(PARENT_PATH)
    parent = parent.loc[parent["fold_id"].eq(args.fold)]
    keys = ["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh"]
    aligned = base.merge(
        parent[[*keys, "prediction_kwh"]],
        on=keys,
        validate="one_to_one",
    )
    capacity = aligned["group_id"].map(CAPACITIES).to_numpy(dtype=float)
    parent_normalized = aligned["prediction_kwh"].to_numpy(dtype=float) / capacity
    validation_groups = aligned["group_id"].to_numpy(dtype=int)
    gate_profiles = {
        "P50": probability[0.50],
        "SQRT50": np.sqrt(probability[0.50]),
        "P70": probability[0.70],
        "SQRT70": np.sqrt(probability[0.70]),
        "HIER": (
            0.50 * probability[0.50]
            + 0.35 * probability[0.70]
            + 0.15 * probability[0.90]
        ),
    }
    policy_predictions: dict[str, np.ndarray] = {"PARENT": aligned["prediction_kwh"]}
    raw_scores: dict[str, dict[str, float]] = {
        "PARENT": _score(aligned.rename(columns={"prediction_kwh": "prediction_kwh"}))
    }
    for quantile, tail_prediction in tail_predictions.items():
        gap = np.maximum(tail_prediction - parent_normalized, 0.0)
        for gate_name, gate in gate_profiles.items():
            for strength in (0.50, 1.00, 1.50, 2.00):
                for cap in (0.15, 0.30):
                    uplift = np.minimum(strength * gate * gap, cap)
                    for scope in ("G3", "ALL"):
                        apply = (
                            validation_groups == 3
                            if scope == "G3"
                            else np.ones(len(aligned), dtype=bool)
                        )
                        normalized = parent_normalized.copy()
                        normalized[apply] += uplift[apply]
                        normalized = np.clip(normalized, 0.0, 1.075)
                        tag = (
                            f"Q{quantile:g}_{gate_name}_S{strength:g}_"
                            f"C{cap:g}_{scope}"
                        )
                        prediction = normalized * capacity
                        policy_predictions[tag] = prediction
                        raw_scores[tag] = _score(
                            aligned[keys].assign(prediction_kwh=prediction)
                        )

    policies = pd.concat(
        [aligned[keys], pd.DataFrame(policy_predictions, index=aligned.index)],
        axis=1,
    )
    raw_best_policy = max(raw_scores, key=lambda name: raw_scores[name]["total"])
    raw_output = aligned[keys].assign(
        prediction_kwh=policy_predictions[raw_best_policy]
    )
    blended, selections = _screen_blends(aligned[keys], policies, parent)
    output = blended.assign(fold_id=args.fold, model_id=args.candidate_id)
    output_path = OUTPUT / f"{args.candidate_id}-{args.fold}.parquet"
    policy_path = OUTPUT / f"{args.candidate_id}-{args.fold}-policies.parquet"
    output.to_parquet(output_path, index=False)
    policies.to_parquet(policy_path, index=False)
    receipt = {
        "schema_version": 1,
        "candidate_id": args.candidate_id,
        "fold_id": args.fold,
        "architecture": "strict_high_output_hurdle_conditional_tail_quantiles",
        "scope": "outer-fold representation screen; same-fold policy scores are diagnostic",
        "strict_preceding_rows": int(preceding.sum()),
        "training_rows": int(training.sum()),
        "tail_training_rows": int(tail_training.sum()),
        "feature_count": len(selected_features),
        "selected_feature_names": selected_features,
        "sitewind_feature_count": len(sitewind_columns),
        "probability_diagnostics": probability_diagnostics,
        "tail_quantiles": list(TAIL_QUANTILES),
        "raw_best_policy": raw_best_policy,
        "raw_best_score": _score(raw_output),
        "raw_best_group_scores": _group_scores(raw_output),
        "oracle_blend_score": _score(output),
        "oracle_blend_group_scores": _group_scores(output),
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
