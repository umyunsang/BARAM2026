"""Add cross-fitted GFS-only and LDAPS-only site-wind calibration features."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from run_alternative_booster_classifier import _feature_names, _fit_model, _probability
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
from run_source_rank_ensemble import _source_columns
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

BASE_COLUMNS = ["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh"]
PARENT_PATH = OUTPUT / "M107_STRICT_TEMPORAL_TOP100-oof.parquet"


def _params(iterations: int, seed: int) -> dict[str, object]:
    return {
        "objective": "l2",
        "n_estimators": iterations,
        "learning_rate": 0.035,
        "num_leaves": 31,
        "min_child_samples": 60,
        "max_bin": 255,
        "subsample": 0.9,
        "subsample_freq": 1,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.1,
        "reg_lambda": 3.0,
        "random_state": seed,
        "n_jobs": 6,
        "deterministic": True,
        "force_col_wise": True,
        "verbosity": -1,
    }


def _select_source_features(
    matrix: pd.DataFrame,
    columns: list[str],
    target: pd.Series,
    training: np.ndarray,
    count: int,
    source_offset: int,
) -> list[str]:
    model = LGBMRegressor(**_params(80, 20260802 + source_offset))
    model.fit(matrix.loc[training, columns], target.loc[training])
    gain = model.booster_.feature_importance(importance_type="gain")
    order = np.argsort(gain)[::-1]
    selected = [columns[index] for index in order[:count]]
    if len(selected) != count or len(set(selected)) != count:
        raise RuntimeError("source site-wind feature selection changed")
    return selected


def _crossfit_source(
    matrix: pd.DataFrame,
    columns: list[str],
    target: pd.Series,
    training: np.ndarray,
    validation: np.ndarray,
    groups: pd.Series,
    iterations: int,
    source_offset: int,
) -> tuple[np.ndarray, dict[str, object]]:
    prediction = np.full(len(matrix), np.nan, dtype="float32")
    diagnostics: dict[str, object] = {}
    for group_id in CAPACITIES:
        group = groups.eq(group_id).to_numpy()
        fit_mask = training & group
        positions = np.flatnonzero(fit_mask)
        splitter = KFold(3, shuffle=True, random_state=20260802 + group_id)
        for split_id, (fit_index, holdout_index) in enumerate(
            splitter.split(positions)
        ):
            model = LGBMRegressor(
                **_params(
                    iterations,
                    20260802 + 100 * source_offset + 10 * group_id + split_id,
                )
            )
            model.fit(
                matrix.iloc[positions[fit_index]][columns],
                target.iloc[positions[fit_index]],
            )
            prediction[positions[holdout_index]] = model.predict(
                matrix.iloc[positions[holdout_index]][columns]
            )
        model = LGBMRegressor(
            **_params(iterations, 20260802 + 100 * source_offset + group_id)
        )
        model.fit(matrix.loc[fit_mask, columns], target.loc[fit_mask])
        apply = validation & group
        prediction[apply] = model.predict(matrix.loc[apply, columns])
        error = prediction[positions] - target.iloc[positions].to_numpy(dtype=float)
        diagnostics[str(group_id)] = {
            "training_rows": len(positions),
            "oof_mae": float(np.mean(np.abs(error))),
            "oof_rmse": float(np.sqrt(np.mean(error**2))),
        }
    required = training | validation
    if not np.isfinite(prediction[required]).all():
        raise RuntimeError("source-specific site-wind prediction is non-finite")
    return prediction, diagnostics


def _gate_matrix(gfs: np.ndarray, ldaps: np.ndarray) -> np.ndarray:
    mean = (gfs + ldaps) / 2.0
    delta = ldaps - gfs
    return np.column_stack(
        [
            gfs,
            ldaps,
            mean,
            delta,
            np.abs(delta),
            gfs**2,
            ldaps**2,
            mean**2,
            gfs * ldaps,
        ]
    ).astype("float32")


def _crossfit_gate(
    gfs: np.ndarray,
    ldaps: np.ndarray,
    target: pd.Series,
    training: np.ndarray,
    validation: np.ndarray,
    groups: pd.Series,
) -> tuple[np.ndarray, dict[str, object]]:
    features = _gate_matrix(gfs, ldaps)
    prediction = np.full(len(features), np.nan, dtype="float32")
    diagnostics: dict[str, object] = {}
    for group_id in CAPACITIES:
        group = groups.eq(group_id).to_numpy()
        positions = np.flatnonzero(training & group)
        splitter = KFold(5, shuffle=True, random_state=20260802 + group_id)
        for fit_index, holdout_index in splitter.split(positions):
            model = Ridge(alpha=10.0)
            model.fit(features[positions[fit_index]], target.iloc[positions[fit_index]])
            prediction[positions[holdout_index]] = model.predict(
                features[positions[holdout_index]]
            )
        model = Ridge(alpha=10.0)
        model.fit(features[positions], target.iloc[positions])
        apply = validation & group
        prediction[apply] = model.predict(features[apply])
        error = prediction[positions] - target.iloc[positions].to_numpy(dtype=float)
        diagnostics[str(group_id)] = {
            "training_rows": len(positions),
            "oof_mae": float(np.mean(np.abs(error))),
            "oof_rmse": float(np.sqrt(np.mean(error**2))),
            "coefficients": model.coef_.tolist(),
        }
    required = training | validation
    prediction[required] = np.clip(prediction[required], 0.0, 40.0)
    if not np.isfinite(prediction[required]).all():
        raise RuntimeError("source gate prediction is non-finite")
    return prediction, diagnostics


def _add_source_sitewind_features(
    matrix: pd.DataFrame,
    gfs: np.ndarray,
    ldaps: np.ndarray,
    gated: np.ndarray,
) -> list[str]:
    additions: dict[str, np.ndarray] = {
        "source_sitewind__gfs": gfs,
        "source_sitewind__ldaps": ldaps,
        "source_sitewind__gated": gated,
        "source_sitewind__mean": (gfs + ldaps) / 2.0,
        "source_sitewind__delta": ldaps - gfs,
        "source_sitewind__disagreement": np.abs(ldaps - gfs),
    }
    for source in ("gfs", "ldaps", "gated", "mean"):
        value = additions[f"source_sitewind__{source}"]
        additions[f"source_sitewind__{source}2"] = value**2
        additions[f"source_sitewind__{source}3"] = value**3
        normalized = np.clip((value - 3.0) / 9.0, 0.0, 1.0)
        additions[f"source_sitewind__{source}_powercurve"] = normalized**3
    addition_frame = pd.DataFrame(additions, index=matrix.index).astype("float32")
    for name in addition_frame:
        matrix[name] = addition_frame[name]
    return addition_frame.columns.tolist()


def _wind_metrics(
    actual: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, float | int]:
    valid = np.isfinite(actual) & np.isfinite(prediction)
    error = prediction[valid] - actual[valid]
    return {
        "count": int(valid.sum()),
        "correlation": float(np.corrcoef(actual[valid], prediction[valid])[0, 1]),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "bias": float(np.mean(error)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--fold", choices=FOLDS, required=True)
    parser.add_argument("--sitewind-cache-id", required=True)
    parser.add_argument("--source-feature-count", type=int, default=120)
    parser.add_argument("--wind-iterations", type=int, default=120)
    parser.add_argument("--power-iterations", type=int, default=120)
    args = parser.parse_args()
    if not 60 <= args.source_feature_count <= 200:
        raise ValueError("source feature count must be between 60 and 200")
    if not 40 <= args.wind_iterations <= 300:
        raise ValueError("wind iterations must be between 40 and 300")
    if not 40 <= args.power_iterations <= 300:
        raise ValueError("power iterations must be between 40 and 300")
    if _sha256(OPEN) != OPEN_SHA or _sha256(BASELINE) != BASELINE_SHA:
        raise RuntimeError("immutable competition input hash mismatch")

    started = time.perf_counter()
    surface, base_columns, _ = _surface()
    validation = _validation_mask(surface, args.fold)
    validation_start = surface.loc[validation, "forecast_kst_dtm"].min()
    preceding = surface["forecast_kst_dtm"].lt(validation_start).to_numpy()
    wind_training = preceding & surface["scada_ws"].notna().to_numpy()
    normalized_target = surface["actual_kwh"] / surface["group_id"].map(CAPACITIES)
    power_training = preceding & normalized_target.ge(0.10).to_numpy()
    matrix = surface[base_columns].astype("float32")
    cached = np.load(
        OUTPUT / f"{args.sitewind_cache_id}-{args.fold}-sitewind-features.npz"
    )
    legacy_sitewind_columns = _add_site_wind_features(
        matrix, cached["legacy"], cached["allweather"]
    )
    source_predictions: dict[str, np.ndarray] = {}
    source_diagnostics: dict[str, object] = {}
    selected_by_source: dict[str, list[str]] = {}
    for source_id, source in enumerate(("gfs", "ldaps"), start=1):
        columns = _source_columns(base_columns, source)
        selected = _select_source_features(
            matrix,
            columns,
            surface["scada_ws"],
            wind_training,
            args.source_feature_count,
            source_id,
        )
        prediction, diagnostics = _crossfit_source(
            matrix,
            selected,
            surface["scada_ws"],
            wind_training,
            validation,
            surface["group_id"],
            args.wind_iterations,
            source_id,
        )
        source_predictions[source] = prediction
        source_diagnostics[source] = diagnostics
        selected_by_source[source] = selected
    gated, gate_diagnostics = _crossfit_gate(
        source_predictions["gfs"],
        source_predictions["ldaps"],
        surface["scada_ws"],
        wind_training,
        validation,
        surface["group_id"],
    )
    source_sitewind_columns = _add_source_sitewind_features(
        matrix,
        source_predictions["gfs"],
        source_predictions["ldaps"],
        gated,
    )
    raw_bins = np.floor(
        (normalized_target.clip(0.10, 1.074999) - 0.10) / 0.02
    ).astype("Int64")
    active_bins = [
        int(value)
        for value in sorted(raw_bins.loc[power_training].dropna().astype(int).unique())
    ]
    bin_to_class = {bin_id: class_id for class_id, bin_id in enumerate(active_bins)}
    classes = raw_bins.map(bin_to_class).astype("Int64")
    centers = np.asarray(
        [
            normalized_target.loc[power_training & classes.eq(class_id)].mean()
            for class_id in range(len(active_bins))
        ]
    )
    selected_features = list(
        dict.fromkeys([*_feature_names(args.fold), *source_sitewind_columns])
    )
    model = _fit_model(
        "xgboost",
        matrix[selected_features],
        classes,
        power_training,
        normalized_target.loc[power_training].clip(lower=0.10).to_numpy(dtype=float),
        len(active_bins),
        args.power_iterations,
    )
    probability = _probability(
        model,
        "xgboost",
        matrix[selected_features],
        validation,
        args.power_iterations,
    )
    base = surface.loc[validation, BASE_COLUMNS].copy()
    raw_output, raw_policy, _, policies = _choose_actions(
        base,
        probability,
        centers,
        normalized_target,
        power_training,
        surface["group_id"],
    )
    parent = pd.read_parquet(PARENT_PATH)
    parent = parent.loc[parent["fold_id"].eq(args.fold)]
    blended, selections = _screen_blends(base, policies, parent)
    output = blended.assign(fold_id=args.fold, model_id=args.candidate_id)
    output_path = OUTPUT / f"{args.candidate_id}-{args.fold}.parquet"
    policy_path = OUTPUT / f"{args.candidate_id}-{args.fold}-policies.parquet"
    output.to_parquet(output_path, index=False)
    policies.to_parquet(policy_path, index=False)
    observed_validation_wind = surface.loc[validation, "scada_ws"].to_numpy(dtype=float)
    wind_metrics = {
        source: _wind_metrics(observed_validation_wind, prediction[validation])
        for source, prediction in {
            **source_predictions,
            "gated": gated,
        }.items()
    }
    receipt = {
        "schema_version": 1,
        "candidate_id": args.candidate_id,
        "fold_id": args.fold,
        "architecture": "crossfit_source_specific_sitewind_xgboost_power_distribution",
        "scope": "unseen-fold source-calibrated site-wind representation screen",
        "source_feature_count": args.source_feature_count,
        "wind_iterations": args.wind_iterations,
        "power_iterations": args.power_iterations,
        "selected_features_by_source": selected_by_source,
        "source_crossfit_diagnostics": source_diagnostics,
        "gate_crossfit_diagnostics": gate_diagnostics,
        "validation_wind_diagnostics_only": wind_metrics,
        "feature_count": len(selected_features),
        "legacy_sitewind_feature_count": len(legacy_sitewind_columns),
        "source_sitewind_feature_count": len(source_sitewind_columns),
        "raw_best_policy": raw_policy,
        "raw_best_score": _score(raw_output),
        "oracle_blend_score": _score(output),
        "oracle_blends": selections,
        "runtime_seconds": round(time.perf_counter() - started, 2),
        "prediction_path": str(output_path.relative_to(Path.cwd())),
        "prediction_sha256": _sha256(output_path),
        "policy_path": str(policy_path.relative_to(Path.cwd())),
        "policy_sha256": _sha256(policy_path),
        "observed_validation_scada_used_for_power_prediction": False,
        "new_2024_evaluation": False,
        "lockbox_reopened": False,
        "external_actions": [],
    }
    receipt_path = OUTPUT / f"{args.candidate_id}-{args.fold}.json"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
