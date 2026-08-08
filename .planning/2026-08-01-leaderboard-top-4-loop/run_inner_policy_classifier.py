"""Fit a classifier and freeze its decision policy on an inner time holdout."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from run_alternative_booster_classifier import _feature_names, _fit_model, _probability
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
from run_site_wind_classifier import (
    DECISION_GAMMAS,
    DECISION_TEMPERATURES,
    FOLDS,
    _add_site_wind_features,
)
from run_site_wind_teacher import _validation_mask

BASE_COLUMNS = ["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh"]
SHIFT_GRID = (0, -2, -1, 1, 2)
ORIGINAL_WEIGHT_GRID = (0.6, 0.7, 0.8, 0.9, 1.0)
SCALE_GRID = (0.98, 1.0, 1.02)
OFFSET_GRID = (-0.015, 0.0, 0.015)


def _group_total(
    actual_normalized: np.ndarray,
    prediction_normalized: np.ndarray,
) -> dict[str, float]:
    valid = np.isfinite(actual_normalized) & (actual_normalized >= 0.10)
    actual = actual_normalized[valid]
    prediction = prediction_normalized[valid]
    error = np.abs(prediction - actual)
    units = np.select([error <= 0.06, error <= 0.08], [4.0, 3.0], default=0.0)
    one_minus_nmae = 1.0 - float(error.mean())
    ficr = float(np.sum(actual * units) / np.sum(actual * 4.0))
    return {
        "total": 0.5 * (one_minus_nmae + ficr),
        "one_minus_nmae": one_minus_nmae,
        "ficr": ficr,
    }


def _policy_values(
    probability: np.ndarray,
    centers: np.ndarray,
    groups: np.ndarray,
    mean_generation: dict[int, float],
) -> dict[str, np.ndarray]:
    actions = np.arange(0.075, 1.076, 0.0025)
    error = np.abs(actions[:, None] - centers[None, :])
    units = np.select([error <= 0.06, error <= 0.08], [4.0, 3.0], default=0.0)
    policies: dict[str, np.ndarray] = {}
    for temperature in DECISION_TEMPERATURES:
        calibrated = probability ** (1.0 / temperature)
        calibrated /= calibrated.sum(axis=1, keepdims=True)
        for gamma in DECISION_GAMMAS:
            chosen = np.empty(len(probability), dtype=float)
            for group_id in CAPACITIES:
                mask = groups == group_id
                group_probability = calibrated[mask]
                utility = -(group_probability @ error.T) + gamma * (
                    group_probability @ (centers[None, :] * units).T
                ) / (4.0 * mean_generation[group_id])
                chosen[mask] = actions[np.argmax(utility, axis=1)]
            policies[f"T{temperature:g}_G{gamma:g}"] = chosen
    return policies


def _shift(
    values: np.ndarray,
    issuance: pd.Series,
    shift_hours: int,
) -> np.ndarray:
    if shift_hours == 0:
        return values.copy()
    series = pd.Series(values, index=issuance.index)
    shifted = series.groupby(issuance, sort=False).shift(shift_hours)
    return shifted.fillna(series).to_numpy(dtype=float)


def _transform(
    values: np.ndarray,
    issuance: pd.Series,
    selection: dict[str, object],
) -> np.ndarray:
    shift_hours = int(selection["shift_hours"])
    original_weight = float(selection["original_weight"])
    shifted = _shift(values, issuance, shift_hours)
    output = original_weight * values + (1.0 - original_weight) * shifted
    output = float(selection["scale"]) * output + float(selection["offset"])
    if bool(selection["snap"]):
        output = np.round(output / 0.0025) * 0.0025
    return np.clip(output, 0.0, 1.075)


def _select_policies(
    base: pd.DataFrame,
    policies: dict[str, np.ndarray],
) -> dict[str, dict[str, object]]:
    selections: dict[str, dict[str, object]] = {}
    groups = base["group_id"].to_numpy(dtype=int)
    for group_id, capacity in CAPACITIES.items():
        mask = groups == group_id
        actual = base.loc[mask, "actual_kwh"].to_numpy(dtype=float) / capacity
        issuance = base.loc[mask, "data_available_kst_dtm"]
        raw_rank: list[tuple[float, str]] = []
        for policy, all_values in policies.items():
            score = _group_total(actual, all_values[mask])
            raw_rank.append((score["total"], policy))
        top_policies = [policy for _, policy in sorted(raw_rank, reverse=True)[:5]]
        best: tuple[float, dict[str, object], dict[str, float]] | None = None
        for policy in top_policies:
            values = policies[policy][mask]
            for shift_hours in SHIFT_GRID:
                weights = (1.0,) if shift_hours == 0 else ORIGINAL_WEIGHT_GRID
                shifted = _shift(values, issuance, shift_hours)
                for original_weight in weights:
                    temporal = (
                        original_weight * values
                        + (1.0 - original_weight) * shifted
                    )
                    for scale in SCALE_GRID:
                        for offset in OFFSET_GRID:
                            adjusted = scale * temporal + offset
                            for snap in (False, True):
                                candidate = np.clip(adjusted, 0.0, 1.075)
                                if snap:
                                    candidate = np.round(candidate / 0.0025) * 0.0025
                                score = _group_total(actual, candidate)
                                selection: dict[str, object] = {
                                    "policy": policy,
                                    "shift_hours": shift_hours,
                                    "original_weight": original_weight,
                                    "scale": scale,
                                    "offset": offset,
                                    "snap": snap,
                                }
                                choice = (score["total"], selection, score)
                                if best is None or choice[0] > best[0]:
                                    best = choice
        assert best is not None
        selections[str(group_id)] = {
            **best[1],
            "inner_group_score": best[2],
            "top_raw_policies": top_policies,
        }
    return selections


def _apply_selections(
    base: pd.DataFrame,
    policies: dict[str, np.ndarray],
    selections: dict[str, dict[str, object]],
) -> pd.DataFrame:
    output = base[BASE_COLUMNS].copy()
    normalized = np.empty(len(base), dtype=float)
    groups = base["group_id"].to_numpy(dtype=int)
    for group_id in CAPACITIES:
        mask = groups == group_id
        selection = selections[str(group_id)]
        policy = str(selection["policy"])
        normalized[mask] = _transform(
            policies[policy][mask],
            base.loc[mask, "data_available_kst_dtm"],
            selection,
        )
    output["prediction_kwh"] = (
        normalized * output["group_id"].map(CAPACITIES).to_numpy(dtype=float)
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--fold", choices=FOLDS, required=True)
    parser.add_argument("--sitewind-cache-id", required=True)
    parser.add_argument("--calibration-days", type=int, default=30)
    parser.add_argument("--class-width", type=float, default=0.02)
    parser.add_argument("--iteration", type=int, default=100)
    args = parser.parse_args()
    if not 14 <= args.calibration_days <= 90:
        raise ValueError("calibration days must be between 14 and 90")
    if not 0.01 <= args.class_width <= 0.04:
        raise ValueError("class width must be between 0.01 and 0.04")
    if not 20 <= args.iteration <= 300:
        raise ValueError("iteration must be between 20 and 300")
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
    calibration_start = validation_start - pd.Timedelta(
        int(args.calibration_days), unit="D"
    )
    calibration = training & surface["forecast_kst_dtm"].ge(calibration_start).to_numpy()
    inner_fit = training & ~calibration
    for group_id in CAPACITIES:
        if int((calibration & surface["group_id"].eq(group_id).to_numpy()).sum()) < 300:
            raise RuntimeError(f"group {group_id} inner calibration is too small")
        if int((inner_fit & surface["group_id"].eq(group_id).to_numpy()).sum()) < 300:
            raise RuntimeError(f"group {group_id} inner fit is too small")

    raw_bins = np.floor(
        (normalized_target.clip(0.10, 1.074999) - 0.10) / args.class_width
    ).astype("Int64")
    active_bins = [
        int(value)
        for value in sorted(raw_bins.loc[training].dropna().astype(int).unique())
    ]
    bin_to_class = {bin_id: class_id for class_id, bin_id in enumerate(active_bins)}
    classes = raw_bins.map(bin_to_class).astype("Int64")
    inner_classes = set(classes.loc[inner_fit].dropna().astype(int))
    if inner_classes != set(range(len(active_bins))):
        raise RuntimeError("inner fit does not cover every active target class")
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
    matrix = matrix[selected_features]

    inner_model = _fit_model(
        "xgboost",
        matrix,
        classes,
        inner_fit,
        normalized_target.loc[inner_fit].clip(lower=0.10).to_numpy(dtype=float),
        len(active_bins),
        args.iteration,
    )
    inner_probability = _probability(
        inner_model, "xgboost", matrix, calibration, args.iteration
    )
    inner_groups = surface.loc[calibration, "group_id"].to_numpy(dtype=int)
    inner_means = {
        group_id: float(
            normalized_target.loc[
                inner_fit & surface["group_id"].eq(group_id).to_numpy()
            ].mean()
        )
        for group_id in CAPACITIES
    }
    inner_policies = _policy_values(
        inner_probability, centers, inner_groups, inner_means
    )
    inner_base = surface.loc[
        calibration, [*BASE_COLUMNS, "data_available_kst_dtm"]
    ].copy()
    selections = _select_policies(inner_base, inner_policies)
    inner_output = _apply_selections(inner_base, inner_policies, selections)

    final_model = _fit_model(
        "xgboost",
        matrix,
        classes,
        training,
        normalized_target.loc[training].clip(lower=0.10).to_numpy(dtype=float),
        len(active_bins),
        args.iteration,
    )
    probability = _probability(
        final_model, "xgboost", matrix, validation, args.iteration
    )
    validation_groups = surface.loc[validation, "group_id"].to_numpy(dtype=int)
    training_means = {
        group_id: float(
            normalized_target.loc[
                training & surface["group_id"].eq(group_id).to_numpy()
            ].mean()
        )
        for group_id in CAPACITIES
    }
    policies = _policy_values(probability, centers, validation_groups, training_means)
    base = surface.loc[
        validation, [*BASE_COLUMNS, "data_available_kst_dtm"]
    ].copy()
    output = _apply_selections(base, policies, selections)
    output["fold_id"] = args.fold
    output["model_id"] = args.candidate_id
    output_path = OUTPUT / f"{args.candidate_id}-{args.fold}.parquet"
    output.to_parquet(output_path, index=False)

    receipt = {
        "schema_version": 1,
        "candidate_id": args.candidate_id,
        "fold_id": args.fold,
        "architecture": "xgboost_multiclass_inner_temporal_policy",
        "scope": "outer validation labels never used for model or policy selection",
        "calibration_days": args.calibration_days,
        "calibration_start": calibration_start.isoformat(),
        "inner_fit_rows": int(inner_fit.sum()),
        "inner_calibration_rows": int(calibration.sum()),
        "training_rows": int(training.sum()),
        "class_width": args.class_width,
        "class_count": len(active_bins),
        "iteration": args.iteration,
        "selections": selections,
        "inner_score": _score(inner_output),
        "fold_score": _score(output),
        "feature_count": len(selected_features),
        "sitewind_feature_count": len(sitewind_columns),
        "runtime_seconds": round(time.perf_counter() - started, 2),
        "prediction_path": str(output_path.relative_to(Path.cwd())),
        "prediction_sha256": _sha256(output_path),
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
