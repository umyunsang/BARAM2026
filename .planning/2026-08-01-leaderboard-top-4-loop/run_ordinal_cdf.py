"""Fit an ordinal CDF and freeze settlement actions on an inner time holdout."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from run_alternative_booster_classifier import _feature_names
from run_inner_policy_classifier import _group_total
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
from sklearn.linear_model import LogisticRegression

BASE_COLUMNS = ["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh"]
CDF_VARIANTS = ("raw", "pooled_platt", "group_platt")
TEMPERATURES = (0.60, 0.75, 0.90, 1.00, 1.15, 1.35)
GAMMAS = (0.50, 0.75, 1.00, 1.25, 1.50, 2.00)
ACTIONS = np.arange(0.075, 1.0751, 0.0025)


@dataclass(frozen=True)
class PlattMap:
    coefficient: float
    intercept: float

    def apply(self, probability: np.ndarray) -> np.ndarray:
        clipped = np.clip(probability, 1e-6, 1.0 - 1e-6)
        logit = np.log(clipped / (1.0 - clipped))
        value = self.coefficient * logit + self.intercept
        return 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))


def _fit_platt(probability: np.ndarray, target: np.ndarray) -> PlattMap | None:
    positives = int(target.sum())
    negatives = int(len(target) - positives)
    if positives < 30 or negatives < 30:
        return None
    clipped = np.clip(probability, 1e-6, 1.0 - 1e-6)
    logit = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    model = LogisticRegression(C=10.0, solver="lbfgs", max_iter=500)
    model.fit(logit, target.astype(int))
    return PlattMap(
        coefficient=float(model.coef_[0, 0]),
        intercept=float(model.intercept_[0]),
    )


def _model(iterations: int, seed: int) -> LGBMClassifier:
    return LGBMClassifier(
        objective="binary",
        n_estimators=iterations,
        learning_rate=0.03,
        num_leaves=15,
        min_child_samples=80,
        subsample=0.90,
        subsample_freq=1,
        colsample_bytree=0.80,
        reg_alpha=0.1,
        reg_lambda=2.0,
        random_state=seed,
        n_jobs=6,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )


def _monotone_survival(internal: np.ndarray) -> np.ndarray:
    clipped = np.clip(internal, 0.0, 1.0)
    return np.minimum.accumulate(clipped, axis=1)


def _survival_to_mass(internal: np.ndarray) -> np.ndarray:
    survival = np.column_stack(
        [
            np.ones(len(internal), dtype=float),
            _monotone_survival(internal),
            np.zeros(len(internal), dtype=float),
        ]
    )
    mass = survival[:, :-1] - survival[:, 1:]
    mass = np.maximum(mass, 0.0)
    denominator = mass.sum(axis=1, keepdims=True)
    if (denominator <= 0.0).any():
        raise RuntimeError("ordinal CDF produced an empty probability row")
    return mass / denominator


def _bin_centers(
    normalized_target: pd.Series,
    groups: pd.Series,
    mask: np.ndarray,
    edges: np.ndarray,
) -> dict[int, np.ndarray]:
    centers: dict[int, np.ndarray] = {}
    midpoint = 0.5 * (edges[:-1] + edges[1:])
    for group_id in CAPACITIES:
        values = normalized_target.loc[mask & groups.eq(group_id).to_numpy()].to_numpy(
            dtype=float
        )
        group_centers = midpoint.copy()
        bins = np.searchsorted(edges, values, side="right") - 1
        bins = np.clip(bins, 0, len(midpoint) - 1)
        for bin_id in range(len(midpoint)):
            selected = values[bins == bin_id]
            if len(selected):
                group_centers[bin_id] = float(selected.mean())
        centers[group_id] = group_centers
    return centers


def _actions_from_mass(
    mass: np.ndarray,
    centers: np.ndarray,
    mean_generation: float,
    temperature: float,
    gamma: float,
) -> np.ndarray:
    calibrated = np.maximum(mass, 1e-12) ** (1.0 / temperature)
    calibrated /= calibrated.sum(axis=1, keepdims=True)
    error = np.abs(ACTIONS[:, None] - centers[None, :])
    units = np.select([error <= 0.06, error <= 0.08], [4.0, 3.0], default=0.0)
    expected_error = calibrated @ error.T
    expected_settlement = calibrated @ (centers[None, :] * units).T
    utility = -expected_error + gamma * expected_settlement / (
        4.0 * mean_generation
    )
    return ACTIONS[np.argmax(utility, axis=1)]


def _variant_internal(
    raw: np.ndarray,
    groups: np.ndarray,
    pooled_maps: list[PlattMap | None],
    group_maps: dict[int, list[PlattMap | None]],
) -> dict[str, np.ndarray]:
    pooled = raw.copy()
    for index, mapping in enumerate(pooled_maps):
        if mapping is not None:
            pooled[:, index] = mapping.apply(raw[:, index])
    grouped = raw.copy()
    for group_id in CAPACITIES:
        rows = groups == group_id
        for index, mapping in enumerate(group_maps[group_id]):
            fallback = pooled_maps[index]
            chosen = mapping if mapping is not None else fallback
            if chosen is not None:
                grouped[rows, index] = chosen.apply(raw[rows, index])
    return {
        "raw": raw,
        "pooled_platt": pooled,
        "group_platt": grouped,
    }


def _select_inner_policy(
    variants: dict[str, np.ndarray],
    base: pd.DataFrame,
    centers: dict[int, np.ndarray],
    means: dict[int, float],
) -> tuple[dict[int, dict[str, float | str]], dict[str, object]]:
    selections: dict[int, dict[str, float | str]] = {}
    diagnostics: dict[str, object] = {}
    groups = base["group_id"].to_numpy(dtype=int)
    for group_id, capacity in CAPACITIES.items():
        rows = groups == group_id
        actual = base.loc[rows, "actual_kwh"].to_numpy(dtype=float) / capacity
        best: tuple[float, str, float, float, dict[str, float]] | None = None
        sweep: dict[str, dict[str, float]] = {}
        for variant in CDF_VARIANTS:
            mass = _survival_to_mass(variants[variant][rows])
            for temperature in TEMPERATURES:
                for gamma in GAMMAS:
                    prediction = _actions_from_mass(
                        mass,
                        centers[group_id],
                        means[group_id],
                        temperature,
                        gamma,
                    )
                    score = _group_total(actual, prediction)
                    tag = f"{variant}_T{temperature:g}_G{gamma:g}"
                    sweep[tag] = score
                    choice = (
                        score["total"],
                        variant,
                        temperature,
                        gamma,
                        score,
                    )
                    if best is None or choice[0] > best[0]:
                        best = choice
        assert best is not None
        selections[group_id] = {
            "variant": best[1],
            "temperature": best[2],
            "gamma": best[3],
        }
        diagnostics[str(group_id)] = {
            "selection": selections[group_id],
            "score": best[4],
            "top_five": [
                {"policy": tag, "score": score}
                for tag, score in sorted(
                    sweep.items(), key=lambda item: item[1]["total"], reverse=True
                )[:5]
            ],
        }
    return selections, diagnostics


def _apply_policy(
    variants: dict[str, np.ndarray],
    groups: np.ndarray,
    centers: dict[int, np.ndarray],
    means: dict[int, float],
    selections: dict[int, dict[str, float | str]],
) -> np.ndarray:
    prediction = np.empty(len(groups), dtype=float)
    for group_id in CAPACITIES:
        rows = groups == group_id
        selection = selections[group_id]
        variant = str(selection["variant"])
        mass = _survival_to_mass(variants[variant][rows])
        prediction[rows] = _actions_from_mass(
            mass,
            centers[group_id],
            means[group_id],
            float(selection["temperature"]),
            float(selection["gamma"]),
        )
    return prediction


def _outer_oracle(
    variants: dict[str, np.ndarray],
    base: pd.DataFrame,
    centers: dict[int, np.ndarray],
    means: dict[int, float],
) -> dict[str, object]:
    groups = base["group_id"].to_numpy(dtype=int)
    normalized = np.empty(len(base), dtype=float)
    selections: dict[str, object] = {}
    for group_id, capacity in CAPACITIES.items():
        rows = groups == group_id
        actual = base.loc[rows, "actual_kwh"].to_numpy(dtype=float) / capacity
        best: tuple[float, np.ndarray, str, float, float, dict[str, float]] | None = None
        for variant in CDF_VARIANTS:
            mass = _survival_to_mass(variants[variant][rows])
            for temperature in TEMPERATURES:
                for gamma in GAMMAS:
                    prediction = _actions_from_mass(
                        mass,
                        centers[group_id],
                        means[group_id],
                        temperature,
                        gamma,
                    )
                    score = _group_total(actual, prediction)
                    choice = (
                        score["total"],
                        prediction,
                        variant,
                        temperature,
                        gamma,
                        score,
                    )
                    if best is None or choice[0] > best[0]:
                        best = choice
        assert best is not None
        normalized[rows] = best[1]
        selections[str(group_id)] = {
            "variant": best[2],
            "temperature": best[3],
            "gamma": best[4],
            "score": best[5],
        }
    output = base[BASE_COLUMNS].copy()
    output["prediction_kwh"] = (
        normalized * output["group_id"].map(CAPACITIES).to_numpy(dtype=float)
    )
    return {"score": _score(output), "selections": selections}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--fold", choices=FOLDS, required=True)
    parser.add_argument("--sitewind-cache-id", required=True)
    parser.add_argument("--iterations", type=int, default=120)
    parser.add_argument("--calibration-days", type=int, default=60)
    parser.add_argument("--bin-width", type=float, choices=(0.025, 0.05), default=0.025)
    args = parser.parse_args()
    if not 40 <= args.iterations <= 300:
        raise ValueError("iterations must be between 40 and 300")
    if not 30 <= args.calibration_days <= 120:
        raise ValueError("calibration-days must be between 30 and 120")
    if _sha256(OPEN) != OPEN_SHA or _sha256(BASELINE) != BASELINE_SHA:
        raise RuntimeError("immutable competition input hash mismatch")

    started = time.perf_counter()
    surface, base_features, _ = _surface()
    validation = _validation_mask(surface, args.fold)
    preceding = _strict_preceding_mask(surface, validation)
    normalized_target = surface["actual_kwh"] / surface["group_id"].map(CAPACITIES)
    eligible = surface["actual_kwh"].notna().to_numpy() & normalized_target.ge(
        0.10
    ).to_numpy()
    training = preceding & eligible
    validation_start = pd.Timestamp(
        surface.loc[validation, "forecast_kst_dtm"].min()
    )
    calibration_start = validation_start - pd.Timedelta(
        days=args.calibration_days
    )
    calibration = training & surface["forecast_kst_dtm"].ge(
        calibration_start
    ).to_numpy()
    inner_fit = training & ~calibration
    for group_id in CAPACITIES:
        if int((inner_fit & surface["group_id"].eq(group_id).to_numpy()).sum()) < 500:
            raise RuntimeError(f"group {group_id} inner fit is too small")
        if int((calibration & surface["group_id"].eq(group_id).to_numpy()).sum()) < 300:
            raise RuntimeError(f"group {group_id} calibration is too small")

    cache = np.load(
        OUTPUT / f"{args.sitewind_cache_id}-{args.fold}-sitewind-features.npz"
    )
    matrix = surface[base_features].astype("float32")
    sitewind_columns = _add_site_wind_features(
        matrix, cache["legacy"], cache["allweather"]
    )
    selected_features = _feature_names(args.fold)
    missing = set(selected_features).difference(matrix.columns)
    if missing:
        raise RuntimeError(f"ordinal CDF is missing fixed features: {sorted(missing)}")
    matrix = matrix[selected_features]

    edges = np.arange(0.10, 1.0751, args.bin_width)
    if not np.isclose(edges[-1], 1.075):
        raise RuntimeError("ordinal edge contract must terminate at 1.075")
    thresholds = edges[1:-1]
    calibration_probability = np.empty((int(calibration.sum()), len(thresholds)))
    validation_probability = np.empty((int(validation.sum()), len(thresholds)))
    pooled_maps: list[PlattMap | None] = []
    group_maps: dict[int, list[PlattMap | None]] = {
        group_id: [] for group_id in CAPACITIES
    }
    calibration_groups = surface.loc[calibration, "group_id"].to_numpy(dtype=int)
    for threshold_index, threshold in enumerate(thresholds):
        inner_target = normalized_target.loc[inner_fit].gt(threshold).astype(int)
        inner_model = _model(args.iterations, 20260803 + threshold_index)
        inner_model.fit(matrix.loc[inner_fit], inner_target)
        calibration_raw = inner_model.predict_proba(matrix.loc[calibration])[:, 1]
        calibration_probability[:, threshold_index] = calibration_raw
        calibration_target = normalized_target.loc[calibration].gt(threshold).to_numpy(
            dtype=int
        )
        pooled_maps.append(_fit_platt(calibration_raw, calibration_target))
        for group_id in CAPACITIES:
            rows = calibration_groups == group_id
            group_maps[group_id].append(
                _fit_platt(calibration_raw[rows], calibration_target[rows])
            )

        final_target = normalized_target.loc[training].gt(threshold).astype(int)
        final_model = _model(args.iterations, 20260803 + threshold_index)
        final_model.fit(matrix.loc[training], final_target)
        validation_probability[:, threshold_index] = final_model.predict_proba(
            matrix.loc[validation]
        )[:, 1]
        if (threshold_index + 1) % 5 == 0 or threshold_index + 1 == len(thresholds):
            print(
                json.dumps(
                    {
                        "threshold_models_completed": threshold_index + 1,
                        "threshold_models_total": len(thresholds),
                    }
                ),
                flush=True,
            )

    calibration_variants = _variant_internal(
        calibration_probability,
        calibration_groups,
        pooled_maps,
        group_maps,
    )
    validation_groups = surface.loc[validation, "group_id"].to_numpy(dtype=int)
    validation_variants = _variant_internal(
        validation_probability,
        validation_groups,
        pooled_maps,
        group_maps,
    )
    centers = _bin_centers(
        normalized_target, surface["group_id"], inner_fit, edges
    )
    inner_means = {
        group_id: float(
            normalized_target.loc[
                inner_fit & surface["group_id"].eq(group_id).to_numpy()
            ].mean()
        )
        for group_id in CAPACITIES
    }
    training_means = {
        group_id: float(
            normalized_target.loc[
                training & surface["group_id"].eq(group_id).to_numpy()
            ].mean()
        )
        for group_id in CAPACITIES
    }
    calibration_base = surface.loc[calibration, BASE_COLUMNS].copy()
    selections, inner_diagnostics = _select_inner_policy(
        calibration_variants,
        calibration_base,
        centers,
        inner_means,
    )
    validation_base = surface.loc[validation, BASE_COLUMNS].copy()
    normalized_prediction = _apply_policy(
        validation_variants,
        validation_groups,
        centers,
        training_means,
        selections,
    )
    output = validation_base.copy()
    output["prediction_kwh"] = (
        normalized_prediction
        * output["group_id"].map(CAPACITIES).to_numpy(dtype=float)
    )
    output["fold_id"] = args.fold
    output["model_id"] = args.candidate_id
    output_path = OUTPUT / f"{args.candidate_id}-{args.fold}.parquet"
    output.to_parquet(output_path, index=False)

    oracle = _outer_oracle(
        validation_variants,
        validation_base,
        centers,
        training_means,
    )
    calibration_maps = {
        "pooled": [None if item is None else item.__dict__ for item in pooled_maps],
        "groups": {
            str(group_id): [
                None if item is None else item.__dict__
                for item in group_maps[group_id]
            ]
            for group_id in CAPACITIES
        },
    }
    receipt = {
        "schema_version": 1,
        "candidate_id": args.candidate_id,
        "fold_id": args.fold,
        "architecture": "strict_shared_lightgbm_ordinal_cdf_inner_platt",
        "scope": "outer labels excluded from model, calibration, and policy selection",
        "fold_score": _score(output),
        "outer_oracle_diagnostic": oracle,
        "inner_policy_selections": {
            str(group_id): value for group_id, value in selections.items()
        },
        "inner_policy_diagnostics": inner_diagnostics,
        "calibration_maps": calibration_maps,
        "calibration_days": args.calibration_days,
        "inner_fit_rows": int(inner_fit.sum()),
        "calibration_rows": int(calibration.sum()),
        "training_rows": int(training.sum()),
        "validation_rows": int(validation.sum()),
        "bin_width": args.bin_width,
        "thresholds": thresholds.tolist(),
        "iterations": args.iterations,
        "feature_count": len(selected_features),
        "selected_feature_names": selected_features,
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
