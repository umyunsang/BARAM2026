"""Strict per-grid power experts with two-stage chronological stacking."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
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
)
from run_site_wind_teacher import _strict_preceding_mask, _validation_mask
from sklearn.linear_model import Ridge
from strict_dev_surface import DEV_CUTOFF, development_surface

BASE_COLUMNS = ["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh"]
TOP_K = (1, 3, 5, 8, 12)
RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
SCALES = (0.98, 1.00, 1.02)
OFFSETS = (-0.01, 0.0, 0.01)
GRID_PATTERN = re.compile(r"^(gfs|ldaps)__grid(\d{2})__")


def _strict_before(surface: pd.DataFrame, cutoff: pd.Timestamp) -> np.ndarray:
    batch_last = surface.groupby("data_available_kst_dtm", sort=False)[
        "forecast_kst_dtm"
    ].transform("max")
    return batch_last.lt(cutoff).to_numpy()


def _common_columns(columns: list[str]) -> list[str]:
    exact = {
        "group_id",
        "hour",
        "month",
        "day_of_year",
        "lead_hour",
        "operating_year",
        "operating_quarter",
        "capacity_kwh",
        "turbine_count",
        "hub_height_m",
        "rotor_diameter_m",
        "latitude_centroid",
        "longitude_centroid",
    }
    return [
        name
        for name in columns
        if name in exact or name.startswith(("cal__", "group_"))
    ]


def _expert_features(columns: list[str]) -> dict[str, list[str]]:
    common = _common_columns(columns)
    prefixes = sorted(
        {
            match.group(0)
            for name in columns
            if (match := GRID_PATTERN.match(name)) is not None
        }
    )
    experts: dict[str, list[str]] = {}
    for prefix in prefixes:
        source = prefix.split("__", maxsplit=1)[0]
        grid = prefix.split("__", maxsplit=2)[1]
        names = [name for name in columns if name.startswith(prefix)]
        names = list(dict.fromkeys([*names, *common]))
        if len(names) < 25:
            raise RuntimeError(f"grid expert {source}-{grid} has too few features")
        experts[f"{source}_{grid}"] = names
    if len(experts) != 25:
        raise RuntimeError(f"expected 25 grid experts, found {len(experts)}")
    return experts


def _group_balance(surface: pd.DataFrame, training: np.ndarray) -> np.ndarray:
    counts = {
        group_id: int((training & surface["group_id"].eq(group_id).to_numpy()).sum())
        for group_id in CAPACITIES
    }
    total = float(sum(counts.values()))
    factors = {group_id: total / (3.0 * count) for group_id, count in counts.items()}
    return surface.loc[training, "group_id"].map(factors).to_numpy(dtype=float)


def _fit_experts(
    matrix: pd.DataFrame,
    target: pd.Series,
    surface: pd.DataFrame,
    training: np.ndarray,
    apply: np.ndarray,
    experts: dict[str, list[str]],
    iterations: int,
    seed: int,
) -> np.ndarray:
    output = np.empty((int(apply.sum()), len(experts)), dtype=float)
    weights = target.loc[training].clip(lower=0.10).to_numpy(dtype=float)
    weights *= _group_balance(surface, training)
    for position, (name, features) in enumerate(experts.items()):
        model = LGBMRegressor(
            objective="l1",
            n_estimators=iterations,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=55,
            max_bin=255,
            subsample=0.90,
            subsample_freq=1,
            colsample_bytree=0.90,
            reg_alpha=0.15,
            reg_lambda=4.0,
            random_state=seed + position,
            n_jobs=6,
            deterministic=True,
            force_col_wise=True,
            verbosity=-1,
        )
        model.fit(
            matrix.loc[training, features],
            target.loc[training],
            sample_weight=weights,
        )
        output[:, position] = np.clip(
            model.predict(matrix.loc[apply, features]), 0.075, 1.075
        )
        if (position + 1) % 5 == 0 or position + 1 == len(experts):
            print(
                json.dumps(
                    {
                        "stage_seed": seed,
                        "experts_complete": position + 1,
                        "expert_count": len(experts),
                        "last_expert": name,
                    }
                ),
                flush=True,
            )
    return output


def _eligible(actual: np.ndarray) -> np.ndarray:
    return np.isfinite(actual) & (actual >= 0.10)


def _expert_ranking(
    predictions: np.ndarray,
    actual: np.ndarray,
    rows: np.ndarray,
) -> list[int]:
    ranking: list[tuple[float, int]] = []
    for position in range(predictions.shape[1]):
        score = _group_total(actual[rows], predictions[rows, position])
        ranking.append((score["total"], position))
    return [position for _, position in sorted(ranking, reverse=True)]


def _base_prediction(
    recipe: dict[str, object],
    fit_predictions: np.ndarray,
    fit_actual: np.ndarray,
    apply_predictions: np.ndarray,
    fit_rows: np.ndarray,
    expert_names: list[str],
) -> tuple[np.ndarray, dict[str, object]]:
    family = str(recipe["family"])
    if family == "top":
        ranking = _expert_ranking(fit_predictions, fit_actual, fit_rows)
        count = min(int(recipe["count"]), len(ranking))
        selected = ranking[:count]
        reducer = str(recipe["reducer"])
        values = apply_predictions[:, selected]
        prediction = (
            np.median(values, axis=1)
            if reducer == "median"
            else np.mean(values, axis=1)
        )
        fitted = {
            "selected_experts": [expert_names[position] for position in selected],
        }
    elif family == "source":
        source = str(recipe["source"])
        selected = [
            position
            for position, name in enumerate(expert_names)
            if name.startswith(f"{source}_")
        ]
        values = apply_predictions[:, selected]
        reducer = str(recipe["reducer"])
        prediction = (
            np.median(values, axis=1)
            if reducer == "median"
            else np.mean(values, axis=1)
        )
        fitted = {
            "selected_experts": [expert_names[position] for position in selected],
        }
    elif family == "ridge":
        valid = fit_rows & _eligible(fit_actual)
        model = Ridge(
            alpha=float(recipe["alpha"]),
            positive=bool(recipe["positive"]),
        )
        model.fit(
            fit_predictions[valid],
            fit_actual[valid],
            sample_weight=fit_actual[valid],
        )
        prediction = model.predict(apply_predictions)
        fitted = {
            "intercept": float(model.intercept_),
            "coefficient_l1": float(np.abs(model.coef_).sum()),
            "coefficient_min": float(model.coef_.min()),
            "coefficient_max": float(model.coef_.max()),
        }
    else:
        raise ValueError(f"unknown expert recipe family: {family}")
    return np.clip(prediction, 0.075, 1.075), fitted


def _recipes() -> list[dict[str, object]]:
    recipes: list[dict[str, object]] = []
    for count in TOP_K:
        for reducer in ("mean", "median"):
            recipes.append({"family": "top", "count": count, "reducer": reducer})
    for source in ("gfs", "ldaps"):
        for reducer in ("mean", "median"):
            recipes.append(
                {"family": "source", "source": source, "reducer": reducer}
            )
    for alpha in RIDGE_ALPHAS:
        for positive in (False, True):
            recipes.append(
                {
                    "family": "ridge",
                    "alpha": alpha,
                    "positive": positive,
                }
            )
    return recipes


def _select_recipe(
    predictions: np.ndarray,
    actual: np.ndarray,
    fit_rows: np.ndarray,
    select_rows: np.ndarray,
    expert_names: list[str],
) -> tuple[dict[str, object], dict[str, object]]:
    best: tuple[float, dict[str, object], dict[str, float], dict[str, object]] | None = None
    ranking: list[tuple[float, str, dict[str, float]]] = []
    for recipe in _recipes():
        base, fitted = _base_prediction(
            recipe,
            predictions,
            actual,
            predictions[select_rows],
            fit_rows,
            expert_names,
        )
        for scale in SCALES:
            for offset in OFFSETS:
                for snap in (False, True):
                    candidate = np.clip(scale * base + offset, 0.075, 1.075)
                    if snap:
                        candidate = np.round(candidate / 0.0025) * 0.0025
                    score = _group_total(actual[select_rows], candidate)
                    selected_recipe = {
                        **recipe,
                        "scale": scale,
                        "offset": offset,
                        "snap": snap,
                    }
                    tag = json.dumps(selected_recipe, sort_keys=True)
                    ranking.append((score["total"], tag, score))
                    choice = (score["total"], selected_recipe, score, fitted)
                    if best is None or choice[0] > best[0]:
                        best = choice
    assert best is not None
    diagnostics: dict[str, object] = {
        "selection_score": best[2],
        "selection_fit": best[3],
        "top_selection_trials": [
            {"recipe": tag, "score": score}
            for _, tag, score in sorted(ranking, reverse=True)[:10]
        ],
    }
    return best[1], diagnostics


def _apply_recipe(
    recipe: dict[str, object],
    calibration_predictions: np.ndarray,
    calibration_actual: np.ndarray,
    apply_predictions: np.ndarray,
    expert_names: list[str],
) -> tuple[np.ndarray, dict[str, object]]:
    fit_rows = np.ones(len(calibration_actual), dtype=bool)
    base, fitted = _base_prediction(
        recipe,
        calibration_predictions,
        calibration_actual,
        apply_predictions,
        fit_rows,
        expert_names,
    )
    output = np.clip(
        float(recipe["scale"]) * base + float(recipe["offset"]),
        0.075,
        1.075,
    )
    if bool(recipe["snap"]):
        output = np.round(output / 0.0025) * 0.0025
    return output, fitted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument(
        "--fold",
        choices=("dev-2023-Q2", "dev-2023-Q3", "dev-2023-Q4"),
        required=True,
    )
    parser.add_argument("--calibration-days", type=int, default=60)
    parser.add_argument("--iterations", type=int, default=180)
    args = parser.parse_args()
    if not 45 <= args.calibration_days <= 120:
        raise ValueError("calibration-days must be between 45 and 120")
    if not 100 <= args.iterations <= 300:
        raise ValueError("iterations must be between 100 and 300")
    if _sha256(OPEN) != OPEN_SHA or _sha256(BASELINE) != BASELINE_SHA:
        raise RuntimeError("immutable competition input hash mismatch")

    started = time.perf_counter()
    surface, feature_columns = development_surface()
    if surface["forecast_kst_dtm"].ge(DEV_CUTOFF).any():
        raise RuntimeError("lockbox row reached grid-expert runner")
    validation = _validation_mask(surface, args.fold)
    outer_history = _strict_preceding_mask(surface, validation)
    target = surface["actual_kwh"] / surface["group_id"].map(CAPACITIES)
    observed_eligible = surface["actual_kwh"].notna().to_numpy() & target.ge(
        0.10
    ).to_numpy()
    outer_training = outer_history & observed_eligible
    validation_cutoff = pd.Timestamp(
        surface.loc[validation, "data_available_kst_dtm"].min()
    )
    calibration_cutoff = validation_cutoff - np.timedelta64(
        args.calibration_days, "D"
    )
    inner_history = _strict_before(surface, calibration_cutoff)
    inner_training = inner_history & observed_eligible
    calibration = outer_training & ~inner_history
    calibration_times = (
        surface.loc[calibration, "forecast_kst_dtm"].drop_duplicates().sort_values()
    )
    split_time = pd.Timestamp(calibration_times.iloc[len(calibration_times) // 2])
    matrix = surface[feature_columns].astype("float32")
    experts = _expert_features(feature_columns)
    expert_names = list(experts)
    calibration_predictions = _fit_experts(
        matrix,
        target,
        surface,
        inner_training,
        calibration,
        experts,
        args.iterations,
        20261000,
    )
    validation_predictions = _fit_experts(
        matrix,
        target,
        surface,
        outer_training,
        validation,
        experts,
        args.iterations,
        20262000,
    )

    calibration_groups = surface.loc[calibration, "group_id"].to_numpy(dtype=int)
    validation_groups = surface.loc[validation, "group_id"].to_numpy(dtype=int)
    calibration_target = target.loc[calibration].to_numpy(dtype=float)
    final_normalized = np.full(int(validation.sum()), np.nan, dtype=float)
    inner_normalized = np.full(int(calibration.sum()), np.nan, dtype=float)
    recipes: dict[str, dict[str, object]] = {}
    diagnostics: dict[str, object] = {}
    for group_id in CAPACITIES:
        calibration_group = calibration_groups == group_id
        validation_group = validation_groups == group_id
        times = surface.loc[calibration, "forecast_kst_dtm"].to_numpy()
        fit_rows = calibration_group & (times < np.datetime64(split_time))
        select_rows = calibration_group & ~fit_rows
        recipe, group_diagnostics = _select_recipe(
            calibration_predictions,
            calibration_target,
            fit_rows,
            select_rows,
            expert_names,
        )
        inner_values, inner_fit = _apply_recipe(
            recipe,
            calibration_predictions[fit_rows],
            calibration_target[fit_rows],
            calibration_predictions[select_rows],
            expert_names,
        )
        inner_normalized[select_rows] = inner_values
        final_values, final_fit = _apply_recipe(
            recipe,
            calibration_predictions[calibration_group],
            calibration_target[calibration_group],
            validation_predictions[validation_group],
            expert_names,
        )
        final_normalized[validation_group] = final_values
        recipes[str(group_id)] = recipe
        diagnostics[str(group_id)] = {
            **group_diagnostics,
            "fit_rows": int(fit_rows.sum()),
            "selection_rows": int(select_rows.sum()),
            "selection_refit": inner_fit,
            "final_refit": final_fit,
        }
        print(
            json.dumps(
                {
                    "group_id": group_id,
                    "recipe": recipe,
                    "selection_score": group_diagnostics["selection_score"],
                }
            ),
            flush=True,
        )

    if not np.isfinite(final_normalized).all():
        raise RuntimeError("grid-expert outer prediction is incomplete")
    selected_inner = np.isfinite(inner_normalized)
    calibration_base = surface.loc[calibration, BASE_COLUMNS].copy()
    inner_output = calibration_base.loc[selected_inner].copy()
    inner_output["prediction_kwh"] = inner_normalized[selected_inner] * inner_output[
        "group_id"
    ].map(CAPACITIES).to_numpy(dtype=float)
    output = surface.loc[validation, BASE_COLUMNS].copy()
    output["prediction_kwh"] = final_normalized * output["group_id"].map(
        CAPACITIES
    ).to_numpy(dtype=float)
    output["fold_id"] = args.fold
    output["model_id"] = args.candidate_id
    output_path = OUTPUT / f"{args.candidate_id}-{args.fold}.parquet"
    output.to_parquet(output_path, index=False)
    receipt = {
        "schema_version": 1,
        "candidate_id": args.candidate_id,
        "fold_id": args.fold,
        "architecture": "strict_25_grid_power_experts_two_stage_stack",
        "scope": "outer labels excluded; stack family selected on second calibration half",
        "physical_development_cutoff": str(DEV_CUTOFF),
        "calibration_days": args.calibration_days,
        "calibration_cutoff": str(calibration_cutoff),
        "calibration_split_time": str(split_time),
        "iterations": args.iterations,
        "expert_count": len(experts),
        "expert_feature_counts": {name: len(features) for name, features in experts.items()},
        "recipes": recipes,
        "diagnostics": diagnostics,
        "inner_selection_score": _score(inner_output),
        "fold_score": _score(output),
        "runtime_seconds": round(time.perf_counter() - started, 2),
        "prediction_path": str(output_path.relative_to(Path.cwd())),
        "prediction_sha256": _sha256(output_path),
        "new_2024_evaluation": False,
        "lockbox_reopened": False,
        "external_actions": [],
    }
    receipt_path = OUTPUT / f"{args.candidate_id}-{args.fold}.json"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
