"""Strict calendar-safe source-rank distribution with seasonal policy selection.

This experiment removes raw calendar fields that tree models cannot extrapolate
through the Q3 boundary while retaining cyclic calendar encodings.  Source
mixture, temperature, and official-utility gamma are selected without outer-Q3
labels: groups 1/2 use Q3 2022, while group 3 (which has no 2022 labels) uses a
preceding May-June 2023 window.  Every fit uses only complete issuance batches
observable before its application window.
"""

from __future__ import annotations

import argparse
import gc
import json
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
from strict_dev_surface import DEV_CUTOFF, development_surface
from xgboost import XGBClassifier

BASE_COLUMNS = ["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh"]
RAW_CALENDAR = {"operating_year", "month", "day_of_year"}
CLASS_WIDTH = 0.02
CLASS_COUNT = 49
MIXTURES = (
    (1.00, 0.50),
    (0.75, 0.25),
    (0.75, 0.50),
    (0.75, 0.75),
    (0.50, 0.25),
    (0.50, 0.50),
    (0.50, 0.75),
    (0.00, 1.00),
    (0.00, 0.00),
)
TEMPERATURES = (0.50, 0.75, 1.00, 1.25)
UTILITY_GAMMAS = (0.00, 0.35, 0.75, 1.00, 1.25, 1.50, 2.00)
ACTIONS = np.arange(0.075, 1.0751, 0.0025)


def _target_classes(target: pd.Series) -> np.ndarray:
    values = target.to_numpy(dtype=float)
    values = np.where(np.isfinite(values), values, 0.10)
    classes = np.floor(
        (np.clip(values, 0.10, 1.074999) - 0.10) / CLASS_WIDTH
    ).astype(int)
    return np.clip(classes, 0, CLASS_COUNT - 1)


def _centers() -> np.ndarray:
    lower = 0.10 + CLASS_WIDTH * np.arange(CLASS_COUNT, dtype=float)
    upper = np.minimum(lower + CLASS_WIDTH, 1.075)
    return (lower + upper) / 2.0


def _active_group_balance(
    surface: pd.DataFrame,
    training: np.ndarray,
) -> np.ndarray:
    counts = {
        group_id: int(
            (training & surface["group_id"].eq(group_id).to_numpy()).sum()
        )
        for group_id in CAPACITIES
    }
    active = {group_id: count for group_id, count in counts.items() if count > 0}
    if not active:
        raise RuntimeError("seasonal source model has no active training group")
    total = float(sum(active.values()))
    factors = {
        group_id: total / (len(active) * count)
        for group_id, count in active.items()
    }
    return surface.loc[training, "group_id"].map(factors).to_numpy(dtype=float)


def _source_columns(columns: list[str], source: str) -> list[str]:
    common = (
        "group_id",
        "group_",
        "capacity",
        "turbine_count",
        "rotor",
        "hub_height",
        "latitude",
        "longitude",
        "hour",
        "lead_hour",
        "cal__",
    )
    selected = [
        name
        for name in columns
        if source in name.lower() or any(token in name for token in common)
    ]
    if len(selected) < 180:
        raise RuntimeError(
            f"{source} calendar-safe source contract resolved {len(selected)} columns"
        )
    return selected


def _feature_screen(
    matrix: pd.DataFrame,
    target: pd.Series,
    surface: pd.DataFrame,
    training: np.ndarray,
    candidates: list[str],
    count: int,
    seed: int,
) -> list[str]:
    weights = target.loc[training].clip(lower=0.10).to_numpy(dtype=float)
    weights *= _active_group_balance(surface, training)
    screen = LGBMRegressor(
        objective="l1",
        n_estimators=180,
        learning_rate=0.035,
        num_leaves=31,
        min_child_samples=80,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.75,
        reg_alpha=0.2,
        reg_lambda=4.0,
        random_state=seed,
        n_jobs=6,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )
    screen.fit(
        matrix.loc[training, candidates],
        target.loc[training],
        sample_weight=weights,
    )
    gains = screen.booster_.feature_importance(importance_type="gain")
    order = np.argsort(gains)[::-1][:count]
    selected = [candidates[position] for position in order]
    if len(selected) != count or len(set(selected)) != count:
        raise RuntimeError("seasonal source feature screen contract changed")
    if RAW_CALENDAR.intersection(selected):
        raise RuntimeError("raw calendar field escaped the feature exclusion")
    del screen
    gc.collect()
    return selected


def _fit_sources(
    matrix: pd.DataFrame,
    target: pd.Series,
    surface: pd.DataFrame,
    training: np.ndarray,
    apply: np.ndarray,
    feature_columns: list[str],
    top_features: int,
    iterations: int,
    seed: int,
    lane: str,
) -> tuple[dict[str, np.ndarray], dict[str, list[str]]]:
    classes = _target_classes(target)
    weights = target.loc[training].clip(lower=0.10).to_numpy(dtype=float)
    weights *= _active_group_balance(surface, training)
    candidate_sets = {
        "global": feature_columns,
        "gfs": _source_columns(feature_columns, "gfs"),
        "ldaps": _source_columns(feature_columns, "ldaps"),
    }
    probabilities: dict[str, np.ndarray] = {}
    selected_by_source: dict[str, list[str]] = {}
    for source_index, (source, candidates) in enumerate(candidate_sets.items()):
        selected = _feature_screen(
            matrix,
            target,
            surface,
            training,
            candidates,
            top_features,
            seed + source_index,
        )
        model = XGBClassifier(
            objective="multi:softprob",
            num_class=CLASS_COUNT,
            n_estimators=iterations,
            learning_rate=0.03,
            max_depth=5,
            min_child_weight=20.0,
            subsample=0.9,
            colsample_bytree=0.8,
            reg_alpha=0.2,
            reg_lambda=5.0,
            max_bin=256,
            tree_method="hist",
            random_state=seed + 20 + source_index,
            n_jobs=6,
        )
        model.fit(
            matrix.loc[training, selected],
            classes[training],
            sample_weight=weights,
        )
        raw = np.asarray(
            model.predict_proba(matrix.loc[apply, selected]), dtype=float
        )
        learned = np.asarray(model.classes_, dtype=int)
        if (
            raw.shape[1] != len(learned)
            or learned.min(initial=0) < 0
            or learned.max(initial=0) >= CLASS_COUNT
        ):
            raise RuntimeError(f"{source} probability class contract changed")
        probability = np.zeros((int(apply.sum()), CLASS_COUNT), dtype=float)
        probability[:, learned] = raw
        row_sum = probability.sum(axis=1, keepdims=True)
        if not np.isfinite(row_sum).all() or np.any(row_sum <= 0.0):
            raise RuntimeError(f"{source} produced invalid probability mass")
        probability /= row_sum
        probabilities[source] = probability
        selected_by_source[source] = selected
        print(
            json.dumps(
                {
                    "lane": lane,
                    "source": source,
                    "training_rows": int(training.sum()),
                    "apply_rows": int(apply.sum()),
                    "feature_count": len(selected),
                }
            ),
            flush=True,
        )
        del model, raw
        gc.collect()
    return probabilities, selected_by_source


def _mixture(
    probabilities: dict[str, np.ndarray],
    global_weight: float,
    gfs_share: float,
) -> np.ndarray:
    source = (
        gfs_share * probabilities["gfs"]
        + (1.0 - gfs_share) * probabilities["ldaps"]
    )
    mixed = global_weight * probabilities["global"] + (1.0 - global_weight) * source
    mixed /= mixed.sum(axis=1, keepdims=True)
    return mixed


def _temper(probability: np.ndarray, temperature: float) -> np.ndarray:
    calibrated = np.clip(probability, 1e-12, 1.0) ** (1.0 / temperature)
    calibrated /= calibrated.sum(axis=1, keepdims=True)
    return calibrated


def _action_components(
    probability: np.ndarray,
    mean_generation: float,
) -> tuple[np.ndarray, np.ndarray]:
    centers = _centers()
    error = np.abs(ACTIONS[:, None] - centers[None, :])
    units = np.select([error <= 0.06, error <= 0.08], [4.0, 3.0], default=0.0)
    expected_error = probability @ error.T
    expected_revenue = (
        probability @ (centers[None, :] * units).T
    ) / (4.0 * mean_generation)
    return expected_error, expected_revenue


def _select_group_policy(
    probabilities: dict[str, np.ndarray],
    base: pd.DataFrame,
    group_id: int,
    mean_generation: float,
) -> tuple[dict[str, float], dict[str, object]]:
    capacity = CAPACITIES[group_id]
    actual = base["actual_kwh"].to_numpy(dtype=float) / capacity
    best: tuple[float, float, float, float, float] | None = None
    trials: list[dict[str, float]] = []
    for global_weight, gfs_share in MIXTURES:
        mixed = _mixture(probabilities, global_weight, gfs_share)
        for temperature in TEMPERATURES:
            tempered = _temper(mixed, temperature)
            expected_error, expected_revenue = _action_components(
                tempered, mean_generation
            )
            for gamma in UTILITY_GAMMAS:
                action = ACTIONS[
                    np.argmin(expected_error - gamma * expected_revenue, axis=1)
                ]
                score = _group_total(actual, action)
                trial = {
                    "global_weight": global_weight,
                    "gfs_share": gfs_share,
                    "temperature": temperature,
                    "utility_gamma": gamma,
                    **score,
                }
                trials.append(trial)
                choice = (
                    score["total"],
                    global_weight,
                    gfs_share,
                    temperature,
                    gamma,
                )
                if best is None or choice[0] > best[0] + 1e-12:
                    best = choice
    if best is None:
        raise RuntimeError(f"group {group_id} policy selection produced no trial")
    selected = {
        "global_weight": best[1],
        "gfs_share": best[2],
        "temperature": best[3],
        "utility_gamma": best[4],
    }
    ranked = sorted(trials, key=lambda item: item["total"], reverse=True)
    return selected, {
        "group_id": group_id,
        "calibration_rows": len(base),
        "selected_score": ranked[0],
        "top_trials": ranked[:10],
    }


def _apply_group_policy(
    probabilities: dict[str, np.ndarray],
    selection: dict[str, float],
    mean_generation: float,
) -> np.ndarray:
    mixed = _mixture(
        probabilities,
        float(selection["global_weight"]),
        float(selection["gfs_share"]),
    )
    calibrated = _temper(mixed, float(selection["temperature"]))
    expected_error, expected_revenue = _action_components(
        calibrated, mean_generation
    )
    gamma = float(selection["utility_gamma"])
    return ACTIONS[np.argmin(expected_error - gamma * expected_revenue, axis=1)]


def _strict_training(
    surface: pd.DataFrame,
    application: np.ndarray,
    observed_eligible: np.ndarray,
    lane: str,
) -> np.ndarray:
    history = _strict_preceding_mask(surface, application)
    training = history & observed_eligible
    cutoff = pd.Timestamp(
        surface.loc[application, "data_available_kst_dtm"].min()
    )
    if not surface.loc[training, "forecast_kst_dtm"].lt(cutoff).all():
        raise RuntimeError(f"{lane} training crossed its availability cutoff")
    return training


def _group_score(
    output: pd.DataFrame,
    group_id: int,
) -> dict[str, float]:
    group = output.loc[output["group_id"].eq(group_id)]
    capacity = CAPACITIES[group_id]
    return _group_total(
        group["actual_kwh"].to_numpy(dtype=float) / capacity,
        group["prediction_kwh"].to_numpy(dtype=float) / capacity,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--fold", choices=("dev-2023-Q3",), required=True)
    parser.add_argument("--top-features", type=int, default=160)
    parser.add_argument("--iterations", type=int, default=150)
    args = parser.parse_args()
    if not 100 <= args.top_features <= 240:
        raise ValueError("top-features must be between 100 and 240")
    if not 100 <= args.iterations <= 240:
        raise ValueError("iterations must be between 100 and 240")
    if _sha256(OPEN) != OPEN_SHA or _sha256(BASELINE) != BASELINE_SHA:
        raise RuntimeError("immutable competition input hash mismatch")

    started = time.perf_counter()
    surface, all_features = development_surface()
    if surface["forecast_kst_dtm"].ge(DEV_CUTOFF).any():
        raise RuntimeError("lockbox row reached seasonal source-rank runner")
    feature_columns = [name for name in all_features if name not in RAW_CALENDAR]
    if len(feature_columns) != len(all_features) - len(RAW_CALENDAR):
        raise RuntimeError("raw calendar exclusion contract changed")
    matrix = surface[feature_columns].astype("float32")
    target = surface["actual_kwh"] / surface["group_id"].map(CAPACITIES)
    observed_eligible = surface["actual_kwh"].notna().to_numpy() & target.ge(
        0.10
    ).to_numpy()

    forecast_time = surface["forecast_kst_dtm"]
    group_values = surface["group_id"].to_numpy(dtype=int)
    seasonal_application = (
        forecast_time.ge(pd.Timestamp("2022-07-01"))
        & forecast_time.lt(pd.Timestamp("2022-10-01"))
        & surface["group_id"].isin((1, 2))
    ).to_numpy()
    group3_application = (
        forecast_time.ge(pd.Timestamp("2023-05-01"))
        & forecast_time.lt(pd.Timestamp("2023-07-01"))
        & surface["group_id"].eq(3)
    ).to_numpy()
    outer_application = _validation_mask(surface, args.fold)
    if set(group_values[outer_application]) != {1, 2, 3}:
        raise RuntimeError("outer Q3 group contract changed")
    if int(seasonal_application.sum()) < 4000 or int(group3_application.sum()) < 1200:
        raise RuntimeError("seasonal calibration window is unexpectedly small")

    seasonal_training = _strict_training(
        surface,
        seasonal_application,
        observed_eligible,
        "Q3-2022",
    )
    group3_training = _strict_training(
        surface,
        group3_application,
        observed_eligible,
        "recent-group3",
    )
    outer_training = _strict_training(
        surface,
        outer_application,
        observed_eligible,
        "outer-Q3",
    )

    seasonal_probability, seasonal_features = _fit_sources(
        matrix,
        target,
        surface,
        seasonal_training,
        seasonal_application,
        feature_columns,
        args.top_features,
        args.iterations,
        20260831,
        "Q3-2022-groups12",
    )
    group3_probability, group3_features = _fit_sources(
        matrix,
        target,
        surface,
        group3_training,
        group3_application,
        feature_columns,
        args.top_features,
        args.iterations,
        20260841,
        "MayJune-2023-group3",
    )

    selections: dict[str, dict[str, float]] = {}
    diagnostics: dict[str, object] = {}
    seasonal_base = surface.loc[seasonal_application, BASE_COLUMNS].copy()
    seasonal_groups = seasonal_base["group_id"].to_numpy(dtype=int)
    for group_id in (1, 2):
        local = seasonal_groups == group_id
        local_probability = {
            name: values[local] for name, values in seasonal_probability.items()
        }
        training_group = seasonal_training & surface["group_id"].eq(group_id).to_numpy()
        mean_generation = float(target.loc[training_group].mean())
        selection, diagnostic = _select_group_policy(
            local_probability,
            seasonal_base.loc[local],
            group_id,
            mean_generation,
        )
        selections[str(group_id)] = selection
        diagnostics[str(group_id)] = diagnostic

    group3_base = surface.loc[group3_application, BASE_COLUMNS].copy()
    group3_mean = float(
        target.loc[
            group3_training & surface["group_id"].eq(3).to_numpy()
        ].mean()
    )
    selection, diagnostic = _select_group_policy(
        group3_probability,
        group3_base,
        3,
        group3_mean,
    )
    selections["3"] = selection
    diagnostics["3"] = diagnostic

    outer_probability, outer_features = _fit_sources(
        matrix,
        target,
        surface,
        outer_training,
        outer_application,
        feature_columns,
        args.top_features,
        args.iterations,
        20260851,
        "outer-Q3-2023",
    )
    output = surface.loc[outer_application, BASE_COLUMNS].copy()
    output_groups = output["group_id"].to_numpy(dtype=int)
    normalized_prediction = np.empty(len(output), dtype=float)
    for group_id in CAPACITIES:
        local = output_groups == group_id
        local_probability = {
            name: values[local] for name, values in outer_probability.items()
        }
        training_group = outer_training & surface["group_id"].eq(group_id).to_numpy()
        mean_generation = float(target.loc[training_group].mean())
        normalized_prediction[local] = _apply_group_policy(
            local_probability,
            selections[str(group_id)],
            mean_generation,
        )
    output["prediction_kwh"] = normalized_prediction * output["group_id"].map(
        CAPACITIES
    ).to_numpy(dtype=float)
    output["fold_id"] = args.fold
    output["model_id"] = args.candidate_id
    output_path = OUTPUT / f"{args.candidate_id}-{args.fold}.parquet"
    output.to_parquet(output_path, index=False)

    fold_score = _score(output)
    group_scores = {
        str(group_id): _group_score(output, group_id) for group_id in CAPACITIES
    }
    receipt = {
        "schema_version": 1,
        "candidate_id": args.candidate_id,
        "fold_id": args.fold,
        "architecture": "strict_calendar_safe_seasonal_source_rank_bayes_action",
        "scope": (
            "outer labels excluded; group policies selected on preceding "
            "seasonal/recent windows"
        ),
        "physical_development_cutoff": str(DEV_CUTOFF),
        "raw_calendar_excluded": sorted(RAW_CALENDAR),
        "cyclic_calendar_retained": ["cal__doy_sin", "cal__doy_cos"],
        "feature_count": len(feature_columns),
        "top_features": args.top_features,
        "iterations": args.iterations,
        "class_width": CLASS_WIDTH,
        "class_count": CLASS_COUNT,
        "seasonal_calibration_window": ["2022-07-01", "2022-10-01"],
        "group3_calibration_window": ["2023-05-01", "2023-07-01"],
        "training_rows": {
            "seasonal_groups12": int(seasonal_training.sum()),
            "recent_group3": int(group3_training.sum()),
            "outer": int(outer_training.sum()),
        },
        "application_rows": {
            "seasonal_groups12": int(seasonal_application.sum()),
            "recent_group3": int(group3_application.sum()),
            "outer": int(outer_application.sum()),
        },
        "policy_selections": selections,
        "policy_selection_diagnostics": diagnostics,
        "seasonal_selected_features": seasonal_features,
        "group3_selected_features": group3_features,
        "outer_selected_features": outer_features,
        "fold_score": fold_score,
        "group_scores": group_scores,
        "runtime_seconds": round(time.perf_counter() - started, 2),
        "prediction_path": str(output_path.relative_to(Path.cwd())),
        "prediction_sha256": _sha256(output_path),
        "new_2024_evaluation": False,
        "lockbox_reopened": False,
        "external_actions": [],
    }
    receipt_path = OUTPUT / f"{args.candidate_id}-{args.fold}.json"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "candidate_id": args.candidate_id,
                "fold_score": fold_score,
                "group_scores": group_scores,
                "policy_selections": selections,
                "runtime_seconds": receipt["runtime_seconds"],
                "prediction_sha256": receipt["prediction_sha256"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
