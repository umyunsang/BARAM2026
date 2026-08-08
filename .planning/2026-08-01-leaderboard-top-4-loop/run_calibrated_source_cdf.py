"""Strict source probabilities with proper-score calibration and Bayes action."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
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
TEMPERATURES = (0.40, 0.60, 0.80, 1.00, 1.20)
PRIOR_STRENGTHS = (0.00, 0.25, 0.50, 0.75, 1.00)


def _strict_before(surface: pd.DataFrame, cutoff: pd.Timestamp) -> np.ndarray:
    batch_last = surface.groupby("data_available_kst_dtm", sort=False)[
        "forecast_kst_dtm"
    ].transform("max")
    mask = batch_last.lt(cutoff).to_numpy()
    if mask.any() and not surface.loc[mask, "forecast_kst_dtm"].lt(cutoff).all():
        raise RuntimeError("inner source model includes an unavailable target")
    return mask


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


def _group_balance(surface: pd.DataFrame, training: np.ndarray) -> np.ndarray:
    counts = {
        group_id: int((training & surface["group_id"].eq(group_id).to_numpy()).sum())
        for group_id in CAPACITIES
    }
    total = float(sum(counts.values()))
    factors = {group_id: total / (3.0 * count) for group_id, count in counts.items()}
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
        "month",
        "day_of_year",
        "lead_hour",
        "cal__",
    )
    selected = [
        name
        for name in columns
        if source in name.lower() or any(token in name for token in common)
    ]
    if len(selected) < 200:
        raise RuntimeError(f"{source} source contract resolved {len(selected)} columns")
    return selected


def _feature_screen(
    matrix: pd.DataFrame,
    target: pd.Series,
    surface: pd.DataFrame,
    training: np.ndarray,
    candidates: list[str],
    count: int,
) -> list[str]:
    weights = target.loc[training].clip(lower=0.10).to_numpy(dtype=float)
    weights *= _group_balance(surface, training)
    model = LGBMRegressor(
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
        random_state=20260803,
        n_jobs=6,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )
    model.fit(
        matrix.loc[training, candidates],
        target.loc[training],
        sample_weight=weights,
    )
    gains = model.booster_.feature_importance(importance_type="gain")
    order = np.argsort(gains)[::-1]
    selected = [candidates[position] for position in order[:count]]
    if len(selected) != count or len(set(selected)) != count:
        raise RuntimeError("source feature screen contract changed")
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
) -> tuple[dict[str, np.ndarray], dict[str, list[str]]]:
    classes = _target_classes(target)
    weights = target.loc[training].clip(lower=0.10).to_numpy(dtype=float)
    weights *= _group_balance(surface, training)
    candidate_sets = {
        "global": feature_columns,
        "gfs": _source_columns(feature_columns, "gfs"),
        "ldaps": _source_columns(feature_columns, "ldaps"),
    }
    probabilities: dict[str, np.ndarray] = {}
    selected_by_source: dict[str, list[str]] = {}
    for source, candidates in candidate_sets.items():
        selected = _feature_screen(
            matrix,
            target,
            surface,
            training,
            candidates,
            top_features,
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
            random_state=20260803,
            n_jobs=6,
        )
        model.fit(
            matrix.loc[training, selected],
            classes[training],
            sample_weight=weights,
        )
        raw_probability = np.asarray(
            model.predict_proba(matrix.loc[apply, selected]), dtype=float
        )
        learned_classes = np.asarray(model.classes_, dtype=int)
        if (
            raw_probability.shape[1] != len(learned_classes)
            or learned_classes.min(initial=0) < 0
            or learned_classes.max(initial=0) >= CLASS_COUNT
        ):
            raise RuntimeError(f"{source} probability class contract changed")
        probability = np.zeros((int(apply.sum()), CLASS_COUNT), dtype=float)
        probability[:, learned_classes] = raw_probability
        probability /= probability.sum(axis=1, keepdims=True)
        probabilities[source] = probability
        selected_by_source[source] = selected
        print(
            json.dumps(
                {
                    "source": source,
                    "training_rows": int(training.sum()),
                    "apply_rows": int(apply.sum()),
                    "feature_count": len(selected),
                }
            ),
            flush=True,
        )
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


def _prior_ratio(
    probability: np.ndarray,
    classes: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    observed = np.bincount(
        classes,
        weights=weights,
        minlength=CLASS_COUNT,
    ).astype(float)
    observed += 2.0
    observed /= observed.sum()
    predicted = np.average(probability, axis=0, weights=weights)
    predicted = np.clip(predicted, 1e-8, None)
    ratio = observed / predicted
    return np.clip(ratio, 0.05, 20.0)


def _apply_prior(
    probability: np.ndarray,
    ratio: np.ndarray,
    strength: float,
) -> np.ndarray:
    adjusted = probability * ratio[None, :] ** strength
    adjusted /= adjusted.sum(axis=1, keepdims=True)
    return adjusted


def _weighted_log_loss(
    probability: np.ndarray,
    classes: np.ndarray,
    weights: np.ndarray,
) -> float:
    likelihood = probability[np.arange(len(classes)), classes]
    return float(np.average(-np.log(np.clip(likelihood, 1e-12, 1.0)), weights=weights))


def _fit_calibration(
    probabilities: dict[str, np.ndarray],
    surface: pd.DataFrame,
    apply: np.ndarray,
    target: pd.Series,
    fit_mask: np.ndarray,
    select_mask: np.ndarray,
) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    apply_positions = np.flatnonzero(apply)
    local_fit = np.searchsorted(apply_positions, np.flatnonzero(fit_mask))
    local_select = np.searchsorted(apply_positions, np.flatnonzero(select_mask))
    classes = _target_classes(target)
    selections: dict[str, dict[str, object]] = {}
    diagnostics: dict[str, object] = {}
    for group_id in CAPACITIES:
        fit_group_global = fit_mask & surface["group_id"].eq(group_id).to_numpy()
        select_group_global = select_mask & surface["group_id"].eq(group_id).to_numpy()
        fit_group = np.searchsorted(
            apply_positions, np.flatnonzero(fit_group_global)
        )
        select_group = np.searchsorted(
            apply_positions, np.flatnonzero(select_group_global)
        )
        if len(fit_group) < 200 or len(select_group) < 200:
            raise RuntimeError(f"group {group_id} calibration split is too small")
        fit_classes = classes[fit_group_global]
        select_classes = classes[select_group_global]
        fit_weights = target.loc[fit_group_global].to_numpy(dtype=float)
        select_weights = target.loc[select_group_global].to_numpy(dtype=float)
        best: tuple[float, float, float, float, float, np.ndarray] | None = None
        trials: dict[str, float] = {}
        for global_weight, gfs_share in MIXTURES:
            mixed = _mixture(probabilities, global_weight, gfs_share)
            for temperature in TEMPERATURES:
                tempered = _temper(mixed, temperature)
                ratio = _prior_ratio(
                    tempered[fit_group], fit_classes, fit_weights
                )
                for strength in PRIOR_STRENGTHS:
                    calibrated = _apply_prior(
                        tempered[select_group], ratio, strength
                    )
                    loss = _weighted_log_loss(
                        calibrated, select_classes, select_weights
                    )
                    tag = (
                        f"GW{global_weight:g}_GS{gfs_share:g}_"
                        f"T{temperature:g}_P{strength:g}"
                    )
                    trials[tag] = loss
                    choice = (
                        loss,
                        global_weight,
                        gfs_share,
                        temperature,
                        strength,
                        ratio,
                    )
                    if best is None or choice[0] < best[0]:
                        best = choice
        assert best is not None
        mixed = _mixture(probabilities, best[1], best[2])
        tempered = _temper(mixed, best[3])
        full_group_global = apply & surface["group_id"].eq(group_id).to_numpy()
        full_group = np.searchsorted(
            apply_positions, np.flatnonzero(full_group_global)
        )
        ratio = _prior_ratio(
            tempered[full_group],
            classes[full_group_global],
            target.loc[full_group_global].to_numpy(dtype=float),
        )
        selections[str(group_id)] = {
            "global_weight": best[1],
            "gfs_share": best[2],
            "temperature": best[3],
            "prior_strength": best[4],
            "prior_ratio": ratio.tolist(),
            "selection_weighted_log_loss": best[0],
        }
        diagnostics[str(group_id)] = {
            "fit_rows": len(fit_group),
            "selection_rows": len(select_group),
            "best_trials": sorted(trials.items(), key=lambda item: item[1])[:5],
        }
    if len(local_fit) + len(local_select) != int(apply.sum()):
        raise RuntimeError("calibration fit/select masks do not partition apply rows")
    return selections, diagnostics


def _apply_calibration(
    probabilities: dict[str, np.ndarray],
    groups: np.ndarray,
    selections: dict[str, dict[str, object]],
) -> np.ndarray:
    output = np.empty_like(probabilities["global"], dtype=float)
    for group_id in CAPACITIES:
        group = groups == group_id
        selection = selections[str(group_id)]
        mixed = _mixture(
            {name: values[group] for name, values in probabilities.items()},
            float(selection["global_weight"]),
            float(selection["gfs_share"]),
        )
        tempered = _temper(mixed, float(selection["temperature"]))
        output[group] = _apply_prior(
            tempered,
            np.asarray(selection["prior_ratio"], dtype=float),
            float(selection["prior_strength"]),
        )
    return output


def _bayes_action(
    probability: np.ndarray,
    groups: np.ndarray,
    mean_generation: dict[int, float],
) -> np.ndarray:
    centers = _centers()
    actions = np.arange(0.075, 1.0751, 0.0025)
    error = np.abs(actions[:, None] - centers[None, :])
    units = np.select([error <= 0.06, error <= 0.08], [4.0, 3.0], default=0.0)
    chosen = np.empty(len(probability), dtype=float)
    for group_id in CAPACITIES:
        group = groups == group_id
        utility = -(probability[group] @ error.T) + (
            probability[group] @ (centers[None, :] * units).T
        ) / (4.0 * mean_generation[group_id])
        chosen[group] = actions[np.argmax(utility, axis=1)]
    return chosen


def _prediction_frame(
    surface: pd.DataFrame,
    apply: np.ndarray,
    normalized: np.ndarray,
) -> pd.DataFrame:
    base = surface.loc[apply, BASE_COLUMNS].copy()
    capacity = base["group_id"].map(CAPACITIES).to_numpy(dtype=float)
    base["prediction_kwh"] = normalized * capacity
    return base


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument(
        "--fold",
        choices=("dev-2023-Q2", "dev-2023-Q3", "dev-2023-Q4"),
        required=True,
    )
    parser.add_argument("--calibration-days", type=int, default=90)
    parser.add_argument("--top-features", type=int, default=160)
    parser.add_argument("--iterations", type=int, default=140)
    args = parser.parse_args()
    if not 60 <= args.calibration_days <= 120:
        raise ValueError("calibration-days must be between 60 and 120")
    if not 80 <= args.top_features <= 240:
        raise ValueError("top-features must be between 80 and 240")
    if not 80 <= args.iterations <= 240:
        raise ValueError("iterations must be between 80 and 240")
    if _sha256(OPEN) != OPEN_SHA or _sha256(BASELINE) != BASELINE_SHA:
        raise RuntimeError("immutable competition input hash mismatch")

    started = time.perf_counter()
    surface, feature_columns = development_surface()
    if surface["forecast_kst_dtm"].ge(DEV_CUTOFF).any():
        raise RuntimeError("lockbox row reached source calibration")
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
    calibration_fit = calibration & surface["forecast_kst_dtm"].lt(
        split_time
    ).to_numpy()
    calibration_select = calibration & ~calibration_fit
    for group_id in CAPACITIES:
        group = surface["group_id"].eq(group_id).to_numpy()
        for name, mask in (
            ("inner training", inner_training),
            ("calibration fit", calibration_fit),
            ("calibration selection", calibration_select),
        ):
            if int((mask & group).sum()) < 200:
                raise RuntimeError(f"group {group_id} {name} is too small")

    matrix = surface[feature_columns].astype("float32")
    calibration_probabilities, inner_features = _fit_sources(
        matrix,
        target,
        surface,
        inner_training,
        calibration,
        feature_columns,
        args.top_features,
        args.iterations,
    )
    selections, calibration_diagnostics = _fit_calibration(
        calibration_probabilities,
        surface,
        calibration,
        target,
        calibration_fit,
        calibration_select,
    )
    calibration_groups = surface.loc[calibration, "group_id"].to_numpy(dtype=int)
    calibrated_inner = _apply_calibration(
        calibration_probabilities, calibration_groups, selections
    )
    inner_means = {
        group_id: float(
            target.loc[
                inner_training & surface["group_id"].eq(group_id).to_numpy()
            ].mean()
        )
        for group_id in CAPACITIES
    }
    inner_action = _bayes_action(calibrated_inner, calibration_groups, inner_means)
    inner_output = _prediction_frame(surface, calibration, inner_action)

    validation_probabilities, final_features = _fit_sources(
        matrix,
        target,
        surface,
        outer_training,
        validation,
        feature_columns,
        args.top_features,
        args.iterations,
    )
    validation_groups = surface.loc[validation, "group_id"].to_numpy(dtype=int)
    calibrated = _apply_calibration(
        validation_probabilities, validation_groups, selections
    )
    outer_means = {
        group_id: float(
            target.loc[
                outer_training & surface["group_id"].eq(group_id).to_numpy()
            ].mean()
        )
        for group_id in CAPACITIES
    }
    action = _bayes_action(calibrated, validation_groups, outer_means)
    output = _prediction_frame(surface, validation, action)
    output["fold_id"] = args.fold
    output["model_id"] = args.candidate_id
    output_path = OUTPUT / f"{args.candidate_id}-{args.fold}.parquet"
    output.to_parquet(output_path, index=False)
    receipt = {
        "schema_version": 1,
        "candidate_id": args.candidate_id,
        "fold_id": args.fold,
        "architecture": "strict_calibrated_gfs_ldaps_global_cdf_bayes_action",
        "scope": "outer labels excluded from models, calibration, source mix, and policy",
        "physical_development_cutoff": str(DEV_CUTOFF),
        "calibration_days": args.calibration_days,
        "calibration_split_time": str(split_time),
        "top_features": args.top_features,
        "iterations": args.iterations,
        "class_width": CLASS_WIDTH,
        "class_count": CLASS_COUNT,
        "calibration_selections": selections,
        "calibration_diagnostics": calibration_diagnostics,
        "inner_calibrated_score": _score(inner_output),
        "inner_selected_features": inner_features,
        "final_selected_features": final_features,
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
