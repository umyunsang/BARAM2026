"""Strict shared classifier with uncertainty-aware group-3 pseudo targets."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from run_pseudo_group3_classifier import _compact_mapper_params, _mapper_features
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
QUANTILES = (0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90)
QUANTILE_MASS = np.asarray((0.15, 0.125, 0.175, 0.10, 0.175, 0.125, 0.15))
CONFIGS = (
    ("observed_only", "none", 0.0),
    ("hard_w010", "hard", 0.10),
    ("hard_w025", "hard", 0.25),
    ("soft_w010", "soft", 0.10),
    ("soft_w025", "soft", 0.25),
    ("soft_w050", "soft", 0.50),
)
TEMPERATURES = (0.40, 0.60, 0.80, 1.00)
GAMMAS = (0.75, 1.00, 1.25)


def _strict_before(surface: pd.DataFrame, cutoff: pd.Timestamp) -> np.ndarray:
    batch_last = surface.groupby("data_available_kst_dtm", sort=False)[
        "forecast_kst_dtm"
    ].transform("max")
    mask = batch_last.lt(cutoff).to_numpy()
    if mask.any() and not surface.loc[mask, "forecast_kst_dtm"].lt(cutoff).all():
        raise RuntimeError("inner history includes an unavailable forecast target")
    return mask


def _pseudo_quantiles(
    surface: pd.DataFrame,
    history: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    labels = surface.loc[
        history, ["forecast_kst_dtm", "group_id", "actual_kwh"]
    ].copy()
    labels["normalized_target"] = labels["actual_kwh"] / labels["group_id"].map(
        CAPACITIES
    )
    wide = labels.pivot(
        index="forecast_kst_dtm", columns="group_id", values="normalized_target"
    )
    features = _mapper_features(wide)
    core = wide[[1, 2]].notna().all(axis=1)
    observed = core & wide[3].notna()
    missing = core & wide[3].isna()
    if int(observed.sum()) < 1000 or int(missing.sum()) < 1000:
        raise RuntimeError("group-3 pseudo mapper lacks chronological support")

    predictions = []
    for quantile in QUANTILES:
        params = {
            **_compact_mapper_params(),
            "objective": "quantile",
            "alpha": quantile,
            "n_estimators": 240,
        }
        model = LGBMRegressor(**params)
        model.fit(
            features.loc[observed],
            wide.loc[observed, 3],
            sample_weight=wide.loc[observed, 3].clip(lower=0.10),
        )
        predictions.append(
            np.clip(model.predict(features.loc[missing]), 0.10, 1.075)
        )
    quantile_values = np.maximum.accumulate(np.column_stack(predictions), axis=1)
    times = features.index[missing]
    time_to_position = {value: position for position, value in enumerate(times)}
    pseudo_rows = (
        history
        & surface["group_id"].eq(3).to_numpy()
        & surface["actual_kwh"].isna().to_numpy()
        & surface["forecast_kst_dtm"].isin(times).to_numpy()
    )
    positions = surface.loc[pseudo_rows, "forecast_kst_dtm"].map(
        time_to_position
    )
    if positions.isna().any():
        raise RuntimeError("pseudo timestamp mapping is incomplete")
    row_positions = np.flatnonzero(pseudo_rows)
    row_quantiles = quantile_values[positions.to_numpy(dtype=int)]
    diagnostics = {
        "mapper_observed_rows": int(observed.sum()),
        "pseudo_rows": len(row_positions),
        "pseudo_mean_q50": float(row_quantiles[:, 3].mean()),
        "pseudo_mean_width_10_90": float(
            np.mean(row_quantiles[:, -1] - row_quantiles[:, 0])
        ),
        "pseudo_q50_min": float(row_quantiles[:, 3].min()),
        "pseudo_q50_max": float(row_quantiles[:, 3].max()),
    }
    return row_positions, row_quantiles, diagnostics


def _class_contract(
    target: pd.Series,
    hard: np.ndarray,
) -> tuple[pd.Series, list[int], np.ndarray]:
    raw = np.floor(
        (target.clip(0.10, 1.074999) - 0.10) / CLASS_WIDTH
    ).astype("Int64")
    active = sorted(raw.loc[hard].dropna().astype(int).unique())
    mapping = {bin_id: class_id for class_id, bin_id in enumerate(active)}
    classes = raw.map(mapping).astype("Int64")
    centers = np.asarray(
        [
            float(target.loc[hard & classes.eq(class_id).to_numpy()].mean())
            for class_id in range(len(active))
        ]
    )
    return classes, active, centers


def _group_balance(surface: pd.DataFrame, hard: np.ndarray) -> dict[int, float]:
    counts = {
        group_id: int((hard & surface["group_id"].eq(group_id).to_numpy()).sum())
        for group_id in CAPACITIES
    }
    total = float(sum(counts.values()))
    return {group_id: total / (3.0 * count) for group_id, count in counts.items()}


def _feature_screen(
    matrix: pd.DataFrame,
    surface: pd.DataFrame,
    target: pd.Series,
    hard: np.ndarray,
    count: int,
) -> list[str]:
    balance = _group_balance(surface, hard)
    weights = target.loc[hard].clip(lower=0.10).to_numpy(dtype=float)
    weights *= surface.loc[hard, "group_id"].map(balance).to_numpy(dtype=float)
    model = LGBMRegressor(
        objective="l1",
        n_estimators=220,
        learning_rate=0.03,
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
    model.fit(matrix.loc[hard], target.loc[hard], sample_weight=weights)
    gains = model.booster_.feature_importance(importance_type="gain")
    order = np.argsort(gains)[::-1]
    selected = [matrix.columns[position] for position in order[:count]]
    if len(selected) != count or len(set(selected)) != count:
        raise RuntimeError("soft-pseudo feature screen contract changed")
    return selected


def _expanded_training(
    matrix: pd.DataFrame,
    surface: pd.DataFrame,
    target: pd.Series,
    hard: np.ndarray,
    selected: list[str],
    active_bins: list[int],
    pseudo_rows: np.ndarray,
    pseudo_quantiles: np.ndarray,
    mode: str,
    pseudo_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    bin_to_class = {bin_id: class_id for class_id, bin_id in enumerate(active_bins)}
    hard_values = target.loc[hard].to_numpy(dtype=float)
    hard_bins = np.floor(
        (np.clip(hard_values, 0.10, 1.074999) - 0.10) / CLASS_WIDTH
    ).astype(int)
    hard_classes = np.asarray([bin_to_class[value] for value in hard_bins], dtype=int)
    balance = _group_balance(surface, hard)
    hard_weights = np.clip(hard_values, 0.10, None)
    hard_weights *= surface.loc[hard, "group_id"].map(balance).to_numpy(dtype=float)
    x_parts = [matrix.loc[hard, selected].to_numpy(dtype="float32")]
    y_parts = [hard_classes]
    w_parts = [hard_weights]
    expanded_pseudo_rows = 0
    if mode != "none" and pseudo_weight > 0:
        if mode == "hard":
            pseudo_values = pseudo_quantiles[:, [3]]
            masses = np.ones(1, dtype=float)
        elif mode == "soft":
            pseudo_values = pseudo_quantiles
            masses = QUANTILE_MASS
        else:
            raise ValueError(f"unknown pseudo mode: {mode}")
        pseudo_matrix = matrix.iloc[pseudo_rows][selected].to_numpy(dtype="float32")
        repeated = np.repeat(pseudo_matrix, pseudo_values.shape[1], axis=0)
        flat_values = pseudo_values.reshape(-1)
        flat_bins = np.floor(
            (np.clip(flat_values, 0.10, 1.074999) - 0.10) / CLASS_WIDTH
        ).astype(int)
        minimum, maximum = min(active_bins), max(active_bins)
        flat_bins = np.clip(flat_bins, minimum, maximum)
        pseudo_classes = np.asarray(
            [bin_to_class[value] for value in flat_bins], dtype=int
        )
        per_row_mass = np.tile(masses / masses.sum(), len(pseudo_rows))
        pseudo_weights = pseudo_weight * np.clip(flat_values, 0.10, None)
        pseudo_weights *= per_row_mass * balance[3]
        x_parts.append(repeated)
        y_parts.append(pseudo_classes)
        w_parts.append(pseudo_weights)
        expanded_pseudo_rows = len(repeated)
    return (
        np.concatenate(x_parts, axis=0),
        np.concatenate(y_parts),
        np.concatenate(w_parts),
        {
            "hard_rows": int(hard.sum()),
            "pseudo_source_rows": len(pseudo_rows) if mode != "none" else 0,
            "expanded_pseudo_rows": expanded_pseudo_rows,
        },
    )


def _fit_probability(
    matrix: pd.DataFrame,
    surface: pd.DataFrame,
    target: pd.Series,
    hard: np.ndarray,
    apply: np.ndarray,
    selected: list[str],
    pseudo_rows: np.ndarray,
    pseudo_quantiles: np.ndarray,
    mode: str,
    pseudo_weight: float,
    iterations: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    _, active_bins, centers = _class_contract(target, hard)
    x_train, y_train, weights, expansion = _expanded_training(
        matrix,
        surface,
        target,
        hard,
        selected,
        active_bins,
        pseudo_rows,
        pseudo_quantiles,
        mode,
        pseudo_weight,
    )
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=len(active_bins),
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
    model.fit(x_train, y_train, sample_weight=weights)
    probability = model.predict_proba(
        matrix.loc[apply, selected].to_numpy(dtype="float32")
    )
    probability = np.asarray(probability, dtype=float)
    probability /= probability.sum(axis=1, keepdims=True)
    return probability, centers, {
        **expansion,
        "class_count": len(active_bins),
        "training_weight_sum": float(weights.sum()),
    }


def _policy_values(
    probability: np.ndarray,
    centers: np.ndarray,
    groups: np.ndarray,
    means: dict[int, float],
) -> dict[str, np.ndarray]:
    actions = np.arange(0.075, 1.0751, 0.0025)
    error = np.abs(actions[:, None] - centers[None, :])
    units = np.select([error <= 0.06, error <= 0.08], [4.0, 3.0], default=0.0)
    policies: dict[str, np.ndarray] = {
        "MODE": centers[np.argmax(probability, axis=1)],
        "MEAN": probability @ centers,
    }
    for temperature in TEMPERATURES:
        calibrated = probability ** (1.0 / temperature)
        calibrated /= calibrated.sum(axis=1, keepdims=True)
        for gamma in GAMMAS:
            chosen = np.empty(len(probability), dtype=float)
            for group_id in CAPACITIES:
                group = groups == group_id
                utility = -(calibrated[group] @ error.T) + gamma * (
                    calibrated[group] @ (centers[None, :] * units).T
                ) / (4.0 * means[group_id])
                chosen[group] = actions[np.argmax(utility, axis=1)]
            policies[f"T{temperature:g}_G{gamma:g}"] = chosen
    return policies


def _group_score(
    base: pd.DataFrame,
    normalized: np.ndarray,
    group_id: int,
) -> dict[str, float]:
    group = base["group_id"].eq(group_id).to_numpy()
    actual = (
        base.loc[group, "actual_kwh"].to_numpy(dtype=float)
        / CAPACITIES[group_id]
    )
    prediction = normalized[group]
    eligible = np.isfinite(actual) & np.isfinite(prediction) & (actual >= 0.10)
    actual = actual[eligible]
    prediction = prediction[eligible]
    error = np.abs(prediction - actual)
    units = np.select([error <= 0.06, error <= 0.08], [4.0, 3.0], default=0.0)
    one_minus_nmae = 1.0 - float(error.mean())
    ficr = float(np.sum(actual * units) / np.sum(actual * 4.0))
    return {
        "total": 0.5 * (one_minus_nmae + ficr),
        "one_minus_nmae": one_minus_nmae,
        "ficr": ficr,
    }


def _select_group_policies(
    base: pd.DataFrame,
    policies: dict[str, np.ndarray],
) -> tuple[dict[str, str], dict[str, dict[str, float]]]:
    selected: dict[str, str] = {}
    selected_scores: dict[str, dict[str, float]] = {}
    for group_id in CAPACITIES:
        scores = {
            name: _group_score(base, values, group_id)
            for name, values in policies.items()
        }
        best = max(scores, key=lambda name: scores[name]["total"])
        selected[str(group_id)] = best
        selected_scores[str(group_id)] = scores[best]
    return selected, selected_scores


def _apply_group_policies(
    base: pd.DataFrame,
    policies: dict[str, np.ndarray],
    selected: dict[str, str],
) -> np.ndarray:
    values = np.empty(len(base), dtype=float)
    groups = base["group_id"].to_numpy(dtype=int)
    for group_id in CAPACITIES:
        group = groups == group_id
        values[group] = policies[selected[str(group_id)]][group]
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument(
        "--fold",
        choices=("dev-2023-Q2", "dev-2023-Q3", "dev-2023-Q4"),
        required=True,
    )
    parser.add_argument("--calibration-days", type=int, default=60)
    parser.add_argument("--feature-count", type=int, default=160)
    parser.add_argument("--iterations", type=int, default=120)
    args = parser.parse_args()
    if not 30 <= args.calibration_days <= 90:
        raise ValueError("calibration-days must be between 30 and 90")
    if not 80 <= args.feature_count <= 240:
        raise ValueError("feature-count must be between 80 and 240")
    if not 60 <= args.iterations <= 240:
        raise ValueError("iterations must be between 60 and 240")
    if _sha256(OPEN) != OPEN_SHA or _sha256(BASELINE) != BASELINE_SHA:
        raise RuntimeError("immutable competition input hash mismatch")

    started = time.perf_counter()
    surface, feature_columns = development_surface()
    if surface["forecast_kst_dtm"].ge(DEV_CUTOFF).any():
        raise RuntimeError("lockbox row reached the soft-pseudo runner")
    validation = _validation_mask(surface, args.fold)
    outer_history = _strict_preceding_mask(surface, validation)
    target = surface["actual_kwh"] / surface["group_id"].map(CAPACITIES)
    observed_eligible = surface["actual_kwh"].notna().to_numpy() & target.ge(
        0.10
    ).to_numpy()
    outer_hard = outer_history & observed_eligible
    validation_cutoff = pd.Timestamp(
        surface.loc[validation, "data_available_kst_dtm"].min()
    )
    calibration_cutoff = validation_cutoff - np.timedelta64(
        args.calibration_days, "D"
    )
    inner_history = _strict_before(surface, calibration_cutoff)
    inner_hard = inner_history & observed_eligible
    inner_calibration = outer_hard & ~inner_history
    for group_id in CAPACITIES:
        group = surface["group_id"].eq(group_id).to_numpy()
        if int((inner_hard & group).sum()) < 300 or int(
            (inner_calibration & group).sum()
        ) < 300:
            raise RuntimeError(f"group {group_id} lacks inner support")

    matrix = surface[feature_columns].astype("float32")
    inner_features = _feature_screen(
        matrix, surface, target, inner_hard, args.feature_count
    )
    inner_pseudo_rows, inner_quantiles, inner_mapper = _pseudo_quantiles(
        surface, inner_history
    )
    inner_base = surface.loc[inner_calibration, BASE_COLUMNS].copy()
    inner_groups = inner_base["group_id"].to_numpy(dtype=int)
    inner_means = {
        group_id: float(
            target.loc[
                inner_hard & surface["group_id"].eq(group_id).to_numpy()
            ].mean()
        )
        for group_id in CAPACITIES
    }
    sweep: dict[str, object] = {}
    best: tuple[float, str, str, float, dict[str, str]] | None = None
    for name, mode, pseudo_weight in CONFIGS:
        probability, centers, expansion = _fit_probability(
            matrix,
            surface,
            target,
            inner_hard,
            inner_calibration,
            inner_features,
            inner_pseudo_rows,
            inner_quantiles,
            mode,
            pseudo_weight,
            args.iterations,
        )
        policies = _policy_values(probability, centers, inner_groups, inner_means)
        selection, group_scores = _select_group_policies(inner_base, policies)
        normalized = _apply_group_policies(inner_base, policies, selection)
        capacity = inner_base["group_id"].map(CAPACITIES).to_numpy(dtype=float)
        score = _score(inner_base.assign(prediction_kwh=normalized * capacity))
        sweep[name] = {
            "mode": mode,
            "pseudo_weight": pseudo_weight,
            "score": score,
            "group_policies": selection,
            "group_scores": group_scores,
            "expansion": expansion,
        }
        choice = (score["total"], name, mode, pseudo_weight, selection)
        if best is None or choice[0] > best[0]:
            best = choice
        print(json.dumps({"inner": name, "score": score}), flush=True)
    assert best is not None

    final_features = _feature_screen(
        matrix, surface, target, outer_hard, args.feature_count
    )
    outer_pseudo_rows, outer_quantiles, outer_mapper = _pseudo_quantiles(
        surface, outer_history
    )
    probability, centers, expansion = _fit_probability(
        matrix,
        surface,
        target,
        outer_hard,
        validation,
        final_features,
        outer_pseudo_rows,
        outer_quantiles,
        best[2],
        best[3],
        args.iterations,
    )
    outer_base = surface.loc[validation, BASE_COLUMNS].copy()
    outer_groups = outer_base["group_id"].to_numpy(dtype=int)
    outer_means = {
        group_id: float(
            target.loc[
                outer_hard & surface["group_id"].eq(group_id).to_numpy()
            ].mean()
        )
        for group_id in CAPACITIES
    }
    policies = _policy_values(probability, centers, outer_groups, outer_means)
    normalized = _apply_group_policies(outer_base, policies, best[4])
    capacity = outer_base["group_id"].map(CAPACITIES).to_numpy(dtype=float)
    output = outer_base.assign(
        prediction_kwh=normalized * capacity,
        fold_id=args.fold,
        model_id=args.candidate_id,
    )
    group_scores = {
        str(group_id): _group_score(outer_base, normalized, group_id)
        for group_id in CAPACITIES
    }
    output_path = OUTPUT / f"{args.candidate_id}-{args.fold}.parquet"
    output.to_parquet(output_path, index=False)
    receipt = {
        "schema_version": 1,
        "candidate_id": args.candidate_id,
        "fold_id": args.fold,
        "architecture": "strict_uncertainty_aware_soft_pseudo_xgboost",
        "scope": "outer labels excluded from mapper, feature, config, and policy selection",
        "physical_development_cutoff": str(DEV_CUTOFF),
        "calibration_days": args.calibration_days,
        "feature_count": args.feature_count,
        "iterations": args.iterations,
        "inner_mapper": inner_mapper,
        "inner_sweep": sweep,
        "selected_config": best[1],
        "selected_mode": best[2],
        "selected_pseudo_weight": best[3],
        "selected_group_policies": best[4],
        "selected_inner_score": best[0],
        "outer_mapper": outer_mapper,
        "outer_expansion": expansion,
        "group_scores": group_scores,
        "fold_score": _score(output),
        "selected_feature_names": final_features,
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
