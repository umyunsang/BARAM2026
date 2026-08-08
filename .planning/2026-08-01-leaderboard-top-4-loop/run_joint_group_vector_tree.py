"""Strict hourly three-group vector-leaf regression probe.

One sample represents one forecast timestamp and the three capacity-normalized
group targets form a dense output vector.  A vector-leaf XGBoost model shares
every tree split across the three targets, allowing the supplied NWP to encode
regional wind structure without using contemporaneous group generation at
inference.  The outer Q3 labels are used only for the final fixed screen.
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
from xgboost import XGBRegressor

BASE_COLUMNS = ["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh"]
RESEARCH_SOURCE = "https://xgboost.readthedocs.io/en/stable/tutorials/multioutput.html"


def _aligned_groups(surface: pd.DataFrame) -> dict[int, pd.DataFrame]:
    groups = {
        group_id: surface.loc[surface["group_id"].eq(group_id)]
        .sort_values(["forecast_kst_dtm", "forecast_id"])
        .reset_index(drop=True)
        for group_id in CAPACITIES
    }
    reference = groups[1][["forecast_id", "forecast_kst_dtm"]]
    for group_id in (2, 3):
        candidate = groups[group_id][["forecast_id", "forecast_kst_dtm"]]
        if not candidate.equals(reference):
            raise RuntimeError(f"group {group_id} hourly key alignment changed")
    return groups


def _wide_masks(
    surface: pd.DataFrame,
    groups: dict[int, pd.DataFrame],
    validation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    strict = _strict_preceding_mask(surface, validation)
    keyed_strict = pd.DataFrame(
        {
            "forecast_id": surface["forecast_id"],
            "group_id": surface["group_id"],
            "strict": strict,
            "validation": validation,
        }
    )
    group1 = keyed_strict.loc[keyed_strict["group_id"].eq(1)].sort_values(
        "forecast_id"
    )
    reference = groups[1].sort_values("forecast_id")
    if not group1["forecast_id"].reset_index(drop=True).equals(
        reference["forecast_id"].reset_index(drop=True)
    ):
        raise RuntimeError("wide mask key alignment changed")
    strict_by_id = dict(zip(group1["forecast_id"], group1["strict"], strict=True))
    validation_by_id = dict(
        zip(group1["forecast_id"], group1["validation"], strict=True)
    )
    ordered_ids = groups[1]["forecast_id"]
    history = ordered_ids.map(strict_by_id).to_numpy(dtype=bool)
    apply = ordered_ids.map(validation_by_id).to_numpy(dtype=bool)
    return history, apply


def _targets(groups: dict[int, pd.DataFrame]) -> np.ndarray:
    return np.column_stack(
        [
            groups[group_id]["actual_kwh"].to_numpy(dtype=float)
            / CAPACITIES[group_id]
            for group_id in CAPACITIES
        ]
    )


def _wide_features(
    groups: dict[int, pd.DataFrame],
    feature_columns: list[str],
    needed: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, int]]:
    base = groups[1].loc[needed, feature_columns].to_numpy(dtype="float32")
    matrices = [base]
    names = [f"{name}__g1" for name in feature_columns]
    varying_counts: dict[str, int] = {}
    for group_id in (2, 3):
        current = groups[group_id].loc[needed, feature_columns].to_numpy(
            dtype="float32"
        )
        difference = current - base
        finite_difference = np.nan_to_num(
            difference, nan=0.0, posinf=0.0, neginf=0.0
        )
        varying = np.max(np.abs(finite_difference), axis=0) > 1e-7
        varying_counts[str(group_id)] = int(varying.sum())
        matrices.append(difference[:, varying])
        names.extend(
            f"{feature_columns[position]}__d{group_id}1"
            for position in np.flatnonzero(varying)
        )
        del current, difference, finite_difference
        gc.collect()
    matrix = np.concatenate(matrices, axis=1)
    output = pd.DataFrame(matrix, columns=names)
    output = output.replace([np.inf, -np.inf], np.nan)
    medians = output.median(axis=0).fillna(0.0)
    output = output.fillna(medians).astype("float32")
    if not np.isfinite(output.to_numpy()).all():
        raise RuntimeError("joint-group feature matrix contains non-finite values")
    return output, varying_counts


def _screen_features(
    matrix: pd.DataFrame,
    target: np.ndarray,
    training: np.ndarray,
    per_target: int,
) -> tuple[list[str], dict[str, list[str]]]:
    selected_by_target: dict[str, list[str]] = {}
    selected: list[str] = []
    sample_weight = np.mean(np.clip(target[training], 0.10, None), axis=1)
    for output_index, group_id in enumerate(CAPACITIES):
        model = LGBMRegressor(
            objective="l1",
            n_estimators=180,
            learning_rate=0.035,
            num_leaves=31,
            min_child_samples=45,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.75,
            reg_alpha=0.2,
            reg_lambda=4.0,
            random_state=20260860 + output_index,
            n_jobs=6,
            deterministic=True,
            force_col_wise=True,
            verbosity=-1,
        )
        model.fit(
            matrix.loc[training],
            target[training, output_index],
            sample_weight=sample_weight,
        )
        gain = model.booster_.feature_importance(importance_type="gain")
        order = np.argsort(gain)[::-1][:per_target]
        names = [matrix.columns[position] for position in order]
        selected_by_target[str(group_id)] = names
        selected.extend(names)
        del model
        gc.collect()
    union = list(dict.fromkeys(selected))
    if len(union) < per_target or len(union) > per_target * 3:
        raise RuntimeError("joint-group feature union contract changed")
    return union, selected_by_target


def _model(strategy: str, iterations: int, seed: int) -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        multi_strategy=strategy,
        n_estimators=iterations,
        learning_rate=0.025,
        max_depth=5,
        min_child_weight=12.0,
        subsample=0.9,
        colsample_bytree=0.85,
        reg_alpha=0.15,
        reg_lambda=5.0,
        max_bin=256,
        tree_method="hist",
        random_state=seed,
        n_jobs=6,
    )


def _prediction_frame(
    groups: dict[int, pd.DataFrame],
    apply: np.ndarray,
    prediction: np.ndarray,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for output_index, group_id in enumerate(CAPACITIES):
        part = groups[group_id].loc[apply, BASE_COLUMNS].copy()
        part["prediction_kwh"] = (
            prediction[:, output_index] * CAPACITIES[group_id]
        )
        parts.append(part)
    return pd.concat(parts, ignore_index=True).sort_values(
        ["forecast_kst_dtm", "group_id"]
    )


def _group_scores(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    scores: dict[str, dict[str, float]] = {}
    for group_id, capacity in CAPACITIES.items():
        group = frame.loc[frame["group_id"].eq(group_id)]
        scores[str(group_id)] = _group_total(
            group["actual_kwh"].to_numpy(dtype=float) / capacity,
            group["prediction_kwh"].to_numpy(dtype=float) / capacity,
        )
    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--fold", choices=("dev-2023-Q3",), required=True)
    parser.add_argument("--features-per-target", type=int, default=80)
    parser.add_argument("--iterations", type=int, default=400)
    args = parser.parse_args()
    if not 40 <= args.features_per_target <= 140:
        raise ValueError("features-per-target must be between 40 and 140")
    if not 150 <= args.iterations <= 600:
        raise ValueError("iterations must be between 150 and 600")
    if _sha256(OPEN) != OPEN_SHA or _sha256(BASELINE) != BASELINE_SHA:
        raise RuntimeError("immutable competition input hash mismatch")

    started = time.perf_counter()
    surface, feature_columns = development_surface()
    if surface["forecast_kst_dtm"].ge(DEV_CUTOFF).any():
        raise RuntimeError("lockbox row reached joint-group vector-tree runner")
    validation = _validation_mask(surface, args.fold)
    groups = _aligned_groups(surface)
    history, apply = _wide_masks(surface, groups, validation)
    target = _targets(groups)
    training = history & np.isfinite(target).all(axis=1)
    if int(training.sum()) < 3500 or int(apply.sum()) < 2000:
        raise RuntimeError("joint-group train/application sample contract changed")
    needed = training | apply
    matrix_needed, varying_counts = _wide_features(
        groups, feature_columns, needed
    )
    local_training = training[needed]
    local_apply = apply[needed]
    target_needed = target[needed]
    selected, selected_by_target = _screen_features(
        matrix_needed,
        target_needed,
        local_training,
        args.features_per_target,
    )
    matrix = matrix_needed[selected]
    sample_weight = np.mean(
        np.clip(target_needed[local_training], 0.10, None), axis=1
    )
    predictions: dict[str, np.ndarray] = {}
    for strategy_index, strategy in enumerate(
        ("multi_output_tree", "one_output_per_tree")
    ):
        model = _model(strategy, args.iterations, 20260870 + strategy_index)
        model.fit(
            matrix.loc[local_training],
            target_needed[local_training],
            sample_weight=sample_weight,
        )
        prediction = np.asarray(model.predict(matrix.loc[local_apply]), dtype=float)
        if prediction.shape != (int(local_apply.sum()), 3):
            raise RuntimeError(f"{strategy} prediction shape changed")
        predictions[strategy] = np.clip(prediction, 0.0, 1.075)
        print(
            json.dumps(
                {
                    "strategy": strategy,
                    "status": "fit",
                    "training_samples": int(local_training.sum()),
                    "application_samples": int(local_apply.sum()),
                    "feature_count": len(selected),
                }
            ),
            flush=True,
        )
        del model
        gc.collect()

    predictions["fixed_half_blend"] = 0.5 * (
        predictions["multi_output_tree"]
        + predictions["one_output_per_tree"]
    )
    frames = {
        name: _prediction_frame(groups, apply, prediction)
        for name, prediction in predictions.items()
    }
    scores = {name: _score(frame) for name, frame in frames.items()}
    group_scores = {
        name: _group_scores(frame) for name, frame in frames.items()
    }

    fixed_name = "multi_output_tree"
    output = frames[fixed_name].copy()
    output["fold_id"] = args.fold
    output["model_id"] = args.candidate_id
    output_path = OUTPUT / f"{args.candidate_id}-{args.fold}.parquet"
    output.to_parquet(output_path, index=False)
    receipt = {
        "schema_version": 1,
        "candidate_id": args.candidate_id,
        "fold_id": args.fold,
        "architecture": "strict_hourly_three_group_xgboost_vector_leaf",
        "scope": (
            "fixed vector-leaf outer screen; Q3 labels excluded from fitting "
            "and feature selection"
        ),
        "research_source": RESEARCH_SOURCE,
        "physical_development_cutoff": str(DEV_CUTOFF),
        "training_samples": int(training.sum()),
        "application_samples": int(apply.sum()),
        "raw_feature_count": matrix_needed.shape[1],
        "varying_delta_feature_counts": varying_counts,
        "features_per_target": args.features_per_target,
        "selected_feature_count": len(selected),
        "selected_features_by_target": selected_by_target,
        "selected_features": selected,
        "iterations": args.iterations,
        "fixed_output": fixed_name,
        "strategy_scores": scores,
        "strategy_group_scores": group_scores,
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
    print(
        json.dumps(
            {
                "candidate_id": args.candidate_id,
                "fixed_score": scores[fixed_name],
                "fixed_group_scores": group_scores[fixed_name],
                "diagnostic_scores": scores,
                "runtime_seconds": receipt["runtime_seconds"],
                "prediction_sha256": receipt["prediction_sha256"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
