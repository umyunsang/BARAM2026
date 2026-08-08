"""Fit a cross-fitted point-plus-residual distribution on strict PLS features."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from run_group_balanced_pls_rank import _frame, _group_scores
from run_multioutput_donor_pls_rank import (
    M195_LATENT,
    M195_RECEIPT,
    _multioutput_latent,
)
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
from run_strict_prequential_source_rank import _screen
from sklearn.model_selection import GroupKFold
from strict_dev_surface import DEV_CUTOFF, development_surface
from xgboost import XGBClassifier

TOP_FEATURES = 220
POINT_ITERATIONS = 300
RESIDUAL_ITERATIONS = 200
RESIDUAL_WIDTH = 0.02
RESIDUAL_MIN = -0.50
RESIDUAL_MAX = 0.50
TEMPERATURE = 0.40
UTILITY_GAMMA = 2.0
ACTIONS = np.arange(0.075, 1.0751, 0.0025)
PARENT_PATH = OUTPUT / "M197_STRICT_FIXED_HALF_PLS-dev-2023-Q3.parquet"


def _point_model(seed: int) -> LGBMRegressor:
    return LGBMRegressor(
        objective="l1",
        n_estimators=POINT_ITERATIONS,
        learning_rate=0.025,
        num_leaves=31,
        min_child_samples=60,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=3.0,
        random_state=seed,
        n_jobs=6,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )


def _residual_probability(
    matrix: pd.DataFrame,
    target: pd.Series,
    training: np.ndarray,
    validation: np.ndarray,
    selected: list[str],
    surface: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    training_positions = np.flatnonzero(training)
    fit_matrix = matrix.loc[training, selected].reset_index(drop=True)
    fit_target = target.loc[training].to_numpy(dtype=float)
    batches = surface.loc[training, "data_available_kst_dtm"].astype(str).to_numpy()
    oof_point = np.full(len(fit_matrix), np.nan, dtype=float)
    splitter = GroupKFold(n_splits=4)
    fold_diagnostics: list[dict[str, object]] = []
    for fold_index, (fit_index, holdout_index) in enumerate(
        splitter.split(fit_matrix, fit_target, batches)
    ):
        model = _point_model(20262600 + fold_index)
        model.fit(fit_matrix.iloc[fit_index], fit_target[fit_index])
        oof_point[holdout_index] = model.predict(fit_matrix.iloc[holdout_index])
        fold_diagnostics.append(
            {
                "fold": fold_index,
                "fit_rows": len(fit_index),
                "holdout_rows": len(holdout_index),
                "holdout_batches": int(np.unique(batches[holdout_index]).size),
            }
        )
        print(json.dumps({"point_crossfit": fold_diagnostics[-1]}), flush=True)
        del model
        gc.collect()
    if not np.isfinite(oof_point).all():
        raise RuntimeError("cross-fitted point predictions are incomplete")

    final_model = _point_model(20262610)
    final_model.fit(fit_matrix, fit_target)
    validation_point = np.clip(
        final_model.predict(matrix.loc[validation, selected]), 0.075, 1.075
    )
    del final_model
    gc.collect()

    residual = np.clip(fit_target - oof_point, RESIDUAL_MIN, RESIDUAL_MAX - 1e-8)
    raw_class = np.floor((residual - RESIDUAL_MIN) / RESIDUAL_WIDTH).astype(int)
    active_classes = np.asarray(sorted(np.unique(raw_class)), dtype=int)
    class_map = {raw: index for index, raw in enumerate(active_classes)}
    classes = np.asarray([class_map[value] for value in raw_class], dtype=int)
    centers = np.asarray(
        [residual[classes == index].mean() for index in range(len(active_classes))],
        dtype=float,
    )
    meta_fit = fit_matrix.copy()
    meta_fit["residual_parent_point"] = oof_point.astype("float32")
    meta_apply = matrix.loc[validation, selected].copy()
    meta_apply["residual_parent_point"] = validation_point.astype("float32")
    classifier = XGBClassifier(
        objective="multi:softprob",
        num_class=len(active_classes),
        n_estimators=RESIDUAL_ITERATIONS,
        learning_rate=0.03,
        max_depth=5,
        min_child_weight=20.0,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=5.0,
        max_bin=256,
        tree_method="hist",
        random_state=20262620,
        n_jobs=6,
    )
    classifier.fit(
        meta_fit,
        classes,
        sample_weight=np.clip(fit_target, 0.10, None),
    )
    raw_probability = np.asarray(classifier.predict_proba(meta_apply), dtype=float)
    learned = np.asarray(classifier.classes_, dtype=int)
    probability = np.zeros((int(validation.sum()), len(active_classes)), dtype=float)
    probability[:, learned] = raw_probability
    probability = np.clip(probability, 1e-12, None) ** (1.0 / TEMPERATURE)
    probability /= probability.sum(axis=1, keepdims=True)
    diagnostics: dict[str, object] = {
        "training_positions_count": len(training_positions),
        "crossfit_folds": fold_diagnostics,
        "oof_point_mae": float(np.mean(np.abs(oof_point - fit_target))),
        "residual_class_count": len(active_classes),
        "residual_center_min": float(centers.min()),
        "residual_center_max": float(centers.max()),
    }
    del classifier, raw_probability, meta_fit, meta_apply, fit_matrix
    gc.collect()
    return validation_point, probability, centers, diagnostics


def _residual_actions(
    point: np.ndarray,
    probability: np.ndarray,
    residual_centers: np.ndarray,
    groups: np.ndarray,
    mean_generation: dict[int, float],
) -> np.ndarray:
    targets = np.clip(point[:, None] + residual_centers[None, :], 0.075, 1.075)
    chosen = np.empty(len(point), dtype=float)
    for group_id in CAPACITIES:
        group_positions = np.flatnonzero(groups == group_id)
        for start in range(0, len(group_positions), 256):
            positions = group_positions[start : start + 256]
            samples = targets[positions]
            mass = probability[positions]
            error = np.abs(ACTIONS[:, None, None] - samples[None, :, :])
            units = np.select(
                [error <= 0.06, error <= 0.08], [4.0, 3.0], default=0.0
            )
            expected_error = np.einsum("arb,rb->ar", error, mass).T
            expected_revenue = np.einsum(
                "arb,rb,rb->ar", units, samples, mass
            ).T / (4.0 * mean_generation[group_id])
            utility = -expected_error + UTILITY_GAMMA * expected_revenue
            chosen[positions] = ACTIONS[np.argmax(utility, axis=1)]
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--fold", choices=("dev-2023-Q3",), required=True)
    args = parser.parse_args()
    if _sha256(OPEN) != OPEN_SHA or _sha256(BASELINE) != BASELINE_SHA:
        raise RuntimeError("immutable competition input hash mismatch")
    if not PARENT_PATH.exists() or not M195_RECEIPT.exists() or not M195_LATENT.exists():
        raise RuntimeError("verified PLS parent surface is missing")

    started = time.perf_counter()
    surface, feature_columns = development_surface()
    if surface["forecast_kst_dtm"].ge(DEV_CUTOFF).any():
        raise RuntimeError("lockbox row reached cross-fitted residual runner")
    validation = _validation_mask(surface, args.fold)
    history = _strict_preceding_mask(surface, validation)
    target = surface["actual_kwh"] / surface["group_id"].map(CAPACITIES)
    training = (
        history
        & surface["actual_kwh"].notna().to_numpy()
        & target.ge(0.10).to_numpy()
    )

    parent_receipt = json.loads(M195_RECEIPT.read_text())
    if _sha256(M195_LATENT) != parent_receipt["latent_checkpoint_sha256"]:
        raise RuntimeError("M195 latent checkpoint hash mismatch")
    cached = np.load(M195_LATENT, allow_pickle=False)
    parent_columns = [str(value) for value in cached["columns"].tolist()]
    parent_values = np.asarray(cached["values"], dtype="float32")
    if parent_values.shape != (len(surface), len(parent_columns)):
        raise RuntimeError("M195 latent checkpoint shape contract changed")
    base_matrix = surface[feature_columns].astype("float32")
    multioutput, multioutput_diagnostics = _multioutput_latent(
        surface,
        base_matrix,
        history,
        validation,
        target,
        feature_columns,
    )
    matrix = pd.concat(
        [
            base_matrix,
            pd.DataFrame(parent_values, columns=parent_columns, index=surface.index),
            multioutput,
        ],
        axis=1,
    )
    del base_matrix, parent_values, multioutput
    gc.collect()

    selected = _screen(
        matrix,
        target,
        training,
        list(matrix.columns),
        TOP_FEATURES,
        20262600,
    )
    point, probability, residual_centers, residual_diagnostics = (
        _residual_probability(
            matrix,
            target,
            training,
            validation,
            selected,
            surface,
        )
    )
    del matrix
    gc.collect()
    groups = surface.loc[validation, "group_id"].to_numpy(dtype=int)
    means = {
        group_id: float(
            target.loc[training & surface["group_id"].eq(group_id).to_numpy()].mean()
        )
        for group_id in CAPACITIES
    }
    normalized = _residual_actions(
        point, probability, residual_centers, groups, means
    )
    point_output = _frame(surface, validation, point)
    residual_output = _frame(surface, validation, normalized)

    parent = pd.read_parquet(PARENT_PATH)
    expected_keys = pd.MultiIndex.from_frame(
        surface.loc[validation, ["forecast_id", "group_id"]]
    )
    parent_keys = pd.MultiIndex.from_frame(parent[["forecast_id", "group_id"]])
    if len(parent) != int(validation.sum()) or not parent_keys.equals(expected_keys):
        raise RuntimeError("M197 parent key contract changed")
    capacity = parent["group_id"].map(CAPACITIES).to_numpy(dtype=float)
    parent_normalized = parent["prediction_kwh"].to_numpy(dtype=float) / capacity
    output = _frame(surface, validation, 0.5 * parent_normalized + 0.5 * normalized)
    output["fold_id"] = args.fold
    output["model_id"] = args.candidate_id
    output_path = OUTPUT / f"{args.candidate_id}-{args.fold}.parquet"
    output.to_parquet(output_path, index=False)

    receipt = {
        "schema_version": 1,
        "candidate_id": args.candidate_id,
        "fold_id": args.fold,
        "architecture": "strict_crossfit_l1_plus_xgb_residual_distribution_half_m197",
        "scope": (
            "fixed official-data-only batch-grouped cross-fitted residual model; "
            "outer Q3 labels excluded from screen, point, residual, policy, and blend"
        ),
        "physical_development_cutoff": str(DEV_CUTOFF),
        "strict_training_rows": int(training.sum()),
        "top_features": TOP_FEATURES,
        "selected_features": selected,
        "selected_latent_feature_count": sum(
            name.startswith(("pls__", "mpls__")) for name in selected
        ),
        "point_iterations": POINT_ITERATIONS,
        "residual_iterations": RESIDUAL_ITERATIONS,
        "residual_width": RESIDUAL_WIDTH,
        "temperature": TEMPERATURE,
        "utility_gamma": UTILITY_GAMMA,
        "residual_diagnostics": residual_diagnostics,
        "multioutput_diagnostics": multioutput_diagnostics,
        "point_score": _score(point_output),
        "point_group_scores": _group_scores(point_output),
        "raw_score": _score(residual_output),
        "raw_group_scores": _group_scores(residual_output),
        "fixed_parent_weight": 0.5,
        "fold_score": _score(output),
        "group_scores": _group_scores(output),
        "runtime_seconds": round(time.perf_counter() - started, 2),
        "prediction_path": str(output_path.relative_to(Path.cwd())),
        "prediction_sha256": _sha256(output_path),
        "parent_path": str(PARENT_PATH.relative_to(Path.cwd())),
        "parent_sha256": _sha256(PARENT_PATH),
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
                "point_score": receipt["point_score"],
                "raw_score": receipt["raw_score"],
                "raw_group_scores": receipt["raw_group_scores"],
                "fold_score": receipt["fold_score"],
                "group_scores": receipt["group_scores"],
                "residual_diagnostics": residual_diagnostics,
                "runtime_seconds": receipt["runtime_seconds"],
                "prediction_sha256": receipt["prediction_sha256"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
