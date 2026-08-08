"""Learn action-conditional settlement events on the strict M195 PLS surface."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from run_group_balanced_pls_rank import _frame, _group_scores
from run_multioutput_donor_pls_rank import M195_LATENT, M195_RECEIPT
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

TRAIN_ACTIONS = np.arange(0.075, 1.0751, 0.025, dtype="float32")
INFERENCE_ACTIONS = np.arange(0.075, 1.0751, 0.01, dtype="float32")
ITERATIONS = 180
PARENT_PATH = OUTPUT / "M197_STRICT_FIXED_HALF_PLS-dev-2023-Q3.parquet"


def _parameters() -> dict[str, object]:
    return {
        "n_estimators": ITERATIONS,
        "learning_rate": 0.035,
        "num_leaves": 31,
        "min_child_samples": 200,
        "max_bin": 127,
        "subsample": 0.9,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.2,
        "reg_lambda": 6.0,
        "random_state": 20262110,
        "n_jobs": 6,
        "deterministic": True,
        "force_col_wise": True,
        "verbosity": -1,
    }


def _expand(
    matrix: np.ndarray,
    anchors: np.ndarray,
    actions: np.ndarray,
) -> np.ndarray:
    rows = len(matrix)
    expanded = np.repeat(matrix, len(actions), axis=0)
    action = np.tile(actions, rows)
    anchor = np.repeat(anchors, len(actions), axis=0)
    anchor_mean = anchor.mean(axis=1)
    anchor_spread = anchor.std(axis=1)
    differences = action[:, None] - anchor
    derived = np.column_stack(
        (
            action,
            np.square(action),
            action**3,
            anchor_mean,
            anchor_spread,
            action - anchor_mean,
            np.abs(action - anchor_mean),
            action * anchor_mean,
            differences,
            np.abs(differences),
        )
    ).astype("float32")
    return np.column_stack((expanded, derived)).astype("float32", copy=False)


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
        raise RuntimeError("lockbox row reached action-event runner")
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
    latent_columns = [str(value) for value in cached["columns"].tolist()]
    latent_values = np.asarray(cached["values"], dtype="float32")
    if latent_values.shape != (len(surface), len(latent_columns)):
        raise RuntimeError("M195 latent checkpoint shape contract changed")
    matrix = pd.concat(
        (
            surface[feature_columns].astype("float32"),
            pd.DataFrame(latent_values, columns=latent_columns, index=surface.index),
        ),
        axis=1,
    )
    del latent_values
    anchor_columns = [name for name in latent_columns if "prediction_p1" in name]
    if len(anchor_columns) != 9:
        raise RuntimeError(f"PLS action-anchor contract resolved {len(anchor_columns)}")
    selected = list(
        dict.fromkeys(
            [*parent_receipt["selected_features"]["global"], *anchor_columns]
        )
    )
    missing = sorted(set(selected).difference(matrix.columns))
    if missing:
        raise RuntimeError(f"M195 selected-feature contract changed: {missing[:3]}")
    raw_matrix = matrix[selected].to_numpy(dtype="float32")
    anchors = matrix[anchor_columns].to_numpy(dtype="float32")
    if not np.isfinite(anchors[training | validation]).all():
        raise RuntimeError("PLS action anchors contain non-finite values")
    del matrix
    gc.collect()

    target_train = target.loc[training].to_numpy(dtype="float32")
    train_groups = surface.loc[training, "group_id"].to_numpy(dtype=int)
    group_counts = {
        group_id: int(np.sum(train_groups == group_id)) for group_id in CAPACITIES
    }
    row_weight = np.asarray(
        [len(train_groups) / (3.0 * group_counts[int(group)]) for group in train_groups],
        dtype="float32",
    )
    expanded_weight = np.repeat(row_weight, len(TRAIN_ACTIONS))
    expanded_train = _expand(
        raw_matrix[training], anchors[training], TRAIN_ACTIONS
    )
    error = np.abs(target_train[:, None] - TRAIN_ACTIONS[None, :])
    within6 = (error <= 0.06).astype("int8").reshape(-1)
    within8 = (error <= 0.08).astype("int8").reshape(-1)
    absolute_error = error.astype("float32").reshape(-1)
    params = _parameters()
    event6_model = LGBMClassifier(objective="binary", **params)
    event8_model = LGBMClassifier(
        objective="binary", **{**params, "random_state": 20262111}
    )
    error_model = LGBMRegressor(
        objective="l2", **{**params, "random_state": 20262112}
    )
    point_model = LGBMRegressor(
        objective="l1",
        **{
            **params,
            "min_child_samples": 80,
            "random_state": 20262113,
        },
    )
    event6_model.fit(expanded_train, within6, sample_weight=expanded_weight)
    print(json.dumps({"action_head": "within6", "status": "fit"}), flush=True)
    event8_model.fit(expanded_train, within8, sample_weight=expanded_weight)
    print(json.dumps({"action_head": "within8", "status": "fit"}), flush=True)
    error_model.fit(expanded_train, absolute_error, sample_weight=expanded_weight)
    print(json.dumps({"action_head": "absolute_error", "status": "fit"}), flush=True)
    point_model.fit(
        raw_matrix[training], target_train, sample_weight=row_weight
    )
    print(json.dumps({"action_head": "point", "status": "fit"}), flush=True)
    del expanded_train, within6, within8, absolute_error, error
    gc.collect()

    validation_positions = np.flatnonzero(validation)
    validation_groups = surface.loc[validation, "group_id"].to_numpy(dtype=int)
    means = {
        group_id: float(target.loc[training & surface["group_id"].eq(group_id)].mean())
        for group_id in CAPACITIES
    }
    group_means = np.asarray(
        [means[int(group)] for group in validation_groups], dtype=float
    )
    normalized = np.empty(len(validation_positions), dtype=float)
    for start in range(0, len(validation_positions), 256):
        stop = min(start + 256, len(validation_positions))
        positions = validation_positions[start:stop]
        expanded = _expand(raw_matrix[positions], anchors[positions], INFERENCE_ACTIONS)
        shape = (len(positions), len(INFERENCE_ACTIONS))
        probability6 = event6_model.predict_proba(expanded)[:, 1].reshape(shape)
        probability8 = event8_model.predict_proba(expanded)[:, 1].reshape(shape)
        probability8 = np.maximum(probability8, probability6)
        expected_error = np.clip(error_model.predict(expanded).reshape(shape), 0.0, None)
        point = np.clip(
            point_model.predict(raw_matrix[positions]), 0.10, 1.075
        )
        settlement = (4.0 * probability6 + 3.0 * (probability8 - probability6)) / 4.0
        utility = -expected_error + (
            point[:, None] / group_means[start:stop, None]
        ) * settlement
        normalized[start:stop] = INFERENCE_ACTIONS[np.argmax(utility, axis=1)]
        del expanded, probability6, probability8, expected_error, settlement, utility
    del event6_model, event8_model, error_model, point_model, raw_matrix, anchors
    gc.collect()
    raw_output = _frame(surface, validation, normalized)

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
        "architecture": "strict_m195_action_conditional_settlement_events",
        "scope": (
            "fixed official-data-only action-event screen; outer Q3 labels "
            "excluded from PLS, all action heads, utility, and blend"
        ),
        "physical_development_cutoff": str(DEV_CUTOFF),
        "strict_training_rows": int(training.sum()),
        "group_training_rows": group_counts,
        "training_action_count": len(TRAIN_ACTIONS),
        "training_action_step": 0.025,
        "inference_action_count": len(INFERENCE_ACTIONS),
        "inference_action_step": 0.01,
        "iterations": ITERATIONS,
        "selected_feature_count": len(selected),
        "anchor_feature_count": len(anchor_columns),
        "expanded_feature_count": len(selected) + 8 + 2 * len(anchor_columns),
        "utility_gamma": 1.0,
        "raw_score": _score(raw_output),
        "raw_group_scores": _group_scores(raw_output),
        "fixed_parent_weight": 0.5,
        "fold_score": _score(output),
        "group_scores": _group_scores(output),
        "runtime_seconds": round(time.perf_counter() - started, 2),
        "prediction_path": str(output_path.relative_to(Path.cwd())),
        "prediction_sha256": _sha256(output_path),
        "parent_path": str(PARENT_PATH.relative_to(Path.cwd())),
        "parent_sha256": _sha256(PARENT_PATH),
        "m195_receipt_sha256": _sha256(M195_RECEIPT),
        "m195_latent_sha256": _sha256(M195_LATENT),
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
                "raw_score": receipt["raw_score"],
                "raw_group_scores": receipt["raw_group_scores"],
                "fold_score": receipt["fold_score"],
                "group_scores": receipt["group_scores"],
                "runtime_seconds": receipt["runtime_seconds"],
                "prediction_sha256": receipt["prediction_sha256"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
