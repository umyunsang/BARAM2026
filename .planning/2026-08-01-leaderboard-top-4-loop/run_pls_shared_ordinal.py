"""Fit one threshold-conditioned ordinal model over verified PLS features."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from run_inner_policy_classifier import _group_total, _policy_values
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
from run_strict_prequential_source_rank import FROZEN_POLICY, _screen
from strict_dev_surface import DEV_CUTOFF, development_surface

BASE_COLUMNS = ["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh"]
CLASS_WIDTH = 0.02
TOP_FEATURES = 180
LOCAL_BOUNDARY_OFFSETS = np.asarray((-4, -3, -2, -1, 0, 1, 2, 3))
GLOBAL_BOUNDARY_COUNT = 4
ITERATIONS = 320
PARENT_PATH = OUTPUT / "M197_STRICT_FIXED_HALF_PLS-dev-2023-Q3.parquet"


def _ordered_probability(survival: np.ndarray) -> np.ndarray:
    """Project threshold survival probabilities to ordered class masses."""
    monotone = np.sort(np.asarray(survival, dtype=float), axis=1)[:, ::-1]
    probability = np.empty((len(monotone), monotone.shape[1] + 1), dtype=float)
    probability[:, 0] = 1.0 - monotone[:, 0]
    probability[:, 1:-1] = monotone[:, :-1] - monotone[:, 1:]
    probability[:, -1] = monotone[:, -1]
    probability = np.clip(probability, 1e-12, None)
    probability /= probability.sum(axis=1, keepdims=True)
    return probability


def _threshold_features(
    base: np.ndarray,
    boundary_indices: np.ndarray,
    boundary_values: np.ndarray,
) -> np.ndarray:
    values = boundary_values[boundary_indices].astype("float32")
    return np.column_stack((base, values, values * values)).astype(
        "float32", copy=False
    )


def _expanded_training(
    base: np.ndarray,
    classes: np.ndarray,
    target: np.ndarray,
    boundary_values: np.ndarray,
    sampling_mode: str,
    weight_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    boundary_count = len(boundary_values)
    row_index = np.arange(len(base), dtype=np.int64)
    if sampling_mode == "local":
        local = np.clip(
            classes[:, None] + LOCAL_BOUNDARY_OFFSETS[None, :],
            0,
            boundary_count - 1,
        )
        global_offsets = np.arange(GLOBAL_BOUNDARY_COUNT, dtype=np.int64)
        global_boundaries = (
            row_index[:, None] * 17 + global_offsets[None, :] * 13
        ) % boundary_count
        sampled = np.column_stack((local, global_boundaries)).astype(np.int64)
    else:
        independent_offsets = np.arange(
            len(LOCAL_BOUNDARY_OFFSETS) + GLOBAL_BOUNDARY_COUNT,
            dtype=np.int64,
        )
        sampled = (
            row_index[:, None] * 17 + independent_offsets[None, :] * 13
        ) % boundary_count
    repeated = np.repeat(base, sampled.shape[1], axis=0)
    flat_boundaries = sampled.reshape(-1)
    expanded = _threshold_features(repeated, flat_boundaries, boundary_values)
    repeated_classes = np.repeat(classes, sampled.shape[1])
    labels = (repeated_classes > flat_boundaries).astype(np.int8)
    row_weights = (
        np.clip(target, 0.10, None)
        if weight_mode == "generation"
        else np.ones(len(target), dtype=float)
    )
    weights = np.repeat(row_weights, sampled.shape[1]).astype("float32")
    diagnostics = {
        "source_rows": len(base),
        "thresholds_per_row": sampled.shape[1],
        "expanded_rows": len(expanded),
        "positive_rows": int(labels.sum()),
        "negative_rows": int((1 - labels).sum()),
        "sampling_mode": sampling_mode,
        "weight_mode": weight_mode,
    }
    return expanded, labels, weights, diagnostics


def _expanded_application(
    base: np.ndarray,
    boundary_values: np.ndarray,
) -> np.ndarray:
    boundary_indices = np.tile(
        np.arange(len(boundary_values), dtype=np.int64), len(base)
    )
    repeated = np.repeat(base, len(boundary_values), axis=0)
    return _threshold_features(repeated, boundary_indices, boundary_values)


def _frame(
    surface: pd.DataFrame,
    validation: np.ndarray,
    normalized: np.ndarray,
) -> pd.DataFrame:
    output = surface.loc[validation, BASE_COLUMNS].copy()
    output["prediction_kwh"] = normalized * output["group_id"].map(
        CAPACITIES
    ).to_numpy(dtype=float)
    return output


def _group_scores(output: pd.DataFrame) -> dict[str, dict[str, float]]:
    scores: dict[str, dict[str, float]] = {}
    for group_id, capacity in CAPACITIES.items():
        group = output.loc[output["group_id"].eq(group_id)]
        scores[str(group_id)] = _group_total(
            group["actual_kwh"].to_numpy(dtype=float) / capacity,
            group["prediction_kwh"].to_numpy(dtype=float) / capacity,
        )
    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--fold", choices=("dev-2023-Q3",), required=True)
    parser.add_argument(
        "--sampling-mode", choices=("local", "uniform"), default="local"
    )
    parser.add_argument(
        "--weight-mode", choices=("generation", "uniform"), default="generation"
    )
    args = parser.parse_args()
    if _sha256(OPEN) != OPEN_SHA or _sha256(BASELINE) != BASELINE_SHA:
        raise RuntimeError("immutable competition input hash mismatch")
    if not PARENT_PATH.exists() or not M195_RECEIPT.exists() or not M195_LATENT.exists():
        raise RuntimeError("verified PLS parent surface is missing")

    started = time.perf_counter()
    surface, feature_columns = development_surface()
    if surface["forecast_kst_dtm"].ge(DEV_CUTOFF).any():
        raise RuntimeError("lockbox row reached shared-ordinal runner")
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
            pd.DataFrame(latent_values, columns=latent_columns, index=surface.index),
            multioutput,
        ],
        axis=1,
    )
    del base_matrix, latent_values, multioutput
    gc.collect()

    raw_bins = np.floor(
        (target.clip(0.10, 1.074999) - 0.10) / CLASS_WIDTH
    ).astype("Int64")
    active_bins = [
        int(value)
        for value in sorted(raw_bins.loc[training].dropna().astype(int).unique())
    ]
    bin_to_class = {bin_id: index for index, bin_id in enumerate(active_bins)}
    classes = raw_bins.map(bin_to_class).astype("Int64")
    class_count = len(active_bins)
    if class_count < 20:
        raise RuntimeError("ordinal class contract unexpectedly collapsed")
    centers = np.asarray(
        [
            target.loc[training & classes.eq(index)].mean()
            for index in range(class_count)
        ],
        dtype=float,
    )
    boundary_values = 0.5 * (centers[:-1] + centers[1:])

    selected = _screen(
        matrix,
        target,
        training,
        list(matrix.columns),
        TOP_FEATURES,
        20261900,
    )
    mandatory = [
        name
        for name in matrix.columns
        if name in {"group_id", "group_1", "group_2", "group_3"}
        or (name.startswith("pls__") and "prediction_p1" in name)
        or (name.startswith("multi_pls__") and "prediction" in name)
    ]
    selected = list(dict.fromkeys([*selected, *mandatory]))
    fit_base = matrix.loc[training, selected].to_numpy(dtype="float32")
    apply_base = matrix.loc[validation, selected].to_numpy(dtype="float32")
    del matrix
    gc.collect()

    expanded_fit, fit_labels, fit_weights, expansion_diagnostics = (
        _expanded_training(
            fit_base,
            classes.loc[training].to_numpy(dtype=np.int64),
            target.loc[training].to_numpy(dtype=float),
            boundary_values,
            args.sampling_mode,
            args.weight_mode,
        )
    )
    del fit_base
    gc.collect()
    model = LGBMClassifier(
        objective="binary",
        n_estimators=ITERATIONS,
        learning_rate=0.025,
        num_leaves=31,
        min_child_samples=240,
        max_bin=255,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=4.0,
        random_state=20261910,
        n_jobs=6,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )
    model.fit(expanded_fit, fit_labels, sample_weight=fit_weights)
    del expanded_fit, fit_labels, fit_weights
    gc.collect()

    expanded_apply = _expanded_application(apply_base, boundary_values)
    del apply_base
    gc.collect()
    raw_survival = model.predict_proba(expanded_apply)[:, 1].reshape(
        int(validation.sum()), class_count - 1
    )
    del expanded_apply, model
    gc.collect()
    probability = _ordered_probability(raw_survival)
    groups = surface.loc[validation, "group_id"].to_numpy(dtype=int)
    means = {
        group_id: float(
            target.loc[training & surface["group_id"].eq(group_id).to_numpy()].mean()
        )
        for group_id in CAPACITIES
    }
    normalized = _policy_values(probability, centers, groups, means)[FROZEN_POLICY]
    ordinal_output = _frame(surface, validation, normalized)

    parent = pd.read_parquet(PARENT_PATH)
    expected_keys = pd.MultiIndex.from_frame(
        surface.loc[validation, ["forecast_id", "group_id"]]
    )
    parent_keys = pd.MultiIndex.from_frame(parent[["forecast_id", "group_id"]])
    if len(parent) != int(validation.sum()) or not parent_keys.equals(expected_keys):
        raise RuntimeError("M197 parent key contract changed")
    parent_normalized = parent["prediction_kwh"].to_numpy(dtype=float) / parent[
        "group_id"
    ].map(CAPACITIES).to_numpy(dtype=float)
    output = _frame(surface, validation, 0.5 * parent_normalized + 0.5 * normalized)
    output["fold_id"] = args.fold
    output["model_id"] = args.candidate_id
    output_path = OUTPUT / f"{args.candidate_id}-{args.fold}.parquet"
    output.to_parquet(output_path, index=False)

    receipt = {
        "schema_version": 1,
        "candidate_id": args.candidate_id,
        "fold_id": args.fold,
        "architecture": "strict_pls_threshold_conditioned_shared_ordinal_half_m197",
        "scope": (
            "fixed official-data-only shared ordinal screen; outer Q3 labels "
            "excluded from PLS, screen, model, monotone projection, policy, and blend"
        ),
        "physical_development_cutoff": str(DEV_CUTOFF),
        "strict_training_rows": int(training.sum()),
        "class_width": CLASS_WIDTH,
        "class_count": class_count,
        "boundary_count": len(boundary_values),
        "sampling_mode": args.sampling_mode,
        "weight_mode": args.weight_mode,
        "iterations": ITERATIONS,
        "top_features": TOP_FEATURES,
        "selected_feature_count": len(selected),
        "selected_latent_feature_count": sum(
            name.startswith(("pls__", "multi_pls__")) for name in selected
        ),
        "selected_features": selected,
        "expansion_diagnostics": expansion_diagnostics,
        "multioutput_diagnostics": multioutput_diagnostics,
        "frozen_policy": FROZEN_POLICY,
        "raw_ordinal_score": _score(ordinal_output),
        "raw_ordinal_group_scores": _group_scores(ordinal_output),
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
                "raw_ordinal_score": receipt["raw_ordinal_score"],
                "raw_ordinal_group_scores": receipt["raw_ordinal_group_scores"],
                "fold_score": receipt["fold_score"],
                "group_scores": receipt["group_scores"],
                "selected_feature_count": receipt["selected_feature_count"],
                "selected_latent_feature_count": receipt[
                    "selected_latent_feature_count"
                ],
                "runtime_seconds": receipt["runtime_seconds"],
                "prediction_sha256": receipt["prediction_sha256"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
