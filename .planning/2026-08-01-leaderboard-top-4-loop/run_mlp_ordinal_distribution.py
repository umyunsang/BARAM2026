"""Fit a shared neural ordinal survival distribution on strict pre-Q3 history."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from run_inner_policy_classifier import _group_total, _policy_values
from run_mlp_distribution import HIDDEN_LAYERS, SEEDS, _screen_features
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
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from strict_dev_surface import DEV_CUTOFF, development_surface

BASE_COLUMNS = ["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh"]
CLASS_WIDTH = 0.025
FROZEN_POLICY = "T0.4_G2"
PARENT_PATH = OUTPUT / "M189_STRICT_PREQUENTIAL_SOURCE_RANK_Q3-dev-2023-Q3.parquet"


def _survival_to_probability(raw: np.ndarray) -> np.ndarray:
    survival = np.minimum.accumulate(np.clip(raw, 1e-8, 1.0 - 1e-8), axis=1)
    extended = np.column_stack(
        [np.ones(len(raw)), survival, np.zeros(len(raw))]
    )
    probability = np.maximum(extended[:, :-1] - extended[:, 1:], 1e-12)
    probability /= probability.sum(axis=1, keepdims=True)
    return probability


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
    parser.add_argument("--feature-count", type=int, default=256)
    args = parser.parse_args()
    if not 160 <= args.feature_count <= 384:
        raise ValueError("feature count must be between 160 and 384")
    if _sha256(OPEN) != OPEN_SHA or _sha256(BASELINE) != BASELINE_SHA:
        raise RuntimeError("immutable competition input hash mismatch")

    started = time.perf_counter()
    surface, feature_columns = development_surface()
    if surface["forecast_kst_dtm"].ge(DEV_CUTOFF).any():
        raise RuntimeError("lockbox row reached neural ordinal runner")
    validation = _validation_mask(surface, args.fold)
    history = _strict_preceding_mask(surface, validation)
    target = surface["actual_kwh"] / surface["group_id"].map(CAPACITIES)
    training = (
        history
        & surface["actual_kwh"].notna().to_numpy()
        & target.ge(0.10).to_numpy()
    )
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
    centers = np.asarray(
        [
            target.loc[training & classes.eq(index)].mean()
            for index in range(class_count)
        ],
        dtype=float,
    )

    raw_matrix = surface[feature_columns].astype("float32")
    selected = _screen_features(raw_matrix, target, training, args.feature_count)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    fit_matrix = imputer.fit_transform(raw_matrix.loc[training, selected])
    apply_matrix = imputer.transform(raw_matrix.loc[validation, selected])
    scaler = StandardScaler()
    fit_matrix = scaler.fit_transform(fit_matrix).astype("float32", copy=False)
    apply_matrix = scaler.transform(apply_matrix).astype("float32", copy=False)
    del raw_matrix
    gc.collect()

    train_class = classes.loc[training].astype(int).to_numpy()
    ordinal_target = np.column_stack(
        [train_class > boundary for boundary in range(class_count - 1)]
    ).astype("int8")
    sample_weight = target.loc[training].clip(lower=0.10).to_numpy(dtype=float)
    survivals: list[np.ndarray] = []
    model_diagnostics: list[dict[str, object]] = []
    for seed in SEEDS:
        model = MLPClassifier(
            hidden_layer_sizes=HIDDEN_LAYERS,
            activation="relu",
            solver="adam",
            alpha=0.001,
            batch_size=256,
            learning_rate="adaptive",
            learning_rate_init=0.0005,
            max_iter=140,
            shuffle=True,
            random_state=seed + 100,
            tol=1e-4,
            early_stopping=False,
            n_iter_no_change=20,
        )
        model.fit(fit_matrix, ordinal_target, sample_weight=sample_weight)
        survival = np.asarray(model.predict_proba(apply_matrix), dtype=float)
        if survival.shape != (int(validation.sum()), class_count - 1):
            raise RuntimeError("neural ordinal output contract changed")
        survivals.append(survival)
        model_diagnostics.append(
            {
                "seed": seed + 100,
                "iterations": int(model.n_iter_),
                "loss": float(model.loss_),
            }
        )
        print(json.dumps({"ordinal_mlp": model_diagnostics[-1]}), flush=True)
        del model
        gc.collect()
    probability = _survival_to_probability(np.mean(survivals, axis=0))

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
        raise RuntimeError("M189 parent key contract changed")
    parent_normalized = parent["prediction_kwh"].to_numpy(dtype=float) / parent[
        "group_id"
    ].map(CAPACITIES).to_numpy(dtype=float)
    blend_output = _frame(surface, validation, 0.5 * normalized + 0.5 * parent_normalized)

    output = ordinal_output.assign(fold_id=args.fold, model_id=args.candidate_id)
    output_path = OUTPUT / f"{args.candidate_id}-{args.fold}.parquet"
    output.to_parquet(output_path, index=False)
    receipt = {
        "schema_version": 1,
        "candidate_id": args.candidate_id,
        "fold_id": args.fold,
        "architecture": "strict_shared_neural_cumulative_ordinal_survival",
        "scope": (
            "fixed official-data-only ordered-distribution screen; outer Q3 labels "
            "excluded from feature selection, preprocessing, and fit"
        ),
        "physical_development_cutoff": str(DEV_CUTOFF),
        "strict_training_rows": int(training.sum()),
        "class_width": CLASS_WIDTH,
        "class_count": class_count,
        "feature_count": len(selected),
        "selected_features": selected,
        "hidden_layers": list(HIDDEN_LAYERS),
        "model_diagnostics": model_diagnostics,
        "frozen_policy": FROZEN_POLICY,
        "fold_score": _score(ordinal_output),
        "group_scores": _group_scores(ordinal_output),
        "fixed_half_m189_blend_score": _score(blend_output),
        "fixed_half_m189_group_scores": _group_scores(blend_output),
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
                "fold_score": receipt["fold_score"],
                "group_scores": receipt["group_scores"],
                "fixed_half_m189_blend_score": receipt[
                    "fixed_half_m189_blend_score"
                ],
                "runtime_seconds": receipt["runtime_seconds"],
                "prediction_sha256": receipt["prediction_sha256"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
