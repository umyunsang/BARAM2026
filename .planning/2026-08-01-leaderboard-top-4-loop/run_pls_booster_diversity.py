"""Fit LightGBM/CatBoost distributions over the verified PLS latent surface."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
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
TOP_FEATURES = 240
ITERATIONS = 180
PARENT_PATH = OUTPUT / "M197_STRICT_FIXED_HALF_PLS-dev-2023-Q3.parquet"


def _align_probability(
    raw: np.ndarray,
    learned: np.ndarray,
    rows: int,
    class_count: int,
) -> np.ndarray:
    probability = np.zeros((rows, class_count), dtype=float)
    probability[:, learned.astype(int)] = np.asarray(raw, dtype=float)
    probability = np.clip(probability, 1e-12, None)
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
    args = parser.parse_args()
    if _sha256(OPEN) != OPEN_SHA or _sha256(BASELINE) != BASELINE_SHA:
        raise RuntimeError("immutable competition input hash mismatch")
    if not PARENT_PATH.exists() or not M195_RECEIPT.exists() or not M195_LATENT.exists():
        raise RuntimeError("verified PLS parent surface is missing")

    started = time.perf_counter()
    surface, feature_columns = development_surface()
    if surface["forecast_kst_dtm"].ge(DEV_CUTOFF).any():
        raise RuntimeError("lockbox row reached PLS booster-diversity runner")
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
    selected = _screen(
        matrix,
        target,
        training,
        list(matrix.columns),
        TOP_FEATURES,
        20261800,
    )
    fit_matrix = matrix.loc[training, selected]
    apply_matrix = matrix.loc[validation, selected]
    sample_weight = target.loc[training].clip(lower=0.10).to_numpy(dtype=float)
    train_class = classes.loc[training].astype(int)

    lgbm = LGBMClassifier(
        objective="multiclass",
        num_class=class_count,
        n_estimators=ITERATIONS,
        learning_rate=0.025,
        num_leaves=15,
        min_child_samples=70,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=3.0,
        random_state=20261810,
        n_jobs=6,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )
    lgbm.fit(fit_matrix, train_class, sample_weight=sample_weight)
    lgbm_probability = _align_probability(
        lgbm.predict_proba(apply_matrix),
        np.asarray(lgbm.classes_),
        int(validation.sum()),
        class_count,
    )
    del lgbm
    gc.collect()

    catboost = CatBoostClassifier(
        loss_function="MultiClass",
        iterations=ITERATIONS,
        learning_rate=0.03,
        depth=7,
        l2_leaf_reg=5.0,
        random_strength=0.2,
        bootstrap_type="Bernoulli",
        subsample=0.9,
        random_seed=20261820,
        thread_count=6,
        allow_writing_files=False,
        verbose=False,
    )
    catboost.fit(fit_matrix, train_class, sample_weight=sample_weight)
    catboost_probability = _align_probability(
        catboost.predict_proba(apply_matrix),
        np.asarray(catboost.classes_),
        int(validation.sum()),
        class_count,
    )
    del catboost, fit_matrix, apply_matrix, matrix
    gc.collect()

    groups = surface.loc[validation, "group_id"].to_numpy(dtype=int)
    means = {
        group_id: float(
            target.loc[training & surface["group_id"].eq(group_id).to_numpy()].mean()
        )
        for group_id in CAPACITIES
    }
    probabilities = {
        "lgbm": lgbm_probability,
        "catboost": catboost_probability,
        "fixed_average": 0.5 * lgbm_probability + 0.5 * catboost_probability,
    }
    booster_outputs: dict[str, pd.DataFrame] = {}
    booster_scores: dict[str, dict[str, float]] = {}
    for name, probability in probabilities.items():
        normalized = _policy_values(probability, centers, groups, means)[FROZEN_POLICY]
        booster_outputs[name] = _frame(surface, validation, normalized)
        booster_scores[name] = _score(booster_outputs[name])

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
    booster_normalized = booster_outputs["fixed_average"][
        "prediction_kwh"
    ].to_numpy(dtype=float) / booster_outputs["fixed_average"]["group_id"].map(
        CAPACITIES
    ).to_numpy(dtype=float)
    output = _frame(
        surface,
        validation,
        0.5 * parent_normalized + 0.5 * booster_normalized,
    )
    output["fold_id"] = args.fold
    output["model_id"] = args.candidate_id
    output_path = OUTPUT / f"{args.candidate_id}-{args.fold}.parquet"
    output.to_parquet(output_path, index=False)
    receipt = {
        "schema_version": 1,
        "candidate_id": args.candidate_id,
        "fold_id": args.fold,
        "architecture": "strict_pls_lgbm_catboost_probability_average_half_m197",
        "scope": (
            "fixed official-data-only booster-diversity screen; outer Q3 labels "
            "excluded from feature/model/policy fit and both 50:50 weights"
        ),
        "physical_development_cutoff": str(DEV_CUTOFF),
        "training_rows": int(training.sum()),
        "class_width": CLASS_WIDTH,
        "class_count": class_count,
        "top_features": TOP_FEATURES,
        "iterations": ITERATIONS,
        "selected_features": selected,
        "selected_pls_feature_count": sum(
            name.startswith(("pls__", "mpls__")) for name in selected
        ),
        "multioutput_diagnostics": multioutput_diagnostics,
        "frozen_policy": FROZEN_POLICY,
        "booster_scores": booster_scores,
        "fixed_average_group_scores": _group_scores(
            booster_outputs["fixed_average"]
        ),
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
                "booster_scores": booster_scores,
                "fixed_average_group_scores": receipt["fixed_average_group_scores"],
                "fold_score": receipt["fold_score"],
                "group_scores": receipt["group_scores"],
                "selected_pls_feature_count": receipt[
                    "selected_pls_feature_count"
                ],
                "runtime_seconds": receipt["runtime_seconds"],
                "prediction_sha256": receipt["prediction_sha256"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
