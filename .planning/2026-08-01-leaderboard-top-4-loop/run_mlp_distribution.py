"""Screen a strict-history neural conditional generation distribution on Q3.

The target is discretized exactly as in the source-rank classifiers, but the
representation is a standardized multilayer perceptron rather than another
tree ensemble.  Feature screening, imputation, scaling, and model fitting use
only complete issuance batches observable before the outer validation fold.
The policy and architecture are frozen constants; Q3 labels are used only for
the final diagnostic score.
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
from run_inner_policy_classifier import _group_total, _policy_values
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
CLASS_WIDTH = 0.02
FROZEN_POLICY = "T0.4_G2"
HIDDEN_LAYERS = (256, 128)
SEEDS = (20261010, 20261011, 20261012)
PARENT_PATH = OUTPUT / "M189_STRICT_PREQUENTIAL_SOURCE_RANK_Q3-dev-2023-Q3.parquet"


def _screen_features(
    matrix: pd.DataFrame,
    target: pd.Series,
    training: np.ndarray,
    feature_count: int,
) -> list[str]:
    model = LGBMRegressor(
        objective="l1",
        n_estimators=180,
        learning_rate=0.035,
        num_leaves=31,
        min_child_samples=60,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=3.0,
        random_state=20261000,
        n_jobs=6,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )
    model.fit(
        matrix.loc[training],
        target.loc[training],
        sample_weight=target.loc[training].clip(lower=0.10),
    )
    gain = model.booster_.feature_importance(importance_type="gain")
    order = np.argsort(gain)[::-1][:feature_count]
    selected = [matrix.columns[position] for position in order]
    del model
    gc.collect()
    return selected


def _align_probability(
    model: MLPClassifier,
    matrix: np.ndarray,
    class_count: int,
) -> np.ndarray:
    raw = np.asarray(model.predict_proba(matrix), dtype=float)
    learned = np.asarray(model.classes_, dtype=int)
    probability = np.zeros((len(matrix), class_count), dtype=float)
    probability[:, learned] = raw
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
    parser.add_argument("--feature-count", type=int, default=256)
    args = parser.parse_args()
    if not 160 <= args.feature_count <= 384:
        raise ValueError("feature count must be between 160 and 384")
    if _sha256(OPEN) != OPEN_SHA or _sha256(BASELINE) != BASELINE_SHA:
        raise RuntimeError("immutable competition input hash mismatch")

    started = time.perf_counter()
    surface, feature_columns = development_surface()
    if surface["forecast_kst_dtm"].ge(DEV_CUTOFF).any():
        raise RuntimeError("lockbox row reached MLP distribution runner")
    validation = _validation_mask(surface, args.fold)
    history = _strict_preceding_mask(surface, validation)
    normalized_target = surface["actual_kwh"] / surface["group_id"].map(CAPACITIES)
    training = (
        history
        & surface["actual_kwh"].notna().to_numpy()
        & normalized_target.ge(0.10).to_numpy()
    )
    if int(training.sum()) < 8000:
        raise RuntimeError("strict MLP history is unexpectedly small")

    raw_bins = np.floor(
        (normalized_target.clip(0.10, 1.074999) - 0.10) / CLASS_WIDTH
    ).astype("Int64")
    active_bins = [
        int(value)
        for value in sorted(raw_bins.loc[training].dropna().astype(int).unique())
    ]
    bin_to_class = {bin_id: index for index, bin_id in enumerate(active_bins)}
    classes = raw_bins.map(bin_to_class).astype("Int64")
    centers = np.asarray(
        [
            normalized_target.loc[training & classes.eq(index)].mean()
            for index in range(len(active_bins))
        ],
        dtype=float,
    )

    raw_matrix = surface[feature_columns].astype("float32")
    selected = _screen_features(
        raw_matrix,
        normalized_target,
        training,
        args.feature_count,
    )
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    fit_matrix = imputer.fit_transform(raw_matrix.loc[training, selected])
    apply_matrix = imputer.transform(raw_matrix.loc[validation, selected])
    scaler = StandardScaler()
    fit_matrix = scaler.fit_transform(fit_matrix).astype("float32", copy=False)
    apply_matrix = scaler.transform(apply_matrix).astype("float32", copy=False)
    del raw_matrix
    gc.collect()

    probabilities: list[np.ndarray] = []
    model_diagnostics: list[dict[str, object]] = []
    sample_weight = normalized_target.loc[training].clip(lower=0.10).to_numpy(float)
    train_classes = classes.loc[training].astype(int).to_numpy()
    for seed in SEEDS:
        model = MLPClassifier(
            hidden_layer_sizes=HIDDEN_LAYERS,
            activation="relu",
            solver="adam",
            alpha=0.001,
            batch_size=256,
            learning_rate="adaptive",
            learning_rate_init=0.0005,
            max_iter=250,
            shuffle=True,
            random_state=seed,
            tol=1e-4,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
        )
        model.fit(fit_matrix, train_classes, sample_weight=sample_weight)
        probabilities.append(
            _align_probability(model, apply_matrix, len(active_bins))
        )
        model_diagnostics.append(
            {
                "seed": seed,
                "iterations": int(model.n_iter_),
                "loss": float(model.loss_),
                "best_validation_score": float(model.best_validation_score_),
            }
        )
        print(json.dumps({"mlp": model_diagnostics[-1]}), flush=True)
        del model
        gc.collect()
    probability = np.mean(probabilities, axis=0)
    probability /= probability.sum(axis=1, keepdims=True)

    groups = surface.loc[validation, "group_id"].to_numpy(dtype=int)
    means = {
        group_id: float(
            normalized_target.loc[
                training & surface["group_id"].eq(group_id).to_numpy()
            ].mean()
        )
        for group_id in CAPACITIES
    }
    policy_values = _policy_values(probability, centers, groups, means)
    fixed_normalized = policy_values[FROZEN_POLICY]
    mlp_output = _frame(surface, validation, fixed_normalized)

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
    blend_normalized = 0.5 * fixed_normalized + 0.5 * parent_normalized
    blend_output = _frame(surface, validation, blend_normalized)

    output = mlp_output.assign(fold_id=args.fold, model_id=args.candidate_id)
    output_path = OUTPUT / f"{args.candidate_id}-{args.fold}.parquet"
    output.to_parquet(output_path, index=False)
    receipt = {
        "schema_version": 1,
        "candidate_id": args.candidate_id,
        "fold_id": args.fold,
        "architecture": "standardized_multiclass_mlp_three_seed_probability_mean",
        "scope": (
            "fixed-architecture official-data-only representation screen; "
            "outer Q3 labels excluded from feature selection, preprocessing, and fit"
        ),
        "physical_development_cutoff": str(DEV_CUTOFF),
        "strict_training_rows": int(training.sum()),
        "strict_training_rows_by_group": {
            str(group_id): int(
                (training & surface["group_id"].eq(group_id).to_numpy()).sum()
            )
            for group_id in CAPACITIES
        },
        "class_width": CLASS_WIDTH,
        "class_count": len(active_bins),
        "feature_count": len(selected),
        "selected_features": selected,
        "hidden_layers": list(HIDDEN_LAYERS),
        "seeds": list(SEEDS),
        "model_diagnostics": model_diagnostics,
        "frozen_policy": FROZEN_POLICY,
        "fold_score": _score(mlp_output),
        "group_scores": _group_scores(mlp_output),
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
