"""Fit ordered label-smoothed strict PLS source-rank classifiers."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from run_correlation_wind_pls_rank import _correlation_union
from run_group_balanced_pls_rank import _frame, _group_scores
from run_inner_policy_classifier import _policy_values
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
from run_strict_prequential_source_rank import (
    FROZEN_MIXTURE,
    FROZEN_POLICY,
    _mixture,
)
from strict_dev_surface import DEV_CUTOFF, development_surface
from xgboost import XGBClassifier

CLASS_WIDTH = 0.02
CORRELATION_FEATURES_PER_GROUP = 80
ITERATIONS = 150
NEIGHBOR_MASS = 0.20
PARENT_PATH = OUTPUT / "M225_LOCAL_Q3_PROB_SMOOTH_BLEND-dev-2023-Q3.parquet"
ORIGINAL_PROBABILITY_PATH = (
    OUTPUT / "M212_STRICT_CORRELATION_WIND_PLS_Q3-dev-2023-Q3-probability.npz"
)
ORIGINAL_RECEIPT_PATH = (
    OUTPUT / "M212_STRICT_CORRELATION_WIND_PLS_Q3-dev-2023-Q3.json"
)


def _smoothed_probability(
    matrix: pd.DataFrame,
    sample_weight: pd.Series,
    classes: pd.Series,
    training: np.ndarray,
    validation: np.ndarray,
    selected: list[str],
    class_count: int,
    seed: int,
    source: str,
) -> np.ndarray:
    fit_matrix = matrix.loc[training, selected].to_numpy(dtype="float32")
    fit_classes = classes.loc[training].astype(int).to_numpy()
    fit_weights = sample_weight.loc[training].to_numpy(dtype=float)
    offsets = np.asarray([-1, 0, 1], dtype=int)
    label_weights = np.asarray(
        [NEIGHBOR_MASS, 1.0 - 2.0 * NEIGHBOR_MASS, NEIGHBOR_MASS],
        dtype=float,
    )
    expanded_matrix = np.repeat(fit_matrix, len(offsets), axis=0)
    expanded_classes = np.clip(
        np.repeat(fit_classes, len(offsets)) + np.tile(offsets, len(fit_classes)),
        0,
        class_count - 1,
    )
    expanded_weights = np.repeat(fit_weights, len(offsets)) * np.tile(
        label_weights,
        len(fit_weights),
    )
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=class_count,
        n_estimators=ITERATIONS,
        learning_rate=0.03,
        max_depth=5,
        min_child_weight=20.0,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=5.0,
        max_bin=256,
        tree_method="hist",
        random_state=seed,
        n_jobs=6,
    )
    model.fit(
        expanded_matrix,
        expanded_classes,
        sample_weight=expanded_weights,
    )
    raw = np.asarray(
        model.predict_proba(matrix.loc[validation, selected]),
        dtype=float,
    )
    learned = np.asarray(model.classes_, dtype=int)
    probability = np.zeros((int(validation.sum()), class_count), dtype=float)
    probability[:, learned] = raw
    probability /= probability.sum(axis=1, keepdims=True)
    print(
        json.dumps(
            {
                "source": source,
                "feature_count": len(selected),
                "strict_training_rows": int(training.sum()),
                "expanded_training_rows": len(expanded_classes),
                "validation_rows": int(validation.sum()),
            }
        ),
        flush=True,
    )
    del (
        model,
        raw,
        fit_matrix,
        expanded_matrix,
        expanded_classes,
        expanded_weights,
    )
    gc.collect()
    return probability


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--fold", choices=("dev-2023-Q3",), required=True)
    args = parser.parse_args()
    if _sha256(OPEN) != OPEN_SHA or _sha256(BASELINE) != BASELINE_SHA:
        raise RuntimeError("immutable competition input hash mismatch")
    required = (
        PARENT_PATH,
        ORIGINAL_PROBABILITY_PATH,
        ORIGINAL_RECEIPT_PATH,
        M195_RECEIPT,
        M195_LATENT,
    )
    if not all(path.exists() for path in required):
        raise RuntimeError("verified label-smoothing parent surface is missing")

    started = time.perf_counter()
    surface, feature_columns = development_surface()
    if surface["forecast_kst_dtm"].ge(DEV_CUTOFF).any():
        raise RuntimeError("lockbox row reached label-smoothed runner")
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
    original_receipt = json.loads(ORIGINAL_RECEIPT_PATH.read_text())
    if (
        _sha256(ORIGINAL_PROBABILITY_PATH)
        != original_receipt["probability_sha256"]
    ):
        raise RuntimeError("M212 probability checkpoint hash mismatch")

    with np.load(M195_LATENT, allow_pickle=False) as cached:
        latent_columns = [str(value) for value in cached["columns"].tolist()]
        latent_values = np.asarray(cached["values"], dtype="float32")
    if latent_values.shape != (len(surface), len(latent_columns)):
        raise RuntimeError("M195 latent checkpoint shape contract changed")
    base_matrix = surface[feature_columns].astype("float32")
    correlated, correlated_by_group = _correlation_union(
        base_matrix,
        surface,
        target,
        training,
        CORRELATION_FEATURES_PER_GROUP,
    )
    matrix = pd.concat(
        [
            base_matrix,
            pd.DataFrame(latent_values, columns=latent_columns, index=surface.index),
        ],
        axis=1,
    )
    del base_matrix, latent_values
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
    centers = np.asarray(
        [
            target.loc[training & classes.eq(index)].mean()
            for index in range(len(active_bins))
        ],
        dtype=float,
    )
    sample_weight = target.clip(lower=0.10).copy()
    additions = {
        "global": correlated,
        "gfs": [name for name in correlated if "gfs" in name.lower()],
        "ldaps": [name for name in correlated if "ldaps" in name.lower()],
    }
    selected_features = {
        source: list(
            dict.fromkeys(
                [*parent_receipt["selected_features"][source], *additions[source]]
            )
        )
        for source in ("global", "gfs", "ldaps")
    }
    smoothed_sources = {
        source: _smoothed_probability(
            matrix,
            sample_weight,
            classes,
            training,
            validation,
            selected_features[source],
            len(active_bins),
            20262900 + 10 * source_index,
            source,
        )
        for source_index, source in enumerate(("global", "gfs", "ldaps"))
    }
    del matrix
    gc.collect()

    smoothed_probability = _mixture(smoothed_sources, *FROZEN_MIXTURE)
    with np.load(ORIGINAL_PROBABILITY_PATH) as cache:
        original_probability = np.asarray(cache["probability"], dtype=float)
    if original_probability.shape != smoothed_probability.shape:
        raise RuntimeError("M212 label/probability shape contract changed")
    ensemble_probability = 0.5 * original_probability + 0.5 * smoothed_probability
    groups = surface.loc[validation, "group_id"].to_numpy(dtype=int)
    means = {
        group_id: float(
            target.loc[training & surface["group_id"].eq(group_id).to_numpy()].mean()
        )
        for group_id in CAPACITIES
    }
    smoothed_normalized = _policy_values(
        smoothed_probability,
        centers,
        groups,
        means,
    )[FROZEN_POLICY]
    ensemble_normalized = _policy_values(
        ensemble_probability,
        centers,
        groups,
        means,
    )[FROZEN_POLICY]
    smoothed_output = _frame(surface, validation, smoothed_normalized)
    ensemble_output = _frame(surface, validation, ensemble_normalized)

    parent = pd.read_parquet(PARENT_PATH)
    expected_keys = pd.MultiIndex.from_frame(
        surface.loc[validation, ["forecast_id", "group_id"]]
    )
    parent_keys = pd.MultiIndex.from_frame(parent[["forecast_id", "group_id"]])
    if len(parent) != int(validation.sum()) or not parent_keys.equals(expected_keys):
        raise RuntimeError("M225 parent key contract changed")
    capacity = parent["group_id"].map(CAPACITIES).to_numpy(dtype=float)
    parent_normalized = parent["prediction_kwh"].to_numpy(dtype=float) / capacity
    output = _frame(
        surface,
        validation,
        0.5 * parent_normalized + 0.5 * ensemble_normalized,
    )
    output["fold_id"] = args.fold
    output["model_id"] = args.candidate_id
    output_path = OUTPUT / f"{args.candidate_id}-{args.fold}.parquet"
    probability_path = OUTPUT / f"{args.candidate_id}-{args.fold}-probability.npz"
    output.to_parquet(output_path, index=False)
    np.savez(
        probability_path,
        probability=ensemble_probability.astype("float32"),
        centers=centers.astype("float32"),
    )

    receipt = {
        "schema_version": 1,
        "candidate_id": args.candidate_id,
        "fold_id": args.fold,
        "architecture": "strict_ordered_label_smoothed_pls_distribution_half_m225",
        "scope": (
            "official-data-only ordered adjacent-bin label smoothing; outer Q3 labels "
            "excluded from fit, fixed probability ensemble, policy, and output blend"
        ),
        "physical_development_cutoff": str(DEV_CUTOFF),
        "strict_training_rows": int(training.sum()),
        "expanded_training_rows_per_source": int(training.sum()) * 3,
        "class_width": CLASS_WIDTH,
        "class_count": len(active_bins),
        "ordered_label_weights": {
            "lower": NEIGHBOR_MASS,
            "current": 1.0 - 2.0 * NEIGHBOR_MASS,
            "upper": NEIGHBOR_MASS,
        },
        "correlation_features_per_group": CORRELATION_FEATURES_PER_GROUP,
        "correlation_union_count": len(correlated),
        "correlation_features_by_group": correlated_by_group,
        "selected_feature_counts": {
            source: len(names) for source, names in selected_features.items()
        },
        "frozen_mixture": {
            "global_weight": FROZEN_MIXTURE[0],
            "gfs_share": FROZEN_MIXTURE[1],
        },
        "frozen_policy": FROZEN_POLICY,
        "smoothed_score": _score(smoothed_output),
        "smoothed_group_scores": _group_scores(smoothed_output),
        "probability_ensemble_score": _score(ensemble_output),
        "probability_ensemble_group_scores": _group_scores(ensemble_output),
        "fixed_parent_weight": 0.5,
        "fold_score": _score(output),
        "group_scores": _group_scores(output),
        "runtime_seconds": round(time.perf_counter() - started, 2),
        "prediction_path": str(output_path.relative_to(Path.cwd())),
        "prediction_sha256": _sha256(output_path),
        "probability_path": str(probability_path.relative_to(Path.cwd())),
        "probability_sha256": _sha256(probability_path),
        "parent_path": str(PARENT_PATH.relative_to(Path.cwd())),
        "parent_sha256": _sha256(PARENT_PATH),
        "original_probability_path": str(
            ORIGINAL_PROBABILITY_PATH.relative_to(Path.cwd())
        ),
        "original_probability_sha256": _sha256(ORIGINAL_PROBABILITY_PATH),
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
                "smoothed_score": receipt["smoothed_score"],
                "smoothed_group_scores": receipt["smoothed_group_scores"],
                "probability_ensemble_score": receipt["probability_ensemble_score"],
                "probability_ensemble_group_scores": receipt[
                    "probability_ensemble_group_scores"
                ],
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
