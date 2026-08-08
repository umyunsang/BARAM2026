"""Ensemble original and half-bin-shifted strict PLS target partitions."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from run_correlation_wind_pls_rank import _correlation_union, _probability
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

CLASS_WIDTH = 0.02
ORIGINAL_ORIGIN = 0.10
SHIFTED_ORIGIN = 0.09
CORRELATION_FEATURES_PER_GROUP = 80
PARENT_PATH = OUTPUT / "M222_LOCAL_Q3_POINT_ANCHOR_BLEND-dev-2023-Q3.parquet"
ORIGINAL_PROBABILITY_PATH = (
    OUTPUT / "M212_STRICT_CORRELATION_WIND_PLS_Q3-dev-2023-Q3-probability.npz"
)
ORIGINAL_RECEIPT_PATH = (
    OUTPUT / "M212_STRICT_CORRELATION_WIND_PLS_Q3-dev-2023-Q3.json"
)


def _class_contract(
    target: pd.Series,
    training: np.ndarray,
    origin: float,
) -> tuple[pd.Series, np.ndarray, list[int]]:
    raw_bins = np.floor(
        (target.clip(0.10, 1.074999) - origin) / CLASS_WIDTH
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
    return classes, centers, active_bins


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
        raise RuntimeError("verified shifted-bin parent surface is missing")

    started = time.perf_counter()
    surface, feature_columns = development_surface()
    if surface["forecast_kst_dtm"].ge(DEV_CUTOFF).any():
        raise RuntimeError("lockbox row reached shifted-bin runner")
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

    shifted_classes, shifted_centers, shifted_bins = _class_contract(
        target,
        training,
        SHIFTED_ORIGIN,
    )
    _, original_centers, original_bins = _class_contract(
        target,
        training,
        ORIGINAL_ORIGIN,
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
    shifted_sources = {
        source: _probability(
            matrix,
            sample_weight,
            shifted_classes,
            training,
            validation,
            selected_features[source],
            len(shifted_bins),
            20262800 + 10 * source_index,
            source,
        )
        for source_index, source in enumerate(("global", "gfs", "ldaps"))
    }
    del matrix
    gc.collect()

    shifted_probability = _mixture(shifted_sources, *FROZEN_MIXTURE)
    with np.load(ORIGINAL_PROBABILITY_PATH) as cache:
        original_probability = np.asarray(cache["probability"], dtype=float)
    if original_probability.shape != (int(validation.sum()), len(original_bins)):
        raise RuntimeError("M212 class/probability shape contract changed")
    combined_probability = np.concatenate(
        [0.5 * original_probability, 0.5 * shifted_probability],
        axis=1,
    )
    combined_centers = np.concatenate([original_centers, shifted_centers])

    groups = surface.loc[validation, "group_id"].to_numpy(dtype=int)
    means = {
        group_id: float(
            target.loc[training & surface["group_id"].eq(group_id).to_numpy()].mean()
        )
        for group_id in CAPACITIES
    }
    shifted_normalized = _policy_values(
        shifted_probability,
        shifted_centers,
        groups,
        means,
    )[FROZEN_POLICY]
    combined_normalized = _policy_values(
        combined_probability,
        combined_centers,
        groups,
        means,
    )[FROZEN_POLICY]
    shifted_output = _frame(surface, validation, shifted_normalized)
    combined_output = _frame(surface, validation, combined_normalized)

    parent = pd.read_parquet(PARENT_PATH)
    expected_keys = pd.MultiIndex.from_frame(
        surface.loc[validation, ["forecast_id", "group_id"]]
    )
    parent_keys = pd.MultiIndex.from_frame(parent[["forecast_id", "group_id"]])
    if len(parent) != int(validation.sum()) or not parent_keys.equals(expected_keys):
        raise RuntimeError("M222 parent key contract changed")
    capacity = parent["group_id"].map(CAPACITIES).to_numpy(dtype=float)
    parent_normalized = parent["prediction_kwh"].to_numpy(dtype=float) / capacity
    output = _frame(
        surface,
        validation,
        0.5 * parent_normalized + 0.5 * combined_normalized,
    )
    output["fold_id"] = args.fold
    output["model_id"] = args.candidate_id
    output_path = OUTPUT / f"{args.candidate_id}-{args.fold}.parquet"
    probability_path = OUTPUT / f"{args.candidate_id}-{args.fold}-probability.npz"
    output.to_parquet(output_path, index=False)
    np.savez(
        probability_path,
        probability=combined_probability.astype("float32"),
        centers=combined_centers.astype("float32"),
    )

    receipt = {
        "schema_version": 1,
        "candidate_id": args.candidate_id,
        "fold_id": args.fold,
        "architecture": "strict_half_bin_shift_pls_distribution_half_m222",
        "scope": (
            "fixed official-data-only half-bin target-partition ensemble; outer Q3 "
            "labels excluded from partition, fit, probability blend, policy, and output blend"
        ),
        "physical_development_cutoff": str(DEV_CUTOFF),
        "strict_training_rows": int(training.sum()),
        "class_width": CLASS_WIDTH,
        "original_origin": ORIGINAL_ORIGIN,
        "shifted_origin": SHIFTED_ORIGIN,
        "original_class_count": len(original_bins),
        "shifted_class_count": len(shifted_bins),
        "partition_weights": {"original": 0.5, "shifted": 0.5},
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
        "shifted_score": _score(shifted_output),
        "shifted_group_scores": _group_scores(shifted_output),
        "partition_ensemble_score": _score(combined_output),
        "partition_ensemble_group_scores": _group_scores(combined_output),
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
                "shifted_score": receipt["shifted_score"],
                "shifted_group_scores": receipt["shifted_group_scores"],
                "partition_ensemble_score": receipt["partition_ensemble_score"],
                "partition_ensemble_group_scores": receipt[
                    "partition_ensemble_group_scores"
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
