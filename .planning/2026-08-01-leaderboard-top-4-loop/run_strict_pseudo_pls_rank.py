"""Fit a strict PLS group-3 head with donor-derived 2022 pseudo targets."""

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
from run_pseudo_group3_classifier import (
    GROUP_ID,
    _pseudo_season_weights,
    _pseudo_targets,
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
from run_strict_prequential_source_rank import (
    FROZEN_MIXTURE,
    FROZEN_POLICY,
    _mixture,
)
from strict_dev_surface import DEV_CUTOFF, development_surface

CLASS_WIDTH = 0.02
CORRELATION_FEATURES_PER_GROUP = 80
PSEUDO_WEIGHT = 0.20
PSEUDO_SEASON_BANDWIDTH_DAYS = 45.0
PSEUDO_PROBABILITY_WEIGHT = 0.50
PARENT_OUTPUT_WEIGHT = 0.50
PARENT_PATH = OUTPUT / "M225_LOCAL_Q3_PROB_SMOOTH_BLEND-dev-2023-Q3.parquet"
BASE_PROBABILITY_PATH = (
    OUTPUT / "M212_STRICT_CORRELATION_WIND_PLS_Q3-dev-2023-Q3-probability.npz"
)
BASE_RECEIPT_PATH = (
    OUTPUT / "M212_STRICT_CORRELATION_WIND_PLS_Q3-dev-2023-Q3.json"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--fold", choices=("dev-2023-Q3",), required=True)
    args = parser.parse_args()
    if _sha256(OPEN) != OPEN_SHA or _sha256(BASELINE) != BASELINE_SHA:
        raise RuntimeError("immutable competition input hash mismatch")
    required = (
        PARENT_PATH,
        BASE_PROBABILITY_PATH,
        BASE_RECEIPT_PATH,
        M195_RECEIPT,
        M195_LATENT,
    )
    if not all(path.exists() for path in required):
        raise RuntimeError("verified strict pseudo parent surface is missing")

    started = time.perf_counter()
    surface, feature_columns = development_surface()
    if surface["forecast_kst_dtm"].ge(DEV_CUTOFF).any():
        raise RuntimeError("lockbox row reached strict pseudo runner")
    validation = _validation_mask(surface, args.fold)
    history = _strict_preceding_mask(surface, validation)
    normalized_target = surface["actual_kwh"] / surface["group_id"].map(CAPACITIES)
    observed_training = (
        history
        & surface["actual_kwh"].notna().to_numpy()
        & normalized_target.ge(0.10).to_numpy()
    )

    parent_receipt = json.loads(M195_RECEIPT.read_text())
    if _sha256(M195_LATENT) != parent_receipt["latent_checkpoint_sha256"]:
        raise RuntimeError("M195 latent checkpoint hash mismatch")
    base_receipt = json.loads(BASE_RECEIPT_PATH.read_text())
    if _sha256(BASE_PROBABILITY_PATH) != base_receipt["probability_sha256"]:
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
        normalized_target,
        observed_training,
        CORRELATION_FEATURES_PER_GROUP,
    )
    matrix = pd.concat(
        [
            base_matrix,
            pd.DataFrame(latent_values, columns=latent_columns, index=surface.index),
        ],
        axis=1,
    )
    del latent_values
    gc.collect()

    pseudo, pseudo_confidence, pseudo_diagnostics = _pseudo_targets(
        surface,
        base_matrix,
        history,
        "compact",
        1.0,
    )
    extended_target = normalized_target.copy()
    extended_target.loc[pseudo.notna()] = pseudo.loc[pseudo.notna()]
    pseudo_mask = pseudo.notna().to_numpy()
    group3 = surface["group_id"].eq(GROUP_ID).to_numpy()
    extended_training = (
        history
        & group3
        & extended_target.ge(0.10).to_numpy()
        & (surface["actual_kwh"].notna().to_numpy() | pseudo_mask)
    )
    season_weights, season_diagnostics = _pseudo_season_weights(
        surface,
        extended_training,
        pseudo_mask,
        validation,
        PSEUDO_SEASON_BANDWIDTH_DAYS,
    )

    observed_bins = np.floor(
        (normalized_target.clip(0.10, 1.074999) - 0.10) / CLASS_WIDTH
    ).astype("Int64")
    active_bins = [
        int(value)
        for value in sorted(
            observed_bins.loc[observed_training].dropna().astype(int).unique()
        )
    ]
    bin_to_class = {bin_id: index for index, bin_id in enumerate(active_bins)}
    extended_bins = np.floor(
        (extended_target.clip(0.10, 1.074999) - 0.10) / CLASS_WIDTH
    ).astype("Int64")
    unexpected_bins = set(
        extended_bins.loc[extended_training].dropna().astype(int).unique()
    ).difference(active_bins)
    if unexpected_bins:
        raise RuntimeError(f"pseudo bins outside observed support: {unexpected_bins}")
    classes = extended_bins.map(bin_to_class).astype("Int64")
    observed_classes = observed_bins.map(bin_to_class).astype("Int64")
    centers = np.asarray(
        [
            normalized_target.loc[
                observed_training & observed_classes.eq(index)
            ].mean()
            for index in range(len(active_bins))
        ],
        dtype=float,
    )
    sample_weight = extended_target.clip(lower=0.10).copy()
    selected_positions = np.flatnonzero(extended_training)
    selected_pseudo = pseudo_mask[extended_training]
    weight_values = sample_weight.iloc[selected_positions].to_numpy(dtype=float)
    weight_values *= np.where(selected_pseudo, PSEUDO_WEIGHT, 1.0)
    weight_values *= pseudo_confidence.iloc[selected_positions].to_numpy(dtype=float)
    weight_values *= season_weights
    sample_weight.iloc[selected_positions] = weight_values

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
    validation_group3 = validation & group3
    pseudo_sources = {
        source: _probability(
            matrix,
            sample_weight,
            classes,
            extended_training,
            validation_group3,
            selected_features[source],
            len(active_bins),
            20263100 + 10 * source_index,
            source,
        )
        for source_index, source in enumerate(("global", "gfs", "ldaps"))
    }
    del matrix, base_matrix
    gc.collect()

    pseudo_probability = _mixture(pseudo_sources, *FROZEN_MIXTURE)
    with np.load(BASE_PROBABILITY_PATH) as cache:
        base_probability = np.asarray(cache["probability"], dtype=float)
    if base_probability.shape != (int(validation.sum()), len(active_bins)):
        raise RuntimeError("M212 probability/class contract changed")
    validation_groups = surface.loc[validation, "group_id"].to_numpy(dtype=int)
    validation_group3_positions = validation_groups == GROUP_ID
    if int(validation_group3_positions.sum()) != len(pseudo_probability):
        raise RuntimeError("group-3 pseudo probability row contract changed")
    combined_probability = base_probability.copy()
    combined_probability[validation_group3_positions] = (
        (1.0 - PSEUDO_PROBABILITY_WEIGHT)
        * base_probability[validation_group3_positions]
        + PSEUDO_PROBABILITY_WEIGHT * pseudo_probability
    )
    combined_probability /= combined_probability.sum(axis=1, keepdims=True)
    means = {
        group_id: float(
            normalized_target.loc[
                observed_training & surface["group_id"].eq(group_id).to_numpy()
            ].mean()
        )
        for group_id in CAPACITIES
    }
    normalized = _policy_values(
        combined_probability,
        centers,
        validation_groups,
        means,
    )[FROZEN_POLICY]
    raw_output = _frame(surface, validation, normalized)

    parent = pd.read_parquet(PARENT_PATH)
    expected_keys = pd.MultiIndex.from_frame(
        surface.loc[validation, ["forecast_id", "group_id"]]
    )
    parent_keys = pd.MultiIndex.from_frame(parent[["forecast_id", "group_id"]])
    if len(parent) != int(validation.sum()) or not parent_keys.equals(expected_keys):
        raise RuntimeError("M225 parent key contract changed")
    capacity = parent["group_id"].map(CAPACITIES).to_numpy(dtype=float)
    parent_normalized = parent["prediction_kwh"].to_numpy(dtype=float) / capacity
    output_normalized = parent_normalized.copy()
    output_normalized[validation_group3_positions] = (
        PARENT_OUTPUT_WEIGHT * parent_normalized[validation_group3_positions]
        + (1.0 - PARENT_OUTPUT_WEIGHT) * normalized[validation_group3_positions]
    )
    output = _frame(surface, validation, output_normalized)
    output["fold_id"] = args.fold
    output["model_id"] = args.candidate_id
    output_path = OUTPUT / f"{args.candidate_id}-{args.fold}.parquet"
    probability_path = OUTPUT / f"{args.candidate_id}-{args.fold}-probability.npz"
    output.to_parquet(output_path, index=False)
    np.savez(
        probability_path,
        probability=combined_probability.astype("float32"),
        centers=centers.astype("float32"),
    )

    receipt = {
        "schema_version": 1,
        "candidate_id": args.candidate_id,
        "fold_id": args.fold,
        "architecture": "strict_group3_donor_pseudo_pls_source_rank_half_m225",
        "scope": (
            "official-data-only donor pseudo targets used only for preceding group-3 "
            "training; outer Q3 labels excluded from mapper, model, fixed policy, and blend"
        ),
        "physical_development_cutoff": str(DEV_CUTOFF),
        "strict_observed_training_rows": int(observed_training.sum()),
        "group3_extended_training_rows": int(extended_training.sum()),
        "group3_observed_rows": int(
            (extended_training & surface["actual_kwh"].notna().to_numpy()).sum()
        ),
        "group3_pseudo_rows": int((extended_training & pseudo_mask).sum()),
        "pseudo_weight": PSEUDO_WEIGHT,
        "pseudo_season_bandwidth_days": PSEUDO_SEASON_BANDWIDTH_DAYS,
        "pseudo_probability_weight": PSEUDO_PROBABILITY_WEIGHT,
        "parent_output_weight": PARENT_OUTPUT_WEIGHT,
        "pseudo_diagnostics": pseudo_diagnostics,
        "season_diagnostics": season_diagnostics,
        "class_width": CLASS_WIDTH,
        "class_count": len(active_bins),
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
        "raw_score": _score(raw_output),
        "raw_group_scores": _group_scores(raw_output),
        "fold_score": _score(output),
        "group_scores": _group_scores(output),
        "runtime_seconds": round(time.perf_counter() - started, 2),
        "prediction_path": str(output_path.relative_to(Path.cwd())),
        "prediction_sha256": _sha256(output_path),
        "probability_path": str(probability_path.relative_to(Path.cwd())),
        "probability_sha256": _sha256(probability_path),
        "parent_path": str(PARENT_PATH.relative_to(Path.cwd())),
        "parent_sha256": _sha256(PARENT_PATH),
        "base_probability_path": str(
            BASE_PROBABILITY_PATH.relative_to(Path.cwd())
        ),
        "base_probability_sha256": _sha256(BASE_PROBABILITY_PATH),
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
                "pseudo_diagnostics": pseudo_diagnostics,
                "season_diagnostics": season_diagnostics,
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
