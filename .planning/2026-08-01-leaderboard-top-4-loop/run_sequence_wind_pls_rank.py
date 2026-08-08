"""Add explicit neighboring-NWP context to strict PLS source-rank heads."""

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
from run_xgb_multiquantile_pls import _screen
from strict_dev_surface import DEV_CUTOFF, development_surface

from baram.features.sequence import add_issuance_sequence_context

CLASS_WIDTH = 0.02
CORRELATION_FEATURES_PER_GROUP = 80
SEQUENCE_FEATURES_PER_SOURCE = 120
PARENT_PATH = OUTPUT / "M229_LOCAL_Q3_SEQUENCE_SMOOTH-dev-2023-Q3.parquet"


def _sequence_frame(
    surface: pd.DataFrame,
    base_matrix: pd.DataFrame,
    correlated: list[str],
) -> tuple[pd.DataFrame, int]:
    keys = ["data_available_kst_dtm", "group_id"]
    sizes = surface.groupby(keys, sort=False)["forecast_id"].transform("size")
    complete = sizes.eq(24).to_numpy()
    sequence_input = pd.concat(
        [
            surface.loc[
                complete,
                ["forecast_kst_dtm", "data_available_kst_dtm", "group_id"],
            ],
            base_matrix.loc[complete, correlated],
        ],
        axis=1,
    )
    contextual = add_issuance_sequence_context(sequence_input, correlated)
    sequence_columns = [name for name in contextual if name.startswith("seq__")]
    frame = contextual[sequence_columns].reindex(surface.index).astype("float32")
    return frame, int((~complete).sum())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--fold", choices=("dev-2023-Q3",), required=True)
    args = parser.parse_args()
    if _sha256(OPEN) != OPEN_SHA or _sha256(BASELINE) != BASELINE_SHA:
        raise RuntimeError("immutable competition input hash mismatch")
    if not PARENT_PATH.exists() or not M195_RECEIPT.exists() or not M195_LATENT.exists():
        raise RuntimeError("verified sequence/PLS parent surface is missing")

    started = time.perf_counter()
    surface, feature_columns = development_surface()
    if surface["forecast_kst_dtm"].ge(DEV_CUTOFF).any():
        raise RuntimeError("lockbox row reached sequence-wind runner")
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
    all_sequence, incomplete_rows = _sequence_frame(surface, base_matrix, correlated)
    sequence_candidates = {
        "global": list(all_sequence.columns),
        "gfs": [name for name in all_sequence if "gfs" in name.lower()],
        "ldaps": [name for name in all_sequence if "ldaps" in name.lower()],
    }
    selected_sequence = {
        source: _screen(
            all_sequence,
            target,
            training,
            candidates,
            min(SEQUENCE_FEATURES_PER_SOURCE, len(candidates)),
            20263400 + 10 * source_index,
        )
        for source_index, (source, candidates) in enumerate(
            sequence_candidates.items()
        )
    }
    selected_sequence_union = list(
        dict.fromkeys(
            name for names in selected_sequence.values() for name in names
        )
    )
    matrix = pd.concat(
        [
            base_matrix,
            pd.DataFrame(latent_values, columns=latent_columns, index=surface.index),
            all_sequence[selected_sequence_union],
        ],
        axis=1,
    )
    del base_matrix, latent_values, all_sequence
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
        "global": [*correlated, *selected_sequence["global"]],
        "gfs": [
            *[name for name in correlated if "gfs" in name.lower()],
            *selected_sequence["gfs"],
        ],
        "ldaps": [
            *[name for name in correlated if "ldaps" in name.lower()],
            *selected_sequence["ldaps"],
        ],
    }
    selected_features = {
        source: list(
            dict.fromkeys(
                [*parent_receipt["selected_features"][source], *additions[source]]
            )
        )
        for source in ("global", "gfs", "ldaps")
    }
    probabilities = {
        source: _probability(
            matrix,
            sample_weight,
            classes,
            training,
            validation,
            selected_features[source],
            len(active_bins),
            20263500 + 10 * source_index,
            source,
        )
        for source_index, source in enumerate(("global", "gfs", "ldaps"))
    }
    del matrix
    gc.collect()

    groups = surface.loc[validation, "group_id"].to_numpy(dtype=int)
    means = {
        group_id: float(
            target.loc[training & surface["group_id"].eq(group_id).to_numpy()].mean()
        )
        for group_id in CAPACITIES
    }
    probability = _mixture(probabilities, *FROZEN_MIXTURE)
    normalized = _policy_values(probability, centers, groups, means)[FROZEN_POLICY]
    raw_output = _frame(surface, validation, normalized)

    parent = pd.read_parquet(PARENT_PATH)
    expected_keys = pd.MultiIndex.from_frame(
        surface.loc[validation, ["forecast_id", "group_id"]]
    )
    parent_keys = pd.MultiIndex.from_frame(parent[["forecast_id", "group_id"]])
    if len(parent) != int(validation.sum()) or not parent_keys.equals(expected_keys):
        raise RuntimeError("M229 parent key contract changed")
    capacity = parent["group_id"].map(CAPACITIES).to_numpy(dtype=float)
    parent_normalized = parent["prediction_kwh"].to_numpy(dtype=float) / capacity
    output = _frame(surface, validation, 0.5 * parent_normalized + 0.5 * normalized)
    output["fold_id"] = args.fold
    output["model_id"] = args.candidate_id
    output_path = OUTPUT / f"{args.candidate_id}-{args.fold}.parquet"
    probability_path = OUTPUT / f"{args.candidate_id}-{args.fold}-probability.npz"
    output.to_parquet(output_path, index=False)
    np.savez(probability_path, probability=probability.astype("float32"))

    receipt = {
        "schema_version": 1,
        "candidate_id": args.candidate_id,
        "fold_id": args.fold,
        "architecture": "strict_neighboring_wind_sequence_pls_source_rank_half_m229",
        "scope": (
            "official-data-only known-future within-issuance NWP sequence context; "
            "outer Q3 labels excluded from sequence screen, heads, policy, and blend"
        ),
        "physical_development_cutoff": str(DEV_CUTOFF),
        "strict_training_rows": int(training.sum()),
        "class_width": CLASS_WIDTH,
        "class_count": len(active_bins),
        "correlation_features_per_group": CORRELATION_FEATURES_PER_GROUP,
        "correlation_union_count": len(correlated),
        "correlation_features_by_group": correlated_by_group,
        "sequence_input_count": len(correlated),
        "sequence_generated_count": sum(
            len(names) for names in sequence_candidates.values()
        ) - len(sequence_candidates["global"]),
        "sequence_incomplete_boundary_rows": incomplete_rows,
        "sequence_candidates_by_source": {
            source: len(names) for source, names in sequence_candidates.items()
        },
        "selected_sequence_features": selected_sequence,
        "selected_sequence_feature_counts": {
            source: len(names) for source, names in selected_sequence.items()
        },
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
                "selected_sequence_feature_counts": receipt[
                    "selected_sequence_feature_counts"
                ],
                "selected_feature_counts": receipt["selected_feature_counts"],
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
