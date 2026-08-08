"""Fit a group-3-only source-rank head over the M195/M196 latent surface."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
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
from run_strict_prequential_source_rank import (
    FROZEN_MIXTURE,
    FROZEN_POLICY,
    _mixture,
    _source_columns,
    _source_probability,
)
from strict_dev_surface import DEV_CUTOFF, development_surface

BASE_COLUMNS = ["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh"]
CLASS_WIDTH = 0.025
PARENT_PATH = OUTPUT / "M197_STRICT_FIXED_HALF_PLS-dev-2023-Q3.parquet"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--fold", choices=("dev-2023-Q3",), required=True)
    parser.add_argument("--iterations", type=int, default=90)
    parser.add_argument("--global-features", type=int, default=120)
    parser.add_argument("--source-features", type=int, default=180)
    args = parser.parse_args()
    if not 60 <= args.iterations <= 160:
        raise ValueError("group-3 head iterations must be between 60 and 160")
    if _sha256(OPEN) != OPEN_SHA or _sha256(BASELINE) != BASELINE_SHA:
        raise RuntimeError("immutable competition input hash mismatch")
    if not PARENT_PATH.exists() or not M195_RECEIPT.exists() or not M195_LATENT.exists():
        raise RuntimeError("verified PLS parent surface is missing")

    started = time.perf_counter()
    surface, feature_columns = development_surface()
    if surface["forecast_kst_dtm"].ge(DEV_CUTOFF).any():
        raise RuntimeError("lockbox row reached group-3 PLS head")
    validation = _validation_mask(surface, args.fold)
    history = _strict_preceding_mask(surface, validation)
    target = surface["actual_kwh"] / surface["group_id"].map(CAPACITIES)
    group3 = surface["group_id"].eq(3).to_numpy()
    training = (
        history
        & group3
        & surface["actual_kwh"].notna().to_numpy()
        & target.ge(0.10).to_numpy()
    )
    application = validation & group3

    parent_receipt = json.loads(M195_RECEIPT.read_text())
    if _sha256(M195_LATENT) != parent_receipt["latent_checkpoint_sha256"]:
        raise RuntimeError("M195 latent checkpoint hash mismatch")
    cached = np.load(M195_LATENT, allow_pickle=False)
    parent_columns = [str(value) for value in cached["columns"].tolist()]
    parent_values = np.asarray(cached["values"], dtype="float32")
    if parent_values.shape != (len(surface), len(parent_columns)):
        raise RuntimeError("M195 latent shape contract changed")
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
    centers = np.asarray(
        [
            target.loc[training & classes.eq(index)].mean()
            for index in range(len(active_bins))
        ],
        dtype=float,
    )
    all_candidates = list(matrix.columns)
    global_latent = [
        name
        for name in all_candidates
        if "__global__" in name and name.startswith(("pls__", "mpls__"))
    ]
    source_specs = {
        "global": (all_candidates, args.global_features),
        "gfs": (
            list(dict.fromkeys([*_source_columns(all_candidates, "gfs"), *global_latent])),
            args.source_features,
        ),
        "ldaps": (
            list(
                dict.fromkeys(
                    [*_source_columns(all_candidates, "ldaps"), *global_latent]
                )
            ),
            args.source_features,
        ),
    }
    probabilities: dict[str, np.ndarray] = {}
    selected_features: dict[str, list[str]] = {}
    for source_index, (source, (candidates, count)) in enumerate(source_specs.items()):
        probability, selected = _source_probability(
            matrix,
            target,
            classes,
            training,
            application,
            candidates,
            count,
            args.iterations,
            len(active_bins),
            20261700 + source_index * 10,
            source,
        )
        probabilities[source] = probability
        selected_features[source] = selected
    del matrix
    gc.collect()

    probability = _mixture(probabilities, *FROZEN_MIXTURE)
    mean = float(target.loc[training].mean())
    means = {group_id: mean for group_id in CAPACITIES}
    group_ids = np.full(int(application.sum()), 3, dtype=int)
    head_normalized = _policy_values(probability, centers, group_ids, means)[
        FROZEN_POLICY
    ]
    group3_base = surface.loc[application, BASE_COLUMNS].copy()
    group3_actual = group3_base["actual_kwh"].to_numpy(dtype=float) / CAPACITIES[3]
    head_score = _group_total(group3_actual, head_normalized)

    parent = pd.read_parquet(PARENT_PATH)
    expected_keys = pd.MultiIndex.from_frame(
        surface.loc[validation, ["forecast_id", "group_id"]]
    )
    parent_keys = pd.MultiIndex.from_frame(parent[["forecast_id", "group_id"]])
    if len(parent) != int(validation.sum()) or not parent_keys.equals(expected_keys):
        raise RuntimeError("M197 parent key contract changed")
    output = parent[[*BASE_COLUMNS, "prediction_kwh"]].copy()
    parent_group3 = parent["group_id"].eq(3).to_numpy()
    parent_group3_normalized = parent.loc[
        parent_group3, "prediction_kwh"
    ].to_numpy(dtype=float) / CAPACITIES[3]
    blended_group3 = 0.5 * parent_group3_normalized + 0.5 * head_normalized
    output.loc[parent_group3, "prediction_kwh"] = blended_group3 * CAPACITIES[3]
    output["fold_id"] = args.fold
    output["model_id"] = args.candidate_id
    output_path = OUTPUT / f"{args.candidate_id}-{args.fold}.parquet"
    output.to_parquet(output_path, index=False)
    receipt = {
        "schema_version": 1,
        "candidate_id": args.candidate_id,
        "fold_id": args.fold,
        "architecture": "strict_group3_source_rank_head_half_m197",
        "scope": (
            "fixed group-3-only official-data PLS head; outer Q3 labels excluded "
            "from feature/model/policy fit and the 50:50 parent weight"
        ),
        "physical_development_cutoff": str(DEV_CUTOFF),
        "training_rows": int(training.sum()),
        "class_width": CLASS_WIDTH,
        "class_count": len(active_bins),
        "iterations": args.iterations,
        "multioutput_diagnostics": multioutput_diagnostics,
        "selected_features": selected_features,
        "selected_pls_feature_counts": {
            source: sum(name.startswith(("pls__", "mpls__")) for name in names)
            for source, names in selected_features.items()
        },
        "frozen_mixture": {
            "global_weight": FROZEN_MIXTURE[0],
            "gfs_share": FROZEN_MIXTURE[1],
        },
        "frozen_policy": FROZEN_POLICY,
        "group3_head_score": head_score,
        "group3_blended_score": _group_total(group3_actual, blended_group3),
        "fold_score": _score(output),
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
                "group3_head_score": head_score,
                "group3_blended_score": receipt["group3_blended_score"],
                "fold_score": receipt["fold_score"],
                "selected_pls_feature_counts": receipt[
                    "selected_pls_feature_counts"
                ],
                "runtime_seconds": receipt["runtime_seconds"],
                "prediction_sha256": receipt["prediction_sha256"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
