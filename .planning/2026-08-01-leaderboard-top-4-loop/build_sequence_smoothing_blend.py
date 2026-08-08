"""Materialize the fixed local-Q3 within-issuance smoothing blend."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from run_group_balanced_pls_rank import _group_scores
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
from run_site_wind_teacher import _validation_mask
from strict_dev_surface import DEV_CUTOFF, development_surface

PARENT_PATH = OUTPUT / "M225_LOCAL_Q3_PROB_SMOOTH_BLEND-dev-2023-Q3.parquet"
RECIPES = {
    1: ("ramp", 0.025),
    2: ("median3", 0.30),
    3: ("gauss5", 0.325),
}


def _smooth(values: np.ndarray, kind: str) -> np.ndarray:
    if kind == "ramp":
        result = values.copy()
        result[1:-1] = (
            0.25 * values[:-2] + 0.50 * values[1:-1] + 0.25 * values[2:]
        )
        return result
    if kind == "median3":
        padded = np.pad(values, (1, 1), mode="edge")
        return np.asarray(
            [np.median(padded[index : index + 3]) for index in range(len(values))]
        )
    if kind == "gauss5":
        return np.convolve(
            np.pad(values, (2, 2), mode="edge"),
            [0.0625, 0.25, 0.375, 0.25, 0.0625],
            mode="valid",
        )
    raise ValueError(f"unknown smoothing recipe: {kind}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--fold", choices=("dev-2023-Q3",), required=True)
    args = parser.parse_args()
    if _sha256(OPEN) != OPEN_SHA or _sha256(BASELINE) != BASELINE_SHA:
        raise RuntimeError("immutable competition input hash mismatch")
    if not PARENT_PATH.exists():
        raise RuntimeError("M225 sequence-smoothing parent is missing")

    surface, _ = development_surface()
    if surface["forecast_kst_dtm"].ge(DEV_CUTOFF).any():
        raise RuntimeError("lockbox row reached sequence-smoothing builder")
    validation = _validation_mask(surface, args.fold)
    validation_surface = surface.loc[validation].reset_index(drop=True)
    parent = pd.read_parquet(PARENT_PATH).reset_index(drop=True)
    expected_keys = pd.MultiIndex.from_frame(
        validation_surface[["forecast_id", "group_id"]]
    )
    parent_keys = pd.MultiIndex.from_frame(parent[["forecast_id", "group_id"]])
    if len(parent) != len(validation_surface) or not parent_keys.equals(expected_keys):
        raise RuntimeError("M225 parent key contract changed")

    capacity = parent["group_id"].map(CAPACITIES).to_numpy(dtype=float)
    normalized = parent["prediction_kwh"].to_numpy(dtype=float) / capacity
    smoothed = normalized.copy()
    grouping = pd.DataFrame(
        {
            "issuance": validation_surface["data_available_kst_dtm"],
            "group_id": parent["group_id"],
        }
    )
    for (_, group_id), indices in grouping.groupby(
        ["issuance", "group_id"],
        sort=False,
    ).groups.items():
        positions = np.asarray(list(indices), dtype=int)
        order = np.argsort(
            validation_surface.loc[positions, "forecast_kst_dtm"].to_numpy()
        )
        ordered = positions[order]
        kind, _ = RECIPES[int(group_id)]
        smoothed[ordered] = _smooth(normalized[ordered], kind)

    output_normalized = normalized.copy()
    for group_id, (_, smoothing_weight) in RECIPES.items():
        mask = parent["group_id"].eq(group_id).to_numpy()
        output_normalized[mask] = (
            (1.0 - smoothing_weight) * normalized[mask]
            + smoothing_weight * smoothed[mask]
        )
    output = parent.copy()
    output["prediction_kwh"] = output_normalized * capacity
    output["fold_id"] = args.fold
    output["model_id"] = args.candidate_id
    output_path = OUTPUT / f"{args.candidate_id}-{args.fold}.parquet"
    output.to_parquet(output_path, index=False)

    receipt = {
        "schema_version": 1,
        "candidate_id": args.candidate_id,
        "fold_id": args.fold,
        "architecture": "fixed_local_within_issuance_sequence_smoothing",
        "scope": (
            "official-data-only deterministic 24-hour within-issuance smoothing; "
            "recipes and weights selected on local Q3, not independent holdout evidence"
        ),
        "recipes": {
            str(group_id): {"kind": kind, "smoothing_weight": weight}
            for group_id, (kind, weight) in RECIPES.items()
        },
        "fold_score": _score(output),
        "group_scores": _group_scores(output),
        "parent_path": str(PARENT_PATH.relative_to(Path.cwd())),
        "parent_sha256": _sha256(PARENT_PATH),
        "prediction_path": str(output_path.relative_to(Path.cwd())),
        "prediction_sha256": _sha256(output_path),
        "observed_scada_feature_count": 0,
        "new_2024_evaluation": False,
        "lockbox_reopened": False,
        "external_actions": [],
    }
    receipt_path = OUTPUT / f"{args.candidate_id}-{args.fold}.json"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
