"""Materialize the fixed local-Q3 M212 probability-smoothing blend."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from run_group_balanced_pls_rank import _frame, _group_scores
from run_inner_policy_classifier import _policy_values
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
from run_strict_prequential_source_rank import FROZEN_POLICY
from strict_dev_surface import DEV_CUTOFF, development_surface

BASE_PARENT = OUTPUT / "M222_LOCAL_Q3_POINT_ANCHOR_BLEND-dev-2023-Q3.parquet"
PROBABILITY_PARENT = (
    OUTPUT / "M212_STRICT_CORRELATION_WIND_PLS_Q3-dev-2023-Q3-probability.npz"
)
PROBABILITY_RECEIPT = (
    OUTPUT / "M212_STRICT_CORRELATION_WIND_PLS_Q3-dev-2023-Q3.json"
)
CLASS_WIDTH = 0.02
SMOOTHING_MASS = 0.20
CHALLENGER_WEIGHTS = {1: 0.0, 2: 0.15, 3: 0.0}


def _smooth(probability: np.ndarray) -> np.ndarray:
    mass = SMOOTHING_MASS
    smoothed = (1.0 - 2.0 * mass) * probability.copy()
    smoothed[:, 1:] += mass * probability[:, :-1]
    smoothed[:, :-1] += mass * probability[:, 1:]
    smoothed[:, 0] += mass * probability[:, 0]
    smoothed[:, -1] += mass * probability[:, -1]
    return smoothed / smoothed.sum(axis=1, keepdims=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--fold", choices=("dev-2023-Q3",), required=True)
    args = parser.parse_args()
    if _sha256(OPEN) != OPEN_SHA or _sha256(BASELINE) != BASELINE_SHA:
        raise RuntimeError("immutable competition input hash mismatch")
    if not all(
        path.exists()
        for path in (BASE_PARENT, PROBABILITY_PARENT, PROBABILITY_RECEIPT)
    ):
        raise RuntimeError("fixed probability-smoothing parent is missing")
    probability_receipt = json.loads(PROBABILITY_RECEIPT.read_text())
    if _sha256(PROBABILITY_PARENT) != probability_receipt["probability_sha256"]:
        raise RuntimeError("M212 probability checkpoint hash mismatch")

    surface, _ = development_surface()
    if surface["forecast_kst_dtm"].ge(DEV_CUTOFF).any():
        raise RuntimeError("lockbox row reached probability-smoothing builder")
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
    centers = np.asarray(
        [
            target.loc[training & classes.eq(index)].mean()
            for index in range(len(active_bins))
        ],
        dtype=float,
    )
    with np.load(PROBABILITY_PARENT) as cache:
        probability = np.asarray(cache["probability"], dtype=float)
    if probability.shape != (int(validation.sum()), len(centers)):
        raise RuntimeError("M212 probability class contract changed")
    smoothed = _smooth(probability)
    groups = surface.loc[validation, "group_id"].to_numpy(dtype=int)
    means = {
        group_id: float(
            target.loc[training & surface["group_id"].eq(group_id).to_numpy()].mean()
        )
        for group_id in CAPACITIES
    }
    normalized = _policy_values(smoothed, centers, groups, means)[FROZEN_POLICY]
    challenger = _frame(surface, validation, normalized)

    base = pd.read_parquet(BASE_PARENT)
    expected_keys = pd.MultiIndex.from_frame(
        surface.loc[validation, ["forecast_id", "group_id"]]
    )
    base_keys = pd.MultiIndex.from_frame(base[["forecast_id", "group_id"]])
    if len(base) != int(validation.sum()) or not base_keys.equals(expected_keys):
        raise RuntimeError("M222 parent key contract changed")
    output = base.copy()
    for group_id, challenger_weight in CHALLENGER_WEIGHTS.items():
        mask = output["group_id"].eq(group_id).to_numpy()
        output.loc[mask, "prediction_kwh"] = (
            (1.0 - challenger_weight)
            * base["prediction_kwh"].to_numpy(dtype=float)[mask]
            + challenger_weight
            * challenger["prediction_kwh"].to_numpy(dtype=float)[mask]
        )
    output["fold_id"] = args.fold
    output["model_id"] = args.candidate_id
    output_path = OUTPUT / f"{args.candidate_id}-{args.fold}.parquet"
    output.to_parquet(output_path, index=False)

    receipt = {
        "schema_version": 1,
        "candidate_id": args.candidate_id,
        "fold_id": args.fold,
        "architecture": "fixed_local_m212_probability_smoothing_group_blend",
        "scope": (
            "official-data-only deterministic M212 probability smoothing; smoothing "
            "and group blend weights selected on local Q3, not independent holdout evidence"
        ),
        "class_width": CLASS_WIDTH,
        "neighbor_smoothing_mass": SMOOTHING_MASS,
        "challenger_weights": {
            str(group_id): weight
            for group_id, weight in CHALLENGER_WEIGHTS.items()
        },
        "frozen_policy": FROZEN_POLICY,
        "challenger_score": _score(challenger),
        "challenger_group_scores": _group_scores(challenger),
        "fold_score": _score(output),
        "group_scores": _group_scores(output),
        "parent_paths": {
            "base": str(BASE_PARENT.relative_to(Path.cwd())),
            "probability": str(PROBABILITY_PARENT.relative_to(Path.cwd())),
        },
        "parent_sha256": {
            "base": _sha256(BASE_PARENT),
            "probability": _sha256(PROBABILITY_PARENT),
        },
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
