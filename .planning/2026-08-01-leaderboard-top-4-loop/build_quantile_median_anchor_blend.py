"""Materialize the fixed local-Q3 point-anchor blend diagnosed after M221."""

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
    OPEN,
    OPEN_SHA,
    OUTPUT,
    _score,
    _sha256,
)

BASE_PARENT = OUTPUT / "M220_LOCAL_Q3_CORR_BIN_WEIGHT_BLEND-dev-2023-Q3.parquet"
QUANTILE_PARENT = (
    OUTPUT / "M206_STRICT_XGB_MULTIQUANTILE_PLS_Q3-dev-2023-Q3-quantiles.npz"
)
QUANTILE_FRAME = OUTPUT / "M206_STRICT_XGB_MULTIQUANTILE_PLS_Q3-dev-2023-Q3.parquet"
ANCHOR_WEIGHTS = {1: 0.05, 2: 0.05, 3: 0.0}
KEYS = ["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--fold", choices=("dev-2023-Q3",), required=True)
    args = parser.parse_args()
    if _sha256(OPEN) != OPEN_SHA or _sha256(BASELINE) != BASELINE_SHA:
        raise RuntimeError("immutable competition input hash mismatch")
    for parent in (BASE_PARENT, QUANTILE_PARENT, QUANTILE_FRAME):
        if not parent.exists():
            raise RuntimeError(f"fixed blend parent is missing: {parent}")

    base = pd.read_parquet(BASE_PARENT)
    quantile_frame = pd.read_parquet(QUANTILE_FRAME)
    if not base[KEYS].equals(quantile_frame[KEYS]):
        raise RuntimeError("M206 quantile order differs from the M220 base contract")

    with np.load(QUANTILE_PARENT) as cache:
        mixed = 0.5 * cache["global_quantiles"] + 0.5 * (
            0.25 * cache["gfs_quantiles"] + 0.75 * cache["ldaps_quantiles"]
        )
    median = np.sort(mixed, axis=1)[:, mixed.shape[1] // 2]
    if len(median) != len(base) or not np.isfinite(median).all():
        raise RuntimeError("invalid M206 median prediction contract")

    output = base.copy()
    for group_id, anchor_weight in ANCHOR_WEIGHTS.items():
        mask = output["group_id"].eq(group_id)
        output.loc[mask, "prediction_kwh"] = (
            (1.0 - anchor_weight)
            * base.loc[mask, "prediction_kwh"].to_numpy(dtype=float)
            + anchor_weight * median[mask.to_numpy()]
        )
    output["fold_id"] = args.fold
    output["model_id"] = args.candidate_id
    output_path = OUTPUT / f"{args.candidate_id}-{args.fold}.parquet"
    output.to_parquet(output_path, index=False)

    receipt = {
        "schema_version": 1,
        "candidate_id": args.candidate_id,
        "fold_id": args.fold,
        "architecture": "fixed_groupwise_m206_median_anchor_blend",
        "scope": (
            "official-data-only fixed parent blend; weights selected on the local "
            "Q3 development fold and therefore not independent holdout evidence"
        ),
        "anchor_weights": {
            str(group_id): weight for group_id, weight in ANCHOR_WEIGHTS.items()
        },
        "parent_paths": {
            "base": str(BASE_PARENT.relative_to(Path.cwd())),
            "quantiles": str(QUANTILE_PARENT.relative_to(Path.cwd())),
            "quantile_frame": str(QUANTILE_FRAME.relative_to(Path.cwd())),
        },
        "parent_sha256": {
            "base": _sha256(BASE_PARENT),
            "quantiles": _sha256(QUANTILE_PARENT),
            "quantile_frame": _sha256(QUANTILE_FRAME),
        },
        "fold_score": _score(output),
        "group_scores": _group_scores(output),
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
