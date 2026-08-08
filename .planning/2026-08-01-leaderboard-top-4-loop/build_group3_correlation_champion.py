"""Materialize the fixed M197 groups 1/2 plus M212 group-3 hybrid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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

BASE_PARENTS = {
    "M197": OUTPUT / "M197_STRICT_FIXED_HALF_PLS-dev-2023-Q3.parquet",
    "M213": OUTPUT / "M213_STRICT_G3_CORRELATION_CHAMPION-dev-2023-Q3.parquet",
}
GROUP3_PARENTS = {
    "M212": OUTPUT / "M212_STRICT_CORRELATION_WIND_PLS_Q3-dev-2023-Q3.parquet",
    "M214": OUTPUT / "M214_STRICT_BALANCED_CORRELATION_WIND_Q3-dev-2023-Q3.parquet",
}
KEYS = ["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--fold", choices=("dev-2023-Q3",), required=True)
    parser.add_argument("--base", choices=tuple(BASE_PARENTS), default="M197")
    parser.add_argument("--group3", choices=tuple(GROUP3_PARENTS), default="M212")
    args = parser.parse_args()
    if _sha256(OPEN) != OPEN_SHA or _sha256(BASELINE) != BASELINE_SHA:
        raise RuntimeError("immutable competition input hash mismatch")
    base_parent = BASE_PARENTS[args.base]
    group3_parent = GROUP3_PARENTS[args.group3]
    if not base_parent.exists() or not group3_parent.exists():
        raise RuntimeError("hybrid parent artifact is missing")

    base = pd.read_parquet(base_parent)
    challenger = pd.read_parquet(group3_parent)
    base_keys = pd.MultiIndex.from_frame(base[["forecast_id", "group_id"]])
    challenger_keys = pd.MultiIndex.from_frame(
        challenger[["forecast_id", "group_id"]]
    )
    if len(base) != len(challenger) or not base_keys.equals(challenger_keys):
        raise RuntimeError("hybrid parent key contract changed")
    if not base[KEYS].equals(challenger[KEYS]):
        raise RuntimeError("hybrid parent label/timestamp contract changed")

    output = base.copy()
    group3 = output["group_id"].eq(3)
    output.loc[group3, "prediction_kwh"] = challenger.loc[
        group3, "prediction_kwh"
    ].to_numpy(dtype=float)
    output["fold_id"] = args.fold
    output["model_id"] = args.candidate_id
    output_path = OUTPUT / f"{args.candidate_id}-{args.fold}.parquet"
    output.to_parquet(output_path, index=False)
    receipt = {
        "schema_version": 1,
        "candidate_id": args.candidate_id,
        "fold_id": args.fold,
        "architecture": f"strict_{args.base.lower()}_groups12_{args.group3.lower()}_group3",
        "scope": (
            "fixed official-data-only group hybrid; no refit, calibration, "
            "lockbox access, or external action"
        ),
        "group_parent_map": {
            "1": str(base_parent.relative_to(Path.cwd())),
            "2": str(base_parent.relative_to(Path.cwd())),
            "3": str(group3_parent.relative_to(Path.cwd())),
        },
        "parent_sha256": {
            "base": _sha256(base_parent),
            "group3": _sha256(group3_parent),
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
