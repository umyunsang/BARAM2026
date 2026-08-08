"""Materialize the predeclared fixed M195/M196 blend from the M196 receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from run_sequence_classifier import OUTPUT, _score, _sha256

PARENTS = (
    OUTPUT / "M195_STRICT_ROW_PLS_G3_Q3-dev-2023-Q3.parquet",
    OUTPUT / "M196_STRICT_MULTI_DONOR_PLS_Q3-dev-2023-Q3.parquet",
)
M196_RECEIPT = OUTPUT / "M196_STRICT_MULTI_DONOR_PLS_Q3-dev-2023-Q3.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--fold", choices=("dev-2023-Q3",), required=True)
    args = parser.parse_args()
    receipt196 = json.loads(M196_RECEIPT.read_text())
    declared = receipt196["fixed_half_m195_blend_score"]
    left, right = (pd.read_parquet(path) for path in PARENTS)
    keys = ["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh"]
    if not left[keys].equals(right[keys]):
        raise RuntimeError("M195/M196 parent key contract changed")
    output = left[keys].copy()
    output["prediction_kwh"] = 0.5 * left["prediction_kwh"].to_numpy() + 0.5 * right[
        "prediction_kwh"
    ].to_numpy()
    output["fold_id"] = args.fold
    output["model_id"] = args.candidate_id
    score = _score(output)
    for metric, value in declared.items():
        if abs(score[metric] - float(value)) > 1e-12:
            raise RuntimeError(f"fixed blend receipt mismatch: {metric}")
    output_path = OUTPUT / f"{args.candidate_id}-{args.fold}.parquet"
    output.to_parquet(output_path, index=False)
    receipt = {
        "schema_version": 1,
        "candidate_id": args.candidate_id,
        "fold_id": args.fold,
        "architecture": "predeclared_fixed_half_m195_m196",
        "scope": "M196-predeclared fixed blend; no label-selected weight or policy",
        "weights": [0.5, 0.5],
        "fold_score": score,
        "parent_paths": [str(path.relative_to(Path.cwd())) for path in PARENTS],
        "parent_sha256": [_sha256(path) for path in PARENTS],
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
