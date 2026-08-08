"""Materialize a fixed groupwise blend of two strict Q3 parent predictions."""

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

RECIPES = {
    "M218": {
        "base": OUTPUT
        / "M215_STRICT_G3_BALANCED_CORRELATION_CHAMPION-dev-2023-Q3.parquet",
        "challenger": OUTPUT
        / "M217_STRICT_BALANCED_CORR_WIND_BIN025_Q3-dev-2023-Q3.parquet",
        "weights": {1: 0.5, 2: 0.0, 3: 0.5},
    },
    "M220": {
        "base": OUTPUT / "M218_LOCAL_Q3_CORR_BIN_BLEND-dev-2023-Q3.parquet",
        "challenger": OUTPUT
        / "M219_STRICT_UNBALANCED_CORR_WIND_BIN025_Q3-dev-2023-Q3.parquet",
        "weights": {1: 0.75, 2: 0.0, 3: 0.0},
    },
}
KEYS = ["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--fold", choices=("dev-2023-Q3",), required=True)
    parser.add_argument("--recipe", choices=tuple(RECIPES), default="M218")
    args = parser.parse_args()
    if _sha256(OPEN) != OPEN_SHA or _sha256(BASELINE) != BASELINE_SHA:
        raise RuntimeError("immutable competition input hash mismatch")
    recipe = RECIPES[args.recipe]
    base_parent = Path(recipe["base"])
    challenger_parent = Path(recipe["challenger"])
    challenger_weights = dict(recipe["weights"])
    if not base_parent.exists() or not challenger_parent.exists():
        raise RuntimeError("fixed blend parent artifact is missing")

    base = pd.read_parquet(base_parent)
    challenger = pd.read_parquet(challenger_parent)
    base_keys = pd.MultiIndex.from_frame(base[["forecast_id", "group_id"]])
    challenger_keys = pd.MultiIndex.from_frame(
        challenger[["forecast_id", "group_id"]]
    )
    if len(base) != len(challenger) or not base_keys.equals(challenger_keys):
        raise RuntimeError("fixed blend parent key contract changed")
    if not base[KEYS].equals(challenger[KEYS]):
        raise RuntimeError("fixed blend parent label/timestamp contract changed")

    output = base.copy()
    for group_id, challenger_weight in challenger_weights.items():
        mask = output["group_id"].eq(group_id)
        output.loc[mask, "prediction_kwh"] = (
            (1.0 - challenger_weight)
            * base.loc[mask, "prediction_kwh"].to_numpy(dtype=float)
            + challenger_weight
            * challenger.loc[mask, "prediction_kwh"].to_numpy(dtype=float)
        )
    output["fold_id"] = args.fold
    output["model_id"] = args.candidate_id
    output_path = OUTPUT / f"{args.candidate_id}-{args.fold}.parquet"
    output.to_parquet(output_path, index=False)
    receipt = {
        "schema_version": 1,
        "candidate_id": args.candidate_id,
        "fold_id": args.fold,
        "architecture": f"fixed_groupwise_blend_{args.recipe.lower()}",
        "scope": (
            "official-data-only fixed parent blend; weights selected on the local "
            "Q3 development fold and therefore not independent holdout evidence"
        ),
        "challenger_weights": {
            str(group_id): weight
            for group_id, weight in challenger_weights.items()
        },
        "parent_paths": {
            "base": str(base_parent.relative_to(Path.cwd())),
            "challenger": str(challenger_parent.relative_to(Path.cwd())),
        },
        "parent_sha256": {
            "base": _sha256(base_parent),
            "challenger": _sha256(challenger_parent),
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
