"""Freeze source-rank policies on preceding folds and apply them forward."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
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

FOLDS = ("dev-2023-Q2", "dev-2023-Q3", "dev-2023-Q4")
KEYS = ["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh"]
POLICY_PATHS = {
    "dev-2023-Q2": OUTPUT / "M150_SOURCE_RANK_XGB_Q2-dev-2023-Q2-policies.parquet",
    "dev-2023-Q3": OUTPUT / "M149_SOURCE_RANK_XGB_Q3-dev-2023-Q3-policies.parquet",
    "dev-2023-Q4": OUTPUT / "M151_SOURCE_RANK_XGB_Q4-dev-2023-Q4-policies.parquet",
}
PARENT_PATH = OUTPUT / "M107_STRICT_TEMPORAL_TOP100-oof.parquet"
PARENT_WEIGHTS = tuple(round(float(value), 1) for value in np.arange(0.0, 1.01, 0.1))


def _load_fold(fold_id: str, common: set[str]) -> pd.DataFrame:
    policies = pd.read_parquet(POLICY_PATHS[fold_id], columns=[*KEYS, *sorted(common)])
    parent = pd.read_parquet(PARENT_PATH)
    parent = parent.loc[parent["fold_id"].eq(fold_id), [*KEYS, "prediction_kwh"]]
    output = policies.merge(
        parent.rename(columns={"prediction_kwh": "parent_prediction_kwh"}),
        on=KEYS,
        validate="one_to_one",
    )
    output["selection_fold_id"] = fold_id
    return output


def _group_score(
    actual_kwh: np.ndarray,
    prediction_kwh: np.ndarray,
    group_id: int,
) -> dict[str, float]:
    capacity = CAPACITIES[group_id]
    actual = actual_kwh / capacity
    prediction = prediction_kwh / capacity
    valid = np.isfinite(actual) & (actual >= 0.10)
    actual = actual[valid]
    error = np.abs(prediction[valid] - actual)
    units = np.select([error <= 0.06, error <= 0.08], [4.0, 3.0], default=0.0)
    one_minus_nmae = 1.0 - float(error.mean())
    ficr = float(np.sum(actual * units) / np.sum(actual * 4.0))
    return {
        "total": 0.5 * (one_minus_nmae + ficr),
        "one_minus_nmae": one_minus_nmae,
        "ficr": ficr,
    }


def _select(
    history: list[pd.DataFrame],
    common: set[str],
    *,
    min_parent_weight: float,
) -> dict[str, dict[str, object]]:
    pooled = pd.concat(history, ignore_index=True)
    selections: dict[str, dict[str, object]] = {}
    for group_id in CAPACITIES:
        group = pooled.loc[pooled["group_id"].eq(group_id)]
        actual = group["actual_kwh"].to_numpy(dtype=float)
        parent = group["parent_prediction_kwh"].to_numpy(dtype=float)
        fold_ids = sorted(group["selection_fold_id"].unique())
        parent_fold_scores = {
            fold_id: _group_score(
                group.loc[group["selection_fold_id"].eq(fold_id), "actual_kwh"].to_numpy(
                    dtype=float
                ),
                group.loc[
                    group["selection_fold_id"].eq(fold_id), "parent_prediction_kwh"
                ].to_numpy(dtype=float),
                group_id,
            )["total"]
            for fold_id in fold_ids
        }
        best: tuple[float, float, float, str, dict[str, float], dict[str, float]] | None = None
        for policy in sorted(common):
            challenger = group[policy].to_numpy(dtype=float)
            for parent_weight in PARENT_WEIGHTS:
                if parent_weight < min_parent_weight:
                    continue
                prediction = (
                    parent_weight * parent
                    + (1.0 - parent_weight) * challenger
                )
                score = _group_score(actual, prediction, group_id)
                fold_deltas: dict[str, float] = {}
                for fold_id in fold_ids:
                    fold_mask = group["selection_fold_id"].eq(fold_id).to_numpy()
                    fold_score = _group_score(
                        actual[fold_mask], prediction[fold_mask], group_id
                    )["total"]
                    fold_deltas[fold_id] = fold_score - parent_fold_scores[fold_id]
                worst_delta = min(fold_deltas.values())
                if worst_delta < -0.001:
                    continue
                choice = (
                    score["total"],
                    worst_delta,
                    parent_weight,
                    policy,
                    score,
                    fold_deltas,
                )
                if best is None or choice[0] > best[0]:
                    best = choice
        assert best is not None
        selections[str(group_id)] = {
            "policy": best[3],
            "parent_weight": best[2],
            "preceding_score": best[4],
            "preceding_fold_deltas": best[5],
            "worst_fold_delta": best[1],
        }
    return selections


def _apply(
    frame: pd.DataFrame,
    selections: dict[str, dict[str, object]],
    fold_id: str,
    candidate_id: str,
) -> pd.DataFrame:
    output = frame[KEYS].copy()
    prediction = np.empty(len(frame), dtype=float)
    for group_id in CAPACITIES:
        mask = frame["group_id"].eq(group_id).to_numpy()
        selection = selections[str(group_id)]
        parent_weight = float(selection["parent_weight"])
        prediction[mask] = (
            parent_weight
            * frame.loc[mask, "parent_prediction_kwh"].to_numpy(dtype=float)
            + (1.0 - parent_weight)
            * frame.loc[mask, str(selection["policy"])].to_numpy(dtype=float)
        )
    output["prediction_kwh"] = prediction
    output["fold_id"] = fold_id
    output["model_id"] = candidate_id
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-id", default="M152_STRICT_SOURCE_RANK")
    args = parser.parse_args()
    if _sha256(OPEN) != OPEN_SHA or _sha256(BASELINE) != BASELINE_SHA:
        raise RuntimeError("immutable competition input hash mismatch")
    started = time.perf_counter()
    policy_columns = {
        fold_id: set(pd.read_parquet(path).columns).difference(KEYS)
        for fold_id, path in POLICY_PATHS.items()
    }
    common = set.intersection(*policy_columns.values())
    if not common:
        raise RuntimeError("source-rank folds have no common policy columns")
    folds = {fold_id: _load_fold(fold_id, common) for fold_id in FOLDS}
    q3_selection = _select(
        [folds[FOLDS[0]]], common, min_parent_weight=0.0
    )
    q3 = _apply(folds[FOLDS[1]], q3_selection, FOLDS[1], args.candidate_id)
    q4_selection = _select(
        [folds[FOLDS[0]], folds[FOLDS[1]]],
        common,
        min_parent_weight=0.5,
    )
    q4 = _apply(folds[FOLDS[2]], q4_selection, FOLDS[2], args.candidate_id)
    parent = pd.read_parquet(PARENT_PATH)
    q2 = parent.loc[parent["fold_id"].eq(FOLDS[0]), [*KEYS, "prediction_kwh"]].copy()
    q2["fold_id"] = FOLDS[0]
    q2["model_id"] = args.candidate_id
    output = pd.concat([q2, q3, q4], ignore_index=True)
    output_path = OUTPUT / f"{args.candidate_id}-oof.parquet"
    output.to_parquet(output_path, index=False)
    receipt = {
        "schema_version": 1,
        "candidate_id": args.candidate_id,
        "scope": "Q2 unchanged; Q3 selected on Q2; Q4 selected on pooled Q2-Q3",
        "common_policy_count": len(common),
        "q4_min_parent_weight": 0.5,
        "q3_selection": q3_selection,
        "q4_selection": q4_selection,
        "fold_scores": {
            fold_id: _score(output.loc[output["fold_id"].eq(fold_id)])
            for fold_id in FOLDS
        },
        "pooled": _score(output),
        "source_sha256": {
            fold_id: _sha256(path) for fold_id, path in POLICY_PATHS.items()
        },
        "parent_sha256": _sha256(PARENT_PATH),
        "runtime_seconds": round(time.perf_counter() - started, 2),
        "prediction_path": str(output_path.relative_to(Path.cwd())),
        "prediction_sha256": _sha256(output_path),
        "observed_scada_feature_count": 0,
        "new_2024_evaluation": False,
        "lockbox_reopened": False,
        "external_actions": [],
    }
    receipt_path = OUTPUT / f"{args.candidate_id}-oof.json"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
