"""Freeze Q2-selected DART policies and parent blends for later folds."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from run_sequence_classifier import OUTPUT, _score, _sha256

REPO = Path(__file__).resolve().parents[2]
PARENT = OUTPUT / "M107_STRICT_TEMPORAL_TOP100-oof.parquet"
FOLDS = ("dev-2023-Q2", "dev-2023-Q3", "dev-2023-Q4")
BOOSTER_ID = "M113_LGBM_DART"
CANDIDATE_ID = "M114_STRICT_DART_BLEND"
SELECTED_ITERATION = 140
BASE_COLUMNS = ["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh"]
CAPACITIES = {1: 21_600.0, 2: 21_600.0, 3: 21_000.0}
PARENT_WEIGHT_GRID = tuple(round(value, 1) for value in np.arange(0.0, 1.01, 0.1))


def _booster_path(fold_id: str) -> Path:
    return OUTPUT / f"{BOOSTER_ID}-{fold_id}-policies.parquet"


def _group_score(frame: pd.DataFrame, group_id: int) -> dict[str, float]:
    capacity = CAPACITIES[group_id]
    valid = frame["actual_kwh"].to_numpy(dtype=float) >= 0.10 * capacity
    actual = frame.loc[valid, "actual_kwh"].to_numpy(dtype=float)
    prediction = frame.loc[valid, "prediction_kwh"].to_numpy(dtype=float)
    error = np.abs(prediction - actual) / capacity
    units = np.select([error <= 0.06, error <= 0.08], [4.0, 3.0], default=0.0)
    one_minus_nmae = 1.0 - float(error.mean())
    ficr = float(np.sum(actual * units) / np.sum(actual * 4.0))
    return {
        "total": 0.5 * (one_minus_nmae + ficr),
        "one_minus_nmae": one_minus_nmae,
        "ficr": ficr,
    }


def _merge_parent_and_policies(fold_id: str, parent: pd.DataFrame) -> pd.DataFrame:
    keys = ["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh"]
    parent_fold = parent.loc[parent["fold_id"].eq(fold_id), [*keys, "prediction_kwh"]]
    policies = pd.read_parquet(_booster_path(fold_id))
    if policies[keys].duplicated().any() or parent_fold[keys].duplicated().any():
        raise RuntimeError(f"duplicate strict-blend key in {fold_id}")
    merged = policies.merge(
        parent_fold.rename(columns={"prediction_kwh": "parent_prediction_kwh"}),
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(parent_fold) or len(merged) != len(policies):
        raise RuntimeError(f"strict-blend row mismatch in {fold_id}")
    return merged


def _policy_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {*BASE_COLUMNS, "parent_prediction_kwh"}
    columns = [column for column in frame.columns if column not in excluded]
    if not columns:
        raise RuntimeError("no DART policy columns found")
    return columns


def _blend(
    frame: pd.DataFrame,
    policy: str,
    parent_weight: float,
) -> pd.DataFrame:
    output = frame[BASE_COLUMNS].copy()
    output["prediction_kwh"] = (
        parent_weight * frame["parent_prediction_kwh"]
        + (1.0 - parent_weight) * frame[policy]
    )
    return output


def _select(frame: pd.DataFrame, group_id: int) -> dict[str, object]:
    group = frame.loc[frame["group_id"].eq(group_id)]
    best: tuple[float, str, float, dict[str, float]] | None = None
    for policy in _policy_columns(frame):
        for parent_weight in PARENT_WEIGHT_GRID:
            score = _group_score(_blend(group, policy, parent_weight), group_id)
            choice = (score["total"], policy, parent_weight, score)
            if best is None or choice[0] > best[0]:
                best = choice
    assert best is not None
    return {
        "policy": best[1],
        "parent_weight": best[2],
        "q2_group_score": best[3],
    }


def main() -> None:
    candidate_id = CANDIDATE_ID
    parent = pd.read_parquet(PARENT)
    q2_candidates = _merge_parent_and_policies(FOLDS[0], parent)
    selections = {
        str(group_id): _select(q2_candidates, group_id)
        for group_id in CAPACITIES
    }

    # Q2 is only a policy-selection/control fold and remains the unchanged parent.
    q2 = parent.loc[
        parent["fold_id"].eq(FOLDS[0]), [*BASE_COLUMNS, "prediction_kwh"]
    ].copy()
    q2["fold_id"] = FOLDS[0]
    output_parts = [q2]
    fold_scores = {FOLDS[0]: _score(q2)}
    group_scores: dict[str, dict[str, dict[str, float]]] = {}

    for fold_id in FOLDS[1:]:
        candidates = _merge_parent_and_policies(fold_id, parent)
        fold_parts: list[pd.DataFrame] = []
        group_scores[fold_id] = {}
        for group_id in CAPACITIES:
            selection = selections[str(group_id)]
            group = candidates.loc[candidates["group_id"].eq(group_id)]
            output = _blend(
                group,
                str(selection["policy"]),
                float(selection["parent_weight"]),
            )
            output["fold_id"] = fold_id
            fold_parts.append(output)
            group_scores[fold_id][str(group_id)] = _group_score(output, group_id)
        fold = pd.concat(fold_parts, ignore_index=True)
        output_parts.append(fold)
        fold_scores[fold_id] = _score(fold)

    output = pd.concat(output_parts, ignore_index=True)
    output["model_id"] = candidate_id
    output_path = OUTPUT / f"{candidate_id}-oof.parquet"
    output[[*BASE_COLUMNS, "prediction_kwh", "fold_id", "model_id"]].to_parquet(
        output_path, index=False
    )
    booster_paths = {fold_id: _booster_path(fold_id) for fold_id in FOLDS}
    receipt = {
        "schema_version": 1,
        "candidate_id": candidate_id,
        "scope": (
            "Q2 policy and parent-weight selection with unchanged Q2 control; "
            f"selected {BOOSTER_ID} blend frozen unchanged for Q3-Q4"
        ),
        "parent_candidate_id": "M107_STRICT_TEMPORAL_TOP100",
        "parent_prediction_sha256": _sha256(PARENT),
        "booster_candidate_id": BOOSTER_ID,
        "booster_policy_sha256": {
            fold_id: _sha256(path) for fold_id, path in booster_paths.items()
        },
        "selection_fold": FOLDS[0],
        "selected_iteration": SELECTED_ITERATION,
        "parent_weight_grid": list(PARENT_WEIGHT_GRID),
        "selections": selections,
        "fold_scores": fold_scores,
        "group_scores": group_scores,
        "pooled": _score(output),
        "prediction_path": str(output_path.relative_to(REPO)),
        "prediction_sha256": _sha256(output_path),
        "observed_scada_feature_count": 0,
        "new_2024_evaluation": False,
        "lockbox_reopened": False,
        "external_actions": [],
    }
    receipt_path = OUTPUT / f"{candidate_id}-oof.json"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
