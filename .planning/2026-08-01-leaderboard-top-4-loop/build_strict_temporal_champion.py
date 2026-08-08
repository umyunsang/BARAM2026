"""Select issuance-local phase corrections on Q2 and freeze them for Q3-Q4."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from run_sequence_classifier import OUTPUT, _score, _sha256

REPO = Path(__file__).resolve().parents[2]
CACHE = (
    REPO
    / "artifacts/cache"
    / "920be0c458d820e855bf79dd25723146f52ce1736138aedba5e6bc853f1f720b"
)
PARENT = OUTPUT / "M103_STRICT_TOP100-oof.parquet"
METADATA = CACHE / "train_features.parquet"
FOLDS = ("dev-2023-Q2", "dev-2023-Q3", "dev-2023-Q4")
CAPACITIES = {1: 21_600.0, 2: 21_600.0, 3: 21_000.0}
BASE_COLUMNS = ["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh"]
SHIFT_GRID = (-2, -1, 1, 2)
ORIGINAL_WEIGHT_GRID = tuple(round(value, 1) for value in np.arange(0.5, 1.01, 0.1))


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


def _apply_policy(
    frame: pd.DataFrame,
    shift_hours: int,
    original_weight: float,
) -> pd.DataFrame:
    output = frame.sort_values(
        ["data_available_kst_dtm", "forecast_kst_dtm"]
    ).copy()
    shifted = (
        output.groupby("data_available_kst_dtm", sort=False)["prediction_kwh"]
        .shift(shift_hours)
        .fillna(output["prediction_kwh"])
    )
    output["prediction_kwh"] = (
        original_weight * output["prediction_kwh"]
        + (1.0 - original_weight) * shifted
    )
    return output


def _select_policy(frame: pd.DataFrame, group_id: int) -> dict[str, object]:
    best: tuple[float, int, float, dict[str, float]] | None = None
    for shift_hours in SHIFT_GRID:
        for original_weight in ORIGINAL_WEIGHT_GRID:
            candidate = _apply_policy(frame, shift_hours, original_weight)
            score = _group_score(candidate, group_id)
            choice = (score["total"], shift_hours, original_weight, score)
            if best is None or choice[0] > best[0]:
                best = choice
    assert best is not None
    return {
        "shift_hours": best[1],
        "original_weight": best[2],
        "q2_group_score": best[3],
    }


def main() -> None:
    candidate_id = "M107_STRICT_TEMPORAL_TOP100"
    parent = pd.read_parquet(PARENT)
    metadata = pd.read_parquet(
        METADATA,
        columns=[
            "forecast_id",
            "forecast_kst_dtm",
            "group_id",
            "data_available_kst_dtm",
        ],
    )
    keys = ["forecast_id", "forecast_kst_dtm", "group_id"]
    parent = parent.merge(metadata, on=keys, validate="one_to_one")
    q2 = parent.loc[parent["fold_id"].eq(FOLDS[0])]
    selections = {
        str(group_id): _select_policy(
            q2.loc[q2["group_id"].eq(group_id)], group_id
        )
        for group_id in CAPACITIES
    }

    # Q2 remains an untouched policy-selection/control fold. The selected policies
    # are applied unchanged to both later folds, with no Q3/Q4 feedback.
    output_parts = [q2.copy()]
    fold_scores = {FOLDS[0]: _score(q2)}
    group_scores: dict[str, dict[str, dict[str, float]]] = {}
    for fold_id in FOLDS[1:]:
        fold_parts: list[pd.DataFrame] = []
        group_scores[fold_id] = {}
        for group_id in CAPACITIES:
            selection = selections[str(group_id)]
            source = parent.loc[
                parent["fold_id"].eq(fold_id)
                & parent["group_id"].eq(group_id)
            ]
            transformed = _apply_policy(
                source,
                int(selection["shift_hours"]),
                float(selection["original_weight"]),
            )
            fold_parts.append(transformed)
            group_scores[fold_id][str(group_id)] = _group_score(
                transformed, group_id
            )
        fold = pd.concat(fold_parts, ignore_index=True)
        output_parts.append(fold)
        fold_scores[fold_id] = _score(fold)

    output = pd.concat(output_parts, ignore_index=True)
    output["model_id"] = candidate_id
    output_path = OUTPUT / f"{candidate_id}-oof.parquet"
    output[[*BASE_COLUMNS, "prediction_kwh", "fold_id", "model_id"]].to_parquet(
        output_path, index=False
    )
    receipt = {
        "schema_version": 1,
        "candidate_id": candidate_id,
        "scope": (
            "Q2 parent control and policy selection; per-group issuance-local "
            "phase policy frozen unchanged for Q3-Q4"
        ),
        "parent_candidate_id": "M103_STRICT_TOP100",
        "parent_prediction_sha256": _sha256(PARENT),
        "metadata_sha256": _sha256(METADATA),
        "selection_fold": FOLDS[0],
        "shift_grid": list(SHIFT_GRID),
        "original_weight_grid": list(ORIGINAL_WEIGHT_GRID),
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
