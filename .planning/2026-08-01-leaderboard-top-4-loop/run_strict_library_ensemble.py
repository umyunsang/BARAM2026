"""Chronology-safe groupwise ensemble over independently trained model families."""

from __future__ import annotations

import argparse
import itertools
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
STRICT_PATHS = {
    "control": OUTPUT / "M107_STRICT_TEMPORAL_TOP100-oof.parquet",
    "dart": OUTPUT / "M114_STRICT_DART_BLEND-oof.parquet",
    "xgboost": OUTPUT / "M116_STRICT_XGBOOST_BLEND-oof.parquet",
    "ordinal": OUTPUT / "M100_STRICT_ORDINAL-oof.parquet",
}
Q2_CALIBRATION_PATHS = {
    "control": OUTPUT / "M107_STRICT_TEMPORAL_TOP100-oof.parquet",
    "dart": OUTPUT / "M113_LGBM_DART-dev-2023-Q2.parquet",
    "xgboost": OUTPUT / "M115_XGBOOST-dev-2023-Q2.parquet",
    "ordinal": OUTPUT / "M98_ORDINAL_BIN025-dev-2023-Q2.parquet",
}
WEIGHTS = tuple(float(value) for value in np.linspace(0.0, 1.0, 21))
SHIFTS = tuple(float(value) for value in np.arange(-0.03, 0.0301, 0.0025))


def _load_fold(path: Path, fold_id: str) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if "fold_id" in frame:
        frame = frame.loc[frame["fold_id"].eq(fold_id)]
    if frame[KEYS].duplicated().any():
        raise RuntimeError(f"duplicate candidate key: {path.name} / {fold_id}")
    return frame[[*KEYS, "prediction_kwh"]].copy()


def _library(paths: dict[str, Path], fold_id: str) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    for name, path in paths.items():
        frame = _load_fold(path, fold_id).rename(
            columns={"prediction_kwh": f"prediction__{name}"}
        )
        if merged is None:
            merged = frame
        else:
            merged = merged.merge(frame, on=KEYS, validate="one_to_one")
    assert merged is not None
    return merged.sort_values(["forecast_kst_dtm", "group_id"]).reset_index(drop=True)


def _group_total(actual: np.ndarray, prediction: np.ndarray) -> float:
    valid = np.isfinite(actual) & (actual >= 0.10)
    actual = actual[valid]
    error = np.abs(prediction[valid] - actual)
    units = np.select([error <= 0.06, error <= 0.08], [4.0, 3.0], default=0.0)
    return 0.5 * (
        1.0
        - float(error.mean())
        + float(np.sum(actual * units) / np.sum(actual * 4.0))
    )


def _select_group(group: pd.DataFrame) -> dict[str, object]:
    group_id = int(group["group_id"].iloc[0])
    capacity = CAPACITIES[group_id]
    actual = group["actual_kwh"].to_numpy(dtype=float) / capacity
    names = tuple(STRICT_PATHS)
    predictions = {
        name: group[f"prediction__{name}"].to_numpy(dtype=float) / capacity
        for name in names
    }
    best: tuple[float, float, str, str, float, bool, float] | None = None
    for left, right in itertools.combinations_with_replacement(names, 2):
        for weight in WEIGHTS:
            base = weight * predictions[left] + (1.0 - weight) * predictions[right]
            for shift in SHIFTS:
                shifted = np.clip(base + shift, 0.0, 1.075)
                for snap in (False, True):
                    candidate = (
                        np.round(shifted / 0.0025) * 0.0025 if snap else shifted
                    )
                    score = _group_total(actual, candidate)
                    choice = (
                        score,
                        -abs(shift),
                        left,
                        right,
                        weight,
                        snap,
                        shift,
                    )
                    if best is None or choice > best:
                        best = choice
    assert best is not None
    return {
        "left": best[2],
        "right": best[3],
        "left_weight": best[4],
        "signed_shift": best[6],
        "snap": best[5],
        "calibration_total": best[0],
    }


def _select(calibration: pd.DataFrame) -> dict[int, dict[str, object]]:
    return {
        int(group_id): _select_group(group)
        for group_id, group in calibration.groupby("group_id", sort=True)
    }


def _apply(
    library: pd.DataFrame,
    selections: dict[int, dict[str, object]],
    fold_id: str,
    candidate_id: str,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for group_id, group in library.groupby("group_id", sort=True):
        selection = selections[int(group_id)]
        capacity = CAPACITIES[int(group_id)]
        normalized = (
            float(selection["left_weight"])
            * group[f"prediction__{selection['left']}"]
            + (1.0 - float(selection["left_weight"]))
            * group[f"prediction__{selection['right']}"]
        ) / capacity
        normalized = np.clip(
            normalized + float(selection["signed_shift"]), 0.0, 1.075
        )
        if bool(selection["snap"]):
            normalized = np.round(normalized / 0.0025) * 0.0025
        output = group[KEYS].copy()
        output["prediction_kwh"] = normalized * capacity
        output["fold_id"] = fold_id
        output["model_id"] = candidate_id
        parts.append(output)
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-id", default="M134_STRICT_LIBRARY_ENSEMBLE")
    args = parser.parse_args()
    if _sha256(OPEN) != OPEN_SHA or _sha256(BASELINE) != BASELINE_SHA:
        raise RuntimeError("immutable competition input hash mismatch")
    started = time.perf_counter()
    candidate_id = args.candidate_id

    strict = {fold: _library(STRICT_PATHS, fold) for fold in FOLDS}
    q2_calibration = _library(Q2_CALIBRATION_PATHS, FOLDS[0])
    q3_selection = _select(q2_calibration)
    q3 = _apply(strict[FOLDS[1]], q3_selection, FOLDS[1], candidate_id)
    q4_selection = _select(strict[FOLDS[1]])
    q4 = _apply(strict[FOLDS[2]], q4_selection, FOLDS[2], candidate_id)
    control = _load_fold(STRICT_PATHS["control"], FOLDS[0])
    q2 = control.assign(fold_id=FOLDS[0], model_id=candidate_id)
    output = pd.concat([q2, q3, q4], ignore_index=True)
    output_path = OUTPUT / f"{candidate_id}-oof.parquet"
    output.to_parquet(output_path, index=False)
    receipt = {
        "schema_version": 1,
        "candidate_id": candidate_id,
        "scope": "Q3 selected on Q2; Q4 selected on Q3; Q2 unchanged control",
        "q3_selection": q3_selection,
        "q4_selection": q4_selection,
        "fold_scores": {
            fold: _score(output.loc[output["fold_id"].eq(fold)]) for fold in FOLDS
        },
        "pooled": _score(output),
        "source_sha256": {
            name: _sha256(path) for name, path in STRICT_PATHS.items()
        },
        "runtime_seconds": round(time.perf_counter() - started, 2),
        "prediction_path": str(output_path.relative_to(Path.cwd())),
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
