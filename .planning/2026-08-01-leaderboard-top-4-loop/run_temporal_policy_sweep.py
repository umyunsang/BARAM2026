"""Apply chronology-safe temporal calibration to the strict M68 candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from baram.evaluation.official import evaluate_official

REPO = Path(__file__).resolve().parents[2]
OUTPUT = REPO / "artifacts/backtests/metric-aligned-probe"
CACHE = (
    REPO
    / "artifacts/cache"
    / "920be0c458d820e855bf79dd25723146f52ce1736138aedba5e6bc853f1f720b"
)
FOLDS = ("dev-2023-Q2", "dev-2023-Q3", "dev-2023-Q4")
CAPACITIES = {1: 21_600.0, 2: 21_600.0, 3: 21_000.0}
METRIC_COLUMNS = [
    "forecast_id",
    "forecast_kst_dtm",
    "group_id",
    "actual_kwh",
    "prediction_kwh",
]
SMOOTHING_GRID = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30)
SCALE_GRID = (0.96, 0.98, 1.00, 1.02, 1.04)
OFFSET_GRID = (-0.02, -0.01, 0.00, 0.01, 0.02)
CAP_GRID: tuple[str, ...] = ("none", "1.01", "capacity")

# These decisions were selected strictly from preceding folds before this stage.
STRICT_M68_POLICY = {
    "dev-2023-Q3": {
        1: ("T0.6_G0.75", 0.00, False),
        2: ("T0.6_G0.35", 0.05, True),
        3: ("T0.6_G1", 0.30, True),
    },
    "dev-2023-Q4": {
        1: ("T0.6_G0.5", 0.00, True),
        2: ("T0.6_G0.35", 0.05, False),
        3: ("T0.6_G0.75", 0.30, True),
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _score(frame: pd.DataFrame) -> dict[str, float]:
    result = evaluate_official(frame[METRIC_COLUMNS], CAPACITIES)
    return {
        "total": result.total,
        "one_minus_nmae": result.one_minus_nmae,
        "ficr": result.ficr,
    }


def _group_components(frame: pd.DataFrame, group_id: int) -> tuple[float, float, float]:
    capacity = CAPACITIES[group_id]
    valid = frame["actual_kwh"].to_numpy(dtype=float) >= 0.10 * capacity
    actual = frame.loc[valid, "actual_kwh"].to_numpy(dtype=float)
    prediction = frame.loc[valid, "prediction_kwh"].to_numpy(dtype=float)
    error = np.abs(prediction - actual) / capacity
    units = np.select([error <= 0.06, error <= 0.08], [4.0, 3.0], default=0.0)
    nmae = float(error.mean())
    ficr = float(np.sum(actual * units) / np.sum(actual * 4.0))
    return 0.5 * (1.0 - nmae) + 0.5 * ficr, nmae, ficr


def _strict_m68() -> pd.DataFrame:
    m50 = pd.read_parquet(OUTPUT / "M50_GEOM_CLASS_TEMP075_G05-oof.parquet")
    outputs = [m50.loc[m50["fold_id"].eq(FOLDS[0])].copy()]
    keys = ["forecast_id", "forecast_kst_dtm", "group_id"]
    for fold_id, group_policy in STRICT_M68_POLICY.items():
        policies = pd.read_parquet(
            OUTPUT / f"M68_SITEWIND_CLASS_ITER-{fold_id}-policies.parquet"
        )
        reference = (
            m50.loc[m50["fold_id"].eq(fold_id), [*keys, "prediction_kwh"]]
            .rename(columns={"prediction_kwh": "m50_prediction_kwh"})
        )
        policies = policies.merge(reference, on=keys, validate="one_to_one")
        prediction = np.empty(len(policies), dtype=float)
        for group_id, (policy, weight, snap) in group_policy.items():
            mask = policies["group_id"].eq(group_id).to_numpy()
            values = (
                (1.0 - weight) * policies.loc[mask, policy].to_numpy(dtype=float)
                + weight
                * policies.loc[mask, "m50_prediction_kwh"].to_numpy(dtype=float)
            )
            if snap:
                capacity = CAPACITIES[group_id]
                values = np.round(values / capacity / 0.0025) * 0.0025 * capacity
            prediction[mask] = values
        policies["prediction_kwh"] = prediction
        policies["fold_id"] = fold_id
        policies["model_id"] = "M68_STRICT_SEQUENTIAL"
        outputs.append(policies[[*METRIC_COLUMNS, "fold_id", "model_id"]])
    combined = pd.concat(outputs, ignore_index=True)
    combined["model_id"] = "M68_STRICT_SEQUENTIAL"
    return combined


def _add_issuance_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    metadata = pd.read_parquet(
        CACHE / "train_features.parquet",
        columns=[
            "forecast_id",
            "forecast_kst_dtm",
            "group_id",
            "data_available_kst_dtm",
        ],
    )
    keys = ["forecast_id", "forecast_kst_dtm", "group_id"]
    return frame.merge(metadata, on=keys, validate="one_to_one")


def _smooth_normalized(
    frame: pd.DataFrame,
    first_neighbor_weight: float,
    second_neighbor_weight: float,
) -> np.ndarray:
    # Resetting the row positions is essential: fold/group subsets retain sparse source indices.
    ordered = frame.reset_index(drop=True).copy()
    capacity = ordered["group_id"].map(CAPACITIES).to_numpy(dtype=float)
    normalized = ordered["prediction_kwh"].to_numpy(dtype=float) / capacity
    result = np.empty(len(ordered), dtype=float)
    for _, part in ordered.groupby("data_available_kst_dtm", sort=False):
        positions = part.index.to_numpy(dtype=int)
        values = normalized[positions]
        numerator = values.copy()
        denominator = np.ones(len(values), dtype=float)
        for distance, weight in (
            (1, first_neighbor_weight),
            (2, second_neighbor_weight),
        ):
            if weight == 0.0:
                continue
            numerator[distance:] += weight * values[:-distance]
            denominator[distance:] += weight
            numerator[:-distance] += weight * values[distance:]
            denominator[:-distance] += weight
        result[positions] = numerator / denominator
    return result


def _apply(
    frame: pd.DataFrame,
    policy: tuple[float, float, float, float, str],
) -> pd.DataFrame:
    first_weight, second_weight, scale, offset, cap_mode = policy
    output = frame.reset_index(drop=True).copy()
    normalized = _smooth_normalized(output, first_weight, second_weight)
    normalized = normalized * scale + offset
    upper = {"none": 1.075, "1.01": 1.01, "capacity": 1.0}[cap_mode]
    normalized = np.clip(normalized, 0.0, upper)
    capacity = output["group_id"].map(CAPACITIES).to_numpy(dtype=float)
    output["prediction_kwh"] = normalized * capacity
    return output


def _select_group_policy(
    history: pd.DataFrame,
    group_id: int,
) -> tuple[tuple[float, float, float, float, str], dict[str, float]]:
    group = history.loc[history["group_id"].eq(group_id)].reset_index(drop=True)
    best: tuple[float, tuple[float, float, float, float, str], float, float] | None = None
    for first_weight in SMOOTHING_GRID:
        for second_weight in SMOOTHING_GRID:
            if second_weight > first_weight:
                continue
            smoothed = _smooth_normalized(group, first_weight, second_weight)
            capacity = CAPACITIES[group_id]
            for scale in SCALE_GRID:
                for offset in OFFSET_GRID:
                    base = smoothed * scale + offset
                    for cap_mode in CAP_GRID:
                        upper = {"none": 1.075, "1.01": 1.01, "capacity": 1.0}[
                            cap_mode
                        ]
                        trial = group.copy()
                        trial["prediction_kwh"] = np.clip(base, 0.0, upper) * capacity
                        total, nmae, ficr = _group_components(trial, group_id)
                        policy = (
                            first_weight,
                            second_weight,
                            scale,
                            offset,
                            cap_mode,
                        )
                        choice = (total, policy, nmae, ficr)
                        if best is None or choice[0] > best[0]:
                            best = choice
    assert best is not None
    return best[1], {"total": best[0], "nmae": best[2], "ficr": best[3]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-id", required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    strict = _add_issuance_metadata(_strict_m68())
    strict_score = _score(strict)
    output_parts = [strict.loc[strict["fold_id"].eq(FOLDS[0])].copy()]
    selections: dict[str, object] = {}
    for fold_index in (1, 2):
        fold_id = FOLDS[fold_index]
        history_folds = FOLDS[:fold_index]
        history = strict.loc[strict["fold_id"].isin(history_folds)]
        application = strict.loc[strict["fold_id"].eq(fold_id)].copy()
        fold_selections: dict[str, object] = {}
        transformed_parts: list[pd.DataFrame] = []
        for group_id in CAPACITIES:
            policy, calibration_score = _select_group_policy(history, group_id)
            transformed = _apply(
                application.loc[application["group_id"].eq(group_id)], policy
            )
            transformed_parts.append(transformed)
            fold_selections[str(group_id)] = {
                "policy": list(policy),
                "preceding_fold_score": calibration_score,
            }
            print(
                json.dumps(
                    {
                        "application_fold": fold_id,
                        "preceding_folds": list(history_folds),
                        "group_id": group_id,
                        **fold_selections[str(group_id)],
                    }
                ),
                flush=True,
            )
        transformed_fold = pd.concat(transformed_parts, ignore_index=True)
        output_parts.append(transformed_fold)
        selections[fold_id] = fold_selections
    combined = pd.concat(output_parts, ignore_index=True)
    combined["model_id"] = args.candidate_id
    output_path = OUTPUT / f"{args.candidate_id}-oof.parquet"
    output_columns = [*METRIC_COLUMNS, "fold_id", "model_id"]
    combined[output_columns].to_parquet(output_path, index=False)
    receipt = {
        "schema_version": 1,
        "candidate_id": args.candidate_id,
        "parent_candidate_id": "M68_STRICT_SEQUENTIAL",
        "scope": "Q2 raw; Q3 calibrated on Q2; Q4 calibrated on pooled Q2-Q3",
        "strict_parent_score": strict_score,
        "selections": selections,
        "fold_scores": {
            fold_id: _score(combined.loc[combined["fold_id"].eq(fold_id)])
            for fold_id in FOLDS
        },
        "pooled": _score(combined),
        "runtime_seconds": round(time.perf_counter() - started, 2),
        "prediction_path": str(output_path.relative_to(REPO)),
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
