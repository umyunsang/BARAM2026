"""Bounded chronology-safe metric-aligned experiment runner.

This file is active experiment state, not a production inference entry point.  It
uses only the prepared competition caches and the frozen 2023 development folds.
The 2024 lockbox is intentionally absent from every code path.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import lightgbm
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from baram.contracts.hashing import sha256_dataframe, sha256_file
from baram.evaluation.official import evaluate_official

REPO = Path(__file__).resolve().parents[2]
CACHE = REPO / "artifacts/cache/920be0c458d820e855bf79dd25723146f52ce1736138aedba5e6bc853f1f720b"
OUTPUT = REPO / "artifacts/backtests/metric-aligned-probe"
CAPACITIES = {1: 21_600.0, 2: 21_600.0, 3: 21_000.0}
FOLDS = ("dev-2023-Q2", "dev-2023-Q3", "dev-2023-Q4")
METRIC_COLUMNS = [
    "forecast_id",
    "forecast_kst_dtm",
    "group_id",
    "actual_kwh",
    "prediction_kwh",
]


def _feature_names(base: pd.DataFrame, geometric: pd.DataFrame, profile: str) -> list[str]:
    metadata = {
        "forecast_kst_dtm",
        "data_available_kst_dtm",
        "operating_day",
        "issuance_batch",
        "forecast_id",
        "manufacturer",
        "model",
    }
    base_numeric = [
        name
        for name in base
        if name not in metadata and pd.api.types.is_numeric_dtype(base[name])
    ]
    geometric_numeric = [
        name
        for name in geometric
        if name not in {"forecast_kst_dtm", "data_available_kst_dtm", "group_id"}
        and pd.api.types.is_numeric_dtype(geometric[name])
    ]
    if profile == "all":
        selected = base_numeric
    elif profile == "wind_geometry":
        static = {
            "operating_year",
            "operating_quarter",
            "hour",
            "month",
            "day_of_year",
            "lead_hour",
            "group_id",
            "capacity_kwh",
            "turbine_count",
            "latitude_centroid",
            "longitude_centroid",
            "hub_height_m",
            "rotor_diameter_m",
            "turbine_capacity_mw",
            "group_capacity_mw",
            "rotor_swept_area_m2",
            "fleet_swept_area_m2",
        }
        tokens = ("speed", "dir_", "phys", "cal__", "source_disagreement")
        selected = [
            name
            for name in base_numeric
            if name in static or any(token in name for token in tokens)
        ]
    else:
        raise ValueError(f"unknown feature profile: {profile}")
    return [*selected, *geometric_numeric]


def _load_surface(profile: str) -> tuple[pd.DataFrame, list[str]]:
    base = pd.read_parquet(CACHE / "train_features.parquet")
    geometric = pd.read_parquet(CACHE / "train_geometric.parquet")
    names = _feature_names(base, geometric, profile)
    joined = base.merge(
        geometric,
        on=["forecast_kst_dtm", "data_available_kst_dtm", "group_id"],
        how="inner",
        validate="one_to_one",
        sort=False,
        suffixes=("", "__duplicate"),
    )
    if len(joined) != len(base) or joined[names].shape[1] != len(names):
        raise RuntimeError("feature join changed the prepared row or column contract")
    labels = pd.read_parquet(CACHE / "labels_long.parquet")
    joined = joined.merge(
        labels[["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh"]],
        on=["forecast_id", "forecast_kst_dtm", "group_id"],
        how="left",
        validate="one_to_one",
        sort=False,
    )
    return joined, names


def _validation_keys() -> pd.DataFrame:
    reference = pd.read_parquet(OUTPUT / "M50_GEOM_CLASS_TEMP075_G05-oof.parquet")
    return reference[["forecast_id", "group_id", "fold_id"]].copy()


def _score_dict(frame: pd.DataFrame) -> dict[str, object]:
    return asdict(evaluate_official(frame[METRIC_COLUMNS], CAPACITIES))


def _fit_one_fold(
    surface: pd.DataFrame,
    feature_names: list[str],
    validation_keys: pd.DataFrame,
    fold_id: str,
    *,
    objective: str,
    alpha: float,
    min_actual_fraction: float,
    weight_power: float,
    num_leaves: int,
) -> tuple[pd.DataFrame, int, float]:
    fold_keys = validation_keys.loc[validation_keys["fold_id"].eq(fold_id)]
    validation = surface.merge(
        fold_keys,
        on=["forecast_id", "group_id"],
        how="inner",
        validate="one_to_one",
        sort=False,
    )
    cutoff = validation["data_available_kst_dtm"].min()
    train = surface.loc[surface["data_available_kst_dtm"].lt(cutoff)].dropna(
        subset=["actual_kwh"]
    )
    capacity = train["group_id"].map(CAPACITIES).to_numpy(dtype=float)
    eligible = train["actual_kwh"].to_numpy(dtype=float) >= min_actual_fraction * capacity
    train = train.loc[eligible].reset_index(drop=True)
    capacity = train["group_id"].map(CAPACITIES).to_numpy(dtype=float)
    target = train["actual_kwh"].to_numpy(dtype=float) / capacity
    weights = np.power(np.maximum(target, 1e-6), weight_power)

    batches = list(dict.fromkeys(train["issuance_batch"].astype(str)))
    stop_count = max(1, int(np.ceil(len(batches) * 0.20)))
    stop_batches = set(batches[-stop_count:])
    stop_mask = train["issuance_batch"].astype(str).isin(stop_batches).to_numpy()
    fit_mask = ~stop_mask
    params: dict[str, object] = {}
    if objective == "quantile":
        params["alpha"] = alpha
    model = LGBMRegressor(
        objective=objective,
        n_estimators=2400,
        learning_rate=0.025,
        num_leaves=num_leaves,
        min_child_samples=40,
        max_bin=255,
        subsample=0.90,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.10,
        reg_lambda=2.0,
        random_state=20260802,
        n_jobs=6,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
        **params,
    )
    started = time.perf_counter()
    model.fit(
        train.loc[fit_mask, feature_names],
        target[fit_mask],
        sample_weight=weights[fit_mask],
        eval_set=[(train.loc[stop_mask, feature_names], target[stop_mask])],
        eval_sample_weight=[weights[stop_mask]],
        eval_metric="l1",
        callbacks=[lightgbm.early_stopping(120, verbose=False)],
    )
    iterations = max(1, int(model.best_iteration_ or 2400))
    refit = model.set_params(n_estimators=iterations)
    refit.fit(train[feature_names], target, sample_weight=weights)
    valid_capacity = validation["group_id"].map(CAPACITIES).to_numpy(dtype=float)
    prediction = np.clip(
        refit.predict(validation[feature_names]) * valid_capacity,
        0.0,
        1.05 * valid_capacity,
    )
    result = validation[["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh"]].copy()
    result["prediction_kwh"] = prediction
    result["fold_id"] = fold_id
    return result, iterations, time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--profile", choices=("all", "wind_geometry"), required=True)
    parser.add_argument("--objective", choices=("l1", "quantile"), default="l1")
    parser.add_argument("--alpha", type=float, default=0.50)
    parser.add_argument("--min-actual-fraction", type=float, default=0.10)
    parser.add_argument("--weight-power", type=float, default=0.0)
    parser.add_argument("--num-leaves", type=int, choices=(15, 31, 63), default=31)
    parser.add_argument("--folds", nargs="+", choices=FOLDS, default=[FOLDS[-1]])
    args = parser.parse_args()
    if not 0.0 <= args.min_actual_fraction <= 0.20:
        raise ValueError("min actual fraction must be in [0, 0.20]")
    if not 0.0 <= args.weight_power <= 2.0:
        raise ValueError("weight power must be in [0, 2]")

    surface, feature_names = _load_surface(args.profile)
    validation_keys = _validation_keys()
    predictions: list[pd.DataFrame] = []
    iterations: dict[str, int] = {}
    runtimes: dict[str, float] = {}
    fold_scores: dict[str, object] = {}
    for fold_id in args.folds:
        part, selected, runtime = _fit_one_fold(
            surface,
            feature_names,
            validation_keys,
            fold_id,
            objective=args.objective,
            alpha=args.alpha,
            min_actual_fraction=args.min_actual_fraction,
            weight_power=args.weight_power,
            num_leaves=args.num_leaves,
        )
        part["model_id"] = args.candidate_id
        predictions.append(part)
        iterations[fold_id] = selected
        runtimes[fold_id] = round(runtime, 2)
        fold_scores[fold_id] = _score_dict(part)
        print(json.dumps({"fold": fold_id, "score": fold_scores[fold_id]}), flush=True)

    combined = pd.concat(predictions, ignore_index=True).sort_values(
        ["forecast_kst_dtm", "group_id"], kind="stable"
    )
    suffix = "oof" if tuple(args.folds) == FOLDS else "q4"
    prediction_path = OUTPUT / f"{args.candidate_id}-{suffix}.parquet"
    receipt_path = OUTPUT / f"{args.candidate_id}-{suffix}.json"
    combined.to_parquet(prediction_path, index=False)
    receipt = {
        "schema_version": 1,
        "candidate_id": args.candidate_id,
        "scope": "2023 chronology-safe metric-aligned OOF only",
        "specification": {
            "profile": args.profile,
            "objective": args.objective,
            "alpha": args.alpha,
            "min_actual_fraction": args.min_actual_fraction,
            "weight_power": args.weight_power,
            "num_leaves": args.num_leaves,
        },
        "feature_count": len(feature_names),
        "fold_scores": fold_scores,
        "pooled": _score_dict(combined) if tuple(args.folds) == FOLDS else None,
        "selected_iterations": iterations,
        "runtime_seconds_by_fold": runtimes,
        "prediction_path": str(prediction_path.relative_to(REPO)),
        "prediction_sha256": sha256_dataframe(combined.reset_index(drop=True)),
        "immutable_hashes": {
            "open": sha256_file(Path("/Users/um-yunsang/Downloads/open.zip")),
            "baseline": sha256_file(Path("/Users/um-yunsang/Downloads/baseline.ipynb")),
        },
        "new_2024_evaluation": False,
        "lockbox_reopened": False,
        "external_actions": [],
    }
    receipt_path.write_text(
        json.dumps(receipt, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
