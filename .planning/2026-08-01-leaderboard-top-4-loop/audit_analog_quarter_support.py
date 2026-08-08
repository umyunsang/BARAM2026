"""Audit quarter-level support for the frozen M234 analog recipes.

The audit is label-free on the official test surface.  It compares each query
day with the leave-one-out neighbor-distance distribution of the historical
analog library in the same fitted representation.  The consumed 2024 lockbox
is never loaded or scored.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from build_robust_analog_test_challenger import CACHE, _test_surface
from run_conditional_daily_analog_profile import (
    BASELINE,
    BASELINE_SHA,
    OPEN,
    OPEN_SHA,
    REPRESENTATIONS,
    Recipe,
    Representation,
    _complete_group_days,
    _cyclic_doy,
    _feature_sets,
    _representation_matrix,
)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from strict_dev_surface import DEV_CUTOFF, development_surface

from baram.contracts.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "artifacts" / "backtests" / "metric-aligned-probe"
M231 = ROOT / "artifacts" / "submissions" / "E0_V2_SEQUENCE_TRANSFER-a5111cee22bc.csv"
M234 = ROOT / "artifacts" / "submissions" / "E0_ROBUST_ANALOG_DEV-08f07b6d9e87.csv"
M234_RECEIPT = M234.with_suffix(".receipt.json")
REPORT = OUTPUT / "M236_ANALOG_QUARTER_SUPPORT_AUDIT.json"
CAPACITIES = {1: 21600.0, 2: 21600.0, 3: 21000.0}
VALIDATION_QUARTERS = (2, 3, 4)
TEST_QUARTERS = (1, 2, 3, 4)


def _projected_distances(
    train_values: np.ndarray,
    query_values: np.ndarray,
    train_issuances: np.ndarray,
    query_issuances: np.ndarray,
    representation: Representation,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    """Return query and train-LOO distances under one train-fitted space."""
    train = _representation_matrix(train_values, representation.mode)
    query = _representation_matrix(query_values, representation.mode)
    median = np.nanmedian(train, axis=0)
    median = np.where(np.isfinite(median), median, 0.0)
    train = np.where(np.isfinite(train), train, median)
    query = np.where(np.isfinite(query), query, median)

    scaler = StandardScaler()
    train = np.clip(scaler.fit_transform(train), -8.0, 8.0)
    query = np.clip(scaler.transform(query), -8.0, 8.0)
    components = min(representation.components, len(train) - 1, train.shape[1])
    if components != representation.components:
        raise RuntimeError("analog support PCA component contract changed")
    pca = PCA(
        n_components=components,
        whiten=True,
        svd_solver="randomized",
        random_state=20260803,
    )
    train = pca.fit_transform(train)
    query = pca.transform(query)
    if representation.season_weight:
        train = np.column_stack(
            [train, representation.season_weight * _cyclic_doy(train_issuances)]
        )
        query = np.column_stack(
            [query, representation.season_weight * _cyclic_doy(query_issuances)]
        )

    query_squared = (
        np.sum(query * query, axis=1, keepdims=True)
        + np.sum(train * train, axis=1)[None, :]
        - 2.0 * query @ train.T
    )
    train_squared = (
        np.sum(train * train, axis=1, keepdims=True)
        + np.sum(train * train, axis=1)[None, :]
        - 2.0 * train @ train.T
    )
    query_squared = np.maximum(query_squared, 0.0)
    train_squared = np.maximum(train_squared, 0.0)
    np.fill_diagonal(train_squared, np.inf)
    return (
        np.sqrt(np.sort(query_squared, axis=1)),
        np.sqrt(np.sort(train_squared, axis=1)),
        {
            "components": components,
            "explained_variance_ratio_sum": float(
                pca.explained_variance_ratio_.sum()
            ),
            "input_dimensions": int(
                _representation_matrix(train_values, representation.mode).shape[1]
            ),
        },
    )


def _percentile(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    ordered = np.sort(reference)
    return np.searchsorted(ordered, values, side="right") / len(ordered)


def _distribution(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "q50": float(np.quantile(values, 0.50)),
        "q75": float(np.quantile(values, 0.75)),
        "q90": float(np.quantile(values, 0.90)),
        "q95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
    }


def _support_summary(
    query_distance: np.ndarray,
    train_loo_distance: np.ndarray,
    neighbors: int,
) -> dict[str, object]:
    if query_distance.shape[1] < neighbors or train_loo_distance.shape[1] < neighbors:
        raise RuntimeError("neighbor count exceeds support audit matrix")
    nearest = query_distance[:, 0]
    kth = query_distance[:, neighbors - 1]
    train_nearest = train_loo_distance[:, 0]
    train_kth = train_loo_distance[:, neighbors - 1]
    nearest_percentile = _percentile(train_nearest, nearest)
    kth_percentile = _percentile(train_kth, kth)
    return {
        "days": len(query_distance),
        "nearest_distance": _distribution(nearest),
        "recipe_kth_distance": _distribution(kth),
        "nearest_train_loo_percentile": {
            **_distribution(nearest_percentile),
            "share_above_0p95": float(np.mean(nearest_percentile > 0.95)),
            "share_above_0p99": float(np.mean(nearest_percentile > 0.99)),
        },
        "recipe_kth_train_loo_percentile": {
            **_distribution(kth_percentile),
            "share_above_0p95": float(np.mean(kth_percentile > 0.95)),
            "share_above_0p99": float(np.mean(kth_percentile > 0.99)),
        },
        "nearest_to_train_loo_median_ratio": float(
            np.median(nearest) / np.median(train_nearest)
        ),
        "recipe_kth_to_train_loo_median_ratio": float(
            np.median(kth) / np.median(train_kth)
        ),
    }


def _operating_quarters(frame: pd.DataFrame, issuances: np.ndarray) -> np.ndarray:
    operating = (
        frame.groupby("data_available_kst_dtm", sort=True)["forecast_kst_dtm"]
        .min()
        .reindex(pd.to_datetime(issuances))
    )
    if operating.isna().any():
        raise RuntimeError("operating-day quarter alignment changed")
    return pd.DatetimeIndex(operating).quarter.to_numpy(dtype=int)


def _validation_support(
    surface: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    group_id: int,
    recipe: Recipe,
    representation: Representation,
) -> dict[str, object]:
    frame, issuances, values, targets = _complete_group_days(
        surface,
        group_id,
        feature_sets[representation.feature_set],
    )
    operating_quarter = _operating_quarters(frame, issuances)
    operating_year = (
        frame.groupby("data_available_kst_dtm", sort=True)["forecast_kst_dtm"]
        .min()
        .reindex(pd.to_datetime(issuances))
        .dt.year.to_numpy(dtype=int)
    )
    day_end = (
        frame.groupby("data_available_kst_dtm", sort=True)["forecast_kst_dtm"]
        .max()
        .reindex(pd.to_datetime(issuances))
        .to_numpy()
    )
    output: dict[str, object] = {}
    for quarter in VALIDATION_QUARTERS:
        query_mask = (operating_year == 2023) & (operating_quarter == quarter)
        if not query_mask.any():
            raise RuntimeError(f"group {group_id} Q{quarter} query is empty")
        cutoff = pd.Timestamp(np.min(issuances[query_mask]))
        train_mask = (day_end < cutoff) & np.isfinite(targets).all(axis=1)
        query_distance, train_loo_distance, projection = _projected_distances(
            values[train_mask],
            values[query_mask],
            issuances[train_mask],
            issuances[query_mask],
            representation,
        )
        output[f"q{quarter}"] = {
            "training_days": int(train_mask.sum()),
            "projection": projection,
            "support": _support_summary(
                query_distance,
                train_loo_distance,
                recipe.neighbors,
            ),
        }
    return output


def _test_support(
    surface: pd.DataFrame,
    test: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    group_id: int,
    recipe: Recipe,
    representation: Representation,
) -> dict[str, object]:
    feature_names = feature_sets[representation.feature_set]
    columns = [
        "forecast_id",
        "forecast_kst_dtm",
        "data_available_kst_dtm",
        "group_id",
        "actual_kwh",
        *feature_names,
    ]
    analog_surface = pd.concat(
        [surface[columns], test[columns]],
        ignore_index=True,
    )
    frame, issuances, values, targets = _complete_group_days(
        analog_surface,
        group_id,
        feature_names,
    )
    operating_year = (
        frame.groupby("data_available_kst_dtm", sort=True)["forecast_kst_dtm"]
        .min()
        .reindex(pd.to_datetime(issuances))
        .dt.year.to_numpy(dtype=int)
    )
    query_mask = operating_year == 2025
    if int(query_mask.sum()) != 365:
        raise RuntimeError(f"group {group_id} test issuance contract changed")
    cutoff = pd.Timestamp(np.min(issuances[query_mask]))
    day_end = (
        frame.groupby("data_available_kst_dtm", sort=True)["forecast_kst_dtm"]
        .max()
        .reindex(pd.to_datetime(issuances))
        .to_numpy()
    )
    train_mask = (day_end < cutoff) & np.isfinite(targets).all(axis=1)
    query_distance, train_loo_distance, projection = _projected_distances(
        values[train_mask],
        values[query_mask],
        issuances[train_mask],
        issuances[query_mask],
        representation,
    )
    quarters = _operating_quarters(frame, issuances[query_mask])
    output: dict[str, object] = {
        "training_days": int(train_mask.sum()),
        "query_days": int(query_mask.sum()),
        "projection": projection,
        "all": _support_summary(
            query_distance,
            train_loo_distance,
            recipe.neighbors,
        ),
    }
    for quarter in TEST_QUARTERS:
        mask = quarters == quarter
        output[f"q{quarter}"] = _support_summary(
            query_distance[mask],
            train_loo_distance,
            recipe.neighbors,
        )
    return output


def _correction_exposure(test: pd.DataFrame) -> dict[str, object]:
    parent = pd.read_csv(M231, encoding="utf-8-sig")
    analog = pd.read_csv(M234, encoding="utf-8-sig")
    key_columns = ["forecast_id", "forecast_kst_dtm"]
    parent["forecast_kst_dtm"] = pd.to_datetime(parent["forecast_kst_dtm"])
    analog["forecast_kst_dtm"] = pd.to_datetime(analog["forecast_kst_dtm"])
    if not parent[key_columns].equals(analog[key_columns]):
        raise RuntimeError("M231/M234 submission key order changed")
    metadata = (
        test.loc[test["group_id"].eq(1), [*key_columns, "data_available_kst_dtm"]]
        .drop_duplicates(key_columns)
        .copy()
    )
    metadata["operating_day"] = metadata.groupby("data_available_kst_dtm")[
        "forecast_kst_dtm"
    ].transform("min")
    metadata["quarter"] = pd.to_datetime(metadata["operating_day"]).dt.quarter
    merged = parent.merge(metadata[[*key_columns, "quarter"]], on=key_columns)
    output: dict[str, object] = {}
    for group_id, capacity in CAPACITIES.items():
        column = f"kpx_group_{group_id}"
        delta = analog[column].to_numpy(dtype=float) - parent[column].to_numpy(
            dtype=float
        )
        group_output: dict[str, object] = {}
        for quarter in TEST_QUARTERS:
            mask = merged["quarter"].to_numpy(dtype=int) == quarter
            absolute = np.abs(delta[mask])
            group_output[f"q{quarter}"] = {
                "rows": int(mask.sum()),
                "mean_signed_delta_kwh": float(np.mean(delta[mask])),
                "mean_absolute_delta_kwh": float(np.mean(absolute)),
                "q90_absolute_delta_kwh": float(np.quantile(absolute, 0.90)),
                "max_absolute_delta_kwh": float(np.max(absolute)),
                "mean_absolute_delta_capacity_fraction": float(
                    np.mean(absolute) / capacity
                ),
                "changed_row_fraction": float(np.mean(absolute > 1e-9)),
            }
        output[str(group_id)] = group_output
    return output


def _aggregate_test_support(groups: dict[str, object]) -> dict[str, object]:
    output: dict[str, object] = {}
    for quarter in TEST_QUARTERS:
        nearest = [
            groups[str(group_id)]["test"][f"q{quarter}"][
                "nearest_train_loo_percentile"
            ]["q50"]
            for group_id in CAPACITIES
        ]
        kth = [
            groups[str(group_id)]["test"][f"q{quarter}"][
                "recipe_kth_train_loo_percentile"
            ]["q50"]
            for group_id in CAPACITIES
        ]
        output[f"q{quarter}"] = {
            "median_nearest_percentile_across_groups": float(np.median(nearest)),
            "median_recipe_kth_percentile_across_groups": float(np.median(kth)),
        }
    return output


def main() -> None:
    if sha256_file(OPEN) != OPEN_SHA or sha256_file(BASELINE) != BASELINE_SHA:
        raise RuntimeError("immutable competition input hash mismatch")
    receipt = json.loads(M234_RECEIPT.read_text(encoding="utf-8"))
    if sha256_file(M234) != receipt["submission_receipt"]["csv_sha256"]:
        raise RuntimeError("M234 CSV hash mismatch")
    surface, numeric = development_surface()
    if surface["forecast_kst_dtm"].ge(DEV_CUTOFF).any():
        raise RuntimeError("lockbox row reached analog support audit")
    feature_sets = _feature_sets(numeric)
    test_features = sorted(set(feature_sets["core"]) | set(feature_sets["extended"]))
    test = _test_surface(test_features)

    recipes = {
        int(group_id): Recipe(**recipe)
        for group_id, recipe in receipt["policy"]["recipes"].items()
    }
    representations = {item.name: item for item in REPRESENTATIONS}
    groups: dict[str, object] = {}
    for group_id, recipe in recipes.items():
        representation = representations[recipe.representation]
        groups[str(group_id)] = {
            "recipe": asdict(recipe),
            "validation": _validation_support(
                surface,
                feature_sets,
                group_id,
                recipe,
                representation,
            ),
            "test": _test_support(
                surface,
                test,
                feature_sets,
                group_id,
                recipe,
                representation,
            ),
        }

    aggregate = _aggregate_test_support(groups)
    report = {
        "schema_version": 1,
        "audit_id": "M236_ANALOG_QUARTER_SUPPORT_AUDIT",
        "purpose": "label-free deployment support audit; not model selection evidence",
        "groups": groups,
        "aggregate_test_support": aggregate,
        "correction_exposure": _correction_exposure(test),
        "interpretation_contract": {
            "distance_reference": "same-fit historical train-day leave-one-out distribution",
            "q1_performance_validated": False,
            "q1_feature_support_only": True,
            "online_score": None,
            "no_external_upload": True,
            "new_2024_evaluation": False,
            "lockbox_reopened": False,
        },
        "inputs": {
            "open_zip_sha256": OPEN_SHA,
            "baseline_ipynb_sha256": BASELINE_SHA,
            "m231_csv_sha256": sha256_file(M231),
            "m234_csv_sha256": sha256_file(M234),
            "m234_receipt_sha256": sha256_file(M234_RECEIPT),
            "test_feature_cache_sha256": sha256_file(CACHE / "test_features.parquet"),
        },
    }
    REPORT.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
