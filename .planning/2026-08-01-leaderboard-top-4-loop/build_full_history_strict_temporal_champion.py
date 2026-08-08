"""Build M261 from the frozen M102/M107 policy on all supplied history.

This is a deployment-only final fit.  The script may use every supplied
2022-2024 label as a training row, but it never scores, slices, compares, or
selects on 2024.  Test inference uses only supplied GFS/LDAPS weather and the
supplied turbine workbook; observed SCADA is a teacher target, never an
inference feature.
"""

from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from pyarrow import parquet as pq
from run_sequence_classifier import (
    BASELINE,
    BASELINE_SHA,
    CAPACITIES,
    OPEN,
    OPEN_SHA,
    _surface,
)
from run_site_wind_teacher import _all_weather_columns
from sklearn.model_selection import KFold

from baram.contracts.hashing import canonical_sha256, sha256_file
from baram.data.archive import read_csv_member, read_info_workbook
from baram.data.canonical import canonicalize_weather
from baram.data.turbines import parse_turbine_workbook
from baram.features.geometric import build_geometric_wind_features
from baram.features.weather import _GFS_VECTOR_PAIRS, _LDAPS_VECTOR_PAIRS
from baram.submission.build import build_submission
from baram.submission.validate import validate_submission
from baram.workflows import _wide_predictions

REPO = Path(__file__).resolve().parents[2]
CACHE = REPO / "artifacts/cache" / OPEN_SHA
EVIDENCE = REPO / "artifacts/backtests/metric-aligned-probe"
SUBMISSIONS = REPO / "artifacts/submissions"

CANDIDATE_ID = "M261_FULL_HISTORY_STRICT_TEMPORAL_TOP100"
CANDIDATE_PATH = SUBMISSIONS / "submission_M261.csv"
RECEIPT_PATH = SUBMISSIONS / "submission_M261.receipt.json"
M102_RECEIPT = EVIDENCE / "M102_TOP100-dev-2023-Q2.json"
M107_RECEIPT = EVIDENCE / "M107_STRICT_TEMPORAL_TOP100-oof.json"
M107_PREDICTION = EVIDENCE / "M107_STRICT_TEMPORAL_TOP100-oof.parquet"
M64B_RECEIPT = EVIDENCE / "M64B_ALLWEATHER_SITEWIND_CLASS-dev-2023-Q2.json"
M252_ONLINE_RECEIPT = REPO / "reports/dacon_m252_online_2026-08-03_receipt.json"

FROZEN_HASHES = {
    "m102_receipt": "17f7ea69f4eca3e1c9500ae2df6f6c2361a6eeca9d8a953551825f476d5b06eb",
    "m107_receipt": "0167aac129b2afd1a004a3612d32bda7d0916757fc20a38a805950a8d92b93ea",
    "m107_prediction": "3539cada59f88a16d4b4181f5aff3c76ff8e9a94954f67f4204ccd09ac8e537d",
    "m64b_receipt": "853122a6952d0995c2f7894797d58e330caded3ff66ec7493910d14713c60ca0",
    "m252_online_receipt": "25694ade390a9386f7d1188bde59314da74c6339a53d8ed60f9b34fc2e4e57f6",
    "train_features": "fec1b0369f6d1c63a3d9e113190c45bb2b84d1c6500098b957c96a6b1d2128e7",
    "test_features": "7aab7538f9cb25d93b0aaeb6882eeb6746c69f2623fb65022a926111daf5948b",
    "train_grid_pivot": "aee0a4e934a23fa058063e4271135eef0add23563ae84296a3226701d1587996",
    "train_geometric": "765b3c3c4d9998d58f61c47698b7de272631c489f48cd5d8f30f14743c60b709",
    "labels_long": "880a6f9368023d1fdab541fb82a7133f18b15c3974d0fd7528c08e6ff8badab2",
    "submission_keys": "f675805cb030ce6d401c735194bc48965cd398a1abd0b4bea21ec546a750ddb7",
}

FROZEN_HELPER_HASHES = {
    "run_sequence_classifier.py": (
        "3a86e4accb470671288819f59df7afaf4956181f69ed9b7bc632466743d98aed"
    ),
    "run_site_wind_classifier.py": (
        "a97193a84970b07c7d55c795688ab27a92fc219247e8e9cc08724cbcc91ce5ef"
    ),
    "run_site_wind_teacher.py": (
        "8749791de2ed005396f238131acf8f53033c61c95a5178c8b1020db2e6faa43b"
    ),
    "run_classifier_iteration_sweep.py": (
        "8f200fd7a11b0bdd820a1155a4e5274e309519aaa750ddeb548be68274b12143"
    ),
    "build_strict_temporal_champion.py": (
        "7dfc568091bba906242d02d6eb15763e5518d9205690fc27725e9e2c2c95b8ab"
    ),
    "run_inner_policy_classifier.py": (
        "29eae0af39221c1193dce59dd513a51f9e928815d5dbe65428fc937a5dac22a9"
    ),
}

FEATURE_SOURCE_HASHES = {
    "src/baram/features/geometric.py": (
        "86e977ed4167c3f9229d0ef017da678cf709fdf39fe774f908a2edc1182406d4"
    ),
    "src/baram/features/weather.py": (
        "d65a18fdf0b025d1863c783d5997f77824a3347ab87e78a4357fe2afeb2d1905"
    ),
    "src/baram/submission/build.py": (
        "b2f73750c583006d872e735b77bfaf3287b3a1275ac6c39d7449c32c0a80cb6c"
    ),
    "src/baram/submission/validate.py": (
        "7285d48e8c3952d94d2789bdf1676c8116d56e197cd0ffcc1c6e95b5fc78ca15"
    ),
}

SITEWIND_ITERATIONS = {1: 395, 2: 170, 3: 164}
CLASS_WIDTH = 0.02
CLASSIFIER_ITERATIONS = 60
ACTION_TEMPERATURE = 0.5
ACTION_GAMMA = 1.5
TEMPORAL_POLICY = {
    1: {"shift_hours": -1, "original_weight": 0.7},
    2: {"shift_hours": -2, "original_weight": 0.8},
    3: {"shift_hours": -2, "original_weight": 0.8},
}


def _array_sha256(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(np.asarray(contiguous.shape, dtype="int64").tobytes())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _expected_paths() -> dict[str, Path]:
    return {
        "m102_receipt": M102_RECEIPT,
        "m107_receipt": M107_RECEIPT,
        "m107_prediction": M107_PREDICTION,
        "m64b_receipt": M64B_RECEIPT,
        "m252_online_receipt": M252_ONLINE_RECEIPT,
        "train_features": CACHE / "train_features.parquet",
        "test_features": CACHE / "test_features.parquet",
        "train_grid_pivot": CACHE / "train_grid_pivot.parquet",
        "train_geometric": CACHE / "train_geometric.parquet",
        "labels_long": CACHE / "labels_long.parquet",
        "submission_keys": CACHE / "submission_keys.parquet",
    }


def _direct_score_calls() -> list[str]:
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    forbidden = {"evaluate_official", "_score"}
    calls: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            name = node.func.attr
        else:
            continue
        if name in forbidden:
            calls.append(f"{name}:{node.lineno}")
    return calls


def _validate_frozen_evidence() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if sha256_file(OPEN) != OPEN_SHA or sha256_file(BASELINE) != BASELINE_SHA:
        raise RuntimeError("immutable competition input hash mismatch")
    for name, path in _expected_paths().items():
        observed = sha256_file(path)
        if observed != FROZEN_HASHES[name]:
            raise RuntimeError(f"M261 frozen input changed: {name}={observed}")
    helper_root = Path(__file__).parent
    for name, expected in FROZEN_HELPER_HASHES.items():
        observed = sha256_file(helper_root / name)
        if observed != expected:
            raise RuntimeError(f"M261 frozen helper changed: {name}={observed}")
    for relative, expected in FEATURE_SOURCE_HASHES.items():
        observed = sha256_file(REPO / relative)
        if observed != expected:
            raise RuntimeError(f"M261 feature/submission source changed: {relative}={observed}")
    if _direct_score_calls():
        raise RuntimeError(f"M261 contains direct score calls: {_direct_score_calls()}")

    m102 = _json(M102_RECEIPT)
    m107 = _json(M107_RECEIPT)
    m64b = _json(M64B_RECEIPT)
    expected_bins = list(range(46))
    if (
        m102.get("candidate_id") != "M102_TOP100"
        or m102.get("fold_id") != "dev-2023-Q2"
        or m102.get("feature_count") != 100
        or m102.get("selected_iteration") != CLASSIFIER_ITERATIONS
        or m102.get("best_policy") != "T0.5_G1.5"
        or m102.get("class_width") != CLASS_WIDTH
        or m102.get("active_bins") != expected_bins
        or m102.get("generation_weight_power") != 1.0
        or m102.get("recency_half_life_days") != 0.0
        or m102.get("seasonal_quarter_boost") != 0.0
        or m102.get("group3_weight") != 1.0
    ):
        raise RuntimeError("M102 frozen classifier contract changed")
    selected = m102.get("selected_feature_names")
    if not isinstance(selected, list) or len(selected) != 100 or len(set(selected)) != 100:
        raise RuntimeError("M102 selected-feature contract changed")
    if any("scada" in str(name).lower() for name in selected):
        raise RuntimeError("M102 selected features expose observed SCADA")
    if (
        m107.get("candidate_id") != "M107_STRICT_TEMPORAL_TOP100"
        or m107.get("parent_candidate_id") != "M103_STRICT_TOP100"
        or m107.get("selection_fold") != "dev-2023-Q2"
    ):
        raise RuntimeError("M107 frozen lineage contract changed")
    # M107 also stores q2_group_score under each selection; compare only the
    # deployment parameters that were frozen for M261.
    selections = m107.get("selections", {})
    compact = {
        int(group): {
            "shift_hours": int(value["shift_hours"]),
            "original_weight": float(value["original_weight"]),
        }
        for group, value in selections.items()
    }
    if compact != TEMPORAL_POLICY:
        raise RuntimeError("M107 frozen temporal-policy contract changed")
    if m107.get("new_2024_evaluation") or m107.get("lockbox_reopened"):
        raise RuntimeError("M107 evidence boundary changed")
    if m64b.get("sitewind_selected_iterations") != {
        str(group): value for group, value in SITEWIND_ITERATIONS.items()
    }:
        raise RuntimeError("M64B frozen site-wind iteration contract changed")
    return m102, m107, m64b


def _source_variable_names(reference: pd.DataFrame, prefix: str) -> list[str]:
    result: list[str] = []
    for name in reference.columns:
        if not name.startswith(f"{prefix}__"):
            continue
        variable = name.split("__", 2)[2]
        if variable not in result:
            result.append(variable)
    return result


def _source_grid_columns(reference: pd.DataFrame, prefix: str) -> list[str]:
    return [name for name in reference.columns if name.startswith(f"{prefix}__")]


def _build_source_grid(
    weather: pd.DataFrame,
    reference: pd.DataFrame,
    prefix: str,
    vector_pairs: Mapping[str, tuple[str, str]],
) -> pd.DataFrame:
    variables = _source_variable_names(reference, prefix)
    prepared = weather.copy()
    for alias, (u_column, v_column) in vector_pairs.items():
        feature = f"{alias}_speed"
        if feature in variables:
            prepared[feature] = np.hypot(
                prepared[u_column].to_numpy(dtype=float),
                prepared[v_column].to_numpy(dtype=float),
            )
    missing = sorted(set(variables).difference(prepared.columns))
    if missing:
        raise RuntimeError(f"M261 {prefix} grid variables are missing: {missing}")
    grid_ids = sorted(prepared["grid_id"].unique())
    parts: list[pd.DataFrame] = []
    for variable in variables:
        pivot = prepared.pivot(
            index="forecast_kst_dtm", columns="grid_id", values=variable
        ).sort_index()
        pivot = pivot.reindex(columns=grid_ids)
        pivot.columns = [
            f"{prefix}__grid{position:02d}__{variable}" for position in range(1, len(grid_ids) + 1)
        ]
        parts.append(pivot)
    result = pd.concat(parts, axis=1).reset_index()
    expected = ["forecast_kst_dtm", *_source_grid_columns(reference, prefix)]
    if list(result.columns) != expected:
        raise RuntimeError(f"M261 {prefix} raw-grid column contract changed")
    values = result.drop(columns="forecast_kst_dtm").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise RuntimeError(f"M261 {prefix} raw-grid values are non-finite after repair")
    return result


def _verify_train_grid_recipe(reference: pd.DataFrame) -> None:
    timestamps = reference["forecast_kst_dtm"].iloc[:8]
    gfs = canonicalize_weather(read_csv_member(OPEN, "train/gfs_train.csv"))
    ldaps = canonicalize_weather(read_csv_member(OPEN, "train/ldaps_train.csv"))
    gfs = gfs.loc[gfs["forecast_kst_dtm"].isin(timestamps)]
    ldaps = ldaps.loc[ldaps["forecast_kst_dtm"].isin(timestamps)]
    rebuilt_gfs = _build_source_grid(gfs, reference, "gfs", _GFS_VECTOR_PAIRS)
    rebuilt_ldaps = _build_source_grid(ldaps, reference, "ldaps", _LDAPS_VECTOR_PAIRS)
    rebuilt = rebuilt_gfs.merge(
        rebuilt_ldaps,
        on="forecast_kst_dtm",
        validate="one_to_one",
        sort=True,
    )[reference.columns]
    expected = reference.loc[reference["forecast_kst_dtm"].isin(timestamps)].reset_index(drop=True)
    rebuilt = rebuilt.reset_index(drop=True)
    if not rebuilt["forecast_kst_dtm"].equals(expected["forecast_kst_dtm"]):
        raise RuntimeError("M261 train-grid timestamp reconstruction changed")
    observed = rebuilt.drop(columns="forecast_kst_dtm").to_numpy(dtype=float)
    frozen = expected.drop(columns="forecast_kst_dtm").to_numpy(dtype=float)
    if not np.array_equal(observed, frozen, equal_nan=True):
        raise RuntimeError("M261 train-grid value reconstruction changed")


def _impute_weather_within_issuance(
    weather: pd.DataFrame, source: str
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Repair supplied NWP gaps using only adjacent leads from the same issuance/grid."""
    identifiers = {
        "grid_id",
        "latitude",
        "longitude",
        "operating_year",
        "operating_quarter",
        "lead_hour",
    }
    value_columns = [
        name
        for name in weather.select_dtypes(include=[np.number]).columns
        if name not in identifiers
    ]
    missing_before = {
        name: int(weather[name].isna().sum())
        for name in value_columns
        if weather[name].isna().any()
    }
    if not missing_before:
        return weather, {
            "source": source,
            "method": "same_issuance_same_grid_linear_with_bidirectional_edge_fill",
            "missing_cells_before": 0,
            "missing_cells_after": 0,
            "features": {},
        }
    result = weather.assign(_original_order=np.arange(len(weather), dtype="int64"))
    result = result.sort_values(["data_available_kst_dtm", "grid_id", "lead_hour"], kind="stable")
    grouped = result.groupby(["data_available_kst_dtm", "grid_id"], sort=False)
    for name in missing_before:
        repaired = grouped[name].transform(
            lambda values: values.interpolate(method="linear", limit_direction="both")
        )
        result[name] = result[name].fillna(repaired)
    result = result.sort_values("_original_order", kind="stable").drop(columns="_original_order")
    missing_after = {name: int(result[name].isna().sum()) for name in missing_before}
    unresolved = {name: count for name, count in missing_after.items() if count}
    if unresolved:
        raise RuntimeError(f"M261 {source} weather repair left gaps: {unresolved}")
    return result, {
        "source": source,
        "method": "same_issuance_same_grid_linear_with_bidirectional_edge_fill",
        "missing_cells_before": int(sum(missing_before.values())),
        "missing_cells_after": 0,
        "features": missing_before,
    }


def _build_test_weather_derivatives() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    reference_grid = pd.read_parquet(CACHE / "train_grid_pivot.parquet")
    gfs = canonicalize_weather(read_csv_member(OPEN, "test/gfs_test.csv"))
    ldaps = canonicalize_weather(read_csv_member(OPEN, "test/ldaps_test.csv"))
    gfs_keys = (
        gfs[["forecast_kst_dtm", "data_available_kst_dtm"]].drop_duplicates().reset_index(drop=True)
    )
    ldaps_keys = (
        ldaps[["forecast_kst_dtm", "data_available_kst_dtm"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    if not gfs_keys.equals(ldaps_keys):
        raise RuntimeError("M261 test GFS/LDAPS time keys changed")
    gfs, gfs_repair = _impute_weather_within_issuance(gfs, "gfs")
    ldaps, ldaps_repair = _impute_weather_within_issuance(ldaps, "ldaps")
    grid_gfs = _build_source_grid(gfs, reference_grid, "gfs", _GFS_VECTOR_PAIRS)
    grid_ldaps = _build_source_grid(ldaps, reference_grid, "ldaps", _LDAPS_VECTOR_PAIRS)
    grid = grid_gfs.merge(
        grid_ldaps,
        on="forecast_kst_dtm",
        validate="one_to_one",
        sort=True,
    )[reference_grid.columns]
    if len(grid) != 8760:
        raise RuntimeError(f"M261 test raw-grid row count changed: {len(grid)}")

    turbines = parse_turbine_workbook(read_info_workbook(OPEN))
    geometric = build_geometric_wind_features(gfs, ldaps, turbines, temporal_context=True)
    reference_geometric_columns = pq.ParquetFile(
        CACHE / "train_geometric.parquet"
    ).schema_arrow.names
    if list(geometric.columns) != reference_geometric_columns:
        raise RuntimeError("M261 train/test geometric column contract changed")
    if len(geometric) != 26280:
        raise RuntimeError(f"M261 test geometric row count changed: {len(geometric)}")
    return grid, geometric, {"gfs": gfs_repair, "ldaps": ldaps_repair}


def _test_surface(grid: pd.DataFrame, geometric: pd.DataFrame) -> pd.DataFrame:
    features = pd.read_parquet(CACHE / "test_features.parquet")
    for frame in (features, grid, geometric):
        frame["forecast_kst_dtm"] = pd.to_datetime(frame["forecast_kst_dtm"])
    support_columns = [
        name
        for name in features
        if name.startswith(
            (
                "gfs_spatial__",
                "ldaps_spatial__",
                "source_disagreement__",
                "phys__",
                "phys_v2__",
            )
        )
    ]
    calendar_columns = [
        name
        for name in (
            "hour",
            "month",
            "day_of_year",
            "lead_hour",
            "cal__hour_sin",
            "cal__hour_cos",
            "cal__doy_sin",
            "cal__doy_cos",
            "turbine_count",
            "latitude_centroid",
            "longitude_centroid",
            "hub_height_m",
            "rotor_diameter_m",
            "turbine_capacity_mw",
            "group_capacity_mw",
            "rotor_swept_area_m2",
            "fleet_swept_area_m2",
        )
        if name in features
    ]
    surface = (
        features[
            [
                "forecast_id",
                "forecast_kst_dtm",
                "data_available_kst_dtm",
                "issuance_batch",
                "group_id",
                *support_columns,
                *calendar_columns,
            ]
        ]
        .merge(grid, on="forecast_kst_dtm", validate="many_to_one")
        .merge(
            geometric,
            on=["forecast_kst_dtm", "data_available_kst_dtm", "group_id"],
            validate="one_to_one",
        )
    )
    for group_id in CAPACITIES:
        surface[f"group_{group_id}"] = surface["group_id"].eq(group_id).astype("int8")
    if len(surface) != 26280:
        raise RuntimeError(f"M261 test surface row count changed: {len(surface)}")
    if surface.duplicated(["forecast_id", "forecast_kst_dtm", "group_id"]).any():
        raise RuntimeError("M261 test surface contains duplicate keys")
    if "actual_kwh" in surface or "scada_ws" in surface:
        raise RuntimeError("M261 test surface exposes an observed target")
    return surface


def _legacy_sitewind(
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    train_matrix = train[columns].astype("float32")
    test_matrix = test[columns].astype("float32")
    eligible = train["scada_ws"].notna().to_numpy()
    positions = np.flatnonzero(eligible)
    train_prediction = np.full(len(train), np.nan, dtype="float32")
    params = {
        "objective": "l2",
        "n_estimators": 400,
        "learning_rate": 0.04,
        "num_leaves": 31,
        "min_child_samples": 60,
        "subsample": 0.9,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 2.0,
        "random_state": 20260801,
        "n_jobs": 6,
        "deterministic": True,
        "force_col_wise": True,
        "verbosity": -1,
    }
    splitter = KFold(3, shuffle=True, random_state=20260801)
    for fit_index, holdout_index in splitter.split(positions):
        model = LGBMRegressor(**params)
        model.fit(
            train_matrix.iloc[positions[fit_index]],
            train["scada_ws"].iloc[positions[fit_index]],
        )
        train_prediction[positions[holdout_index]] = model.predict(
            train_matrix.iloc[positions[holdout_index]]
        )
    final_model = LGBMRegressor(**params)
    final_model.fit(train_matrix.loc[eligible], train.loc[eligible, "scada_ws"])
    test_prediction = final_model.predict(test_matrix).astype("float32")
    return train_prediction, test_prediction


def _allweather_sitewind(
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    train_matrix = train[columns].astype("float32")
    test_matrix = test[columns].astype("float32")
    train_prediction = np.full(len(train), np.nan, dtype="float32")
    test_prediction = np.full(len(test), np.nan, dtype="float32")
    for group_id in CAPACITIES:
        train_group = train["group_id"].eq(group_id).to_numpy()
        test_group = test["group_id"].eq(group_id).to_numpy()
        eligible = train_group & train["scada_ws"].notna().to_numpy()
        positions = np.flatnonzero(eligible)
        params = {
            "objective": "l2",
            "n_estimators": SITEWIND_ITERATIONS[group_id],
            "learning_rate": 0.025,
            "num_leaves": 63,
            "min_child_samples": 40,
            "max_bin": 255,
            "subsample": 0.9,
            "subsample_freq": 1,
            "colsample_bytree": 0.85,
            "reg_alpha": 0.1,
            "reg_lambda": 3.0,
            "random_state": 20260802,
            "n_jobs": 6,
            "deterministic": True,
            "force_col_wise": True,
            "verbosity": -1,
        }
        splitter = KFold(3, shuffle=True, random_state=20260802 + group_id)
        for fit_index, holdout_index in splitter.split(positions):
            model = LGBMRegressor(**params)
            model.fit(
                train_matrix.iloc[positions[fit_index]],
                train["scada_ws"].iloc[positions[fit_index]],
            )
            train_prediction[positions[holdout_index]] = model.predict(
                train_matrix.iloc[positions[holdout_index]]
            )
        final_model = LGBMRegressor(**params)
        final_model.fit(train_matrix.loc[eligible], train.loc[eligible, "scada_ws"])
        test_prediction[test_group] = final_model.predict(test_matrix.loc[test_group]).astype(
            "float32"
        )
    return train_prediction, test_prediction


def _add_sitewind_features(
    matrix: pd.DataFrame,
    legacy: np.ndarray,
    allweather: np.ndarray,
) -> list[str]:
    matrix["sitewind__legacy"] = legacy
    matrix["sitewind__allweather"] = allweather
    matrix["sitewind__mean"] = (legacy + allweather) / 2.0
    matrix["sitewind__delta"] = allweather - legacy
    matrix["sitewind__disagreement"] = np.abs(allweather - legacy)
    for source in ("legacy", "allweather", "mean"):
        value = matrix[f"sitewind__{source}"]
        matrix[f"sitewind__{source}2"] = value**2
        matrix[f"sitewind__{source}3"] = value**3
        normalized = np.clip((value - 3.0) / 9.0, 0.0, 1.0)
        matrix[f"sitewind__{source}_powercurve"] = normalized**3
    return [name for name in matrix if name.startswith("sitewind__")]


def _fixed_actions(
    probability: np.ndarray,
    centers: np.ndarray,
    groups: np.ndarray,
    mean_generation: dict[int, float],
) -> np.ndarray:
    actions = np.arange(0.075, 1.076, 0.0025)
    error = np.abs(actions[:, None] - centers[None, :])
    units = np.select([error <= 0.06, error <= 0.08], [4.0, 3.0], default=0.0)
    calibrated = probability ** (1.0 / ACTION_TEMPERATURE)
    calibrated /= calibrated.sum(axis=1, keepdims=True)
    chosen = np.empty(len(probability), dtype=float)
    for group_id in CAPACITIES:
        mask = groups == group_id
        group_probability = calibrated[mask]
        utility = -(group_probability @ error.T) + ACTION_GAMMA * (
            group_probability @ (centers[None, :] * units).T
        ) / (4.0 * mean_generation[group_id])
        chosen[mask] = actions[np.argmax(utility, axis=1)]
    return chosen


def _apply_temporal_policy(frame: pd.DataFrame) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for group_id, selection in TEMPORAL_POLICY.items():
        part = (
            frame.loc[frame["group_id"].eq(group_id)]
            .sort_values(["data_available_kst_dtm", "forecast_kst_dtm"])
            .copy()
        )
        shifted = (
            part.groupby("data_available_kst_dtm", sort=False)["prediction_normalized"]
            .shift(int(selection["shift_hours"]))
            .fillna(part["prediction_normalized"])
        )
        weight = float(selection["original_weight"])
        part["prediction_normalized"] = (
            weight * part["prediction_normalized"] + (1.0 - weight) * shifted
        )
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def _fit_classifier(
    train: pd.DataFrame,
    test: pd.DataFrame,
    selected_features: list[str],
    train_legacy: np.ndarray,
    test_legacy: np.ndarray,
    train_allweather: np.ndarray,
    test_allweather: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    non_sitewind = [name for name in selected_features if not name.startswith("sitewind__")]
    missing_train = sorted(set(non_sitewind).difference(train.columns))
    missing_test = sorted(set(non_sitewind).difference(test.columns))
    if missing_train or missing_test:
        raise RuntimeError(
            f"M261 selected base features are missing: train={missing_train}, test={missing_test}"
        )
    train_matrix = train[non_sitewind].astype("float32")
    test_matrix = test[non_sitewind].astype("float32")
    train_sitewind = _add_sitewind_features(train_matrix, train_legacy, train_allweather)
    test_sitewind = _add_sitewind_features(test_matrix, test_legacy, test_allweather)
    if train_sitewind != test_sitewind or len(train_sitewind) != 14:
        raise RuntimeError("M261 train/test site-wind feature contract changed")
    missing = sorted(set(selected_features).difference(train_matrix.columns))
    if missing:
        raise RuntimeError(f"M261 frozen selected features are missing: {missing}")
    train_matrix = train_matrix[selected_features]
    test_matrix = test_matrix[selected_features]

    normalized = train["actual_kwh"] / train["group_id"].map(CAPACITIES)
    training = train["actual_kwh"].notna().to_numpy() & normalized.ge(0.10).to_numpy()
    raw_bins = np.floor((normalized.clip(0.10, 1.074999) - 0.10) / CLASS_WIDTH).astype("Int64")
    active_bins = [
        int(value) for value in sorted(raw_bins.loc[training].dropna().astype(int).unique())
    ]
    if active_bins != list(range(46)):
        raise RuntimeError(f"M261 full-history active bins changed: {active_bins}")
    bin_to_class = {bin_id: class_id for class_id, bin_id in enumerate(active_bins)}
    classes = raw_bins.map(bin_to_class).astype("Int64")
    centers = np.asarray(
        [
            normalized.loc[training & classes.eq(class_id)].mean()
            for class_id in range(len(active_bins))
        ],
        dtype=float,
    )
    if not np.isfinite(centers).all():
        raise RuntimeError("M261 class centers contain non-finite values")
    params = {
        "objective": "multiclass",
        "num_class": len(active_bins),
        "n_estimators": CLASSIFIER_ITERATIONS,
        "learning_rate": 0.025,
        "num_leaves": 15,
        "min_child_samples": 80,
        "subsample": 0.9,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 2.0,
        "random_state": 20260801,
        "n_jobs": 6,
        "deterministic": True,
        "force_col_wise": True,
        "verbosity": -1,
    }
    classifier = LGBMClassifier(**params)
    sample_weight = normalized.loc[training].clip(lower=0.10).to_numpy(dtype=float)
    classifier.fit(
        train_matrix.loc[training],
        classes.loc[training].astype(int),
        sample_weight=sample_weight,
    )
    if classifier.classes_.tolist() != list(range(46)):
        raise RuntimeError("M261 fitted classifier class order changed")
    probability = classifier.predict_proba(test_matrix)
    mean_generation = {
        group_id: float(normalized.loc[training & train["group_id"].eq(group_id)].mean())
        for group_id in CAPACITIES
    }
    chosen = _fixed_actions(
        probability,
        centers,
        test["group_id"].to_numpy(dtype=int),
        mean_generation,
    )
    output = test[
        [
            "forecast_id",
            "forecast_kst_dtm",
            "data_available_kst_dtm",
            "group_id",
        ]
    ].copy()
    output["prediction_normalized"] = chosen
    output = _apply_temporal_policy(output)
    output["prediction_kwh"] = output["prediction_normalized"] * output["group_id"].map(
        CAPACITIES
    ).to_numpy(dtype=float)
    diagnostics = {
        "training_rows": int(training.sum()),
        "training_rows_by_group": {
            str(group): int((training & train["group_id"].eq(group).to_numpy()).sum())
            for group in CAPACITIES
        },
        "final_fit_operating_years": sorted(
            int(value)
            for value in (
                train.loc[training, "forecast_kst_dtm"] - pd.Timedelta(hours=1)
            ).dt.year.unique()
        ),
        "active_bins": active_bins,
        "class_counts": {
            str(class_id): int((training & classes.eq(class_id).to_numpy()).sum())
            for class_id in range(len(active_bins))
        },
        "class_centers_sha256": _array_sha256(centers),
        "mean_generation": {str(group): value for group, value in mean_generation.items()},
        "classifier_probability_sha256": _array_sha256(probability),
        "selected_feature_count": len(selected_features),
        "selected_feature_sha256": canonical_sha256(selected_features),
        "train_matrix_nonfinite_count": int(
            (~np.isfinite(train_matrix.to_numpy(dtype="float32"))).sum()
        ),
        "test_matrix_nonfinite_count": int(
            (~np.isfinite(test_matrix.to_numpy(dtype="float32"))).sum()
        ),
    }
    return output, diagnostics


def _artifact_kib() -> int:
    return (
        sum(path.stat().st_size for path in (REPO / "artifacts").rglob("*") if path.is_file())
        // 1024
    )


def _self_test() -> None:
    m102, m107, m64b = _validate_frozen_evidence()
    reference = pd.read_parquet(CACHE / "train_grid_pivot.parquet")
    _verify_train_grid_recipe(reference)
    result = {
        "state": "M261_SELF_TEST_PASS",
        "m102_feature_count": len(m102["selected_feature_names"]),
        "m107_pooled": m107["pooled"],
        "sitewind_iterations": m64b["sitewind_selected_iterations"],
        "direct_score_calls": _direct_score_calls(),
        "train_grid_recipe_rows_verified": 8,
    }
    print(json.dumps(result, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        return

    m102, m107, m64b = _validate_frozen_evidence()
    train, base_columns, auxiliary_columns = _surface()
    grid, geometric, weather_repair = _build_test_weather_derivatives()
    test = _test_surface(grid, geometric)
    del grid, geometric
    gc.collect()

    allweather_columns = _all_weather_columns(train)
    for profile, columns in (
        ("legacy", auxiliary_columns),
        ("allweather", allweather_columns),
        ("windgeom", base_columns),
    ):
        missing = sorted(set(columns).difference(test.columns))
        if missing:
            raise RuntimeError(f"M261 test {profile} feature parity failed: {missing}")

    print(
        json.dumps(
            {
                "stage": "feature_contract_pass",
                "train_rows": len(train),
                "test_rows": len(test),
                "legacy_features": len(auxiliary_columns),
                "allweather_features": len(allweather_columns),
                "windgeom_features": len(base_columns),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    train_legacy, test_legacy = _legacy_sitewind(train, test, auxiliary_columns)
    print(
        json.dumps(
            {
                "stage": "legacy_sitewind_complete",
                "train_feature_sha256": _array_sha256(train_legacy),
                "test_feature_sha256": _array_sha256(test_legacy),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    train_allweather, test_allweather = _allweather_sitewind(train, test, allweather_columns)
    print(
        json.dumps(
            {
                "stage": "allweather_sitewind_complete",
                "train_feature_sha256": _array_sha256(train_allweather),
                "test_feature_sha256": _array_sha256(test_allweather),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    selected_features = [str(value) for value in m102["selected_feature_names"]]
    output, fit_diagnostics = _fit_classifier(
        train,
        test,
        selected_features,
        train_legacy,
        test_legacy,
        train_allweather,
        test_allweather,
    )
    long = output[["forecast_id", "forecast_kst_dtm", "group_id", "prediction_kwh"]]
    wide = _wide_predictions(long)
    sample = pd.read_parquet(CACHE / "submission_keys.parquet")

    policy = {
        "architecture": "full_history_m102_top100_classifier_with_m107_temporal_policy",
        "selection_source": "Q2_frozen_M102_and_M107_evidence_only",
        "selected_feature_names": selected_features,
        "class_width": CLASS_WIDTH,
        "active_bins": list(range(46)),
        "classifier_iterations": CLASSIFIER_ITERATIONS,
        "classifier_parameters": {
            "objective": "multiclass",
            "learning_rate": 0.025,
            "num_leaves": 15,
            "min_child_samples": 80,
            "subsample": 0.9,
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 2.0,
            "random_state": 20260801,
            "n_jobs": 6,
            "deterministic": True,
            "force_col_wise": True,
        },
        "generation_weight_power": 1.0,
        "action_policy": "T0.5_G1.5",
        "sitewind_iterations": {str(group): value for group, value in SITEWIND_ITERATIONS.items()},
        "sitewind_crossfit": "three_fold_kfold_original_seeds",
        "sitewind_test_fit": "full_supplied_history",
        "temporal_policy": {str(group): value for group, value in TEMPORAL_POLICY.items()},
        "final_fit_label_years": [2022, 2023, 2024],
        "observed_scada_inference_features": [],
        "m102_receipt_sha256": FROZEN_HASHES["m102_receipt"],
        "m107_receipt_sha256": FROZEN_HASHES["m107_receipt"],
        "m64b_receipt_sha256": FROZEN_HASHES["m64b_receipt"],
    }
    policy_sha = canonical_sha256(policy)
    csv_sha = build_submission(sample, wide, CANDIDATE_PATH)
    validation = validate_submission(
        CANDIDATE_PATH,
        sample,
        candidate_id=CANDIDATE_ID,
        source_sha256=OPEN_SHA,
        champion_policy_sha256=policy_sha,
        cap_modes={1: "nonnegative_only", 2: "nonnegative_only", 3: "nonnegative_only"},
    )
    if validation.csv_sha256 != csv_sha:
        raise RuntimeError("M261 build and validation hashes differ")

    capacity = output["group_id"].map(CAPACITIES).to_numpy(dtype=float)
    normalized_prediction = output["prediction_kwh"].to_numpy(dtype=float) / capacity
    receipt = {
        "schema_version": 1,
        "state": "LOCAL_FROZEN_M107_POLICY_FULL_HISTORY_FINAL_FIT_BUILT_NOT_UPLOADED",
        "candidate_id": CANDIDATE_ID,
        "candidate_path": str(CANDIDATE_PATH.relative_to(REPO)),
        "policy": policy,
        "policy_sha256": policy_sha,
        "fit_diagnostics": fit_diagnostics,
        "sitewind_diagnostics": {
            "legacy_train_sha256": _array_sha256(train_legacy),
            "legacy_test_sha256": _array_sha256(test_legacy),
            "allweather_train_sha256": _array_sha256(train_allweather),
            "allweather_test_sha256": _array_sha256(test_allweather),
            "train_scada_observed_rows": int(train["scada_ws"].notna().sum()),
            "test_scada_observed_rows": 0,
            "observed_scada_feature_count": 0,
        },
        "test_weather_repair": weather_repair,
        "prediction_diagnostics": {
            "row_count": len(output),
            "normalized_min": float(normalized_prediction.min()),
            "normalized_max": float(normalized_prediction.max()),
            "above_capacity_rows": int((normalized_prediction > 1.0).sum()),
            "prediction_sha256": _array_sha256(output["prediction_kwh"].to_numpy()),
        },
        "submission_receipt": asdict(validation),
        "inputs": {
            "open_zip_sha256": OPEN_SHA,
            "baseline_ipynb_sha256": BASELINE_SHA,
            **FROZEN_HASHES,
            "helper_source_sha256": FROZEN_HELPER_HASHES,
            "feature_submission_source_sha256": FEATURE_SOURCE_HASHES,
            "builder_code_sha256": sha256_file(Path(__file__)),
        },
        "source_evidence": {
            "m102_best_policy": m102["best_policy"],
            "m102_selected_iteration": m102["selected_iteration"],
            "m107_pooled_development_diagnostic": m107["pooled"],
            "m107_selections": m107["selections"],
            "m64b_sitewind_iterations": m64b["sitewind_selected_iterations"],
            "m252_online_receipt_sha256": FROZEN_HASHES["m252_online_receipt"],
        },
        "evaluation_contract": {
            "target_total_strictly_greater_than": 0.66,
            "direct_score_calls": _direct_score_calls(),
            "score_function_calls": 0,
            "metrics_computed_on_2024": False,
            "2024_slice_or_comparison_created": False,
            "selection_after_final_fit": False,
            "parameter_search_after_final_fit": False,
            "local_score": None,
            "online_score": None,
            "target_status": "UNVERIFIED_REQUIRES_EXTERNAL_DACON_RESULT",
        },
        "artifact_budget": {
            "limit_kib": 6 * 1024 * 1024,
            "candidate_bytes": CANDIDATE_PATH.stat().st_size,
            "status": "PASS_AT_BUILD",
        },
        "online_score": None,
        "new_2024_evaluation": False,
        "lockbox_reopened": False,
        "no_external_upload": True,
        "external_actions": [],
    }
    if _artifact_kib() > receipt["artifact_budget"]["limit_kib"]:
        raise RuntimeError("M261 artifact budget exceeded")
    RECEIPT_PATH.write_text(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    print(json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
