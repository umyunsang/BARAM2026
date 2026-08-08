"""Fail-closed S17-N22 M115 terrain materialization and strict assessment."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from xgboost import XGBClassifier

from baram.constants import CAPACITIES_KWH
from baram.evaluation.official import evaluate_official
from baram.evaluation.prequential import run_prequential_protocol
from baram.loop.events import EventStore

FOLDS = ("dev-2023-Q2", "dev-2023-Q3", "dev-2023-Q4")
OUTER = ("dev-2023-Q3", "dev-2023-Q4")
FAMILY = (
    "CHAMPION",
    "M115_REFIT_ZERO",
    "TERRAIN_SX300_H8_M115_REPLACED",
)
KEYS = ("fold_id", "group_id", "forecast_kst_dtm")
TERRAIN_NAME = "terrain__sx300_h8_mean16"
PREFIX_ROWS = 52_560
FULL_CACHE_ROWS = 78_912
FORECAST_END = pd.Timestamp("2024-01-01 00:00:00")
WEIGHT = 0.70 / 3.0
ACTIONS_CF = np.arange(0.075, 1.076, 0.0025)
MODEL_PARAMS = {
    "objective": "multi:softprob",
    "n_estimators": 100,
    "learning_rate": 0.03,
    "max_depth": 5,
    "min_child_weight": 20.0,
    "subsample": 0.9,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 5.0,
    "max_bin": 256,
    "tree_method": "hist",
    "random_state": 20260802,
    "n_jobs": 6,
}


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def verify_inputs(repo: Path, predeclaration: Path) -> dict[str, Any]:
    frozen = json.loads(predeclaration.read_text())
    observed = {
        relative: sha256_path(repo / relative)
        for relative in frozen["input_bundle"]["files"]
    }
    if observed != frozen["input_bundle"]["files"]:
        raise RuntimeError("N22 input hash mismatch")
    if canonical_hash(observed) != frozen["input_bundle"]["sha256"]:
        raise RuntimeError("N22 input bundle mismatch")
    spec_path = repo / frozen["family_spec"]["path"]
    if sha256_path(spec_path) != frozen["family_spec"]["sha256"]:
        raise RuntimeError("N22 family spec mutation")
    return frozen


def read_npy_prefix(npz_path: Path, member: str) -> tuple[np.ndarray, dict[str, Any]]:
    """Decode exactly the allowed prefix and never request the 2024 tail."""
    with zipfile.ZipFile(npz_path) as archive:
        info = archive.getinfo(member)
        with archive.open(info) as stream:
            version = np.lib.format.read_magic(stream)
            if version == (1, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
            else:
                shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
            if shape != (FULL_CACHE_ROWS,) or fortran or dtype != np.dtype("float32"):
                raise RuntimeError(f"N22 unexpected cache member header: {member}")
            requested = PREFIX_ROWS * dtype.itemsize
            payload = stream.read(requested)
            if len(payload) != requested:
                raise RuntimeError("N22 bounded cache prefix truncated")
    values = np.frombuffer(payload, dtype=dtype).copy()
    if len(values) != PREFIX_ROWS:
        raise RuntimeError("N22 bounded cache prefix row mismatch")
    return values, {
        "path": str(npz_path),
        "member": member,
        "full_elements": FULL_CACHE_ROWS,
        "decoded_elements": len(values),
        "decoded_bytes": len(payload),
        "forbidden_tail_elements": FULL_CACHE_ROWS - len(values),
        "tail_values_decoded": 0,
        "dtype": str(dtype),
        "compression": info.compress_type,
    }


def selected_features(repo: Path) -> dict[str, list[str]]:
    base = repo / "artifacts/backtests/metric-aligned-probe"
    result: dict[str, list[str]] = {}
    for fold in FOLDS:
        m102 = json.loads((base / f"M102_TOP100-{fold}.json").read_text())
        m115 = json.loads((base / f"M115_XGBOOST-{fold}.json").read_text())
        names = list(m115["selected_feature_names"])
        if names != list(m102["selected_feature_names"]):
            raise RuntimeError(f"N22 {fold} M102/M115 feature mismatch")
        if len(names) != 100 or len(set(names)) != 100 or TERRAIN_NAME in names:
            raise RuntimeError(f"N22 {fold} feature contract mismatch")
        if int(m115["selected_iteration"]) != 100:
            raise RuntimeError(f"N22 {fold} iteration contract mismatch")
        result[fold] = names
    return result


def _parquet_columns(path: Path) -> set[str]:
    return set(pq.ParquetFile(path).schema.names)


def read_parquet_prefix(
    path: Path,
    columns: list[str],
    rows: int,
) -> pd.DataFrame:
    """Expose exactly the chronological prefix from a one-row-group frozen file."""
    parquet = pq.ParquetFile(path)
    batches = parquet.iter_batches(
        batch_size=rows,
        columns=columns,
        use_threads=False,
    )
    first = next(batches)
    if first.num_rows != rows:
        raise RuntimeError(f"N22 Parquet prefix truncated: {path} {first.num_rows}/{rows}")
    return first.to_pandas()


def load_bounded_surface(
    repo: Path,
    feature_names: dict[str, list[str]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cache = (
        repo
        / "artifacts/cache/920be0c458d820e855bf79dd25723146f52ce1736138aedba5e6bc853f1f720b"
    )
    features_path = cache / "train_features.parquet"
    grid_path = cache / "train_grid_pivot.parquet"
    geometric_path = cache / "train_geometric.parquet"
    wanted = set().union(*map(set, feature_names.values()))
    wanted = {name for name in wanted if not name.startswith("sitewind__")}
    feature_schema = _parquet_columns(features_path)
    grid_schema = _parquet_columns(grid_path)
    geometric_schema = _parquet_columns(geometric_path)
    dynamic = {"group_1", "group_2", "group_3"}
    missing = sorted(wanted - feature_schema - grid_schema - geometric_schema - dynamic)
    if missing:
        raise RuntimeError(f"N22 selected feature missing from bounded schemas: {missing}")
    base_keys = [
        "forecast_id",
        "forecast_kst_dtm",
        "data_available_kst_dtm",
        "issuance_batch",
        "group_id",
    ]
    feature_columns = list(
        dict.fromkeys([*base_keys, *sorted(wanted & feature_schema)])
    )
    u_columns = [
        f"ldaps__grid{grid:02d}__heightAboveGround_10_10u"
        for grid in range(1, 17)
    ]
    v_columns = [
        f"ldaps__grid{grid:02d}__heightAboveGround_10_10v"
        for grid in range(1, 17)
    ]
    grid_columns = list(
        dict.fromkeys(
            [
                "forecast_kst_dtm",
                *sorted(wanted & grid_schema),
                *u_columns,
                *v_columns,
            ]
        )
    )
    geometric_keys = ["forecast_kst_dtm", "data_available_kst_dtm", "group_id"]
    geometric_columns = list(
        dict.fromkeys([*geometric_keys, *sorted(wanted & geometric_schema)])
    )
    features = read_parquet_prefix(features_path, feature_columns, PREFIX_ROWS)
    grid = read_parquet_prefix(grid_path, grid_columns, 17_520)
    geometric = read_parquet_prefix(geometric_path, geometric_columns, PREFIX_ROWS)
    for frame in (features, grid, geometric):
        frame["forecast_kst_dtm"] = pd.to_datetime(frame["forecast_kst_dtm"])
    for frame in (features, geometric):
        frame["data_available_kst_dtm"] = pd.to_datetime(
            frame["data_available_kst_dtm"]
        )
    if len(features) != PREFIX_ROWS or len(grid) != 17_520 or len(geometric) != PREFIX_ROWS:
        raise RuntimeError(
            f"N22 bounded source rows changed: {len(features)}/{len(grid)}/{len(geometric)}"
        )
    if not features["forecast_kst_dtm"].is_monotonic_increasing:
        raise RuntimeError("N22 bounded feature order is not chronological")
    expected_groups = np.tile(np.asarray([1, 2, 3], dtype=int), 17_520)
    if not np.array_equal(features["group_id"].to_numpy(dtype=int), expected_groups):
        raise RuntimeError("N22 bounded feature group interleave changed")
    original_keys = pd.MultiIndex.from_frame(features[["forecast_kst_dtm", "group_id"]])
    surface = features.merge(
        grid,
        on="forecast_kst_dtm",
        validate="many_to_one",
    ).merge(
        geometric,
        on=["forecast_kst_dtm", "data_available_kst_dtm", "group_id"],
        validate="one_to_one",
    )
    observed_keys = pd.MultiIndex.from_frame(surface[["forecast_kst_dtm", "group_id"]])
    if not original_keys.equals(observed_keys):
        raise RuntimeError("N22 bounded merge changed row order")
    for group in (1, 2, 3):
        surface[f"group_{group}"] = surface["group_id"].eq(group).astype("int8")
    if len(surface) != PREFIX_ROWS or surface["forecast_kst_dtm"].max() != FORECAST_END:
        raise RuntimeError("N22 bounded surface contract failed")
    if surface["forecast_kst_dtm"].min() != pd.Timestamp("2022-01-01 01:00:00"):
        raise RuntimeError("N22 bounded surface start changed")
    return surface, {
        "rows": len(surface),
        "unique_timestamps": int(surface["forecast_kst_dtm"].nunique()),
        "start": surface["forecast_kst_dtm"].min().isoformat(),
        "end": surface["forecast_kst_dtm"].max().isoformat(),
        "groups": sorted(surface["group_id"].unique().astype(int).tolist()),
        "u_columns": u_columns,
        "v_columns": v_columns,
        "2024_value_rows": 0,
        "row_order": "forecast_kst_dtm ascending, group 1/2/3 interleaved",
    }


def add_sitewind(
    matrix: pd.DataFrame,
    legacy: np.ndarray,
    allweather: np.ndarray,
) -> None:
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


def terrain_feature(surface: pd.DataFrame, lookup: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    u_columns = [
        f"ldaps__grid{grid:02d}__heightAboveGround_10_10u"
        for grid in range(1, 17)
    ]
    v_columns = [
        f"ldaps__grid{grid:02d}__heightAboveGround_10_10v"
        for grid in range(1, 17)
    ]
    u = surface[u_columns].to_numpy(dtype=np.float64)
    v = surface[v_columns].to_numpy(dtype=np.float64)
    finite = np.isfinite(u) & np.isfinite(v)
    nonzero = np.hypot(u, v) > 0.0
    if not finite.all() or not nonzero.all():
        raise RuntimeError(
            f"N22 terrain vector gate failed: finite={finite.sum()} nonzero={nonzero.sum()}"
        )
    direction = np.degrees(np.arctan2(-u, -v)) % 360.0
    bins = (np.floor((direction + 2.5) / 5.0).astype(np.int16) % 72)
    per_grid = lookup[np.arange(16)[None, :], bins]
    if per_grid.shape != (PREFIX_ROWS, 16) or not np.isfinite(per_grid).all():
        raise RuntimeError("N22 terrain lookup materialization failed")
    value = per_grid.mean(axis=1).astype(np.float32)
    return value, {
        "rows": len(value),
        "grid_values": int(per_grid.size),
        "finite_vectors": int(finite.sum()),
        "nonzero_vectors": int(nonzero.sum()),
        "min": float(value.min()),
        "max": float(value.max()),
        "mean": float(value.mean()),
        "unique_direction_bins": int(np.unique(bins).size),
    }


def original_validation_keys(repo: Path, fold: str) -> pd.DataFrame:
    path = (
        repo
        / "artifacts/backtests/metric-aligned-probe"
        / f"M115_XGBOOST-{fold}-policies.parquet"
    )
    keys = pd.read_parquet(
        path,
        columns=["forecast_id", "forecast_kst_dtm", "group_id"],
    )
    keys["forecast_kst_dtm"] = pd.to_datetime(keys["forecast_kst_dtm"])
    if keys.duplicated(["forecast_id", "group_id"]).any():
        raise RuntimeError(f"N22 {fold} original validation keys duplicate")
    return keys


def load_training_labels(
    repo: Path,
    fold_start: pd.Timestamp,
    expected_rows: int,
) -> pd.DataFrame:
    path = (
        repo
        / "artifacts/cache/920be0c458d820e855bf79dd25723146f52ce1736138aedba5e6bc853f1f720b"
        / "labels_long.parquet"
    )
    labels = read_parquet_prefix(
        path,
        [
            "forecast_kst_dtm",
            "group_id",
            "actual_kwh",
            "operating_year",
        ],
        expected_rows,
    )
    labels["forecast_kst_dtm"] = pd.to_datetime(labels["forecast_kst_dtm"])
    if len(labels) != expected_rows:
        raise RuntimeError("N22 bounded training-label prefix row mismatch")
    if labels["forecast_kst_dtm"].max() >= fold_start:
        raise RuntimeError("N22 future training label read")
    if labels["operating_year"].max() > 2023:
        raise RuntimeError("N22 2024 training label read")
    if labels.duplicated(["forecast_kst_dtm", "group_id"]).any():
        raise RuntimeError("N22 training labels duplicate")
    return labels


def fold_frames(
    surface: pd.DataFrame,
    validation_keys: pd.DataFrame,
    labels: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    training = surface.loc[
        surface["forecast_kst_dtm"].lt(validation_keys["forecast_kst_dtm"].min())
    ].merge(
        labels[["forecast_kst_dtm", "group_id", "actual_kwh"]],
        on=["forecast_kst_dtm", "group_id"],
        how="inner",
        validate="one_to_one",
    )
    validation = validation_keys.merge(
        surface,
        on=["forecast_id", "forecast_kst_dtm", "group_id"],
        how="left",
        validate="one_to_one",
    )
    if len(validation) != len(validation_keys) or validation.isna().all(axis=1).any():
        raise RuntimeError("N22 validation surface alignment failed")
    return training, validation


def training_contract(training: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    capacity = training["group_id"].map(CAPACITIES_KWH).to_numpy(dtype=np.float64)
    rate = training["actual_kwh"].to_numpy(dtype=np.float64) / capacity
    eligible = np.isfinite(rate) & (rate >= 0.10)
    clipped = np.clip(rate, 0.10, 1.074999)
    raw_bins = np.floor((clipped - 0.10) / 0.02).astype(np.int16)
    active_bins = np.asarray(sorted(np.unique(raw_bins[eligible])), dtype=np.int16)
    mapping = {int(bin_id): class_id for class_id, bin_id in enumerate(active_bins)}
    classes = np.asarray([mapping[int(value)] for value in raw_bins[eligible]], dtype=np.int32)
    centers = np.asarray(
        [
            float(np.mean(rate[eligible][classes == class_id]))
            for class_id in range(len(active_bins))
        ],
        dtype=np.float64,
    )
    return eligible, classes, centers


def make_model(class_count: int) -> XGBClassifier:
    return XGBClassifier(num_class=class_count, **MODEL_PARAMS)


def fixed_action(
    probability: np.ndarray,
    centers: np.ndarray,
    validation_groups: np.ndarray,
    training: pd.DataFrame,
) -> np.ndarray:
    eligible_centers = centers >= 0.10
    probability = probability[:, eligible_centers].astype(np.float64)
    probability /= probability.sum(axis=1, keepdims=True)
    centers = centers[eligible_centers]
    calibrated = probability ** (1.0 / 0.75)
    calibrated /= calibrated.sum(axis=1, keepdims=True)
    error = np.abs(ACTIONS_CF[:, None] - centers[None, :])
    units = np.select([error <= 0.06, error <= 0.08], [4.0, 3.0], default=0.0)
    capacity = training["group_id"].map(CAPACITIES_KWH).to_numpy(dtype=np.float64)
    rate = training["actual_kwh"].to_numpy(dtype=np.float64) / capacity
    mean_generation = {
        group: float(np.mean(rate[(training["group_id"].to_numpy() == group) & (rate >= 0.10)]))
        for group in (1, 2, 3)
    }
    chosen = np.empty(len(probability), dtype=np.float64)
    for group in (1, 2, 3):
        selected = validation_groups == group
        group_probability = calibrated[selected]
        utility = -(group_probability @ error.T) + 2.0 * (
            group_probability @ (centers[None, :] * units).T
        ) / (4.0 * mean_generation[group])
        chosen[selected] = ACTIONS_CF[np.argmax(utility, axis=1)]
    return chosen * pd.Series(validation_groups).map(CAPACITIES_KWH).to_numpy(dtype=float)


def fit_arm(
    matrix: pd.DataFrame,
    features: list[str],
    training: pd.DataFrame,
    validation: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, Any]]:
    eligible, classes, centers = training_contract(training)
    train_matrix = training[features].astype("float32")
    validation_matrix = validation[features].astype("float32")
    rate = (
        training["actual_kwh"].to_numpy(dtype=np.float64)
        / training["group_id"].map(CAPACITIES_KWH).to_numpy(dtype=np.float64)
    )
    sample_weight = np.clip(rate[eligible], 0.10, None)
    model = make_model(len(centers))
    model.fit(
        train_matrix.loc[eligible],
        classes,
        sample_weight=sample_weight,
    )
    probability = model.predict_proba(
        validation_matrix,
        iteration_range=(0, 100),
    )
    if probability.shape != (len(validation), len(centers)):
        raise RuntimeError("N22 probability shape mismatch")
    if not np.isfinite(probability).all():
        raise RuntimeError("N22 probability nonfinite")
    action = fixed_action(
        probability,
        centers,
        validation["group_id"].to_numpy(dtype=int),
        training,
    )
    details = {
        "training_rows_total": len(training),
        "training_rows_eligible": int(eligible.sum()),
        "validation_rows": len(validation),
        "class_count": len(centers),
        "center_min": float(centers.min()),
        "center_max": float(centers.max()),
        "feature_count": len(features),
        "feature_names_sha256": canonical_hash(features),
        "model_params": {**MODEL_PARAMS, "num_class": len(centers)},
        "policy": "T0.75_G2",
        "policy_search_calls": 0,
    }
    del model, probability, train_matrix, validation_matrix
    gc.collect()
    return action, details


def assessment_actions(repo: Path) -> pd.DataFrame:
    path = repo / "artifacts/backtests/s17_n7_strict_actions/actions.parquet"
    columns = [
        "fold_id",
        "group_id",
        "forecast_kst_dtm",
        "M115_XGBOOST",
        "CHAMPION",
    ]
    frame = pd.read_parquet(path, columns=columns)
    frame["forecast_kst_dtm"] = pd.to_datetime(frame["forecast_kst_dtm"])
    if frame.duplicated(list(KEYS)).any() or len(frame) != 19_440:
        raise RuntimeError("N22 N7 action-key contract failed")
    return frame


def vector_hash(frame: pd.DataFrame, column: str, fold: str) -> str:
    part = frame.loc[frame["fold_id"].eq(fold)].sort_values(list(KEYS), kind="stable")
    key_bytes = part[list(KEYS)].astype(str).agg("|".join, axis=1).str.cat(sep="\n").encode()
    values = np.ascontiguousarray(part[column].to_numpy(dtype="<f8")).tobytes()
    return hashlib.sha256(key_bytes + b"\n" + values).hexdigest()


def prepare(repo: Path, frozen: dict[str, Any]) -> dict[str, Any]:
    names = selected_features(repo)
    surface, surface_details = load_bounded_surface(repo, names)
    lookup = np.load(repo / "artifacts/external/copdem_s17_n20/lookup.npy", allow_pickle=False)
    if lookup.shape != (16, 72) or not np.isfinite(lookup).all():
        raise RuntimeError("N22 lookup contract failed")
    terrain, terrain_details = terrain_feature(surface, lookup)
    fold_details: dict[str, Any] = {}
    prefix_details: dict[str, Any] = {}
    base = repo / "artifacts/backtests/metric-aligned-probe"
    for fold in FOLDS:
        legacy, legacy_detail = read_npy_prefix(
            base / f"M64B_ALLWEATHER_SITEWIND_CLASS-{fold}-sitewind-features.npz",
            "legacy.npy",
        )
        allweather, allweather_detail = read_npy_prefix(
            base / f"M64B_ALLWEATHER_SITEWIND_CLASS-{fold}-sitewind-features.npz",
            "allweather.npy",
        )
        matrix = surface.copy()
        add_sitewind(matrix, legacy, allweather)
        matrix[TERRAIN_NAME] = terrain
        missing = sorted(set(names[fold]) - set(matrix.columns))
        if missing:
            raise RuntimeError(f"N22 {fold} materialized feature missing: {missing}")
        validation = original_validation_keys(repo, fold)
        start = validation["forecast_kst_dtm"].min()
        validation_surface = validation.merge(
            matrix,
            on=["forecast_id", "forecast_kst_dtm", "group_id"],
            how="left",
            validate="one_to_one",
        )
        assessment = assessment_actions(repo)
        retained = assessment.loc[assessment["fold_id"].eq(fold), list(KEYS)]
        retained_check = retained.merge(
            validation[["forecast_kst_dtm", "group_id"]],
            on=["forecast_kst_dtm", "group_id"],
            how="left",
            indicator=True,
        )
        if not retained_check["_merge"].eq("both").all():
            raise RuntimeError(f"N22 {fold} retained keys absent from original validation")
        fold_details[fold] = {
            "original_validation_rows": len(validation),
            "retained_assessment_rows": len(retained),
            "fold_start": start.isoformat(),
            "control_feature_count": len(names[fold]),
            "candidate_feature_count": len(names[fold]) + 1,
            "control_missing_values": int(
                validation_surface[names[fold]].isna().sum().sum()
            ),
            "terrain_missing_values": int(
                validation_surface[TERRAIN_NAME].isna().sum()
            ),
        }
        prefix_details[fold] = {
            "legacy": legacy_detail,
            "allweather": allweather_detail,
        }
        del matrix, validation_surface, legacy, allweather
        gc.collect()
    return {
        "surface": surface_details,
        "terrain": terrain_details,
        "folds": fold_details,
        "prefixes": prefix_details,
        "decoded_2024_values": 0,
        "assessment_actual_values_read": 0,
        "fits": 0,
        "predict_calls": 0,
        "policy_or_score_calls": 0,
        "input_bundle_sha256": frozen["input_bundle"]["sha256"],
    }


def materialize(
    repo: Path,
    predeclaration: Path,
    output_dir: Path,
) -> dict[str, Any]:
    frozen = verify_inputs(repo, predeclaration)
    names = selected_features(repo)
    surface, surface_details = load_bounded_surface(repo, names)
    lookup = np.load(repo / "artifacts/external/copdem_s17_n20/lookup.npy", allow_pickle=False)
    terrain, terrain_details = terrain_feature(surface, lookup)
    n7 = assessment_actions(repo)
    base = repo / "artifacts/backtests/metric-aligned-probe"
    fold_outputs: list[pd.DataFrame] = []
    fit_details: dict[str, Any] = {}
    prefix_details: dict[str, Any] = {}
    total_fits = 0
    total_predict_calls = 0
    for fold in FOLDS:
        legacy, legacy_detail = read_npy_prefix(
            base / f"M64B_ALLWEATHER_SITEWIND_CLASS-{fold}-sitewind-features.npz",
            "legacy.npy",
        )
        allweather, allweather_detail = read_npy_prefix(
            base / f"M64B_ALLWEATHER_SITEWIND_CLASS-{fold}-sitewind-features.npz",
            "allweather.npy",
        )
        matrix = surface.copy()
        add_sitewind(matrix, legacy, allweather)
        matrix[TERRAIN_NAME] = terrain
        validation_keys = original_validation_keys(repo, fold)
        fold_start = validation_keys["forecast_kst_dtm"].min()
        expected_training_rows = int(surface["forecast_kst_dtm"].lt(fold_start).sum())
        labels = load_training_labels(repo, fold_start, expected_training_rows)
        training, validation = fold_frames(matrix, validation_keys, labels)
        control_action, control_details = fit_arm(
            matrix,
            names[fold],
            training,
            validation,
        )
        total_fits += 1
        total_predict_calls += 1
        candidate_action, candidate_details = fit_arm(
            matrix,
            [*names[fold], TERRAIN_NAME],
            training,
            validation,
        )
        total_fits += 1
        total_predict_calls += 1
        predicted = validation[
            [
                "forecast_id",
                "forecast_kst_dtm",
                "group_id",
                "data_available_kst_dtm",
            ]
        ].copy()
        predicted["M115_CONTROL"] = control_action
        predicted["M115_TERRAIN"] = candidate_action
        retained = n7.loc[n7["fold_id"].eq(fold)].merge(
            predicted,
            on=["forecast_kst_dtm", "group_id"],
            how="left",
            validate="one_to_one",
        )
        if retained[["M115_CONTROL", "M115_TERRAIN"]].isna().any().any():
            raise RuntimeError(f"N22 {fold} prediction alignment failed")
        operating_day = (
            retained["forecast_kst_dtm"] - pd.Timedelta(hours=1)
        ).dt.normalize()
        basis = operating_day - pd.Timedelta(days=1) + pd.Timedelta(hours=14)
        feature_availability_safe = bool(
            retained["data_available_kst_dtm"].le(basis).all()
        )
        label_operating_day_max = (
            labels["forecast_kst_dtm"].max() - pd.Timedelta(hours=1)
        ).normalize()
        label_available_max = label_operating_day_max + pd.Timedelta(days=1)
        first_retained_basis = basis.min()
        label_chronology_safe = bool(label_available_max <= first_retained_basis)
        control_error = float(
            np.max(
                np.abs(
                    retained["M115_CONTROL"].to_numpy(dtype=float)
                    - retained["M115_XGBOOST"].to_numpy(dtype=float)
                )
            )
        )
        capacity = retained["group_id"].map(CAPACITIES_KWH).to_numpy(dtype=float)
        zero = retained["CHAMPION"].to_numpy(dtype=float) + WEIGHT * (
            retained["M115_CONTROL"].to_numpy(dtype=float)
            - retained["M115_XGBOOST"].to_numpy(dtype=float)
        )
        treatment = retained["CHAMPION"].to_numpy(dtype=float) + WEIGHT * (
            retained["M115_TERRAIN"].to_numpy(dtype=float)
            - retained["M115_CONTROL"].to_numpy(dtype=float)
        )
        zero = np.clip(zero, 0.0, 1.075 * capacity)
        treatment = np.clip(treatment, 0.0, 1.075 * capacity)
        retained["M115_REFIT_ZERO"] = zero
        retained["TERRAIN_SX300_H8_M115_REPLACED"] = treatment
        zero_error = float(
            np.max(
                np.abs(
                    retained["M115_REFIT_ZERO"].to_numpy(dtype=float)
                    - retained["CHAMPION"].to_numpy(dtype=float)
                )
            )
        )
        if fold == "dev-2023-Q2":
            retained["M115_REFIT_ZERO"] = retained["CHAMPION"]
            retained["TERRAIN_SX300_H8_M115_REPLACED"] = retained["CHAMPION"]
        output = retained[
            [
                *KEYS,
                "CHAMPION",
                "M115_REFIT_ZERO",
                "TERRAIN_SX300_H8_M115_REPLACED",
                "M115_XGBOOST",
                "M115_CONTROL",
                "M115_TERRAIN",
            ]
        ].copy()
        fold_outputs.append(output)
        fit_details[fold] = {
            "fold_start": fold_start.isoformat(),
            "fit_label_max_forecast_time": labels["forecast_kst_dtm"].max().isoformat(),
            "fit_label_available_max": label_available_max.isoformat(),
            "first_retained_basis": first_retained_basis.isoformat(),
            "feature_availability_safe": feature_availability_safe,
            "label_chronology_safe": label_chronology_safe,
            "fit_label_rows_read": len(labels),
            "assessment_actual_values_read": 0,
            "control": control_details,
            "terrain": candidate_details,
            "control_vs_n7_m115_max_abs_kwh": control_error,
            "refit_zero_vs_champion_max_abs_kwh_before_q2_burnin": zero_error,
            "retained_rows": len(retained),
        }
        prefix_details[fold] = {
            "legacy": legacy_detail,
            "allweather": allweather_detail,
        }
        del matrix, labels, training, validation, retained, legacy, allweather
        gc.collect()
    predictions = pd.concat(fold_outputs, ignore_index=True)
    predictions = predictions.sort_values(list(KEYS), kind="stable").reset_index(drop=True)
    if total_fits != 6 or total_predict_calls != 6:
        raise RuntimeError("N22 fit/predict count mismatch")
    if len(predictions) != 19_440 or predictions.duplicated(list(KEYS)).any():
        raise RuntimeError("N22 output key contract failed")
    q2 = predictions["fold_id"].eq("dev-2023-Q2")
    q2_zero_error = float(
        np.max(
            np.abs(
                predictions.loc[q2, "M115_REFIT_ZERO"].to_numpy(dtype=float)
                - predictions.loc[q2, "CHAMPION"].to_numpy(dtype=float)
            )
        )
    )
    q2_treatment_error = float(
        np.max(
            np.abs(
                predictions.loc[q2, "TERRAIN_SX300_H8_M115_REPLACED"].to_numpy(dtype=float)
                - predictions.loc[q2, "CHAMPION"].to_numpy(dtype=float)
            )
        )
    )
    control_max = max(
        details["control_vs_n7_m115_max_abs_kwh"] for details in fit_details.values()
    )
    zero_outer_max = max(
        fit_details[fold]["refit_zero_vs_champion_max_abs_kwh_before_q2_burnin"]
        for fold in OUTER
    )
    finite = np.isfinite(predictions[list(FAMILY)].to_numpy(dtype=float)).all()
    guards = {
        "input_hashes_exact": True,
        "surface_rows_52560": surface_details["rows"] == PREFIX_ROWS,
        "prefix_values_per_member_52560": all(
            part[member]["decoded_elements"] == PREFIX_ROWS
            for part in prefix_details.values()
            for member in ("legacy", "allweather")
        ),
        "decoded_2024_values_zero": all(
            part[member]["tail_values_decoded"] == 0
            for part in prefix_details.values()
            for member in ("legacy", "allweather")
        ),
        "terrain_vectors_finite_nonzero": (
            terrain_details["finite_vectors"] == PREFIX_ROWS * 16
            and terrain_details["nonzero_vectors"] == PREFIX_ROWS * 16
        ),
        "provenance_strictly_past_by_retained_basis": all(
            details["feature_availability_safe"]
            and details["label_chronology_safe"]
            for details in fit_details.values()
        ),
        "control_vs_n7_m115_max_abs_le_1e_6": control_max <= 1e-6,
        "zero_vs_champion_max_abs_le_1e_9": zero_outer_max <= 1e-9,
        "q2_all_equal_champion": q2_zero_error == 0.0 and q2_treatment_error == 0.0,
        "predictions_finite": bool(finite),
        "assessment_actual_not_read": True,
        "fit_count_six": total_fits == 6,
        "policy_search_calls_zero": True,
        "score_calls_zero": True,
    }
    details_path = output_dir / "materialization_details.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    details = {
        "schema_version": 1,
        "node_id": frozen["node_id"],
        "predeclaration_sha256": sha256_path(predeclaration),
        "family_spec_sha256": frozen["family_spec"]["sha256"],
        "input_bundle_sha256": frozen["input_bundle"]["sha256"],
        "surface": surface_details,
        "terrain": terrain_details,
        "prefixes": prefix_details,
        "fits": fit_details,
        "fit_count": total_fits,
        "predict_calls": total_predict_calls,
        "policy_search_calls": 0,
        "score_calls": 0,
        "assessment_actual_values_read": 0,
        "control_max_abs_kwh": control_max,
        "zero_outer_max_abs_kwh": zero_outer_max,
        "q2_zero_max_abs_kwh": q2_zero_error,
        "q2_treatment_max_abs_kwh": q2_treatment_error,
        "guards": guards,
        "all_guards_pass": all(guards.values()),
        "comparison_index_consumed": False,
        "actions": {
            "model_fits": total_fits,
            "predict_calls": total_predict_calls,
            "policy_or_score_calls": 0,
            "decoded_2024_values": 0,
            "assessment_actual_values": 0,
            "external_requests": 0,
            "test_access": False,
            "rejected_ecmwf_access": False,
            "quarantined_n10_access": False,
            "dependency_changes": False,
            "dacon_actions": [],
        },
    }
    details_path.write_text(json.dumps(details, ensure_ascii=False, indent=2) + "\n")
    if not details["all_guards_pass"]:
        details["verdict"] = "DIAGNOSTIC_INCONCLUSIVE_PRE_SCORE_GUARD_FAILED"
        details_path.write_text(json.dumps(details, ensure_ascii=False, indent=2) + "\n")
        return details
    predictions_path = output_dir / "predictions.parquet"
    predictions.to_parquet(predictions_path, index=False)
    provenance_rows: list[dict[str, Any]] = []
    inherited = pd.read_parquet(
        repo / "artifacts/backtests/s17_n7_strict_actions/procedure_provenance.parquet"
    )
    for fold in OUTER:
        champion = inherited.loc[
            inherited["model_id"].eq("CHAMPION") & inherited["test_fold"].eq(fold)
        ]
        if len(champion) != 1:
            raise RuntimeError(f"N22 {fold} inherited Champion provenance missing")
        provenance_rows.append(champion.iloc[0].to_dict())
        fit_time = fit_details[fold]["fit_label_available_max"]
        selection_time = "2023-07-01T00:00:00"
        for model, policy in (
            ("M115_REFIT_ZERO", "FIXED_T0.75_G2_M115_REFIT_ZERO"),
            (
                "TERRAIN_SX300_H8_M115_REPLACED",
                "FIXED_T0.75_G2_TERRAIN_MEAN16_REPLACE_M115",
            ),
        ):
            provenance_rows.append(
                {
                    "model_id": model,
                    "test_fold": fold,
                    "fit_max_time": fit_time,
                    "selection_max_time": selection_time,
                    "policy_id": policy,
                    "predeclaration_sha256": sha256_path(predeclaration),
                    "prediction_sha256": vector_hash(predictions, model, fold),
                    "weights_fit": "past_only_expanding",
                }
            )
    provenance = pd.DataFrame(provenance_rows)
    provenance_path = output_dir / "procedure_provenance.parquet"
    provenance.to_parquet(provenance_path, index=False)
    details["output_hashes"] = {
        "predictions.parquet": sha256_path(predictions_path),
        "procedure_provenance.parquet": sha256_path(provenance_path),
    }
    details_path.write_text(json.dumps(details, ensure_ascii=False, indent=2) + "\n")
    family_manifest = {
        "schema_version": 1,
        "node_id": frozen["node_id"],
        "predeclaration_sha256": sha256_path(predeclaration),
        "family_spec_sha256": frozen["family_spec"]["sha256"],
        "input_bundle_sha256": frozen["input_bundle"]["sha256"],
        "family": list(FAMILY),
        "comparison_index_if_assessed": 4,
        "materialization_guards_pass": True,
        "assessment_actual_values_read": 0,
        "fits": total_fits,
        "policy_search_calls": 0,
        "score_calls": 0,
        "output_hashes": {
            "predictions.parquet": sha256_path(predictions_path),
            "procedure_provenance.parquet": sha256_path(provenance_path),
            "materialization_details.json": sha256_path(details_path),
        },
    }
    family_path = output_dir / "family_manifest.json"
    family_path.write_text(json.dumps(family_manifest, ensure_ascii=False, indent=2) + "\n")
    return family_manifest


def metric_score(frame: pd.DataFrame, column: str) -> dict[str, float]:
    metric = frame[["forecast_kst_dtm", "group_id", "actual_kwh"]].copy()
    metric.insert(0, "forecast_id", np.arange(len(metric), dtype=np.int64))
    metric["prediction_kwh"] = frame[column].to_numpy(dtype=float)
    result = evaluate_official(metric, CAPACITIES_KWH)
    return {
        "total": float(result.total),
        "one_minus_nmae": float(result.one_minus_nmae),
        "ficr": float(result.ficr),
    }


def evaluate(
    repo: Path,
    predeclaration: Path,
    output_dir: Path,
    family_sha256: str,
) -> dict[str, Any]:
    frozen = verify_inputs(repo, predeclaration)
    family_path = output_dir / "family_manifest.json"
    if sha256_path(family_path) != family_sha256:
        raise RuntimeError("N22 family manifest freeze mismatch")
    family = json.loads(family_path.read_text())
    for name, digest in family["output_hashes"].items():
        if sha256_path(output_dir / name) != digest:
            raise RuntimeError(f"N22 frozen output mutation: {name}")
    predictions = pd.read_parquet(output_dir / "predictions.parquet")
    predictions["forecast_kst_dtm"] = pd.to_datetime(predictions["forecast_kst_dtm"])
    actual = pd.read_parquet(
        repo / "artifacts/backtests/s17_n7_strict_actions/actions.parquet",
        columns=["fold_id", "group_id", "forecast_kst_dtm", "actual_kwh"],
    )
    actual["forecast_kst_dtm"] = pd.to_datetime(actual["forecast_kst_dtm"])
    assessment = predictions.merge(
        actual,
        on=list(KEYS),
        how="left",
        validate="one_to_one",
    )
    if assessment["actual_kwh"].isna().any() or len(assessment) != 19_440:
        raise RuntimeError("N22 assessment actual alignment failed")
    provenance = pd.read_parquet(output_dir / "procedure_provenance.parquet")
    store = EventStore(repo, repo / "artifacts/registry/loop_events_s17.sqlite")
    protocol = run_prequential_protocol(
        assessment,
        prediction_columns=list(FAMILY),
        incumbent="CHAMPION",
        capacities=CAPACITIES_KWH,
        procedure_provenance=provenance,
        family_manifest_sha256=family_sha256,
        comparison_index=4,
        event_store=store,
        n_rep=4999,
        seed=20260808,
        block_lengths=(3, 7, 14),
        margin_total=0.001635,
    )
    outer = assessment.loc[assessment["fold_id"].isin(OUTER)].copy()
    scores = {model: metric_score(outer, model) for model in FAMILY}
    deltas = {
        model: scores[model]["total"] - scores["CHAMPION"]["total"]
        for model in FAMILY
        if model != "CHAMPION"
    }
    candidate = "TERRAIN_SX300_H8_M115_REPLACED"
    protocol_delta = protocol["blocks"]["7"]["joint_max_t"]["candidates"][candidate][
        "observed_delta_total"
    ]
    if abs(protocol_delta - deltas[candidate]) >= 1e-12:
        raise RuntimeError("N22 point/protocol delta disagreement")
    promotion = bool(
        deltas[candidate] >= 0.001635
        and protocol["promotion_stable_all_blocks"][candidate]
        and protocol["inference"] == "SUPPORTED"
    )
    goal_reached = bool(promotion and scores[candidate]["total"] >= 0.66)
    result = {
        "schema_version": 1,
        "node_id": frozen["node_id"],
        "family_manifest_sha256": family_sha256,
        "comparison_index": 4,
        "scores": scores,
        "deltas_vs_champion": deltas,
        "promotion_supported": promotion,
        "goal_total_0p66_reached": goal_reached,
        "protocol": protocol,
        "score_calls": {
            "outer_official": len(FAMILY),
            "strict_protocol": 1,
            "materialization_or_policy_selection": 0,
        },
        "assessment_actual_values_read": len(assessment),
        "comparison_consumed": True,
        "verdict": (
            "SUPPORTED_GOAL_REACHED"
            if goal_reached
            else ("SUPPORTED_BELOW_GOAL" if promotion else "REFUTED")
        ),
        "next_handoff": (
            ["S17_TARGET_DELIVERY"]
            if goal_reached
            else ["S17-N23_POST_TERRAIN_FRONTIER_RESEARCH_INTAKE"]
        ),
        "forbidden_access": {
            "2024_values": False,
            "test": False,
            "rejected_ecmwf": False,
            "quarantined_n10": False,
            "external_requests": 0,
            "dependency_changes": False,
            "dacon_actions": [],
        },
    }
    evaluation_path = output_dir / "evaluation.json"
    evaluation_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "materialize", "evaluate"))
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--predeclaration",
        type=Path,
        default=Path("reports/s17_n22_m115_terrain_strict_predeclaration.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/backtests/s17_n22_m115_terrain"),
    )
    parser.add_argument("--family-sha256")
    args = parser.parse_args()
    repo = args.repo.resolve()
    predeclaration = args.predeclaration
    output_dir = args.output_dir
    if not predeclaration.is_absolute():
        predeclaration = repo / predeclaration
    if not output_dir.is_absolute():
        output_dir = repo / output_dir
    frozen = verify_inputs(repo, predeclaration)
    if args.mode == "preflight":
        result = prepare(repo, frozen)
    elif args.mode == "materialize":
        result = materialize(repo, predeclaration, output_dir)
    else:
        if not args.family_sha256:
            raise RuntimeError("N22 evaluate requires --family-sha256")
        result = evaluate(
            repo,
            predeclaration,
            output_dir,
            args.family_sha256,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
