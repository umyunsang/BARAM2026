"""Screen layout-aware wake-sector features from supplied turbine geometry and NWP."""

from __future__ import annotations

import argparse
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from run_alternative_booster_classifier import _feature_names, _fit_model, _probability
from run_consensus_classifier import _screen_blends
from run_sequence_classifier import (
    BASELINE,
    BASELINE_SHA,
    CAPACITIES,
    OPEN,
    OPEN_SHA,
    OUTPUT,
    _score,
    _sha256,
    _surface,
)
from run_site_wind_classifier import FOLDS, _add_site_wind_features, _choose_actions
from run_site_wind_teacher import _validation_mask

from baram.data.turbines import parse_turbine_workbook

BASE_COLUMNS = ["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh"]
PARENT_PATH = OUTPUT / "M107_STRICT_TEMPORAL_TOP100-oof.parquet"
VECTORS = {
    "gfs10": ("geom__gfs__wind10__mean_u", "geom__gfs__wind10__mean_v"),
    "gfs80": ("geom__gfs__wind80__mean_u", "geom__gfs__wind80__mean_v"),
    "gfs100": ("geom__gfs__wind100__mean_u", "geom__gfs__wind100__mean_v"),
    "ldaps10": ("geom__ldaps__wind10__mean_u", "geom__ldaps__wind10__mean_v"),
    "ldaps50max": (
        "geom__ldaps__wind50max__mean_u",
        "geom__ldaps__wind50max__mean_v",
    ),
    "ldaps50min": (
        "geom__ldaps__wind50min__mean_u",
        "geom__ldaps__wind50min__mean_v",
    ),
}


def _turbines() -> pd.DataFrame:
    with zipfile.ZipFile(OPEN) as archive:
        return parse_turbine_workbook(archive.read("info.xlsx"))


def _local_xy(group: pd.DataFrame) -> np.ndarray:
    latitude = group["latitude"].to_numpy(dtype=float)
    longitude = group["longitude"].to_numpy(dtype=float)
    center_latitude = float(latitude.mean())
    east = (longitude - longitude.mean()) * 111_320.0 * np.cos(
        np.deg2rad(center_latitude)
    )
    north = (latitude - latitude.mean()) * 111_320.0
    return np.column_stack([east, north])


def _wake_values(
    u: np.ndarray,
    v: np.ndarray,
    group: pd.DataFrame,
    expansion: float,
) -> dict[str, np.ndarray]:
    xy = _local_xy(group)
    diameter = float(group["rotor_diameter_m"].mean())
    pair_source: list[int] = []
    pair_target: list[int] = []
    delta: list[np.ndarray] = []
    for target in range(len(xy)):
        for source in range(len(xy)):
            if source == target:
                continue
            pair_source.append(source)
            pair_target.append(target)
            delta.append(xy[target] - xy[source])
    del pair_source
    delta_array = np.asarray(delta, dtype=float)
    target_index = np.asarray(pair_target, dtype=int)
    speed = np.hypot(u, v)
    unit_u = np.divide(u, speed, out=np.zeros_like(u), where=speed > 1e-8)
    unit_v = np.divide(v, speed, out=np.zeros_like(v), where=speed > 1e-8)
    downwind = unit_u[:, None] * delta_array[None, :, 0] + unit_v[
        :, None
    ] * delta_array[None, :, 1]
    crosswind = np.abs(
        -unit_v[:, None] * delta_array[None, :, 0]
        + unit_u[:, None] * delta_array[None, :, 1]
    )
    positive = downwind > 0.0
    radius = diameter / 2.0 + expansion * np.maximum(downwind, 0.0)
    decay = (diameter / (diameter + 2.0 * expansion * np.maximum(downwind, 0.0))) ** 2
    deficit = np.where(
        positive,
        decay * np.exp(-0.5 * (crosswind / np.maximum(radius, 1.0)) ** 2),
        0.0,
    )
    turbine_exposure = np.zeros((len(u), len(group)), dtype=float)
    for target in range(len(group)):
        turbine_exposure[:, target] = deficit[:, target_index == target].sum(axis=1)
    angle = np.arctan2(unit_v, unit_u)
    mean_exposure = turbine_exposure.mean(axis=1)
    return {
        "mean": mean_exposure,
        "max": turbine_exposure.max(axis=1),
        "std": turbine_exposure.std(axis=1),
        "active_fraction": (deficit > 0.10).mean(axis=1),
        "speed_weighted": mean_exposure * speed,
        "power_weighted": mean_exposure * speed**3,
        "sector_sin2": np.sin(2.0 * angle),
        "sector_cos2": np.cos(2.0 * angle),
        "sector_sin3": np.sin(3.0 * angle),
        "sector_cos3": np.cos(3.0 * angle),
    }


def _wake_features(
    surface: pd.DataFrame,
    turbines: pd.DataFrame,
    expansion: float,
) -> pd.DataFrame:
    additions: dict[str, np.ndarray] = {}
    for alias, (u_name, v_name) in VECTORS.items():
        if u_name not in surface or v_name not in surface:
            raise RuntimeError(f"missing wake vector: {alias}")
        alias_values = {
            name: np.full(len(surface), np.nan, dtype="float32")
            for name in (
                "mean",
                "max",
                "std",
                "active_fraction",
                "speed_weighted",
                "power_weighted",
                "sector_sin2",
                "sector_cos2",
                "sector_sin3",
                "sector_cos3",
            )
        }
        for group_id in CAPACITIES:
            mask = surface["group_id"].eq(group_id).to_numpy()
            values = _wake_values(
                surface.loc[mask, u_name].to_numpy(dtype=float),
                surface.loc[mask, v_name].to_numpy(dtype=float),
                turbines.loc[turbines["group_id"].eq(group_id)],
                expansion,
            )
            for name, value in values.items():
                alias_values[name][mask] = value
        for name, value in alias_values.items():
            additions[f"wake__{alias}__{name}"] = value
    result = pd.DataFrame(additions, index=surface.index)
    if not np.isfinite(result.to_numpy()).all():
        raise RuntimeError("wake-sector features contain non-finite values")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--fold", choices=FOLDS, required=True)
    parser.add_argument("--sitewind-cache-id", required=True)
    parser.add_argument("--iterations", type=int, default=120)
    parser.add_argument("--wake-expansion", type=float, default=0.075)
    args = parser.parse_args()
    if not 40 <= args.iterations <= 300:
        raise ValueError("iterations must be between 40 and 300")
    if not 0.02 <= args.wake_expansion <= 0.20:
        raise ValueError("wake expansion must be between 0.02 and 0.20")
    if _sha256(OPEN) != OPEN_SHA or _sha256(BASELINE) != BASELINE_SHA:
        raise RuntimeError("immutable competition input hash mismatch")

    started = time.perf_counter()
    surface, base_columns, _ = _surface()
    validation = _validation_mask(surface, args.fold)
    validation_start = surface.loc[validation, "forecast_kst_dtm"].min()
    target = surface["actual_kwh"] / surface["group_id"].map(CAPACITIES)
    training = (
        surface["forecast_kst_dtm"].lt(validation_start).to_numpy()
        & target.ge(0.10).to_numpy()
    )
    raw_bins = np.floor((target.clip(0.10, 1.074999) - 0.10) / 0.02).astype(
        "Int64"
    )
    active_bins = [
        int(value)
        for value in sorted(raw_bins.loc[training].dropna().astype(int).unique())
    ]
    bin_to_class = {bin_id: class_id for class_id, bin_id in enumerate(active_bins)}
    classes = raw_bins.map(bin_to_class).astype("Int64")
    centers = np.asarray(
        [
            target.loc[training & classes.eq(class_id)].mean()
            for class_id in range(len(active_bins))
        ]
    )
    cached = np.load(
        OUTPUT / f"{args.sitewind_cache_id}-{args.fold}-sitewind-features.npz"
    )
    matrix = surface[base_columns].astype("float32")
    sitewind_columns = _add_site_wind_features(
        matrix, cached["legacy"], cached["allweather"]
    )
    wake = _wake_features(surface, _turbines(), args.wake_expansion)
    matrix = pd.concat([matrix, wake], axis=1)
    selected_features = list(
        dict.fromkeys([*_feature_names(args.fold), *wake.columns.tolist()])
    )
    model = _fit_model(
        "xgboost",
        matrix[selected_features],
        classes,
        training,
        target.loc[training].clip(lower=0.10).to_numpy(dtype=float),
        len(active_bins),
        args.iterations,
    )
    probability = _probability(
        model,
        "xgboost",
        matrix[selected_features],
        validation,
        args.iterations,
    )
    base = surface.loc[validation, BASE_COLUMNS].copy()
    raw_output, raw_policy, _, policies = _choose_actions(
        base,
        probability,
        centers,
        target,
        training,
        surface["group_id"],
    )
    parent = pd.read_parquet(PARENT_PATH)
    parent = parent.loc[parent["fold_id"].eq(args.fold)]
    blended, selections = _screen_blends(base, policies, parent)
    output = blended.assign(fold_id=args.fold, model_id=args.candidate_id)
    output_path = OUTPUT / f"{args.candidate_id}-{args.fold}.parquet"
    policy_path = OUTPUT / f"{args.candidate_id}-{args.fold}-policies.parquet"
    output.to_parquet(output_path, index=False)
    policies.to_parquet(policy_path, index=False)
    receipt = {
        "schema_version": 1,
        "candidate_id": args.candidate_id,
        "fold_id": args.fold,
        "architecture": "xgboost_multiclass_supplied_turbine_wake_sectors",
        "scope": "unseen-fold supplied-geometry wake representation screen",
        "iterations": args.iterations,
        "wake_expansion": args.wake_expansion,
        "wake_feature_count": len(wake.columns),
        "feature_count": len(selected_features),
        "sitewind_feature_count": len(sitewind_columns),
        "raw_best_policy": raw_policy,
        "raw_best_score": _score(raw_output),
        "oracle_blend_score": _score(output),
        "oracle_blends": selections,
        "runtime_seconds": round(time.perf_counter() - started, 2),
        "prediction_path": str(output_path.relative_to(Path.cwd())),
        "prediction_sha256": _sha256(output_path),
        "policy_path": str(policy_path.relative_to(Path.cwd())),
        "policy_sha256": _sha256(policy_path),
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
