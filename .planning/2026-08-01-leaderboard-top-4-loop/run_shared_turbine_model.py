"""Pool all supplied turbine SCADA targets with turbine-local NWP interpolation."""

from __future__ import annotations

import argparse
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from run_alternative_booster_classifier import _feature_names
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
from run_site_wind_classifier import FOLDS, _add_site_wind_features
from run_site_wind_teacher import _validation_mask
from run_turbine_decomposition import _hourly_turbine_targets
from run_xgb_point_regressor import _best_raw, _policies

from baram.data.turbines import parse_turbine_workbook
from baram.features.spatial import haversine_km

BASE_COLUMNS = ["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh"]
PARENT_PATH = OUTPUT / "M107_STRICT_TEMPORAL_TOP100-oof.parquet"
SOURCE_VARIABLES = {
    "gfs": (
        "heightAboveGround_10_10u",
        "heightAboveGround_10_10v",
        "heightAboveGround_80_u",
        "heightAboveGround_80_v",
        "heightAboveGround_100_100u",
        "heightAboveGround_100_100v",
        "surface_0_gust",
        "heightAboveGround_2_2t",
        "surface_0_sp",
    ),
    "ldaps": (
        "heightAboveGround_10_10u",
        "heightAboveGround_10_10v",
        "heightAboveGround_50_50MUmax",
        "heightAboveGround_50_50MVmax",
        "heightAboveGround_50_50MUmin",
        "heightAboveGround_50_50MVmin",
        "heightAboveGround_5_XBLWS",
        "heightAboveGround_5_YBLWS",
        "heightAboveGround_2_t",
        "surface_0_sp",
        "etc_0_blh",
    ),
}
VECTOR_PAIRS = {
    "gfs10": ("gfs__heightAboveGround_10_10u", "gfs__heightAboveGround_10_10v"),
    "gfs80": ("gfs__heightAboveGround_80_u", "gfs__heightAboveGround_80_v"),
    "gfs100": (
        "gfs__heightAboveGround_100_100u",
        "gfs__heightAboveGround_100_100v",
    ),
    "ldaps10": (
        "ldaps__heightAboveGround_10_10u",
        "ldaps__heightAboveGround_10_10v",
    ),
    "ldaps50max": (
        "ldaps__heightAboveGround_50_50MUmax",
        "ldaps__heightAboveGround_50_50MVmax",
    ),
    "ldaps50min": (
        "ldaps__heightAboveGround_50_50MUmin",
        "ldaps__heightAboveGround_50_50MVmin",
    ),
    "ldaps5": (
        "ldaps__heightAboveGround_5_XBLWS",
        "ldaps__heightAboveGround_5_YBLWS",
    ),
}


def _metadata_and_geometry() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    geometry: dict[str, pd.DataFrame] = {}
    with zipfile.ZipFile(OPEN) as archive:
        turbines = parse_turbine_workbook(archive.read("info.xlsx"))
        for source, count in (("gfs", 9), ("ldaps", 16)):
            with archive.open(f"train/{source}_train.csv") as stream:
                frame = pd.read_csv(
                    stream,
                    usecols=["grid_id", "latitude", "longitude"],
                    nrows=count,
                )
            geometry[source] = frame.sort_values("grid_id").reset_index(drop=True)
    return turbines, geometry


def _turbine_weather(
    surface: pd.DataFrame,
    turbine: pd.Series,
    geometry: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    additions: dict[str, np.ndarray] = {}
    for source, variables in SOURCE_VARIABLES.items():
        grid = geometry[source]
        distance = np.asarray(
            haversine_km(
                float(turbine["latitude"]),
                float(turbine["longitude"]),
                grid["latitude"].to_numpy(dtype=float),
                grid["longitude"].to_numpy(dtype=float),
            ),
            dtype=float,
        )
        raw_weight = 1.0 / np.maximum(distance, 0.10) ** 2
        weight = raw_weight / raw_weight.sum()
        grid_ids = grid["grid_id"].to_numpy(dtype=int)
        for variable in variables:
            columns = [
                f"{source}__grid{grid_id:02d}__{variable}" for grid_id in grid_ids
            ]
            missing = set(columns).difference(surface.columns)
            if missing:
                raise RuntimeError(f"missing turbine NWP columns: {sorted(missing)}")
            values = surface[columns].to_numpy(dtype="float32")
            additions[f"{source}__{variable}"] = values @ weight
    for alias, (u_name, v_name) in VECTOR_PAIRS.items():
        u = additions[u_name]
        v = additions[v_name]
        speed = np.hypot(u, v)
        additions[f"turbine_wind__{alias}__speed"] = speed
        additions[f"turbine_wind__{alias}__speed2"] = speed**2
        additions[f"turbine_wind__{alias}__speed3"] = speed**3
        additions[f"turbine_wind__{alias}__dir_sin"] = np.divide(
            v, speed, out=np.zeros_like(v), where=speed > 1e-8
        )
        additions[f"turbine_wind__{alias}__dir_cos"] = np.divide(
            u, speed, out=np.zeros_like(u), where=speed > 1e-8
        )
    return pd.DataFrame(additions, index=surface.index).astype("float32")


def _static_features(turbine: pd.Series, rows: int) -> pd.DataFrame:
    values = {
        "turbine_static__number": float(turbine["turbine_number"]),
        "turbine_static__latitude": float(turbine["latitude"]),
        "turbine_static__longitude": float(turbine["longitude"]),
        "turbine_static__rotor": float(turbine["rotor_diameter_m"]),
        "turbine_static__capacity": float(turbine["capacity_mw"]),
        "turbine_static__unison": float(turbine["manufacturer"] == "UNISON"),
    }
    return pd.DataFrame(
        {name: np.full(rows, value, dtype="float32") for name, value in values.items()}
    )


def _turbine_target_map(targets: pd.DataFrame, turbine: pd.Series) -> pd.Series:
    selected = targets.loc[
        targets["group_id"].eq(int(turbine["group_id"]))
        & targets["turbine_id"].eq(int(turbine["turbine_number"])),
        ["forecast_kst_dtm", "turbine_kwh"],
    ]
    if selected.empty or selected["forecast_kst_dtm"].duplicated().any():
        raise RuntimeError("turbine target identity changed")
    return selected.set_index("forecast_kst_dtm")["turbine_kwh"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--fold", choices=FOLDS, required=True)
    parser.add_argument("--sitewind-cache-id", required=True)
    parser.add_argument("--iterations", nargs="+", type=int, default=[100, 200, 300])
    parser.add_argument("--num-leaves", type=int, default=63)
    args = parser.parse_args()
    checkpoints = sorted(set(args.iterations))
    if not checkpoints or checkpoints[0] < 50 or checkpoints[-1] > 600:
        raise ValueError("iterations must be between 50 and 600")
    if args.num_leaves not in {31, 63, 127}:
        raise ValueError("num-leaves must be 31, 63, or 127")
    if _sha256(OPEN) != OPEN_SHA or _sha256(BASELINE) != BASELINE_SHA:
        raise RuntimeError("immutable competition input hash mismatch")

    started = time.perf_counter()
    surface, base_columns, _ = _surface()
    validation = _validation_mask(surface, args.fold)
    validation_start = surface.loc[validation, "forecast_kst_dtm"].min()
    preceding = surface["forecast_kst_dtm"].lt(validation_start).to_numpy()
    cached = np.load(
        OUTPUT / f"{args.sitewind_cache_id}-{args.fold}-sitewind-features.npz"
    )
    group_matrix = surface[base_columns].astype("float32")
    sitewind_columns = _add_site_wind_features(
        group_matrix, cached["legacy"], cached["allweather"]
    )
    selected_features = _feature_names(args.fold)
    turbines, geometry = _metadata_and_geometry()
    targets = _hourly_turbine_targets()
    training_matrices: list[np.ndarray] = []
    training_targets: list[np.ndarray] = []
    training_groups: list[np.ndarray] = []
    validation_parts: list[tuple[pd.Series, np.ndarray, np.ndarray, float]] = []
    expanded_feature_names: list[str] | None = None
    turbine_diagnostics: dict[str, object] = {}
    for _, turbine in turbines.sort_values("turbine_id").iterrows():
        group_id = int(turbine["group_id"])
        group = surface["group_id"].eq(group_id).to_numpy()
        local_weather = _turbine_weather(surface, turbine, geometry)
        positions = np.flatnonzero(group)
        static = _static_features(turbine, len(positions))
        local = pd.concat(
            [
                group_matrix.iloc[positions][selected_features].reset_index(drop=True),
                local_weather.iloc[positions].reset_index(drop=True),
                static,
            ],
            axis=1,
        )
        names = local.columns.tolist()
        if expanded_feature_names is None:
            expanded_feature_names = names
        elif names != expanded_feature_names:
            raise RuntimeError("shared turbine feature contract changed")
        target_map = _turbine_target_map(targets, turbine)
        turbine_target = surface.iloc[positions]["forecast_kst_dtm"].map(target_map)
        capacity_kwh = float(turbine["capacity_mw"]) * 1000.0
        normalized_target = turbine_target / capacity_kwh
        fit = preceding[positions] & normalized_target.notna().to_numpy()
        apply = validation[positions]
        training_matrices.append(local.loc[fit].to_numpy(dtype="float32"))
        training_targets.append(normalized_target.loc[fit].to_numpy(dtype="float32"))
        training_groups.append(np.full(int(fit.sum()), group_id, dtype="int8"))
        validation_parts.append(
            (
                turbine,
                positions[apply],
                local.loc[apply].to_numpy(dtype="float32"),
                capacity_kwh,
            )
        )
        turbine_diagnostics[str(turbine["turbine_id"])] = {
            "group_id": group_id,
            "training_rows": int(fit.sum()),
            "validation_rows": int(apply.sum()),
            "target_mean": float(normalized_target.loc[fit].mean()),
        }
    x_train = np.concatenate(training_matrices)
    y_train = np.concatenate(training_targets)
    group_train = np.concatenate(training_groups)
    group_counts = {
        group_id: int((group_train == group_id).sum()) for group_id in CAPACITIES
    }
    sample_weight = np.asarray(
        [len(group_train) / (3.0 * group_counts[int(group)]) for group in group_train],
        dtype="float32",
    ) * np.clip(y_train, 0.10, None)
    model = LGBMRegressor(
        objective="l1",
        n_estimators=max(checkpoints),
        learning_rate=0.025,
        num_leaves=args.num_leaves,
        min_child_samples=100,
        max_bin=255,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.2,
        reg_lambda=5.0,
        random_state=20260803,
        n_jobs=6,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )
    model.fit(x_train, y_train, sample_weight=sample_weight)
    base = surface.loc[validation, BASE_COLUMNS].copy()
    base_positions = np.flatnonzero(validation)
    base_position_map = {position: index for index, position in enumerate(base_positions)}
    parent = pd.read_parquet(PARENT_PATH)
    parent = parent.loc[parent["fold_id"].eq(args.fold)]
    sweep: dict[str, object] = {}
    best: tuple[float, int, pd.DataFrame, pd.DataFrame, str, dict[str, object]] | None = None
    for checkpoint in checkpoints:
        group_prediction = np.zeros(len(base), dtype=float)
        for _turbine, global_positions, values, capacity_kwh in validation_parts:
            normalized = np.clip(
                model.predict(values, num_iteration=checkpoint), 0.0, 1.075
            )
            local_positions = np.asarray(
                [base_position_map[int(position)] for position in global_positions],
                dtype=int,
            )
            group_prediction[local_positions] += normalized * capacity_kwh
        normalized_group = group_prediction / base["group_id"].map(CAPACITIES).to_numpy(
            dtype=float
        )
        policies = _policies(base, normalized_group)
        raw_policy, raw_output = _best_raw(base, policies)
        blended, selections = _screen_blends(base, policies, parent)
        raw_score = _score(raw_output)
        blend_score = _score(blended)
        sweep[str(checkpoint)] = {
            "raw_policy": raw_policy,
            "raw_score": raw_score,
            "oracle_blend_score": blend_score,
            "oracle_blends": selections,
        }
        choice = (
            blend_score["total"],
            checkpoint,
            blended,
            policies,
            raw_policy,
            selections,
        )
        if best is None or choice[0] > best[0]:
            best = choice
        print(json.dumps({"checkpoint": checkpoint, **sweep[str(checkpoint)]}), flush=True)
    assert best is not None and expanded_feature_names is not None
    output = best[2].assign(fold_id=args.fold, model_id=args.candidate_id)
    output_path = OUTPUT / f"{args.candidate_id}-{args.fold}.parquet"
    policy_path = OUTPUT / f"{args.candidate_id}-{args.fold}-policies.parquet"
    output.to_parquet(output_path, index=False)
    best[3].to_parquet(policy_path, index=False)
    receipt = {
        "schema_version": 1,
        "candidate_id": args.candidate_id,
        "fold_id": args.fold,
        "architecture": "shared_turbine_scada_l1_turbine_local_nwp",
        "scope": "unseen-fold shared-turbine transfer representation screen",
        "selected_checkpoint": best[1],
        "selected_raw_policy": best[4],
        "selected_oracle_blends": best[5],
        "selected_oracle_blend_score": _score(output),
        "sweep": sweep,
        "feature_count": len(expanded_feature_names),
        "group_base_feature_count": len(selected_features),
        "turbine_local_feature_count": len(expanded_feature_names) - len(selected_features),
        "sitewind_feature_count": len(sitewind_columns),
        "training_rows": len(x_train),
        "group_training_rows": group_counts,
        "turbine_diagnostics": turbine_diagnostics,
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
