"""Stack strict turbine-level NWP-to-wind teachers into empirical power curves."""

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
from run_inner_policy_classifier import _group_total
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
from run_shared_turbine_model import _metadata_and_geometry, _turbine_weather
from run_site_wind_classifier import FOLDS
from run_site_wind_teacher import _strict_preceding_mask, _validation_mask

BASE_COLUMNS = ["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh"]
TEACHER_PROFILES = ("global", "gfs", "ldaps")
POWER_OBJECTIVES = ("l1", "huber", "l2")
SCALES = tuple(float(value) for value in np.arange(0.90, 1.1001, 0.025))
OFFSETS = tuple(float(value) for value in np.arange(-0.05, 0.0501, 0.005))


def _read_scada_before(
    archive: zipfile.ZipFile,
    member: str,
    cutoff: pd.Timestamp,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    with archive.open(member) as stream:
        for chunk in pd.read_csv(stream, parse_dates=["kst_dtm"], chunksize=100_000):
            selected = chunk.loc[chunk["kst_dtm"].lt(cutoff)]
            if not selected.empty:
                parts.append(selected)
            if chunk["kst_dtm"].max() >= cutoff:
                break
    if not parts:
        raise RuntimeError(f"no preceding SCADA rows found in {member}")
    result = pd.concat(parts, ignore_index=True)
    if not result["kst_dtm"].lt(cutoff).all():
        raise RuntimeError("SCADA cutoff filtering failed")
    return result


def _aggregate_turbine_scada(cutoff: pd.Timestamp) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    with zipfile.ZipFile(OPEN) as archive:
        configurations = (
            (
                "VESTAS",
                _read_scada_before(
                    archive, "train/scada_vestas_train.csv", cutoff
                ),
                tuple(range(1, 13)),
            ),
            (
                "UNISON",
                _read_scada_before(
                    archive, "train/scada_unison_train.csv", cutoff
                ),
                tuple(range(1, 6)),
            ),
        )
    for manufacturer, frame, turbine_numbers in configurations:
        prefix = manufacturer.lower()
        timestamp = (
            frame["kst_dtm"].dt.ceil("h")
            if manufacturer == "VESTAS"
            else frame["kst_dtm"].dt.floor("h") + np.timedelta64(1, "h")
        )
        for turbine_number in turbine_numbers:
            stem = f"{prefix}_wtg{turbine_number:02d}"
            power = frame[f"{stem}_power_kw10m"].where(
                lambda values: values.between(-10.0, 750.0)
            ).clip(lower=0.0)
            wind_speed = frame[f"{stem}_ws"].where(
                lambda values: values.between(0.0, 45.0)
            )
            direction = np.deg2rad(
                frame[f"{stem}_wd"].where(
                    lambda values: values.between(0.0, 360.0)
                )
            )
            raw = pd.DataFrame(
                {
                    "forecast_kst_dtm": timestamp,
                    "power": power,
                    "wind_speed": wind_speed,
                    "direction_sin": np.sin(direction),
                    "direction_cos": np.cos(direction),
                }
            )
            grouped = raw.groupby("forecast_kst_dtm", sort=True).agg(
                power_sum=("power", "sum"),
                power_count=("power", "count"),
                wind_speed=("wind_speed", "mean"),
                wind_count=("wind_speed", "count"),
                direction_sin=("direction_sin", "mean"),
                direction_cos=("direction_cos", "mean"),
                direction_count=("direction_sin", "count"),
            )
            grouped["turbine_kwh"] = grouped["power_sum"].where(
                grouped["power_count"].eq(6)
            )
            grouped["wind_speed"] = grouped["wind_speed"].where(
                grouped["wind_count"].eq(6)
            )
            grouped[["direction_sin", "direction_cos"]] = grouped[
                ["direction_sin", "direction_cos"]
            ].where(grouped["direction_count"].eq(6), np.nan)
            parts.append(
                grouped.reset_index()[
                    [
                        "forecast_kst_dtm",
                        "turbine_kwh",
                        "wind_speed",
                        "direction_sin",
                        "direction_cos",
                    ]
                ].assign(
                    manufacturer=manufacturer,
                    turbine_number=turbine_number,
                )
            )
    output = pd.concat(parts, ignore_index=True)
    output = output.loc[output["forecast_kst_dtm"].lt(cutoff)].reset_index(
        drop=True
    )
    if not output["forecast_kst_dtm"].lt(cutoff).all():
        raise RuntimeError("hourly SCADA aggregation crossed the outer cutoff")
    return output


def _teacher_model(iterations: int, seed: int) -> LGBMRegressor:
    return LGBMRegressor(
        objective="l2",
        n_estimators=iterations,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=40,
        max_bin=255,
        subsample=0.90,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=3.0,
        random_state=seed,
        n_jobs=6,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )


def _teacher_feature_profiles(matrix: pd.DataFrame) -> dict[str, list[str]]:
    common = [
        name
        for name in matrix
        if name in {"lead_hour", "cal__hour_sin", "cal__hour_cos", "cal__doy_sin", "cal__doy_cos"}
        or name.startswith("turbine_static__")
    ]
    gfs = [
        name
        for name in matrix
        if name.startswith("gfs") or name.startswith("turbine_wind__gfs")
    ]
    ldaps = [
        name
        for name in matrix
        if name.startswith("ldaps") or name.startswith("turbine_wind__ldaps")
    ]
    global_names = [
        name
        for name in matrix
        if not name.startswith("turbine_static__")
    ]
    profiles = {
        "global": list(dict.fromkeys([*global_names, *common])),
        "gfs": list(dict.fromkeys([*gfs, *common])),
        "ldaps": list(dict.fromkeys([*ldaps, *common])),
    }
    for profile, names in profiles.items():
        if len(names) < 20:
            raise RuntimeError(f"{profile} turbine teacher has too few features")
    return profiles


def _strict_before(
    batch_last_forecast: pd.Series,
    cutoff: pd.Timestamp,
) -> np.ndarray:
    return batch_last_forecast.lt(cutoff).to_numpy()


def _crossfit_teacher(
    matrix: pd.DataFrame,
    profiles: dict[str, list[str]],
    target: pd.Series,
    training: np.ndarray,
    validation: np.ndarray,
    available: pd.Series,
    batch_last_forecast: pd.Series,
    iterations: int,
    seed_offset: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, object]]:
    training_batches = (
        available.loc[training & target.notna().to_numpy()]
        .drop_duplicates()
        .sort_values()
    )
    if len(training_batches) < 120:
        raise RuntimeError("turbine wind teacher has too few preceding batches")
    boundaries = [int(len(training_batches) * fraction) for fraction in (0.25, 0.50, 0.75)]
    oof = {profile: np.full(len(matrix), np.nan, dtype="float32") for profile in profiles}
    outer = {
        profile: np.full(int(validation.sum()), np.nan, dtype="float32")
        for profile in profiles
    }
    diagnostics: dict[str, object] = {}
    for profile_index, (profile, names) in enumerate(profiles.items()):
        blocks: list[dict[str, object]] = []
        for block_index, start_index in enumerate(boundaries):
            stop_index = (
                boundaries[block_index + 1]
                if block_index + 1 < len(boundaries)
                else len(training_batches)
            )
            block_batches = set(training_batches.iloc[start_index:stop_index])
            apply = training & available.isin(block_batches).to_numpy()
            cutoff = pd.Timestamp(training_batches.iloc[start_index])
            fit = (
                training
                & target.notna().to_numpy()
                & _strict_before(batch_last_forecast, cutoff)
            )
            if int(fit.sum()) < 500 or int(apply.sum()) < 100:
                raise RuntimeError("turbine wind teacher temporal block is too small")
            model = _teacher_model(
                iterations,
                20260803 + 1000 * seed_offset + 100 * profile_index + block_index,
            )
            model.fit(matrix.loc[fit, names], target.loc[fit])
            oof[profile][apply] = model.predict(matrix.loc[apply, names])
            blocks.append(
                {
                    "fit_rows": int(fit.sum()),
                    "apply_rows": int(apply.sum()),
                    "cutoff": str(cutoff),
                }
            )
        final_fit = training & target.notna().to_numpy()
        model = _teacher_model(
            iterations,
            20260803 + 1000 * seed_offset + 100 * profile_index + 99,
        )
        model.fit(matrix.loc[final_fit, names], target.loc[final_fit])
        outer[profile][:] = model.predict(matrix.loc[validation, names])
        oof_rows = np.isfinite(oof[profile]) & target.notna().to_numpy()
        error = oof[profile][oof_rows] - target.loc[oof_rows].to_numpy(dtype=float)
        diagnostics[profile] = {
            "feature_count": len(names),
            "blocks": blocks,
            "oof_rows": int(oof_rows.sum()),
            "oof_mae": float(np.mean(np.abs(error))),
            "oof_rmse": float(np.sqrt(np.mean(error**2))),
            "outer_fit_rows": int(final_fit.sum()),
        }
    return oof, outer, diagnostics


def _power_features(
    teacher: dict[str, np.ndarray],
    local: pd.DataFrame,
    turbine: pd.Series,
) -> pd.DataFrame:
    global_wind = teacher["global"]
    gfs_wind = teacher["gfs"]
    ldaps_wind = teacher["ldaps"]
    mean_wind = (gfs_wind + ldaps_wind) / 2.0
    data: dict[str, np.ndarray] = {
        "teacher__global": global_wind,
        "teacher__gfs": gfs_wind,
        "teacher__ldaps": ldaps_wind,
        "teacher__source_mean": mean_wind,
        "teacher__source_delta": ldaps_wind - gfs_wind,
        "teacher__source_disagreement": np.abs(ldaps_wind - gfs_wind),
    }
    for source, values in (
        ("global", global_wind),
        ("gfs", gfs_wind),
        ("ldaps", ldaps_wind),
        ("source_mean", mean_wind),
    ):
        data[f"teacher__{source}2"] = values**2
        data[f"teacher__{source}3"] = values**3
    direction_names = [
        name
        for name in local
        if name.startswith("turbine_wind__")
        and name.endswith(("__speed", "__dir_sin", "__dir_cos"))
    ]
    for name in direction_names:
        data[name] = local[name].to_numpy(dtype="float32")
    rows = len(local)
    data.update(
        {
            "static__manufacturer_unison": np.full(
                rows, float(turbine["manufacturer"] == "UNISON")
            ),
            "static__group": np.full(rows, float(turbine["group_id"])),
            "static__turbine_number": np.full(
                rows, float(turbine["turbine_number"])
            ),
            "static__latitude": np.full(rows, float(turbine["latitude"])),
            "static__longitude": np.full(rows, float(turbine["longitude"])),
            "static__rotor": np.full(rows, float(turbine["rotor_diameter_m"])),
        }
    )
    for name in ("lead_hour", "cal__hour_sin", "cal__hour_cos", "cal__doy_sin", "cal__doy_cos"):
        if name in local:
            data[name] = local[name].to_numpy(dtype="float32")
    return pd.DataFrame(data, index=local.index).astype("float32")


def _power_model(objective: str, iterations: int, seed: int) -> LGBMRegressor:
    return LGBMRegressor(
        objective=objective,
        n_estimators=iterations,
        learning_rate=0.025,
        num_leaves=31,
        min_child_samples=100,
        max_bin=255,
        subsample=0.90,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.2,
        reg_lambda=4.0,
        random_state=seed,
        n_jobs=6,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )


def _sample_weight(groups: np.ndarray, target: np.ndarray) -> np.ndarray:
    counts = {group_id: int((groups == group_id).sum()) for group_id in CAPACITIES}
    inverse_group = np.asarray(
        [len(groups) / (3.0 * counts[int(group)]) for group in groups],
        dtype="float32",
    )
    return inverse_group * (0.25 + 0.75 * np.clip(target, 0.0, 1.0))


def _aggregate_turbines(
    metadata: pd.DataFrame,
    prediction: np.ndarray,
) -> pd.DataFrame:
    work = metadata.copy()
    work["prediction_kwh"] = np.clip(prediction, 0.0, 1.075) * work[
        "turbine_capacity_kwh"
    ].to_numpy(dtype=float)
    counts = work.groupby(["forecast_kst_dtm", "group_id"])["turbine_id"].nunique()
    expected = {1: 6, 2: 6, 3: 5}
    for group_id, turbine_count in expected.items():
        part = counts.loc[counts.index.get_level_values("group_id") == group_id]
        if not part.eq(turbine_count).all():
            raise RuntimeError(f"group {group_id} turbine aggregation is incomplete")
    return (
        work.groupby(["forecast_kst_dtm", "group_id"], as_index=False)[
            "prediction_kwh"
        ]
        .sum()
        .sort_values(["forecast_kst_dtm", "group_id"])
    )


def _join_actual(surface: pd.DataFrame, prediction: pd.DataFrame) -> pd.DataFrame:
    actual = surface[
        ["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh"]
    ].drop_duplicates(["forecast_kst_dtm", "group_id"])
    output = actual.merge(
        prediction,
        on=["forecast_kst_dtm", "group_id"],
        how="inner",
        validate="one_to_one",
    )
    return output.sort_values(["forecast_kst_dtm", "group_id"]).reset_index(drop=True)


def _select_calibration(
    frames: dict[str, pd.DataFrame],
) -> tuple[dict[int, dict[str, float | str | bool]], dict[str, object]]:
    selections: dict[int, dict[str, float | str | bool]] = {}
    diagnostics: dict[str, object] = {}
    for group_id, capacity in CAPACITIES.items():
        best: tuple[
            float,
            str,
            float,
            float,
            bool,
            dict[str, float],
        ] | None = None
        ranked: list[tuple[float, str, dict[str, float]]] = []
        for objective, frame in frames.items():
            part = frame.loc[frame["group_id"].eq(group_id)]
            actual = part["actual_kwh"].to_numpy(dtype=float) / capacity
            raw = part["prediction_kwh"].to_numpy(dtype=float) / capacity
            for scale in SCALES:
                for offset in OFFSETS:
                    adjusted = np.clip(scale * raw + offset, 0.0, 1.075)
                    for snap in (False, True):
                        prediction = (
                            np.round(adjusted / 0.0025) * 0.0025
                            if snap
                            else adjusted
                        )
                        score = _group_total(actual, prediction)
                        tag = f"{objective}_S{scale:.3f}_O{offset:+.3f}_{int(snap)}"
                        ranked.append((score["total"], tag, score))
                        choice = (
                            score["total"],
                            objective,
                            scale,
                            offset,
                            snap,
                            score,
                        )
                        if best is None or choice[0] > best[0]:
                            best = choice
        assert best is not None
        selections[group_id] = {
            "objective": best[1],
            "scale": best[2],
            "offset": best[3],
            "snap": best[4],
        }
        diagnostics[str(group_id)] = {
            "selection": selections[group_id],
            "score": best[5],
            "top_five": [
                {"policy": tag, "score": score}
                for _, tag, score in sorted(ranked, reverse=True)[:5]
            ],
        }
    return selections, diagnostics


def _apply_selections(
    frames: dict[str, pd.DataFrame],
    selections: dict[int, dict[str, float | str | bool]],
) -> pd.DataFrame:
    reference = next(iter(frames.values()))
    output = reference[BASE_COLUMNS].copy()
    prediction = np.empty(len(output), dtype=float)
    for group_id, capacity in CAPACITIES.items():
        rows = output["group_id"].eq(group_id).to_numpy()
        selection = selections[group_id]
        source = frames[str(selection["objective"])]
        normalized = (
            source.loc[rows, "prediction_kwh"].to_numpy(dtype=float) / capacity
        )
        normalized = np.clip(
            float(selection["scale"]) * normalized + float(selection["offset"]),
            0.0,
            1.075,
        )
        if bool(selection["snap"]):
            normalized = np.round(normalized / 0.0025) * 0.0025
        prediction[rows] = normalized * capacity
    output["prediction_kwh"] = prediction
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--fold", choices=FOLDS, required=True)
    parser.add_argument("--teacher-iterations", type=int, default=240)
    parser.add_argument("--power-iterations", type=int, default=400)
    parser.add_argument("--calibration-days", type=int, default=60)
    args = parser.parse_args()
    if not 80 <= args.teacher_iterations <= 500:
        raise ValueError("teacher iterations must be between 80 and 500")
    if not 100 <= args.power_iterations <= 800:
        raise ValueError("power iterations must be between 100 and 800")
    if not 30 <= args.calibration_days <= 120:
        raise ValueError("calibration days must be between 30 and 120")
    if _sha256(OPEN) != OPEN_SHA or _sha256(BASELINE) != BASELINE_SHA:
        raise RuntimeError("immutable competition input hash mismatch")

    started = time.perf_counter()
    surface, _, _ = _surface()
    validation = _validation_mask(surface, args.fold)
    preceding = _strict_preceding_mask(surface, validation)
    validation_start = pd.Timestamp(
        surface.loc[validation, "forecast_kst_dtm"].min()
    )
    scada = _aggregate_turbine_scada(validation_start)
    turbines, geometry = _metadata_and_geometry()
    batch_last_forecast = surface.groupby(
        "data_available_kst_dtm", sort=False
    )["forecast_kst_dtm"].transform("max")
    base_features = [
        name
        for name in _feature_names(args.fold)
        if not name.startswith("sitewind__")
    ]

    oof_feature_parts: list[pd.DataFrame] = []
    oof_metadata_parts: list[pd.DataFrame] = []
    oof_target_parts: list[np.ndarray] = []
    validation_feature_parts: list[pd.DataFrame] = []
    validation_metadata_parts: list[pd.DataFrame] = []
    teacher_diagnostics: dict[str, object] = {}
    for turbine_index, (_, turbine) in enumerate(
        turbines.sort_values("turbine_id").iterrows()
    ):
        group_id = int(turbine["group_id"])
        group_positions = np.flatnonzero(surface["group_id"].eq(group_id).to_numpy())
        group_surface = surface.iloc[group_positions].reset_index(drop=True)
        group_preceding = preceding[group_positions]
        group_validation = validation[group_positions]
        group_batch_last = batch_last_forecast.iloc[group_positions].reset_index(drop=True)
        local_weather = _turbine_weather(group_surface, turbine, geometry).reset_index(
            drop=True
        )
        selected_group_features = [
            name for name in base_features if name in group_surface.columns
        ]
        local = pd.concat(
            [
                group_surface[selected_group_features].reset_index(drop=True),
                local_weather,
            ],
            axis=1,
        )
        local["turbine_static__latitude"] = float(turbine["latitude"])
        local["turbine_static__longitude"] = float(turbine["longitude"])
        local["turbine_static__rotor"] = float(turbine["rotor_diameter_m"])
        local["turbine_static__unison"] = float(
            turbine["manufacturer"] == "UNISON"
        )
        profiles = _teacher_feature_profiles(local)
        turbine_scada = scada.loc[
            scada["manufacturer"].eq(turbine["manufacturer"])
            & scada["turbine_number"].eq(int(turbine["turbine_number"])),
            ["forecast_kst_dtm", "turbine_kwh", "wind_speed"],
        ].drop_duplicates("forecast_kst_dtm")
        scada_map = turbine_scada.set_index("forecast_kst_dtm")
        wind_target = group_surface["forecast_kst_dtm"].map(scada_map["wind_speed"])
        power_target = group_surface["forecast_kst_dtm"].map(scada_map["turbine_kwh"])
        oof_teacher, outer_teacher, diagnostics = _crossfit_teacher(
            local,
            profiles,
            wind_target,
            group_preceding,
            group_validation,
            group_surface["data_available_kst_dtm"],
            group_batch_last,
            args.teacher_iterations,
            turbine_index,
        )
        teacher_diagnostics[str(turbine["turbine_id"])] = diagnostics
        oof_rows = np.logical_and.reduce(
            [np.isfinite(oof_teacher[profile]) for profile in TEACHER_PROFILES]
        )
        oof_features = _power_features(
            {profile: oof_teacher[profile][oof_rows] for profile in TEACHER_PROFILES},
            local.loc[oof_rows].reset_index(drop=True),
            turbine,
        )
        oof_feature_parts.append(oof_features)
        turbine_capacity = float(turbine["capacity_mw"]) * 1000.0
        oof_target_parts.append(
            power_target.loc[oof_rows].to_numpy(dtype=float) / turbine_capacity
        )
        oof_metadata_parts.append(
            pd.DataFrame(
                {
                    "forecast_kst_dtm": group_surface.loc[
                        oof_rows, "forecast_kst_dtm"
                    ].to_numpy(),
                    "group_id": group_id,
                    "turbine_id": str(turbine["turbine_id"]),
                    "turbine_capacity_kwh": turbine_capacity,
                }
            )
        )
        validation_features = _power_features(
            outer_teacher,
            local.loc[group_validation].reset_index(drop=True),
            turbine,
        )
        validation_feature_parts.append(validation_features)
        validation_metadata_parts.append(
            pd.DataFrame(
                {
                    "forecast_kst_dtm": group_surface.loc[
                        group_validation, "forecast_kst_dtm"
                    ].to_numpy(),
                    "group_id": group_id,
                    "turbine_id": str(turbine["turbine_id"]),
                    "turbine_capacity_kwh": turbine_capacity,
                }
            )
        )
        print(
            json.dumps(
                {
                    "turbine_completed": turbine_index + 1,
                    "turbine_total": len(turbines),
                    "turbine_id": str(turbine["turbine_id"]),
                    "oof_rows": int(oof_rows.sum()),
                }
            ),
            flush=True,
        )

    x_oof = pd.concat(oof_feature_parts, ignore_index=True)
    oof_metadata = pd.concat(oof_metadata_parts, ignore_index=True)
    y_oof = np.concatenate(oof_target_parts)
    x_validation = pd.concat(validation_feature_parts, ignore_index=True)
    validation_metadata = pd.concat(validation_metadata_parts, ignore_index=True)
    if list(x_oof.columns) != list(x_validation.columns):
        raise RuntimeError("power-stack feature contract changed between OOF and outer")
    labeled = np.isfinite(y_oof)
    calibration_start = validation_start - pd.Timedelta(days=args.calibration_days)
    inner_fit = labeled & oof_metadata["forecast_kst_dtm"].lt(
        calibration_start
    ).to_numpy()
    calibration_rows = oof_metadata["forecast_kst_dtm"].ge(
        calibration_start
    ).to_numpy()
    if int(inner_fit.sum()) < 10_000 or int(calibration_rows.sum()) < 5_000:
        raise RuntimeError("power-stack inner split is too small")

    inner_frames: dict[str, pd.DataFrame] = {}
    outer_frames: dict[str, pd.DataFrame] = {}
    power_diagnostics: dict[str, object] = {}
    for objective_index, objective in enumerate(POWER_OBJECTIVES):
        inner_model = _power_model(
            objective,
            args.power_iterations,
            20260803 + objective_index,
        )
        inner_weight = _sample_weight(
            oof_metadata.loc[inner_fit, "group_id"].to_numpy(dtype=int),
            y_oof[inner_fit],
        )
        inner_model.fit(
            x_oof.loc[inner_fit],
            y_oof[inner_fit],
            sample_weight=inner_weight,
        )
        inner_prediction = inner_model.predict(x_oof.loc[calibration_rows])
        inner_group = _aggregate_turbines(
            oof_metadata.loc[calibration_rows].reset_index(drop=True),
            inner_prediction,
        )
        inner_frames[objective] = _join_actual(surface, inner_group)

        final_model = _power_model(
            objective,
            args.power_iterations,
            20260803 + objective_index,
        )
        final_weight = _sample_weight(
            oof_metadata.loc[labeled, "group_id"].to_numpy(dtype=int),
            y_oof[labeled],
        )
        final_model.fit(
            x_oof.loc[labeled],
            y_oof[labeled],
            sample_weight=final_weight,
        )
        outer_prediction = final_model.predict(x_validation)
        outer_group = _aggregate_turbines(validation_metadata, outer_prediction)
        outer_frames[objective] = _join_actual(surface.loc[validation], outer_group)
        power_diagnostics[objective] = {
            "inner_fit_rows": int(inner_fit.sum()),
            "calibration_rows": int(calibration_rows.sum()),
            "final_fit_rows": int(labeled.sum()),
            "feature_count": x_oof.shape[1],
        }

    selections, calibration_diagnostics = _select_calibration(inner_frames)
    output = _apply_selections(outer_frames, selections)
    output["fold_id"] = args.fold
    output["model_id"] = args.candidate_id
    output_path = OUTPUT / f"{args.candidate_id}-{args.fold}.parquet"
    output.to_parquet(output_path, index=False)
    outer_oracle_selections, outer_oracle_diagnostics = _select_calibration(
        outer_frames
    )
    outer_oracle = _apply_selections(outer_frames, outer_oracle_selections)
    receipt = {
        "schema_version": 1,
        "candidate_id": args.candidate_id,
        "fold_id": args.fold,
        "architecture": "strict_per_turbine_nwp_wind_to_empirical_power_stack",
        "scope": "outer labels and outer SCADA excluded from model and policy selection",
        "fold_score": _score(output),
        "outer_oracle_diagnostic": {
            "score": _score(outer_oracle),
            "selections": {
                str(group_id): value
                for group_id, value in outer_oracle_selections.items()
            },
            "diagnostics": outer_oracle_diagnostics,
        },
        "inner_selections": {
            str(group_id): value for group_id, value in selections.items()
        },
        "inner_calibration_diagnostics": calibration_diagnostics,
        "teacher_diagnostics": teacher_diagnostics,
        "power_diagnostics": power_diagnostics,
        "teacher_iterations": args.teacher_iterations,
        "power_iterations": args.power_iterations,
        "calibration_days": args.calibration_days,
        "power_feature_count": x_oof.shape[1],
        "power_feature_names": x_oof.columns.tolist(),
        "oof_rows": len(x_oof),
        "oof_labeled_rows": int(labeled.sum()),
        "validation_turbine_rows": len(x_validation),
        "runtime_seconds": round(time.perf_counter() - started, 2),
        "prediction_path": str(output_path.relative_to(Path.cwd())),
        "prediction_sha256": _sha256(output_path),
        "observed_validation_scada_used_for_power_prediction": False,
        "new_2024_evaluation": False,
        "lockbox_reopened": False,
        "external_actions": [],
    }
    receipt_path = OUTPUT / f"{args.candidate_id}-{args.fold}.json"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
