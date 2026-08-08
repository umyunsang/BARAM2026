"""Predict official group generation through supplied turbine-level SCADA targets."""

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

BASE_COLUMNS = ["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh"]
PARENT_PATH = OUTPUT / "M107_STRICT_TEMPORAL_TOP100-oof.parquet"
TURBINES = {1: tuple(range(1, 7)), 2: tuple(range(7, 13)), 3: tuple(range(1, 6))}


def _hourly_turbine_targets() -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    with zipfile.ZipFile(OPEN) as archive:
        with archive.open("train/scada_vestas_train.csv") as stream:
            vestas = pd.read_csv(stream, parse_dates=["kst_dtm"])
        for group_id in (1, 2):
            columns = [
                f"vestas_wtg{number:02d}_power_kw10m"
                for number in TURBINES[group_id]
            ]
            values = vestas[columns].where(
                lambda frame: (frame >= -10.0) & (frame <= 650.0)
            ).clip(lower=0.0)
            values["forecast_kst_dtm"] = vestas["kst_dtm"].dt.ceil("h")
            grouped = values.groupby("forecast_kst_dtm", sort=True).agg(
                ["sum", "count"]
            )
            for number, column in zip(TURBINES[group_id], columns, strict=True):
                target = grouped[(column, "sum")].where(
                    grouped[(column, "count")].eq(6)
                )
                parts.append(
                    pd.DataFrame(
                        {
                            "forecast_kst_dtm": grouped.index,
                            "group_id": group_id,
                            "turbine_id": number,
                            "turbine_kwh": target.to_numpy(dtype=float),
                        }
                    )
                )
        with archive.open("train/scada_unison_train.csv") as stream:
            unison = pd.read_csv(stream, parse_dates=["kst_dtm"])
        columns = [
            f"unison_wtg{number:02d}_power_kw10m" for number in TURBINES[3]
        ]
        values = unison[columns].where(
            lambda frame: (frame >= 0.0) & (frame <= 750.0)
        )
        values["forecast_kst_dtm"] = (
            unison["kst_dtm"].dt.floor("h") + pd.Timedelta(1, unit="h")
        )
        grouped = values.groupby("forecast_kst_dtm", sort=True).agg(["sum", "count"])
        for number, column in zip(TURBINES[3], columns, strict=True):
            target = grouped[(column, "sum")].where(grouped[(column, "count")].eq(6))
            parts.append(
                pd.DataFrame(
                    {
                        "forecast_kst_dtm": grouped.index,
                        "group_id": 3,
                        "turbine_id": number,
                        "turbine_kwh": target.to_numpy(dtype=float),
                    }
                )
            )
    return pd.concat(parts, ignore_index=True)


def _fit_turbines(
    surface: pd.DataFrame,
    matrix: pd.DataFrame,
    selected_features: list[str],
    validation: np.ndarray,
    validation_start: pd.Timestamp,
    targets: pd.DataFrame,
    objective: str,
    iterations: int,
    leaves: int,
) -> tuple[np.ndarray, dict[str, object]]:
    normalized_prediction = np.full(len(surface), np.nan, dtype="float32")
    diagnostics: dict[str, object] = {}
    for group_id, turbine_ids in TURBINES.items():
        positions = np.flatnonzero(surface["group_id"].eq(group_id).to_numpy())
        group_surface = surface.iloc[positions]
        group_matrix = matrix.iloc[positions][selected_features]
        apply = validation[positions]
        if int(apply.sum()) == 0:
            continue
        turbine_capacity = CAPACITIES[group_id] / len(turbine_ids)
        group_predictions: list[np.ndarray] = []
        group_diagnostics: dict[str, object] = {}
        for turbine_id in turbine_ids:
            target = targets.loc[
                targets["group_id"].eq(group_id)
                & targets["turbine_id"].eq(turbine_id),
                ["forecast_kst_dtm", "turbine_kwh"],
            ]
            target_map = dict(
                zip(target["forecast_kst_dtm"], target["turbine_kwh"], strict=True)
            )
            turbine_target = group_surface["forecast_kst_dtm"].map(target_map)
            training = (
                group_surface["forecast_kst_dtm"].lt(validation_start).to_numpy()
                & turbine_target.notna().to_numpy()
            )
            normalized_target = turbine_target / turbine_capacity
            model = LGBMRegressor(
                objective=objective,
                n_estimators=iterations,
                learning_rate=0.025,
                num_leaves=leaves,
                min_child_samples=80,
                max_bin=255,
                subsample=0.9,
                subsample_freq=1,
                colsample_bytree=0.8,
                reg_alpha=0.2,
                reg_lambda=4.0,
                random_state=20260802 + 10 * group_id + turbine_id,
                n_jobs=6,
                deterministic=True,
                force_col_wise=True,
                verbosity=-1,
            )
            model.fit(
                group_matrix.loc[training],
                normalized_target.loc[training],
                sample_weight=normalized_target.loc[training].clip(lower=0.10),
            )
            prediction = np.clip(model.predict(group_matrix.loc[apply]), 0.0, 1.075)
            group_predictions.append(prediction)
            group_diagnostics[str(turbine_id)] = {
                "training_rows": int(training.sum()),
                "target_mean": float(normalized_target.loc[training].mean()),
                "target_missing_rows": int(
                    group_surface["forecast_kst_dtm"]
                    .lt(validation_start)
                    .sum()
                    - training.sum()
                ),
            }
        mean_prediction = np.mean(np.column_stack(group_predictions), axis=1)
        normalized_prediction[positions[apply]] = mean_prediction
        diagnostics[str(group_id)] = group_diagnostics
    if not np.isfinite(normalized_prediction[validation]).all():
        raise RuntimeError("turbine decomposition produced non-finite validation values")
    return normalized_prediction, diagnostics


def _policy_values(point: np.ndarray) -> dict[str, np.ndarray]:
    policies: dict[str, np.ndarray] = {}
    for scale in (0.90, 0.95, 1.00, 1.05, 1.10):
        for shift in np.arange(-0.05, 0.0501, 0.005):
            normalized = np.clip(scale * point + shift, 0.075, 1.075)
            tag = f"S{scale:g}_O{shift:+.3f}"
            policies[tag] = normalized
            policies[f"{tag}_SNAP"] = np.round(normalized / 0.0025) * 0.0025
    return policies


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--fold", choices=FOLDS, required=True)
    parser.add_argument("--sitewind-cache-id", required=True)
    parser.add_argument("--objective", choices=("l1", "l2", "huber"), default="l1")
    parser.add_argument("--iterations", type=int, default=240)
    parser.add_argument("--leaves", type=int, choices=(7, 15, 31), default=15)
    args = parser.parse_args()
    if not 80 <= args.iterations <= 500:
        raise ValueError("iterations must be between 80 and 500")
    if _sha256(OPEN) != OPEN_SHA or _sha256(BASELINE) != BASELINE_SHA:
        raise RuntimeError("immutable competition input hash mismatch")

    started = time.perf_counter()
    surface, base_columns, _ = _surface()
    validation = _validation_mask(surface, args.fold)
    validation_start = pd.Timestamp(
        surface.loc[validation, "forecast_kst_dtm"].min()
    )
    cached = np.load(
        OUTPUT / f"{args.sitewind_cache_id}-{args.fold}-sitewind-features.npz"
    )
    matrix = surface[base_columns].astype("float32")
    sitewind_columns = _add_site_wind_features(
        matrix, cached["legacy"], cached["allweather"]
    )
    selected_features = _feature_names(args.fold)
    missing = set(selected_features).difference(matrix.columns)
    if missing:
        raise RuntimeError(f"missing fixed features: {sorted(missing)}")
    targets = _hourly_turbine_targets()
    normalized, diagnostics = _fit_turbines(
        surface,
        matrix,
        selected_features,
        validation,
        validation_start,
        targets,
        args.objective,
        args.iterations,
        args.leaves,
    )
    base = surface.loc[validation, BASE_COLUMNS].copy()
    point = normalized[validation]
    policy_predictions: dict[str, np.ndarray] = {}
    raw_scores: dict[str, dict[str, float]] = {}
    capacity = base["group_id"].map(CAPACITIES).to_numpy(dtype=float)
    for policy, candidate in _policy_values(point).items():
        prediction = candidate * capacity
        policy_predictions[policy] = prediction
        raw_scores[policy] = _score(base.assign(prediction_kwh=prediction))
    policies = pd.concat(
        [base, pd.DataFrame(policy_predictions, index=base.index)], axis=1
    )
    raw_best_policy = max(raw_scores, key=lambda name: raw_scores[name]["total"])
    raw_output = base.assign(prediction_kwh=policy_predictions[raw_best_policy])
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
        "architecture": "turbine_level_scada_supervision_lgbm_aggregation",
        "scope": (
            "same-fold representation screen; turbine SCADA is training target only "
            "and inference uses supplied NWP-derived features"
        ),
        "objective": args.objective,
        "iterations": args.iterations,
        "leaves": args.leaves,
        "turbine_count": sum(map(len, TURBINES.values())),
        "turbine_diagnostics": diagnostics,
        "feature_count": len(selected_features),
        "selected_feature_names": selected_features,
        "raw_best_policy": raw_best_policy,
        "raw_best_score": _score(raw_output),
        "oracle_blend_score": _score(output),
        "oracle_blends": selections,
        "sitewind_feature_count": len(sitewind_columns),
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
