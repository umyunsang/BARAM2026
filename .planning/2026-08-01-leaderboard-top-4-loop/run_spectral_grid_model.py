"""Strict low-frequency grid-sequence model inspired by Fourier NWP fusion."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
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
)
from run_site_wind_teacher import _strict_preceding_mask, _validation_mask
from strict_dev_surface import DEV_CUTOFF, development_surface

BASE_COLUMNS = ["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh"]
GRID_SPEED = re.compile(r"^(gfs|ldaps)__grid\d{2}__.*_speed$")
RESEARCH_SOURCE = "https://arxiv.org/abs/2607.17095"


def _spectral_context(
    surface: pd.DataFrame,
    speed_columns: list[str],
) -> pd.DataFrame:
    ordered = surface.sort_values(
        ["data_available_kst_dtm", "group_id", "forecast_kst_dtm"],
        kind="stable",
    )
    generated: dict[str, np.ndarray] = {
        f"spectral__{name}__{suffix}": np.empty(len(surface), dtype="float32")
        for name in speed_columns
        for suffix in ("low1", "low3", "high3", "energy1", "energy2", "energy3")
    }
    for _, part in ordered.groupby(
        ["data_available_kst_dtm", "group_id"], sort=False
    ):
        positions = part.index.to_numpy(dtype=int)
        values = part[speed_columns].to_numpy(dtype=float)
        means = np.nanmean(values, axis=0)
        missing = np.where(~np.isfinite(values))
        if missing[0].size:
            values[missing] = means[missing[1]]
        spectrum = np.fft.rfft(values, axis=0)
        total_energy = np.maximum(np.sum(np.abs(spectrum) ** 2, axis=0), 1e-12)
        reconstructions: dict[str, np.ndarray] = {}
        for cutoff, name in ((1, "low1"), (3, "low3")):
            low = spectrum.copy()
            low[cutoff + 1 :] = 0.0
            reconstructions[name] = np.fft.irfft(low, n=len(values), axis=0)
        reconstructions["high3"] = values - reconstructions["low3"]
        for column_index, column in enumerate(speed_columns):
            for name in ("low1", "low3", "high3"):
                generated[f"spectral__{column}__{name}"][positions] = reconstructions[
                    name
                ][:, column_index]
            for mode in (1, 2, 3):
                ratio = (
                    np.abs(spectrum[mode, column_index]) ** 2 / total_energy[column_index]
                    if mode < len(spectrum)
                    else 0.0
                )
                generated[f"spectral__{column}__energy{mode}"][positions] = ratio
    context = pd.DataFrame(generated, index=surface.index)
    if not np.isfinite(context.to_numpy(dtype=float)).all():
        raise RuntimeError("spectral context contains a non-finite value")
    return context


def _model(iterations: int, seed: int) -> LGBMRegressor:
    return LGBMRegressor(
        objective="l1",
        n_estimators=iterations,
        learning_rate=0.025,
        num_leaves=31,
        min_child_samples=45,
        max_bin=255,
        subsample=0.90,
        subsample_freq=1,
        colsample_bytree=0.82,
        reg_alpha=0.15,
        reg_lambda=4.0,
        random_state=seed,
        n_jobs=6,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument(
        "--fold",
        choices=("dev-2023-Q2", "dev-2023-Q3", "dev-2023-Q4"),
        required=True,
    )
    parser.add_argument("--top-features", type=int, default=240)
    parser.add_argument("--iterations", type=int, default=320)
    args = parser.parse_args()
    if not 120 <= args.top_features <= 360:
        raise ValueError("top-features must be between 120 and 360")
    if not 160 <= args.iterations <= 600:
        raise ValueError("iterations must be between 160 and 600")
    if _sha256(OPEN) != OPEN_SHA or _sha256(BASELINE) != BASELINE_SHA:
        raise RuntimeError("immutable competition input hash mismatch")

    started = time.perf_counter()
    surface, base_features = development_surface()
    if surface["forecast_kst_dtm"].ge(DEV_CUTOFF).any():
        raise RuntimeError("lockbox row reached spectral runner")
    speed_columns = [name for name in base_features if GRID_SPEED.match(name)]
    if len(speed_columns) != 118:
        raise RuntimeError(
            f"grid-speed contract changed: expected 118, got {len(speed_columns)}"
        )
    spectral = _spectral_context(surface, speed_columns)
    matrix = pd.concat(
        [surface[base_features].astype("float32"), spectral], axis=1
    )
    validation = _validation_mask(surface, args.fold)
    history = _strict_preceding_mask(surface, validation)
    target = surface["actual_kwh"] / surface["group_id"].map(CAPACITIES)
    observed = surface["actual_kwh"].notna().to_numpy()
    output = surface.loc[validation, BASE_COLUMNS].copy()
    validation_groups = output["group_id"].to_numpy(dtype=int)
    normalized_prediction = np.empty(len(output), dtype=float)
    diagnostics: dict[str, object] = {}

    for group_id in CAPACITIES:
        group = surface["group_id"].eq(group_id).to_numpy()
        training = history & observed & group
        application = validation & group
        screen = _model(160, 20261000 + group_id)
        screen.fit(
            matrix.loc[training],
            target.loc[training],
            sample_weight=target.loc[training].clip(lower=0.10),
        )
        gain = screen.booster_.feature_importance(importance_type="gain")
        positions = np.argsort(gain)[::-1][: args.top_features]
        selected = [matrix.columns[position] for position in positions]
        model = _model(args.iterations, 20261100 + group_id)
        model.fit(
            matrix.loc[training, selected],
            target.loc[training],
            sample_weight=target.loc[training].clip(lower=0.10),
        )
        prediction = np.clip(
            model.predict(matrix.loc[application, selected]), 0.0, 1.075
        )
        normalized_prediction[validation_groups == group_id] = prediction
        actual = target.loc[application].to_numpy(dtype=float)
        diagnostics[str(group_id)] = {
            "training_rows": int(training.sum()),
            "selected_feature_names": selected,
            "selected_spectral_feature_count": int(
                sum(name.startswith("spectral__") for name in selected)
            ),
            "group_score": _group_total(actual, prediction),
        }
        print(
            json.dumps(
                {
                    "group_id": group_id,
                    "training_rows": int(training.sum()),
                    "selected_spectral_feature_count": diagnostics[str(group_id)][
                        "selected_spectral_feature_count"
                    ],
                    "group_score": diagnostics[str(group_id)]["group_score"],
                }
            ),
            flush=True,
        )

    output["prediction_kwh"] = (
        normalized_prediction
        * output["group_id"].map(CAPACITIES).to_numpy(dtype=float)
    )
    output["fold_id"] = args.fold
    output["model_id"] = args.candidate_id
    output_path = OUTPUT / f"{args.candidate_id}-{args.fold}.parquet"
    output.to_parquet(output_path, index=False)
    receipt = {
        "schema_version": 1,
        "candidate_id": args.candidate_id,
        "fold_id": args.fold,
        "architecture": "strict_grid_temporal_fourier_lowpass_lightgbm",
        "scope": "fixed representation screen; outer labels excluded from fitting",
        "research_source": RESEARCH_SOURCE,
        "grid_speed_feature_count": len(speed_columns),
        "spectral_feature_count": spectral.shape[1],
        "base_feature_count": len(base_features),
        "top_features": args.top_features,
        "iterations": args.iterations,
        "diagnostics": diagnostics,
        "fold_score": _score(output),
        "runtime_seconds": round(time.perf_counter() - started, 2),
        "prediction_path": str(output_path.relative_to(Path.cwd())),
        "prediction_sha256": _sha256(output_path),
        "new_2024_evaluation": False,
        "lockbox_reopened": False,
        "external_actions": [],
    }
    receipt_path = OUTPUT / f"{args.candidate_id}-{args.fold}.json"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
