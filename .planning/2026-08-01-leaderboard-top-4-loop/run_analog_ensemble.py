"""Screen a supplied-NWP analog ensemble with metric-aware neighbor decisions."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
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
from sklearn.neighbors import NearestNeighbors

BASE_COLUMNS = ["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh"]
NEIGHBOR_COUNTS = (20, 40, 80, 160)
KERNELS = ("uniform", "inverse", "exponential")
GAMMAS = (0.0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)


def _feature_names(surface: pd.DataFrame) -> list[str]:
    preferred = [
        "sitewind__legacy",
        "sitewind__allweather",
        "sitewind__mean",
        "sitewind__delta",
        "sitewind__disagreement",
        "gfs_spatial__idw__wind10_speed",
        "gfs_spatial__idw__wind10_dir_sin",
        "gfs_spatial__idw__wind10_dir_cos",
        "gfs_spatial__idw__wind80_speed",
        "gfs_spatial__idw__wind80_dir_sin",
        "gfs_spatial__idw__wind80_dir_cos",
        "gfs_spatial__idw__wind100_speed",
        "gfs_spatial__idw__wind100_dir_sin",
        "gfs_spatial__idw__wind100_dir_cos",
        "ldaps_spatial__idw__wind5_speed",
        "ldaps_spatial__idw__wind5_dir_sin",
        "ldaps_spatial__idw__wind5_dir_cos",
        "ldaps_spatial__idw__wind10_speed",
        "ldaps_spatial__idw__wind10_dir_sin",
        "ldaps_spatial__idw__wind10_dir_cos",
        "ldaps_spatial__idw__wind50max_speed",
        "ldaps_spatial__idw__wind50max_dir_sin",
        "ldaps_spatial__idw__wind50max_dir_cos",
        "ldaps_spatial__idw__wind50min_speed",
        "ldaps_spatial__idw__wind50min_dir_sin",
        "ldaps_spatial__idw__wind50min_dir_cos",
        "source_disagreement__wind10_speed_idw",
        "source_disagreement__wind10_speed_idw__abs",
        "phys__hub117_speed",
        "phys__speed_shear_100_80",
        "phys__air_density",
        "phys_v2__shear_alpha_100_80",
        "phys_v2__hub117_speed",
        "phys_v2__air_density",
        "geom__gfs__wind100__vector_spread",
        "geom__gfs__wind100__gradient_norm",
        "geom__ldaps__wind50max__vector_spread",
        "geom__ldaps__wind50max__gradient_norm",
        "geom__ldaps__wind50min__vector_spread",
        "geom__ldaps__wind50min__gradient_norm",
        "lead_hour",
        "cal__hour_sin",
        "cal__hour_cos",
        "cal__doy_sin",
        "cal__doy_cos",
    ]
    selected = [name for name in preferred if name in surface]
    if len(selected) < 35:
        raise RuntimeError(f"analog feature contract resolved only {len(selected)} columns")
    return selected


def _robust_transform(
    train: pd.DataFrame,
    valid: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    median = train.median(axis=0)
    filled_train = train.fillna(median)
    filled_valid = valid.fillna(median)
    lower = filled_train.quantile(0.10)
    upper = filled_train.quantile(0.90)
    scale = (upper - lower).mask(lambda values: values.abs() < 1e-6, 1.0)
    return (
        np.clip(((filled_train - median) / scale).to_numpy(dtype="float32"), -8, 8),
        np.clip(((filled_valid - median) / scale).to_numpy(dtype="float32"), -8, 8),
    )


def _weights(distances: np.ndarray, kernel: str) -> np.ndarray:
    if kernel == "uniform":
        raw = np.ones_like(distances)
    elif kernel == "inverse":
        raw = 1.0 / np.maximum(distances, 1e-3)
    else:
        bandwidth = np.maximum(np.median(distances, axis=1, keepdims=True), 1e-3)
        raw = np.exp(-distances / bandwidth)
    return raw / raw.sum(axis=1, keepdims=True)


def _decide_all(
    neighbor_targets: np.ndarray,
    weights: np.ndarray,
    mean_generation: float,
) -> dict[float, np.ndarray]:
    actions = np.arange(0.075, 1.076, 0.005)
    predictions = {
        gamma: np.empty(len(neighbor_targets), dtype=float) for gamma in GAMMAS
    }
    for row in range(len(neighbor_targets)):
        target = neighbor_targets[row]
        weight = weights[row]
        error = np.abs(actions[:, None] - target[None, :])
        units = np.select([error <= 0.06, error <= 0.08], [4.0, 3.0], default=0.0)
        expected_error = error @ weight
        expected_settlement = (units * target[None, :]) @ weight
        for gamma in GAMMAS:
            utility = (
                -expected_error
                + gamma * expected_settlement / (4.0 * mean_generation)
            )
            predictions[gamma][row] = actions[int(np.argmax(utility))]
    return predictions


def _group_score(
    actual_kwh: np.ndarray,
    normalized_prediction: np.ndarray,
    group_id: int,
) -> dict[str, float]:
    capacity = CAPACITIES[group_id]
    actual = actual_kwh / capacity
    valid = np.isfinite(actual) & (actual >= 0.10)
    actual = actual[valid]
    prediction = normalized_prediction[valid]
    error = np.abs(actual - prediction)
    units = np.select([error <= 0.06, error <= 0.08], [4.0, 3.0], default=0.0)
    one_minus_nmae = 1.0 - float(error.mean())
    ficr = float(np.sum(actual * units) / (4.0 * np.sum(actual)))
    return {
        "total": 0.5 * (one_minus_nmae + ficr),
        "one_minus_nmae": one_minus_nmae,
        "ficr": ficr,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--fold", choices=FOLDS, default=FOLDS[-1])
    parser.add_argument("--sitewind-cache-id", required=True)
    args = parser.parse_args()
    if _sha256(OPEN) != OPEN_SHA or _sha256(BASELINE) != BASELINE_SHA:
        raise RuntimeError("immutable competition input hash mismatch")
    started = time.perf_counter()
    surface, base_columns, _ = _surface()
    validation = _validation_mask(surface, args.fold)
    start = surface.loc[validation, "forecast_kst_dtm"].min()
    normalized_target = surface["actual_kwh"] / surface["group_id"].map(CAPACITIES)
    training = (
        surface["forecast_kst_dtm"].lt(start).to_numpy()
        & surface["actual_kwh"].notna().to_numpy()
        & normalized_target.ge(0.10).to_numpy()
    )
    cached = np.load(
        OUTPUT / f"{args.sitewind_cache_id}-{args.fold}-sitewind-features.npz"
    )
    matrix = surface[base_columns].astype("float32")
    sitewind_columns = _add_site_wind_features(
        matrix, cached["legacy"], cached["allweather"]
    )
    selected_features = _feature_names(matrix)
    matrix = matrix[selected_features]
    base = surface.loc[validation, BASE_COLUMNS].copy()
    normalized_predictions = np.empty(len(base), dtype=float)
    group_receipts: dict[str, object] = {}
    for group_id in CAPACITIES:
        group_training = training & surface["group_id"].eq(group_id).to_numpy()
        group_validation = validation & surface["group_id"].eq(group_id).to_numpy()
        train_matrix, valid_matrix = _robust_transform(
            matrix.loc[group_training], matrix.loc[group_validation]
        )
        target = normalized_target.loc[group_training].to_numpy(dtype=float)
        maximum_neighbors = min(max(NEIGHBOR_COUNTS), len(target))
        neighbors = NearestNeighbors(
            n_neighbors=maximum_neighbors,
            algorithm="auto",
            metric="euclidean",
            n_jobs=6,
        )
        neighbors.fit(train_matrix)
        distances, indices = neighbors.kneighbors(valid_matrix)
        neighbor_targets = target[indices]
        actual = surface.loc[group_validation, "actual_kwh"].to_numpy(dtype=float)
        best: tuple[float, np.ndarray, int, str, float, dict[str, float]] | None = None
        sweep: dict[str, object] = {}
        for neighbor_count in NEIGHBOR_COUNTS:
            if neighbor_count > maximum_neighbors:
                continue
            count_scores: dict[str, object] = {}
            for kernel in KERNELS:
                weight = _weights(distances[:, :neighbor_count], kernel)
                kernel_scores: dict[str, dict[str, float]] = {}
                predictions = _decide_all(
                    neighbor_targets[:, :neighbor_count],
                    weight,
                    float(target.mean()),
                )
                for gamma in GAMMAS:
                    prediction = predictions[gamma]
                    score = _group_score(actual, prediction, group_id)
                    kernel_scores[f"G{gamma:g}"] = score
                    choice = (
                        score["total"],
                        prediction,
                        neighbor_count,
                        kernel,
                        gamma,
                        score,
                    )
                    if best is None or choice[0] > best[0]:
                        best = choice
                count_scores[kernel] = kernel_scores
            sweep[str(neighbor_count)] = count_scores
        assert best is not None
        positions = np.flatnonzero(base["group_id"].eq(group_id).to_numpy())
        normalized_predictions[positions] = best[1]
        group_receipts[str(group_id)] = {
            "training_rows": int(group_training.sum()),
            "selected_neighbor_count": best[2],
            "selected_kernel": best[3],
            "selected_gamma": best[4],
            "selected_score": best[5],
            "sweep": sweep,
        }
        print(
            json.dumps(
                {
                    "group_id": group_id,
                    "selected_neighbor_count": best[2],
                    "selected_kernel": best[3],
                    "selected_gamma": best[4],
                    "selected_score": best[5],
                }
            ),
            flush=True,
        )
    output = base.copy()
    output["prediction_kwh"] = (
        normalized_predictions
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
        "architecture": "metric_aware_nwp_analog_ensemble",
        "scope": "same-fold representation screen; group settings are oracle diagnostics",
        "feature_count": len(selected_features),
        "selected_feature_names": selected_features,
        "sitewind_feature_count": len(sitewind_columns),
        "group_receipts": group_receipts,
        "fold_score": _score(output),
        "runtime_seconds": round(time.perf_counter() - started, 2),
        "prediction_path": str(output_path.relative_to(Path.cwd())),
        "prediction_sha256": _sha256(output_path),
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
