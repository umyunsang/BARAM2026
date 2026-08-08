"""Chronology-safe analog residual distributions for settlement decisions."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
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
)
from sklearn.neighbors import NearestNeighbors

FOLDS = ("dev-2023-Q2", "dev-2023-Q3", "dev-2023-Q4")
QUANTILES = np.linspace(0.025, 0.975, 21)
OFFSETS = np.arange(-0.10, 0.1001, 0.0025)


@dataclass(frozen=True)
class AnalogPolicy:
    profile: str
    neighbors: int
    residual_mode: bool
    gamma: float


PROFILES = {
    "level": ("p_self",),
    "level_time": ("p_self", "hour_sin", "hour_cos"),
    "cross": ("p_self", "p1", "p2", "p3", "p12_mean", "p12_diff"),
    "all": (
        "p_self",
        "p1",
        "p2",
        "p3",
        "p12_mean",
        "p12_diff",
        "hour_sin",
        "hour_cos",
        "doy_sin",
        "doy_cos",
    ),
}


def _surface(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    capacity = output["group_id"].map(CAPACITIES).astype(float)
    output["p_self"] = output["prediction_kwh"] / capacity
    output["target"] = output["actual_kwh"] / capacity
    pivot = (
        output.pivot(
            index="forecast_kst_dtm", columns="group_id", values="p_self"
        )
        .rename(columns={1: "p1", 2: "p2", 3: "p3"})
        .reset_index()
    )
    output = output.merge(
        pivot, on="forecast_kst_dtm", how="left", validate="many_to_one"
    )
    output["p12_mean"] = (output["p1"] + output["p2"]) / 2.0
    output["p12_diff"] = output["p2"] - output["p1"]
    hour = output["forecast_kst_dtm"].dt.hour
    day = output["forecast_kst_dtm"].dt.dayofyear
    output["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    output["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    output["doy_sin"] = np.sin(2.0 * np.pi * day / 365.25)
    output["doy_cos"] = np.cos(2.0 * np.pi * day / 365.25)
    return output.sort_values(["forecast_kst_dtm", "group_id"]).reset_index(drop=True)


def _analog_distribution(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    profile: str,
    neighbors: int,
    residual_mode: bool,
) -> np.ndarray:
    columns = list(PROFILES[profile])
    distribution = np.empty((len(valid), len(QUANTILES)), dtype="float32")
    for group_id in CAPACITIES:
        fit = train["group_id"].eq(group_id) & train["target"].ge(0.10)
        apply = valid["group_id"].eq(group_id)
        x_train = train.loc[fit, columns].to_numpy(dtype=float)
        x_valid = valid.loc[apply, columns].to_numpy(dtype=float)
        center = x_train.mean(axis=0)
        scale = x_train.std(axis=0)
        scale[scale < 1e-6] = 1.0
        x_train = (x_train - center) / scale
        x_valid = (x_valid - center) / scale
        count = min(neighbors, len(x_train))
        model = NearestNeighbors(n_neighbors=count, algorithm="auto", n_jobs=6)
        model.fit(x_train)
        indices = model.kneighbors(x_valid, return_distance=False)
        target = train.loc[fit, "target"].to_numpy(dtype=float)
        samples = target[indices]
        if residual_mode:
            source = train.loc[fit, "p_self"].to_numpy(dtype=float)
            samples = samples - source[indices]
            samples += valid.loc[apply, "p_self"].to_numpy(dtype=float)[:, None]
        quantiles = np.quantile(samples, QUANTILES, axis=1).T
        distribution[apply.to_numpy()] = np.clip(quantiles, 0.0, 1.075)
    return distribution


def _decision(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    distribution: np.ndarray,
    gamma: float,
) -> np.ndarray:
    base = valid["p_self"].to_numpy(dtype=float)
    actions = np.clip(base[:, None] + OFFSETS[None, :], 0.0, 1.075)
    prediction = np.empty(len(valid), dtype=float)
    for group_id in CAPACITIES:
        apply = valid["group_id"].eq(group_id).to_numpy()
        sample = distribution[apply]
        candidate = actions[apply]
        error = np.abs(candidate[:, :, None] - sample[:, None, :])
        unit = np.select(
            [error <= 0.06, error <= 0.08], [4.0, 3.0], default=0.0
        )
        mean_generation = float(
            train.loc[
                train["group_id"].eq(group_id) & train["target"].ge(0.10),
                "target",
            ].mean()
        )
        utility = -error.mean(axis=2) + gamma * (
            (sample[:, None, :] * unit).mean(axis=2) / (4.0 * mean_generation)
        )
        prediction[apply] = candidate[np.arange(int(apply.sum())), utility.argmax(axis=1)]
    return prediction


def _prediction_frame(frame: pd.DataFrame, normalized: np.ndarray) -> pd.DataFrame:
    output = frame[
        ["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh"]
    ].copy()
    output["prediction_kwh"] = (
        normalized * output["group_id"].map(CAPACITIES).to_numpy(dtype=float)
    )
    return output


def _select(
    train: pd.DataFrame,
    calibration: pd.DataFrame,
) -> tuple[AnalogPolicy | None, dict[str, object]]:
    base_score = _score(
        _prediction_frame(
            calibration, calibration["p_self"].to_numpy(dtype=float)
        )
    )
    diagnostics: dict[str, object] = {"identity": base_score}
    best: tuple[float, AnalogPolicy | None] = (base_score["total"], None)
    for profile in PROFILES:
        for neighbors in (50, 100, 200, 400):
            for residual_mode in (False, True):
                distribution = _analog_distribution(
                    train, calibration, profile, neighbors, residual_mode
                )
                config_best: tuple[float, float, dict[str, float]] | None = None
                for gamma in (0.0, 0.35, 0.6, 0.85, 1.0, 1.25, 1.5, 2.0):
                    prediction = _decision(
                        train, calibration, distribution, gamma
                    )
                    score = _score(_prediction_frame(calibration, prediction))
                    choice = (score["total"], gamma, score)
                    if config_best is None or choice[0] > config_best[0]:
                        config_best = choice
                assert config_best is not None
                key = f"{profile}|k={neighbors}|residual={residual_mode}"
                diagnostics[key] = {
                    "gamma": config_best[1],
                    "score": config_best[2],
                }
                policy = AnalogPolicy(
                    profile, neighbors, residual_mode, config_best[1]
                )
                if config_best[0] > best[0]:
                    best = (config_best[0], policy)
    return best[1], diagnostics


def _stage(
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    application: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, object]]:
    policy, diagnostics = _select(train, calibration)
    if policy is None:
        prediction = application["p_self"].to_numpy(dtype=float)
    else:
        combined = pd.concat([train, calibration], ignore_index=True)
        distribution = _analog_distribution(
            combined,
            application,
            policy.profile,
            policy.neighbors,
            policy.residual_mode,
        )
        prediction = _decision(
            combined, application, distribution, policy.gamma
        )
    return prediction, {
        "selected_policy": None if policy is None else policy.__dict__,
        "calibration_diagnostics": diagnostics,
        "application_score": _score(_prediction_frame(application, prediction)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-id", default="M106_KNN_RESIDUAL_UTILITY")
    args = parser.parse_args()
    if _sha256(OPEN) != OPEN_SHA or _sha256(BASELINE) != BASELINE_SHA:
        raise RuntimeError("immutable competition input hash mismatch")
    started = time.perf_counter()
    parent = pd.read_parquet(OUTPUT / "M103_STRICT_TOP100-oof.parquet")
    surface = _surface(parent)
    quarter = {
        fold: surface.loc[surface["fold_id"].eq(fold)].copy() for fold in FOLDS
    }
    q2_times = quarter[FOLDS[0]]["forecast_kst_dtm"].drop_duplicates().sort_values()
    cutoff = q2_times.iloc[int(len(q2_times) * 0.60)]
    q2_early = quarter[FOLDS[0]].loc[
        quarter[FOLDS[0]]["forecast_kst_dtm"].lt(cutoff)
    ]
    q2_late = quarter[FOLDS[0]].loc[
        quarter[FOLDS[0]]["forecast_kst_dtm"].ge(cutoff)
    ]
    q3_prediction, q3_receipt = _stage(
        q2_early, q2_late, quarter[FOLDS[1]]
    )
    q4_prediction, q4_receipt = _stage(
        quarter[FOLDS[0]], quarter[FOLDS[1]], quarter[FOLDS[2]]
    )
    outputs = [
        parent.loc[parent["fold_id"].eq(FOLDS[0])].copy(),
        _prediction_frame(quarter[FOLDS[1]], q3_prediction).assign(
            fold_id=FOLDS[1]
        ),
        _prediction_frame(quarter[FOLDS[2]], q4_prediction).assign(
            fold_id=FOLDS[2]
        ),
    ]
    output = pd.concat(outputs, ignore_index=True)
    output["model_id"] = args.candidate_id
    output_path = OUTPUT / f"{args.candidate_id}-oof.parquet"
    output.to_parquet(output_path, index=False)
    receipt = {
        "schema_version": 1,
        "candidate_id": args.candidate_id,
        "scope": "Q2 control; Q3 selected within Q2; Q4 selected on Q3",
        "parent_candidate_id": "M103_STRICT_TOP100",
        "q3_stage": q3_receipt,
        "q4_stage": q4_receipt,
        "fold_scores": {
            fold: _score(output.loc[output["fold_id"].eq(fold)]) for fold in FOLDS
        },
        "pooled": _score(output),
        "runtime_seconds": round(time.perf_counter() - started, 2),
        "prediction_path": str(output_path.relative_to(Path.cwd())),
        "prediction_sha256": _sha256(output_path),
        "observed_scada_feature_count": 0,
        "new_2024_evaluation": False,
        "lockbox_reopened": False,
        "external_actions": [],
    }
    receipt_path = OUTPUT / f"{args.candidate_id}-oof.json"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
