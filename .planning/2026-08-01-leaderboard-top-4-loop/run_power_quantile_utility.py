"""Fit a compact conditional power distribution and optimize settlement utility."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
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

QUANTILES = (0.10, 0.30, 0.50, 0.70, 0.90)
ITERATIONS = (80, 120, 180, 240)
SPREAD_SCALES = (0.75, 1.0, 1.25)
GAMMAS = (0.0, 0.2, 0.35, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)


def _decision_policies(
    base: pd.DataFrame,
    samples: np.ndarray,
    mean_generation: dict[int, float],
) -> tuple[pd.DataFrame, str, dict[str, dict[str, float]], pd.DataFrame]:
    actions = np.arange(0.075, 1.076, 0.0025)
    median = samples[:, len(QUANTILES) // 2]
    predictions: dict[str, np.ndarray] = {"MEDIAN": median}
    for spread_scale in SPREAD_SCALES:
        distribution = np.clip(
            median[:, None] + spread_scale * (samples - median[:, None]),
            0.075,
            1.075,
        )
        for gamma in GAMMAS:
            chosen = np.empty(len(base), dtype=float)
            for group_id in CAPACITIES:
                mask = base["group_id"].eq(group_id).to_numpy()
                group_samples = distribution[mask]
                error = np.abs(
                    actions[None, :, None] - group_samples[:, None, :]
                )
                units = np.select(
                    [error <= 0.06, error <= 0.08], [4.0, 3.0], default=0.0
                )
                utility = -error.mean(axis=2) + gamma * (
                    group_samples[:, None, :] * units
                ).mean(axis=2) / (4.0 * mean_generation[group_id])
                chosen[mask] = actions[np.argmax(utility, axis=1)]
            tag = f"S{spread_scale:g}_G{gamma:g}"
            predictions[tag] = chosen
    scores: dict[str, dict[str, float]] = {}
    policy_frame = base[
        ["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh"]
    ].copy()
    capacity = base["group_id"].map(CAPACITIES).to_numpy(dtype=float)
    for tag, normalized in predictions.items():
        candidate = base[
            ["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh"]
        ].copy()
        candidate["prediction_kwh"] = normalized * capacity
        scores[tag] = _score(candidate)
        policy_frame[tag] = normalized * capacity
    best_policy = max(scores, key=lambda name: scores[name]["total"])
    output = base[
        ["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh"]
    ].copy()
    output["prediction_kwh"] = predictions[best_policy] * capacity
    return output, best_policy, scores, policy_frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--fold", choices=FOLDS, default=FOLDS[-1])
    parser.add_argument("--sitewind-cache-id", required=True)
    parser.add_argument("--num-leaves", type=int, choices=(15, 31, 63), default=31)
    parser.add_argument(
        "--iterations", nargs="+", type=int, default=list(ITERATIONS)
    )
    args = parser.parse_args()
    iterations = tuple(sorted(set(args.iterations)))
    if not iterations or iterations[0] < 1 or iterations[-1] > 400:
        raise ValueError("iterations must be unique positive values no greater than 400")
    if _sha256(OPEN) != OPEN_SHA or _sha256(BASELINE) != BASELINE_SHA:
        raise RuntimeError("immutable competition input hash mismatch")
    started = time.perf_counter()
    surface, base_columns, _ = _surface()
    validation = _validation_mask(surface, args.fold)
    start = surface.loc[validation, "forecast_kst_dtm"].min()
    preceding = surface["forecast_kst_dtm"].lt(start).to_numpy()
    normalized_target = surface["actual_kwh"] / surface["group_id"].map(CAPACITIES)
    training = (
        preceding
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
    base = surface.loc[
        validation, ["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh"]
    ].copy()
    prediction_by_iteration = {
        iteration: np.empty((int(validation.sum()), len(QUANTILES)), dtype=float)
        for iteration in iterations
    }
    params = {
        "objective": "quantile",
        "n_estimators": max(iterations),
        "learning_rate": 0.03,
        "num_leaves": args.num_leaves,
        "min_child_samples": 80,
        "subsample": 0.9,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 2.0,
        "random_state": 20260802,
        "n_jobs": 6,
        "deterministic": True,
        "force_col_wise": True,
        "verbosity": -1,
    }
    sample_weight = normalized_target.loc[training].clip(lower=0.10)
    for column_index, quantile in enumerate(QUANTILES):
        model = LGBMRegressor(**{**params, "alpha": quantile})
        model.fit(
            matrix.loc[training],
            normalized_target.loc[training],
            sample_weight=sample_weight,
        )
        for iteration in iterations:
            prediction_by_iteration[iteration][:, column_index] = model.predict(
                matrix.loc[validation], num_iteration=iteration
            )
        print(json.dumps({"quantile": quantile, "status": "fit"}), flush=True)
    mean_generation = {
        group_id: float(
            normalized_target.loc[training & surface["group_id"].eq(group_id).to_numpy()].mean()
        )
        for group_id in CAPACITIES
    }
    sweep: dict[str, object] = {}
    best: tuple[float, int, str, pd.DataFrame, pd.DataFrame] | None = None
    for iteration in iterations:
        samples = np.sort(prediction_by_iteration[iteration], axis=1)
        samples = np.clip(samples, 0.075, 1.075)
        output, policy, scores, policies = _decision_policies(
            base, samples, mean_generation
        )
        score = scores[policy]
        sweep[str(iteration)] = {
            "best_policy": policy,
            "best_score": score,
            "scores": scores,
        }
        choice = (score["total"], iteration, policy, output, policies)
        if best is None or choice[0] > best[0]:
            best = choice
        print(
            json.dumps({"iteration": iteration, "policy": policy, "score": score}),
            flush=True,
        )
    assert best is not None
    output = best[3]
    policies = best[4]
    output["fold_id"] = args.fold
    output["model_id"] = args.candidate_id
    output_path = OUTPUT / f"{args.candidate_id}-{args.fold}.parquet"
    policy_path = OUTPUT / f"{args.candidate_id}-{args.fold}-policies.parquet"
    output.to_parquet(output_path, index=False)
    policies.to_parquet(policy_path, index=False)
    receipt = {
        "schema_version": 1,
        "candidate_id": args.candidate_id,
        "fold_id": args.fold,
        "architecture": "conditional_power_quantiles",
        "quantiles": list(QUANTILES),
        "feature_count": matrix.shape[1],
        "sitewind_feature_count": len(sitewind_columns),
        "num_leaves": args.num_leaves,
        "selected_iteration": best[1],
        "best_policy": best[2],
        "sweep": sweep,
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
