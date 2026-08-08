"""Sweep independent group classifiers against the exact official metric."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
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
from xgboost import XGBClassifier

ITERATIONS = (40, 60, 80, 100)


def _distribution(
    normalized_target: pd.Series,
    training: np.ndarray,
    width: float,
) -> tuple[pd.Series, np.ndarray, list[int]]:
    raw_bins = np.floor(
        (normalized_target.clip(0.10, 1.074999) - 0.10) / width
    ).astype("Int64")
    active_bins = [
        int(value)
        for value in sorted(raw_bins.loc[training].dropna().astype(int).unique())
    ]
    bin_to_class = {bin_id: class_id for class_id, bin_id in enumerate(active_bins)}
    classes = raw_bins.map(bin_to_class).astype("Int64")
    centers = np.asarray(
        [
            normalized_target.loc[training & classes.eq(class_id)].mean()
            if (training & classes.eq(class_id)).any()
            else 0.10 + (bin_id + 0.5) * width
            for class_id, bin_id in enumerate(active_bins)
        ]
    )
    return classes, centers, active_bins


def _choose_actions(
    base: pd.DataFrame,
    probabilities: dict[int, np.ndarray],
    centers: dict[int, np.ndarray],
    mean_generation: dict[int, float],
) -> tuple[pd.DataFrame, str, dict[str, dict[str, float]], pd.DataFrame]:
    actions = np.arange(0.075, 1.076, 0.0025)
    results: dict[str, dict[str, float]] = {}
    predictions: dict[str, np.ndarray] = {}
    for temperature in (1.0, 0.75, 0.6):
        for gamma in (0.35, 0.5, 0.75, 1.0):
            chosen = np.empty(len(base), dtype=float)
            for group_id in CAPACITIES:
                mask = base["group_id"].eq(group_id).to_numpy()
                probability = probabilities[group_id] ** (1.0 / temperature)
                probability /= probability.sum(axis=1, keepdims=True)
                group_centers = centers[group_id]
                error = np.abs(actions[:, None] - group_centers[None, :])
                units = np.select(
                    [error <= 0.06, error <= 0.08], [4.0, 3.0], default=0.0
                )
                utility = -(probability @ error.T) + gamma * (
                    probability @ (group_centers[None, :] * units).T
                ) / (4.0 * mean_generation[group_id])
                chosen[mask] = actions[np.argmax(utility, axis=1)]
            candidate = base[
                ["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh"]
            ].copy()
            candidate["prediction_kwh"] = (
                chosen * candidate["group_id"].map(CAPACITIES).to_numpy(dtype=float)
            )
            tag = f"T{temperature:g}_G{gamma:g}"
            results[tag] = _score(candidate)
            predictions[tag] = chosen
            print(json.dumps({"policy": tag, "score": results[tag]}), flush=True)
    best_policy = max(results, key=lambda name: results[name]["total"])
    output = base[
        ["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh"]
    ].copy()
    capacity = output["group_id"].map(CAPACITIES).to_numpy(dtype=float)
    output["prediction_kwh"] = predictions[best_policy] * capacity
    policy_frame = base[
        ["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh"]
    ].copy()
    for tag, normalized in predictions.items():
        policy_frame[tag] = normalized * capacity
    return output, best_policy, results, policy_frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--fold", choices=FOLDS, default=FOLDS[-1])
    parser.add_argument("--sitewind-cache-id", required=True)
    parser.add_argument("--class-width", type=float, default=0.02)
    parser.add_argument("--num-leaves", type=int, choices=(7, 15, 31), default=15)
    parser.add_argument("--family", choices=("lgbm", "xgboost"), default="lgbm")
    parser.add_argument("--top-features", type=int, default=0)
    parser.add_argument("--iterations", nargs="+", type=int, default=list(ITERATIONS))
    args = parser.parse_args()
    iterations = tuple(sorted(set(args.iterations)))
    if not iterations or iterations[0] < 1 or iterations[-1] > 400:
        raise ValueError("iterations must be unique positive values no greater than 400")
    if not 0.005 <= args.class_width <= 0.10:
        raise ValueError("class-width must be between 0.005 and 0.10")
    if args.top_features < 0 or args.top_features > 629:
        raise ValueError("top-features must be between zero and 629")
    if _sha256(OPEN) != OPEN_SHA or _sha256(BASELINE) != BASELINE_SHA:
        raise RuntimeError("immutable competition input hash mismatch")
    started = time.perf_counter()
    surface, base_columns, _ = _surface()
    validation = _validation_mask(surface, args.fold)
    start = surface.loc[validation, "forecast_kst_dtm"].min()
    preceding = surface["forecast_kst_dtm"].lt(start).to_numpy()
    normalized_target = surface["actual_kwh"] / surface["group_id"].map(CAPACITIES)
    cached = np.load(
        OUTPUT / f"{args.sitewind_cache_id}-{args.fold}-sitewind-features.npz"
    )
    matrix = surface[base_columns].astype("float32")
    sitewind_columns = _add_site_wind_features(
        matrix, cached["legacy"], cached["allweather"]
    )
    selected_feature_names = list(matrix.columns)
    if args.top_features:
        receipt = json.loads(
            (OUTPUT / f"M102_TOP100-{args.fold}.json").read_text()
        )
        selected_feature_names = receipt["selected_feature_names"][: args.top_features]
        missing = set(selected_feature_names).difference(matrix.columns)
        if missing:
            raise RuntimeError(f"missing selected group features: {sorted(missing)}")
        matrix = matrix[selected_feature_names]
    base = surface.loc[
        validation, ["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh"]
    ].copy()
    group_probabilities: dict[int, dict[int, np.ndarray]] = {
        iteration: {} for iteration in iterations
    }
    group_centers: dict[int, np.ndarray] = {}
    group_bins: dict[str, list[int]] = {}
    mean_generation: dict[int, float] = {}
    for group_id in CAPACITIES:
        group = surface["group_id"].eq(group_id).to_numpy()
        training = (
            preceding
            & group
            & surface["actual_kwh"].notna().to_numpy()
            & normalized_target.ge(0.10).to_numpy()
        )
        classes, centers, active_bins = _distribution(
            normalized_target, training, args.class_width
        )
        group_centers[group_id] = centers
        group_bins[str(group_id)] = active_bins
        mean_generation[group_id] = float(normalized_target.loc[training].mean())
        if args.family == "lgbm":
            classifier = LGBMClassifier(
                objective="multiclass",
                num_class=len(active_bins),
                n_estimators=max(iterations),
                learning_rate=0.025,
                num_leaves=args.num_leaves,
                min_child_samples=60,
                subsample=0.9,
                subsample_freq=1,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=2.0,
                random_state=20260801 + group_id,
                n_jobs=6,
                deterministic=True,
                force_col_wise=True,
                verbosity=-1,
            )
        else:
            classifier = XGBClassifier(
                objective="multi:softprob",
                num_class=len(active_bins),
                n_estimators=max(iterations),
                learning_rate=0.03,
                max_depth=5,
                min_child_weight=10.0,
                subsample=0.9,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=5.0,
                max_bin=256,
                tree_method="hist",
                random_state=20260801 + group_id,
                n_jobs=6,
            )
        classifier.fit(
            matrix.loc[training],
            classes.loc[training].astype(int),
            sample_weight=normalized_target.loc[training].clip(lower=0.10),
        )
        apply = validation & group
        for iteration in iterations:
            if args.family == "lgbm":
                probability = classifier.predict_proba(
                    matrix.loc[apply], num_iteration=iteration
                )
            else:
                probability = classifier.predict_proba(
                    matrix.loc[apply], iteration_range=(0, iteration)
                )
            group_probabilities[iteration][group_id] = probability
    parent = pd.read_parquet(OUTPUT / "M107_STRICT_TEMPORAL_TOP100-oof.parquet")
    parent = parent.loc[parent["fold_id"].eq(args.fold)]
    sweep: dict[str, object] = {}
    best: tuple[
        float,
        int,
        str,
        pd.DataFrame,
        pd.DataFrame,
        dict[str, object],
        dict[str, float],
    ] | None = None
    for iteration in iterations:
        output, best_policy, scores, policies = _choose_actions(
            base,
            group_probabilities[iteration],
            group_centers,
            mean_generation,
        )
        score = scores[best_policy]
        blended, selections = _screen_blends(base, policies, parent)
        blend_score = _score(blended)
        sweep[str(iteration)] = {
            "best_policy": best_policy,
            "best_score": score,
            "oracle_blend_score": blend_score,
            "oracle_blends": selections,
            "scores": scores,
        }
        choice = (
            blend_score["total"],
            iteration,
            best_policy,
            blended,
            policies,
            selections,
            score,
        )
        if best is None or choice[0] > best[0]:
            best = choice
        print(
            json.dumps(
                {
                    "iteration": iteration,
                    "policy": best_policy,
                    "score": score,
                    "oracle_blend_score": blend_score,
                }
            ),
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
        "architecture": f"independent_group_{args.family}_classifiers",
        "feature_count": matrix.shape[1],
        "sitewind_feature_count": len(sitewind_columns),
        "class_width": args.class_width,
        "active_bins": group_bins,
        "num_leaves": args.num_leaves,
        "family": args.family,
        "top_features": args.top_features,
        "selected_feature_names": selected_feature_names,
        "selected_iteration": best[1],
        "best_policy": best[2],
        "selected_raw_score": best[6],
        "selected_oracle_blends": best[5],
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
