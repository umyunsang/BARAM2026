"""Model ordered generation bins with cumulative binary LightGBM classifiers."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from run_sequence_classifier import (
    BASELINE,
    BASELINE_SHA,
    CAPACITIES,
    OPEN,
    OPEN_SHA,
    OUTPUT,
    _sha256,
    _surface,
)
from run_site_wind_classifier import (
    FOLDS,
    _add_site_wind_features,
    _choose_actions,
)
from run_site_wind_teacher import _validation_mask

ITERATIONS = (40, 60, 80, 120)


def _ordered_probability(probability_gt: np.ndarray) -> np.ndarray:
    """Turn possibly crossing P(Y > threshold) curves into ordered bin masses."""
    monotone = np.sort(probability_gt, axis=1)[:, ::-1]
    probability = np.empty(
        (len(monotone), monotone.shape[1] + 1), dtype="float64"
    )
    probability[:, 0] = 1.0 - monotone[:, 0]
    probability[:, 1:-1] = monotone[:, :-1] - monotone[:, 1:]
    probability[:, -1] = monotone[:, -1]
    probability = np.clip(probability, 1e-12, None)
    probability /= probability.sum(axis=1, keepdims=True)
    return probability


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--fold", choices=FOLDS, default=FOLDS[-1])
    parser.add_argument("--sitewind-cache-id", required=True)
    parser.add_argument("--num-leaves", type=int, choices=(7, 15, 31), default=15)
    parser.add_argument("--class-width", type=float, default=0.02)
    parser.add_argument("--generation-weight-power", type=float, default=1.0)
    parser.add_argument("--top-features", type=int, default=0)
    parser.add_argument("--iterations", nargs="+", type=int, default=list(ITERATIONS))
    args = parser.parse_args()
    iterations = tuple(sorted(set(args.iterations)))
    if not iterations or iterations[0] < 1 or iterations[-1] > 240:
        raise ValueError("iterations must be unique positive values no greater than 240")
    if not 0.01 <= args.class_width <= 0.10:
        raise ValueError("class-width must be between 0.01 and 0.10")
    if not 0.0 <= args.generation_weight_power <= 2.0:
        raise ValueError("generation-weight-power must be between zero and two")
    if args.top_features < 0:
        raise ValueError("top-features must be nonnegative")
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
    width = args.class_width
    raw_bins = np.floor(
        (normalized_target.clip(0.10, 1.074999) - 0.10) / width
    ).astype("Int64")
    active_bins = [
        int(value)
        for value in sorted(raw_bins.loc[training].dropna().astype(int).unique())
    ]
    bin_to_class = {bin_id: index for index, bin_id in enumerate(active_bins)}
    classes = raw_bins.map(bin_to_class).astype("Int64")
    class_count = len(active_bins)
    centers = np.asarray(
        [
            normalized_target.loc[training & classes.eq(class_id)].mean()
            for class_id in range(class_count)
        ],
        dtype=float,
    )
    validation_count = int(validation.sum())
    cumulative = {
        iteration: np.empty((validation_count, class_count - 1), dtype="float32")
        for iteration in iterations
    }
    params = {
        "objective": "binary",
        "n_estimators": max(iterations),
        "learning_rate": 0.025,
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
    sample_weight = normalized_target.loc[training].clip(lower=0.10) ** (
        args.generation_weight_power
    )
    training_class = classes.loc[training].astype(int)
    selected_feature_names = list(matrix.columns)
    if 0 < args.top_features < matrix.shape[1]:
        screening_params = {
            **params,
            "objective": "multiclass",
            "num_class": class_count,
            "n_estimators": min(max(iterations), 60),
            "random_state": 20260803,
        }
        screening = LGBMClassifier(**screening_params)
        screening.fit(
            matrix.loc[training],
            training_class,
            sample_weight=sample_weight,
        )
        gain = screening.booster_.feature_importance(importance_type="gain")
        selected_positions = np.argsort(gain)[::-1][: args.top_features]
        selected_feature_names = [matrix.columns[index] for index in selected_positions]
        matrix = matrix[selected_feature_names]
    for boundary in range(class_count - 1):
        model = LGBMClassifier(**params)
        model.fit(
            matrix.loc[training],
            training_class.gt(boundary).astype(int),
            sample_weight=sample_weight,
        )
        for iteration in iterations:
            cumulative[iteration][:, boundary] = model.predict_proba(
                matrix.loc[validation], num_iteration=iteration
            )[:, 1]
        if boundary % 8 == 0 or boundary == class_count - 2:
            print(
                json.dumps(
                    {
                        "boundary": boundary + 1,
                        "boundary_count": class_count - 1,
                        "status": "fit",
                    }
                ),
                flush=True,
            )
    base = surface.loc[
        validation, ["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh"]
    ].copy()
    sweep: dict[str, object] = {}
    best: tuple[float, int, str, object] | None = None
    policy_columns: dict[str, np.ndarray] = {}
    for iteration in iterations:
        probability = _ordered_probability(cumulative[iteration])
        output, best_policy, scores, policies = _choose_actions(
            base,
            probability,
            centers,
            normalized_target,
            training,
            surface["group_id"],
        )
        score = scores[best_policy]
        sweep[str(iteration)] = {
            "best_policy": best_policy,
            "best_score": score,
            "scores": scores,
        }
        policy_columns.update(
            {
                f"I{iteration}__{column}": policies[column].to_numpy(dtype=float)
                for column in policies.columns
                if column not in base.columns
            }
        )
        choice = (score["total"], iteration, best_policy, (output, policies))
        if best is None or choice[0] > best[0]:
            best = choice
        print(
            json.dumps(
                {"iteration": iteration, "policy": best_policy, "score": score}
            ),
            flush=True,
        )
    assert best is not None
    output, _ = best[3]
    output["fold_id"] = args.fold
    output["model_id"] = args.candidate_id
    output_path = OUTPUT / f"{args.candidate_id}-{args.fold}.parquet"
    policy_path = OUTPUT / f"{args.candidate_id}-{args.fold}-policies.parquet"
    output.to_parquet(output_path, index=False)
    policy_frame = pd.concat(
        [base.reset_index(drop=True), pd.DataFrame(policy_columns)], axis=1
    )
    policy_frame.to_parquet(policy_path, index=False)
    receipt = {
        "schema_version": 1,
        "candidate_id": args.candidate_id,
        "fold_id": args.fold,
        "architecture": "cumulative_ordinal_binary",
        "feature_count": matrix.shape[1],
        "top_features": args.top_features,
        "selected_feature_names": selected_feature_names,
        "sitewind_feature_count": len(sitewind_columns),
        "num_leaves": args.num_leaves,
        "class_width": width,
        "class_count": class_count,
        "active_bins": active_bins,
        "generation_weight_power": args.generation_weight_power,
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
