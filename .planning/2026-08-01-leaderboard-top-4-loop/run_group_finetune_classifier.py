"""Screen shared-to-site LightGBM continuation for metric-aware classification."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import lightgbm
import numpy as np
from lightgbm import LGBMClassifier
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
from run_site_wind_classifier import (
    DECISION_GAMMAS,
    DECISION_TEMPERATURES,
    FOLDS,
    _add_site_wind_features,
)
from run_site_wind_teacher import _validation_mask

BASE_COLUMNS = ["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh"]
FINE_ITERATIONS = (5, 10, 20, 40, 60)
PROBABILITY_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)


def _feature_names(fold: str) -> list[str]:
    receipt = json.loads((OUTPUT / f"M102_TOP100-{fold}.json").read_text())
    names = receipt["selected_feature_names"]
    if len(names) != 100 or len(set(names)) != 100:
        raise RuntimeError("fixed M102 feature contract changed")
    return names


def _group_score(
    actual_kwh: np.ndarray,
    prediction: np.ndarray,
    group_id: int,
) -> dict[str, float]:
    capacity = CAPACITIES[group_id]
    actual = actual_kwh / capacity
    valid = np.isfinite(actual) & (actual >= 0.10)
    actual = actual[valid]
    prediction = prediction[valid]
    error = np.abs(actual - prediction)
    units = np.select([error <= 0.06, error <= 0.08], [4.0, 3.0], default=0.0)
    one_minus_nmae = 1.0 - float(error.mean())
    ficr = float(np.sum(actual * units) / (4.0 * np.sum(actual)))
    return {
        "total": 0.5 * (one_minus_nmae + ficr),
        "one_minus_nmae": one_minus_nmae,
        "ficr": ficr,
    }


def _choose_group_action(
    probability: np.ndarray,
    centers: np.ndarray,
    mean_generation: float,
    actual_kwh: np.ndarray,
    group_id: int,
) -> tuple[np.ndarray, str, dict[str, float]]:
    actions = np.arange(0.075, 1.076, 0.0025)
    error = np.abs(actions[:, None] - centers[None, :])
    units = np.select([error <= 0.06, error <= 0.08], [4.0, 3.0], default=0.0)
    best: tuple[float, np.ndarray, str, dict[str, float]] | None = None
    for temperature in DECISION_TEMPERATURES:
        calibrated = probability ** (1.0 / temperature)
        calibrated /= calibrated.sum(axis=1, keepdims=True)
        for gamma in DECISION_GAMMAS:
            utility = -(calibrated @ error.T) + gamma * (
                calibrated @ (centers[None, :] * units).T
            ) / (4.0 * mean_generation)
            chosen = actions[np.argmax(utility, axis=1)]
            score = _group_score(actual_kwh, chosen, group_id)
            tag = f"T{temperature:g}_G{gamma:g}"
            choice = (score["total"], chosen, tag, score)
            if best is None or choice[0] > best[0]:
                best = choice
    assert best is not None
    return best[1], best[2], best[3]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--fold", choices=FOLDS, default=FOLDS[-1])
    parser.add_argument("--sitewind-cache-id", required=True)
    parser.add_argument("--shared-iterations", type=int, default=60)
    parser.add_argument("--fine-learning-rate", type=float, default=0.01)
    args = parser.parse_args()
    if not 20 <= args.shared_iterations <= 200:
        raise ValueError("shared-iterations must be between twenty and two hundred")
    if not 0.002 <= args.fine_learning_rate <= 0.05:
        raise ValueError("fine-learning-rate must be between 0.002 and 0.05")
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
    raw_bins = np.floor(
        (normalized_target.clip(0.10, 1.074999) - 0.10) / 0.02
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
    selected_features = _feature_names(args.fold)
    matrix = matrix[selected_features]
    shared_params = {
        "objective": "multiclass",
        "num_class": len(active_bins),
        "n_estimators": args.shared_iterations,
        "learning_rate": 0.025,
        "num_leaves": 15,
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
    shared = LGBMClassifier(**shared_params)
    shared.fit(
        matrix.loc[training],
        classes.loc[training].astype(int),
        sample_weight=normalized_target.loc[training].clip(lower=0.10),
    )
    base = surface.loc[validation, BASE_COLUMNS].copy()
    normalized_predictions = np.empty(len(base), dtype=float)
    group_receipts: dict[str, object] = {}
    for group_id in CAPACITIES:
        group_training = training & surface["group_id"].eq(group_id).to_numpy()
        group_validation = validation & surface["group_id"].eq(group_id).to_numpy()
        shared_probability = shared.predict_proba(matrix.loc[group_validation])
        group_positions = np.flatnonzero(group_training)
        class_anchors = np.asarray(
            [
                np.flatnonzero(training & classes.eq(class_id).to_numpy())[0]
                for class_id in range(len(active_bins))
            ],
            dtype=int,
        )
        fine_positions = np.concatenate([group_positions, class_anchors])
        fine_weights = np.concatenate(
            [
                normalized_target.iloc[group_positions]
                .clip(lower=0.10)
                .to_numpy(dtype=float),
                np.full(len(class_anchors), 1e-12),
            ]
        )
        fine_params = {
            **shared_params,
            "n_estimators": max(FINE_ITERATIONS),
            "learning_rate": args.fine_learning_rate,
            "min_child_samples": 40,
            "reg_lambda": 4.0,
            "random_state": 20260802 + group_id,
        }
        fine = LGBMClassifier(**fine_params)
        fine.fit(
            matrix.iloc[fine_positions],
            classes.iloc[fine_positions].astype(int),
            sample_weight=fine_weights,
            init_model=shared.booster_,
            callbacks=[lightgbm.log_evaluation(period=0)],
        )
        actual = surface.loc[group_validation, "actual_kwh"].to_numpy(dtype=float)
        mean_generation = float(normalized_target.loc[group_training].mean())
        best: tuple[float, np.ndarray, int, float, str, dict[str, float]] | None = None
        sweep: dict[str, object] = {}
        for fine_iteration in FINE_ITERATIONS:
            probability = fine.predict_proba(
                matrix.loc[group_validation],
                num_iteration=args.shared_iterations + fine_iteration,
            )
            iteration_scores: dict[str, object] = {}
            for fine_weight in PROBABILITY_WEIGHTS:
                blended_probability = (
                    (1.0 - fine_weight) * shared_probability
                    + fine_weight * probability
                )
                chosen, policy, score = _choose_group_action(
                    blended_probability,
                    centers,
                    mean_generation,
                    actual,
                    group_id,
                )
                iteration_scores[f"W{fine_weight:g}"] = {
                    "policy": policy,
                    "score": score,
                }
                choice = (
                    score["total"],
                    chosen,
                    fine_iteration,
                    fine_weight,
                    policy,
                    score,
                )
                if best is None or choice[0] > best[0]:
                    best = choice
            sweep[str(fine_iteration)] = iteration_scores
        assert best is not None
        positions = np.flatnonzero(base["group_id"].eq(group_id).to_numpy())
        normalized_predictions[positions] = best[1]
        group_receipts[str(group_id)] = {
            "training_rows": int(group_training.sum()),
            "selected_fine_iteration": best[2],
            "selected_fine_probability_weight": best[3],
            "selected_policy": best[4],
            "selected_score": best[5],
            "sweep": sweep,
        }
        print(json.dumps({"group_id": group_id, **group_receipts[str(group_id)]}), flush=True)
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
        "architecture": "shared_classifier_site_continuation",
        "scope": "same-fold transfer-learning screen; per-group choices are oracle diagnostics",
        "shared_iterations": args.shared_iterations,
        "fine_learning_rate": args.fine_learning_rate,
        "fine_iterations": list(FINE_ITERATIONS),
        "probability_weights": list(PROBABILITY_WEIGHTS),
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
