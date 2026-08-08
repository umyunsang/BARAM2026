"""Screen M50 with cross-fitted all-weather site-wind features."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import lightgbm
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
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
from run_site_wind_teacher import (
    _all_weather_columns,
    _strict_preceding_mask,
    _validation_mask,
)
from sklearn.model_selection import KFold

FOLDS = ("dev-2023-Q2", "dev-2023-Q3", "dev-2023-Q4")
DECISION_TEMPERATURES = (1.2, 1.0, 0.85, 0.75, 0.6, 0.5, 0.4)
DECISION_GAMMAS = (0.0, 0.2, 0.35, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)


def _legacy_prediction(
    surface: pd.DataFrame,
    columns: list[str],
    preceding: np.ndarray,
    validation: np.ndarray,
) -> np.ndarray:
    matrix = surface[columns].astype("float32")
    training = preceding & surface["scada_ws"].notna().to_numpy()
    positions = np.flatnonzero(training)
    prediction = np.full(len(surface), np.nan, dtype="float32")
    params = {
        "objective": "l2",
        "n_estimators": 400,
        "learning_rate": 0.04,
        "num_leaves": 31,
        "min_child_samples": 60,
        "subsample": 0.9,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 2.0,
        "random_state": 20260801,
        "n_jobs": 6,
        "deterministic": True,
        "force_col_wise": True,
        "verbosity": -1,
    }
    splitter = KFold(3, shuffle=True, random_state=20260801)
    for fit_index, holdout_index in splitter.split(positions):
        model = LGBMRegressor(**params)
        model.fit(
            matrix.iloc[positions[fit_index]],
            surface["scada_ws"].iloc[positions[fit_index]],
        )
        prediction[positions[holdout_index]] = model.predict(
            matrix.iloc[positions[holdout_index]]
        )
    model = LGBMRegressor(**params)
    model.fit(matrix.loc[training], surface.loc[training, "scada_ws"])
    prediction[validation] = model.predict(matrix.loc[validation])
    return prediction


def _allweather_prediction(
    surface: pd.DataFrame,
    columns: list[str],
    preceding: np.ndarray,
    validation: np.ndarray,
    selected_iterations: dict[int, int],
) -> np.ndarray:
    matrix = surface[columns].astype("float32")
    prediction = np.full(len(surface), np.nan, dtype="float32")
    for group_id in CAPACITIES:
        group = surface["group_id"].eq(group_id).to_numpy()
        training = preceding & group & surface["scada_ws"].notna().to_numpy()
        positions = np.flatnonzero(training)
        params = {
            "objective": "l2",
            "n_estimators": selected_iterations[group_id],
            "learning_rate": 0.025,
            "num_leaves": 63,
            "min_child_samples": 40,
            "max_bin": 255,
            "subsample": 0.9,
            "subsample_freq": 1,
            "colsample_bytree": 0.85,
            "reg_alpha": 0.1,
            "reg_lambda": 3.0,
            "random_state": 20260802,
            "n_jobs": 6,
            "deterministic": True,
            "force_col_wise": True,
            "verbosity": -1,
        }
        splitter = KFold(3, shuffle=True, random_state=20260802 + group_id)
        for fit_index, holdout_index in splitter.split(positions):
            model = LGBMRegressor(**params)
            model.fit(
                matrix.iloc[positions[fit_index]],
                surface["scada_ws"].iloc[positions[fit_index]],
            )
            prediction[positions[holdout_index]] = model.predict(
                matrix.iloc[positions[holdout_index]]
            )
        model = LGBMRegressor(**params)
        model.fit(matrix.loc[training], surface.loc[training, "scada_ws"])
        apply = validation & group
        prediction[apply] = model.predict(matrix.loc[apply])
    return prediction


def _select_allweather_iterations(
    surface: pd.DataFrame,
    columns: list[str],
    preceding: np.ndarray,
) -> dict[int, int]:
    matrix = surface[columns].astype("float32")
    selected: dict[int, int] = {}
    for group_id in CAPACITIES:
        group = surface["group_id"].eq(group_id).to_numpy()
        training = preceding & group & surface["scada_ws"].notna().to_numpy()
        batches = (
            surface.loc[training, "data_available_kst_dtm"]
            .drop_duplicates()
            .sort_values()
        )
        cutoff = batches.iloc[int(len(batches) * 0.80)]
        stop = training & surface["data_available_kst_dtm"].ge(cutoff).to_numpy()
        fit = training & ~stop
        model = LGBMRegressor(
            objective="l2",
            n_estimators=1800,
            learning_rate=0.025,
            num_leaves=63,
            min_child_samples=40,
            max_bin=255,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=3.0,
            random_state=20260802,
            n_jobs=6,
            deterministic=True,
            force_col_wise=True,
            verbosity=-1,
        )
        model.fit(
            matrix.loc[fit],
            surface.loc[fit, "scada_ws"],
            eval_set=[(matrix.loc[stop], surface.loc[stop, "scada_ws"])],
            eval_metric="l1",
            callbacks=[lightgbm.early_stopping(100, verbose=False)],
        )
        selected[group_id] = max(1, int(model.best_iteration_ or 1800))
    return selected


def _add_site_wind_features(
    matrix: pd.DataFrame,
    legacy: np.ndarray,
    allweather: np.ndarray,
) -> list[str]:
    matrix["sitewind__legacy"] = legacy
    matrix["sitewind__allweather"] = allweather
    matrix["sitewind__mean"] = (legacy + allweather) / 2.0
    matrix["sitewind__delta"] = allweather - legacy
    matrix["sitewind__disagreement"] = np.abs(allweather - legacy)
    for source in ("legacy", "allweather", "mean"):
        value = matrix[f"sitewind__{source}"]
        matrix[f"sitewind__{source}2"] = value**2
        matrix[f"sitewind__{source}3"] = value**3
        normalized = np.clip((value - 3.0) / 9.0, 0.0, 1.0)
        matrix[f"sitewind__{source}_powercurve"] = normalized**3
    return [name for name in matrix if name.startswith("sitewind__")]


def _choose_actions(
    base: pd.DataFrame,
    probability: np.ndarray,
    centers: np.ndarray,
    normalized_target: pd.Series,
    training: np.ndarray,
    training_groups: pd.Series,
) -> tuple[
    pd.DataFrame,
    str,
    dict[str, dict[str, float]],
    pd.DataFrame,
]:
    actions = np.arange(0.075, 1.076, 0.0025)
    error = np.abs(actions[:, None] - centers[None, :])
    units = np.select([error <= 0.06, error <= 0.08], [4.0, 3.0], default=0.0)
    mean_generation = {
        group_id: float(
            normalized_target.loc[training & training_groups.eq(group_id)].mean()
        )
        for group_id in CAPACITIES
    }
    results: dict[str, dict[str, float]] = {}
    predictions: dict[str, np.ndarray] = {}
    for temperature in DECISION_TEMPERATURES:
        calibrated = probability ** (1.0 / temperature)
        calibrated /= calibrated.sum(axis=1, keepdims=True)
        for gamma in DECISION_GAMMAS:
            chosen = np.empty(len(base), dtype=float)
            for group_id in CAPACITIES:
                mask = base["group_id"].eq(group_id).to_numpy()
                group_probability = calibrated[mask]
                utility = -(group_probability @ error.T) + gamma * (
                    group_probability @ (centers[None, :] * units).T
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
    best = max(results, key=lambda name: results[name]["total"])
    output = base[
        ["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh"]
    ].copy()
    output["prediction_kwh"] = (
        predictions[best] * output["group_id"].map(CAPACITIES).to_numpy(dtype=float)
    )
    policy_frame = base[
        ["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh"]
    ].copy()
    capacity = policy_frame["group_id"].map(CAPACITIES).to_numpy(dtype=float)
    for tag, normalized in predictions.items():
        policy_frame[tag] = normalized * capacity
    return output, best, results, policy_frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--fold", choices=FOLDS, default=FOLDS[-1])
    parser.add_argument("--sitewind-only", action="store_true")
    args = parser.parse_args()
    if _sha256(OPEN) != OPEN_SHA or _sha256(BASELINE) != BASELINE_SHA:
        raise RuntimeError("immutable competition input hash mismatch")
    started = time.perf_counter()
    surface, base_columns, auxiliary_columns = _surface()
    validation = _validation_mask(surface, args.fold)
    preceding = _strict_preceding_mask(surface, validation)
    normalized_target = surface["actual_kwh"] / surface["group_id"].map(CAPACITIES)

    sitewind_cache = OUTPUT / f"{args.candidate_id}-{args.fold}-sitewind-features.npz"
    if sitewind_cache.exists():
        cached = np.load(sitewind_cache)
        legacy = cached["legacy"]
        allweather = cached["allweather"]
        iteration_values = cached["iterations"]
        sitewind_iterations = {
            group_id: int(iteration_values[group_id - 1]) for group_id in CAPACITIES
        }
    else:
        legacy = _legacy_prediction(surface, auxiliary_columns, preceding, validation)
        allweather_columns = _all_weather_columns(surface)
        sitewind_iterations = _select_allweather_iterations(
            surface, allweather_columns, preceding
        )
        allweather = _allweather_prediction(
            surface,
            allweather_columns,
            preceding,
            validation,
            sitewind_iterations,
        )
        np.savez_compressed(
            sitewind_cache,
            legacy=legacy,
            allweather=allweather,
            iterations=np.asarray(
                [sitewind_iterations[group_id] for group_id in CAPACITIES], dtype=int
            ),
        )
    if args.sitewind_only:
        print(
            json.dumps(
                {
                    "candidate_id": args.candidate_id,
                    "fold_id": args.fold,
                    "sitewind_cache": str(sitewind_cache.relative_to(Path.cwd())),
                    "sitewind_selected_iterations": sitewind_iterations,
                    "observed_scada_feature_count": 0,
                    "new_2024_evaluation": False,
                    "lockbox_reopened": False,
                }
            ),
            flush=True,
        )
        return
    matrix = surface[base_columns].astype("float32")
    sitewind_columns = _add_site_wind_features(matrix, legacy, allweather)
    training = preceding & surface["actual_kwh"].notna().to_numpy() & normalized_target.ge(
        0.10
    ).to_numpy()
    width = 0.025
    classes = np.floor((normalized_target.clip(0.10, 1.074999) - 0.10) / width).astype(
        "Int64"
    )
    class_count = int(classes.loc[training].max()) + 1
    centers = np.asarray(
        [
            normalized_target.loc[training & classes.eq(class_id)].mean()
            if (training & classes.eq(class_id)).any()
            else 0.10 + (class_id + 0.5) * width
            for class_id in range(class_count)
        ]
    )
    batches = surface.loc[training, "data_available_kst_dtm"].drop_duplicates().sort_values()
    cutoff = batches.iloc[int(len(batches) * 0.80)]
    inner_fit = training & surface["data_available_kst_dtm"].lt(cutoff).to_numpy()
    inner_stop = training & ~inner_fit
    missing_inner_classes = set(classes.loc[inner_stop].dropna().astype(int)) - set(
        classes.loc[inner_fit].dropna().astype(int)
    )
    for class_id in sorted(missing_inner_classes):
        position = int(np.flatnonzero(inner_stop & classes.eq(class_id).to_numpy())[0])
        inner_fit[position] = True
        inner_stop[position] = False
    params = {
        "objective": "multiclass",
        "num_class": class_count,
        "n_estimators": 800,
        "learning_rate": 0.025,
        "num_leaves": 15,
        "min_child_samples": 80,
        "subsample": 0.9,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 2.0,
        "random_state": 20260801,
        "n_jobs": 6,
        "deterministic": True,
        "force_col_wise": True,
        "verbosity": -1,
    }
    classifier = LGBMClassifier(**params)
    classifier.fit(
        matrix.loc[inner_fit],
        classes.loc[inner_fit].astype(int),
        sample_weight=normalized_target.loc[inner_fit].clip(lower=0.10),
        eval_set=[(matrix.loc[inner_stop], classes.loc[inner_stop].astype(int))],
        eval_sample_weight=[normalized_target.loc[inner_stop].clip(lower=0.10)],
        callbacks=[lightgbm.early_stopping(50, verbose=False)],
    )
    best_iteration = max(1, int(classifier.best_iteration_ or params["n_estimators"]))
    classifier = LGBMClassifier(**{**params, "n_estimators": best_iteration})
    classifier.fit(
        matrix.loc[training],
        classes.loc[training].astype(int),
        sample_weight=normalized_target.loc[training].clip(lower=0.10),
    )
    probability = classifier.predict_proba(matrix.loc[validation])
    base = surface.loc[
        validation, ["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh"]
    ].copy()
    output, best_policy, scores, policies = _choose_actions(
        base,
        probability,
        centers,
        normalized_target,
        training,
        surface["group_id"],
    )
    output["fold_id"] = args.fold
    output["model_id"] = args.candidate_id
    output_path = OUTPUT / f"{args.candidate_id}-{args.fold}.parquet"
    output.to_parquet(output_path, index=False)
    policy_path = OUTPUT / f"{args.candidate_id}-{args.fold}-policies.parquet"
    policies.to_parquet(policy_path, index=False)
    receipt = {
        "schema_version": 1,
        "candidate_id": args.candidate_id,
        "fold_id": args.fold,
        "feature_count": matrix.shape[1],
        "sitewind_feature_count": len(sitewind_columns),
        "sitewind_selected_iterations": sitewind_iterations,
        "selected_iteration": best_iteration,
        "best_policy": best_policy,
        "scores": scores,
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
