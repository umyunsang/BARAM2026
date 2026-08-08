"""Strict settlement-band classifier with inner-only configuration selection."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

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
)
from run_site_wind_teacher import _strict_preceding_mask, _validation_mask
from strict_dev_surface import DEV_CUTOFF, development_surface
from xgboost import XGBClassifier

BASE_COLUMNS = ["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh"]
WIDTHS = (0.10, 0.12, 0.14, 0.16)
FAMILIES = ("lgbm", "xgboost")
POLICIES = ("mode_center", "expected_utility")


def _classes(
    target: pd.Series,
    eligible: np.ndarray,
    width: float,
) -> tuple[pd.Series, np.ndarray]:
    raw = np.floor((target.clip(0.10, 1.074999) - 0.10) / width).astype("Int64")
    active = sorted(raw.loc[eligible].dropna().astype(int).unique())
    mapping = {bin_id: class_id for class_id, bin_id in enumerate(active)}
    classes = raw.map(mapping).astype("Int64")
    centers = []
    for bin_id in active:
        lower = 0.10 + bin_id * width
        upper = min(lower + width, 1.075)
        centers.append((lower + upper) / 2.0)
    return classes, np.asarray(centers, dtype=float)


def _feature_screen(
    matrix: pd.DataFrame,
    target: pd.Series,
    fit: np.ndarray,
    count: int,
) -> list[str]:
    model = LGBMRegressor(
        objective="l1",
        n_estimators=180,
        learning_rate=0.035,
        num_leaves=31,
        min_child_samples=80,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.75,
        reg_alpha=0.2,
        reg_lambda=4.0,
        random_state=20260803,
        n_jobs=6,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )
    model.fit(
        matrix.loc[fit],
        target.loc[fit],
        sample_weight=target.loc[fit].clip(lower=0.10),
    )
    gains = model.booster_.feature_importance(importance_type="gain")
    order = np.argsort(gains)[::-1]
    selected = [matrix.columns[position] for position in order[:count]]
    if len(selected) != count or len(set(selected)) != count:
        raise RuntimeError("feature screen did not resolve a unique fixed set")
    return selected


def _model(
    family: str,
    class_count: int,
    iterations: int,
) -> LGBMClassifier | XGBClassifier:
    if family == "lgbm":
        return LGBMClassifier(
            objective="multiclass",
            num_class=class_count,
            n_estimators=iterations,
            learning_rate=0.035,
            num_leaves=31,
            min_child_samples=80,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_alpha=0.2,
            reg_lambda=4.0,
            random_state=20260803,
            n_jobs=6,
            deterministic=True,
            force_col_wise=True,
            verbosity=-1,
        )
    if family == "xgboost":
        return XGBClassifier(
            objective="multi:softprob",
            num_class=class_count,
            n_estimators=iterations,
            learning_rate=0.035,
            max_depth=5,
            min_child_weight=20.0,
            subsample=0.9,
            colsample_bytree=0.8,
            reg_alpha=0.2,
            reg_lambda=5.0,
            max_bin=256,
            tree_method="hist",
            random_state=20260803,
            n_jobs=6,
        )
    raise ValueError(f"unknown family: {family}")


def _policy_predictions(
    probability: np.ndarray,
    centers: np.ndarray,
    groups: np.ndarray,
    mean_generation: dict[int, float],
) -> dict[str, np.ndarray]:
    output = {"mode_center": centers[np.argmax(probability, axis=1)]}
    actions = np.arange(0.075, 1.0751, 0.0025)
    error = np.abs(actions[:, None] - centers[None, :])
    units = np.select([error <= 0.06, error <= 0.08], [4.0, 3.0], default=0.0)
    chosen = np.empty(len(probability), dtype=float)
    for group_id in CAPACITIES:
        group = groups == group_id
        utility = -(probability[group] @ error.T) + (
            probability[group] @ (centers[None, :] * units).T
        ) / (4.0 * mean_generation[group_id])
        chosen[group] = actions[np.argmax(utility, axis=1)]
    output["expected_utility"] = chosen
    return output


def _fit_predict(
    matrix: pd.DataFrame,
    target: pd.Series,
    groups: pd.Series,
    fit: np.ndarray,
    apply: np.ndarray,
    selected: list[str],
    family: str,
    width: float,
    iterations: int,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    classes, centers = _classes(target, fit, width)
    if classes.loc[fit].isna().any():
        raise RuntimeError("eligible fit row has no settlement class")
    model = _model(family, len(centers), iterations)
    model.fit(
        matrix.loc[fit, selected],
        classes.loc[fit].astype(int),
        sample_weight=target.loc[fit].clip(lower=0.10),
    )
    probability = model.predict_proba(matrix.loc[apply, selected])
    probability = np.asarray(probability, dtype=float)
    probability /= probability.sum(axis=1, keepdims=True)
    means = {
        group_id: float(target.loc[fit & groups.eq(group_id).to_numpy()].mean())
        for group_id in CAPACITIES
    }
    policies = _policy_predictions(
        probability,
        centers,
        groups.loc[apply].to_numpy(dtype=int),
        means,
    )
    return policies, {
        "class_count": len(centers),
        "centers": centers.tolist(),
        "mean_generation": means,
    }


def _score_policies(
    surface: pd.DataFrame,
    apply: np.ndarray,
    policies: dict[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    base = surface.loc[apply, BASE_COLUMNS].copy()
    capacity = base["group_id"].map(CAPACITIES).to_numpy(dtype=float)
    return {
        name: _score(base.assign(prediction_kwh=prediction * capacity))
        for name, prediction in policies.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument(
        "--fold",
        choices=("dev-2023-Q2", "dev-2023-Q3", "dev-2023-Q4"),
        required=True,
    )
    parser.add_argument("--calibration-days", type=int, default=60)
    parser.add_argument("--feature-count", type=int, default=180)
    parser.add_argument("--iterations", type=int, default=160)
    args = parser.parse_args()
    if not 30 <= args.calibration_days <= 90:
        raise ValueError("calibration-days must be between 30 and 90")
    if not 80 <= args.feature_count <= 300:
        raise ValueError("feature-count must be between 80 and 300")
    if not 60 <= args.iterations <= 300:
        raise ValueError("iterations must be between 60 and 300")
    if _sha256(OPEN) != OPEN_SHA or _sha256(BASELINE) != BASELINE_SHA:
        raise RuntimeError("immutable competition input hash mismatch")

    started = time.perf_counter()
    surface, feature_columns = development_surface()
    if surface["forecast_kst_dtm"].ge(DEV_CUTOFF).any():
        raise RuntimeError("lockbox row reached the coarse classifier")
    validation = _validation_mask(surface, args.fold)
    preceding = _strict_preceding_mask(surface, validation)
    target = surface["actual_kwh"] / surface["group_id"].map(CAPACITIES)
    eligible = surface["actual_kwh"].notna().to_numpy() & target.ge(0.10).to_numpy()
    training = preceding & eligible
    calibration_cutoff = pd.Timestamp(
        surface.loc[validation, "data_available_kst_dtm"].min()
    ) - np.timedelta64(args.calibration_days, "D")
    inner_calibration = training & surface["forecast_kst_dtm"].ge(
        calibration_cutoff
    ).to_numpy()
    inner_fit = training & ~inner_calibration
    for group_id in CAPACITIES:
        group = surface["group_id"].eq(group_id).to_numpy()
        if int((inner_fit & group).sum()) < 300 or int(
            (inner_calibration & group).sum()
        ) < 300:
            raise RuntimeError(f"group {group_id} lacks inner chronology support")

    matrix = surface[feature_columns].astype("float32")
    inner_features = _feature_screen(
        matrix, target, inner_fit, args.feature_count
    )
    inner_sweep: dict[str, object] = {}
    best: tuple[float, str, float, str] | None = None
    for family in FAMILIES:
        for width in WIDTHS:
            policies, diagnostics = _fit_predict(
                matrix,
                target,
                surface["group_id"],
                inner_fit,
                inner_calibration,
                inner_features,
                family,
                width,
                args.iterations,
            )
            scores = _score_policies(surface, inner_calibration, policies)
            tag = f"{family}_W{width:g}"
            inner_sweep[tag] = {"scores": scores, **diagnostics}
            for policy in POLICIES:
                choice = (scores[policy]["total"], family, width, policy)
                if best is None or choice[0] > best[0]:
                    best = choice
            print(json.dumps({"inner": tag, "scores": scores}), flush=True)
    assert best is not None
    selected_family, selected_width, selected_policy = best[1:]

    final_features = _feature_screen(
        matrix, target, training, args.feature_count
    )
    outer_policies, outer_diagnostics = _fit_predict(
        matrix,
        target,
        surface["group_id"],
        training,
        validation,
        final_features,
        selected_family,
        selected_width,
        args.iterations,
    )
    outer_scores = _score_policies(surface, validation, outer_policies)
    base = surface.loc[validation, BASE_COLUMNS].copy()
    capacity = base["group_id"].map(CAPACITIES).to_numpy(dtype=float)
    output = base.assign(
        prediction_kwh=outer_policies[selected_policy] * capacity,
        fold_id=args.fold,
        model_id=args.candidate_id,
    )
    output_path = OUTPUT / f"{args.candidate_id}-{args.fold}.parquet"
    output.to_parquet(output_path, index=False)
    receipt = {
        "schema_version": 1,
        "candidate_id": args.candidate_id,
        "fold_id": args.fold,
        "architecture": "strict_coarse_settlement_band_classifier",
        "scope": "outer labels excluded from feature, configuration, and policy selection",
        "physical_development_cutoff": str(DEV_CUTOFF),
        "calibration_days": args.calibration_days,
        "feature_count": args.feature_count,
        "iterations": args.iterations,
        "inner_sweep": inner_sweep,
        "selected_family": selected_family,
        "selected_width": selected_width,
        "selected_policy": selected_policy,
        "selected_inner_score": best[0],
        "outer_policy_scores": outer_scores,
        "fold_score": outer_scores[selected_policy],
        "outer_diagnostics": outer_diagnostics,
        "selected_feature_names": final_features,
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
