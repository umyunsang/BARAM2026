"""Screen direct normalized-power XGBoost regressors on unseen development folds."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
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
from xgboost import XGBRegressor

BASE_COLUMNS = ["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh"]
PARENT_PATH = OUTPUT / "M107_STRICT_TEMPORAL_TOP100-oof.parquet"


def _model(
    objective: str,
    iterations: int,
    max_depth: int,
    min_child_weight: float,
    seed: int,
    quantile_alpha: float,
) -> XGBRegressor:
    parameters: dict[str, object] = {
        "objective": f"reg:{objective}",
        "n_estimators": iterations,
        "learning_rate": 0.025,
        "max_depth": max_depth,
        "min_child_weight": min_child_weight,
        "subsample": 0.9,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 5.0,
        "max_bin": 256,
        "tree_method": "hist",
        "random_state": seed,
        "n_jobs": 6,
    }
    if objective == "quantileerror":
        parameters["quantile_alpha"] = quantile_alpha
    return XGBRegressor(**parameters)


def _fit_predict(
    matrix: pd.DataFrame,
    target: pd.Series,
    training: np.ndarray,
    validation: np.ndarray,
    groups: pd.Series,
    architecture: str,
    objective: str,
    iterations: int,
    checkpoints: list[int],
    sample_power: float,
    max_depth: int,
    min_child_weight: float,
    quantile_alpha: float,
) -> tuple[dict[int, np.ndarray], dict[str, object]]:
    predictions = {
        checkpoint: np.full(len(matrix), np.nan, dtype="float32")
        for checkpoint in checkpoints
    }
    diagnostics: dict[str, object] = {}
    model_groups = (0,) if architecture == "shared" else tuple(CAPACITIES)
    for model_group in model_groups:
        if model_group == 0:
            fit_mask = training
            apply_mask = validation
        else:
            fit_mask = training & groups.eq(model_group).to_numpy()
            apply_mask = validation & groups.eq(model_group).to_numpy()
        weights = np.power(
            target.loc[fit_mask].clip(lower=0.10).to_numpy(dtype=float),
            sample_power,
        )
        model = _model(
            objective,
            iterations,
            max_depth,
            min_child_weight,
            20260802 + model_group,
            quantile_alpha,
        )
        model.fit(matrix.loc[fit_mask], target.loc[fit_mask], sample_weight=weights)
        for checkpoint in checkpoints:
            prediction = model.predict(
                matrix.loc[apply_mask], iteration_range=(0, checkpoint)
            )
            predictions[checkpoint][apply_mask] = prediction
        diagnostics[str(model_group)] = {
            "training_rows": int(fit_mask.sum()),
            "validation_rows": int(apply_mask.sum()),
            "target_mean": float(target.loc[fit_mask].mean()),
            "weight_mean": float(weights.mean()),
        }
    for checkpoint, prediction in predictions.items():
        if not np.isfinite(prediction[validation]).all():
            raise RuntimeError(
                f"non-finite direct-regression prediction at checkpoint {checkpoint}"
            )
    return predictions, diagnostics


def _policies(base: pd.DataFrame, normalized_point: np.ndarray) -> pd.DataFrame:
    capacity = base["group_id"].map(CAPACITIES).to_numpy(dtype=float)
    additions: dict[str, np.ndarray] = {}
    for scale in np.arange(0.85, 1.151, 0.05):
        for offset in np.arange(-0.08, 0.0801, 0.005):
            normalized = np.clip(scale * normalized_point + offset, 0.075, 1.075)
            additions[f"S{scale:.2f}_O{offset:+.3f}"] = normalized * capacity
    return pd.concat([base.reset_index(drop=True), pd.DataFrame(additions)], axis=1)


def _best_raw(base: pd.DataFrame, policies: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    best: tuple[float, str, pd.DataFrame] | None = None
    for name in sorted(set(policies.columns).difference(BASE_COLUMNS)):
        candidate = base.copy()
        candidate["prediction_kwh"] = policies[name].to_numpy(dtype=float)
        score = _score(candidate)
        choice = (score["total"], name, candidate)
        if best is None or choice[0] > best[0]:
            best = choice
    assert best is not None
    return best[1], best[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--fold", choices=FOLDS, default=FOLDS[-1])
    parser.add_argument("--sitewind-cache-id", required=True)
    parser.add_argument(
        "--architecture", choices=("shared", "group"), default="group"
    )
    parser.add_argument(
        "--objective",
        choices=("absoluteerror", "pseudohubererror", "squarederror", "quantileerror"),
        default="absoluteerror",
    )
    parser.add_argument("--quantile-alpha", type=float, default=0.5)
    parser.add_argument("--iterations", nargs="+", type=int, default=[100, 200, 300, 400])
    parser.add_argument("--sample-power", type=float, default=0.0)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--min-child-weight", type=float, default=20.0)
    args = parser.parse_args()
    checkpoints = sorted(set(args.iterations))
    if not checkpoints or checkpoints[0] < 1 or checkpoints[-1] > 800:
        raise ValueError("iterations must be between one and 800")
    if not 0.0 <= args.sample_power <= 2.0:
        raise ValueError("sample-power must be between zero and two")
    if not 0.05 <= args.quantile_alpha <= 0.95:
        raise ValueError("quantile-alpha must be between 0.05 and 0.95")
    if not 2 <= args.max_depth <= 10:
        raise ValueError("max-depth must be between two and ten")
    if not 1.0 <= args.min_child_weight <= 200.0:
        raise ValueError("min-child-weight must be between one and 200")
    if _sha256(OPEN) != OPEN_SHA or _sha256(BASELINE) != BASELINE_SHA:
        raise RuntimeError("immutable competition input hash mismatch")

    started = time.perf_counter()
    surface, base_columns, _ = _surface()
    validation = _validation_mask(surface, args.fold)
    validation_start = surface.loc[validation, "forecast_kst_dtm"].min()
    normalized_target = surface["actual_kwh"] / surface["group_id"].map(CAPACITIES)
    training = (
        surface["forecast_kst_dtm"].lt(validation_start).to_numpy()
        & normalized_target.ge(0.10).to_numpy()
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
    predictions, diagnostics = _fit_predict(
        matrix,
        normalized_target,
        training,
        validation,
        surface["group_id"],
        args.architecture,
        args.objective,
        max(checkpoints),
        checkpoints,
        args.sample_power,
        args.max_depth,
        args.min_child_weight,
        args.quantile_alpha,
    )
    base = surface.loc[validation, BASE_COLUMNS].copy()
    parent = pd.read_parquet(PARENT_PATH)
    parent = parent.loc[parent["fold_id"].eq(args.fold)]
    sweep: dict[str, object] = {}
    best: tuple[float, int, pd.DataFrame, pd.DataFrame, str, dict[str, object]] | None = None
    for checkpoint in checkpoints:
        normalized_point = predictions[checkpoint][validation]
        policies = _policies(base, normalized_point)
        raw_policy, raw_output = _best_raw(base, policies)
        blended, selections = _screen_blends(base, policies, parent)
        raw_score = _score(raw_output)
        blend_score = _score(blended)
        sweep[str(checkpoint)] = {
            "raw_policy": raw_policy,
            "raw_score": raw_score,
            "oracle_blend_score": blend_score,
            "oracle_blends": selections,
        }
        choice = (
            blend_score["total"],
            checkpoint,
            blended,
            policies,
            raw_policy,
            selections,
        )
        if best is None or choice[0] > best[0]:
            best = choice
        print(json.dumps({"checkpoint": checkpoint, **sweep[str(checkpoint)]}), flush=True)
    assert best is not None
    output = best[2]
    output["fold_id"] = args.fold
    output["model_id"] = args.candidate_id
    output_path = OUTPUT / f"{args.candidate_id}-{args.fold}.parquet"
    policy_path = OUTPUT / f"{args.candidate_id}-{args.fold}-policies.parquet"
    output.to_parquet(output_path, index=False)
    best[3].to_parquet(policy_path, index=False)
    receipt = {
        "schema_version": 1,
        "candidate_id": args.candidate_id,
        "fold_id": args.fold,
        "architecture": f"xgboost_{args.architecture}_normalized_{args.objective}",
        "scope": (
            "unseen-fold direct-regression representation screen; "
            "oracle policies not promoted"
        ),
        "selected_checkpoint": best[1],
        "selected_raw_policy": best[4],
        "selected_oracle_blends": best[5],
        "selected_oracle_blend_score": _score(output),
        "sweep": sweep,
        "feature_count": len(selected_features),
        "selected_feature_names": selected_features,
        "sitewind_feature_count": len(sitewind_columns),
        "sample_power": args.sample_power,
        "quantile_alpha": args.quantile_alpha,
        "max_depth": args.max_depth,
        "min_child_weight": args.min_child_weight,
        "model_diagnostics": diagnostics,
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
