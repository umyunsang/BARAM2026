"""Strict conditional Gaussian-mixture residual distribution with Bayes actions.

The point forecast and every residual used to estimate the conditional density
are produced without access to the outer validation labels.  Residual density
contexts are group-specific point-forecast quantile bins crossed with broad
lead-time bands.  A smooth mixture is shifted around each final point forecast
and converted to an action using the official group-level FICR denominator.
"""

from __future__ import annotations

import argparse
import json
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
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
from strict_dev_surface import DEV_CUTOFF, development_surface

BASE_COLUMNS = ["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh"]
TARGET_CENTERS = np.arange(0.0775, 1.0751, 0.005)
ACTIONS = np.arange(0.075, 1.0751, 0.0025)
LEAD_EDGES = np.asarray((11.5, 19.5, 27.5, 35.5))
RESEARCH_SOURCE = (
    "https://www.sciencedirect.com/science/article/pii/S0306261917317555"
)


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


def _screen(
    matrix: pd.DataFrame,
    target: pd.Series,
    fit: np.ndarray,
    top_features: int,
    seed: int,
) -> list[str]:
    model = _model(140, seed)
    model.fit(
        matrix.loc[fit],
        target.loc[fit],
        sample_weight=target.loc[fit].clip(lower=0.10),
    )
    gain = model.booster_.feature_importance(importance_type="gain")
    order = np.argsort(gain)[::-1][:top_features]
    selected = [matrix.columns[position] for position in order]
    if len(selected) != top_features or len(set(selected)) != top_features:
        raise RuntimeError("conditional-GMM feature screen contract changed")
    return selected


def _crossfit_point(
    surface: pd.DataFrame,
    matrix: pd.DataFrame,
    target: pd.Series,
    fit: np.ndarray,
    top_features: int,
    iterations: int,
    seed: int,
) -> tuple[np.ndarray, list[dict[str, int]]]:
    positions = np.flatnonzero(fit)
    batches = np.asarray(
        sorted(surface.loc[fit, "data_available_kst_dtm"].unique())
    )
    if len(batches) < 150:
        raise RuntimeError("conditional-GMM history has too few issuance batches")
    oof = np.full(len(surface), np.nan, dtype=float)
    diagnostics: list[dict[str, int]] = []
    splitter = KFold(n_splits=5, shuffle=True, random_state=seed)
    for fold_index, (train_index, apply_index) in enumerate(splitter.split(batches)):
        train_batches = set(batches[train_index])
        apply_batches = set(batches[apply_index])
        local_fit = fit & surface["data_available_kst_dtm"].isin(train_batches).to_numpy()
        local_apply = fit & surface["data_available_kst_dtm"].isin(apply_batches).to_numpy()
        selected = _screen(
            matrix,
            target,
            local_fit,
            top_features,
            seed + 100 + fold_index,
        )
        model = _model(iterations, seed + 200 + fold_index)
        model.fit(
            matrix.loc[local_fit, selected],
            target.loc[local_fit],
            sample_weight=target.loc[local_fit].clip(lower=0.10),
        )
        oof[local_apply] = np.clip(
            model.predict(matrix.loc[local_apply, selected]), 0.075, 1.075
        )
        diagnostics.append(
            {
                "fold": fold_index,
                "fit_rows": int(local_fit.sum()),
                "apply_rows": int(local_apply.sum()),
            }
        )
    if not np.isfinite(oof[positions]).all():
        raise RuntimeError("conditional-GMM OOF point forecast is incomplete")
    return oof, diagnostics


def _final_point(
    matrix: pd.DataFrame,
    target: pd.Series,
    fit: np.ndarray,
    apply: np.ndarray,
    top_features: int,
    iterations: int,
    seed: int,
) -> tuple[np.ndarray, list[str]]:
    selected = _screen(matrix, target, fit, top_features, seed)
    model = _model(iterations, seed + 1)
    model.fit(
        matrix.loc[fit, selected],
        target.loc[fit],
        sample_weight=target.loc[fit].clip(lower=0.10),
    )
    prediction = np.clip(
        model.predict(matrix.loc[apply, selected]), 0.075, 1.075
    )
    return prediction, selected


def _point_edges(values: np.ndarray, bin_count: int) -> np.ndarray:
    raw = np.quantile(values, np.linspace(0.0, 1.0, bin_count + 1))
    raw[0] = -np.inf
    raw[-1] = np.inf
    for index in range(1, len(raw) - 1):
        raw[index] = max(raw[index], raw[index - 1] + 1e-6)
    return raw


def _context_ids(
    point: np.ndarray,
    lead: np.ndarray,
    point_edges: np.ndarray,
) -> np.ndarray:
    point_id = np.clip(
        np.searchsorted(point_edges, point, side="right") - 1,
        0,
        len(point_edges) - 2,
    )
    lead_id = np.clip(
        np.searchsorted(LEAD_EDGES, lead, side="right") - 1,
        0,
        len(LEAD_EDGES) - 2,
    )
    return point_id * (len(LEAD_EDGES) - 1) + lead_id


def _fit_mixture(values: np.ndarray, components: int, seed: int) -> GaussianMixture:
    component_count = min(components, max(1, len(values) // 80))
    model = GaussianMixture(
        n_components=component_count,
        covariance_type="full",
        reg_covar=2.5e-4,
        max_iter=300,
        n_init=3,
        random_state=seed,
    )
    model.fit(values.reshape(-1, 1))
    return model


def _conditional_probability(
    train_point: np.ndarray,
    train_target: np.ndarray,
    train_lead: np.ndarray,
    apply_point: np.ndarray,
    apply_lead: np.ndarray,
    bin_count: int,
    components: int,
    spread_scale: float,
    seed: int,
) -> tuple[np.ndarray, dict[str, object]]:
    point_edges = _point_edges(train_point, bin_count)
    train_context = _context_ids(train_point, train_lead, point_edges)
    apply_context = _context_ids(apply_point, apply_lead, point_edges)
    residual = train_target - train_point
    global_model = _fit_mixture(residual, components, seed)
    models: dict[int, GaussianMixture] = {}
    counts: dict[str, int] = {}
    for context in np.unique(train_context):
        selected = residual[train_context == context]
        counts[str(int(context))] = len(selected)
        if len(selected) >= 120:
            models[int(context)] = _fit_mixture(
                selected, components, seed + 10 + int(context)
            )

    probability = np.empty((len(apply_point), len(TARGET_CENTERS)), dtype=float)
    for context in np.unique(apply_context):
        rows = np.flatnonzero(apply_context == context)
        model = models.get(int(context), global_model)
        weights = model.weights_.reshape(-1)
        means = model.means_.reshape(-1)
        scales = np.sqrt(model.covariances_.reshape(-1)) * spread_scale
        scales = np.maximum(scales, 0.01)
        target_residual = (
            TARGET_CENTERS[None, None, :]
            - apply_point[rows, None, None]
            - means[None, :, None]
        )
        density = np.exp(-0.5 * (target_residual / scales[None, :, None]) ** 2)
        density /= scales[None, :, None]
        probability[rows] = np.sum(density * weights[None, :, None], axis=1)
    probability = np.clip(probability, 1e-15, None)
    probability /= probability.sum(axis=1, keepdims=True)
    return probability, {
        "point_edges": point_edges.tolist(),
        "context_counts": counts,
        "fitted_context_count": len(models),
        "global_components": int(global_model.n_components),
    }


def _bayes_action(
    probability: np.ndarray,
    mean_generation: float,
    gamma: float,
) -> np.ndarray:
    error = np.abs(ACTIONS[:, None] - TARGET_CENTERS[None, :])
    units = np.select([error <= 0.06, error <= 0.08], [4.0, 3.0], default=0.0)
    utility = -(probability @ error.T)
    utility += gamma * (
        probability @ (TARGET_CENTERS[None, :] * units).T
    ) / (4.0 * mean_generation)
    return ACTIONS[np.argmax(utility, axis=1)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument(
        "--fold",
        choices=("dev-2023-Q2", "dev-2023-Q3", "dev-2023-Q4"),
        required=True,
    )
    parser.add_argument("--top-features", type=int, default=180)
    parser.add_argument("--iterations", type=int, default=320)
    parser.add_argument("--point-bins", type=int, default=8)
    parser.add_argument("--components", type=int, default=3)
    parser.add_argument("--spread-scale", type=float, default=1.0)
    parser.add_argument("--utility-gamma", type=float, default=1.0)
    args = parser.parse_args()
    if not 100 <= args.top_features <= 260:
        raise ValueError("top-features must be between 100 and 260")
    if not 180 <= args.iterations <= 500:
        raise ValueError("iterations must be between 180 and 500")
    if not 4 <= args.point_bins <= 12:
        raise ValueError("point-bins must be between four and twelve")
    if args.components not in {1, 2, 3, 4}:
        raise ValueError("components must be one through four")
    if not 0.6 <= args.spread_scale <= 1.6:
        raise ValueError("spread-scale must be between 0.6 and 1.6")
    if not 0.0 <= args.utility_gamma <= 2.0:
        raise ValueError("utility-gamma must be between zero and two")
    if _sha256(OPEN) != OPEN_SHA or _sha256(BASELINE) != BASELINE_SHA:
        raise RuntimeError("immutable competition input hash mismatch")

    started = time.perf_counter()
    surface, feature_columns = development_surface()
    if surface["forecast_kst_dtm"].ge(DEV_CUTOFF).any():
        raise RuntimeError("lockbox row reached conditional-GMM runner")
    validation = _validation_mask(surface, args.fold)
    history = _strict_preceding_mask(surface, validation)
    target = surface["actual_kwh"] / surface["group_id"].map(CAPACITIES)
    eligible = surface["actual_kwh"].notna().to_numpy() & target.ge(0.10).to_numpy()
    matrix = surface[feature_columns].astype("float32")
    output = surface.loc[validation, BASE_COLUMNS].copy()
    output_groups = output["group_id"].to_numpy(dtype=int)
    normalized = np.empty(len(output), dtype=float)
    diagnostics: dict[str, object] = {}

    for group_id in CAPACITIES:
        group = surface["group_id"].eq(group_id).to_numpy()
        fit = history & eligible & group
        apply = validation & group
        oof, crossfit_diagnostics = _crossfit_point(
            surface,
            matrix,
            target,
            fit,
            args.top_features,
            args.iterations,
            20261200 + group_id * 100,
        )
        point, selected = _final_point(
            matrix,
            target,
            fit,
            apply,
            args.top_features,
            args.iterations,
            20261300 + group_id * 100,
        )
        probability, mixture_diagnostics = _conditional_probability(
            oof[fit],
            target.loc[fit].to_numpy(dtype=float),
            surface.loc[fit, "lead_hour"].to_numpy(dtype=float),
            point,
            surface.loc[apply, "lead_hour"].to_numpy(dtype=float),
            args.point_bins,
            args.components,
            args.spread_scale,
            20261400 + group_id * 100,
        )
        action = _bayes_action(
            probability,
            float(target.loc[fit].mean()),
            args.utility_gamma,
        )
        normalized[output_groups == group_id] = action
        actual = target.loc[apply].to_numpy(dtype=float)
        diagnostics[str(group_id)] = {
            "fit_rows": int(fit.sum()),
            "apply_rows": int(apply.sum()),
            "selected_feature_names": selected,
            "crossfit": crossfit_diagnostics,
            "oof_point_mae": float(
                np.mean(np.abs(oof[fit] - target.loc[fit].to_numpy(dtype=float)))
            ),
            "point_score": _group_total(actual, point),
            "action_score": _group_total(actual, action),
            "mixture": mixture_diagnostics,
        }
        print(
            json.dumps(
                {
                    "group_id": group_id,
                    "fit_rows": int(fit.sum()),
                    "point_score": diagnostics[str(group_id)]["point_score"],
                    "action_score": diagnostics[str(group_id)]["action_score"],
                }
            ),
            flush=True,
        )

    output["prediction_kwh"] = normalized * output["group_id"].map(
        CAPACITIES
    ).to_numpy(dtype=float)
    output["fold_id"] = args.fold
    output["model_id"] = args.candidate_id
    output_path = OUTPUT / f"{args.candidate_id}-{args.fold}.parquet"
    output.to_parquet(output_path, index=False)
    receipt = {
        "schema_version": 1,
        "candidate_id": args.candidate_id,
        "fold_id": args.fold,
        "architecture": "strict_group_conditional_residual_gmm_bayes_action",
        "scope": "fixed policy; outer labels excluded from fitting and selection",
        "research_source": RESEARCH_SOURCE,
        "physical_development_cutoff": str(DEV_CUTOFF),
        "top_features": args.top_features,
        "iterations": args.iterations,
        "point_bins": args.point_bins,
        "lead_bands": LEAD_EDGES.tolist(),
        "components": args.components,
        "spread_scale": args.spread_scale,
        "utility_gamma": args.utility_gamma,
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
