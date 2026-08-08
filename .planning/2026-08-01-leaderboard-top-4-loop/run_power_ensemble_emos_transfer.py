"""Evaluate one fixed final-power ensemble moment calibration over M244."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from build_v2_transfer_sequence_challenger import (
    METRIC_COLUMNS,
    OOF,
    _group_score,
    _paired_issuance_bootstrap,
    _score,
)
from run_conditional_daily_analog_profile import (
    BASELINE,
    BASELINE_SHA,
    OPEN,
    OPEN_SHA,
    REPRESENTATIONS,
    Recipe,
    _combine,
    _complete_group_days,
    _distances,
    _feature_sets,
    _kernel_weights,
    _profile_frame,
    _profile_heads,
)
from run_rare_event_corrected_analog_transfer import (
    HUB_FEATURE,
    SLOPE_CAP,
    TAIL_QUANTILE,
    _ols_slope,
    _rare_event_profile,
)
from run_recency_spread_analog_transfer import _composed_profile, _parents
from run_recency_weighted_analog_transfer import HALF_LIFE_DAYS
from run_spread_shrunk_analog_transfer import (
    REFERENCE_QUANTILE,
    _apply_spread_recipe,
    _spread_adjusted_profile,
)
from run_stable_daily_analog_profile import DISTRIBUTION
from scipy.optimize import nnls
from sklearn.linear_model import LinearRegression
from strict_dev_surface import DEV_CUTOFF, development_surface

from baram.contracts.hashing import canonical_sha256, sha256_file

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "artifacts" / "backtests" / "metric-aligned-probe"
M244_RECEIPT = OUTPUT / "M244_RARE_EVENT_CORRECTED_ANALOG_Q234.json"
M244_RECEIPT_SHA = (
    "8489f2759ce39fe4abb7c5179386cde0e1a39d22a5d07673a782b4163410a371"
)
MODEL_ID = "M246_FINAL_POWER_EMOS_ANALOG_Q234"
FOLDS = ("q2", "q3", "q4")
INITIAL_HISTORY_DAYS = 80
CALIBRATION_BLOCK_DAYS = 90
MIN_CALIBRATION_ROWS = 120


def _weighted_moments(
    members: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mean = np.sum(members * weights[:, :, None], axis=1)
    variance = np.sum(
        weights[:, :, None] * (members - mean[:, None, :]) ** 2,
        axis=1,
    )
    return mean, np.maximum(variance, 0.0)


def _corrected_members(
    reference_values: np.ndarray,
    reference_targets: np.ndarray,
    reference_issuances: np.ndarray,
    query_values: np.ndarray,
    query_issuances: np.ndarray,
    representation: object,
    recipe: Recipe,
    hub_index: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    order, distance, retrieval = _distances(
        reference_values,
        query_values,
        reference_issuances,
        query_issuances,
        representation,
    )
    neighbor_indices = order[:, : recipe.neighbors]
    neighbor_distance = distance[:, : recipe.neighbors]
    base_weights = _kernel_weights(neighbor_distance, recipe.kernel)
    age_days = (
        query_issuances[:, None] - reference_issuances[neighbor_indices]
    ) / np.timedelta64(1, "D")
    age_days = np.asarray(age_days, dtype=float)
    if np.any(age_days <= 0.0):
        raise RuntimeError("M246 found a non-historical analog neighbor")
    weights = base_weights * np.exp2(-age_days / HALF_LIFE_DAYS)
    weights /= weights.sum(axis=1, keepdims=True)

    reference_hub = reference_values[:, :, hub_index].copy()
    query_hub = query_values[:, :, hub_index].copy()
    hub_median = float(np.nanmedian(reference_hub))
    reference_hub = np.where(
        np.isfinite(reference_hub), reference_hub, hub_median
    )
    query_hub = np.where(np.isfinite(query_hub), query_hub, hub_median)
    raw_slope = _ols_slope(
        reference_hub.reshape(-1),
        reference_targets.reshape(-1),
    )
    slope = float(np.clip(raw_slope, 0.0, SLOPE_CAP))
    threshold = float(np.quantile(reference_hub, TAIL_QUANTILE))
    high = query_hub > threshold
    neighbor_hub = reference_hub[neighbor_indices]
    neighbor_targets = reference_targets[neighbor_indices]
    adjustment = slope * (query_hub[:, None, :] - neighbor_hub)
    corrected = np.where(
        high[:, None, :],
        neighbor_targets + adjustment,
        neighbor_targets,
    )
    corrected = np.clip(corrected, 0.0, 1.0)
    return corrected, weights, {
        **retrieval,
        "reference_days": len(reference_issuances),
        "query_days": len(query_issuances),
        "raw_ols_slope": raw_slope,
        "applied_slope": slope,
        "hub_threshold": threshold,
        "high_query_fraction": float(np.mean(high)),
        "corrected_clip_fraction": float(
            np.mean(
                (neighbor_targets + np.where(high[:, None, :], adjustment, 0.0)
                < 0.0)
                | (
                    neighbor_targets
                    + np.where(high[:, None, :], adjustment, 0.0)
                    > 1.0
                )
            )
        ),
    }


def _prequential_calibration_pairs(
    train_values: np.ndarray,
    train_targets: np.ndarray,
    train_issuances: np.ndarray,
    representation: object,
    recipe: Recipe,
    hub_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    if len(train_issuances) <= INITIAL_HISTORY_DAYS:
        raise RuntimeError("M246 calibration history is too short")
    means: list[np.ndarray] = []
    variances: list[np.ndarray] = []
    actuals: list[np.ndarray] = []
    blocks: list[dict[str, object]] = []
    for start in range(
        INITIAL_HISTORY_DAYS,
        len(train_issuances),
        CALIBRATION_BLOCK_DAYS,
    ):
        stop = min(start + CALIBRATION_BLOCK_DAYS, len(train_issuances))
        members, weights, diagnostics = _corrected_members(
            train_values[:start],
            train_targets[:start],
            train_issuances[:start],
            train_values[start:stop],
            train_issuances[start:stop],
            representation,
            recipe,
            hub_index,
        )
        mean, variance = _weighted_moments(members, weights)
        means.append(mean)
        variances.append(variance)
        actuals.append(train_targets[start:stop])
        blocks.append(
            {
                **diagnostics,
                "first_query_issuance": str(train_issuances[start]),
                "last_query_issuance": str(train_issuances[stop - 1]),
            }
        )
    mean = np.concatenate(means, axis=0)
    variance = np.concatenate(variances, axis=0)
    actual = np.concatenate(actuals, axis=0)
    if mean.size < MIN_CALIBRATION_ROWS or not np.isfinite(actual).all():
        raise RuntimeError("M246 calibration-pair support changed")
    return mean, variance, actual, {
        "initial_history_days": INITIAL_HISTORY_DAYS,
        "block_days": CALIBRATION_BLOCK_DAYS,
        "calibration_days": len(mean),
        "calibration_rows": mean.size,
        "blocks": blocks,
    }


def _fit_moment_calibration(
    mean: np.ndarray,
    variance: np.ndarray,
    actual: np.ndarray,
) -> tuple[dict[str, float], dict[str, object]]:
    x = mean.reshape(-1, 1)
    y = actual.reshape(-1)
    model = LinearRegression(positive=True)
    model.fit(x, y)
    fitted = model.predict(x)
    residual = y - fitted
    variance_design = np.column_stack(
        [np.ones(len(y), dtype=float), variance.reshape(-1)]
    )
    variance_coefficients, variance_residual = nnls(
        variance_design,
        residual * residual,
    )
    parameters = {
        "mean_intercept": float(model.intercept_),
        "mean_slope": float(model.coef_[0]),
        "variance_intercept": float(variance_coefficients[0]),
        "variance_slope": float(variance_coefficients[1]),
    }
    diagnostics = {
        "raw_mean_mae": float(np.mean(np.abs(y - x[:, 0]))),
        "calibrated_mean_mae": float(np.mean(np.abs(residual))),
        "raw_residual_mean": float(np.mean(y - x[:, 0])),
        "calibrated_residual_mean": float(np.mean(residual)),
        "raw_member_variance_mean": float(np.mean(variance)),
        "calibration_residual_variance": float(np.mean(residual * residual)),
        "nnls_residual_norm": float(variance_residual),
    }
    return parameters, diagnostics


def _calibrated_profile(
    surface: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    group_id: int,
    query_issuances: np.ndarray,
    recipe: Recipe,
) -> tuple[pd.DataFrame, dict[str, object]]:
    representation = next(
        item for item in REPRESENTATIONS if item.name == recipe.representation
    )
    features = feature_sets[representation.feature_set]
    if HUB_FEATURE not in features:
        raise RuntimeError("M246 frozen hub-speed feature is absent")
    hub_index = features.index(HUB_FEATURE)
    frame, issuances, values, targets = _complete_group_days(
        surface,
        group_id,
        features,
    )
    query_mask = np.isin(issuances, query_issuances)
    cutoff = pd.Timestamp(np.min(query_issuances))
    day_end = (
        frame.groupby("data_available_kst_dtm", sort=True)["forecast_kst_dtm"]
        .max()
        .reindex(pd.to_datetime(issuances))
        .to_numpy()
    )
    train_mask = (day_end < cutoff) & np.isfinite(targets).all(axis=1)
    if int(train_mask.sum()) <= INITIAL_HISTORY_DAYS or not query_mask.any():
        raise RuntimeError(
            f"group {group_id} M246 days changed: "
            f"train={int(train_mask.sum())}, query={int(query_mask.sum())}"
        )
    train_values = values[train_mask]
    train_targets = targets[train_mask]
    train_issuances = issuances[train_mask]
    query_values = values[query_mask]
    query_days = issuances[query_mask]

    calibration_mean, calibration_variance, calibration_actual, block_diagnostics = (
        _prequential_calibration_pairs(
            train_values,
            train_targets,
            train_issuances,
            representation,
            recipe,
            hub_index,
        )
    )
    parameters, fit_diagnostics = _fit_moment_calibration(
        calibration_mean,
        calibration_variance,
        calibration_actual,
    )
    members, weights, query_diagnostics = _corrected_members(
        train_values,
        train_targets,
        train_issuances,
        query_values,
        query_days,
        representation,
        recipe,
        hub_index,
    )
    raw_mean, raw_variance = _weighted_moments(members, weights)
    calibrated_mean = (
        parameters["mean_intercept"] + parameters["mean_slope"] * raw_mean
    )
    calibrated_variance = (
        parameters["variance_intercept"]
        + parameters["variance_slope"] * raw_variance
    )
    scale = np.sqrt(
        np.maximum(calibrated_variance, 0.0)
        / np.maximum(raw_variance, 1e-8)
    )
    calibrated_members = calibrated_mean[:, None, :] + scale[:, None, :] * (
        members - raw_mean[:, None, :]
    )
    clip_fraction = float(
        np.mean((calibrated_members < 0.0) | (calibrated_members > 1.0))
    )
    calibrated_members = np.clip(calibrated_members, 0.0, 1.0)
    heads = _profile_heads(
        calibrated_members,
        weights,
        float(np.nanmean(train_targets)),
    )
    profile = _profile_frame(
        frame,
        query_days,
        heads[recipe.head],
        group_id,
    )
    return profile, {
        "training_days": int(train_mask.sum()),
        "query_days": int(query_mask.sum()),
        "calibration": block_diagnostics,
        "parameters": parameters,
        "fit": fit_diagnostics,
        "query": query_diagnostics,
        "query_raw_mean_summary": {
            "min": float(np.min(raw_mean)),
            "mean": float(np.mean(raw_mean)),
            "max": float(np.max(raw_mean)),
        },
        "query_calibrated_mean_summary": {
            "min": float(np.min(calibrated_mean)),
            "mean": float(np.mean(calibrated_mean)),
            "max": float(np.max(calibrated_mean)),
        },
        "query_scale_summary": {
            "min": float(np.min(scale)),
            "mean": float(np.mean(scale)),
            "median": float(np.median(scale)),
            "max": float(np.max(scale)),
        },
        "member_clip_fraction": clip_fraction,
    }


def main() -> None:
    if sha256_file(OPEN) != OPEN_SHA or sha256_file(BASELINE) != BASELINE_SHA:
        raise RuntimeError("immutable competition input hash mismatch")
    if sha256_file(M244_RECEIPT) != M244_RECEIPT_SHA:
        raise RuntimeError("M244 promoted receipt hash mismatch")
    m244_receipt = json.loads(M244_RECEIPT.read_text(encoding="utf-8"))
    recipes = {
        int(group_id): Recipe(**raw)
        for group_id, raw in m244_receipt["policy"]["recipes"].items()
    }

    surface, numeric = development_surface()
    if surface["forecast_kst_dtm"].ge(DEV_CUTOFF).any():
        raise RuntimeError("lockbox row reached M246 power-ensemble runner")
    feature_sets = _feature_sets(numeric)
    parents = _parents(surface)
    query_issuances = {
        fold: np.sort(parent["data_available_kst_dtm"].unique())
        for fold, parent in parents.items()
    }

    m244_replacements: dict[str, list[pd.DataFrame]] = {
        fold: [] for fold in FOLDS
    }
    calibrated_replacements: dict[str, list[pd.DataFrame]] = {
        fold: [] for fold in FOLDS
    }
    diagnostics: dict[str, object] = {}
    for group_id, recipe in recipes.items():
        group_diagnostics: dict[str, object] = {"recipe": asdict(recipe)}
        for fold in FOLDS:
            parent = parents[fold]
            m244_profile, m244_retrieval = _rare_event_profile(
                surface,
                feature_sets,
                group_id,
                query_issuances[fold],
                recipe,
            )
            calibrated_profile, calibrated_retrieval = _calibrated_profile(
                surface,
                feature_sets,
                group_id,
                query_issuances[fold],
                recipe,
            )
            spread_profile, spread_retrieval = _spread_adjusted_profile(
                surface,
                feature_sets,
                group_id,
                query_issuances[fold],
                recipe,
            )
            m244_profile = _composed_profile(m244_profile, spread_profile)
            calibrated_profile = _composed_profile(
                calibrated_profile,
                spread_profile,
            )
            m244 = _apply_spread_recipe(
                parent,
                m244_profile,
                group_id,
                recipe,
            )
            calibrated = _apply_spread_recipe(
                parent,
                calibrated_profile,
                group_id,
                recipe,
            )
            m244_replacements[fold].append(m244)
            calibrated_replacements[fold].append(calibrated)
            group_diagnostics[fold] = {
                "parent": _group_score(parent, group_id),
                "m244": _group_score(m244, group_id),
                "power_ensemble_calibrated": _group_score(
                    calibrated,
                    group_id,
                ),
                "m244_retrieval": m244_retrieval,
                "spread_retrieval": spread_retrieval,
                "calibrated_retrieval": calibrated_retrieval,
            }
        diagnostics[str(group_id)] = group_diagnostics

    m244_outputs = {
        fold: _combine(parents[fold], m244_replacements[fold]) for fold in FOLDS
    }
    calibrated_outputs = {
        fold: _combine(parents[fold], calibrated_replacements[fold])
        for fold in FOLDS
    }
    scores: dict[str, object] = {}
    m244_deltas: dict[str, float] = {}
    calibrated_deltas: dict[str, float] = {}
    for fold in FOLDS:
        parent_score = _score(parents[fold])
        m244_score = _score(m244_outputs[fold])
        calibrated_score = _score(calibrated_outputs[fold])
        expected = m244_receipt["scores"][fold]["rare_event_corrected"]
        if abs(m244_score["total"] - expected["total"]) > 1e-12:
            raise RuntimeError(f"M244 {fold} reproduction score changed")
        m244_deltas[fold] = m244_score["total"] - parent_score["total"]
        calibrated_deltas[fold] = (
            calibrated_score["total"] - parent_score["total"]
        )
        scores[fold] = {
            "parent": parent_score,
            "m244": m244_score,
            "power_ensemble_calibrated": calibrated_score,
            "m244_total_delta": m244_deltas[fold],
            "calibrated_total_delta": calibrated_deltas[fold],
            "calibrated_minus_m244_total": (
                calibrated_score["total"] - m244_score["total"]
            ),
        }

    m244_q4_bootstrap = _paired_issuance_bootstrap(
        parents["q4"],
        m244_outputs["q4"],
    )
    calibrated_q4_bootstrap = _paired_issuance_bootstrap(
        parents["q4"],
        calibrated_outputs["q4"],
    )
    expected_bootstrap = m244_receipt["corrected_q4_paired_bootstrap"]
    if (
        abs(m244_q4_bootstrap["mean"] - expected_bootstrap["mean"]) > 1e-12
        or m244_q4_bootstrap["positive_fraction"]
        != expected_bootstrap["positive_fraction"]
    ):
        raise RuntimeError("M244 Q4 bootstrap reproduction changed")
    all_positive = all(delta > 0.0 for delta in calibrated_deltas.values())
    improves_worst_fold = min(calibrated_deltas.values()) > min(
        m244_deltas.values()
    )
    improves_q4_robustness = (
        calibrated_deltas["q4"] > m244_deltas["q4"]
        and calibrated_q4_bootstrap["positive_fraction"]
        > m244_q4_bootstrap["positive_fraction"]
    )
    promoted = all_positive and (improves_worst_fold or improves_q4_robustness)
    promotion = {
        "promoted": promoted,
        "all_q2_q3_q4_deltas_positive": all_positive,
        "m244_worst_fold_delta": min(m244_deltas.values()),
        "calibrated_worst_fold_delta": min(calibrated_deltas.values()),
        "improves_worst_fold": improves_worst_fold,
        "improves_q4_robustness": improves_q4_robustness,
        "rule": (
            "all calibrated fold deltas positive and either calibrated "
            "worst-fold delta strictly exceeds M244 or both Q4 delta and "
            "paired-bootstrap positive fraction strictly exceed M244"
        ),
    }

    prediction_path = OUTPUT / f"{MODEL_ID}-dev-2023-Q3.parquet"
    calibrated_outputs["q3"].assign(
        fold_id="dev-2023-Q3",
        model_id=MODEL_ID,
    )[[*METRIC_COLUMNS, "fold_id", "model_id", "data_available_kst_dtm"]].to_parquet(
        prediction_path,
        index=False,
    )
    policy = {
        "architecture": "exact_m244_with_final_power_member_EMOS_moments",
        "recipes": {
            str(group_id): asdict(recipe) for group_id, recipe in recipes.items()
        },
        "initial_history_days": INITIAL_HISTORY_DAYS,
        "calibration_block_days": CALIBRATION_BLOCK_DAYS,
        "mean_model": "group_affine_nonnegative_slope",
        "variance_model": "group_nnls_c_plus_d_raw_member_variance",
        "member_mapping": "affine_match_fitted_mean_and_variance_then_clip_0_1",
        "recency_half_life_days": HALF_LIFE_DAYS,
        "rare_event_tail_quantile": TAIL_QUANTILE,
        "rare_event_slope_bounds": [0.0, SLOPE_CAP],
        "spread_reference_quantile": REFERENCE_QUANTILE,
        "hour_or_lead_fit": False,
        "calibration_window_search": False,
        "distribution_search": False,
        "recipe_or_group_exception_search": False,
        "q1_policy_if_promoted": "retain_M245_scope_extended_q1_exactly",
    }
    receipt = {
        "schema_version": 1,
        "candidate_id": MODEL_ID,
        "state": (
            "LOCAL_POWER_ENSEMBLE_EMOS_PROMOTED_FOR_TEST_BUILD"
            if promoted
            else "LOCAL_POWER_ENSEMBLE_EMOS_REJECTED_NO_TEST_BUILD"
        ),
        "policy": policy,
        "policy_sha256": canonical_sha256(policy),
        "scores": scores,
        "diagnostics": diagnostics,
        "m244_q4_paired_bootstrap": m244_q4_bootstrap,
        "calibrated_q4_paired_bootstrap": calibrated_q4_bootstrap,
        "promotion": promotion,
        "prediction_path": str(prediction_path.relative_to(ROOT)),
        "prediction_sha256": sha256_file(prediction_path),
        "inputs": {
            "open_zip_sha256": OPEN_SHA,
            "baseline_ipynb_sha256": BASELINE_SHA,
            "distribution_parent_sha256": sha256_file(DISTRIBUTION),
            "v2_parent_oof_sha256": sha256_file(OOF),
            "m244_receipt_sha256": sha256_file(M244_RECEIPT),
        },
        "online_score": None,
        "new_2024_evaluation": False,
        "lockbox_reopened": False,
        "no_external_upload": True,
        "external_actions": [],
    }
    receipt_path = OUTPUT / f"{MODEL_ID}.json"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    print(json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
