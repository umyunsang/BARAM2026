"""Evaluate one fixed multivariate analog-regression correction over M243.

The runner keeps exact M243 retrieval, recency weighting, spread shrinkage,
decision heads, transforms, blend weights, and parents.  It changes only the
analog-member value: a strictly preceding group/hour ridge model translates
each historical member from its physical NWP state to the query state.  No
test submission is built by this runner.
"""

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
from run_rare_event_corrected_analog_transfer import _rare_event_profile
from run_recency_spread_analog_transfer import _composed_profile, _parents
from run_recency_weighted_analog_transfer import HALF_LIFE_DAYS, _recency_profile
from run_spread_shrunk_analog_transfer import (
    REFERENCE_QUANTILE,
    _apply_spread_recipe,
    _spread_adjusted_profile,
)
from run_stable_daily_analog_profile import DISTRIBUTION
from strict_dev_surface import DEV_CUTOFF, development_surface

from baram.contracts.hashing import canonical_sha256, sha256_file

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "artifacts" / "backtests" / "metric-aligned-probe"
M243_RECEIPT = OUTPUT / "M243_RECENCY_SPREAD_ANALOG_Q234.json"
M244_RECEIPT = OUTPUT / "M244_RARE_EVENT_CORRECTED_ANALOG_Q234.json"
M243_RECEIPT_SHA = (
    "be4a590ec53d74a27e4d9d9e280536a1f1e9af3fd73d333889b92d5e143565c0"
)
M244_RECEIPT_SHA = (
    "8489f2759ce39fe4abb7c5179386cde0e1a39d22a5d07673a782b4163410a371"
)
MODEL_ID = "M260_MULTIVARIATE_ANALOG_RIDGE_Q234"
FOLDS = ("q2", "q3", "q4")
RIDGE_PENALTY = 10.0
CORRECTION_FEATURES = (
    "phys_v2__hub117_speed",
    "phys_v2__shear_alpha_100_80",
    "phys_v2__air_density",
    "phys_v2__fleet_power_proxy_w",
)


def _ridge_coefficients(
    values: np.ndarray,
    targets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, float]]]:
    """Fit one standardized ridge model per target hour."""
    if values.ndim != 3 or values.shape[1:] != (24, len(CORRECTION_FEATURES)):
        raise RuntimeError("M260 physical tensor shape changed")
    if targets.shape != values.shape[:2]:
        raise RuntimeError("M260 ridge target shape changed")
    centers = np.empty((24, len(CORRECTION_FEATURES)), dtype=float)
    scales = np.empty_like(centers)
    intercepts = np.empty(24, dtype=float)
    slopes = np.empty_like(centers)
    diagnostics: list[dict[str, float]] = []
    for hour in range(24):
        x = np.asarray(values[:, hour, :], dtype=float)
        y = np.asarray(targets[:, hour], dtype=float)
        valid_y = np.isfinite(y)
        if int(valid_y.sum()) < 80:
            raise RuntimeError(f"M260 hour {hour} ridge support changed")
        x = x[valid_y]
        y = y[valid_y]
        center = np.nanmean(x, axis=0)
        if not np.isfinite(center).all():
            raise RuntimeError(f"M260 hour {hour} physical center is non-finite")
        x = np.where(np.isfinite(x), x, center)
        scale = np.std(x, axis=0, ddof=0)
        scale = np.where(scale > 1e-8, scale, 1.0)
        standardized = (x - center) / scale
        design = np.column_stack([np.ones(len(standardized)), standardized])
        penalty = np.diag([0.0, *([RIDGE_PENALTY] * len(CORRECTION_FEATURES))])
        coefficients = np.linalg.solve(
            design.T @ design + penalty,
            design.T @ y,
        )
        fitted = design @ coefficients
        centers[hour] = center
        scales[hour] = scale
        intercepts[hour] = coefficients[0]
        slopes[hour] = coefficients[1:]
        denominator = float(np.sum((y - float(np.mean(y))) ** 2))
        r2 = (
            1.0 - float(np.sum((y - fitted) ** 2)) / denominator
            if denominator > 1e-12
            else 0.0
        )
        diagnostics.append(
            {
                "hour": float(hour),
                "rows": float(len(y)),
                "r2": r2,
                "intercept": float(intercepts[hour]),
                "slope_l2": float(np.linalg.norm(slopes[hour])),
            }
        )
    return centers, scales, intercepts, slopes, diagnostics


def _ridge_profile(
    surface: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    group_id: int,
    query_issuances: np.ndarray,
    recipe: Recipe,
) -> tuple[pd.DataFrame, dict[str, object]]:
    representation = next(
        item for item in REPRESENTATIONS if item.name == recipe.representation
    )
    retrieval_features = feature_sets[representation.feature_set]
    missing = set(CORRECTION_FEATURES).difference(surface.columns)
    if missing:
        raise RuntimeError(f"M260 physical features are absent: {sorted(missing)}")
    frame, issuances, retrieval_values, targets = _complete_group_days(
        surface,
        group_id,
        retrieval_features,
    )
    (
        correction_frame,
        correction_issuances,
        correction_values,
        correction_targets,
    ) = _complete_group_days(surface, group_id, list(CORRECTION_FEATURES))
    if (
        not np.array_equal(issuances, correction_issuances)
        or not frame[["forecast_id", "group_id"]].equals(
            correction_frame[["forecast_id", "group_id"]]
        )
        or not np.allclose(targets, correction_targets, equal_nan=True)
    ):
        raise RuntimeError("M260 physical-day alignment changed")
    query_mask = np.isin(issuances, query_issuances)
    cutoff = pd.Timestamp(np.min(query_issuances))
    day_end = (
        frame.groupby("data_available_kst_dtm", sort=True)["forecast_kst_dtm"]
        .max()
        .reindex(pd.to_datetime(issuances))
        .to_numpy()
    )
    train_mask = (day_end < cutoff) & np.isfinite(targets).all(axis=1)
    if int(train_mask.sum()) < 80 or not query_mask.any():
        raise RuntimeError(
            f"group {group_id} M260 days changed: "
            f"train={int(train_mask.sum())}, query={int(query_mask.sum())}"
        )
    train_values = retrieval_values[train_mask]
    query_values = retrieval_values[query_mask]
    train_targets = targets[train_mask]
    train_issuances = issuances[train_mask]
    query_days = issuances[query_mask]
    order, distance, retrieval = _distances(
        train_values,
        query_values,
        train_issuances,
        query_days,
        representation,
    )
    neighbor_indices = order[:, : recipe.neighbors]
    neighbor_distance = distance[:, : recipe.neighbors]
    base_weights = _kernel_weights(neighbor_distance, recipe.kernel)
    age_days = (
        query_days[:, None] - train_issuances[neighbor_indices]
    ) / np.timedelta64(1, "D")
    age_days = np.asarray(age_days, dtype=float)
    if np.any(age_days <= 0.0):
        raise RuntimeError("M260 found a non-historical analog neighbor")
    weights = base_weights * np.exp2(-age_days / HALF_LIFE_DAYS)
    weights /= weights.sum(axis=1, keepdims=True)

    train_physical = np.asarray(correction_values[train_mask], dtype=float)
    query_physical = np.asarray(correction_values[query_mask], dtype=float)
    centers, scales, intercepts, slopes, ridge_diagnostics = _ridge_coefficients(
        train_physical,
        train_targets,
    )
    train_physical = np.where(
        np.isfinite(train_physical),
        train_physical,
        centers[None, :, :],
    )
    query_physical = np.where(
        np.isfinite(query_physical),
        query_physical,
        centers[None, :, :],
    )
    neighbor_physical = train_physical[neighbor_indices]
    standardized_difference = (
        query_physical[:, None, :, :] - neighbor_physical
    ) / scales[None, None, :, :]
    adjustment = np.einsum(
        "qkhf,hf->qkh",
        standardized_difference,
        slopes,
        optimize=True,
    )
    neighbor_targets = train_targets[neighbor_indices]
    raw_corrected = neighbor_targets + adjustment
    corrected_targets = np.clip(raw_corrected, 0.0, 1.0)
    heads = _profile_heads(
        corrected_targets,
        weights,
        float(np.nanmean(train_targets)),
    )
    profile = _profile_frame(frame, query_days, heads[recipe.head], group_id)
    return profile, {
        **retrieval,
        "training_days": int(train_mask.sum()),
        "query_days": int(query_mask.sum()),
        "correction_features": list(CORRECTION_FEATURES),
        "ridge_penalty": RIDGE_PENALTY,
        "ridge_diagnostics": ridge_diagnostics,
        "intercept_summary": {
            "min": float(np.min(intercepts)),
            "mean": float(np.mean(intercepts)),
            "max": float(np.max(intercepts)),
        },
        "adjustment_summary": {
            "min": float(np.min(adjustment)),
            "mean": float(np.mean(adjustment)),
            "max": float(np.max(adjustment)),
            "mean_absolute": float(np.mean(np.abs(adjustment))),
        },
        "corrected_clip_fraction": float(
            np.mean((raw_corrected < 0.0) | (raw_corrected > 1.0))
        ),
    }


def _self_test() -> None:
    values = np.ones((100, 24, len(CORRECTION_FEATURES)), dtype=float)
    values[:, :, 0] = np.linspace(-1.0, 1.0, 100)[:, None]
    targets = 0.5 + 0.1 * values[:, :, 0]
    centers, scales, _, slopes, _ = _ridge_coefficients(values, targets)
    identical = (values[:2, None] - values[:2, None]) / scales[None, None]
    adjustment = np.einsum("qkhf,hf->qkh", identical, slopes, optimize=True)
    if not np.array_equal(adjustment, np.zeros_like(adjustment)):
        raise RuntimeError("M260 identical-state adjustment self-test failed")
    if not np.isfinite(centers).all() or not np.isfinite(slopes).all():
        raise RuntimeError("M260 ridge finiteness self-test failed")


def main() -> None:
    _self_test()
    if sha256_file(OPEN) != OPEN_SHA or sha256_file(BASELINE) != BASELINE_SHA:
        raise RuntimeError("immutable competition input hash mismatch")
    if sha256_file(M243_RECEIPT) != M243_RECEIPT_SHA:
        raise RuntimeError("M243 promoted receipt hash mismatch")
    if sha256_file(M244_RECEIPT) != M244_RECEIPT_SHA:
        raise RuntimeError("M244 promoted receipt hash mismatch")
    m243_receipt = json.loads(M243_RECEIPT.read_text(encoding="utf-8"))
    m244_receipt = json.loads(M244_RECEIPT.read_text(encoding="utf-8"))
    recipes = {
        int(group_id): Recipe(**raw)
        for group_id, raw in m243_receipt["policy"]["recipes"].items()
    }

    surface, numeric = development_surface()
    if surface["forecast_kst_dtm"].ge(DEV_CUTOFF).any():
        raise RuntimeError("lockbox row reached M260 ridge runner")
    feature_sets = _feature_sets(numeric)
    parents = _parents(surface)
    query_issuances = {
        fold: np.sort(parent["data_available_kst_dtm"].unique())
        for fold, parent in parents.items()
    }

    m243_replacements: dict[str, list[pd.DataFrame]] = {fold: [] for fold in FOLDS}
    m244_replacements: dict[str, list[pd.DataFrame]] = {fold: [] for fold in FOLDS}
    ridge_replacements: dict[str, list[pd.DataFrame]] = {fold: [] for fold in FOLDS}
    diagnostics: dict[str, object] = {}
    for group_id, recipe in recipes.items():
        group_diagnostics: dict[str, object] = {"recipe": asdict(recipe)}
        for fold in FOLDS:
            parent = parents[fold]
            recency_profile, recency_retrieval = _recency_profile(
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
            rare_profile, rare_retrieval = _rare_event_profile(
                surface,
                feature_sets,
                group_id,
                query_issuances[fold],
                recipe,
            )
            ridge_profile, ridge_retrieval = _ridge_profile(
                surface,
                feature_sets,
                group_id,
                query_issuances[fold],
                recipe,
            )
            m243_profile = _composed_profile(recency_profile, spread_profile)
            m244_profile = _composed_profile(rare_profile, spread_profile)
            corrected_profile = _composed_profile(ridge_profile, spread_profile)
            m243 = _apply_spread_recipe(parent, m243_profile, group_id, recipe)
            m244 = _apply_spread_recipe(parent, m244_profile, group_id, recipe)
            corrected = _apply_spread_recipe(
                parent,
                corrected_profile,
                group_id,
                recipe,
            )
            m243_replacements[fold].append(m243)
            m244_replacements[fold].append(m244)
            ridge_replacements[fold].append(corrected)
            group_diagnostics[fold] = {
                "parent": _group_score(parent, group_id),
                "m243": _group_score(m243, group_id),
                "m244": _group_score(m244, group_id),
                "multivariate_ridge": _group_score(corrected, group_id),
                "recency_retrieval": recency_retrieval,
                "spread_retrieval": spread_retrieval,
                "rare_retrieval": rare_retrieval,
                "ridge_retrieval": ridge_retrieval,
            }
        diagnostics[str(group_id)] = group_diagnostics

    m243_outputs = {
        fold: _combine(parents[fold], m243_replacements[fold]) for fold in FOLDS
    }
    m244_outputs = {
        fold: _combine(parents[fold], m244_replacements[fold]) for fold in FOLDS
    }
    ridge_outputs = {
        fold: _combine(parents[fold], ridge_replacements[fold]) for fold in FOLDS
    }
    scores: dict[str, object] = {}
    m244_deltas: dict[str, float] = {}
    ridge_deltas: dict[str, float] = {}
    for fold in FOLDS:
        parent_score = _score(parents[fold])
        m243_score = _score(m243_outputs[fold])
        m244_score = _score(m244_outputs[fold])
        ridge_score = _score(ridge_outputs[fold])
        expected_m243 = m243_receipt["scores"][fold]["recency_plus_spread"]
        expected_m244 = m244_receipt["scores"][fold]["rare_event_corrected"]
        if abs(m243_score["total"] - expected_m243["total"]) > 1e-12:
            raise RuntimeError(f"M243 {fold} reproduction score changed")
        if abs(m244_score["total"] - expected_m244["total"]) > 1e-12:
            raise RuntimeError(f"M244 {fold} reproduction score changed")
        m244_deltas[fold] = m244_score["total"] - parent_score["total"]
        ridge_deltas[fold] = ridge_score["total"] - parent_score["total"]
        scores[fold] = {
            "parent": parent_score,
            "m243": m243_score,
            "m244": m244_score,
            "multivariate_ridge": ridge_score,
            "m244_total_delta": m244_deltas[fold],
            "ridge_total_delta": ridge_deltas[fold],
            "ridge_minus_m244_total": ridge_score["total"] - m244_score["total"],
        }

    m244_q4_bootstrap = _paired_issuance_bootstrap(parents["q4"], m244_outputs["q4"])
    ridge_q4_bootstrap = _paired_issuance_bootstrap(parents["q4"], ridge_outputs["q4"])
    expected_bootstrap = m244_receipt["corrected_q4_paired_bootstrap"]
    if (
        abs(m244_q4_bootstrap["mean"] - expected_bootstrap["mean"]) > 1e-12
        or m244_q4_bootstrap["positive_fraction"]
        != expected_bootstrap["positive_fraction"]
    ):
        raise RuntimeError("M244 Q4 bootstrap reproduction changed")
    all_positive = all(delta > 0.0 for delta in ridge_deltas.values())
    improves_worst_fold = min(ridge_deltas.values()) > min(m244_deltas.values())
    improves_q4_robustness = (
        ridge_deltas["q4"] > m244_deltas["q4"]
        and ridge_q4_bootstrap["positive_fraction"]
        > m244_q4_bootstrap["positive_fraction"]
    )
    promoted = all_positive and (improves_worst_fold or improves_q4_robustness)
    promotion = {
        "promoted": promoted,
        "all_q2_q3_q4_deltas_positive": all_positive,
        "m244_worst_fold_delta": min(m244_deltas.values()),
        "ridge_worst_fold_delta": min(ridge_deltas.values()),
        "improves_worst_fold": improves_worst_fold,
        "improves_q4_robustness": improves_q4_robustness,
        "rule": (
            "all ridge-corrected fold deltas positive and either its worst-fold "
            "delta strictly exceeds M244 or both Q4 delta and paired-bootstrap "
            "positive fraction strictly exceed M244"
        ),
    }

    prediction_path = OUTPUT / f"{MODEL_ID}-dev-2023-Q3.parquet"
    ridge_outputs["q3"].assign(
        fold_id="dev-2023-Q3",
        model_id=MODEL_ID,
    )[[*METRIC_COLUMNS, "fold_id", "model_id", "data_available_kst_dtm"]].to_parquet(
        prediction_path,
        index=False,
    )
    policy = {
        "architecture": "exact_m243_with_group_hour_multivariate_analog_ridge",
        "recipes": {
            str(group_id): asdict(recipe) for group_id, recipe in recipes.items()
        },
        "correction_features": list(CORRECTION_FEATURES),
        "ridge_penalty": RIDGE_PENALTY,
        "ridge_scope": "strictly_preceding_complete_days_by_group_and_hour",
        "member_adjustment": "standardized_beta_dot_query_minus_neighbor_state",
        "member_clip": [0.0, 1.0],
        "recency_half_life_days": HALF_LIFE_DAYS,
        "spread_reference_quantile": REFERENCE_QUANTILE,
        "feature_search": False,
        "ridge_search": False,
        "hour_pooling_search": False,
        "coefficient_cap_search": False,
        "correction_blend_search": False,
        "group_exception_search": False,
        "q1_policy_if_promoted": "unconditional M231 fallback",
    }
    receipt = {
        "schema_version": 1,
        "candidate_id": MODEL_ID,
        "state": (
            "LOCAL_MULTIVARIATE_ANALOG_RIDGE_PROMOTED_FOR_TEST_BUILD"
            if promoted
            else "LOCAL_MULTIVARIATE_ANALOG_RIDGE_REJECTED_NO_TEST_BUILD"
        ),
        "policy": policy,
        "policy_sha256": canonical_sha256(policy),
        "scores": scores,
        "diagnostics": diagnostics,
        "m244_q4_paired_bootstrap": m244_q4_bootstrap,
        "ridge_q4_paired_bootstrap": ridge_q4_bootstrap,
        "promotion": promotion,
        "prediction_path": str(prediction_path.relative_to(ROOT)),
        "prediction_sha256": sha256_file(prediction_path),
        "inputs": {
            "open_zip_sha256": OPEN_SHA,
            "baseline_ipynb_sha256": BASELINE_SHA,
            "distribution_parent_sha256": sha256_file(DISTRIBUTION),
            "v2_parent_oof_sha256": sha256_file(OOF),
            "m243_receipt_sha256": sha256_file(M243_RECEIPT),
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
