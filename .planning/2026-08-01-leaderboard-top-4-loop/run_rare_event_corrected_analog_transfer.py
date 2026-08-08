"""Evaluate one fixed rare-event regression correction over M243 analogs."""

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
M243_RECEIPT_SHA = (
    "be4a590ec53d74a27e4d9d9e280536a1f1e9af3fd73d333889b92d5e143565c0"
)
MODEL_ID = "M244_RARE_EVENT_CORRECTED_ANALOG_Q234"
FOLDS = ("q2", "q3", "q4")
HUB_FEATURE = "phys_v2__hub117_speed"
TAIL_QUANTILE = 0.90
SLOPE_CAP = 0.20


def _ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) < 100:
        raise RuntimeError("M244 rare-event slope support changed")
    centered_x = x - float(np.mean(x))
    denominator = float(centered_x @ centered_x)
    if denominator <= 1e-8:
        raise RuntimeError("M244 rare-event hub-speed variance collapsed")
    return float(centered_x @ (y - float(np.mean(y))) / denominator)


def _rare_event_profile(
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
        raise RuntimeError("M244 frozen hub-speed feature is absent")
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
    if int(train_mask.sum()) < 80 or not query_mask.any():
        raise RuntimeError(
            f"group {group_id} rare-event days changed: "
            f"train={int(train_mask.sum())}, query={int(query_mask.sum())}"
        )
    train_values = values[train_mask]
    query_values = values[query_mask]
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
        raise RuntimeError("M244 found a non-historical analog neighbor")
    weights = base_weights * np.exp2(-age_days / HALF_LIFE_DAYS)
    weights /= weights.sum(axis=1, keepdims=True)

    train_hub = train_values[:, :, hub_index].copy()
    query_hub = query_values[:, :, hub_index].copy()
    hub_median = float(np.nanmedian(train_hub))
    train_hub = np.where(np.isfinite(train_hub), train_hub, hub_median)
    query_hub = np.where(np.isfinite(query_hub), query_hub, hub_median)
    raw_slope = _ols_slope(train_hub.reshape(-1), train_targets.reshape(-1))
    slope = float(np.clip(raw_slope, 0.0, SLOPE_CAP))
    threshold = float(np.quantile(train_hub, TAIL_QUANTILE))
    high = query_hub > threshold
    neighbor_hub = train_hub[neighbor_indices]
    neighbor_targets = train_targets[neighbor_indices]
    adjustment = slope * (query_hub[:, None, :] - neighbor_hub)
    corrected_targets = np.where(
        high[:, None, :],
        neighbor_targets + adjustment,
        neighbor_targets,
    )
    corrected_targets = np.clip(corrected_targets, 0.0, 1.0)
    heads = _profile_heads(
        corrected_targets,
        weights,
        float(np.nanmean(train_targets)),
    )
    profile = _profile_frame(
        frame,
        query_days,
        heads[recipe.head],
        group_id,
    )
    applied_adjustment = np.where(high[:, None, :], adjustment, 0.0)
    return profile, {
        **retrieval,
        "training_days": int(train_mask.sum()),
        "query_days": int(query_mask.sum()),
        "hub_feature": HUB_FEATURE,
        "tail_quantile": TAIL_QUANTILE,
        "hub_threshold": threshold,
        "raw_ols_slope": raw_slope,
        "applied_slope": slope,
        "slope_cap": SLOPE_CAP,
        "high_query_fraction": float(np.mean(high)),
        "adjustment_summary": {
            "min": float(np.min(applied_adjustment)),
            "mean": float(np.mean(applied_adjustment)),
            "max": float(np.max(applied_adjustment)),
            "mean_absolute": float(np.mean(np.abs(applied_adjustment))),
        },
        "corrected_clip_fraction": float(
            np.mean(
                (neighbor_targets + applied_adjustment < 0.0)
                | (neighbor_targets + applied_adjustment > 1.0)
            )
        ),
    }


def main() -> None:
    if sha256_file(OPEN) != OPEN_SHA or sha256_file(BASELINE) != BASELINE_SHA:
        raise RuntimeError("immutable competition input hash mismatch")
    if sha256_file(M243_RECEIPT) != M243_RECEIPT_SHA:
        raise RuntimeError("M243 promoted receipt hash mismatch")
    m243_receipt = json.loads(M243_RECEIPT.read_text(encoding="utf-8"))
    recipes = {
        int(group_id): Recipe(**raw)
        for group_id, raw in m243_receipt["policy"]["recipes"].items()
    }

    surface, numeric = development_surface()
    if surface["forecast_kst_dtm"].ge(DEV_CUTOFF).any():
        raise RuntimeError("lockbox row reached M244 rare-event runner")
    feature_sets = _feature_sets(numeric)
    parents = _parents(surface)
    query_issuances = {
        fold: np.sort(parent["data_available_kst_dtm"].unique())
        for fold, parent in parents.items()
    }

    m243_replacements: dict[str, list[pd.DataFrame]] = {
        fold: [] for fold in FOLDS
    }
    corrected_replacements: dict[str, list[pd.DataFrame]] = {
        fold: [] for fold in FOLDS
    }
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
            corrected_profile, corrected_retrieval = _rare_event_profile(
                surface,
                feature_sets,
                group_id,
                query_issuances[fold],
                recipe,
            )
            m243_profile = _composed_profile(recency_profile, spread_profile)
            corrected_profile = _composed_profile(
                corrected_profile,
                spread_profile,
            )
            m243 = _apply_spread_recipe(parent, m243_profile, group_id, recipe)
            corrected = _apply_spread_recipe(
                parent,
                corrected_profile,
                group_id,
                recipe,
            )
            m243_replacements[fold].append(m243)
            corrected_replacements[fold].append(corrected)
            group_diagnostics[fold] = {
                "parent": _group_score(parent, group_id),
                "m243": _group_score(m243, group_id),
                "rare_event_corrected": _group_score(corrected, group_id),
                "recency_retrieval": recency_retrieval,
                "spread_retrieval": spread_retrieval,
                "corrected_retrieval": corrected_retrieval,
            }
        diagnostics[str(group_id)] = group_diagnostics

    m243_outputs = {
        fold: _combine(parents[fold], m243_replacements[fold]) for fold in FOLDS
    }
    corrected_outputs = {
        fold: _combine(parents[fold], corrected_replacements[fold]) for fold in FOLDS
    }
    scores: dict[str, object] = {}
    m243_deltas: dict[str, float] = {}
    corrected_deltas: dict[str, float] = {}
    for fold in FOLDS:
        parent_score = _score(parents[fold])
        m243_score = _score(m243_outputs[fold])
        corrected_score = _score(corrected_outputs[fold])
        expected = m243_receipt["scores"][fold]["recency_plus_spread"]
        if abs(m243_score["total"] - expected["total"]) > 1e-12:
            raise RuntimeError(f"M243 {fold} reproduction score changed")
        m243_deltas[fold] = m243_score["total"] - parent_score["total"]
        corrected_deltas[fold] = corrected_score["total"] - parent_score["total"]
        scores[fold] = {
            "parent": parent_score,
            "m243": m243_score,
            "rare_event_corrected": corrected_score,
            "m243_total_delta": m243_deltas[fold],
            "corrected_total_delta": corrected_deltas[fold],
            "corrected_minus_m243_total": (
                corrected_score["total"] - m243_score["total"]
            ),
        }

    m243_q4_bootstrap = _paired_issuance_bootstrap(
        parents["q4"], m243_outputs["q4"]
    )
    corrected_q4_bootstrap = _paired_issuance_bootstrap(
        parents["q4"], corrected_outputs["q4"]
    )
    expected_bootstrap = m243_receipt["composed_q4_paired_bootstrap"]
    if (
        abs(m243_q4_bootstrap["mean"] - expected_bootstrap["mean"]) > 1e-12
        or m243_q4_bootstrap["positive_fraction"]
        != expected_bootstrap["positive_fraction"]
    ):
        raise RuntimeError("M243 Q4 bootstrap reproduction changed")
    all_positive = all(delta > 0.0 for delta in corrected_deltas.values())
    improves_worst_fold = min(corrected_deltas.values()) > min(m243_deltas.values())
    improves_q4_robustness = (
        corrected_deltas["q4"] > m243_deltas["q4"]
        and corrected_q4_bootstrap["positive_fraction"]
        > m243_q4_bootstrap["positive_fraction"]
    )
    promoted = all_positive and (improves_worst_fold or improves_q4_robustness)
    promotion = {
        "promoted": promoted,
        "all_q2_q3_q4_deltas_positive": all_positive,
        "m243_worst_fold_delta": min(m243_deltas.values()),
        "corrected_worst_fold_delta": min(corrected_deltas.values()),
        "improves_worst_fold": improves_worst_fold,
        "improves_q4_robustness": improves_q4_robustness,
        "rule": (
            "all corrected fold deltas positive and either corrected worst-fold "
            "delta strictly exceeds M243 or both Q4 delta and paired-bootstrap "
            "positive fraction strictly exceed M243"
        ),
    }

    prediction_path = OUTPUT / f"{MODEL_ID}-dev-2023-Q3.parquet"
    corrected_outputs["q3"].assign(
        fold_id="dev-2023-Q3",
        model_id=MODEL_ID,
    )[[*METRIC_COLUMNS, "fold_id", "model_id", "data_available_kst_dtm"]].to_parquet(
        prediction_path,
        index=False,
    )
    # M270 additive output only. Q2 and Q4 are already computed above; persisting them
    # lets the M244/M245 policy be scored pooled, which is the local counterpart of the
    # already-uploaded M252. Written to separate paths so the archived Q3 artifact and its
    # hash stay the reproduction reference.
    for _extra in ("q2", "q4"):
        corrected_outputs[_extra].assign(
            fold_id=f"dev-2023-{_extra.upper()}",
            model_id=MODEL_ID,
        )[[*METRIC_COLUMNS, "fold_id", "model_id", "data_available_kst_dtm"]].to_parquet(
            OUTPUT / f"{MODEL_ID}-dev-2023-{_extra.upper()}.parquet",
            index=False,
        )
    policy = {
        "architecture": "exact_m243_with_train_only_rare_event_analog_mos",
        "recipes": {
            str(group_id): asdict(recipe) for group_id, recipe in recipes.items()
        },
        "hub_feature": HUB_FEATURE,
        "tail_quantile": TAIL_QUANTILE,
        "slope_fit": "one_group_ols_normalized_generation_on_hub_speed",
        "slope_bounds": [0.0, SLOPE_CAP],
        "member_adjustment": "slope_times_query_minus_neighbor_hub_speed",
        "recency_half_life_days": HALF_LIFE_DAYS,
        "spread_reference_quantile": REFERENCE_QUANTILE,
        "threshold_search": False,
        "proxy_search": False,
        "slope_cap_search": False,
        "lead_slope_search": False,
        "group_exception_search": False,
        "q1_policy_if_promoted": "unconditional M231 fallback",
    }
    receipt = {
        "schema_version": 1,
        "candidate_id": MODEL_ID,
        "state": (
            "LOCAL_RARE_EVENT_ANALOG_MOS_PROMOTED_FOR_TEST_BUILD"
            if promoted
            else "LOCAL_RARE_EVENT_ANALOG_MOS_REJECTED_NO_TEST_BUILD"
        ),
        "policy": policy,
        "policy_sha256": canonical_sha256(policy),
        "scores": scores,
        "diagnostics": diagnostics,
        "m243_q4_paired_bootstrap": m243_q4_bootstrap,
        "corrected_q4_paired_bootstrap": corrected_q4_bootstrap,
        "promotion": promotion,
        "prediction_path": str(prediction_path.relative_to(ROOT)),
        "prediction_sha256": sha256_file(prediction_path),
        "inputs": {
            "open_zip_sha256": OPEN_SHA,
            "baseline_ipynb_sha256": BASELINE_SHA,
            "distribution_parent_sha256": sha256_file(DISTRIBUTION),
            "v2_parent_oof_sha256": sha256_file(OOF),
            "m243_receipt_sha256": sha256_file(M243_RECEIPT),
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
