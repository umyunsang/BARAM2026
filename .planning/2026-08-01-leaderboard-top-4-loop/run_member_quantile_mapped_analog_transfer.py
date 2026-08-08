"""Evaluate one rank-wise empirical quantile map over exact M244 members."""

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
    _feature_sets,
    _profile_frame,
    _profile_heads,
)
from run_power_ensemble_emos_transfer import (
    CALIBRATION_BLOCK_DAYS,
    INITIAL_HISTORY_DAYS,
    MIN_CALIBRATION_ROWS,
    _corrected_members,
)
from run_rare_event_corrected_analog_transfer import (
    HUB_FEATURE,
    SLOPE_CAP,
    TAIL_QUANTILE,
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
from strict_dev_surface import DEV_CUTOFF, development_surface

from baram.contracts.hashing import canonical_sha256, sha256_file

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "artifacts" / "backtests" / "metric-aligned-probe"
M244_RECEIPT = OUTPUT / "M244_RARE_EVENT_CORRECTED_ANALOG_Q234.json"
M244_RECEIPT_SHA = (
    "8489f2759ce39fe4abb7c5179386cde0e1a39d22a5d07673a782b4163410a371"
)
MODEL_ID = "M250_MEMBER_QUANTILE_MAPPED_ANALOG_Q234"
FOLDS = ("q2", "q3", "q4")


def _prequential_member_pairs(
    train_values: np.ndarray,
    train_targets: np.ndarray,
    train_issuances: np.ndarray,
    representation: object,
    recipe: Recipe,
    hub_index: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    if len(train_issuances) <= INITIAL_HISTORY_DAYS:
        raise RuntimeError("M250 calibration history is too short")
    members: list[np.ndarray] = []
    actuals: list[np.ndarray] = []
    blocks: list[dict[str, object]] = []
    for start in range(
        INITIAL_HISTORY_DAYS,
        len(train_issuances),
        CALIBRATION_BLOCK_DAYS,
    ):
        stop = min(start + CALIBRATION_BLOCK_DAYS, len(train_issuances))
        block_members, _, diagnostics = _corrected_members(
            train_values[:start],
            train_targets[:start],
            train_issuances[:start],
            train_values[start:stop],
            train_issuances[start:stop],
            representation,
            recipe,
            hub_index,
        )
        members.append(block_members)
        actuals.append(train_targets[start:stop])
        blocks.append(
            {
                **diagnostics,
                "first_query_issuance": str(train_issuances[start]),
                "last_query_issuance": str(train_issuances[stop - 1]),
            }
        )
    member_array = np.concatenate(members, axis=0)
    actual_array = np.concatenate(actuals, axis=0)
    if (
        member_array.shape[0] != actual_array.shape[0]
        or member_array.shape[1] != recipe.neighbors
        or member_array.shape[2] != 24
        or member_array.size < MIN_CALIBRATION_ROWS
        or not np.isfinite(member_array).all()
        or not np.isfinite(actual_array).all()
    ):
        raise RuntimeError("M250 prequential member-pair support changed")
    return member_array, actual_array, {
        "initial_history_days": INITIAL_HISTORY_DAYS,
        "block_days": CALIBRATION_BLOCK_DAYS,
        "calibration_days": len(member_array),
        "calibration_rows_per_rank": actual_array.size,
        "neighbor_ranks": recipe.neighbors,
        "blocks": blocks,
    }


def _empirical_quantile_map(
    calibration_members: np.ndarray,
    calibration_actual: np.ndarray,
    query_members: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    if calibration_members.shape[1] != query_members.shape[1]:
        raise RuntimeError("M250 rank count changed between calibration and query")
    actual_sorted = np.sort(calibration_actual.reshape(-1))
    sample_count = len(actual_sorted)
    if sample_count < MIN_CALIBRATION_ROWS:
        raise RuntimeError("M250 empirical target CDF support changed")
    mapped = np.empty_like(query_members, dtype=float)
    rank_diagnostics: list[dict[str, float | int]] = []
    for rank in range(query_members.shape[1]):
        member_sorted = np.sort(calibration_members[:, rank, :].reshape(-1))
        if len(member_sorted) != sample_count:
            raise RuntimeError("M250 member/actual empirical CDF lengths differ")
        query = query_members[:, rank, :]
        right = np.searchsorted(member_sorted, query, side="right")
        probability = np.clip(
            (right.astype(float) - 0.5) / sample_count,
            0.0,
            1.0,
        )
        position = probability * (sample_count - 1)
        lower = np.floor(position).astype(int)
        upper = np.ceil(position).astype(int)
        fraction = position - lower
        transformed = (
            (1.0 - fraction) * actual_sorted[lower]
            + fraction * actual_sorted[upper]
        )
        transformed = np.clip(transformed, 0.0, 1.0)
        mapped[:, rank, :] = transformed
        difference = transformed - query
        rank_diagnostics.append(
            {
                "rank": rank + 1,
                "calibration_member_mean": float(np.mean(member_sorted)),
                "calibration_actual_mean": float(np.mean(actual_sorted)),
                "query_mean": float(np.mean(query)),
                "mapped_mean": float(np.mean(transformed)),
                "mean_correction": float(np.mean(difference)),
                "mean_absolute_correction": float(np.mean(np.abs(difference))),
                "max_absolute_correction": float(np.max(np.abs(difference))),
            }
        )
    difference = mapped - query_members
    return mapped, {
        "method": "exact_empirical_CDF_member_rank_mapping",
        "calibration_samples_per_rank": sample_count,
        "rank_diagnostics": rank_diagnostics,
        "overall_correction": {
            "mean": float(np.mean(difference)),
            "mean_absolute": float(np.mean(np.abs(difference))),
            "max_absolute": float(np.max(np.abs(difference))),
            "changed_fraction": float(np.mean(difference != 0.0)),
        },
    }


def _quantile_mapped_profile(
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
        raise RuntimeError("M250 frozen hub-speed feature is absent")
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
            f"group {group_id} M250 days changed: "
            f"train={int(train_mask.sum())}, query={int(query_mask.sum())}"
        )
    train_values = values[train_mask]
    train_targets = targets[train_mask]
    train_issuances = issuances[train_mask]
    query_values = values[query_mask]
    query_days = issuances[query_mask]

    calibration_members, calibration_actual, calibration_diagnostics = (
        _prequential_member_pairs(
            train_values,
            train_targets,
            train_issuances,
            representation,
            recipe,
            hub_index,
        )
    )
    query_members, query_weights, query_diagnostics = _corrected_members(
        train_values,
        train_targets,
        train_issuances,
        query_values,
        query_days,
        representation,
        recipe,
        hub_index,
    )
    mapped_members, mapping_diagnostics = _empirical_quantile_map(
        calibration_members,
        calibration_actual,
        query_members,
    )
    heads = _profile_heads(
        mapped_members,
        query_weights,
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
        "calibration": calibration_diagnostics,
        "query": query_diagnostics,
        "mapping": mapping_diagnostics,
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
        raise RuntimeError("lockbox row reached M250 quantile-map runner")
    feature_sets = _feature_sets(numeric)
    parents = _parents(surface)
    query_issuances = {
        fold: np.sort(parent["data_available_kst_dtm"].unique())
        for fold, parent in parents.items()
    }

    m244_replacements: dict[str, list[pd.DataFrame]] = {
        fold: [] for fold in FOLDS
    }
    mapped_replacements: dict[str, list[pd.DataFrame]] = {
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
            mapped_profile, mapped_retrieval = _quantile_mapped_profile(
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
            mapped_profile = _composed_profile(mapped_profile, spread_profile)
            m244 = _apply_spread_recipe(parent, m244_profile, group_id, recipe)
            mapped = _apply_spread_recipe(parent, mapped_profile, group_id, recipe)
            m244_replacements[fold].append(m244)
            mapped_replacements[fold].append(mapped)
            group_diagnostics[fold] = {
                "parent": _group_score(parent, group_id),
                "m244": _group_score(m244, group_id),
                "member_quantile_mapped": _group_score(mapped, group_id),
                "m244_retrieval": m244_retrieval,
                "mapped_retrieval": mapped_retrieval,
                "spread_retrieval": spread_retrieval,
            }
        diagnostics[str(group_id)] = group_diagnostics

    m244_outputs = {
        fold: _combine(parents[fold], m244_replacements[fold]) for fold in FOLDS
    }
    mapped_outputs = {
        fold: _combine(parents[fold], mapped_replacements[fold]) for fold in FOLDS
    }
    scores: dict[str, object] = {}
    m244_deltas: dict[str, float] = {}
    mapped_deltas: dict[str, float] = {}
    for fold in FOLDS:
        parent_score = _score(parents[fold])
        m244_score = _score(m244_outputs[fold])
        mapped_score = _score(mapped_outputs[fold])
        expected = m244_receipt["scores"][fold]["rare_event_corrected"]
        if abs(m244_score["total"] - expected["total"]) > 1e-12:
            raise RuntimeError(f"M244 {fold} reproduction score changed")
        m244_deltas[fold] = m244_score["total"] - parent_score["total"]
        mapped_deltas[fold] = mapped_score["total"] - parent_score["total"]
        scores[fold] = {
            "parent": parent_score,
            "m244": m244_score,
            "member_quantile_mapped": mapped_score,
            "m244_total_delta": m244_deltas[fold],
            "mapped_total_delta": mapped_deltas[fold],
            "mapped_minus_m244_total": mapped_score["total"] - m244_score["total"],
        }

    m244_q4_bootstrap = _paired_issuance_bootstrap(
        parents["q4"], m244_outputs["q4"]
    )
    mapped_q4_bootstrap = _paired_issuance_bootstrap(
        parents["q4"], mapped_outputs["q4"]
    )
    expected_bootstrap = m244_receipt["corrected_q4_paired_bootstrap"]
    if (
        abs(m244_q4_bootstrap["mean"] - expected_bootstrap["mean"]) > 1e-12
        or m244_q4_bootstrap["positive_fraction"]
        != expected_bootstrap["positive_fraction"]
    ):
        raise RuntimeError("M244 Q4 bootstrap reproduction changed")
    all_positive = all(delta > 0.0 for delta in mapped_deltas.values())
    improves_worst_fold = min(mapped_deltas.values()) > min(m244_deltas.values())
    improves_q4_robustness = (
        mapped_deltas["q4"] > m244_deltas["q4"]
        and mapped_q4_bootstrap["positive_fraction"]
        > m244_q4_bootstrap["positive_fraction"]
    )
    promoted = all_positive and (improves_worst_fold or improves_q4_robustness)
    promotion = {
        "promoted": promoted,
        "all_q2_q3_q4_deltas_positive": all_positive,
        "m244_worst_fold_delta": min(m244_deltas.values()),
        "mapped_worst_fold_delta": min(mapped_deltas.values()),
        "improves_worst_fold": improves_worst_fold,
        "improves_q4_robustness": improves_q4_robustness,
        "rule": (
            "all mapped fold deltas positive and either mapped worst-fold delta "
            "strictly exceeds M244 or both Q4 delta and paired-bootstrap positive "
            "fraction strictly exceed M244"
        ),
    }

    prediction_path = OUTPUT / f"{MODEL_ID}-dev-2023-Q3.parquet"
    mapped_outputs["q3"].assign(
        fold_id="dev-2023-Q3",
        model_id=MODEL_ID,
    )[[*METRIC_COLUMNS, "fold_id", "model_id", "data_available_kst_dtm"]].to_parquet(
        prediction_path,
        index=False,
    )
    policy = {
        "architecture": "exact_m244_with_member_rank_empirical_quantile_mapping",
        "recipes": {
            str(group_id): asdict(recipe) for group_id, recipe in recipes.items()
        },
        "initial_history_days": INITIAL_HISTORY_DAYS,
        "calibration_block_days": CALIBRATION_BLOCK_DAYS,
        "mapping_scope": "group_by_analog_distance_rank_all_24_leads_pooled",
        "mapping": "exact_sorted_empirical_CDF_to_actual_empirical_CDF",
        "physical_clip": [0.0, 1.0],
        "recency_half_life_days": HALF_LIFE_DAYS,
        "rare_event_hub_feature": HUB_FEATURE,
        "rare_event_tail_quantile": TAIL_QUANTILE,
        "rare_event_slope_bounds": [0.0, SLOPE_CAP],
        "spread_reference_quantile": REFERENCE_QUANTILE,
        "probability_grid_search": False,
        "lead_or_hour_mapping_search": False,
        "correction_cap_search": False,
        "constraint_search": False,
        "shrinkage_search": False,
        "rank_pooling_search": False,
        "group_exception_search": False,
        "q1_policy_if_promoted": "unconditional M231 fallback pending scope audit",
    }
    receipt = {
        "schema_version": 1,
        "candidate_id": MODEL_ID,
        "state": (
            "LOCAL_MEMBER_QUANTILE_MAP_PROMOTED_FOR_SCOPE_AUDIT"
            if promoted
            else "LOCAL_MEMBER_QUANTILE_MAP_REJECTED_NO_TEST_BUILD"
        ),
        "policy": policy,
        "policy_sha256": canonical_sha256(policy),
        "scores": scores,
        "diagnostics": diagnostics,
        "m244_q4_paired_bootstrap": m244_q4_bootstrap,
        "mapped_q4_paired_bootstrap": mapped_q4_bootstrap,
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
