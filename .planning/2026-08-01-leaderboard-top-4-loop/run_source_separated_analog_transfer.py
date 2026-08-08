"""Evaluate one equal-mass GFS/LDAPS source-separated analog pool."""

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
    Recipe,
    Representation,
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
from strict_dev_surface import DEV_CUTOFF, development_surface

from baram.contracts.hashing import canonical_sha256, sha256_file

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "artifacts" / "backtests" / "metric-aligned-probe"
M244_RECEIPT = OUTPUT / "M244_RARE_EVENT_CORRECTED_ANALOG_Q234.json"
M244_RECEIPT_SHA = (
    "8489f2759ce39fe4abb7c5179386cde0e1a39d22a5d07673a782b4163410a371"
)
MODEL_ID = "M247_SOURCE_SEPARATED_ANALOG_Q234"
FOLDS = ("q2", "q3", "q4")
SOURCE_MASS = {"gfs": 0.5, "ldaps": 0.5}
SOURCE_REPRESENTATION = Representation(
    "source_specific",
    "raw_delta",
    24,
    2.5,
)


def _source_features(core: list[str]) -> dict[str, list[str]]:
    gfs = [
        name
        for name in core
        if name.startswith("gfs_spatial__idw__wind")
        or name in {"phys__hub117_speed", "phys_v2__hub117_speed"}
    ]
    ldaps = [
        name for name in core if name.startswith("ldaps_spatial__idw__wind")
    ]
    if len(gfs) != 11 or len(ldaps) != 12:
        raise RuntimeError(
            f"M247 source feature contract changed: gfs={len(gfs)}, ldaps={len(ldaps)}"
        )
    if set(gfs) & set(ldaps) or HUB_FEATURE not in gfs:
        raise RuntimeError("M247 source separation changed")
    return {"gfs": gfs, "ldaps": ldaps}


def _source_pool_profile(
    surface: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    group_id: int,
    query_issuances: np.ndarray,
    recipe: Recipe,
) -> tuple[pd.DataFrame, dict[str, object]]:
    core = feature_sets["core"]
    sources = _source_features(core)
    frame, issuances, values, targets = _complete_group_days(
        surface,
        group_id,
        core,
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
            f"group {group_id} M247 days changed: "
            f"train={int(train_mask.sum())}, query={int(query_mask.sum())}"
        )
    train_values = values[train_mask]
    query_values = values[query_mask]
    train_targets = targets[train_mask]
    train_issuances = issuances[train_mask]
    query_days = issuances[query_mask]
    hub_index = core.index(HUB_FEATURE)
    train_hub = train_values[:, :, hub_index].copy()
    query_hub = query_values[:, :, hub_index].copy()
    hub_median = float(np.nanmedian(train_hub))
    train_hub = np.where(np.isfinite(train_hub), train_hub, hub_median)
    query_hub = np.where(np.isfinite(query_hub), query_hub, hub_median)
    raw_slope = _ols_slope(train_hub.reshape(-1), train_targets.reshape(-1))
    slope = float(np.clip(raw_slope, 0.0, SLOPE_CAP))
    threshold = float(np.quantile(train_hub, TAIL_QUANTILE))
    high = query_hub > threshold

    member_blocks: list[np.ndarray] = []
    weight_blocks: list[np.ndarray] = []
    neighbor_sets: dict[str, np.ndarray] = {}
    retrieval: dict[str, object] = {}
    clip_counts: dict[str, float] = {}
    for source, feature_names in sources.items():
        positions = [core.index(name) for name in feature_names]
        order, distance, source_retrieval = _distances(
            train_values[:, :, positions],
            query_values[:, :, positions],
            train_issuances,
            query_days,
            SOURCE_REPRESENTATION,
        )
        neighbor_indices = order[:, : recipe.neighbors]
        neighbor_distance = distance[:, : recipe.neighbors]
        base_weights = _kernel_weights(neighbor_distance, recipe.kernel)
        age_days = (
            query_days[:, None] - train_issuances[neighbor_indices]
        ) / np.timedelta64(1, "D")
        age_days = np.asarray(age_days, dtype=float)
        if np.any(age_days <= 0.0):
            raise RuntimeError(f"M247 {source} found a non-historical neighbor")
        weights = base_weights * np.exp2(-age_days / HALF_LIFE_DAYS)
        weights /= weights.sum(axis=1, keepdims=True)
        neighbor_targets = train_targets[neighbor_indices]
        neighbor_hub = train_hub[neighbor_indices]
        adjustment = slope * (query_hub[:, None, :] - neighbor_hub)
        corrected = np.where(
            high[:, None, :],
            neighbor_targets + adjustment,
            neighbor_targets,
        )
        applied_adjustment = np.where(high[:, None, :], adjustment, 0.0)
        clip_counts[source] = float(
            np.mean(
                (neighbor_targets + applied_adjustment < 0.0)
                | (neighbor_targets + applied_adjustment > 1.0)
            )
        )
        member_blocks.append(np.clip(corrected, 0.0, 1.0))
        weight_blocks.append(weights * SOURCE_MASS[source])
        neighbor_sets[source] = neighbor_indices
        retrieval[source] = {
            **source_retrieval,
            "feature_count": len(feature_names),
            "source_mass": SOURCE_MASS[source],
            "neighbor_age_days": {
                "min": float(np.min(age_days)),
                "median": float(np.median(age_days)),
                "max": float(np.max(age_days)),
            },
        }

    members = np.concatenate(member_blocks, axis=1)
    weights = np.concatenate(weight_blocks, axis=1)
    if not np.allclose(weights.sum(axis=1), 1.0, atol=1e-12, rtol=0.0):
        raise RuntimeError("M247 pooled source mass does not sum to one")
    heads = _profile_heads(
        members,
        weights,
        float(np.nanmean(train_targets)),
    )
    profile = _profile_frame(
        frame,
        query_days,
        heads[recipe.head],
        group_id,
    )
    gfs_neighbors = neighbor_sets["gfs"]
    ldaps_neighbors = neighbor_sets["ldaps"]
    overlap = np.asarray(
        [
            len(set(gfs_row.tolist()) & set(ldaps_row.tolist()))
            / recipe.neighbors
            for gfs_row, ldaps_row in zip(
                gfs_neighbors,
                ldaps_neighbors,
                strict=True,
            )
        ],
        dtype=float,
    )
    return profile, {
        "training_days": int(train_mask.sum()),
        "query_days": int(query_mask.sum()),
        "source_mass": SOURCE_MASS,
        "source_features": sources,
        "source_retrieval": retrieval,
        "neighbor_overlap_fraction": {
            "min": float(np.min(overlap)),
            "mean": float(np.mean(overlap)),
            "median": float(np.median(overlap)),
            "max": float(np.max(overlap)),
        },
        "hub_feature": HUB_FEATURE,
        "tail_quantile": TAIL_QUANTILE,
        "hub_threshold": threshold,
        "raw_ols_slope": raw_slope,
        "applied_slope": slope,
        "high_query_fraction": float(np.mean(high)),
        "corrected_clip_fraction_by_source": clip_counts,
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
        raise RuntimeError("lockbox row reached M247 source-separated runner")
    feature_sets = _feature_sets(numeric)
    parents = _parents(surface)
    query_issuances = {
        fold: np.sort(parent["data_available_kst_dtm"].unique())
        for fold, parent in parents.items()
    }

    m244_replacements: dict[str, list[pd.DataFrame]] = {
        fold: [] for fold in FOLDS
    }
    source_replacements: dict[str, list[pd.DataFrame]] = {
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
            source_profile, source_retrieval = _source_pool_profile(
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
            source_profile = _composed_profile(source_profile, spread_profile)
            m244 = _apply_spread_recipe(
                parent,
                m244_profile,
                group_id,
                recipe,
            )
            source = _apply_spread_recipe(
                parent,
                source_profile,
                group_id,
                recipe,
            )
            m244_replacements[fold].append(m244)
            source_replacements[fold].append(source)
            group_diagnostics[fold] = {
                "parent": _group_score(parent, group_id),
                "m244": _group_score(m244, group_id),
                "source_separated": _group_score(source, group_id),
                "m244_retrieval": m244_retrieval,
                "spread_retrieval": spread_retrieval,
                "source_retrieval": source_retrieval,
            }
        diagnostics[str(group_id)] = group_diagnostics

    m244_outputs = {
        fold: _combine(parents[fold], m244_replacements[fold]) for fold in FOLDS
    }
    source_outputs = {
        fold: _combine(parents[fold], source_replacements[fold]) for fold in FOLDS
    }
    scores: dict[str, object] = {}
    m244_deltas: dict[str, float] = {}
    source_deltas: dict[str, float] = {}
    for fold in FOLDS:
        parent_score = _score(parents[fold])
        m244_score = _score(m244_outputs[fold])
        source_score = _score(source_outputs[fold])
        expected = m244_receipt["scores"][fold]["rare_event_corrected"]
        if abs(m244_score["total"] - expected["total"]) > 1e-12:
            raise RuntimeError(f"M244 {fold} reproduction score changed")
        m244_deltas[fold] = m244_score["total"] - parent_score["total"]
        source_deltas[fold] = source_score["total"] - parent_score["total"]
        scores[fold] = {
            "parent": parent_score,
            "m244": m244_score,
            "source_separated": source_score,
            "m244_total_delta": m244_deltas[fold],
            "source_total_delta": source_deltas[fold],
            "source_minus_m244_total": source_score["total"] - m244_score["total"],
        }

    m244_q4_bootstrap = _paired_issuance_bootstrap(
        parents["q4"],
        m244_outputs["q4"],
    )
    source_q4_bootstrap = _paired_issuance_bootstrap(
        parents["q4"],
        source_outputs["q4"],
    )
    expected_bootstrap = m244_receipt["corrected_q4_paired_bootstrap"]
    if (
        abs(m244_q4_bootstrap["mean"] - expected_bootstrap["mean"]) > 1e-12
        or m244_q4_bootstrap["positive_fraction"]
        != expected_bootstrap["positive_fraction"]
    ):
        raise RuntimeError("M244 Q4 bootstrap reproduction changed")
    all_positive = all(delta > 0.0 for delta in source_deltas.values())
    improves_worst_fold = min(source_deltas.values()) > min(m244_deltas.values())
    improves_q4_robustness = (
        source_deltas["q4"] > m244_deltas["q4"]
        and source_q4_bootstrap["positive_fraction"]
        > m244_q4_bootstrap["positive_fraction"]
    )
    promoted = all_positive and (improves_worst_fold or improves_q4_robustness)
    promotion = {
        "promoted": promoted,
        "all_q2_q3_q4_deltas_positive": all_positive,
        "m244_worst_fold_delta": min(m244_deltas.values()),
        "source_worst_fold_delta": min(source_deltas.values()),
        "improves_worst_fold": improves_worst_fold,
        "improves_q4_robustness": improves_q4_robustness,
        "rule": (
            "all source-separated fold deltas positive and either source "
            "worst-fold delta strictly exceeds M244 or both Q4 delta and "
            "paired-bootstrap positive fraction strictly exceed M244"
        ),
    }

    prediction_path = OUTPUT / f"{MODEL_ID}-dev-2023-Q3.parquet"
    source_outputs["q3"].assign(
        fold_id="dev-2023-Q3",
        model_id=MODEL_ID,
    )[[*METRIC_COLUMNS, "fold_id", "model_id", "data_available_kst_dtm"]].to_parquet(
        prediction_path,
        index=False,
    )
    policy = {
        "architecture": "equal_mass_source_separated_gfs_ldaps_analog_pool",
        "recipes": {
            str(group_id): asdict(recipe) for group_id, recipe in recipes.items()
        },
        "source_mass": SOURCE_MASS,
        "source_representation": asdict(SOURCE_REPRESENTATION),
        "gfs_features": "all_9_core_GFS_IDW_wind_plus_2_GFS_hub_speeds",
        "ldaps_features": "all_12_core_LDAPS_IDW_wind_variables",
        "recency_half_life_days": HALF_LIFE_DAYS,
        "rare_event_tail_quantile": TAIL_QUANTILE,
        "rare_event_slope_bounds": [0.0, SLOPE_CAP],
        "spread_reference_quantile": REFERENCE_QUANTILE,
        "source_weight_search": False,
        "source_feature_or_neighbor_search": False,
        "full_core_mixture_search": False,
        "group_exception_search": False,
        "q1_policy_if_promoted": "retain_M245_scope_extended_q1_exactly",
    }
    receipt = {
        "schema_version": 1,
        "candidate_id": MODEL_ID,
        "state": (
            "LOCAL_SOURCE_SEPARATED_ANALOG_PROMOTED_FOR_TEST_BUILD"
            if promoted
            else "LOCAL_SOURCE_SEPARATED_ANALOG_REJECTED_NO_TEST_BUILD"
        ),
        "policy": policy,
        "policy_sha256": canonical_sha256(policy),
        "scores": scores,
        "diagnostics": diagnostics,
        "m244_q4_paired_bootstrap": m244_q4_bootstrap,
        "source_q4_paired_bootstrap": source_q4_bootstrap,
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
