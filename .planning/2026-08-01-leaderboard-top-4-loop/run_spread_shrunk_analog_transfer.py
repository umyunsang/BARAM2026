"""Evaluate one frozen analog-target-spread shrinkage over M234.

The exact M234 analogs and profile heads are retained.  Only the effective
blend is reduced when the retrieved historical target ensemble is more
dispersed than an hour-specific, train-day leave-one-out reference.  The 2024
lockbox is never loaded or scored.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from build_v2_transfer_sequence_challenger import (
    CAPACITIES,
    EXPECTED_SELECTIONS,
    METRIC_COLUMNS,
    OOF,
    _apply_long,
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
    _apply_recipe,
    _combine,
    _complete_group_days,
    _distances,
    _feature_sets,
    _kernel_weights,
    _profile_frame,
    _profile_heads,
    _selected_profile,
)
from run_stable_daily_analog_profile import DISTRIBUTION, _fold_parent
from strict_dev_surface import DEV_CUTOFF, development_surface

from baram.contracts.hashing import canonical_sha256, sha256_file

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "artifacts" / "backtests" / "metric-aligned-probe"
M234 = ROOT / "artifacts" / "submissions" / "E0_ROBUST_ANALOG_DEV-08f07b6d9e87.csv"
M234_RECEIPT = M234.with_suffix(".receipt.json")
MODEL_ID = "M240_SPREAD_SHRUNK_ANALOG_Q234"
FOLDS = ("q2", "q3", "q4")
REFERENCE_QUANTILE = 0.75


def _without_self_neighbors(
    order: np.ndarray,
    distance: np.ndarray,
    neighbors: int,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.empty((len(order), neighbors), dtype=int)
    distances = np.empty((len(order), neighbors), dtype=float)
    for row in range(len(order)):
        keep = order[row] != row
        if int(np.sum(keep)) < neighbors:
            raise RuntimeError("M240 leave-one-out neighbor contract changed")
        indices[row] = order[row, keep][:neighbors]
        distances[row] = distance[row, keep][:neighbors]
    return indices, distances


def _weighted_spread(
    neighbor_targets: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    mean = np.sum(neighbor_targets * weights[:, :, None], axis=1)
    variance = np.sum(
        weights[:, :, None] * (neighbor_targets - mean[:, None, :]) ** 2,
        axis=1,
    )
    return np.sqrt(np.maximum(variance, 0.0))


def _spread_adjusted_profile(
    surface: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    group_id: int,
    query_issuances: np.ndarray,
    recipe: Recipe,
) -> tuple[pd.DataFrame, dict[str, object]]:
    representation = next(
        item for item in REPRESENTATIONS if item.name == recipe.representation
    )
    frame, issuances, values, targets = _complete_group_days(
        surface,
        group_id,
        feature_sets[representation.feature_set],
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
            f"group {group_id} spread days changed: "
            f"train={int(train_mask.sum())}, query={int(query_mask.sum())}"
        )
    train_values = values[train_mask]
    train_targets = targets[train_mask]
    train_issuances = issuances[train_mask]
    query_values = values[query_mask]
    query_days = issuances[query_mask]

    query_order, query_distance, query_retrieval = _distances(
        train_values,
        query_values,
        train_issuances,
        query_days,
        representation,
    )
    query_indices = query_order[:, : recipe.neighbors]
    query_distances = query_distance[:, : recipe.neighbors]
    query_weights = _kernel_weights(query_distances, recipe.kernel)
    query_neighbor_targets = train_targets[query_indices]
    heads = _profile_heads(
        query_neighbor_targets,
        query_weights,
        float(np.nanmean(train_targets)),
    )

    loo_order, loo_distance, loo_retrieval = _distances(
        train_values,
        train_values,
        train_issuances,
        train_issuances,
        representation,
    )
    loo_indices, loo_distances = _without_self_neighbors(
        loo_order,
        loo_distance,
        recipe.neighbors,
    )
    loo_weights = _kernel_weights(loo_distances, recipe.kernel)
    loo_spread = _weighted_spread(train_targets[loo_indices], loo_weights)
    reference = np.quantile(loo_spread, REFERENCE_QUANTILE, axis=0)
    reference = np.maximum(reference, 1e-6)
    query_spread = _weighted_spread(query_neighbor_targets, query_weights)
    multiplier = np.minimum(1.0, reference[None, :] / np.maximum(query_spread, 1e-8))

    profile = _profile_frame(
        frame,
        query_days,
        heads[recipe.head],
        group_id,
    )
    if len(profile) != multiplier.size:
        raise RuntimeError("M240 spread multiplier alignment changed")
    profile["analog_blend_multiplier"] = multiplier.reshape(-1)
    return profile, {
        **query_retrieval,
        "training_days": int(train_mask.sum()),
        "query_days": int(query_mask.sum()),
        "reference_quantile": REFERENCE_QUANTILE,
        "reference_spread_by_hour": [float(value) for value in reference],
        "reference_spread_summary": {
            "min": float(np.min(reference)),
            "median": float(np.median(reference)),
            "max": float(np.max(reference)),
        },
        "query_spread_summary": {
            "min": float(np.min(query_spread)),
            "median": float(np.median(query_spread)),
            "max": float(np.max(query_spread)),
        },
        "multiplier_summary": {
            "min": float(np.min(multiplier)),
            "mean": float(np.mean(multiplier)),
            "median": float(np.median(multiplier)),
            "fraction_below_one": float(np.mean(multiplier < 1.0)),
        },
        "loo_retrieval": loo_retrieval,
    }


def _apply_spread_recipe(
    parent: pd.DataFrame,
    analog: pd.DataFrame,
    group_id: int,
    recipe: Recipe,
) -> pd.DataFrame:
    group = parent.loc[parent["group_id"].eq(group_id)].copy()
    group = group.merge(
        analog[
            [
                "forecast_id",
                "group_id",
                "analog_normalized",
                "analog_blend_multiplier",
            ]
        ],
        on=["forecast_id", "group_id"],
        how="left",
        validate="one_to_one",
    )
    parent_normalized = group["prediction_kwh"].to_numpy(dtype=float) / CAPACITIES[
        group_id
    ]
    analog_normalized = group["analog_normalized"].to_numpy(dtype=float)
    multiplier = group["analog_blend_multiplier"].to_numpy(dtype=float)
    transformed = parent_normalized.copy()
    for positions in group.groupby("data_available_kst_dtm", sort=False).indices.values():
        positions = np.asarray(positions, dtype=int)
        p = parent_normalized[positions]
        a = analog_normalized[positions]
        m = multiplier[positions]
        if not np.isfinite(a).all() or not np.isfinite(m).all():
            continue
        if recipe.transform == "level":
            target = a
        elif recipe.transform == "shape":
            target = p.mean() + (a - a.mean())
        elif recipe.transform == "scaled":
            target = a * p.mean() / max(float(a.mean()), 1e-4)
        else:
            raise ValueError(f"unknown transform: {recipe.transform}")
        effective_weight = recipe.blend_weight * np.clip(m, 0.0, 1.0)
        transformed[positions] = (
            (1.0 - effective_weight) * p + effective_weight * target
        )
    group["prediction_kwh"] = np.clip(
        transformed * CAPACITIES[group_id],
        0.0,
        CAPACITIES[group_id],
    )
    return group[parent.columns]


def _parents(surface: pd.DataFrame) -> dict[str, pd.DataFrame]:
    metadata = surface[
        ["forecast_id", "forecast_kst_dtm", "group_id", "data_available_kst_dtm"]
    ]
    distribution = pd.read_parquet(DISTRIBUTION)
    q2_parent = _fold_parent(distribution, metadata, "dev-2023-Q2")
    oof = pd.read_parquet(OOF).merge(
        metadata,
        on=["forecast_id", "forecast_kst_dtm", "group_id"],
        validate="one_to_one",
    )
    q3_parent = _apply_long(
        oof.loc[oof["fold_id"].eq("dev-2023-Q3")].reset_index(drop=True),
        EXPECTED_SELECTIONS,
    )
    q4_parent = _apply_long(
        oof.loc[oof["fold_id"].eq("dev-2023-Q4")].reset_index(drop=True),
        EXPECTED_SELECTIONS,
    )
    return {"q2": q2_parent, "q3": q3_parent, "q4": q4_parent}


def main() -> None:
    if sha256_file(OPEN) != OPEN_SHA or sha256_file(BASELINE) != BASELINE_SHA:
        raise RuntimeError("immutable competition input hash mismatch")
    m234_receipt = json.loads(M234_RECEIPT.read_text(encoding="utf-8"))
    if sha256_file(M234) != m234_receipt["submission_receipt"]["csv_sha256"]:
        raise RuntimeError("M234 CSV hash mismatch")
    recipes = {
        int(group_id): Recipe(**raw)
        for group_id, raw in m234_receipt["policy"]["recipes"].items()
    }

    surface, numeric = development_surface()
    if surface["forecast_kst_dtm"].ge(DEV_CUTOFF).any():
        raise RuntimeError("lockbox row reached M240 spread-shrinkage runner")
    feature_sets = _feature_sets(numeric)
    parents = _parents(surface)
    query_issuances = {
        fold: np.sort(parent["data_available_kst_dtm"].unique())
        for fold, parent in parents.items()
    }

    full_replacements: dict[str, list[pd.DataFrame]] = {fold: [] for fold in FOLDS}
    spread_replacements: dict[str, list[pd.DataFrame]] = {
        fold: [] for fold in FOLDS
    }
    diagnostics: dict[str, object] = {}
    for group_id, recipe in recipes.items():
        group_diagnostics: dict[str, object] = {"recipe": asdict(recipe)}
        for fold in FOLDS:
            parent = parents[fold]
            full_profile, full_retrieval = _selected_profile(
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
            full = _apply_recipe(parent, full_profile, group_id, recipe)
            spread = _apply_spread_recipe(parent, spread_profile, group_id, recipe)
            full_replacements[fold].append(full)
            spread_replacements[fold].append(spread)
            group_diagnostics[fold] = {
                "parent": _group_score(parent, group_id),
                "full_analog": _group_score(full, group_id),
                "spread_shrunk": _group_score(spread, group_id),
                "full_retrieval": full_retrieval,
                "spread_retrieval": spread_retrieval,
            }
        diagnostics[str(group_id)] = group_diagnostics

    full_outputs = {
        fold: _combine(parents[fold], full_replacements[fold]) for fold in FOLDS
    }
    spread_outputs = {
        fold: _combine(parents[fold], spread_replacements[fold]) for fold in FOLDS
    }
    scores: dict[str, object] = {}
    full_deltas: dict[str, float] = {}
    spread_deltas: dict[str, float] = {}
    for fold in FOLDS:
        parent_score = _score(parents[fold])
        full_score = _score(full_outputs[fold])
        spread_score = _score(spread_outputs[fold])
        expected = m234_receipt["development_scores"][fold]
        if abs(full_score["total"] - expected["selected"]["total"]) > 1e-12:
            raise RuntimeError(f"M234 {fold} reproduction score changed")
        full_deltas[fold] = full_score["total"] - parent_score["total"]
        spread_deltas[fold] = spread_score["total"] - parent_score["total"]
        scores[fold] = {
            "parent": parent_score,
            "full_analog": full_score,
            "spread_shrunk": spread_score,
            "full_total_delta": full_deltas[fold],
            "spread_total_delta": spread_deltas[fold],
            "spread_minus_full_total": spread_score["total"] - full_score["total"],
        }

    full_q4_bootstrap = _paired_issuance_bootstrap(parents["q4"], full_outputs["q4"])
    spread_q4_bootstrap = _paired_issuance_bootstrap(
        parents["q4"], spread_outputs["q4"]
    )
    all_positive = all(delta > 0.0 for delta in spread_deltas.values())
    improves_worst_fold = min(spread_deltas.values()) > min(full_deltas.values())
    improves_q4_robustness = (
        spread_deltas["q4"] > full_deltas["q4"]
        and spread_q4_bootstrap["positive_fraction"]
        > full_q4_bootstrap["positive_fraction"]
    )
    promoted = all_positive and (improves_worst_fold or improves_q4_robustness)
    promotion = {
        "promoted": promoted,
        "all_q2_q3_q4_deltas_positive": all_positive,
        "full_worst_fold_delta": min(full_deltas.values()),
        "spread_worst_fold_delta": min(spread_deltas.values()),
        "improves_worst_fold": improves_worst_fold,
        "improves_q4_robustness": improves_q4_robustness,
        "rule": (
            "all spread-shrunk fold deltas positive and either spread worst-fold "
            "delta strictly exceeds full M234 or both Q4 delta and paired-bootstrap "
            "positive fraction strictly exceed full M234"
        ),
    }

    prediction_path = OUTPUT / f"{MODEL_ID}-dev-2023-Q3.parquet"
    spread_outputs["q3"].assign(
        fold_id="dev-2023-Q3",
        model_id=MODEL_ID,
    )[[*METRIC_COLUMNS, "fold_id", "model_id", "data_available_kst_dtm"]].to_parquet(
        prediction_path,
        index=False,
    )
    policy = {
        "architecture": "frozen_m234_with_train_loo_target_spread_shrinkage",
        "recipes": {
            str(group_id): asdict(recipe) for group_id, recipe in recipes.items()
        },
        "spread": "weighted_hourly_standard_deviation_of_neighbor_targets",
        "reference": "train_day_leave_one_out_hourly_quantile",
        "reference_quantile": REFERENCE_QUANTILE,
        "blend_multiplier": "min(1, reference_spread/query_spread)",
        "percentile_search": False,
        "aggregation_search": False,
        "shrink_function_search": False,
        "q1_policy_if_promoted": "unconditional M231 fallback",
    }
    receipt = {
        "schema_version": 1,
        "candidate_id": MODEL_ID,
        "state": (
            "LOCAL_SPREAD_SHRINKAGE_PROMOTED_FOR_TEST_BUILD"
            if promoted
            else "LOCAL_SPREAD_SHRINKAGE_REJECTED_NO_TEST_BUILD"
        ),
        "policy": policy,
        "policy_sha256": canonical_sha256(policy),
        "scores": scores,
        "diagnostics": diagnostics,
        "full_q4_paired_bootstrap": full_q4_bootstrap,
        "spread_q4_paired_bootstrap": spread_q4_bootstrap,
        "promotion": promotion,
        "prediction_path": str(prediction_path.relative_to(ROOT)),
        "prediction_sha256": sha256_file(prediction_path),
        "inputs": {
            "open_zip_sha256": OPEN_SHA,
            "baseline_ipynb_sha256": BASELINE_SHA,
            "distribution_parent_sha256": sha256_file(DISTRIBUTION),
            "v2_parent_oof_sha256": sha256_file(OOF),
            "m234_csv_sha256": sha256_file(M234),
            "m234_receipt_sha256": sha256_file(M234_RECEIPT),
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
