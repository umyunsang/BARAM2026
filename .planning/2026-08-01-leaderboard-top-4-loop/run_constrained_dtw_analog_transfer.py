"""Evaluate one fixed constrained-DTW alignment over exact M234 analogs.

M255 preserves every M234 day neighbor, distance weight, profile head,
transform, parent prediction, and blend weight.  It changes only the mapping
from each neighbor's 24-hour historical target profile to the query hours by
using an endpoint-anchored Sakoe-Chiba DTW path over the same standardized
25-core NWP trajectories.  No test submission is built by this runner.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from build_v2_transfer_sequence_challenger import (
    METRIC_COLUMNS,
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
from run_phase_aligned_analog_transfer import (
    FOLDS,
    M234,
    M234_RECEIPT,
    _parents,
    _standardize_trajectories,
)
from strict_dev_surface import DEV_CUTOFF, development_surface

from baram.contracts.hashing import canonical_sha256, sha256_file

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "artifacts" / "backtests" / "metric-aligned-probe"
MODEL_ID = "M255_CONSTRAINED_DTW_ANALOG_Q234"
DTW_RADIUS = 2
MOVE_ORDER = ("diagonal", "vertical", "horizontal")


def _dtw_path(local_cost: np.ndarray) -> list[tuple[int, int]]:
    """Return the fixed endpoint-anchored radius-two minimum-cost path."""
    cost = np.asarray(local_cost, dtype=float)
    if cost.shape != (24, 24) or not np.isfinite(cost).all():
        raise RuntimeError("M255 DTW local-cost contract changed")

    accumulated = np.full((24, 24), np.inf, dtype=float)
    predecessor = np.full((24, 24), -1, dtype=np.int8)
    for query_hour in range(24):
        lower = max(0, query_hour - DTW_RADIUS)
        upper = min(24, query_hour + DTW_RADIUS + 1)
        for neighbor_hour in range(lower, upper):
            if query_hour == 0 and neighbor_hour == 0:
                accumulated[0, 0] = cost[0, 0]
                continue
            candidates = (
                accumulated[query_hour - 1, neighbor_hour - 1]
                if query_hour > 0 and neighbor_hour > 0
                else np.inf,
                accumulated[query_hour - 1, neighbor_hour]
                if query_hour > 0
                else np.inf,
                accumulated[query_hour, neighbor_hour - 1]
                if neighbor_hour > 0
                else np.inf,
            )
            move = int(np.argmin(candidates))
            if not np.isfinite(candidates[move]):
                continue
            accumulated[query_hour, neighbor_hour] = (
                cost[query_hour, neighbor_hour] + candidates[move]
            )
            predecessor[query_hour, neighbor_hour] = move

    if not np.isfinite(accumulated[-1, -1]):
        raise RuntimeError("M255 DTW endpoint is unreachable")
    query_hour = 23
    neighbor_hour = 23
    reversed_path = [(query_hour, neighbor_hour)]
    while query_hour != 0 or neighbor_hour != 0:
        move = int(predecessor[query_hour, neighbor_hour])
        if move == 0:
            query_hour -= 1
            neighbor_hour -= 1
        elif move == 1:
            query_hour -= 1
        elif move == 2:
            neighbor_hour -= 1
        else:
            raise RuntimeError("M255 DTW predecessor chain changed")
        reversed_path.append((query_hour, neighbor_hour))
    path = list(reversed(reversed_path))
    if path[0] != (0, 0) or path[-1] != (23, 23):
        raise RuntimeError("M255 DTW endpoint anchor changed")
    if any(abs(query - neighbor) > DTW_RADIUS for query, neighbor in path):
        raise RuntimeError("M255 DTW path escaped its fixed radius")
    return path


def _warp_neighbor_targets(
    train_values: np.ndarray,
    query_values: np.ndarray,
    neighbor_indices: np.ndarray,
    neighbor_targets: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    """Map neighbor targets to query hours with one fixed DTW path per pair."""
    if neighbor_targets.shape[:2] != neighbor_indices.shape:
        raise RuntimeError("M255 neighbor-target topology changed")
    if neighbor_targets.shape[2] != 24:
        raise RuntimeError("M255 neighbor target length changed")
    train, query, scaling = _standardize_trajectories(train_values, query_values)
    warped = np.empty_like(neighbor_targets, dtype=float)
    path_lengths: list[int] = []
    mean_absolute_offsets: list[float] = []
    nonzero_offset_fractions: list[float] = []
    normalized_costs: list[float] = []

    for query_position in range(len(query)):
        query_trajectory = query[query_position]
        for neighbor_position, train_position in enumerate(
            neighbor_indices[query_position]
        ):
            neighbor_trajectory = train[int(train_position)]
            difference = (
                query_trajectory[:, None, :] - neighbor_trajectory[None, :, :]
            )
            local_cost = np.mean(difference * difference, axis=2)
            path = _dtw_path(local_cost)
            buckets: list[list[float]] = [[] for _ in range(24)]
            for query_hour, neighbor_hour in path:
                buckets[query_hour].append(
                    float(
                        neighbor_targets[
                            query_position, neighbor_position, neighbor_hour
                        ]
                    )
                )
            if any(not values for values in buckets):
                raise RuntimeError("M255 DTW path did not cover every query hour")
            warped[query_position, neighbor_position] = np.asarray(
                [float(np.mean(values)) for values in buckets],
                dtype=float,
            )
            offsets = np.asarray(
                [abs(query_hour - neighbor_hour) for query_hour, neighbor_hour in path],
                dtype=float,
            )
            path_lengths.append(len(path))
            mean_absolute_offsets.append(float(np.mean(offsets)))
            nonzero_offset_fractions.append(float(np.mean(offsets > 0.0)))
            normalized_costs.append(
                float(
                    np.mean(
                        [
                            local_cost[query_hour, neighbor_hour]
                            for query_hour, neighbor_hour in path
                        ]
                    )
                )
            )
    if not np.isfinite(warped).all():
        raise RuntimeError("M255 warped neighbor target is non-finite")
    diagnostics = {
        "path_count": len(path_lengths),
        "path_length": {
            "min": int(np.min(path_lengths)),
            "mean": float(np.mean(path_lengths)),
            "max": int(np.max(path_lengths)),
        },
        "mean_absolute_offset_hours": float(np.mean(mean_absolute_offsets)),
        "mean_nonzero_offset_fraction": float(
            np.mean(nonzero_offset_fractions)
        ),
        "mean_path_local_cost": float(np.mean(normalized_costs)),
        "standardization": scaling,
    }
    return warped, diagnostics


def _self_test_dtw() -> None:
    zero_cost = np.zeros((24, 24), dtype=float)
    path = _dtw_path(zero_cost)
    if path != [(hour, hour) for hour in range(24)]:
        raise RuntimeError("M255 diagonal-first DTW self-test failed")
    trajectory = np.arange(24, dtype=float)[:, None]
    targets = np.arange(24, dtype=float)[None, None, :]
    warped, _ = _warp_neighbor_targets(
        trajectory[None, :, :],
        trajectory[None, :, :],
        np.zeros((1, 1), dtype=int),
        targets,
    )
    if not np.array_equal(warped, targets):
        raise RuntimeError("M255 identity-warp self-test failed")


def _dtw_profile(
    surface: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    group_id: int,
    query_issuances: np.ndarray,
    recipe: Recipe,
) -> tuple[pd.DataFrame, dict[str, object]]:
    representation = next(
        item for item in REPRESENTATIONS if item.name == recipe.representation
    )
    if representation.feature_set != "core":
        raise RuntimeError("M255 predeclared core-feature contract changed")
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
            f"group {group_id} DTW days changed: "
            f"train={int(train_mask.sum())}, query={int(query_mask.sum())}"
        )

    order, distance, retrieval = _distances(
        values[train_mask],
        values[query_mask],
        issuances[train_mask],
        issuances[query_mask],
        representation,
    )
    neighbor_indices = order[:, : recipe.neighbors]
    weights = _kernel_weights(distance[:, : recipe.neighbors], recipe.kernel)
    neighbor_targets = targets[train_mask][neighbor_indices]
    warped_targets, dtw = _warp_neighbor_targets(
        values[train_mask],
        values[query_mask],
        neighbor_indices,
        neighbor_targets,
    )
    heads = _profile_heads(
        warped_targets,
        weights,
        float(np.nanmean(targets[train_mask])),
    )
    profile = _profile_frame(
        frame,
        issuances[query_mask],
        heads[recipe.head],
        group_id,
    )
    return profile, {
        **retrieval,
        "training_days": int(train_mask.sum()),
        "query_days": int(query_mask.sum()),
        "dtw": dtw,
    }


def main() -> None:
    _self_test_dtw()
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
        raise RuntimeError("lockbox row reached M255 DTW runner")
    feature_sets = _feature_sets(numeric)
    if len(feature_sets["core"]) != 25:
        raise RuntimeError("M255 frozen core feature count changed")
    parents = _parents(surface)
    query_issuances = {
        fold: np.sort(parent["data_available_kst_dtm"].unique())
        for fold, parent in parents.items()
    }

    full_replacements: dict[str, list[pd.DataFrame]] = {fold: [] for fold in FOLDS}
    dtw_replacements: dict[str, list[pd.DataFrame]] = {fold: [] for fold in FOLDS}
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
            dtw_profile, dtw_retrieval = _dtw_profile(
                surface,
                feature_sets,
                group_id,
                query_issuances[fold],
                recipe,
            )
            full = _apply_recipe(parent, full_profile, group_id, recipe)
            dtw = _apply_recipe(parent, dtw_profile, group_id, recipe)
            full_replacements[fold].append(full)
            dtw_replacements[fold].append(dtw)
            group_diagnostics[fold] = {
                "parent": _group_score(parent, group_id),
                "full_analog": _group_score(full, group_id),
                "constrained_dtw": _group_score(dtw, group_id),
                "full_retrieval": full_retrieval,
                "dtw_retrieval": dtw_retrieval,
            }
        diagnostics[str(group_id)] = group_diagnostics

    full_outputs = {
        fold: _combine(parents[fold], full_replacements[fold]) for fold in FOLDS
    }
    dtw_outputs = {
        fold: _combine(parents[fold], dtw_replacements[fold]) for fold in FOLDS
    }
    scores: dict[str, object] = {}
    full_deltas: dict[str, float] = {}
    dtw_deltas: dict[str, float] = {}
    for fold in FOLDS:
        parent_score = _score(parents[fold])
        full_score = _score(full_outputs[fold])
        dtw_score = _score(dtw_outputs[fold])
        expected = m234_receipt["development_scores"][fold]
        if abs(full_score["total"] - expected["selected"]["total"]) > 1e-12:
            raise RuntimeError(f"M234 {fold} reproduction score changed")
        full_deltas[fold] = full_score["total"] - parent_score["total"]
        dtw_deltas[fold] = dtw_score["total"] - parent_score["total"]
        scores[fold] = {
            "parent": parent_score,
            "full_analog": full_score,
            "constrained_dtw": dtw_score,
            "full_total_delta": full_deltas[fold],
            "dtw_total_delta": dtw_deltas[fold],
            "dtw_minus_full_total": dtw_score["total"] - full_score["total"],
        }

    full_q4_bootstrap = _paired_issuance_bootstrap(parents["q4"], full_outputs["q4"])
    dtw_q4_bootstrap = _paired_issuance_bootstrap(parents["q4"], dtw_outputs["q4"])
    all_positive = all(delta > 0.0 for delta in dtw_deltas.values())
    improves_worst_fold = min(dtw_deltas.values()) > min(full_deltas.values())
    improves_q4_robustness = (
        dtw_deltas["q4"] > full_deltas["q4"]
        and dtw_q4_bootstrap["positive_fraction"]
        > full_q4_bootstrap["positive_fraction"]
    )
    promoted = all_positive and (improves_worst_fold or improves_q4_robustness)
    promotion = {
        "promoted": promoted,
        "all_q2_q3_q4_deltas_positive": all_positive,
        "full_worst_fold_delta": min(full_deltas.values()),
        "dtw_worst_fold_delta": min(dtw_deltas.values()),
        "improves_worst_fold": improves_worst_fold,
        "improves_q4_robustness": improves_q4_robustness,
        "rule": (
            "all constrained-DTW fold deltas positive and either DTW worst-fold "
            "delta strictly exceeds full M234 or both Q4 delta and paired-bootstrap "
            "positive fraction strictly exceed full M234"
        ),
    }

    prediction_path = OUTPUT / f"{MODEL_ID}-dev-2023-Q3.parquet"
    dtw_outputs["q3"].assign(
        fold_id="dev-2023-Q3",
        model_id=MODEL_ID,
    )[[*METRIC_COLUMNS, "fold_id", "model_id", "data_available_kst_dtm"]].to_parquet(
        prediction_path,
        index=False,
    )
    policy = {
        "architecture": "frozen_m234_with_endpoint_anchored_constrained_dtw",
        "recipes": {
            str(group_id): asdict(recipe) for group_id, recipe in recipes.items()
        },
        "features": "all_25_core_features_from_frozen_m234_contract",
        "local_cost": "mean_squared_train_standardized_nwp_difference",
        "path_endpoints": [[0, 0], [23, 23]],
        "sakoe_chiba_radius_hours": DTW_RADIUS,
        "allowed_moves_and_tie_order": list(MOVE_ORDER),
        "target_mapping": "mean_neighbor_target_for_each_query_hour_path_bucket",
        "path_search": False,
        "radius_search": False,
        "cost_search": False,
        "feature_search": False,
        "neighbor_search": False,
        "group_exception_search": False,
        "q1_policy_if_promoted": "no_test_build_M255_screen_only",
    }
    receipt = {
        "schema_version": 1,
        "candidate_id": MODEL_ID,
        "state": (
            "LOCAL_CONSTRAINED_DTW_PROMOTED_FOR_M244_INTEGRATION_AUDIT"
            if promoted
            else "LOCAL_CONSTRAINED_DTW_REJECTED_NO_TEST_BUILD"
        ),
        "policy": policy,
        "policy_sha256": canonical_sha256(policy),
        "scores": scores,
        "diagnostics": diagnostics,
        "full_q4_paired_bootstrap": full_q4_bootstrap,
        "dtw_q4_paired_bootstrap": dtw_q4_bootstrap,
        "promotion": promotion,
        "prediction_path": str(prediction_path.relative_to(ROOT)),
        "prediction_sha256": sha256_file(prediction_path),
        "inputs": {
            "open_zip_sha256": OPEN_SHA,
            "baseline_ipynb_sha256": BASELINE_SHA,
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
