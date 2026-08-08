"""Evaluate one frozen NWP phase alignment over the M234 analog recipes.

The experiment preserves M234 day retrieval, kernel weights, profile head,
transform, and blend.  For each query/neighbor pair it selects exactly one
lag from a predeclared five-value set by train-standardized NWP trajectory
MSE, then shifts only that neighbor's historical target profile.  The 2024
lockbox is never loaded or scored.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from build_v2_transfer_sequence_challenger import (
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
MODEL_ID = "M238_PHASE_ALIGNED_ANALOG_Q234"
FOLDS = ("q2", "q3", "q4")
LAGS = (0, -1, 1, -2, 2)


def _standardize_trajectories(
    train_values: np.ndarray,
    query_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Fit all imputation and scaling statistics on historical days only."""
    median = np.nanmedian(train_values, axis=(0, 1))
    median = np.where(np.isfinite(median), median, 0.0)
    train = np.where(np.isfinite(train_values), train_values, median[None, None, :])
    query = np.where(np.isfinite(query_values), query_values, median[None, None, :])
    center = np.mean(train, axis=(0, 1))
    scale = np.std(train, axis=(0, 1))
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0)
    train = np.clip((train - center[None, None, :]) / scale[None, None, :], -8.0, 8.0)
    query = np.clip((query - center[None, None, :]) / scale[None, None, :], -8.0, 8.0)
    return train, query, {
        "feature_count": int(train.shape[2]),
        "imputation": "per_feature_train_median",
        "centering": "per_feature_train_mean_after_imputation",
        "scaling": "per_feature_train_population_standard_deviation",
        "clipping": [-8.0, 8.0],
        "constant_feature_count": int(np.sum(scale == 1.0)),
    }


def _choose_neighbor_lags(
    train_values: np.ndarray,
    query_values: np.ndarray,
    neighbor_indices: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    train, query, scaling = _standardize_trajectories(train_values, query_values)
    neighbor = train[neighbor_indices]
    scores = np.empty((*neighbor_indices.shape, len(LAGS)), dtype=float)
    for position, lag in enumerate(LAGS):
        if lag >= 0:
            query_slice = slice(0, 24 - lag)
            neighbor_slice = slice(lag, 24)
        else:
            query_slice = slice(-lag, 24)
            neighbor_slice = slice(0, 24 + lag)
        difference = (
            query[:, None, query_slice, :] - neighbor[:, :, neighbor_slice, :]
        )
        scores[:, :, position] = np.mean(difference * difference, axis=(2, 3))
    selected_positions = np.argmin(scores, axis=2)
    selected_lags = np.asarray(LAGS, dtype=int)[selected_positions]
    selected_scores = np.take_along_axis(
        scores, selected_positions[:, :, None], axis=2
    )[:, :, 0]
    counts = {
        str(lag): int(np.sum(selected_lags == lag))
        for lag in LAGS
    }
    total = int(selected_lags.size)
    diagnostics = {
        "lag_candidates_and_tie_order": list(LAGS),
        "lag_counts": counts,
        "lag_fractions": {
            key: float(value / total) for key, value in counts.items()
        },
        "zero_lag_fraction": float(np.mean(selected_lags == 0)),
        "mean_absolute_lag_hours": float(np.mean(np.abs(selected_lags))),
        "mean_selected_trajectory_mse": float(np.mean(selected_scores)),
        "median_selected_trajectory_mse": float(np.median(selected_scores)),
        "standardization": scaling,
    }
    return selected_lags, diagnostics


def _shift_neighbor_targets(
    neighbor_targets: np.ndarray,
    selected_lags: np.ndarray,
) -> np.ndarray:
    if neighbor_targets.shape[:2] != selected_lags.shape:
        raise RuntimeError("neighbor-target and phase-lag topology changed")
    shifted = np.empty_like(neighbor_targets)
    hours = np.arange(24, dtype=int)
    for lag in LAGS:
        mask = selected_lags == lag
        shifted_for_lag = neighbor_targets[:, :, np.clip(hours + lag, 0, 23)]
        shifted[mask] = shifted_for_lag[mask]
    if not np.isfinite(shifted).all():
        raise RuntimeError("phase-aligned neighbor target contains non-finite values")
    return shifted


def _phase_aligned_profile(
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
        raise RuntimeError("M238 predeclared core-feature contract changed")
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
            f"group {group_id} phase days changed: "
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
    selected_lags, phase = _choose_neighbor_lags(
        values[train_mask],
        values[query_mask],
        neighbor_indices,
    )
    shifted_targets = _shift_neighbor_targets(
        targets[train_mask][neighbor_indices],
        selected_lags,
    )
    heads = _profile_heads(
        shifted_targets,
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
        "phase": phase,
    }


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
        raise RuntimeError("lockbox row reached M238 phase-alignment runner")
    feature_sets = _feature_sets(numeric)
    parents = _parents(surface)
    query_issuances = {
        fold: np.sort(parent["data_available_kst_dtm"].unique())
        for fold, parent in parents.items()
    }

    full_replacements: dict[str, list[pd.DataFrame]] = {fold: [] for fold in FOLDS}
    phase_replacements: dict[str, list[pd.DataFrame]] = {fold: [] for fold in FOLDS}
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
            phase_profile, phase_retrieval = _phase_aligned_profile(
                surface,
                feature_sets,
                group_id,
                query_issuances[fold],
                recipe,
            )
            full = _apply_recipe(parent, full_profile, group_id, recipe)
            phase = _apply_recipe(parent, phase_profile, group_id, recipe)
            full_replacements[fold].append(full)
            phase_replacements[fold].append(phase)
            group_diagnostics[fold] = {
                "parent": _group_score(parent, group_id),
                "full_analog": _group_score(full, group_id),
                "phase_aligned": _group_score(phase, group_id),
                "full_retrieval": full_retrieval,
                "phase_retrieval": phase_retrieval,
            }
        diagnostics[str(group_id)] = group_diagnostics

    full_outputs = {
        fold: _combine(parents[fold], full_replacements[fold]) for fold in FOLDS
    }
    phase_outputs = {
        fold: _combine(parents[fold], phase_replacements[fold]) for fold in FOLDS
    }
    scores: dict[str, object] = {}
    full_deltas: dict[str, float] = {}
    phase_deltas: dict[str, float] = {}
    for fold in FOLDS:
        parent_score = _score(parents[fold])
        full_score = _score(full_outputs[fold])
        phase_score = _score(phase_outputs[fold])
        expected = m234_receipt["development_scores"][fold]
        if abs(full_score["total"] - expected["selected"]["total"]) > 1e-12:
            raise RuntimeError(f"M234 {fold} reproduction score changed")
        full_deltas[fold] = full_score["total"] - parent_score["total"]
        phase_deltas[fold] = phase_score["total"] - parent_score["total"]
        scores[fold] = {
            "parent": parent_score,
            "full_analog": full_score,
            "phase_aligned": phase_score,
            "full_total_delta": full_deltas[fold],
            "phase_total_delta": phase_deltas[fold],
            "phase_minus_full_total": phase_score["total"] - full_score["total"],
        }

    full_q4_bootstrap = _paired_issuance_bootstrap(parents["q4"], full_outputs["q4"])
    phase_q4_bootstrap = _paired_issuance_bootstrap(
        parents["q4"], phase_outputs["q4"]
    )
    all_positive = all(delta > 0.0 for delta in phase_deltas.values())
    improves_worst_fold = min(phase_deltas.values()) > min(full_deltas.values())
    improves_q4_robustness = (
        phase_deltas["q4"] > full_deltas["q4"]
        and phase_q4_bootstrap["positive_fraction"]
        > full_q4_bootstrap["positive_fraction"]
    )
    promoted = all_positive and (improves_worst_fold or improves_q4_robustness)
    promotion = {
        "promoted": promoted,
        "all_q2_q3_q4_deltas_positive": all_positive,
        "full_worst_fold_delta": min(full_deltas.values()),
        "phase_worst_fold_delta": min(phase_deltas.values()),
        "improves_worst_fold": improves_worst_fold,
        "improves_q4_robustness": improves_q4_robustness,
        "rule": (
            "all phase-aligned fold deltas positive and either phase worst-fold "
            "delta strictly exceeds full M234 or both Q4 delta and paired-bootstrap "
            "positive fraction strictly exceed full M234"
        ),
    }

    prediction_path = OUTPUT / f"{MODEL_ID}-dev-2023-Q3.parquet"
    phase_outputs["q3"].assign(
        fold_id="dev-2023-Q3",
        model_id=MODEL_ID,
    )[[*METRIC_COLUMNS, "fold_id", "model_id", "data_available_kst_dtm"]].to_parquet(
        prediction_path,
        index=False,
    )
    policy = {
        "architecture": "frozen_m234_with_nwp_phase_aligned_neighbor_targets",
        "recipes": {
            str(group_id): asdict(recipe) for group_id, recipe in recipes.items()
        },
        "phase_features": "all_core_features_from_frozen_m234_feature_contract",
        "lag_candidates_and_tie_order": list(LAGS),
        "trajectory_score": "mean_squared_error_over_valid_standardized_nwp_cells",
        "target_shift": "historical_target[t+lag]_with_edge_replication",
        "lag_search": False,
        "feature_search": False,
        "target_aware_phase_selection": False,
        "q1_policy_if_promoted": "unconditional M231 fallback",
    }
    receipt = {
        "schema_version": 1,
        "candidate_id": MODEL_ID,
        "state": (
            "LOCAL_PHASE_ALIGNMENT_PROMOTED_FOR_TEST_BUILD"
            if promoted
            else "LOCAL_PHASE_ALIGNMENT_REJECTED_NO_TEST_BUILD"
        ),
        "policy": policy,
        "policy_sha256": canonical_sha256(policy),
        "scores": scores,
        "diagnostics": diagnostics,
        "full_q4_paired_bootstrap": full_q4_bootstrap,
        "phase_q4_paired_bootstrap": phase_q4_bootstrap,
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
