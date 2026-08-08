"""Evaluate one fixed same-lead three-hour-window analog formulation."""

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
    Recipe,
    _apply_recipe,
    _combine,
    _complete_group_days,
    _cyclic_doy,
    _feature_sets,
    _profile_frame,
    _profile_heads,
    _selected_profile,
)
from run_stable_daily_analog_profile import DISTRIBUTION, _fold_parent
from sklearn.preprocessing import StandardScaler
from strict_dev_surface import DEV_CUTOFF, development_surface

from baram.contracts.hashing import canonical_sha256, sha256_file

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "artifacts" / "backtests" / "metric-aligned-probe"
M234 = ROOT / "artifacts" / "submissions" / "E0_ROBUST_ANALOG_DEV-08f07b6d9e87.csv"
M234_RECEIPT = M234.with_suffix(".receipt.json")
MODEL_ID = "M242_LEAD_LOCAL_3H_ANALOG_Q234"
FOLDS = ("q2", "q3", "q4")
NEIGHBORS = 20
SEASON_WEIGHT = 2.5


def _three_hour_windows(values: np.ndarray) -> np.ndarray:
    previous = np.concatenate([values[:, :1, :], values[:, :-1, :]], axis=1)
    following = np.concatenate([values[:, 1:, :], values[:, -1:, :]], axis=1)
    return np.concatenate([previous, values, following], axis=2)


def _lead_local_profile(
    surface: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    group_id: int,
    query_issuances: np.ndarray,
    recipe: Recipe,
) -> tuple[pd.DataFrame, dict[str, object]]:
    frame, issuances, values, targets = _complete_group_days(
        surface,
        group_id,
        feature_sets["core"],
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
            f"group {group_id} lead-local days changed: "
            f"train={int(train_mask.sum())}, query={int(query_mask.sum())}"
        )
    train_values = _three_hour_windows(values[train_mask])
    query_values = _three_hour_windows(values[query_mask])
    train_targets = targets[train_mask]
    train_issuances = issuances[train_mask]
    query_days = issuances[query_mask]
    if train_values.shape[2] != 75:
        raise RuntimeError("M242 frozen 25-core three-hour feature count changed")

    train_season = SEASON_WEIGHT * _cyclic_doy(train_issuances)
    query_season = SEASON_WEIGHT * _cyclic_doy(query_days)
    neighbor_targets = np.empty(
        (len(query_days), NEIGHBORS, 24),
        dtype=float,
    )
    nearest_distances = np.empty((len(query_days), 24), dtype=float)
    for hour in range(24):
        train = train_values[:, hour, :]
        query = query_values[:, hour, :]
        median = np.nanmedian(train, axis=0)
        median = np.where(np.isfinite(median), median, 0.0)
        train = np.where(np.isfinite(train), train, median)
        query = np.where(np.isfinite(query), query, median)
        scaler = StandardScaler()
        train = np.clip(scaler.fit_transform(train), -8.0, 8.0)
        query = np.clip(scaler.transform(query), -8.0, 8.0)
        train = np.column_stack([train, train_season])
        query = np.column_stack([query, query_season])
        squared = (
            np.sum(query * query, axis=1, keepdims=True)
            + np.sum(train * train, axis=1)[None, :]
            - 2.0 * query @ train.T
        )
        squared = np.maximum(squared, 0.0)
        order = np.argsort(squared, axis=1)[:, :NEIGHBORS]
        nearest_distances[:, hour] = np.sqrt(
            np.take_along_axis(squared, order[:, :1], axis=1)[:, 0]
        )
        neighbor_targets[:, :, hour] = train_targets[order, hour]
    weights = np.full(
        (len(query_days), NEIGHBORS),
        1.0 / NEIGHBORS,
        dtype=float,
    )
    heads = _profile_heads(
        neighbor_targets,
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
        "architecture": "same_lead_edge_replicated_three_hour_window",
        "training_days": int(train_mask.sum()),
        "query_days": int(query_mask.sum()),
        "core_feature_count": 25,
        "window_feature_count": int(train_values.shape[2]),
        "neighbors": NEIGHBORS,
        "kernel": "uniform",
        "season_weight": SEASON_WEIGHT,
        "median_nearest_distance": float(np.median(nearest_distances)),
        "nearest_distance_by_lead": [
            float(value) for value in np.median(nearest_distances, axis=0)
        ],
        "preprocessing": {
            "imputation": "per_lead_window_cell_train_median",
            "scaling": "per_lead_window_cell_train_standard_scaler",
            "clipping": [-8.0, 8.0],
        },
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
        raise RuntimeError("lockbox row reached M242 lead-local runner")
    feature_sets = _feature_sets(numeric)
    if len(feature_sets["core"]) != 25:
        raise RuntimeError("M242 frozen core feature count changed")
    parents = _parents(surface)
    query_issuances = {
        fold: np.sort(parent["data_available_kst_dtm"].unique())
        for fold, parent in parents.items()
    }

    full_replacements: dict[str, list[pd.DataFrame]] = {fold: [] for fold in FOLDS}
    local_replacements: dict[str, list[pd.DataFrame]] = {fold: [] for fold in FOLDS}
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
            local_profile, local_retrieval = _lead_local_profile(
                surface,
                feature_sets,
                group_id,
                query_issuances[fold],
                recipe,
            )
            full = _apply_recipe(parent, full_profile, group_id, recipe)
            local = _apply_recipe(parent, local_profile, group_id, recipe)
            full_replacements[fold].append(full)
            local_replacements[fold].append(local)
            group_diagnostics[fold] = {
                "parent": _group_score(parent, group_id),
                "full_analog": _group_score(full, group_id),
                "lead_local": _group_score(local, group_id),
                "full_retrieval": full_retrieval,
                "lead_local_retrieval": local_retrieval,
            }
        diagnostics[str(group_id)] = group_diagnostics

    full_outputs = {
        fold: _combine(parents[fold], full_replacements[fold]) for fold in FOLDS
    }
    local_outputs = {
        fold: _combine(parents[fold], local_replacements[fold]) for fold in FOLDS
    }
    scores: dict[str, object] = {}
    full_deltas: dict[str, float] = {}
    local_deltas: dict[str, float] = {}
    for fold in FOLDS:
        parent_score = _score(parents[fold])
        full_score = _score(full_outputs[fold])
        local_score = _score(local_outputs[fold])
        expected = m234_receipt["development_scores"][fold]
        if abs(full_score["total"] - expected["selected"]["total"]) > 1e-12:
            raise RuntimeError(f"M234 {fold} reproduction score changed")
        full_deltas[fold] = full_score["total"] - parent_score["total"]
        local_deltas[fold] = local_score["total"] - parent_score["total"]
        scores[fold] = {
            "parent": parent_score,
            "full_analog": full_score,
            "lead_local": local_score,
            "full_total_delta": full_deltas[fold],
            "lead_local_total_delta": local_deltas[fold],
            "lead_local_minus_full_total": local_score["total"] - full_score["total"],
        }

    full_q4_bootstrap = _paired_issuance_bootstrap(parents["q4"], full_outputs["q4"])
    local_q4_bootstrap = _paired_issuance_bootstrap(
        parents["q4"], local_outputs["q4"]
    )
    all_positive = all(delta > 0.0 for delta in local_deltas.values())
    improves_worst_fold = min(local_deltas.values()) > min(full_deltas.values())
    improves_q4_robustness = (
        local_deltas["q4"] > full_deltas["q4"]
        and local_q4_bootstrap["positive_fraction"]
        > full_q4_bootstrap["positive_fraction"]
    )
    promoted = all_positive and (improves_worst_fold or improves_q4_robustness)
    promotion = {
        "promoted": promoted,
        "all_q2_q3_q4_deltas_positive": all_positive,
        "full_worst_fold_delta": min(full_deltas.values()),
        "lead_local_worst_fold_delta": min(local_deltas.values()),
        "improves_worst_fold": improves_worst_fold,
        "improves_q4_robustness": improves_q4_robustness,
        "rule": (
            "all lead-local fold deltas positive and either lead-local worst-fold "
            "delta strictly exceeds full M234 or both Q4 delta and paired-bootstrap "
            "positive fraction strictly exceed full M234"
        ),
    }

    prediction_path = OUTPUT / f"{MODEL_ID}-dev-2023-Q3.parquet"
    local_outputs["q3"].assign(
        fold_id="dev-2023-Q3",
        model_id=MODEL_ID,
    )[[*METRIC_COLUMNS, "fold_id", "model_id", "data_available_kst_dtm"]].to_parquet(
        prediction_path,
        index=False,
    )
    policy = {
        "architecture": "same_lead_edge_replicated_three_hour_window_analog",
        "recipes": {
            str(group_id): asdict(recipe) for group_id, recipe in recipes.items()
        },
        "features": "all_25_core_previous_current_next",
        "neighbors": NEIGHBORS,
        "kernel": "uniform",
        "same_lead_only": True,
        "season_weight": SEASON_WEIGHT,
        "neighbor_search": False,
        "kernel_search": False,
        "window_search": False,
        "feature_search": False,
        "q1_policy_if_promoted": "unconditional M231 fallback",
    }
    receipt = {
        "schema_version": 1,
        "candidate_id": MODEL_ID,
        "state": (
            "LOCAL_LEAD_LOCAL_ANALOG_PROMOTED_FOR_TEST_BUILD"
            if promoted
            else "LOCAL_LEAD_LOCAL_ANALOG_REJECTED_NO_TEST_BUILD"
        ),
        "policy": policy,
        "policy_sha256": canonical_sha256(policy),
        "scores": scores,
        "diagnostics": diagnostics,
        "full_q4_paired_bootstrap": full_q4_bootstrap,
        "lead_local_q4_paired_bootstrap": local_q4_bootstrap,
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
