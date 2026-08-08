"""Evaluate one fixed one-year recency half-life over M234 analog weights."""

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
MODEL_ID = "M241_RECENCY_WEIGHTED_ANALOG_Q234"
FOLDS = ("q2", "q3", "q4")
HALF_LIFE_DAYS = 365.0


def _recency_profile(
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
            f"group {group_id} recency days changed: "
            f"train={int(train_mask.sum())}, query={int(query_mask.sum())}"
        )
    train_issuances = issuances[train_mask]
    query_days = issuances[query_mask]
    order, distance, retrieval = _distances(
        values[train_mask],
        values[query_mask],
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
        raise RuntimeError("M241 found a non-historical analog neighbor")
    decay = np.exp2(-age_days / HALF_LIFE_DAYS)
    weights = base_weights * decay
    weights /= weights.sum(axis=1, keepdims=True)
    heads = _profile_heads(
        targets[train_mask][neighbor_indices],
        weights,
        float(np.nanmean(targets[train_mask])),
    )
    profile = _profile_frame(
        frame,
        query_days,
        heads[recipe.head],
        group_id,
    )
    effective_age = np.sum(weights * age_days, axis=1)
    base_effective_age = np.sum(base_weights * age_days, axis=1)
    return profile, {
        **retrieval,
        "training_days": int(train_mask.sum()),
        "query_days": int(query_mask.sum()),
        "half_life_days": HALF_LIFE_DAYS,
        "neighbor_age_days": {
            "min": float(np.min(age_days)),
            "median": float(np.median(age_days)),
            "max": float(np.max(age_days)),
        },
        "base_effective_age_days_mean": float(np.mean(base_effective_age)),
        "recency_effective_age_days_mean": float(np.mean(effective_age)),
        "mean_effective_age_reduction_days": float(
            np.mean(base_effective_age - effective_age)
        ),
        "decay_range": [float(np.min(decay)), float(np.max(decay))],
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
        raise RuntimeError("lockbox row reached M241 recency runner")
    feature_sets = _feature_sets(numeric)
    parents = _parents(surface)
    query_issuances = {
        fold: np.sort(parent["data_available_kst_dtm"].unique())
        for fold, parent in parents.items()
    }

    full_replacements: dict[str, list[pd.DataFrame]] = {fold: [] for fold in FOLDS}
    recency_replacements: dict[str, list[pd.DataFrame]] = {
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
            recency_profile, recency_retrieval = _recency_profile(
                surface,
                feature_sets,
                group_id,
                query_issuances[fold],
                recipe,
            )
            full = _apply_recipe(parent, full_profile, group_id, recipe)
            recency = _apply_recipe(parent, recency_profile, group_id, recipe)
            full_replacements[fold].append(full)
            recency_replacements[fold].append(recency)
            group_diagnostics[fold] = {
                "parent": _group_score(parent, group_id),
                "full_analog": _group_score(full, group_id),
                "recency_weighted": _group_score(recency, group_id),
                "full_retrieval": full_retrieval,
                "recency_retrieval": recency_retrieval,
            }
        diagnostics[str(group_id)] = group_diagnostics

    full_outputs = {
        fold: _combine(parents[fold], full_replacements[fold]) for fold in FOLDS
    }
    recency_outputs = {
        fold: _combine(parents[fold], recency_replacements[fold]) for fold in FOLDS
    }
    scores: dict[str, object] = {}
    full_deltas: dict[str, float] = {}
    recency_deltas: dict[str, float] = {}
    for fold in FOLDS:
        parent_score = _score(parents[fold])
        full_score = _score(full_outputs[fold])
        recency_score = _score(recency_outputs[fold])
        expected = m234_receipt["development_scores"][fold]
        if abs(full_score["total"] - expected["selected"]["total"]) > 1e-12:
            raise RuntimeError(f"M234 {fold} reproduction score changed")
        full_deltas[fold] = full_score["total"] - parent_score["total"]
        recency_deltas[fold] = recency_score["total"] - parent_score["total"]
        scores[fold] = {
            "parent": parent_score,
            "full_analog": full_score,
            "recency_weighted": recency_score,
            "full_total_delta": full_deltas[fold],
            "recency_total_delta": recency_deltas[fold],
            "recency_minus_full_total": recency_score["total"] - full_score["total"],
        }

    full_q4_bootstrap = _paired_issuance_bootstrap(parents["q4"], full_outputs["q4"])
    recency_q4_bootstrap = _paired_issuance_bootstrap(
        parents["q4"], recency_outputs["q4"]
    )
    all_positive = all(delta > 0.0 for delta in recency_deltas.values())
    improves_worst_fold = min(recency_deltas.values()) > min(full_deltas.values())
    improves_q4_robustness = (
        recency_deltas["q4"] > full_deltas["q4"]
        and recency_q4_bootstrap["positive_fraction"]
        > full_q4_bootstrap["positive_fraction"]
    )
    promoted = all_positive and (improves_worst_fold or improves_q4_robustness)
    promotion = {
        "promoted": promoted,
        "all_q2_q3_q4_deltas_positive": all_positive,
        "full_worst_fold_delta": min(full_deltas.values()),
        "recency_worst_fold_delta": min(recency_deltas.values()),
        "improves_worst_fold": improves_worst_fold,
        "improves_q4_robustness": improves_q4_robustness,
        "rule": (
            "all recency-weighted fold deltas positive and either recency "
            "worst-fold delta strictly exceeds full M234 or both Q4 delta and "
            "paired-bootstrap positive fraction strictly exceed full M234"
        ),
    }

    prediction_path = OUTPUT / f"{MODEL_ID}-dev-2023-Q3.parquet"
    recency_outputs["q3"].assign(
        fold_id="dev-2023-Q3",
        model_id=MODEL_ID,
    )[[*METRIC_COLUMNS, "fold_id", "model_id", "data_available_kst_dtm"]].to_parquet(
        prediction_path,
        index=False,
    )
    policy = {
        "architecture": "frozen_m234_with_neighbor_recency_weighting",
        "recipes": {
            str(group_id): asdict(recipe) for group_id, recipe in recipes.items()
        },
        "recency_multiplier": "2**(-age_days/365)",
        "half_life_days": HALF_LIFE_DAYS,
        "age_reference": "query_issuance_minus_historical_neighbor_issuance",
        "half_life_search": False,
        "hard_window_search": False,
        "spread_combination_search": False,
        "q1_policy_if_promoted": "unconditional M231 fallback",
    }
    receipt = {
        "schema_version": 1,
        "candidate_id": MODEL_ID,
        "state": (
            "LOCAL_RECENCY_WEIGHTING_PROMOTED_FOR_TEST_BUILD"
            if promoted
            else "LOCAL_RECENCY_WEIGHTING_REJECTED_NO_TEST_BUILD"
        ),
        "policy": policy,
        "policy_sha256": canonical_sha256(policy),
        "scores": scores,
        "diagnostics": diagnostics,
        "full_q4_paired_bootstrap": full_q4_bootstrap,
        "recency_q4_paired_bootstrap": recency_q4_bootstrap,
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
