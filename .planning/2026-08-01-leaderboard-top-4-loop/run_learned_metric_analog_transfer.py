"""Evaluate one frozen PLS similarity metric over the M234 analog recipes.

Only the analog-day distance changes.  The metric is fitted on complete days
ending before each query fold, and query days are transformed from NWP alone.
M234 neighbor counts, kernels, heads, transforms, and blend weights remain
fixed.  The consumed 2024 lockbox is never loaded or scored.
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
    _cyclic_doy,
    _feature_sets,
    _kernel_weights,
    _profile_frame,
    _profile_heads,
    _representation_matrix,
    _selected_profile,
)
from run_stable_daily_analog_profile import DISTRIBUTION, _fold_parent
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from strict_dev_surface import DEV_CUTOFF, development_surface

from baram.contracts.hashing import canonical_sha256, sha256_file

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "artifacts" / "backtests" / "metric-aligned-probe"
M234 = ROOT / "artifacts" / "submissions" / "E0_ROBUST_ANALOG_DEV-08f07b6d9e87.csv"
M234_RECEIPT = M234.with_suffix(".receipt.json")
MODEL_ID = "M239_PLS_LEARNED_ANALOG_METRIC_Q234"
FOLDS = ("q2", "q3", "q4")
PLS_COMPONENTS = 8
SEASON_WEIGHT = 2.5


def _learned_distances(
    train_values: np.ndarray,
    query_values: np.ndarray,
    train_targets: np.ndarray,
    train_issuances: np.ndarray,
    query_issuances: np.ndarray,
    neighbors: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    train = _representation_matrix(train_values, "raw_delta")
    query = _representation_matrix(query_values, "raw_delta")
    median = np.nanmedian(train, axis=0)
    median = np.where(np.isfinite(median), median, 0.0)
    train = np.where(np.isfinite(train), train, median)
    query = np.where(np.isfinite(query), query, median)
    scaler = StandardScaler()
    train = np.clip(scaler.fit_transform(train), -8.0, 8.0)
    query = np.clip(scaler.transform(query), -8.0, 8.0)
    if PLS_COMPONENTS > min(len(train) - 1, train.shape[1], train_targets.shape[1]):
        raise RuntimeError("M239 PLS component contract cannot be supported")
    model = PLSRegression(
        n_components=PLS_COMPONENTS,
        scale=False,
        max_iter=1_000,
        tol=1e-7,
        copy=True,
    )
    model.fit(train, train_targets)
    train_scores = np.asarray(model.transform(train), dtype=float)
    query_scores = np.asarray(model.transform(query), dtype=float)
    score_center = np.mean(train_scores, axis=0)
    score_scale = np.std(train_scores, axis=0)
    score_scale = np.where(score_scale > 1e-8, score_scale, 1.0)
    train_scores = np.clip((train_scores - score_center) / score_scale, -8.0, 8.0)
    query_scores = np.clip((query_scores - score_center) / score_scale, -8.0, 8.0)
    train_scores = np.column_stack(
        [train_scores, SEASON_WEIGHT * _cyclic_doy(train_issuances)]
    )
    query_scores = np.column_stack(
        [query_scores, SEASON_WEIGHT * _cyclic_doy(query_issuances)]
    )
    squared = (
        np.sum(query_scores * query_scores, axis=1, keepdims=True)
        + np.sum(train_scores * train_scores, axis=1)[None, :]
        - 2.0 * query_scores @ train_scores.T
    )
    squared = np.maximum(squared, 0.0)
    order = np.argsort(squared, axis=1)[:, :neighbors]
    distance = np.sqrt(np.take_along_axis(squared, order, axis=1))
    prediction = np.asarray(model.predict(query), dtype=float)
    return order, distance, {
        "input_dimensions": int(train.shape[1]),
        "pls_components": PLS_COMPONENTS,
        "latent_dimensions_with_season": int(train_scores.shape[1]),
        "season_weight": SEASON_WEIGHT,
        "component_iterations": [int(value) for value in model.n_iter_],
        "training_target_mean": float(np.mean(train_targets)),
        "query_direct_profile_mean": float(np.mean(prediction)),
        "query_direct_profile_range": [
            float(np.min(prediction)),
            float(np.max(prediction)),
        ],
        "median_nearest_distance": float(np.median(distance[:, 0])),
        "preprocessing": {
            "imputation": "per_flattened_cell_train_median",
            "scaling": "per_flattened_cell_train_standard_scaler",
            "input_clipping": [-8.0, 8.0],
            "score_scaling": "per_pls_x_score_train_population_standard_deviation",
            "score_clipping": [-8.0, 8.0],
        },
    }


def _learned_profile(
    surface: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    group_id: int,
    query_issuances: np.ndarray,
    recipe: Recipe,
) -> tuple[pd.DataFrame, dict[str, object]]:
    representation = next(
        item for item in REPRESENTATIONS if item.name == recipe.representation
    )
    if (
        representation.feature_set != "core"
        or representation.mode != "raw_delta"
        or representation.season_weight != SEASON_WEIGHT
    ):
        raise RuntimeError("M239 frozen M234 representation contract changed")
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
            f"group {group_id} learned-metric days changed: "
            f"train={int(train_mask.sum())}, query={int(query_mask.sum())}"
        )
    order, distance, diagnostics = _learned_distances(
        values[train_mask],
        values[query_mask],
        targets[train_mask],
        issuances[train_mask],
        issuances[query_mask],
        recipe.neighbors,
    )
    weights = _kernel_weights(distance, recipe.kernel)
    heads = _profile_heads(
        targets[train_mask][order],
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
        **diagnostics,
        "training_days": int(train_mask.sum()),
        "query_days": int(query_mask.sum()),
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
        raise RuntimeError("lockbox row reached M239 learned-metric runner")
    feature_sets = _feature_sets(numeric)
    if len(feature_sets["core"]) != 25:
        raise RuntimeError("M239 frozen core feature count changed")
    parents = _parents(surface)
    query_issuances = {
        fold: np.sort(parent["data_available_kst_dtm"].unique())
        for fold, parent in parents.items()
    }

    full_replacements: dict[str, list[pd.DataFrame]] = {fold: [] for fold in FOLDS}
    learned_replacements: dict[str, list[pd.DataFrame]] = {
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
            learned_profile, learned_retrieval = _learned_profile(
                surface,
                feature_sets,
                group_id,
                query_issuances[fold],
                recipe,
            )
            full = _apply_recipe(parent, full_profile, group_id, recipe)
            learned = _apply_recipe(parent, learned_profile, group_id, recipe)
            full_replacements[fold].append(full)
            learned_replacements[fold].append(learned)
            group_diagnostics[fold] = {
                "parent": _group_score(parent, group_id),
                "full_analog": _group_score(full, group_id),
                "learned_metric": _group_score(learned, group_id),
                "full_retrieval": full_retrieval,
                "learned_retrieval": learned_retrieval,
            }
        diagnostics[str(group_id)] = group_diagnostics

    full_outputs = {
        fold: _combine(parents[fold], full_replacements[fold]) for fold in FOLDS
    }
    learned_outputs = {
        fold: _combine(parents[fold], learned_replacements[fold]) for fold in FOLDS
    }
    scores: dict[str, object] = {}
    full_deltas: dict[str, float] = {}
    learned_deltas: dict[str, float] = {}
    for fold in FOLDS:
        parent_score = _score(parents[fold])
        full_score = _score(full_outputs[fold])
        learned_score = _score(learned_outputs[fold])
        expected = m234_receipt["development_scores"][fold]
        if abs(full_score["total"] - expected["selected"]["total"]) > 1e-12:
            raise RuntimeError(f"M234 {fold} reproduction score changed")
        full_deltas[fold] = full_score["total"] - parent_score["total"]
        learned_deltas[fold] = learned_score["total"] - parent_score["total"]
        scores[fold] = {
            "parent": parent_score,
            "full_analog": full_score,
            "learned_metric": learned_score,
            "full_total_delta": full_deltas[fold],
            "learned_total_delta": learned_deltas[fold],
            "learned_minus_full_total": learned_score["total"] - full_score["total"],
        }

    full_q4_bootstrap = _paired_issuance_bootstrap(parents["q4"], full_outputs["q4"])
    learned_q4_bootstrap = _paired_issuance_bootstrap(
        parents["q4"], learned_outputs["q4"]
    )
    all_positive = all(delta > 0.0 for delta in learned_deltas.values())
    improves_worst_fold = min(learned_deltas.values()) > min(full_deltas.values())
    improves_q4_robustness = (
        learned_deltas["q4"] > full_deltas["q4"]
        and learned_q4_bootstrap["positive_fraction"]
        > full_q4_bootstrap["positive_fraction"]
    )
    promoted = all_positive and (improves_worst_fold or improves_q4_robustness)
    promotion = {
        "promoted": promoted,
        "all_q2_q3_q4_deltas_positive": all_positive,
        "full_worst_fold_delta": min(full_deltas.values()),
        "learned_worst_fold_delta": min(learned_deltas.values()),
        "improves_worst_fold": improves_worst_fold,
        "improves_q4_robustness": improves_q4_robustness,
        "rule": (
            "all learned-metric fold deltas positive and either learned worst-fold "
            "delta strictly exceeds full M234 or both Q4 delta and paired-bootstrap "
            "positive fraction strictly exceed full M234"
        ),
    }

    prediction_path = OUTPUT / f"{MODEL_ID}-dev-2023-Q3.parquet"
    learned_outputs["q3"].assign(
        fold_id="dev-2023-Q3",
        model_id=MODEL_ID,
    )[[*METRIC_COLUMNS, "fold_id", "model_id", "data_available_kst_dtm"]].to_parquet(
        prediction_path,
        index=False,
    )
    policy = {
        "architecture": "frozen_m234_with_pls_learned_analog_similarity",
        "recipes": {
            str(group_id): asdict(recipe) for group_id, recipe in recipes.items()
        },
        "features": "all_25_core_raw_plus_delta_daily_nwp_cells",
        "pls_components": PLS_COMPONENTS,
        "latent_score_scaling": "train_population_standard_deviation",
        "season_weight": SEASON_WEIGHT,
        "distance": "euclidean_in_standardized_pls_x_scores_plus_cyclic_season",
        "component_search": False,
        "feature_search": False,
        "metric_blend_search": False,
        "q1_policy_if_promoted": "unconditional M231 fallback",
    }
    receipt = {
        "schema_version": 1,
        "candidate_id": MODEL_ID,
        "state": (
            "LOCAL_LEARNED_METRIC_PROMOTED_FOR_TEST_BUILD"
            if promoted
            else "LOCAL_LEARNED_METRIC_REJECTED_NO_TEST_BUILD"
        ),
        "policy": policy,
        "policy_sha256": canonical_sha256(policy),
        "scores": scores,
        "diagnostics": diagnostics,
        "full_q4_paired_bootstrap": full_q4_bootstrap,
        "learned_q4_paired_bootstrap": learned_q4_bootstrap,
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
