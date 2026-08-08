"""Select daily analog-profile corrections only when Q2 and Q3 both improve."""

from __future__ import annotations

import json
import time
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
    BLEND_WEIGHTS,
    HEADS,
    KERNELS,
    NEIGHBOR_COUNTS,
    OPEN,
    OPEN_SHA,
    REPRESENTATIONS,
    TRANSFORMS,
    Recipe,
    _apply_recipe,
    _combine,
    _complete_group_days,
    _distances,
    _feature_sets,
    _group_total_from_arrays,
    _kernel_weights,
    _profile_frame,
    _profile_heads,
    _selected_profile,
)
from strict_dev_surface import DEV_CUTOFF, development_surface

from baram.contracts.hashing import canonical_sha256, sha256_file

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "artifacts" / "backtests" / "metric-aligned-probe"
CANDIDATE_ID = "M233_STABLE_DAILY_ANALOG_PROFILE_Q3"
DISTRIBUTION = (
    ROOT
    / "artifacts"
    / "backtests"
    / "distribution-v2"
    / "baram-v2-20260801-01"
    / "D1_LGBM_SHARED_BASE-oof.parquet"
)


def _candidate_scores(
    surface: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    parent: pd.DataFrame,
    group_id: int,
    query_issuances: np.ndarray,
) -> tuple[dict[Recipe, float], dict[str, object]]:
    parent_group = parent.loc[parent["group_id"].eq(group_id)].copy()
    scores: dict[Recipe, float] = {}
    representation_diagnostics: dict[str, object] = {}
    for representation in REPRESENTATIONS:
        frame, issuances, values, targets = _complete_group_days(
            surface, group_id, feature_sets[representation.feature_set]
        )
        query_mask = np.isin(issuances, query_issuances)
        cutoff = pd.Timestamp(np.min(query_issuances))
        day_end = frame.groupby("data_available_kst_dtm", sort=True)[
            "forecast_kst_dtm"
        ].max().to_numpy()
        train_mask = (day_end < cutoff) & np.isfinite(targets).all(axis=1)
        if int(train_mask.sum()) < 80 or not query_mask.any():
            raise RuntimeError(
                f"group {group_id} stable analog days changed: "
                f"train={int(train_mask.sum())}, query={int(query_mask.sum())}"
            )
        order, distance, diagnostics = _distances(
            values[train_mask],
            values[query_mask],
            issuances[train_mask],
            issuances[query_mask],
            representation,
        )
        representation_diagnostics[representation.name] = {
            **diagnostics,
            "training_days": int(train_mask.sum()),
            "query_days": int(query_mask.sum()),
        }
        available = min(distance.shape[1], max(NEIGHBOR_COUNTS))
        for neighbors in NEIGHBOR_COUNTS:
            if neighbors > available:
                continue
            neighbor_indices = order[:, :neighbors]
            neighbor_targets = targets[train_mask][neighbor_indices]
            mean_generation = float(np.nanmean(targets[train_mask]))
            for kernel in KERNELS:
                weights = _kernel_weights(distance[:, :neighbors], kernel)
                heads = _profile_heads(neighbor_targets, weights, mean_generation)
                for head in HEADS:
                    analog = _profile_frame(
                        frame,
                        issuances[query_mask],
                        heads[head],
                        group_id,
                    )
                    merged = parent_group.merge(
                        analog[["forecast_id", "group_id", "analog_normalized"]],
                        on=["forecast_id", "group_id"],
                        how="left",
                        validate="one_to_one",
                    )
                    if merged["analog_normalized"].isna().any():
                        raise RuntimeError("stable analog merge is incomplete")
                    actual = merged["actual_kwh"].to_numpy(dtype=float)
                    parent_normalized = (
                        merged["prediction_kwh"].to_numpy(dtype=float)
                        / CAPACITIES[group_id]
                    )
                    analog_normalized = merged["analog_normalized"].to_numpy(
                        dtype=float
                    )
                    day_positions = list(
                        merged.groupby(
                            "data_available_kst_dtm", sort=False
                        ).indices.values()
                    )
                    for transform in TRANSFORMS:
                        target = analog_normalized.copy()
                        if transform != "level":
                            for positions in day_positions:
                                positions = np.asarray(positions, dtype=int)
                                p = parent_normalized[positions]
                                a = analog_normalized[positions]
                                if transform == "shape":
                                    target[positions] = p.mean() + (a - a.mean())
                                else:
                                    target[positions] = (
                                        a * p.mean() / max(float(a.mean()), 1e-4)
                                    )
                        for blend_weight in BLEND_WEIGHTS:
                            recipe = Recipe(
                                representation=representation.name,
                                neighbors=neighbors,
                                kernel=kernel,
                                head=head,
                                transform=transform,
                                blend_weight=blend_weight,
                            )
                            candidate = np.clip(
                                (1.0 - blend_weight) * parent_normalized
                                + blend_weight * target,
                                0.0,
                                1.0,
                            )
                            scores[recipe] = _group_total_from_arrays(
                                actual,
                                candidate * CAPACITIES[group_id],
                                group_id,
                            )
    return scores, representation_diagnostics


def _stable_selection(
    q2_scores: dict[Recipe, float],
    q3_scores: dict[Recipe, float],
    q2_baseline: float,
    q3_baseline: float,
) -> tuple[Recipe | None, dict[str, float | int | None]]:
    common = sorted(
        set(q2_scores) & set(q3_scores),
        key=lambda recipe: (
            recipe.representation,
            recipe.neighbors,
            recipe.kernel,
            recipe.head,
            recipe.transform,
            recipe.blend_weight,
        ),
    )
    stable = [
        recipe
        for recipe in common
        if q2_scores[recipe] > q2_baseline and q3_scores[recipe] > q3_baseline
    ]
    if not stable:
        return None, {
            "common_candidates": len(common),
            "stable_positive_candidates": 0,
            "q2_delta": None,
            "q3_delta": None,
        }
    selected = max(
        stable,
        key=lambda recipe: (
            min(
                q2_scores[recipe] - q2_baseline,
                q3_scores[recipe] - q3_baseline,
            ),
            0.5
            * (
                q2_scores[recipe]
                - q2_baseline
                + q3_scores[recipe]
                - q3_baseline
            ),
        ),
    )
    return selected, {
        "common_candidates": len(common),
        "stable_positive_candidates": len(stable),
        "q2_delta": q2_scores[selected] - q2_baseline,
        "q3_delta": q3_scores[selected] - q3_baseline,
        "worst_fold_delta": min(
            q2_scores[selected] - q2_baseline,
            q3_scores[selected] - q3_baseline,
        ),
    }


def _fold_parent(
    distribution: pd.DataFrame,
    metadata: pd.DataFrame,
    fold_id: str,
) -> pd.DataFrame:
    parent = distribution.loc[
        distribution["fold_id"].eq(fold_id)
        & distribution["quantile"].eq(0.50)
    ].drop(columns=["quantile"])
    parent = parent.merge(
        metadata,
        on=["forecast_id", "forecast_kst_dtm", "group_id"],
        validate="one_to_one",
    )
    return _apply_long(parent.reset_index(drop=True), EXPECTED_SELECTIONS)


def main() -> None:
    if sha256_file(OPEN) != OPEN_SHA or sha256_file(BASELINE) != BASELINE_SHA:
        raise RuntimeError("immutable competition input hash mismatch")
    started = time.perf_counter()
    surface, numeric = development_surface()
    if surface["forecast_kst_dtm"].ge(DEV_CUTOFF).any():
        raise RuntimeError("lockbox row reached stable daily analog runner")
    feature_sets = _feature_sets(numeric)
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
    q3 = oof.loc[oof["fold_id"].eq("dev-2023-Q3")].reset_index(drop=True)
    q4 = oof.loc[oof["fold_id"].eq("dev-2023-Q4")].reset_index(drop=True)
    q3_parent = _apply_long(q3, EXPECTED_SELECTIONS)
    q4_parent = _apply_long(q4, EXPECTED_SELECTIONS)
    parents = {"q2": q2_parent, "q3": q3_parent, "q4": q4_parent}
    issuances = {
        name: np.sort(frame["data_available_kst_dtm"].unique())
        for name, frame in parents.items()
    }

    selections: dict[int, Recipe] = {}
    selection_diagnostics: dict[str, object] = {}
    q2_replacements: list[pd.DataFrame] = []
    q3_replacements: list[pd.DataFrame] = []
    q4_replacements: list[pd.DataFrame] = []
    for group_id in CAPACITIES:
        q2_scores, q2_repr = _candidate_scores(
            surface,
            feature_sets,
            q2_parent,
            group_id,
            issuances["q2"],
        )
        q3_scores, q3_repr = _candidate_scores(
            surface,
            feature_sets,
            q3_parent,
            group_id,
            issuances["q3"],
        )
        q2_baseline = _group_score(q2_parent, group_id)["total"]
        q3_baseline = _group_score(q3_parent, group_id)["total"]
        recipe, stability = _stable_selection(
            q2_scores,
            q3_scores,
            q2_baseline,
            q3_baseline,
        )
        selection_diagnostics[str(group_id)] = {
            "q2_parent": _group_score(q2_parent, group_id),
            "q3_parent": _group_score(q3_parent, group_id),
            "stability": stability,
            "selected_recipe": asdict(recipe) if recipe is not None else None,
            "q2_representations": q2_repr,
            "q3_representations": q3_repr,
        }
        if recipe is None:
            continue
        selections[group_id] = recipe
        for fold_name, parent, replacements in (
            ("q2", q2_parent, q2_replacements),
            ("q3", q3_parent, q3_replacements),
            ("q4", q4_parent, q4_replacements),
        ):
            profile, retrieval = _selected_profile(
                surface,
                feature_sets,
                group_id,
                issuances[fold_name],
                recipe,
            )
            replacement = _apply_recipe(parent, profile, group_id, recipe)
            replacements.append(replacement)
            selection_diagnostics[str(group_id)][f"{fold_name}_selected"] = (
                _group_score(replacement, group_id)
            )
            selection_diagnostics[str(group_id)][f"{fold_name}_retrieval"] = retrieval

    q2_output = _combine(q2_parent, q2_replacements)
    q3_output = _combine(q3_parent, q3_replacements)
    q4_output = _combine(q4_parent, q4_replacements)
    output = q3_output.assign(
        fold_id="dev-2023-Q3",
        model_id=CANDIDATE_ID,
    )
    output_path = OUTPUT / f"{CANDIDATE_ID}-dev-2023-Q3.parquet"
    output[[*METRIC_COLUMNS, "fold_id", "model_id", "data_available_kst_dtm"]].to_parquet(
        output_path,
        index=False,
    )
    policy = {
        str(group_id): asdict(recipe) for group_id, recipe in selections.items()
    }
    receipt = {
        "schema_version": 1,
        "candidate_id": CANDIDATE_ID,
        "architecture": "q2_q3_stable_daily_nwp_analog_profile_v2_sequence_correction",
        "selection_folds": ["dev-2023-Q2", "dev-2023-Q3"],
        "frozen_transfer_fold": "dev-2023-Q4",
        "selection_rule": "maximize the worst positive group-total delta over Q2 and Q3",
        "selected_recipes": policy,
        "selection_diagnostics": selection_diagnostics,
        "q2_parent_score": _score(q2_parent),
        "q2_selected_score": _score(q2_output),
        "q3_parent_score": _score(q3_parent),
        "q3_selected_score": _score(q3_output),
        "q4_parent_score": _score(q4_parent),
        "q4_frozen_score": _score(q4_output),
        "q4_total_delta": _score(q4_output)["total"] - _score(q4_parent)["total"],
        "q4_paired_bootstrap": _paired_issuance_bootstrap(q4_parent, q4_output),
        "feature_sets": feature_sets,
        "representations": [
            asdict(item) | {"name": item.name} for item in REPRESENTATIONS
        ],
        "candidate_count_per_complete_representation": (
            len(NEIGHBOR_COUNTS)
            * len(KERNELS)
            * len(HEADS)
            * len(TRANSFORMS)
            * len(BLEND_WEIGHTS)
        ),
        "training_scope": "complete pre-cutoff issuance days from physical pre-2024 surface",
        "prediction_path": str(output_path.relative_to(ROOT)),
        "prediction_sha256": sha256_file(output_path),
        "policy_sha256": canonical_sha256(policy),
        "distribution_parent_sha256": sha256_file(DISTRIBUTION),
        "v2_parent_oof_sha256": sha256_file(OOF),
        "runtime_seconds": round(time.perf_counter() - started, 2),
        "new_2024_evaluation": False,
        "lockbox_reopened": False,
        "external_actions": [],
    }
    receipt_path = OUTPUT / f"{CANDIDATE_ID}-dev-2023-Q3.json"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
