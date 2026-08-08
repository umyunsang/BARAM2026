"""Build a conservative three-fold analog-profile test challenger.

This is a development challenger, not independent promotion evidence: Q2, Q3,
and Q4 are all used to choose the recipe.  The analog target library remains
strictly pre-2024 and the consumed 2024 lockbox is not materialized.
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
    _feature_sets,
    _selected_profile,
)
from run_stable_daily_analog_profile import (
    DISTRIBUTION,
    _candidate_scores,
    _fold_parent,
)
from strict_dev_surface import DEV_CUTOFF, development_surface

from baram.contracts.hashing import canonical_sha256, sha256_file
from baram.submission.build import build_submission
from baram.submission.validate import validate_submission

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "artifacts" / "cache" / OPEN_SHA
OUTPUT = ROOT / "artifacts" / "backtests" / "metric-aligned-probe"
SUBMISSIONS = ROOT / "artifacts" / "submissions"
PARENT_CSV = SUBMISSIONS / "E0_V2_SEQUENCE_TRANSFER-a5111cee22bc.csv"
PARENT_RECEIPT = SUBMISSIONS / "E0_V2_SEQUENCE_TRANSFER-a5111cee22bc.receipt.json"
MODEL_ID = "M234_ROBUST_ANALOG_PROFILE_Q3"


def _select_three_fold(
    scores: dict[str, dict[Recipe, float]],
    baselines: dict[str, float],
) -> tuple[Recipe | None, dict[str, object]]:
    common = set.intersection(*(set(fold_scores) for fold_scores in scores.values()))
    stable = [
        recipe
        for recipe in common
        if all(scores[fold][recipe] > baselines[fold] for fold in scores)
    ]
    if not stable:
        return None, {
            "common_candidates": len(common),
            "three_fold_positive_candidates": 0,
        }
    selected = max(
        stable,
        key=lambda recipe: (
            min(scores[fold][recipe] - baselines[fold] for fold in scores),
            float(
                np.mean(
                    [scores[fold][recipe] - baselines[fold] for fold in scores]
                )
            ),
            -recipe.blend_weight,
            recipe.neighbors,
            recipe.representation,
            recipe.kernel,
            recipe.head,
            recipe.transform,
        ),
    )
    deltas = {
        fold: scores[fold][selected] - baselines[fold] for fold in scores
    }
    return selected, {
        "common_candidates": len(common),
        "three_fold_positive_candidates": len(stable),
        "deltas": deltas,
        "worst_fold_delta": min(deltas.values()),
        "mean_fold_delta": float(np.mean(list(deltas.values()))),
    }


def _complete_issuances(
    surface: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    group_id: int,
) -> set[pd.Timestamp]:
    _, issuances, _, _ = _complete_group_days(
        surface,
        group_id,
        feature_sets["core"],
    )
    return set(pd.to_datetime(issuances))


def _test_surface(feature_names: list[str]) -> pd.DataFrame:
    columns = [
        "forecast_id",
        "forecast_kst_dtm",
        "data_available_kst_dtm",
        "group_id",
        *feature_names,
    ]
    test = pd.read_parquet(CACHE / "test_features.parquet", columns=columns)
    test["forecast_kst_dtm"] = pd.to_datetime(test["forecast_kst_dtm"])
    test["data_available_kst_dtm"] = pd.to_datetime(
        test["data_available_kst_dtm"]
    )
    test["actual_kwh"] = np.nan
    if (
        test["forecast_kst_dtm"].min() != pd.Timestamp("2025-01-01 01:00:00")
        or test["forecast_kst_dtm"].max() != pd.Timestamp("2026-01-01 00:00:00")
        or int(test["forecast_kst_dtm"].dt.year.eq(2026).sum()) != 3
    ):
        raise RuntimeError("test inference year contract changed")
    sizes = test.groupby(["data_available_kst_dtm", "group_id"], sort=False).size()
    if len(sizes) != 365 * 3 or not sizes.eq(24).all():
        raise RuntimeError("test daily analog topology changed")
    return test


def _test_parent_long(test: pd.DataFrame) -> pd.DataFrame:
    parent = pd.read_csv(PARENT_CSV, encoding="utf-8-sig")
    parent["forecast_kst_dtm"] = pd.to_datetime(parent["forecast_kst_dtm"])
    merged = test[
        ["forecast_id", "forecast_kst_dtm", "data_available_kst_dtm", "group_id"]
    ].merge(
        parent,
        on=["forecast_id", "forecast_kst_dtm"],
        how="left",
        validate="many_to_one",
    )
    if merged[[f"kpx_group_{group_id}" for group_id in CAPACITIES]].isna().any().any():
        raise RuntimeError("test parent join is incomplete")
    merged["prediction_kwh"] = np.select(
        [merged["group_id"].eq(group_id) for group_id in CAPACITIES],
        [merged[f"kpx_group_{group_id}"] for group_id in CAPACITIES],
        default=np.nan,
    )
    if merged["prediction_kwh"].isna().any():
        raise RuntimeError("test parent long conversion failed")
    return merged[
        [
            "forecast_id",
            "forecast_kst_dtm",
            "group_id",
            "data_available_kst_dtm",
            "prediction_kwh",
        ]
    ]


def _wide_submission(long: pd.DataFrame) -> pd.DataFrame:
    wide = long.pivot(
        index=["forecast_id", "forecast_kst_dtm"],
        columns="group_id",
        values="prediction_kwh",
    ).reset_index()
    wide.columns.name = None
    wide = wide.rename(
        columns={group_id: f"kpx_group_{group_id}" for group_id in CAPACITIES}
    )
    return wide[
        [
            "forecast_id",
            "forecast_kst_dtm",
            "kpx_group_1",
            "kpx_group_2",
            "kpx_group_3",
        ]
    ]


def main() -> None:
    if sha256_file(OPEN) != OPEN_SHA or sha256_file(BASELINE) != BASELINE_SHA:
        raise RuntimeError("immutable competition input hash mismatch")
    parent_receipt = json.loads(PARENT_RECEIPT.read_text(encoding="utf-8"))
    parent_csv_sha = parent_receipt["submission_receipt"]["csv_sha256"]
    if sha256_file(PARENT_CSV) != parent_csv_sha:
        raise RuntimeError("active sequence parent CSV hash mismatch")
    surface, numeric = development_surface()
    if surface["forecast_kst_dtm"].ge(DEV_CUTOFF).any():
        raise RuntimeError("lockbox row reached robust analog builder")
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
    q3_parent = _apply_long(
        oof.loc[oof["fold_id"].eq("dev-2023-Q3")].reset_index(drop=True),
        EXPECTED_SELECTIONS,
    )
    q4_parent = _apply_long(
        oof.loc[oof["fold_id"].eq("dev-2023-Q4")].reset_index(drop=True),
        EXPECTED_SELECTIONS,
    )
    parents = {"q2": q2_parent, "q3": q3_parent, "q4": q4_parent}

    selections: dict[int, Recipe] = {}
    diagnostics: dict[str, object] = {}
    development_replacements: dict[str, list[pd.DataFrame]] = {
        fold: [] for fold in parents
    }
    for group_id in CAPACITIES:
        complete = _complete_issuances(surface, feature_sets, group_id)
        fold_scores: dict[str, dict[Recipe, float]] = {}
        baselines: dict[str, float] = {}
        scoring_parents: dict[str, pd.DataFrame] = {}
        for fold, parent in parents.items():
            scoring_parent = parent.loc[
                parent["data_available_kst_dtm"].isin(complete)
            ].reset_index(drop=True)
            scoring_parents[fold] = scoring_parent
            query_issuances = np.sort(
                scoring_parent["data_available_kst_dtm"].unique()
            )
            fold_scores[fold], _ = _candidate_scores(
                surface,
                feature_sets,
                scoring_parent,
                group_id,
                query_issuances,
            )
            baselines[fold] = _group_score(scoring_parent, group_id)["total"]
        recipe, stability = _select_three_fold(fold_scores, baselines)
        diagnostics[str(group_id)] = {
            "selection": stability,
            "recipe": asdict(recipe) if recipe is not None else None,
            "scoring_parent_rows": {
                fold: len(frame) for fold, frame in scoring_parents.items()
            },
        }
        if recipe is None:
            continue
        selections[group_id] = recipe
        for fold, parent in parents.items():
            query_issuances = np.sort(parent["data_available_kst_dtm"].unique())
            profile, retrieval = _selected_profile(
                surface,
                feature_sets,
                group_id,
                query_issuances,
                recipe,
            )
            replacement = _apply_recipe(parent, profile, group_id, recipe)
            development_replacements[fold].append(replacement)
            diagnostics[str(group_id)][f"{fold}_parent"] = _group_score(
                parent, group_id
            )
            diagnostics[str(group_id)][f"{fold}_selected"] = _group_score(
                replacement, group_id
            )
            diagnostics[str(group_id)][f"{fold}_retrieval"] = retrieval

    development_outputs = {
        fold: _combine(parents[fold], replacements)
        for fold, replacements in development_replacements.items()
    }
    development_path = OUTPUT / f"{MODEL_ID}-dev-2023-Q3.parquet"
    development_outputs["q3"].assign(
        fold_id="dev-2023-Q3",
        model_id=MODEL_ID,
    ).to_parquet(development_path, index=False)

    all_features = sorted(
        set(feature_sets["core"]) | set(feature_sets["extended"])
    )
    test = _test_surface(all_features)
    train_columns = [
        "forecast_id",
        "forecast_kst_dtm",
        "data_available_kst_dtm",
        "group_id",
        "actual_kwh",
        *all_features,
    ]
    analog_surface = pd.concat(
        [surface[train_columns], test[train_columns]],
        ignore_index=True,
    )
    if set(analog_surface["forecast_kst_dtm"].dt.year.unique()) != {
        2022,
        2023,
        2025,
        2026,
    }:
        raise RuntimeError("analog inference surface crossed the frozen year boundary")
    test_parent = _test_parent_long(test)
    test_replacements: list[pd.DataFrame] = []
    test_issuances = np.sort(test["data_available_kst_dtm"].unique())
    test_retrieval: dict[str, object] = {}
    for group_id, recipe in selections.items():
        profile, retrieval = _selected_profile(
            analog_surface,
            feature_sets,
            group_id,
            test_issuances,
            recipe,
        )
        test_replacements.append(
            _apply_recipe(test_parent, profile, group_id, recipe)
        )
        test_retrieval[str(group_id)] = retrieval
    test_output = _combine(test_parent, test_replacements)
    wide = _wide_submission(test_output)

    policy = {
        "architecture": "q2_q3_q4_worst_fold_robust_daily_analog_profile",
        "selection_rule": "maximize worst positive delta over Q2, Q3, and Q4",
        "recipes": {
            str(group_id): asdict(recipe)
            for group_id, recipe in selections.items()
        },
        "analog_target_history": "2022-2023 complete issuance days only",
        "parent_csv_sha256": parent_csv_sha,
        "q4_used_in_recipe_selection": True,
        "independent_transfer_evidence": False,
    }
    policy_sha = canonical_sha256(policy)
    candidate_id = f"E0_ROBUST_ANALOG_DEV-{policy_sha[:12]}"
    candidate_path = SUBMISSIONS / f"{candidate_id}.csv"
    sample = pd.read_parquet(CACHE / "submission_keys.parquet")
    csv_sha = build_submission(sample, wide, candidate_path)
    validation = validate_submission(
        candidate_path,
        sample,
        candidate_id=candidate_id,
        source_sha256=OPEN_SHA,
        champion_policy_sha256=policy_sha,
        cap_modes={1: "capacity", 2: "capacity", 3: "capacity"},
    )
    if validation.csv_sha256 != csv_sha:
        raise RuntimeError("robust analog build and validation hashes differ")
    receipt = {
        "schema_version": 1,
        "state": "LOCAL_ROBUST_ANALOG_DEVELOPMENT_CHALLENGER_BUILT_NOT_UPLOADED",
        "candidate_id": candidate_id,
        "candidate_path": str(candidate_path.relative_to(ROOT)),
        "policy": policy,
        "policy_sha256": policy_sha,
        "selection_diagnostics": diagnostics,
        "development_scores": {
            fold: {
                "parent": _score(parents[fold]),
                "selected": _score(development_outputs[fold]),
                "total_delta": _score(development_outputs[fold])["total"]
                - _score(parents[fold])["total"],
            }
            for fold in parents
        },
        "q4_paired_bootstrap": _paired_issuance_bootstrap(
            q4_parent, development_outputs["q4"]
        ),
        "test_retrieval": test_retrieval,
        "submission_receipt": asdict(validation),
        "development_prediction_path": str(development_path.relative_to(ROOT)),
        "development_prediction_sha256": sha256_file(development_path),
        "parent_path": str(PARENT_CSV.relative_to(ROOT)),
        "parent_receipt_path": str(PARENT_RECEIPT.relative_to(ROOT)),
        "parent_model_lineage_sha256": parent_receipt[
            "parent_model_lineage_sha256"
        ],
        "online_score": None,
        "q4_used_in_recipe_selection": True,
        "independent_transfer_evidence": False,
        "new_2024_evaluation": False,
        "lockbox_reopened": False,
        "no_external_upload": True,
        "external_actions": [],
    }
    receipt_path = candidate_path.with_suffix(".receipt.json")
    receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
