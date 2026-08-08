"""Select daily analog corrections on untouched 2022 group-1/2 diagnostics.

The recipe must improve both a shared-L1 and a shared-Q50 parent on each of
2022 Q2, Q3, and Q4.  Only after that six-surface selection is frozen is it
applied once to the 2023 Q2/Q3/Q4 M231-lineage parents.  Group 3 is unchanged.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from build_robust_analog_test_challenger import (
    _complete_issuances,
    _select_three_fold,
)
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
    _feature_sets,
    _selected_profile,
)
from run_stable_daily_analog_profile import DISTRIBUTION, _candidate_scores, _fold_parent
from strict_dev_surface import DEV_CUTOFF, development_surface

from baram.contracts.hashing import canonical_sha256, sha256_file
from baram.contracts.types import FoldSpec
from baram.models.baselines import predict_bundle
from baram.models.distribution_oof import load_distribution_v2_specs
from baram.models.lightgbm import fit_lgbm_bundle, load_point_v2_specs
from baram.models.oof import filter_complete_validation_rows

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "artifacts" / "backtests" / "metric-aligned-probe"
SPLITS = ROOT / "artifacts" / "manifests" / "splits.json"
POINT_MANIFEST = ROOT / "artifacts" / "manifests" / "backtest-point-v2.json"
POINT_CONFIG = ROOT / "configs" / "models" / "point_v2.yaml"
DIST_CONFIG = ROOT / "configs" / "models" / "distribution_v2.yaml"
MODEL_ID = "M235_GROUP12_ANALOG_TRANSFER"
SEED = 20260801
N_JOBS = 6
GROUP12 = (1, 2)


def _folds() -> tuple[FoldSpec, ...]:
    payload = json.loads(SPLITS.read_text(encoding="utf-8"))
    raw_folds = payload.get("group12_diagnostics")
    if not isinstance(raw_folds, list):
        raise RuntimeError("split receipt has no group12 diagnostic folds")
    folds = tuple(
        FoldSpec(
            fold_id=str(raw["fold_id"]),
            train_batches=tuple(str(value) for value in raw["train_batches"]),
            validation_batches=tuple(str(value) for value in raw["validation_batches"]),
            eligible_groups=tuple(int(value) for value in raw["eligible_groups"]),
            official_total_eligible=bool(raw["official_total_eligible"]),
            is_lockbox=bool(raw.get("is_lockbox", False)),
        )
        for raw in raw_folds
    )
    if tuple(fold.fold_id for fold in folds) != (
        "dev-2022-Q2",
        "dev-2022-Q3",
        "dev-2022-Q4",
    ):
        raise RuntimeError("group12 diagnostic fold identity changed")
    if any(
        fold.eligible_groups != GROUP12
        or fold.official_total_eligible
        or fold.is_lockbox
        for fold in folds
    ):
        raise RuntimeError("group12 diagnostic scope changed")
    return folds


def _model_contract() -> tuple[tuple[str, ...], dict[str, dict[str, object]]]:
    point = json.loads(POINT_MANIFEST.read_text(encoding="utf-8"))
    feature_names = tuple(str(value) for value in point["feature_names"])
    if len(feature_names) != 704:
        raise RuntimeError("frozen v2 point feature count changed")
    point_specs = {
        item.candidate_id: item for item in load_point_v2_specs(POINT_CONFIG)
    }
    distribution_specs = {
        item.candidate_id: item
        for item in load_distribution_v2_specs(DIST_CONFIG)
    }
    l1 = point_specs["P0_SHARED_L1"]
    q50 = distribution_specs["D1_LGBM_SHARED_BASE"]
    if l1.architecture != "shared" or q50.architecture != "shared":
        raise RuntimeError("shared parent architecture changed")
    return feature_names, {
        "shared_l1": dict(l1.params),
        "shared_q50": {
            **dict(q50.params),
            "objective": "quantile",
            "alpha": 0.50,
        },
    }


def _parent_oof(
    surface: pd.DataFrame,
    feature_names: tuple[str, ...],
    folds: tuple[FoldSpec, ...],
    contracts: dict[str, dict[str, object]],
) -> tuple[pd.DataFrame, dict[str, object]]:
    shared_features = (*feature_names, "group_id", "capacity_kwh")
    outputs: list[pd.DataFrame] = []
    diagnostics: dict[str, object] = {}
    for family, params in contracts.items():
        diagnostics[family] = {}
        for fold in folds:
            train = surface.loc[
                surface["issuance_batch"].isin(fold.train_batches)
                & surface["group_id"].isin(GROUP12)
                & surface["actual_kwh"].notna()
            ].reset_index(drop=True)
            valid = surface.loc[
                surface["issuance_batch"].isin(fold.validation_batches)
            ].reset_index(drop=True)
            valid = filter_complete_validation_rows(valid, GROUP12)
            if train.empty or valid.empty:
                raise RuntimeError(f"empty 2022 parent fold: {family}/{fold.fold_id}")
            if (
                valid["operating_year"].ne(2022).any()
                or valid["forecast_kst_dtm"].max()
                > pd.Timestamp("2023-01-01 00:00:00")
            ):
                raise RuntimeError("non-2022 operating period reached group12 parent")
            target = train["actual_kwh"] / train["capacity_kwh"]
            bundle = fit_lgbm_bundle(
                train[list(shared_features)],
                target,
                train["issuance_batch"],
                shared_features,
                fold.fold_id,
                None,
                1.0,
                params,
                SEED,
                N_JOBS,
            )
            normalized = predict_bundle(
                bundle,
                valid[list(shared_features)].reset_index(drop=True),
                fold.fold_id,
            )
            parent = valid[
                [
                    "forecast_id",
                    "forecast_kst_dtm",
                    "group_id",
                    "actual_kwh",
                    "data_available_kst_dtm",
                ]
            ].copy()
            parent["prediction_kwh"] = np.clip(
                normalized * valid["capacity_kwh"].to_numpy(dtype=float),
                0.0,
                valid["capacity_kwh"].to_numpy(dtype=float),
            )
            parent = _apply_long(
                parent,
                {group_id: EXPECTED_SELECTIONS[group_id] for group_id in GROUP12},
            )
            parent["fold_id"] = fold.fold_id
            parent["parent_family"] = family
            outputs.append(parent)
            diagnostics[family][fold.fold_id] = {
                "training_rows": len(train),
                "validation_rows": len(parent),
                "selected_iteration": bundle.estimator.n_estimators_,
                "model_manifest": asdict(bundle.manifest),
            }
    combined = pd.concat(outputs, ignore_index=True).sort_values(
        ["parent_family", "fold_id", "forecast_kst_dtm", "group_id"],
        kind="stable",
    )
    keys = ["parent_family", "fold_id", "forecast_id", "group_id"]
    if combined.duplicated(keys).any():
        raise RuntimeError("group12 parent OOF keys are duplicated")
    return combined.reset_index(drop=True), diagnostics


def _parents_2023(
    surface: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    metadata = surface[
        ["forecast_id", "forecast_kst_dtm", "group_id", "data_available_kst_dtm"]
    ]
    distribution = pd.read_parquet(DISTRIBUTION)
    q2 = _fold_parent(distribution, metadata, "dev-2023-Q2")
    oof = pd.read_parquet(OOF).merge(
        metadata,
        on=["forecast_id", "forecast_kst_dtm", "group_id"],
        validate="one_to_one",
    )
    q3 = _apply_long(
        oof.loc[oof["fold_id"].eq("dev-2023-Q3")].reset_index(drop=True),
        EXPECTED_SELECTIONS,
    )
    q4 = _apply_long(
        oof.loc[oof["fold_id"].eq("dev-2023-Q4")].reset_index(drop=True),
        EXPECTED_SELECTIONS,
    )
    return {"q2": q2, "q3": q3, "q4": q4}


def main() -> None:
    if sha256_file(OPEN) != OPEN_SHA or sha256_file(BASELINE) != BASELINE_SHA:
        raise RuntimeError("immutable competition input hash mismatch")
    started = time.perf_counter()
    surface, numeric = development_surface()
    if surface["forecast_kst_dtm"].ge(DEV_CUTOFF).any():
        raise RuntimeError("lockbox row reached group12 analog runner")
    folds = _folds()
    feature_names, contracts = _model_contract()
    missing = sorted(set(feature_names) - set(surface))
    if missing:
        raise RuntimeError(f"frozen feature contract missing: {missing[:5]}")
    feature_sets = _feature_sets(numeric)
    parent_2022, model_diagnostics = _parent_oof(
        surface,
        feature_names,
        folds,
        contracts,
    )

    selections: dict[int, Recipe] = {}
    selection_diagnostics: dict[str, object] = {}
    surface_names = [
        f"{family}:{fold.fold_id}"
        for family in contracts
        for fold in folds
    ]
    for group_id in GROUP12:
        fold_scores: dict[str, dict[Recipe, float]] = {}
        baselines: dict[str, float] = {}
        for family in contracts:
            for fold in folds:
                name = f"{family}:{fold.fold_id}"
                parent = parent_2022.loc[
                    parent_2022["parent_family"].eq(family)
                    & parent_2022["fold_id"].eq(fold.fold_id)
                ].reset_index(drop=True)
                complete = _complete_issuances(surface, feature_sets, group_id)
                parent = parent.loc[
                    parent["data_available_kst_dtm"].isin(complete)
                ].reset_index(drop=True)
                query_issuances = np.sort(
                    parent["data_available_kst_dtm"].unique()
                )
                fold_scores[name], _ = _candidate_scores(
                    surface,
                    feature_sets,
                    parent,
                    group_id,
                    query_issuances,
                )
                baselines[name] = _group_score(parent, group_id)["total"]
        recipe, stability = _select_three_fold(fold_scores, baselines)
        selection_diagnostics[str(group_id)] = {
            "selection": stability,
            "recipe": asdict(recipe) if recipe is not None else None,
            "selection_surfaces": surface_names,
        }
        if recipe is not None:
            selections[group_id] = recipe

    parents = _parents_2023(surface)
    replacements: dict[str, list[pd.DataFrame]] = {fold: [] for fold in parents}
    transfer_diagnostics: dict[str, object] = {}
    for group_id, recipe in selections.items():
        transfer_diagnostics[str(group_id)] = {}
        for fold, parent in parents.items():
            issuances = np.sort(parent["data_available_kst_dtm"].unique())
            profile, retrieval = _selected_profile(
                surface,
                feature_sets,
                group_id,
                issuances,
                recipe,
            )
            replacement = _apply_recipe(parent, profile, group_id, recipe)
            replacements[fold].append(replacement)
            before = _group_score(parent, group_id)
            after = _group_score(replacement, group_id)
            transfer_diagnostics[str(group_id)][fold] = {
                "parent": before,
                "selected": after,
                "total_delta": after["total"] - before["total"],
                "retrieval": retrieval,
            }
    outputs = {
        fold: _combine(parents[fold], replacements[fold]) for fold in parents
    }
    development_scores = {
        fold: {
            "parent": _score(parents[fold]),
            "selected": _score(outputs[fold]),
            "total_delta": _score(outputs[fold])["total"]
            - _score(parents[fold])["total"],
        }
        for fold in parents
    }
    transfer_pass = bool(selections) and all(
        development_scores[fold]["total_delta"] > 0.0 for fold in parents
    ) and all(
        transfer_diagnostics[str(group_id)][fold]["total_delta"] > 0.0
        for group_id in selections
        for fold in parents
    )

    parent_path = OUTPUT / f"{MODEL_ID}-group12-parents.parquet"
    parent_2022.to_parquet(parent_path, index=False)
    prediction_path = OUTPUT / f"{MODEL_ID}-dev-2023-Q3.parquet"
    outputs["q3"].assign(
        fold_id="dev-2023-Q3",
        model_id=MODEL_ID,
    )[[*METRIC_COLUMNS, "fold_id", "model_id", "data_available_kst_dtm"]].to_parquet(
        prediction_path,
        index=False,
    )
    policy = {
        "architecture": "six_surface_2022_group12_daily_analog_transfer",
        "selection_surfaces": surface_names,
        "selection_rule": "positive on every surface, maximize worst then mean delta",
        "recipes": {
            str(group_id): asdict(recipe)
            for group_id, recipe in selections.items()
        },
        "group3_action": "identity_over_M231_parent",
    }
    receipt = {
        "schema_version": 1,
        "candidate_id": MODEL_ID,
        "state": (
            "LOCAL_INDEPENDENT_TRANSFER_PASS_TEST_BUILD_ELIGIBLE"
            if transfer_pass
            else "LOCAL_INDEPENDENT_TRANSFER_REJECTED"
        ),
        "policy": policy,
        "policy_sha256": canonical_sha256(policy),
        "model_diagnostics": model_diagnostics,
        "selection_diagnostics": selection_diagnostics,
        "transfer_diagnostics": transfer_diagnostics,
        "development_scores": development_scores,
        "q4_paired_bootstrap": _paired_issuance_bootstrap(
            parents["q4"], outputs["q4"]
        ),
        "transfer_pass": transfer_pass,
        "test_build_eligible": transfer_pass,
        "group12_parent_path": str(parent_path.relative_to(ROOT)),
        "group12_parent_sha256": sha256_file(parent_path),
        "prediction_path": str(prediction_path.relative_to(ROOT)),
        "prediction_sha256": sha256_file(prediction_path),
        "input_hashes": {
            "open_zip": OPEN_SHA,
            "baseline_notebook": BASELINE_SHA,
            "splits": sha256_file(SPLITS),
            "point_manifest": sha256_file(POINT_MANIFEST),
            "point_config": sha256_file(POINT_CONFIG),
            "distribution_config": sha256_file(DIST_CONFIG),
            "distribution_parent": sha256_file(DISTRIBUTION),
            "v2_parent_oof": sha256_file(OOF),
        },
        "selection_year": 2022,
        "transfer_year": 2023,
        "q4_used_in_recipe_selection": False,
        "independent_transfer_evidence": True,
        "new_2024_evaluation": False,
        "lockbox_reopened": False,
        "no_external_upload": True,
        "external_actions": [],
    }
    receipt_path = OUTPUT / f"{MODEL_ID}-dev-2023-Q3.json"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "receipt": receipt,
                "runtime_seconds": round(time.perf_counter() - started, 2),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
