"""Evaluate one predeclared support gate over the frozen M234 recipes.

The gate applies an M234 group-day correction only when both its nearest and
recipe-kth analog distances are within the historical train-day leave-one-out
95th percentile.  No threshold search is performed, and the consumed 2024
lockbox is never loaded or scored.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from audit_analog_quarter_support import _percentile, _projected_distances
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
    _feature_sets,
    _selected_profile,
)
from run_stable_daily_analog_profile import DISTRIBUTION, _fold_parent
from strict_dev_surface import DEV_CUTOFF, development_surface

from baram.contracts.hashing import canonical_sha256, sha256_file

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "artifacts" / "backtests" / "metric-aligned-probe"
M234 = ROOT / "artifacts" / "submissions" / "E0_ROBUST_ANALOG_DEV-08f07b6d9e87.csv"
M234_RECEIPT = M234.with_suffix(".receipt.json")
MODEL_ID = "M237_SUPPORT_GATED_ANALOG_Q234"
SUPPORT_THRESHOLD = 0.95
FOLDS = ("q2", "q3", "q4")


def _support_gate(
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
            f"group {group_id} support-gate days changed: "
            f"train={int(train_mask.sum())}, query={int(query_mask.sum())}"
        )
    query_distance, train_loo_distance, projection = _projected_distances(
        values[train_mask],
        values[query_mask],
        issuances[train_mask],
        issuances[query_mask],
        representation,
    )
    nearest_percentile = _percentile(
        train_loo_distance[:, 0],
        query_distance[:, 0],
    )
    kth_percentile = _percentile(
        train_loo_distance[:, recipe.neighbors - 1],
        query_distance[:, recipe.neighbors - 1],
    )
    supported = (nearest_percentile <= SUPPORT_THRESHOLD) & (
        kth_percentile <= SUPPORT_THRESHOLD
    )
    gate = pd.DataFrame(
        {
            "data_available_kst_dtm": pd.to_datetime(issuances[query_mask]),
            "nearest_loo_percentile": nearest_percentile,
            "recipe_kth_loo_percentile": kth_percentile,
            "support_gate": supported,
        }
    )
    return gate, {
        "training_days": int(train_mask.sum()),
        "query_days": int(query_mask.sum()),
        "supported_days": int(supported.sum()),
        "fallback_days": int((~supported).sum()),
        "support_fraction": float(np.mean(supported)),
        "threshold": SUPPORT_THRESHOLD,
        "rule": "nearest_loo_percentile<=0.95 and recipe_kth_loo_percentile<=0.95",
        "nearest_percentile_max": float(np.max(nearest_percentile)),
        "recipe_kth_percentile_max": float(np.max(kth_percentile)),
        "projection": projection,
    }


def _apply_support_gate(
    parent: pd.DataFrame,
    full_analog: pd.DataFrame,
    gate: pd.DataFrame,
    group_id: int,
) -> pd.DataFrame:
    keys = [
        "forecast_id",
        "forecast_kst_dtm",
        "group_id",
        "data_available_kst_dtm",
    ]
    parent_group = parent.loc[parent["group_id"].eq(group_id), [*keys, "prediction_kwh"]]
    candidate = full_analog.merge(
        parent_group.rename(columns={"prediction_kwh": "parent_prediction_kwh"}),
        on=keys,
        how="left",
        validate="one_to_one",
    ).merge(
        gate[["data_available_kst_dtm", "support_gate"]],
        on="data_available_kst_dtm",
        how="left",
        validate="many_to_one",
    )
    if candidate["parent_prediction_kwh"].isna().any():
        raise RuntimeError(f"group {group_id} parent alignment changed")
    supported = candidate["support_gate"].eq(True).to_numpy(dtype=bool)
    candidate["prediction_kwh"] = np.where(
        supported,
        candidate["prediction_kwh"],
        candidate["parent_prediction_kwh"],
    )
    return candidate[parent.columns]


def _parents(
    surface: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
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
    return {"q2": q2_parent, "q3": q3_parent, "q4": q4_parent}, metadata


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
        raise RuntimeError("lockbox row reached M237 support-gate runner")
    feature_sets = _feature_sets(numeric)
    parents, _ = _parents(surface)
    query_issuances = {
        fold: np.sort(parent["data_available_kst_dtm"].unique())
        for fold, parent in parents.items()
    }

    full_replacements: dict[str, list[pd.DataFrame]] = {fold: [] for fold in FOLDS}
    gated_replacements: dict[str, list[pd.DataFrame]] = {fold: [] for fold in FOLDS}
    gate_diagnostics: dict[str, object] = {}
    for group_id, recipe in recipes.items():
        group_diagnostics: dict[str, object] = {"recipe": asdict(recipe)}
        for fold in FOLDS:
            parent = parents[fold]
            profile, retrieval = _selected_profile(
                surface,
                feature_sets,
                group_id,
                query_issuances[fold],
                recipe,
            )
            full = _apply_recipe(parent, profile, group_id, recipe)
            gate, support = _support_gate(
                surface,
                feature_sets,
                group_id,
                query_issuances[fold],
                recipe,
            )
            gated = _apply_support_gate(parent, full, gate, group_id)
            full_replacements[fold].append(full)
            gated_replacements[fold].append(gated)
            group_diagnostics[fold] = {
                "parent": _group_score(parent, group_id),
                "full_analog": _group_score(full, group_id),
                "support_gated": _group_score(gated, group_id),
                "retrieval": retrieval,
                "support": support,
            }
        gate_diagnostics[str(group_id)] = group_diagnostics

    full_outputs = {
        fold: _combine(parents[fold], full_replacements[fold]) for fold in FOLDS
    }
    gated_outputs = {
        fold: _combine(parents[fold], gated_replacements[fold]) for fold in FOLDS
    }
    scores: dict[str, object] = {}
    full_deltas: dict[str, float] = {}
    gated_deltas: dict[str, float] = {}
    for fold in FOLDS:
        parent_score = _score(parents[fold])
        full_score = _score(full_outputs[fold])
        gated_score = _score(gated_outputs[fold])
        expected = m234_receipt["development_scores"][fold]
        if abs(full_score["total"] - expected["selected"]["total"]) > 1e-12:
            raise RuntimeError(f"M234 {fold} reproduction score changed")
        full_deltas[fold] = full_score["total"] - parent_score["total"]
        gated_deltas[fold] = gated_score["total"] - parent_score["total"]
        scores[fold] = {
            "parent": parent_score,
            "full_analog": full_score,
            "support_gated": gated_score,
            "full_total_delta": full_deltas[fold],
            "gated_total_delta": gated_deltas[fold],
            "gated_minus_full_total": gated_score["total"] - full_score["total"],
        }

    full_q4_bootstrap = _paired_issuance_bootstrap(parents["q4"], full_outputs["q4"])
    gated_q4_bootstrap = _paired_issuance_bootstrap(
        parents["q4"], gated_outputs["q4"]
    )
    all_positive = all(delta > 0.0 for delta in gated_deltas.values())
    improves_worst_fold = min(gated_deltas.values()) > min(full_deltas.values())
    improves_q4_robustness = (
        gated_deltas["q4"] > full_deltas["q4"]
        and gated_q4_bootstrap["positive_fraction"]
        > full_q4_bootstrap["positive_fraction"]
    )
    promoted = all_positive and (improves_worst_fold or improves_q4_robustness)
    promotion = {
        "promoted": promoted,
        "all_q2_q3_q4_deltas_positive": all_positive,
        "full_worst_fold_delta": min(full_deltas.values()),
        "gated_worst_fold_delta": min(gated_deltas.values()),
        "improves_worst_fold": improves_worst_fold,
        "improves_q4_robustness": improves_q4_robustness,
        "rule": (
            "all gated fold deltas positive and either gated worst-fold delta "
            "strictly exceeds full M234 or both Q4 delta and paired-bootstrap "
            "positive fraction strictly exceed full M234"
        ),
    }

    prediction_path = OUTPUT / f"{MODEL_ID}-dev-2023-Q3.parquet"
    gated_outputs["q3"].assign(
        fold_id="dev-2023-Q3",
        model_id=MODEL_ID,
    )[[*METRIC_COLUMNS, "fold_id", "model_id", "data_available_kst_dtm"]].to_parquet(
        prediction_path,
        index=False,
    )
    policy = {
        "architecture": "frozen_m234_with_train_loo_support_fallback",
        "recipes": {str(group_id): asdict(recipe) for group_id, recipe in recipes.items()},
        "support_threshold": SUPPORT_THRESHOLD,
        "support_rule": (
            "nearest_loo_percentile<=0.95 and "
            "recipe_kth_loo_percentile<=0.95"
        ),
        "fallback": "M231 group-day prediction",
        "q1_policy_if_promoted": "unconditional M231 fallback",
        "threshold_search": False,
    }
    receipt = {
        "schema_version": 1,
        "candidate_id": MODEL_ID,
        "state": (
            "LOCAL_SUPPORT_GATE_PROMOTED_FOR_TEST_BUILD"
            if promoted
            else "LOCAL_SUPPORT_GATE_REJECTED_NO_TEST_BUILD"
        ),
        "policy": policy,
        "policy_sha256": canonical_sha256(policy),
        "scores": scores,
        "gate_diagnostics": gate_diagnostics,
        "full_q4_paired_bootstrap": full_q4_bootstrap,
        "gated_q4_paired_bootstrap": gated_q4_bootstrap,
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
