"""Evaluate one exact composition of the promoted M241 and M240 policies."""

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
    _feature_sets,
    _selected_profile,
)
from run_recency_weighted_analog_transfer import (
    HALF_LIFE_DAYS,
    _recency_profile,
)
from run_spread_shrunk_analog_transfer import (
    REFERENCE_QUANTILE,
    _apply_spread_recipe,
    _spread_adjusted_profile,
)
from run_stable_daily_analog_profile import DISTRIBUTION, _fold_parent
from strict_dev_surface import DEV_CUTOFF, development_surface

from baram.contracts.hashing import canonical_sha256, sha256_file

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "artifacts" / "backtests" / "metric-aligned-probe"
M234 = ROOT / "artifacts" / "submissions" / "E0_ROBUST_ANALOG_DEV-08f07b6d9e87.csv"
M234_RECEIPT = M234.with_suffix(".receipt.json")
M240_RECEIPT = OUTPUT / "M240_SPREAD_SHRUNK_ANALOG_Q234.json"
M241_RECEIPT = OUTPUT / "M241_RECENCY_WEIGHTED_ANALOG_Q234.json"
M240_RECEIPT_SHA = (
    "5f62fb05e1233e9c35a3dcab2cf011968142edf598b4b4df9ba76b3fe9c1a62d"
)
M241_RECEIPT_SHA = (
    "6e487d0cbdf179c22c6d21a1a76fd653b961980e201b855af814155e838e7564"
)
MODEL_ID = "M243_RECENCY_SPREAD_ANALOG_Q234"
FOLDS = ("q2", "q3", "q4")


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


def _composed_profile(
    recency_profile: pd.DataFrame,
    spread_profile: pd.DataFrame,
) -> pd.DataFrame:
    multiplier = spread_profile[
        ["forecast_id", "group_id", "analog_blend_multiplier"]
    ]
    combined = recency_profile.merge(
        multiplier,
        on=["forecast_id", "group_id"],
        how="left",
        validate="one_to_one",
    )
    if combined["analog_blend_multiplier"].isna().any():
        raise RuntimeError("M243 spread multiplier alignment changed")
    return combined


def main() -> None:
    if sha256_file(OPEN) != OPEN_SHA or sha256_file(BASELINE) != BASELINE_SHA:
        raise RuntimeError("immutable competition input hash mismatch")
    if sha256_file(M240_RECEIPT) != M240_RECEIPT_SHA:
        raise RuntimeError("M240 promoted receipt hash mismatch")
    if sha256_file(M241_RECEIPT) != M241_RECEIPT_SHA:
        raise RuntimeError("M241 promoted receipt hash mismatch")
    m234_receipt = json.loads(M234_RECEIPT.read_text(encoding="utf-8"))
    m241_receipt = json.loads(M241_RECEIPT.read_text(encoding="utf-8"))
    if sha256_file(M234) != m234_receipt["submission_receipt"]["csv_sha256"]:
        raise RuntimeError("M234 CSV hash mismatch")
    recipes = {
        int(group_id): Recipe(**raw)
        for group_id, raw in m234_receipt["policy"]["recipes"].items()
    }

    surface, numeric = development_surface()
    if surface["forecast_kst_dtm"].ge(DEV_CUTOFF).any():
        raise RuntimeError("lockbox row reached M243 composition runner")
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
    composed_replacements: dict[str, list[pd.DataFrame]] = {
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
            spread_profile, spread_retrieval = _spread_adjusted_profile(
                surface,
                feature_sets,
                group_id,
                query_issuances[fold],
                recipe,
            )
            composed_profile = _composed_profile(recency_profile, spread_profile)
            full = _apply_recipe(parent, full_profile, group_id, recipe)
            recency = _apply_recipe(parent, recency_profile, group_id, recipe)
            composed = _apply_spread_recipe(
                parent,
                composed_profile,
                group_id,
                recipe,
            )
            full_replacements[fold].append(full)
            recency_replacements[fold].append(recency)
            composed_replacements[fold].append(composed)
            group_diagnostics[fold] = {
                "parent": _group_score(parent, group_id),
                "full_analog": _group_score(full, group_id),
                "recency_weighted": _group_score(recency, group_id),
                "recency_plus_spread": _group_score(composed, group_id),
                "full_retrieval": full_retrieval,
                "recency_retrieval": recency_retrieval,
                "spread_retrieval": spread_retrieval,
            }
        diagnostics[str(group_id)] = group_diagnostics

    full_outputs = {
        fold: _combine(parents[fold], full_replacements[fold]) for fold in FOLDS
    }
    recency_outputs = {
        fold: _combine(parents[fold], recency_replacements[fold]) for fold in FOLDS
    }
    composed_outputs = {
        fold: _combine(parents[fold], composed_replacements[fold]) for fold in FOLDS
    }
    scores: dict[str, object] = {}
    recency_deltas: dict[str, float] = {}
    composed_deltas: dict[str, float] = {}
    for fold in FOLDS:
        parent_score = _score(parents[fold])
        full_score = _score(full_outputs[fold])
        recency_score = _score(recency_outputs[fold])
        composed_score = _score(composed_outputs[fold])
        expected_full = m234_receipt["development_scores"][fold]["selected"]
        expected_recency = m241_receipt["scores"][fold]["recency_weighted"]
        if abs(full_score["total"] - expected_full["total"]) > 1e-12:
            raise RuntimeError(f"M234 {fold} reproduction score changed")
        if abs(recency_score["total"] - expected_recency["total"]) > 1e-12:
            raise RuntimeError(f"M241 {fold} reproduction score changed")
        recency_deltas[fold] = recency_score["total"] - parent_score["total"]
        composed_deltas[fold] = composed_score["total"] - parent_score["total"]
        scores[fold] = {
            "parent": parent_score,
            "full_analog": full_score,
            "recency_weighted": recency_score,
            "recency_plus_spread": composed_score,
            "recency_total_delta": recency_deltas[fold],
            "composed_total_delta": composed_deltas[fold],
            "composed_minus_recency_total": (
                composed_score["total"] - recency_score["total"]
            ),
        }

    recency_q4_bootstrap = _paired_issuance_bootstrap(
        parents["q4"], recency_outputs["q4"]
    )
    composed_q4_bootstrap = _paired_issuance_bootstrap(
        parents["q4"], composed_outputs["q4"]
    )
    expected_bootstrap = m241_receipt["recency_q4_paired_bootstrap"]
    if (
        abs(recency_q4_bootstrap["mean"] - expected_bootstrap["mean"]) > 1e-12
        or recency_q4_bootstrap["positive_fraction"]
        != expected_bootstrap["positive_fraction"]
    ):
        raise RuntimeError("M241 Q4 bootstrap reproduction changed")
    all_positive = all(delta > 0.0 for delta in composed_deltas.values())
    improves_worst_fold = min(composed_deltas.values()) > min(
        recency_deltas.values()
    )
    improves_q4_robustness = (
        composed_deltas["q4"] > recency_deltas["q4"]
        and composed_q4_bootstrap["positive_fraction"]
        > recency_q4_bootstrap["positive_fraction"]
    )
    promoted = all_positive and (improves_worst_fold or improves_q4_robustness)
    promotion = {
        "promoted": promoted,
        "all_q2_q3_q4_deltas_positive": all_positive,
        "recency_worst_fold_delta": min(recency_deltas.values()),
        "composed_worst_fold_delta": min(composed_deltas.values()),
        "improves_worst_fold": improves_worst_fold,
        "improves_q4_robustness": improves_q4_robustness,
        "rule": (
            "all composed fold deltas positive and either composed worst-fold "
            "delta strictly exceeds M241 or both Q4 delta and paired-bootstrap "
            "positive fraction strictly exceed M241"
        ),
    }

    prediction_path = OUTPUT / f"{MODEL_ID}-dev-2023-Q3.parquet"
    composed_outputs["q3"].assign(
        fold_id="dev-2023-Q3",
        model_id=MODEL_ID,
    )[[*METRIC_COLUMNS, "fold_id", "model_id", "data_available_kst_dtm"]].to_parquet(
        prediction_path,
        index=False,
    )
    policy = {
        "architecture": "exact_m241_head_with_exact_m240_blend_multiplier",
        "recipes": {
            str(group_id): asdict(recipe) for group_id, recipe in recipes.items()
        },
        "recency_half_life_days": HALF_LIFE_DAYS,
        "recency_policy": "exact_M241_neighbor_weighting",
        "spread_reference_quantile": REFERENCE_QUANTILE,
        "spread_policy": "exact_M240_base_neighbor_spread_multiplier",
        "composition": "M241_analog_normalized_plus_M240_blend_multiplier",
        "parameter_search": False,
        "interaction_search": False,
        "group_exception_search": False,
        "q1_policy_if_promoted": "unconditional M231 fallback",
    }
    receipt = {
        "schema_version": 1,
        "candidate_id": MODEL_ID,
        "state": (
            "LOCAL_RECENCY_SPREAD_COMPOSITION_PROMOTED_FOR_TEST_BUILD"
            if promoted
            else "LOCAL_RECENCY_SPREAD_COMPOSITION_REJECTED_NO_TEST_BUILD"
        ),
        "policy": policy,
        "policy_sha256": canonical_sha256(policy),
        "scores": scores,
        "diagnostics": diagnostics,
        "recency_q4_paired_bootstrap": recency_q4_bootstrap,
        "composed_q4_paired_bootstrap": composed_q4_bootstrap,
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
            "m240_receipt_sha256": sha256_file(M240_RECEIPT),
            "m241_receipt_sha256": sha256_file(M241_RECEIPT),
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
