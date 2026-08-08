"""Screen one strict convex ensemble of exact M107 and exact M244 lineages.

M134 predates the later analog lineage.  This runner reconstructs the complete
M244 Q2-Q4 prediction surface, verifies its recorded scores, and selects one
small per-group M244 mass that improves both Q2 and Q3 over M107.  The frozen
mass transfers once to Q4.  No calibration, dynamic gating, or test build is
performed here.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from build_v2_transfer_sequence_challenger import (
    CAPACITIES,
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
    Recipe,
    _combine,
    _feature_sets,
)
from run_rare_event_corrected_analog_transfer import _rare_event_profile
from run_recency_spread_analog_transfer import _composed_profile, _parents
from run_spread_shrunk_analog_transfer import (
    _apply_spread_recipe,
    _spread_adjusted_profile,
)
from run_strict_parent_analog_transfer import FOLD_MAP, _pooled, _strict_parents
from run_temporal_reconciliation import _verified_m107_receipt
from strict_dev_surface import DEV_CUTOFF, development_surface

from baram.contracts.hashing import canonical_sha256, sha256_file

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "artifacts" / "backtests" / "metric-aligned-probe"
MODEL_ID = "M263_INDEPENDENT_LINEAGE_CONVEX_ENSEMBLE"
M244_RECEIPT = OUTPUT / "M244_RARE_EVENT_CORRECTED_ANALOG_Q234.json"
M244_RECEIPT_SHA = (
    "8489f2759ce39fe4abb7c5179386cde0e1a39d22a5d07673a782b4163410a371"
)
M244_PREDICTION_SHA = (
    "fb007b62d07c9d15757b0d449b0f11cc165925e6631f173a9c2f6e87b9eb7598"
)
M244_MASSES = (0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25)
FOLDS = ("q2", "q3", "q4")
KEYS = ["forecast_id", "forecast_kst_dtm", "group_id"]


def _verified_m244_receipt() -> dict[str, object]:
    if sha256_file(M244_RECEIPT) != M244_RECEIPT_SHA:
        raise RuntimeError("M244 promoted receipt hash mismatch")
    receipt = json.loads(M244_RECEIPT.read_text(encoding="utf-8"))
    if (
        receipt["candidate_id"] != "M244_RARE_EVENT_CORRECTED_ANALOG_Q234"
        or receipt["state"] != "LOCAL_RARE_EVENT_ANALOG_MOS_PROMOTED_FOR_TEST_BUILD"
        or receipt["prediction_sha256"] != M244_PREDICTION_SHA
        or not receipt["promotion"]["promoted"]
        or receipt.get("new_2024_evaluation")
        or receipt.get("lockbox_reopened")
        or receipt.get("external_actions")
    ):
        raise RuntimeError("M244 promoted evidence boundary changed")
    return receipt


def _reconstruct_m244(
    surface: pd.DataFrame,
    numeric: list[str],
    receipt: dict[str, object],
) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    recipes = {
        int(group_id): Recipe(**raw)
        for group_id, raw in receipt["policy"]["recipes"].items()
    }
    feature_sets = _feature_sets(numeric)
    parents = _parents(surface)
    query_issuances = {
        fold: np.sort(parent["data_available_kst_dtm"].unique())
        for fold, parent in parents.items()
    }
    replacements: dict[str, list[pd.DataFrame]] = {fold: [] for fold in FOLDS}
    diagnostics: dict[str, object] = {}
    for group_id, recipe in recipes.items():
        group_diagnostics: dict[str, object] = {"recipe": asdict(recipe)}
        for fold in FOLDS:
            rare_profile, rare_diagnostics = _rare_event_profile(
                surface,
                feature_sets,
                group_id,
                query_issuances[fold],
                recipe,
            )
            spread_profile, spread_diagnostics = _spread_adjusted_profile(
                surface,
                feature_sets,
                group_id,
                query_issuances[fold],
                recipe,
            )
            profile = _composed_profile(rare_profile, spread_profile)
            replacement = _apply_spread_recipe(
                parents[fold], profile, group_id, recipe
            )
            replacements[fold].append(replacement)
            group_diagnostics[fold] = {
                "rare_event": rare_diagnostics,
                "spread": spread_diagnostics,
                "score": _group_score(replacement, group_id),
            }
        diagnostics[str(group_id)] = group_diagnostics
    outputs = {
        fold: _combine(parents[fold], replacements[fold]) for fold in FOLDS
    }
    for fold in FOLDS:
        score = _score(outputs[fold])
        expected = receipt["scores"][fold]["rare_event_corrected"]
        for metric in ("total", "one_minus_nmae", "ficr"):
            if abs(score[metric] - expected[metric]) > 1e-12:
                raise RuntimeError(f"M244 {fold} {metric} reproduction changed")
    return outputs, diagnostics


def _align(
    m107: pd.DataFrame,
    m244: pd.DataFrame,
    fold: str,
) -> pd.DataFrame:
    left = m107[[*METRIC_COLUMNS, "data_available_kst_dtm"]].rename(
        columns={"prediction_kwh": "m107_prediction_kwh"}
    )
    right = m244[[*KEYS, "actual_kwh", "prediction_kwh"]].rename(
        columns={
            "actual_kwh": "m244_actual_kwh",
            "prediction_kwh": "m244_prediction_kwh",
        }
    )
    if left[KEYS].duplicated().any() or right[KEYS].duplicated().any():
        raise RuntimeError(f"M263 {fold} lineage keys are not unique")
    merged = left.merge(
        right,
        on=KEYS,
        how="left",
        validate="one_to_one",
        sort=False,
        indicator=True,
    )
    missing = merged["_merge"].eq("left_only")
    expected_boundary = (
        fold == "q4"
        and int(missing.sum()) == 3
        and set(merged.loc[missing, "group_id"].astype(int)) == set(CAPACITIES)
        and merged.loc[missing, "forecast_kst_dtm"].eq(DEV_CUTOFF).all()
        and len(right) == len(left) - 3
    )
    if missing.any() and not expected_boundary:
        raise RuntimeError(f"M263 {fold} lineage key coverage changed")
    if not missing.any() and len(merged) != len(right):
        raise RuntimeError(f"M263 {fold} right-lineage key coverage changed")
    if not np.allclose(
        merged.loc[~missing, "actual_kwh"].to_numpy(dtype=float),
        merged.loc[~missing, "m244_actual_kwh"].to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError(f"M263 {fold} lineage actual values changed")
    merged.loc[missing, "m244_prediction_kwh"] = merged.loc[
        missing, "m107_prediction_kwh"
    ].to_numpy(dtype=float)
    return merged.drop(columns=["m244_actual_kwh", "_merge"])


def _apply_mass(
    library: pd.DataFrame,
    group_id: int,
    mass: float,
) -> pd.DataFrame:
    output = library[["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh"]].copy()
    output["prediction_kwh"] = library["m107_prediction_kwh"].to_numpy(dtype=float)
    affected = output["group_id"].eq(group_id).to_numpy()
    output.loc[affected, "prediction_kwh"] = (
        (1.0 - mass)
        * library.loc[affected, "m107_prediction_kwh"].to_numpy(dtype=float)
        + mass
        * library.loc[affected, "m244_prediction_kwh"].to_numpy(dtype=float)
    )
    output["data_available_kst_dtm"] = library["data_available_kst_dtm"].to_numpy()
    return output


def _select_masses(
    libraries: dict[str, pd.DataFrame],
) -> tuple[dict[int, float], dict[str, object]]:
    selections = {group_id: 0.0 for group_id in CAPACITIES}
    diagnostics: dict[str, object] = {}
    for group_id in CAPACITIES:
        q2_parent = _group_score(_apply_mass(libraries["q2"], group_id, 0.0), group_id)
        q3_parent = _group_score(_apply_mass(libraries["q3"], group_id, 0.0), group_id)
        candidates: list[dict[str, float]] = []
        for mass in M244_MASSES:
            q2 = _group_score(_apply_mass(libraries["q2"], group_id, mass), group_id)
            q3 = _group_score(_apply_mass(libraries["q3"], group_id, mass), group_id)
            candidates.append(
                {
                    "m244_mass": mass,
                    "q2_total": q2["total"],
                    "q3_total": q3["total"],
                    "q2_delta": q2["total"] - q2_parent["total"],
                    "q3_delta": q3["total"] - q3_parent["total"],
                }
            )
        stable = [
            item
            for item in candidates
            if item["m244_mass"] > 0.0
            and item["q2_delta"] > 0.0
            and item["q3_delta"] > 0.0
        ]
        if stable:
            selected = max(
                stable,
                key=lambda item: (
                    min(item["q2_delta"], item["q3_delta"]),
                    0.5 * (item["q2_delta"] + item["q3_delta"]),
                    -item["m244_mass"],
                ),
            )
            selections[group_id] = selected["m244_mass"]
        diagnostics[str(group_id)] = {
            "q2_parent": q2_parent,
            "q3_parent": q3_parent,
            "candidate_scores": candidates,
            "stable_positive_candidates": len(stable),
            "selected_m244_mass": selections[group_id],
        }
    return selections, diagnostics


def _apply_policy(
    library: pd.DataFrame,
    selections: dict[int, float],
) -> pd.DataFrame:
    output = library[["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh"]].copy()
    output["prediction_kwh"] = library["m107_prediction_kwh"].to_numpy(dtype=float)
    for group_id, mass in selections.items():
        affected = output["group_id"].eq(group_id).to_numpy()
        output.loc[affected, "prediction_kwh"] = (
            (1.0 - mass)
            * library.loc[affected, "m107_prediction_kwh"].to_numpy(dtype=float)
            + mass
            * library.loc[affected, "m244_prediction_kwh"].to_numpy(dtype=float)
        )
    output["data_available_kst_dtm"] = library["data_available_kst_dtm"].to_numpy()
    return output


def main() -> None:
    if sha256_file(OPEN) != OPEN_SHA or sha256_file(BASELINE) != BASELINE_SHA:
        raise RuntimeError("immutable competition input hash mismatch")
    started = time.perf_counter()
    m107_receipt = _verified_m107_receipt()
    m244_receipt = _verified_m244_receipt()
    surface, numeric = development_surface()
    if surface["forecast_kst_dtm"].ge(DEV_CUTOFF).any():
        raise RuntimeError("lockbox row reached M263 runner")
    m107, boundary_fallback = _strict_parents(surface)
    m244, m244_diagnostics = _reconstruct_m244(surface, numeric, m244_receipt)
    libraries = {fold: _align(m107[fold], m244[fold], fold) for fold in FOLDS}
    selections, selection_diagnostics = _select_masses(libraries)
    outputs = {fold: _apply_policy(libraries[fold], selections) for fold in FOLDS}

    fold_scores: dict[str, object] = {}
    fold_deltas: dict[str, float] = {}
    for fold in FOLDS:
        parent_score = _score(m107[fold])
        selected_score = _score(outputs[fold])
        expected = m107_receipt["fold_scores"][FOLD_MAP[fold]]
        for metric in ("total", "one_minus_nmae", "ficr"):
            if abs(parent_score[metric] - expected[metric]) > 1e-12:
                raise RuntimeError(f"M107 {fold} {metric} reproduction changed")
        deltas = {
            metric: selected_score[metric] - parent_score[metric]
            for metric in ("total", "one_minus_nmae", "ficr")
        }
        fold_deltas[fold] = deltas["total"]
        fold_scores[fold] = {
            "parent": parent_score,
            "m244": _score(m244[fold]),
            "selected": selected_score,
            "deltas": deltas,
        }

    pooled_parent_frame = _pooled(m107)
    pooled_selected_frame = _pooled(outputs)
    pooled_parent = _score(pooled_parent_frame)
    pooled_selected = _score(pooled_selected_frame)
    for metric in ("total", "one_minus_nmae", "ficr"):
        if abs(pooled_parent[metric] - m107_receipt["pooled"][metric]) > 1e-12:
            raise RuntimeError(f"M107 pooled {metric} reproduction changed")
    pooled_deltas = {
        metric: pooled_selected[metric] - pooled_parent[metric]
        for metric in ("total", "one_minus_nmae", "ficr")
    }
    q4_bootstrap = _paired_issuance_bootstrap(m107["q4"], outputs["q4"])
    all_folds_positive = all(delta > 0.0 for delta in fold_deltas.values())
    pooled_positive = pooled_deltas["total"] > 0.0
    bootstrap_positive = q4_bootstrap["positive_fraction"] > 0.50
    promoted = all_folds_positive and pooled_positive and bootstrap_positive

    prediction_path = OUTPUT / f"{MODEL_ID}-oof.parquet"
    prediction_output = pooled_selected_frame.copy()
    prediction_output["model_id"] = MODEL_ID
    prediction_output[
        [*METRIC_COLUMNS, "fold_id", "model_id", "data_available_kst_dtm"]
    ].to_parquet(prediction_path, index=False)
    policy = {
        "architecture": "strict_convex_ensemble_of_independent_full_prediction_lineages",
        "left_parent": "M107_STRICT_TEMPORAL_TOP100",
        "right_parent": "M244_RARE_EVENT_CORRECTED_ANALOG_Q234",
        "m244_mass_grid": list(M244_MASSES),
        "selection_folds": [FOLD_MAP["q2"], FOLD_MAP["q3"]],
        "frozen_transfer_fold": FOLD_MAP["q4"],
        "selection_rule": (
            "require positive Q2 and Q3 group-total deltas; maximize worst delta, "
            "then mean delta, then prefer smaller M244 mass"
        ),
        "shift": None,
        "snap": False,
        "scale_or_offset": None,
        "dynamic_gate": False,
        "component_specific_weight": False,
        "group_or_fold_exception_search": False,
        "test_build": False,
    }
    receipt = {
        "schema_version": 1,
        "candidate_id": MODEL_ID,
        "state": (
            "LOCAL_INDEPENDENT_LINEAGE_ENSEMBLE_PROMOTED_NO_TEST_BUILD"
            if promoted
            else "LOCAL_INDEPENDENT_LINEAGE_ENSEMBLE_REJECTED_NO_TEST_BUILD"
        ),
        "policy": policy,
        "policy_sha256": canonical_sha256(policy),
        "selected_m244_masses": {
            str(group): mass for group, mass in selections.items()
        },
        "selection_diagnostics": selection_diagnostics,
        "m244_reconstruction_diagnostics": m244_diagnostics,
        "boundary_fallback": boundary_fallback,
        "fold_scores": fold_scores,
        "pooled": {
            "parent": pooled_parent,
            "selected": pooled_selected,
            "deltas": pooled_deltas,
        },
        "q4_paired_bootstrap": q4_bootstrap,
        "promotion": {
            "promoted": promoted,
            "all_q2_q3_q4_total_deltas_positive": all_folds_positive,
            "pooled_total_delta_positive": pooled_positive,
            "q4_bootstrap_positive_fraction_above_half": bootstrap_positive,
            "rule": (
                "Q2, Q3, Q4, and pooled Total deltas are positive and Q4 paired "
                "issuance bootstrap positive fraction exceeds 0.50"
            ),
        },
        "source_receipts": {
            "open_zip_sha256": OPEN_SHA,
            "baseline_ipynb_sha256": BASELINE_SHA,
            "m107_prediction_sha256": m107_receipt["prediction_sha256"],
            "m244_receipt_sha256": M244_RECEIPT_SHA,
            "m244_q3_prediction_sha256": M244_PREDICTION_SHA,
        },
        "prediction_path": str(prediction_path.relative_to(ROOT)),
        "prediction_sha256": sha256_file(prediction_path),
        "completion_target": {
            "metric": "Dacon Total",
            "strict_threshold": 0.66000,
            "status": "UNVERIFIED_REQUIRES_EXTERNAL_DACON_RESULT",
        },
        "online_score": None,
        "no_external_upload": True,
        "new_2024_evaluation": False,
        "lockbox_reopened": False,
        "external_actions": [],
        "runtime_seconds": round(time.perf_counter() - started, 2),
    }
    receipt_path = OUTPUT / f"{MODEL_ID}.json"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    print(json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
