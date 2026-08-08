"""Transfer the exact M244 analog correction onto the M107 strict parent.

M251 changes only the parent prediction surface. Every analog retrieval,
member correction, reliability multiplier, recipe, transform, and blend weight
is inherited byte-pinned from M244. No test submission is built by this runner.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from build_v2_transfer_sequence_challenger import (
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
from run_recency_spread_analog_transfer import _composed_profile
from run_spread_shrunk_analog_transfer import (
    _apply_spread_recipe,
    _spread_adjusted_profile,
)
from strict_dev_surface import DEV_CUTOFF, development_surface

from baram.contracts.hashing import canonical_sha256, sha256_file

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "artifacts" / "backtests" / "metric-aligned-probe"
MODEL_ID = "M251_STRICT_PARENT_ANALOG_TRANSFER"
M107_PREDICTION = OUTPUT / "M107_STRICT_TEMPORAL_TOP100-oof.parquet"
M107_PREDICTION_SHA = (
    "3539cada59f88a16d4b4181f5aff3c76ff8e9a94954f67f4204ccd09ac8e537d"
)
M107_RECEIPT = OUTPUT / "M107_STRICT_TEMPORAL_TOP100-oof.json"
M107_RECEIPT_SHA = (
    "0167aac129b2afd1a004a3612d32bda7d0916757fc20a38a805950a8d92b93ea"
)
M244_RECEIPT = OUTPUT / "M244_RARE_EVENT_CORRECTED_ANALOG_Q234.json"
M244_RECEIPT_SHA = (
    "8489f2759ce39fe4abb7c5179386cde0e1a39d22a5d07673a782b4163410a371"
)
FOLD_MAP = {
    "q2": "dev-2023-Q2",
    "q3": "dev-2023-Q3",
    "q4": "dev-2023-Q4",
}
KEYS = ["forecast_id", "forecast_kst_dtm", "group_id"]
TARGET_TOTAL = 0.65971
TARGET_ONE_MINUS_NMAE = 0.87991
TARGET_FICR = 0.43952


def _verified_receipts() -> tuple[dict[str, object], dict[str, object]]:
    if sha256_file(M107_PREDICTION) != M107_PREDICTION_SHA:
        raise RuntimeError("M107 strict prediction hash mismatch")
    if sha256_file(M107_RECEIPT) != M107_RECEIPT_SHA:
        raise RuntimeError("M107 strict receipt hash mismatch")
    if sha256_file(M244_RECEIPT) != M244_RECEIPT_SHA:
        raise RuntimeError("M244 promoted receipt hash mismatch")
    m107 = json.loads(M107_RECEIPT.read_text(encoding="utf-8"))
    m244 = json.loads(M244_RECEIPT.read_text(encoding="utf-8"))
    if (
        m107["candidate_id"] != "M107_STRICT_TEMPORAL_TOP100"
        or m107["prediction_sha256"] != M107_PREDICTION_SHA
        or m107.get("new_2024_evaluation")
        or m107.get("lockbox_reopened")
        or m107.get("external_actions")
    ):
        raise RuntimeError("M107 strict evidence boundary changed")
    if (
        m244["candidate_id"] != "M244_RARE_EVENT_CORRECTED_ANALOG_Q234"
        or m244["state"]
        != "LOCAL_RARE_EVENT_ANALOG_MOS_PROMOTED_FOR_TEST_BUILD"
        or not m244["promotion"]["promoted"]
        or m244.get("new_2024_evaluation")
        or m244.get("lockbox_reopened")
        or m244.get("external_actions")
    ):
        raise RuntimeError("M244 promoted evidence boundary changed")
    return m107, m244


def _strict_parents(
    surface: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    metadata = surface[
        [*KEYS, "actual_kwh", "data_available_kst_dtm"]
    ].copy()
    if metadata[KEYS].duplicated().any():
        raise RuntimeError("development metadata contains duplicate keys")
    predictions = pd.read_parquet(M107_PREDICTION)
    if predictions[[*KEYS, "fold_id"]].duplicated().any():
        raise RuntimeError("M107 contains duplicate fold keys")
    parents: dict[str, pd.DataFrame] = {}
    boundary_fallback: dict[str, object] = {
        "policy": (
            "retain exact M107 for the final 2023-Q4 issuance when its last "
            "forecast timestamp equals the development cutoff"
        ),
        "rows": 0,
        "issuances": [],
    }
    for short_fold, fold_id in FOLD_MAP.items():
        prediction = predictions.loc[predictions["fold_id"].eq(fold_id)].copy()
        prediction = prediction.merge(
            metadata.rename(columns={"actual_kwh": "surface_actual_kwh"}),
            on=KEYS,
            how="left",
            validate="one_to_one",
        )
        missing = prediction["data_available_kst_dtm"].isna()
        if missing.any():
            boundary = prediction.loc[missing]
            expected_boundary = (
                short_fold == "q4"
                and len(boundary) == 3
                and set(boundary["group_id"].astype(int)) == {1, 2, 3}
                and boundary["forecast_kst_dtm"].eq(DEV_CUTOFF).all()
            )
            if not expected_boundary:
                raise RuntimeError(f"M107 {short_fold} metadata coverage changed")
            prediction = prediction.sort_values(
                ["group_id", "forecast_kst_dtm"]
            ).reset_index(drop=True)
            prediction["data_available_kst_dtm"] = prediction.groupby(
                "group_id", sort=False
            )["data_available_kst_dtm"].ffill()
            if prediction["data_available_kst_dtm"].isna().any():
                raise RuntimeError("M107 boundary issuance could not be resolved")
            boundary_issuances = sorted(
                str(value)
                for value in prediction.loc[
                    prediction["forecast_kst_dtm"].eq(DEV_CUTOFF),
                    "data_available_kst_dtm",
                ].unique()
            )
            if len(boundary_issuances) != 1:
                raise RuntimeError("M107 boundary rows crossed issuance batches")
            boundary_fallback = {
                **boundary_fallback,
                "rows": 3,
                "forecast_kst_dtm": str(DEV_CUTOFF),
                "issuances": boundary_issuances,
            }
            missing = prediction["forecast_kst_dtm"].eq(DEV_CUTOFF)
        if not np.allclose(
            prediction.loc[~missing, "actual_kwh"].to_numpy(dtype=float),
            prediction.loc[~missing, "surface_actual_kwh"].to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-12,
        ):
            raise RuntimeError(f"M107 {short_fold} actual values changed")
        parent = prediction[
            [*METRIC_COLUMNS, "data_available_kst_dtm"]
        ].copy()
        if parent[METRIC_COLUMNS[:3]].duplicated().any():
            raise RuntimeError(f"M107 {short_fold} parent keys are not unique")
        parents[short_fold] = parent
    return parents, boundary_fallback


def _pooled(outputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for fold, output in outputs.items():
        part = output.copy()
        part["fold_id"] = FOLD_MAP[fold]
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    if sha256_file(OPEN) != OPEN_SHA or sha256_file(BASELINE) != BASELINE_SHA:
        raise RuntimeError("immutable competition input hash mismatch")
    m107_receipt, m244_receipt = _verified_receipts()
    recipes = {
        int(group_id): Recipe(**raw)
        for group_id, raw in m244_receipt["policy"]["recipes"].items()
    }

    surface, numeric = development_surface()
    if surface["forecast_kst_dtm"].ge(DEV_CUTOFF).any():
        raise RuntimeError("lockbox row reached M251 parent-transfer runner")
    feature_sets = _feature_sets(numeric)
    parents, boundary_fallback = _strict_parents(surface)
    query_issuances = {
        fold: np.sort(parent["data_available_kst_dtm"].unique())
        for fold, parent in parents.items()
    }

    replacements: dict[str, list[pd.DataFrame]] = {
        fold: [] for fold in FOLD_MAP
    }
    diagnostics: dict[str, object] = {}
    for group_id, recipe in recipes.items():
        group_diagnostics: dict[str, object] = {"recipe": asdict(recipe)}
        for fold in FOLD_MAP:
            corrected_profile, corrected_retrieval = _rare_event_profile(
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
            profile = _composed_profile(corrected_profile, spread_profile)
            corrected = _apply_spread_recipe(
                parents[fold],
                profile,
                group_id,
                recipe,
            )
            replacements[fold].append(corrected)
            group_diagnostics[fold] = {
                "parent": _group_score(parents[fold], group_id),
                "transferred": _group_score(corrected, group_id),
                "corrected_retrieval": corrected_retrieval,
                "spread_retrieval": spread_retrieval,
            }
        diagnostics[str(group_id)] = group_diagnostics

    transferred = {
        fold: _combine(parents[fold], replacements[fold]) for fold in FOLD_MAP
    }
    fold_scores: dict[str, object] = {}
    fold_deltas: dict[str, float] = {}
    for fold in FOLD_MAP:
        parent_score = _score(parents[fold])
        transferred_score = _score(transferred[fold])
        fold_deltas[fold] = transferred_score["total"] - parent_score["total"]
        expected_parent = m107_receipt["fold_scores"][FOLD_MAP[fold]]
        for metric in ("total", "one_minus_nmae", "ficr"):
            if abs(parent_score[metric] - expected_parent[metric]) > 1e-12:
                raise RuntimeError(
                    f"M107 {fold} {metric} reproduction changed"
                )
        fold_scores[fold] = {
            "parent": parent_score,
            "transferred": transferred_score,
            "total_delta": fold_deltas[fold],
            "one_minus_nmae_delta": (
                transferred_score["one_minus_nmae"]
                - parent_score["one_minus_nmae"]
            ),
            "ficr_delta": transferred_score["ficr"] - parent_score["ficr"],
        }

    pooled_parent_frame = _pooled(parents)
    pooled_transferred_frame = _pooled(transferred)
    pooled_parent = _score(pooled_parent_frame)
    pooled_transferred = _score(pooled_transferred_frame)
    for metric in ("total", "one_minus_nmae", "ficr"):
        if abs(pooled_parent[metric] - m107_receipt["pooled"][metric]) > 1e-12:
            raise RuntimeError(f"M107 pooled {metric} reproduction changed")
    pooled_deltas = {
        metric: pooled_transferred[metric] - pooled_parent[metric]
        for metric in ("total", "one_minus_nmae", "ficr")
    }
    q4_bootstrap = _paired_issuance_bootstrap(
        parents["q4"], transferred["q4"]
    )

    all_fold_totals_positive = all(delta > 0.0 for delta in fold_deltas.values())
    pooled_components_positive = all(delta > 0.0 for delta in pooled_deltas.values())
    q4_bootstrap_positive = q4_bootstrap["positive_fraction"] > 0.50
    promoted = (
        all_fold_totals_positive
        and pooled_components_positive
        and q4_bootstrap_positive
    )
    promotion = {
        "promoted": promoted,
        "all_q2_q3_q4_total_deltas_positive": all_fold_totals_positive,
        "pooled_total_one_minus_nmae_ficr_all_improve": (
            pooled_components_positive
        ),
        "q4_bootstrap_positive_fraction_above_half": q4_bootstrap_positive,
        "rule": (
            "all Q2/Q3/Q4 Total deltas over M107 positive; pooled Total, "
            "1-NMAE, and FICR all strictly improve; and Q4 paired issuance "
            "bootstrap positive fraction exceeds 0.50"
        ),
    }
    goal_check = {
        "total_strictly_above_0p65971": pooled_transferred["total"]
        > TARGET_TOTAL,
        "one_minus_nmae_strictly_above_0p87991": (
            pooled_transferred["one_minus_nmae"] > TARGET_ONE_MINUS_NMAE
        ),
        "ficr_strictly_above_0p43952": pooled_transferred["ficr"] > TARGET_FICR,
    }
    goal_check["all_simultaneous"] = all(goal_check.values())

    prediction_path = OUTPUT / f"{MODEL_ID}-oof.parquet"
    prediction_output = _pooled(transferred)
    prediction_output["model_id"] = MODEL_ID
    prediction_output[
        [*METRIC_COLUMNS, "fold_id", "model_id", "data_available_kst_dtm"]
    ].to_parquet(prediction_path, index=False)
    policy = {
        "architecture": "exact_M244_correction_on_independent_M107_parent",
        "parent": "M107_STRICT_TEMPORAL_TOP100",
        "analog_policy": "exact_M244_RARE_EVENT_CORRECTED_ANALOG_Q234",
        "recipes": {
            str(group_id): asdict(recipe) for group_id, recipe in recipes.items()
        },
        "parent_blend_search": False,
        "analog_parameter_search": False,
        "shift_search": False,
        "group_exception_search": False,
        "quarter_gate_search": False,
        "test_build": False,
    }
    receipt = {
        "schema_version": 1,
        "candidate_id": MODEL_ID,
        "state": (
            "LOCAL_STRICT_PARENT_ANALOG_TRANSFER_PROMOTED_NO_TEST_BUILD"
            if promoted
            else "LOCAL_STRICT_PARENT_ANALOG_TRANSFER_REJECTED_NO_TEST_BUILD"
        ),
        "policy": policy,
        "policy_sha256": canonical_sha256(policy),
        "fold_scores": fold_scores,
        "pooled": {
            "parent": pooled_parent,
            "transferred": pooled_transferred,
            "deltas": pooled_deltas,
        },
        "q4_paired_bootstrap": q4_bootstrap,
        "promotion": promotion,
        "goal_check": goal_check,
        "diagnostics": diagnostics,
        "boundary_fallback": boundary_fallback,
        "source_receipts": {
            "m107_prediction_sha256": M107_PREDICTION_SHA,
            "m107_receipt_sha256": M107_RECEIPT_SHA,
            "m244_receipt_sha256": M244_RECEIPT_SHA,
        },
        "prediction_path": str(prediction_path.relative_to(ROOT)),
        "prediction_sha256": sha256_file(prediction_path),
        "online_score": None,
        "no_external_upload": True,
        "new_2024_evaluation": False,
        "lockbox_reopened": False,
        "external_actions": [],
    }
    receipt_path = OUTPUT / f"{MODEL_ID}.json"
    receipt_path.write_text(
        json.dumps(receipt, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
