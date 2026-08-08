"""Evaluate one fixed D1 predictive-width shrink over exact M244 corrections.

M259 leaves Q2 exactly at M244. For Q3 and Q4 it shrinks only the
M244-minus-parent correction when the aligned D1 q90-q10 width exceeds a
chronology-safe group/hour 75th-percentile reference. No test submission is
built by this runner.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from build_v2_transfer_sequence_challenger import (
    METRIC_COLUMNS,
    OOF,
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
from run_stable_daily_analog_profile import DISTRIBUTION
from strict_dev_surface import DEV_CUTOFF, development_surface

from baram.constants import CAPACITIES_KWH
from baram.contracts.hashing import canonical_sha256, sha256_file

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "artifacts" / "backtests" / "metric-aligned-probe"
MODEL_ID = "M259_D1_WIDTH_SHRUNK_M244"
M244_RECEIPT = OUTPUT / "M244_RARE_EVENT_CORRECTED_ANALOG_Q234.json"
M244_RECEIPT_SHA = (
    "8489f2759ce39fe4abb7c5179386cde0e1a39d22a5d07673a782b4163410a371"
)
D1_PREDICTION = (
    ROOT
    / "artifacts"
    / "backtests"
    / "distribution-v2"
    / "baram-v2-20260801-01"
    / "D1_LGBM_SHARED_BASE-oof.parquet"
)
D1_PREDICTION_SHA = (
    "8d419b185b34b7d7f6c5f02482d744443cdae26b2cce03d8f46d81cef7dc3f83"
)
D1_RECEIPT = D1_PREDICTION.parent / "D1_LGBM_SHARED_BASE.json"
D1_RECEIPT_SHA = (
    "9fdcb5fbe4f32eeb6c111a1a22ad89bf6f55a9b2181da7cac1ab0d30c87511f8"
)
M252_ONLINE_RECEIPT = ROOT / "reports" / "dacon_m252_online_2026-08-03_receipt.json"
M252_ONLINE_RECEIPT_SHA = (
    "25694ade390a9386f7d1188bde59314da74c6339a53d8ed60f9b34fc2e4e57f6"
)
FOLD_MAP = {
    "q2": "dev-2023-Q2",
    "q3": "dev-2023-Q3",
    "q4": "dev-2023-Q4",
}
KEYS = ["forecast_id", "forecast_kst_dtm", "group_id"]
LOW_QUANTILE = 0.10
HIGH_QUANTILE = 0.90
REFERENCE_QUANTILE = 0.75
MIN_REFERENCE_SUPPORT = 80
TINY = np.finfo(float).tiny


def _self_test() -> None:
    width = np.array([0.0, 0.20, 0.40, 0.80], dtype=float)
    reference = np.array([0.30, 0.30, 0.30, 0.30], dtype=float)
    actual = _multipliers(width, reference)
    expected = np.array([1.0, 1.0, 0.75, 0.375], dtype=float)
    if not np.allclose(actual, expected, rtol=0.0, atol=1e-12):
        raise RuntimeError("M259 multiplier self-test failed")
    print("M259_SELF_TEST_PASS", flush=True)


def _multipliers(width: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if not np.isfinite(width).all() or not np.isfinite(reference).all():
        raise RuntimeError("M259 width inputs contain non-finite values")
    if (width < 0.0).any() or (reference < 0.0).any():
        raise RuntimeError("M259 width inputs contain negative values")
    return np.minimum(1.0, reference / np.maximum(width, TINY))


def _verified_inputs() -> dict[str, object]:
    expected = {
        "m244_receipt": (M244_RECEIPT, M244_RECEIPT_SHA),
        "d1_prediction": (D1_PREDICTION, D1_PREDICTION_SHA),
        "d1_receipt": (D1_RECEIPT, D1_RECEIPT_SHA),
        "m252_online_receipt": (M252_ONLINE_RECEIPT, M252_ONLINE_RECEIPT_SHA),
    }
    changed = {
        name: sha256_file(path)
        for name, (path, frozen_hash) in expected.items()
        if sha256_file(path) != frozen_hash
    }
    if changed:
        raise RuntimeError(f"M259 frozen input changed: {changed}")
    receipt = json.loads(M244_RECEIPT.read_text(encoding="utf-8"))
    if (
        receipt["candidate_id"] != "M244_RARE_EVENT_CORRECTED_ANALOG_Q234"
        or not receipt["promotion"]["promoted"]
        or receipt.get("new_2024_evaluation")
        or receipt.get("lockbox_reopened")
        or receipt.get("external_actions")
    ):
        raise RuntimeError("M244 promoted evidence boundary changed")
    return receipt


def _d1_widths() -> dict[str, pd.DataFrame]:
    frame = pd.read_parquet(
        D1_PREDICTION,
        columns=[
            *KEYS,
            "actual_kwh",
            "fold_id",
            "quantile",
            "prediction_kwh",
        ],
    )
    if set(frame["quantile"].unique()) != {
        0.05,
        0.10,
        0.25,
        0.50,
        0.75,
        0.90,
        0.95,
    }:
        raise RuntimeError("D1 quantile schema changed")
    selected = frame.loc[
        frame["quantile"].isin([LOW_QUANTILE, HIGH_QUANTILE])
    ].copy()
    if selected[[*KEYS, "fold_id", "quantile"]].duplicated().any():
        raise RuntimeError("D1 contains duplicate quantile keys")
    wide = selected.pivot(
        index=[*KEYS, "actual_kwh", "fold_id"],
        columns="quantile",
        values="prediction_kwh",
    ).reset_index()
    if LOW_QUANTILE not in wide or HIGH_QUANTILE not in wide:
        raise RuntimeError("D1 width quantiles are incomplete")
    low = wide[LOW_QUANTILE].to_numpy(dtype=float)
    high = wide[HIGH_QUANTILE].to_numpy(dtype=float)
    if not np.isfinite(low).all() or not np.isfinite(high).all():
        raise RuntimeError("D1 width quantiles contain non-finite values")
    if np.any(high + 1e-12 < low):
        raise RuntimeError("D1 q90 is below q10")
    capacity = wide["group_id"].map(CAPACITIES_KWH).to_numpy(dtype=float)
    if not np.isfinite(capacity).all():
        raise RuntimeError("D1 width capacity mapping changed")
    wide["width_fraction"] = np.maximum(0.0, high - low) / capacity
    wide["target_hour"] = pd.to_datetime(wide["forecast_kst_dtm"]).dt.hour
    result: dict[str, pd.DataFrame] = {}
    for short_fold, fold_id in FOLD_MAP.items():
        part = wide.loc[wide["fold_id"].eq(fold_id)].copy()
        if part[KEYS].duplicated().any() or part.empty:
            raise RuntimeError(f"D1 {short_fold} width keys changed")
        result[short_fold] = part[
            [*KEYS, "actual_kwh", "target_hour", "width_fraction"]
        ].reset_index(drop=True)
    return result


def _reference_table(history: pd.DataFrame) -> pd.DataFrame:
    grouped = history.groupby(["group_id", "target_hour"], sort=True)[
        "width_fraction"
    ]
    reference = grouped.quantile(REFERENCE_QUANTILE).rename(
        "reference_width_fraction"
    )
    support = grouped.size().rename("reference_support")
    table = pd.concat([reference, support], axis=1).reset_index()
    if len(table) != 3 * 24:
        raise RuntimeError("M259 reference group/hour coverage changed")
    if int(table["reference_support"].min()) < MIN_REFERENCE_SUPPORT:
        raise RuntimeError("M259 reference support changed")
    if not np.isfinite(table["reference_width_fraction"]).all():
        raise RuntimeError("M259 reference width contains non-finite values")
    return table


def _apply_width_shrink(
    parent: pd.DataFrame,
    m244: pd.DataFrame,
    widths: pd.DataFrame,
    reference: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    ordered = m244.copy().reset_index(drop=True)
    ordered["_row_order"] = np.arange(len(ordered), dtype=np.int64)
    parent_prediction = parent[[*KEYS, "prediction_kwh"]].rename(
        columns={"prediction_kwh": "parent_prediction_kwh"}
    )
    merged = ordered.merge(
        parent_prediction,
        on=KEYS,
        how="left",
        validate="one_to_one",
    ).merge(
        widths[[*KEYS, "target_hour", "width_fraction"]],
        on=KEYS,
        how="left",
        validate="one_to_one",
    ).merge(
        reference,
        on=["group_id", "target_hour"],
        how="left",
        validate="many_to_one",
    )
    merged = merged.sort_values("_row_order", kind="stable").reset_index(drop=True)
    required = [
        "parent_prediction_kwh",
        "width_fraction",
        "reference_width_fraction",
        "reference_support",
    ]
    if merged[required].isna().any().any():
        raise RuntimeError("M259 parent/width/reference alignment changed")
    if not merged[KEYS].equals(ordered[KEYS]):
        raise RuntimeError("M259 merge changed M244 key order")
    width = merged["width_fraction"].to_numpy(dtype=float)
    ref = merged["reference_width_fraction"].to_numpy(dtype=float)
    multiplier = _multipliers(width, ref)
    parent_values = merged["parent_prediction_kwh"].to_numpy(dtype=float)
    m244_values = merged["prediction_kwh"].to_numpy(dtype=float)
    candidate_values = parent_values + multiplier * (m244_values - parent_values)
    candidate = ordered.drop(columns="_row_order").copy()
    candidate["prediction_kwh"] = candidate_values
    diagnostics: dict[str, object] = {
        "rows": len(candidate),
        "changed_rows_vs_m244": int(
            np.count_nonzero(np.abs(candidate_values - m244_values) > 1e-12)
        ),
        "multiplier": {
            "min": float(np.min(multiplier)),
            "mean": float(np.mean(multiplier)),
            "median": float(np.median(multiplier)),
            "fraction_below_one": float(np.mean(multiplier < 1.0 - 1e-15)),
        },
        "width_fraction": {
            "min": float(np.min(width)),
            "q25": float(np.quantile(width, 0.25)),
            "median": float(np.median(width)),
            "q75": float(np.quantile(width, 0.75)),
            "max": float(np.max(width)),
        },
        "reference_width_fraction": {
            "min": float(np.min(ref)),
            "median": float(np.median(ref)),
            "max": float(np.max(ref)),
        },
    }
    return candidate, diagnostics


def _pooled(outputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for short_fold, fold_id in FOLD_MAP.items():
        part = outputs[short_fold].copy()
        part["fold_id"] = fold_id
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    if sha256_file(OPEN) != OPEN_SHA or sha256_file(BASELINE) != BASELINE_SHA:
        raise RuntimeError("immutable competition input hash mismatch")
    m244_receipt = _verified_inputs()
    recipes = {
        int(group_id): Recipe(**raw)
        for group_id, raw in m244_receipt["policy"]["recipes"].items()
    }
    surface, numeric = development_surface()
    if surface["forecast_kst_dtm"].ge(DEV_CUTOFF).any():
        raise RuntimeError("lockbox row reached M259 runner")
    feature_sets = _feature_sets(numeric)
    parents = _parents(surface)
    query_issuances = {
        fold: np.sort(parent["data_available_kst_dtm"].unique())
        for fold, parent in parents.items()
    }

    replacements: dict[str, list[pd.DataFrame]] = {
        fold: [] for fold in FOLD_MAP
    }
    retrieval_diagnostics: dict[str, object] = {}
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
                "corrected_retrieval": corrected_retrieval,
                "spread_retrieval": spread_retrieval,
            }
        retrieval_diagnostics[str(group_id)] = group_diagnostics

    m244_outputs = {
        fold: _combine(parents[fold], replacements[fold]) for fold in FOLD_MAP
    }
    widths = _d1_widths()
    references = {
        "q3": _reference_table(widths["q2"]),
        "q4": _reference_table(pd.concat([widths["q2"], widths["q3"]])),
    }
    candidate_outputs = {"q2": m244_outputs["q2"].copy()}
    uncertainty_diagnostics: dict[str, object] = {
        "q2": {
            "policy": "exact_M244_control",
            "rows": len(m244_outputs["q2"]),
            "changed_rows_vs_m244": 0,
        }
    }
    for fold in ("q3", "q4"):
        candidate, diagnostics = _apply_width_shrink(
            parents[fold],
            m244_outputs[fold],
            widths[fold],
            references[fold],
        )
        candidate_outputs[fold] = candidate
        uncertainty_diagnostics[fold] = diagnostics

    fold_scores: dict[str, object] = {}
    fold_deltas: dict[str, float] = {}
    for fold in FOLD_MAP:
        m244_score = _score(m244_outputs[fold])
        expected = m244_receipt["scores"][fold]["rare_event_corrected"]
        for metric in ("total", "one_minus_nmae", "ficr"):
            if abs(m244_score[metric] - expected[metric]) > 1e-12:
                raise RuntimeError(f"M244 {fold} {metric} reproduction changed")
        candidate_score = _score(candidate_outputs[fold])
        deltas = {
            metric: candidate_score[metric] - m244_score[metric]
            for metric in ("total", "one_minus_nmae", "ficr")
        }
        fold_deltas[fold] = deltas["total"]
        fold_scores[fold] = {
            "m244": m244_score,
            "width_shrunk": candidate_score,
            "deltas": deltas,
        }

    pooled_m244 = _score(_pooled(m244_outputs))
    pooled_candidate = _score(_pooled(candidate_outputs))
    pooled_deltas = {
        metric: pooled_candidate[metric] - pooled_m244[metric]
        for metric in ("total", "one_minus_nmae", "ficr")
    }
    q4_bootstrap = _paired_issuance_bootstrap(
        m244_outputs["q4"], candidate_outputs["q4"]
    )
    promoted = (
        fold_deltas["q3"] > 0.0
        and fold_deltas["q4"] > 0.0
        and pooled_deltas["total"] > 0.0
        and q4_bootstrap["positive_fraction"] > 0.50
    )
    promotion = {
        "promoted": promoted,
        "q3_total_delta_positive": fold_deltas["q3"] > 0.0,
        "q4_total_delta_positive": fold_deltas["q4"] > 0.0,
        "pooled_total_delta_positive": pooled_deltas["total"] > 0.0,
        "q4_bootstrap_positive_fraction_above_half": (
            q4_bootstrap["positive_fraction"] > 0.50
        ),
        "rule": (
            "Q2 remains exact M244; Q3 and Q4 Total deltas versus M244 are "
            "strictly positive; pooled Q2-Q4 Total improves; and Q4 paired "
            "issuance bootstrap positive fraction exceeds 0.50"
        ),
    }

    prediction_path = OUTPUT / f"{MODEL_ID}-oof.parquet"
    prediction_output = _pooled(candidate_outputs)
    prediction_output["model_id"] = MODEL_ID
    prediction_output[
        [*METRIC_COLUMNS, "fold_id", "model_id", "data_available_kst_dtm"]
    ].to_parquet(prediction_path, index=False)
    policy = {
        "architecture": "exact_M244_correction_shrunk_by_D1_predictive_width",
        "parent": "exact_M244_fold_parent",
        "comparator": "M244_RARE_EVENT_CORRECTED_ANALOG_Q234",
        "distribution": "D1_LGBM_SHARED_BASE",
        "width_fraction": "(q90_minus_q10)_divided_by_group_capacity",
        "low_quantile": LOW_QUANTILE,
        "high_quantile": HIGH_QUANTILE,
        "reference_quantile": REFERENCE_QUANTILE,
        "q3_reference_history": ["dev-2023-Q2"],
        "q4_reference_history": ["dev-2023-Q2", "dev-2023-Q3"],
        "reference_grain": ["group_id", "target_hour"],
        "multiplier": "min(1, reference_width/max(query_width, tiny))",
        "q2_policy": "exact_M244_control",
        "recipes": {
            str(group_id): asdict(recipe) for group_id, recipe in recipes.items()
        },
        "quantile_pair_search": False,
        "reference_quantile_search": False,
        "multiplier_search": False,
        "group_hour_exception_search": False,
        "parent_or_analog_search": False,
        "test_build": False,
    }
    receipt = {
        "schema_version": 1,
        "candidate_id": MODEL_ID,
        "state": (
            "LOCAL_D1_WIDTH_SHRUNK_ANALOG_PROMOTED_NO_TEST_BUILD"
            if promoted
            else "LOCAL_D1_WIDTH_SHRUNK_ANALOG_REJECTED_NO_TEST_BUILD"
        ),
        "policy": policy,
        "policy_sha256": canonical_sha256(policy),
        "fold_scores": fold_scores,
        "pooled": {
            "m244": pooled_m244,
            "width_shrunk": pooled_candidate,
            "deltas": pooled_deltas,
        },
        "q4_paired_bootstrap": q4_bootstrap,
        "promotion": promotion,
        "uncertainty_diagnostics": uncertainty_diagnostics,
        "retrieval_diagnostics": retrieval_diagnostics,
        "prediction_path": str(prediction_path.relative_to(ROOT)),
        "prediction_sha256": sha256_file(prediction_path),
        "inputs": {
            "open_zip_sha256": OPEN_SHA,
            "baseline_ipynb_sha256": BASELINE_SHA,
            "d1_prediction_sha256": D1_PREDICTION_SHA,
            "d1_receipt_sha256": D1_RECEIPT_SHA,
            "distribution_parent_sha256": sha256_file(DISTRIBUTION),
            "v2_parent_oof_sha256": sha256_file(OOF),
            "m244_receipt_sha256": M244_RECEIPT_SHA,
            "m252_online_receipt_sha256": M252_ONLINE_RECEIPT_SHA,
            "runner_sha256": sha256_file(Path(__file__)),
        },
        "completion_target": {
            "dacon_total_strictly_greater_than": 0.66,
            "m252_online_total": 0.6268784092,
        },
        "online_score": None,
        "no_external_upload": True,
        "new_2024_evaluation": False,
        "lockbox_reopened": False,
        "external_actions": [],
    }
    receipt_path = OUTPUT / f"{MODEL_ID}.json"
    receipt_path.write_text(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    print(json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        _self_test()
    else:
        main()
