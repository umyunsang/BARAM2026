"""Augment exact M263 with the Q2-selected M114 DART group-3 lineage."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from build_v2_transfer_sequence_challenger import (
    METRIC_COLUMNS,
    _group_score,
    _paired_issuance_bootstrap,
    _score,
)
from run_sequence_classifier import BASELINE, BASELINE_SHA, OPEN, OPEN_SHA

from baram.contracts.hashing import canonical_sha256, sha256_file

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "artifacts" / "backtests" / "metric-aligned-probe"
MODEL_ID = "M265_DART_GROUP3_AUGMENTATION"
M263_PREDICTION = OUTPUT / "M263_INDEPENDENT_LINEAGE_CONVEX_ENSEMBLE-oof.parquet"
M263_RECEIPT = OUTPUT / "M263_INDEPENDENT_LINEAGE_CONVEX_ENSEMBLE.json"
M114_PREDICTION = OUTPUT / "M114_STRICT_DART_BLEND-oof.parquet"
M114_RECEIPT = OUTPUT / "M114_STRICT_DART_BLEND-oof.json"
FROZEN_HASHES = {
    "m263_prediction": "5b09c1b55621766e09bbd39a25d43f6eaf36d14621577604d2474d3bea2348a8",
    "m263_receipt": "9169edab2b983dc5c47907295b8524ffba9c01615eda87d8eb288a742848e47e",
    "m114_prediction": "d37bb7253a185913aeb01656203601749824ff63d771463fd68108dc6232f7f7",
    "m114_receipt": "ac12b47fde57bc6f4b553c17f05e7f65fb91524c2665d9378df963513d729d93",
}
FOLDS = ("dev-2023-Q2", "dev-2023-Q3", "dev-2023-Q4")
KEYS = ["forecast_id", "forecast_kst_dtm", "group_id"]


def _read_receipts() -> tuple[dict[str, object], dict[str, object]]:
    paths = {
        "m263_prediction": M263_PREDICTION,
        "m263_receipt": M263_RECEIPT,
        "m114_prediction": M114_PREDICTION,
        "m114_receipt": M114_RECEIPT,
    }
    for name, path in paths.items():
        observed = sha256_file(path)
        if observed != FROZEN_HASHES[name]:
            raise RuntimeError(f"M265 frozen {name} hash changed: {observed}")
    m263 = json.loads(M263_RECEIPT.read_text(encoding="utf-8"))
    m114 = json.loads(M114_RECEIPT.read_text(encoding="utf-8"))
    if (
        m263.get("candidate_id") != "M263_INDEPENDENT_LINEAGE_CONVEX_ENSEMBLE"
        or m263.get("state")
        != "LOCAL_INDEPENDENT_LINEAGE_ENSEMBLE_PROMOTED_NO_TEST_BUILD"
        or not m263.get("promotion", {}).get("promoted")
        or m263.get("prediction_sha256") != FROZEN_HASHES["m263_prediction"]
        or m263.get("new_2024_evaluation")
        or m263.get("lockbox_reopened")
        or m263.get("external_actions")
    ):
        raise RuntimeError("M265 M263 evidence boundary changed")
    selection = m114.get("selections", {}).get("3", {})
    if (
        m114.get("candidate_id") != "M114_STRICT_DART_BLEND"
        or m114.get("scope")
        != (
            "Q2 policy and parent-weight selection with unchanged Q2 control; "
            "selected DART blend frozen unchanged for Q3-Q4"
        )
        or m114.get("selection_fold") != "dev-2023-Q2"
        or m114.get("selected_iteration") != 140
        or selection.get("policy") != "T0.6_G0.5"
        or abs(float(selection.get("parent_weight", -1.0)) - 0.6) > 1e-12
        or m114.get("prediction_sha256") != FROZEN_HASHES["m114_prediction"]
        or m114.get("new_2024_evaluation")
        or m114.get("lockbox_reopened")
        or m114.get("external_actions")
    ):
        raise RuntimeError("M265 M114 chronology/evidence boundary changed")
    return m263, m114


def _aligned_library() -> pd.DataFrame:
    m263 = pd.read_parquet(M263_PREDICTION)
    m114 = pd.read_parquet(M114_PREDICTION)
    required = {*METRIC_COLUMNS, "fold_id", "model_id"}
    if not required.issubset(m263.columns) or not required.issubset(m114.columns):
        raise RuntimeError("M265 parent schema changed")
    if m263[KEYS].duplicated().any() or m114[KEYS].duplicated().any():
        raise RuntimeError("M265 parent key uniqueness changed")
    right = m114[[*KEYS, "actual_kwh", "prediction_kwh", "fold_id"]].rename(
        columns={
            "actual_kwh": "m114_actual_kwh",
            "prediction_kwh": "m114_prediction_kwh",
            "fold_id": "m114_fold_id",
        }
    )
    library = m263.rename(columns={"prediction_kwh": "m263_prediction_kwh"}).merge(
        right,
        on=KEYS,
        how="inner",
        validate="one_to_one",
        sort=False,
    )
    if len(library) != len(m263) or len(library) != len(m114):
        raise RuntimeError("M265 parent key coverage changed")
    if set(library["fold_id"]) != set(FOLDS) or not library["fold_id"].eq(
        library["m114_fold_id"]
    ).all():
        raise RuntimeError("M265 parent fold alignment changed")
    if not np.allclose(
        library["actual_kwh"].to_numpy(dtype=float),
        library["m114_actual_kwh"].to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError("M265 parent actual alignment changed")
    return library


def _metric_frame(library: pd.DataFrame, use_m114_group3: bool) -> pd.DataFrame:
    output = library[[*KEYS, "actual_kwh", "fold_id", "data_available_kst_dtm"]].copy()
    output["prediction_kwh"] = library["m263_prediction_kwh"].to_numpy(dtype=float)
    if use_m114_group3:
        affected = library["group_id"].eq(3) & library["fold_id"].isin(FOLDS[1:])
        output.loc[affected, "prediction_kwh"] = library.loc[
            affected, "m114_prediction_kwh"
        ].to_numpy(dtype=float)
    return output


def main() -> None:
    if sha256_file(OPEN) != OPEN_SHA or sha256_file(BASELINE) != BASELINE_SHA:
        raise RuntimeError("immutable competition input hash mismatch")
    m263_receipt, m114_receipt = _read_receipts()
    library = _aligned_library()
    parent = _metric_frame(library, use_m114_group3=False)
    selected = _metric_frame(library, use_m114_group3=True)

    fold_scores: dict[str, object] = {}
    fold_deltas: dict[str, float] = {}
    for fold in FOLDS:
        fold_mask = library["fold_id"].eq(fold)
        parent_fold = parent.loc[fold_mask]
        selected_fold = selected.loc[fold_mask]
        m114_fold = library.loc[fold_mask, [*KEYS, "actual_kwh"]].copy()
        m114_fold["prediction_kwh"] = library.loc[
            fold_mask, "m114_prediction_kwh"
        ].to_numpy(dtype=float)
        parent_score = _score(parent_fold)
        selected_score = _score(selected_fold)
        m114_score = _score(m114_fold)
        key = fold.lower().rsplit("-", maxsplit=1)[-1]
        expected_parent = m263_receipt["fold_scores"][key]["selected"]
        expected_m114 = m114_receipt["fold_scores"][fold]
        for metric in ("total", "one_minus_nmae", "ficr"):
            if abs(parent_score[metric] - expected_parent[metric]) > 1e-12:
                raise RuntimeError(f"M265 M263 {fold} {metric} reproduction changed")
            if abs(m114_score[metric] - expected_m114[metric]) > 1e-12:
                raise RuntimeError(f"M265 M114 {fold} {metric} reproduction changed")
        deltas = {
            metric: selected_score[metric] - parent_score[metric]
            for metric in ("total", "one_minus_nmae", "ficr")
        }
        fold_deltas[fold] = deltas["total"]
        fold_scores[fold] = {
            "parent": parent_score,
            "selected": selected_score,
            "deltas": deltas,
            "group3_parent": _group_score(parent_fold, 3),
            "group3_selected": _group_score(selected_fold, 3),
        }

    parent_pooled = _score(parent)
    selected_pooled = _score(selected)
    pooled_deltas = {
        metric: selected_pooled[metric] - parent_pooled[metric]
        for metric in ("total", "one_minus_nmae", "ficr")
    }
    q4_parent = parent.loc[parent["fold_id"].eq(FOLDS[2])]
    q4_selected = selected.loc[selected["fold_id"].eq(FOLDS[2])]
    q4_bootstrap = _paired_issuance_bootstrap(q4_parent, q4_selected)
    q2_exact = np.array_equal(
        parent.loc[parent["fold_id"].eq(FOLDS[0]), "prediction_kwh"].to_numpy(),
        selected.loc[selected["fold_id"].eq(FOLDS[0]), "prediction_kwh"].to_numpy(),
    )
    later_positive = all(fold_deltas[fold] > 0.0 for fold in FOLDS[1:])
    pooled_positive = pooled_deltas["total"] > 0.0
    bootstrap_positive = q4_bootstrap["positive_fraction"] > 0.50
    promoted = q2_exact and later_positive and pooled_positive and bootstrap_positive

    output_path = OUTPUT / f"{MODEL_ID}-oof.parquet"
    artifact = selected.copy()
    artifact["model_id"] = MODEL_ID
    artifact[[*METRIC_COLUMNS, "fold_id", "model_id", "data_available_kst_dtm"]].to_parquet(
        output_path, index=False
    )
    policy = {
        "architecture": "exact_m263_plus_q2_selected_m114_dart_group3",
        "parent": "M263_INDEPENDENT_LINEAGE_CONVEX_ENSEMBLE",
        "replacement_lineage": "M114_STRICT_DART_BLEND",
        "replacement_group": 3,
        "q2_policy": "exact_parent",
        "q3_q4_policy": {
            "booster": "M113_LGBM_DART",
            "iterations": 140,
            "action_policy": "T0.6_G0.5",
            "m107_parent_weight": 0.6,
            "selection_fold": "dev-2023-Q2",
        },
        "groups_1_2": "exact_parent",
        "additional_search": False,
        "test_build": False,
    }
    receipt = {
        "schema_version": 1,
        "candidate_id": MODEL_ID,
        "state": (
            "LOCAL_DART_GROUP3_AUGMENTATION_PROMOTED_NO_TEST_BUILD"
            if promoted
            else "LOCAL_DART_GROUP3_AUGMENTATION_REJECTED_NO_TEST_BUILD"
        ),
        "policy": policy,
        "policy_sha256": canonical_sha256(policy),
        "fold_scores": fold_scores,
        "pooled": {
            "parent": parent_pooled,
            "selected": selected_pooled,
            "deltas": pooled_deltas,
        },
        "q4_paired_bootstrap": q4_bootstrap,
        "promotion": {
            "promoted": promoted,
            "q2_exact_parent": q2_exact,
            "q3_q4_total_deltas_positive": later_positive,
            "pooled_total_delta_positive": pooled_positive,
            "q4_bootstrap_positive_fraction_above_half": bootstrap_positive,
            "rule": (
                "Q2 is exact parent; Q3, Q4, and pooled Total deltas are positive; "
                "Q4 paired-issuance bootstrap positive fraction exceeds 0.50"
            ),
        },
        "source_sha256": {
            "open_zip": OPEN_SHA,
            "baseline_ipynb": BASELINE_SHA,
            **FROZEN_HASHES,
        },
        "prediction_path": str(output_path.relative_to(ROOT)),
        "prediction_sha256": sha256_file(output_path),
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
    }
    receipt_path = OUTPUT / f"{MODEL_ID}.json"
    receipt_path.write_text(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    print(json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
