"""Augment exact M263 with the chronology-selected M95 group-1 lineage.

M95 selected its Q3 group-1 policy on Q2 and its Q4 group-1 policy on pooled
Q2+Q3.  This runner preserves exact M263 everywhere except group 1 on Q3/Q4,
leaves Q2 byte-equivalent, and applies one frozen promotion gate.  It performs
no policy, weight, calibration, or test-period search.
"""

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
MODEL_ID = "M264_STRICT_FAMILY_GROUP1_AUGMENTATION"
M263_PREDICTION = OUTPUT / "M263_INDEPENDENT_LINEAGE_CONVEX_ENSEMBLE-oof.parquet"
M263_RECEIPT = OUTPUT / "M263_INDEPENDENT_LINEAGE_CONVEX_ENSEMBLE.json"
M95_PREDICTION = OUTPUT / "M95_STRICT_FAMILY-oof.parquet"
M95_RECEIPT = OUTPUT / "M95_STRICT_FAMILY-oof.json"
FROZEN_HASHES = {
    "m263_prediction": "5b09c1b55621766e09bbd39a25d43f6eaf36d14621577604d2474d3bea2348a8",
    "m263_receipt": "9169edab2b983dc5c47907295b8524ffba9c01615eda87d8eb288a742848e47e",
    "m95_prediction": "5ada716b01442eb73f05a3ac8d1de05dbfb3f29323f551541015acf5a0223cb7",
    "m95_receipt": "c3624185cbaa185d221dd5455005c443a1af3ba357cfd981d99c605b24176b9b",
}
FOLDS = ("dev-2023-Q2", "dev-2023-Q3", "dev-2023-Q4")
KEYS = ["forecast_id", "forecast_kst_dtm", "group_id"]


def _read_receipts() -> tuple[dict[str, object], dict[str, object]]:
    paths = {
        "m263_prediction": M263_PREDICTION,
        "m263_receipt": M263_RECEIPT,
        "m95_prediction": M95_PREDICTION,
        "m95_receipt": M95_RECEIPT,
    }
    for name, path in paths.items():
        observed = sha256_file(path)
        if observed != FROZEN_HASHES[name]:
            raise RuntimeError(f"M264 frozen {name} hash changed: {observed}")
    m263 = json.loads(M263_RECEIPT.read_text(encoding="utf-8"))
    m95 = json.loads(M95_RECEIPT.read_text(encoding="utf-8"))
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
        raise RuntimeError("M264 M263 evidence boundary changed")
    selections = m95.get("selections", {})
    if (
        m95.get("candidate_id") != "M95_STRICT_FAMILY"
        or m95.get("scope") != "Q2 fixed; Q3 selected on Q2; Q4 selected on pooled Q2-Q3"
        or m95.get("prediction_sha256") != FROZEN_HASHES["m95_prediction"]
        or selections.get("dev-2023-Q3", {}).get("1", {}).get("policy")
        != "M72:T0.6_G0.5"
        or selections.get("dev-2023-Q4", {}).get("1", {}).get("policy")
        != "M68:T0.6_G0.5"
        or selections.get("dev-2023-Q4", {}).get("1", {}).get("history_folds")
        != ["dev-2023-Q2", "dev-2023-Q3"]
        or m95.get("new_2024_evaluation")
        or m95.get("lockbox_reopened")
        or m95.get("external_actions")
    ):
        raise RuntimeError("M264 M95 chronology/evidence boundary changed")
    return m263, m95


def _aligned_library() -> pd.DataFrame:
    m263 = pd.read_parquet(M263_PREDICTION)
    m95 = pd.read_parquet(M95_PREDICTION)
    required = {*METRIC_COLUMNS, "fold_id", "model_id"}
    if not required.issubset(m263.columns) or not required.issubset(m95.columns):
        raise RuntimeError("M264 parent schema changed")
    if m263[KEYS].duplicated().any() or m95[KEYS].duplicated().any():
        raise RuntimeError("M264 parent key uniqueness changed")
    right = m95[[*KEYS, "actual_kwh", "prediction_kwh", "fold_id"]].rename(
        columns={
            "actual_kwh": "m95_actual_kwh",
            "prediction_kwh": "m95_prediction_kwh",
            "fold_id": "m95_fold_id",
        }
    )
    library = m263.rename(columns={"prediction_kwh": "m263_prediction_kwh"}).merge(
        right,
        on=KEYS,
        how="inner",
        validate="one_to_one",
        sort=False,
    )
    if len(library) != len(m263) or len(library) != len(m95):
        raise RuntimeError("M264 parent key coverage changed")
    if set(library["fold_id"]) != set(FOLDS) or not library["fold_id"].eq(
        library["m95_fold_id"]
    ).all():
        raise RuntimeError("M264 parent fold alignment changed")
    if not np.allclose(
        library["actual_kwh"].to_numpy(dtype=float),
        library["m95_actual_kwh"].to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError("M264 parent actual alignment changed")
    return library


def _metric_frame(library: pd.DataFrame, use_m95_group1: bool) -> pd.DataFrame:
    output = library[[*KEYS, "actual_kwh", "fold_id", "data_available_kst_dtm"]].copy()
    output["prediction_kwh"] = library["m263_prediction_kwh"].to_numpy(dtype=float)
    if use_m95_group1:
        affected = library["group_id"].eq(1) & library["fold_id"].isin(FOLDS[1:])
        output.loc[affected, "prediction_kwh"] = library.loc[
            affected, "m95_prediction_kwh"
        ].to_numpy(dtype=float)
    return output


def main() -> None:
    if sha256_file(OPEN) != OPEN_SHA or sha256_file(BASELINE) != BASELINE_SHA:
        raise RuntimeError("immutable competition input hash mismatch")
    m263_receipt, m95_receipt = _read_receipts()
    library = _aligned_library()
    parent = _metric_frame(library, use_m95_group1=False)
    selected = _metric_frame(library, use_m95_group1=True)

    fold_scores: dict[str, object] = {}
    fold_deltas: dict[str, float] = {}
    for fold in FOLDS:
        parent_fold = parent.loc[parent["fold_id"].eq(fold)]
        selected_fold = selected.loc[selected["fold_id"].eq(fold)]
        m95_fold = library.loc[library["fold_id"].eq(fold), [*KEYS, "actual_kwh"]].copy()
        m95_fold["prediction_kwh"] = library.loc[
            library["fold_id"].eq(fold), "m95_prediction_kwh"
        ].to_numpy(dtype=float)
        parent_score = _score(parent_fold)
        selected_score = _score(selected_fold)
        m95_score = _score(m95_fold)
        key = fold.lower().rsplit("-", maxsplit=1)[-1]
        expected_parent = m263_receipt["fold_scores"][key]["selected"]
        expected_m95 = m95_receipt["fold_scores"][fold]
        for metric in ("total", "one_minus_nmae", "ficr"):
            if abs(parent_score[metric] - expected_parent[metric]) > 1e-12:
                raise RuntimeError(f"M264 M263 {fold} {metric} reproduction changed")
            if abs(m95_score[metric] - expected_m95[metric]) > 1e-12:
                raise RuntimeError(f"M264 M95 {fold} {metric} reproduction changed")
        deltas = {
            metric: selected_score[metric] - parent_score[metric]
            for metric in ("total", "one_minus_nmae", "ficr")
        }
        fold_deltas[fold] = deltas["total"]
        fold_scores[fold] = {
            "parent": parent_score,
            "selected": selected_score,
            "deltas": deltas,
            "group1_parent": _group_score(parent_fold, 1),
            "group1_selected": _group_score(selected_fold, 1),
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
        "architecture": "exact_m263_plus_chronology_selected_m95_group1",
        "parent": "M263_INDEPENDENT_LINEAGE_CONVEX_ENSEMBLE",
        "replacement_lineage": "M95_STRICT_FAMILY",
        "replacement_group": 1,
        "q2_policy": "exact_parent",
        "q3_policy": "M72:T0.6_G0.5 selected on Q2",
        "q4_policy": "M68:T0.6_G0.5 selected on pooled Q2+Q3",
        "groups_2_3": "exact_parent",
        "blend_mass": None,
        "shift_or_snap": None,
        "temporal_transform": None,
        "post_result_rescue": False,
        "test_build": False,
    }
    receipt = {
        "schema_version": 1,
        "candidate_id": MODEL_ID,
        "state": (
            "LOCAL_STRICT_FAMILY_GROUP1_AUGMENTATION_PROMOTED_NO_TEST_BUILD"
            if promoted
            else "LOCAL_STRICT_FAMILY_GROUP1_AUGMENTATION_REJECTED_NO_TEST_BUILD"
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
