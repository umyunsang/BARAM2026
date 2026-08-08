"""Evaluate the frozen M231 within-issuance transform on exact M265 OOF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from build_v2_transfer_sequence_challenger import (
    CAPACITIES,
    EXPECTED_SELECTIONS,
    METRIC_COLUMNS,
    _group_score,
    _paired_issuance_bootstrap,
    _score,
    _smooth,
)
from run_sequence_classifier import BASELINE, BASELINE_SHA, OPEN, OPEN_SHA

from baram.contracts.hashing import canonical_sha256, sha256_file

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "artifacts" / "backtests" / "metric-aligned-probe"
MODEL_ID = "M267_FIXED_M231_SEQUENCE_TRANSFER"
M265_PREDICTION = OUTPUT / "M265_DART_GROUP3_AUGMENTATION-oof.parquet"
M265_RECEIPT = OUTPUT / "M265_DART_GROUP3_AUGMENTATION.json"
M231_RECEIPT = (
    ROOT
    / "artifacts"
    / "submissions"
    / "E0_V2_SEQUENCE_TRANSFER-a5111cee22bc.receipt.json"
)
M231_BUILDER = Path(__file__).parent / "build_v2_transfer_sequence_challenger.py"
FROZEN_HASHES = {
    "m265_prediction": "123c0a4f8a4d42fa2e2bb164016e4d6f13c7bc11948700dbd2166c579c616b51",
    "m265_receipt": "fd936854d340ba41462cfd1562cc7c0fd2da54af5795326bc01b673e81ffbd75",
    "m231_receipt": "60bbc5fafb1ffaa4aa881bc7f7329f0817f0e399e099facf60ec76d8304441eb",
    "m231_builder": "39fee6708df8ae146c987fa0ea6af6ff320165f7218f0f30526e24870b1c3e8d",
}
FOLDS = ("dev-2023-Q2", "dev-2023-Q3", "dev-2023-Q4")
KEYS = ["forecast_id", "forecast_kst_dtm", "group_id"]
SELECTIONS = {
    1: ("mean5", 0.47500000000000003),
    2: ("median5", 0.5),
    3: ("median5", 0.325),
}


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_frozen_evidence() -> tuple[dict[str, Any], dict[str, Any]]:
    paths = {
        "m265_prediction": M265_PREDICTION,
        "m265_receipt": M265_RECEIPT,
        "m231_receipt": M231_RECEIPT,
        "m231_builder": M231_BUILDER,
    }
    for name, path in paths.items():
        observed = sha256_file(path)
        if observed != FROZEN_HASHES[name]:
            raise RuntimeError(f"M267 frozen {name} hash changed: {observed}")
    m265 = _json(M265_RECEIPT)
    m231 = _json(M231_RECEIPT)
    if (
        m265.get("candidate_id") != "M265_DART_GROUP3_AUGMENTATION"
        or m265.get("state")
        != "LOCAL_DART_GROUP3_AUGMENTATION_PROMOTED_NO_TEST_BUILD"
        or not m265.get("promotion", {}).get("promoted")
        or m265.get("prediction_sha256") != FROZEN_HASHES["m265_prediction"]
        or m265.get("new_2024_evaluation")
        or m265.get("lockbox_reopened")
        or m265.get("external_actions")
    ):
        raise RuntimeError("M267 M265 promotion boundary changed")
    recipes = m231.get("policy", {}).get("recipes")
    expected_recipes = {
        str(group_id): {"kind": kind, "smoothing_weight": weight}
        for group_id, (kind, weight) in SELECTIONS.items()
    }
    transfer = m231.get("transfer_check", {})
    if (
        EXPECTED_SELECTIONS != SELECTIONS
        or m231.get("candidate_id") != "E0_V2_SEQUENCE_TRANSFER-a5111cee22bc"
        or m231.get("state")
        != "LOCAL_V2_SEQUENCE_TRANSFER_CHALLENGER_BUILT_NOT_UPLOADED"
        or recipes != expected_recipes
        or transfer.get("selection_fold") != "dev-2023-Q3"
        or transfer.get("check_fold") != "dev-2023-Q4"
        or abs(float(transfer.get("q4_total_delta", 0.0)) - 0.002546778938880334)
        > 1e-15
        or float(transfer.get("paired_bootstrap", {}).get("positive_fraction", 0.0))
        != 0.913
        or m231.get("online_score") is not None
        or m231.get("new_2024_evaluation")
        or m231.get("lockbox_reopened")
        or m231.get("external_actions")
    ):
        raise RuntimeError("M267 M231 transfer evidence changed")
    return m265, m231


def _apply_frozen_transform(
    parent: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    output = parent.reset_index(drop=True).copy()
    diagnostics: dict[str, dict[str, int]] = {}
    for fold in FOLDS:
        diagnostics[fold] = {}
        for group_id, (kind, weight) in SELECTIONS.items():
            group_mask = output["fold_id"].eq(fold) & output["group_id"].eq(group_id)
            group_positions = np.flatnonzero(group_mask.to_numpy())
            group = output.loc[group_mask].reset_index(drop=True)
            normalized = (
                group["prediction_kwh"].to_numpy(dtype=float) / CAPACITIES[group_id]
            )
            transformed = normalized.copy()
            complete_blocks = 0
            complete_rows = 0
            incomplete_blocks = 0
            incomplete_rows = 0
            for indices in group.groupby("data_available_kst_dtm", sort=False).groups.values():
                positions = np.asarray(list(indices), dtype=int)
                order = np.argsort(group.loc[positions, "forecast_kst_dtm"].to_numpy())
                ordered = positions[order]
                times = pd.to_datetime(group.loc[ordered, "forecast_kst_dtm"])
                time_ns = times.astype("int64").to_numpy()
                complete = (
                    len(ordered) == 24
                    and times.nunique() == 24
                    and np.all(np.diff(time_ns) == 3_600_000_000_000)
                )
                if not complete:
                    incomplete_blocks += 1
                    incomplete_rows += len(ordered)
                    continue
                smoothed = _smooth(normalized[ordered], kind)
                transformed[ordered] = (
                    (1.0 - weight) * normalized[ordered] + weight * smoothed
                )
                complete_blocks += 1
                complete_rows += len(ordered)
            output.loc[group_positions, "prediction_kwh"] = np.clip(
                transformed * CAPACITIES[group_id],
                0.0,
                CAPACITIES[group_id],
            )
            diagnostics[fold][str(group_id)] = {
                "complete_blocks": complete_blocks,
                "complete_rows": complete_rows,
                "incomplete_blocks": incomplete_blocks,
                "incomplete_rows_exact_parent": incomplete_rows,
            }
    return output, diagnostics


def _self_test() -> None:
    _validate_frozen_evidence()
    values = np.asarray([0.1, 0.2, 0.9, 0.2, 0.1], dtype=float)
    expected = np.asarray([0.1, 0.2, 0.2, 0.2, 0.1], dtype=float)
    observed = _smooth(values, "median5")
    if not np.array_equal(observed, expected):
        raise RuntimeError("M267 median5 self-test changed")
    print(
        json.dumps(
            {
                "state": "M267_SELF_TEST_PASS",
                "selections": SELECTIONS,
                "median5": observed.tolist(),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if sha256_file(OPEN) != OPEN_SHA or sha256_file(BASELINE) != BASELINE_SHA:
        raise RuntimeError("immutable competition input hash mismatch")
    if args.self_test:
        _self_test()
        return

    _validate_frozen_evidence()
    parent = pd.read_parquet(M265_PREDICTION)
    required = {*METRIC_COLUMNS, "fold_id", "model_id", "data_available_kst_dtm"}
    if not required.issubset(parent.columns):
        raise RuntimeError("M267 M265 schema changed")
    if parent[KEYS].duplicated().any() or set(parent["fold_id"]) != set(FOLDS):
        raise RuntimeError("M267 M265 key/fold contract changed")
    transformed, topology = _apply_frozen_transform(parent)
    if not parent[[*KEYS, "actual_kwh", "fold_id"]].equals(
        transformed[[*KEYS, "actual_kwh", "fold_id"]]
    ):
        raise RuntimeError("M267 transformed key/actual contract changed")

    fold_scores: dict[str, Any] = {}
    fold_deltas: dict[str, float] = {}
    for fold in FOLDS:
        mask = parent["fold_id"].eq(fold)
        parent_fold = parent.loc[mask]
        transformed_fold = transformed.loc[mask]
        parent_score = _score(parent_fold)
        transformed_score = _score(transformed_fold)
        deltas = {
            metric: transformed_score[metric] - parent_score[metric]
            for metric in ("total", "one_minus_nmae", "ficr")
        }
        fold_deltas[fold] = deltas["total"]
        fold_scores[fold] = {
            "parent": parent_score,
            "transformed": transformed_score,
            "deltas": deltas,
            "group_scores": {
                str(group_id): {
                    "parent": _group_score(parent_fold, group_id),
                    "transformed": _group_score(transformed_fold, group_id),
                }
                for group_id in CAPACITIES
            },
        }

    parent_pooled = _score(parent)
    transformed_pooled = _score(transformed)
    pooled_deltas = {
        metric: transformed_pooled[metric] - parent_pooled[metric]
        for metric in ("total", "one_minus_nmae", "ficr")
    }
    q4 = parent["fold_id"].eq(FOLDS[2])
    q4_bootstrap = _paired_issuance_bootstrap(parent.loc[q4], transformed.loc[q4])
    all_fold_positive = all(delta > 0.0 for delta in fold_deltas.values())
    pooled_positive = pooled_deltas["total"] > 0.0
    bootstrap_positive = q4_bootstrap["positive_fraction"] > 0.50
    promoted = all_fold_positive and pooled_positive and bootstrap_positive

    output_path = OUTPUT / f"{MODEL_ID}-oof.parquet"
    artifact = transformed.copy()
    artifact["model_id"] = MODEL_ID
    artifact[[*METRIC_COLUMNS, "fold_id", "model_id", "data_available_kst_dtm"]].to_parquet(
        output_path,
        index=False,
    )
    policy = {
        "architecture": "fixed_m231_complete_issuance_sequence_transfer_on_m265",
        "parent": "M265_DART_GROUP3_AUGMENTATION",
        "recipes": {
            str(group_id): {"kind": kind, "smoothing_weight": weight}
            for group_id, (kind, weight) in SELECTIONS.items()
        },
        "complete_block_requirement": "exactly_24_unique_consecutive_hourly_rows",
        "incomplete_block_policy": "exact_parent",
        "clip_normalized_to_capacity": True,
        "additional_search": False,
        "test_build": False,
    }
    receipt = {
        "schema_version": 1,
        "candidate_id": MODEL_ID,
        "state": (
            "LOCAL_FIXED_M231_SEQUENCE_TRANSFER_PROMOTED_NO_TEST_BUILD"
            if promoted
            else "LOCAL_FIXED_M231_SEQUENCE_TRANSFER_REJECTED_NO_TEST_BUILD"
        ),
        "policy": policy,
        "policy_sha256": canonical_sha256(policy),
        "topology": topology,
        "fold_scores": fold_scores,
        "pooled": {
            "parent": parent_pooled,
            "transformed": transformed_pooled,
            "deltas": pooled_deltas,
        },
        "q4_paired_bootstrap": q4_bootstrap,
        "promotion": {
            "promoted": promoted,
            "all_fold_total_deltas_positive": all_fold_positive,
            "pooled_total_delta_positive": pooled_positive,
            "q4_bootstrap_positive_fraction_above_half": bootstrap_positive,
            "rule": (
                "Q2, Q3, Q4, and pooled Total deltas are positive and Q4 "
                "paired-issuance bootstrap positive fraction exceeds 0.50"
            ),
        },
        "source_sha256": {
            "open_zip": OPEN_SHA,
            "baseline_ipynb": BASELINE_SHA,
            **FROZEN_HASHES,
            "runner": sha256_file(Path(__file__)),
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
