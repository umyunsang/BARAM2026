"""Build M268 by applying the frozen promoted M267 transform to M266."""

from __future__ import annotations

import argparse
import ast
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from build_v2_transfer_sequence_challenger import EXPECTED_SELECTIONS, _smooth
from run_sequence_classifier import BASELINE_SHA, CAPACITIES, OPEN_SHA

from baram.contracts.hashing import canonical_sha256, sha256_file
from baram.submission.build import build_submission
from baram.submission.validate import validate_submission

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "artifacts" / "cache" / OPEN_SHA
SUBMISSIONS = ROOT / "artifacts" / "submissions"
EVIDENCE = ROOT / "artifacts" / "backtests" / "metric-aligned-probe"
CANDIDATE_ID = "M268_FULL_HISTORY_M231_SEQUENCE_TRANSFER"
CANDIDATE_PATH = SUBMISSIONS / "submission_M268.csv"
RECEIPT_PATH = SUBMISSIONS / "submission_M268.receipt.json"
M266_SUBMISSION = SUBMISSIONS / "submission_M266.csv"
M266_RECEIPT = SUBMISSIONS / "submission_M266.receipt.json"
M267_PREDICTION = EVIDENCE / "M267_FIXED_M231_SEQUENCE_TRANSFER-oof.parquet"
M267_RECEIPT = EVIDENCE / "M267_FIXED_M231_SEQUENCE_TRANSFER.json"
M231_RECEIPT = SUBMISSIONS / "E0_V2_SEQUENCE_TRANSFER-a5111cee22bc.receipt.json"
M231_BUILDER = Path(__file__).parent / "build_v2_transfer_sequence_challenger.py"
SUBMISSION_KEYS = CACHE / "submission_keys.parquet"
TEST_FEATURES = CACHE / "test_features.parquet"
FROZEN_HASHES = {
    "m266_submission": "10955b934b035e273317f8a25f98652775ca73bad3854ab683869fe862a84cb3",
    "m266_receipt": "f70ca20ed5c4ea534ecdb83f53fa3d9b1c436ad8808bc3c845738df8b754c34d",
    "m267_prediction": "1d1ba64dd2e3544ecbf87239612f380b8558f2362f91c8b808dec299959173bb",
    "m267_receipt": "452512369655b95f1204b8f215774ef5cf75cc63ed685c0fe4d79cca92b738f5",
    "m231_receipt": "60bbc5fafb1ffaa4aa881bc7f7329f0817f0e399e099facf60ec76d8304441eb",
    "m231_builder": "39fee6708df8ae146c987fa0ea6af6ff320165f7218f0f30526e24870b1c3e8d",
    "submission_keys": "f675805cb030ce6d401c735194bc48965cd398a1abd0b4bea21ec546a750ddb7",
    "test_features": "7aab7538f9cb25d93b0aaeb6882eeb6746c69f2623fb65022a926111daf5948b",
}
SELECTIONS = {
    1: ("mean5", 0.47500000000000003),
    2: ("median5", 0.5),
    3: ("median5", 0.325),
}


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _artifact_kib() -> int:
    return (
        sum(
            path.stat().st_size
            for path in (ROOT / "artifacts").rglob("*")
            if path.is_file()
        )
        // 1024
    )


def _direct_score_calls() -> list[str]:
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    forbidden = {"evaluate_official", "_score"}
    calls: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            name = node.func.attr
        else:
            continue
        if name in forbidden:
            calls.append(f"{name}:{node.lineno}")
    return calls


def _validate_frozen_evidence() -> tuple[dict[str, Any], dict[str, Any]]:
    paths = {
        "m266_submission": M266_SUBMISSION,
        "m266_receipt": M266_RECEIPT,
        "m267_prediction": M267_PREDICTION,
        "m267_receipt": M267_RECEIPT,
        "m231_receipt": M231_RECEIPT,
        "m231_builder": M231_BUILDER,
        "submission_keys": SUBMISSION_KEYS,
        "test_features": TEST_FEATURES,
    }
    for name, path in paths.items():
        observed = sha256_file(path)
        if observed != FROZEN_HASHES[name]:
            raise RuntimeError(f"M268 frozen {name} hash changed: {observed}")
    if _direct_score_calls():
        raise RuntimeError(f"M268 contains direct score calls: {_direct_score_calls()}")
    m266 = _json(M266_RECEIPT)
    m267 = _json(M267_RECEIPT)
    m231 = _json(M231_RECEIPT)
    if (
        m266.get("candidate_id") != "M266_FULL_HISTORY_DART_GROUP3_AUGMENTATION"
        or m266.get("state")
        != "LOCAL_FROZEN_M266_FULL_HISTORY_DART_GROUP3_BUILT_NOT_UPLOADED"
        or m266.get("submission_receipt", {}).get("csv_sha256")
        != FROZEN_HASHES["m266_submission"]
        or m266.get("online_score") is not None
        or m266.get("new_2024_evaluation")
        or m266.get("lockbox_reopened")
        or m266.get("external_actions")
    ):
        raise RuntimeError("M268 M266 deployment boundary changed")
    if (
        m267.get("candidate_id") != "M267_FIXED_M231_SEQUENCE_TRANSFER"
        or m267.get("state")
        != "LOCAL_FIXED_M231_SEQUENCE_TRANSFER_PROMOTED_NO_TEST_BUILD"
        or not m267.get("promotion", {}).get("promoted")
        or m267.get("prediction_sha256") != FROZEN_HASHES["m267_prediction"]
        or m267.get("policy", {}).get("recipes")
        != {
            str(group_id): {"kind": kind, "smoothing_weight": weight}
            for group_id, (kind, weight) in SELECTIONS.items()
        }
        or m267.get("new_2024_evaluation")
        or m267.get("lockbox_reopened")
        or m267.get("external_actions")
    ):
        raise RuntimeError("M268 M267 promotion boundary changed")
    if (
        EXPECTED_SELECTIONS != SELECTIONS
        or m231.get("candidate_id") != "E0_V2_SEQUENCE_TRANSFER-a5111cee22bc"
        or m231.get("policy", {}).get("recipes")
        != m267.get("policy", {}).get("recipes")
        or m231.get("new_2024_evaluation")
        or m231.get("lockbox_reopened")
        or m231.get("external_actions")
    ):
        raise RuntimeError("M268 M231 recipe boundary changed")
    return m266, m267


def _issuance_map() -> tuple[pd.DataFrame, dict[str, Any]]:
    test = pd.read_parquet(
        TEST_FEATURES,
        columns=[
            "forecast_id",
            "forecast_kst_dtm",
            "data_available_kst_dtm",
            "group_id",
        ],
    )
    if len(test) != 26_280:
        raise RuntimeError("M268 test feature cardinality changed")
    grouped = test.groupby(["forecast_id", "forecast_kst_dtm"], sort=False).agg(
        issuance_count=("data_available_kst_dtm", "nunique"),
        group_count=("group_id", "nunique"),
        issuance=("data_available_kst_dtm", "first"),
    )
    if not (
        grouped["issuance_count"].eq(1).all()
        and grouped["group_count"].eq(3).all()
    ):
        raise RuntimeError("M268 test forecast/group/issuance topology changed")
    mapping = grouped.reset_index()[
        ["forecast_id", "forecast_kst_dtm", "issuance"]
    ]
    sizes = mapping.groupby("issuance", sort=False).size()
    if len(sizes) != 365 or not sizes.eq(24).all():
        raise RuntimeError("M268 test issuance blocks are not 365 complete days")
    return mapping, {
        "test_long_rows": len(test),
        "forecast_rows": len(mapping),
        "issuance_blocks": len(sizes),
        "rows_per_issuance": 24,
        "groups_per_forecast": 3,
    }


def _transform(parent: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    working = parent.merge(
        mapping,
        on=["forecast_id", "forecast_kst_dtm"],
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if working["issuance"].isna().any():
        raise RuntimeError("M268 parent/issuance join changed")
    for group_id, (kind, weight) in SELECTIONS.items():
        column = f"kpx_group_{group_id}"
        normalized = working[column].to_numpy(dtype=float) / CAPACITIES[group_id]
        transformed = normalized.copy()
        for indices in working.groupby("issuance", sort=False).groups.values():
            positions = np.asarray(list(indices), dtype=int)
            order = np.argsort(working.loc[positions, "forecast_kst_dtm"].to_numpy())
            ordered = positions[order]
            if len(ordered) != 24:
                raise RuntimeError("M268 incomplete test issuance reached smoother")
            smoothed = _smooth(normalized[ordered], kind)
            transformed[ordered] = (
                (1.0 - weight) * normalized[ordered] + weight * smoothed
            )
        working[column] = np.clip(
            transformed * CAPACITIES[group_id],
            0.0,
            CAPACITIES[group_id],
        )
    return working[parent.columns.tolist()]


def _self_test() -> None:
    _validate_frozen_evidence()
    values = np.asarray([0.1, 0.2, 0.9, 0.2, 0.1], dtype=float)
    observed = _smooth(values, "median5")
    expected = np.asarray([0.1, 0.2, 0.2, 0.2, 0.1], dtype=float)
    if not np.array_equal(observed, expected):
        raise RuntimeError("M268 smoother self-test changed")
    print(
        json.dumps(
            {
                "state": "M268_SELF_TEST_PASS",
                "direct_score_calls": _direct_score_calls(),
                "selections": SELECTIONS,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        return

    m266, m267 = _validate_frozen_evidence()
    parent = pd.read_csv(M266_SUBMISSION, encoding="utf-8-sig")
    expected_columns = [
        "forecast_id",
        "forecast_kst_dtm",
        "kpx_group_1",
        "kpx_group_2",
        "kpx_group_3",
    ]
    if parent.columns.tolist() != expected_columns or len(parent) != 8_760:
        raise RuntimeError("M268 M266 schema changed")
    parent["forecast_kst_dtm"] = pd.to_datetime(
        parent["forecast_kst_dtm"], errors="raise"
    )
    if parent[["forecast_id", "forecast_kst_dtm"]].duplicated().any():
        raise RuntimeError("M268 M266 keys are not unique")
    mapping, topology = _issuance_map()
    wide = _transform(parent, mapping)
    sample = pd.read_parquet(SUBMISSION_KEYS)
    policy = {
        "architecture": "full_history_m267_fixed_m231_sequence_transfer",
        "parent": "submission_M266.csv",
        "parent_csv_sha256": FROZEN_HASHES["m266_submission"],
        "recipes": {
            str(group_id): {"kind": kind, "smoothing_weight": weight}
            for group_id, (kind, weight) in SELECTIONS.items()
        },
        "topology": "365_complete_24_hour_test_issuance_blocks",
        "clip_bounds": {str(group): [0.0, capacity] for group, capacity in CAPACITIES.items()},
        "model_fit": False,
        "score_calls": 0,
        "selection_after_parent_build": False,
    }
    policy_sha = canonical_sha256(policy)
    csv_sha = build_submission(sample, wide, CANDIDATE_PATH)
    validation = validate_submission(
        CANDIDATE_PATH,
        sample,
        candidate_id=CANDIDATE_ID,
        source_sha256=OPEN_SHA,
        champion_policy_sha256=policy_sha,
        cap_modes={1: "capacity", 2: "capacity", 3: "capacity"},
    )
    if validation.csv_sha256 != csv_sha:
        raise RuntimeError("M268 build and validation hashes differ")
    parsed = pd.read_csv(CANDIDATE_PATH, encoding="utf-8-sig")
    for column in ("kpx_group_1", "kpx_group_2", "kpx_group_3"):
        error = np.abs(
            parsed[column].to_numpy(dtype=float)
            - wide[column].to_numpy(dtype=float)
        )
        if float(error.max()) > 1e-10:
            raise RuntimeError(f"M268 {column} CSV round-trip changed")
    values = parsed[["kpx_group_1", "kpx_group_2", "kpx_group_3"]].to_numpy(
        dtype=float
    )
    receipt = {
        "schema_version": 1,
        "state": "LOCAL_FROZEN_M268_SEQUENCE_TRANSFER_BUILT_NOT_UPLOADED",
        "candidate_id": CANDIDATE_ID,
        "candidate_path": str(CANDIDATE_PATH.relative_to(ROOT)),
        "policy": policy,
        "policy_sha256": policy_sha,
        "topology": topology,
        "prediction_diagnostics": {
            "row_count": len(parsed),
            "minimum_kwh": float(values.min()),
            "maximum_kwh": float(values.max()),
            "changed_rows_by_group": {
                str(group_id): int(
                    (
                        parsed[f"kpx_group_{group_id}"].to_numpy(dtype=float)
                        != parent[f"kpx_group_{group_id}"].to_numpy(dtype=float)
                    ).sum()
                )
                for group_id in CAPACITIES
            },
        },
        "submission_receipt": asdict(validation),
        "inputs": {
            "open_zip_sha256": OPEN_SHA,
            "baseline_ipynb_sha256": BASELINE_SHA,
            **FROZEN_HASHES,
            "builder_code_sha256": sha256_file(Path(__file__)),
        },
        "source_evidence": {
            "m266_state": m266["state"],
            "m267_state": m267["state"],
            "m267_policy_sha256": m267["policy_sha256"],
        },
        "evaluation_contract": {
            "target_total_strictly_greater_than": 0.66,
            "direct_score_calls": _direct_score_calls(),
            "score_function_calls": 0,
            "metrics_computed_on_2024": False,
            "2024_slice_or_comparison_created": False,
            "model_fit": False,
            "selection_after_parent_build": False,
            "local_score": None,
            "online_score": None,
            "target_status": "UNVERIFIED_REQUIRES_EXTERNAL_DACON_RESULT",
        },
        "artifact_budget": {
            "limit_kib": 6 * 1024 * 1024,
            "candidate_bytes": CANDIDATE_PATH.stat().st_size,
            "status": "PASS_AT_BUILD",
        },
        "online_score": None,
        "new_2024_evaluation": False,
        "lockbox_reopened": False,
        "no_external_upload": True,
        "external_actions": [],
    }
    if _artifact_kib() > receipt["artifact_budget"]["limit_kib"]:
        raise RuntimeError("M268 artifact budget exceeded")
    RECEIPT_PATH.write_text(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    print(json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
