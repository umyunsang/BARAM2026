"""Build a deployment-support-constrained analog challenger.

M234 was evaluated only on operating quarters Q2-Q4.  This deterministic
candidate therefore retains the independently transferred M231 parent for Q1
and applies M234 only to complete Q2-Q4 operating-day issuance batches.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from build_v2_transfer_sequence_challenger import (
    BASELINE,
    BASELINE_SHA,
    OPEN,
    OPEN_SHA,
    _test_topology,
)

from baram.contracts.hashing import canonical_sha256, sha256_file
from baram.submission.build import SUBMISSION_COLUMNS, build_submission
from baram.submission.validate import validate_submission

ROOT = Path(__file__).resolve().parents[2]
SUBMISSIONS = ROOT / "artifacts" / "submissions"
PARENT = SUBMISSIONS / "E0_V2_SEQUENCE_TRANSFER-a5111cee22bc.csv"
PARENT_RECEIPT = PARENT.with_suffix(".receipt.json")
FULL_ANALOG = SUBMISSIONS / "E0_ROBUST_ANALOG_DEV-08f07b6d9e87.csv"
FULL_ANALOG_RECEIPT = FULL_ANALOG.with_suffix(".receipt.json")
SUPPORTED_QUARTERS = (2, 3, 4)
ONLINE_BASELINE = 0.65971
M231_Q4_DELTA = 0.0025467789388804
M234_Q4_INCREMENT = 0.002682607515930857


def _verified_receipt(path: Path, candidate: Path) -> dict[str, object]:
    receipt = json.loads(path.read_text(encoding="utf-8"))
    expected = receipt["submission_receipt"]["csv_sha256"]
    if sha256_file(candidate) != expected:
        raise RuntimeError(f"candidate hash differs from receipt: {candidate.name}")
    if receipt.get("online_score") is not None:
        raise RuntimeError("local parent receipt unexpectedly contains an online score")
    return receipt


def _operating_support() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    sample, issuance_map = _test_topology()
    issuance_map["forecast_kst_dtm"] = pd.to_datetime(
        issuance_map["forecast_kst_dtm"]
    )
    issuance_map["issuance"] = pd.to_datetime(issuance_map["issuance"])
    batches = (
        issuance_map.groupby("issuance", sort=True)
        .agg(
            operating_day=("forecast_kst_dtm", "min"),
            forecast_rows=("forecast_id", "size"),
        )
        .reset_index()
    )
    if len(batches) != 365 or not batches["forecast_rows"].eq(24).all():
        raise RuntimeError("official test operating-day topology changed")
    if (
        batches["operating_day"].min() != pd.Timestamp("2025-01-01 01:00:00")
        or batches["operating_day"].max()
        != pd.Timestamp("2025-12-31 01:00:00")
    ):
        raise RuntimeError("official test operating-day boundary changed")
    batches["operating_quarter"] = batches["operating_day"].dt.quarter
    batches["analog_supported"] = batches["operating_quarter"].isin(
        SUPPORTED_QUARTERS
    )
    support = issuance_map.merge(
        batches[["issuance", "operating_day", "operating_quarter", "analog_supported"]],
        on="issuance",
        validate="many_to_one",
    )
    if support.duplicated(["forecast_id", "forecast_kst_dtm"]).any():
        raise RuntimeError("test support map contains duplicate forecast keys")
    diagnostics = {
        "total_issuance_days": len(batches),
        "supported_issuance_days": int(batches["analog_supported"].sum()),
        "fallback_issuance_days": int((~batches["analog_supported"]).sum()),
        "supported_forecast_rows": int(support["analog_supported"].sum()),
        "fallback_forecast_rows": int((~support["analog_supported"]).sum()),
        "operating_days_by_quarter": {
            str(quarter): int(batches["operating_quarter"].eq(quarter).sum())
            for quarter in range(1, 5)
        },
        "first_operating_day": str(batches["operating_day"].min()),
        "last_operating_day": str(batches["operating_day"].max()),
    }
    return sample, support, diagnostics


def _blend(support: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    parent = pd.read_csv(PARENT, encoding="utf-8-sig")
    full = pd.read_csv(FULL_ANALOG, encoding="utf-8-sig")
    if list(parent.columns) != list(SUBMISSION_COLUMNS) or list(full.columns) != list(
        SUBMISSION_COLUMNS
    ):
        raise RuntimeError("parent submission schema changed")
    for frame in (parent, full):
        frame["forecast_kst_dtm"] = pd.to_datetime(frame["forecast_kst_dtm"])
    keys = ["forecast_id", "forecast_kst_dtm"]
    if not parent[keys].equals(full[keys]):
        raise RuntimeError("parent and full-analog keys differ")
    combined = parent.merge(
        full,
        on=keys,
        suffixes=("__parent", "__analog"),
        validate="one_to_one",
    ).merge(
        support[[*keys, "operating_day", "operating_quarter", "analog_supported"]],
        on=keys,
        validate="one_to_one",
    )
    output = combined[keys].copy()
    active = combined["analog_supported"].to_numpy(dtype=bool)
    difference_diagnostics: dict[str, object] = {}
    for group_id in (1, 2, 3):
        column = f"kpx_group_{group_id}"
        parent_values = combined[f"{column}__parent"].to_numpy(dtype=float)
        analog_values = combined[f"{column}__analog"].to_numpy(dtype=float)
        output[column] = np.where(active, analog_values, parent_values)
        delta = analog_values - parent_values
        difference_diagnostics[str(group_id)] = {
            "supported_mean_absolute_delta_kwh": float(
                np.mean(np.abs(delta[active]))
            ),
            "supported_max_absolute_delta_kwh": float(
                np.max(np.abs(delta[active]))
            ),
            "fallback_max_absolute_delta_kwh": float(
                np.max(np.abs(output.loc[~active, column] - parent_values[~active]))
            ),
        }
    if not np.array_equal(
        output.loc[~active, list(SUBMISSION_COLUMNS[2:])].to_numpy(),
        parent.loc[~active, list(SUBMISSION_COLUMNS[2:])].to_numpy(),
    ):
        raise RuntimeError("Q1 fallback differs from the M231 parent")
    if not np.array_equal(
        output.loc[active, list(SUBMISSION_COLUMNS[2:])].to_numpy(),
        full.loc[active, list(SUBMISSION_COLUMNS[2:])].to_numpy(),
    ):
        raise RuntimeError("Q2-Q4 output differs from the M234 challenger")
    return output, difference_diagnostics


def main() -> None:
    if sha256_file(OPEN) != OPEN_SHA or sha256_file(BASELINE) != BASELINE_SHA:
        raise RuntimeError("immutable competition input hash mismatch")
    parent_receipt = _verified_receipt(PARENT_RECEIPT, PARENT)
    analog_receipt = _verified_receipt(FULL_ANALOG_RECEIPT, FULL_ANALOG)
    if not bool(analog_receipt.get("q4_used_in_recipe_selection")):
        raise RuntimeError("M234 evidence classification changed")
    sample, support, support_diagnostics = _operating_support()
    predictions, difference_diagnostics = _blend(support)

    policy = {
        "architecture": "operating_quarter_support_constrained_daily_analog",
        "q1_action": "retain_M231_parent",
        "q2_q3_q4_action": "apply_M234_full_analog",
        "supported_operating_quarters": list(SUPPORTED_QUARTERS),
        "operating_day_contract": "minimum forecast timestamp per complete issuance batch",
        "parent_csv_sha256": parent_receipt["submission_receipt"]["csv_sha256"],
        "full_analog_csv_sha256": analog_receipt["submission_receipt"]["csv_sha256"],
        "q1_analog_exposure": False,
    }
    policy_sha = canonical_sha256(policy)
    candidate_id = f"E0_SUPPORTED_SEASON_ANALOG-{policy_sha[:12]}"
    candidate_path = SUBMISSIONS / f"{candidate_id}.csv"
    csv_sha = build_submission(sample, predictions, candidate_path)
    validation = validate_submission(
        candidate_path,
        sample,
        candidate_id=candidate_id,
        source_sha256=OPEN_SHA,
        champion_policy_sha256=policy_sha,
        cap_modes={1: "capacity", 2: "capacity", 3: "capacity"},
    )
    if validation.csv_sha256 != csv_sha:
        raise RuntimeError("supported-season build and validation hashes differ")
    supported_fraction = (
        support_diagnostics["supported_issuance_days"]
        / support_diagnostics["total_issuance_days"]
    )
    conditional_proxy = (
        ONLINE_BASELINE
        + M231_Q4_DELTA
        + supported_fraction * M234_Q4_INCREMENT
    )
    receipt = {
        "schema_version": 1,
        "state": "LOCAL_SUPPORT_CONSTRAINED_ANALOG_CHALLENGER_BUILT_NOT_UPLOADED",
        "candidate_id": candidate_id,
        "candidate_path": str(candidate_path.relative_to(ROOT)),
        "policy": policy,
        "policy_sha256": policy_sha,
        "support_diagnostics": support_diagnostics,
        "difference_diagnostics": difference_diagnostics,
        "submission_receipt": asdict(validation),
        "parent_path": str(PARENT.relative_to(ROOT)),
        "parent_receipt_path": str(PARENT_RECEIPT.relative_to(ROOT)),
        "full_analog_path": str(FULL_ANALOG.relative_to(ROOT)),
        "full_analog_receipt_path": str(FULL_ANALOG_RECEIPT.relative_to(ROOT)),
        "conditional_proxy": {
            "value": conditional_proxy,
            "formula": "0.65971 + M231_Q4_delta + (275/365)*M234_Q4_increment",
            "evidence_class": "heuristic_prioritization_only_not_online_score",
        },
        "q4_used_in_analog_recipe_selection": True,
        "q1_analog_exposure": False,
        "online_score": None,
        "new_2024_evaluation": False,
        "lockbox_reopened": False,
        "no_external_upload": True,
        "external_actions": [],
    }
    receipt_path = candidate_path.with_suffix(".receipt.json")
    receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
