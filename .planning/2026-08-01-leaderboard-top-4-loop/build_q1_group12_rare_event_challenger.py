"""Build the no-upload M245 scope-extended rare-event analog candidate.

The frozen M244 correction is applied to Q1 only for groups 1 and 2, whose
scope passed the strict 2022-to-2023-Q1 audit. Group 3 retains the exact M231
Q1 parent, while every Q2-Q4 value must remain exactly equal to M244.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from build_rare_event_corrected_analog_challenger import _verified_m244
from build_robust_analog_test_challenger import (
    _test_parent_long,
    _test_surface,
    _wide_submission,
)
from build_spread_shrunk_analog_challenger import _verified_parent
from build_supported_season_analog_challenger import _operating_support
from build_v2_transfer_sequence_challenger import (
    BASELINE,
    BASELINE_SHA,
    OPEN,
    OPEN_SHA,
)
from run_conditional_daily_analog_profile import Recipe, _combine, _feature_sets
from run_rare_event_corrected_analog_transfer import _rare_event_profile
from run_recency_spread_analog_transfer import _composed_profile
from run_spread_shrunk_analog_transfer import (
    _apply_spread_recipe,
    _spread_adjusted_profile,
)
from strict_dev_surface import DEV_CUTOFF, development_surface

from baram.contracts.hashing import canonical_sha256, sha256_file
from baram.submission.build import SUBMISSION_COLUMNS, build_submission
from baram.submission.validate import validate_submission

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "artifacts" / "backtests" / "metric-aligned-probe"
SUBMISSIONS = ROOT / "artifacts" / "submissions"
PARENT = SUBMISSIONS / "E0_V2_SEQUENCE_TRANSFER-a5111cee22bc.csv"
PARENT_RECEIPT = PARENT.with_suffix(".receipt.json")
M244_CANDIDATE = SUBMISSIONS / "E0_RARE_EVENT_ANALOG-5f6a9679a463.csv"
M244_CANDIDATE_SHA = (
    "1c9c0a509a71ee9ebe2eb0e15bdc37eadfed42687644b9ef0423b89de20a3ab3"
)
M245_RECEIPT = OUTPUT / "M245_Q1_GROUP12_RARE_EVENT_AUDIT.json"
M245_RECEIPT_SHA = (
    "568a18c5476eae5d90c13cfcbe254c18d292bdd581523f7a16b7d85ab1c9733d"
)
Q1_ANALOG_GROUPS = (1, 2)
Q1_FALLBACK_GROUPS = (3,)
SUPPORTED_QUARTERS = (2, 3, 4)
ONLINE_BASELINE = 0.65971
M231_Q4_DELTA = 0.0025467789388804
M244_Q4_INCREMENT = 0.004435001716904918


def _verified_m245() -> dict[str, object]:
    if sha256_file(M245_RECEIPT) != M245_RECEIPT_SHA:
        raise RuntimeError("M245 Q1 scope receipt hash mismatch")
    receipt = json.loads(M245_RECEIPT.read_text(encoding="utf-8"))
    policy = receipt["policy"]
    if (
        receipt["state"]
        != "LOCAL_Q1_GROUP12_SCOPE_PASS_TEST_EXTENSION_ELIGIBLE"
        or not receipt["scope_pass"]
        or policy["groups"] != [1, 2]
        or policy["validation_days"] != 90
        or policy["group3_policy"] != "unconditional_M231_fallback"
    ):
        raise RuntimeError("M245 Q1 scope-promotion contract changed")
    if (
        receipt.get("online_score") is not None
        or receipt.get("new_2024_evaluation")
        or receipt.get("lockbox_reopened")
        or not receipt.get("no_external_upload")
    ):
        raise RuntimeError("M245 external-evidence boundary changed")
    return receipt


def _enforce_group3_q1_fallback(
    parent: pd.DataFrame,
    output: pd.DataFrame,
    support: pd.DataFrame,
) -> pd.DataFrame:
    keys = ["forecast_id", "forecast_kst_dtm"]
    working = output.merge(
        parent[
            [
                *keys,
                "group_id",
                "data_available_kst_dtm",
                "prediction_kwh",
            ]
        ].rename(columns={"prediction_kwh": "parent_prediction_kwh"}),
        on=[*keys, "group_id", "data_available_kst_dtm"],
        validate="one_to_one",
    ).merge(
        support[[*keys, "analog_supported"]],
        on=keys,
        validate="many_to_one",
    )
    fallback = (
        ~working["analog_supported"].to_numpy(dtype=bool)
        & working["group_id"].eq(3).to_numpy()
    )
    working.loc[fallback, "prediction_kwh"] = working.loc[
        fallback, "parent_prediction_kwh"
    ].to_numpy(dtype=float)
    return working[output.columns]


def _exact_scope_diagnostics(
    candidate_path: Path,
    support: pd.DataFrame,
) -> dict[str, object]:
    parent = pd.read_csv(PARENT, encoding="utf-8-sig")
    m244 = pd.read_csv(M244_CANDIDATE, encoding="utf-8-sig")
    candidate = pd.read_csv(candidate_path, encoding="utf-8-sig")
    for frame in (parent, m244, candidate):
        frame["forecast_kst_dtm"] = pd.to_datetime(frame["forecast_kst_dtm"])
    keys = ["forecast_id", "forecast_kst_dtm"]
    if not parent[keys].equals(m244[keys]) or not parent[keys].equals(
        candidate[keys]
    ):
        raise RuntimeError("M245 candidate keys differ from a frozen parent")
    flags = candidate[keys].merge(
        support[[*keys, "analog_supported"]],
        on=keys,
        validate="one_to_one",
    )
    q1 = ~flags["analog_supported"].to_numpy(dtype=bool)
    q234 = ~q1
    columns = list(SUBMISSION_COLUMNS[2:])
    if not np.array_equal(
        candidate.loc[q234, columns].to_numpy(),
        m244.loc[q234, columns].to_numpy(),
    ):
        raise RuntimeError("M245 Q2-Q4 values differ from M244")
    diagnostics: dict[str, object] = {}
    for group_id in (1, 2, 3):
        column = f"kpx_group_{group_id}"
        delta = (
            candidate[column].to_numpy(dtype=float)
            - parent[column].to_numpy(dtype=float)
        )
        q1_delta = delta[q1]
        if group_id == 3 and not np.array_equal(
            q1_delta, np.zeros(len(q1_delta), dtype=float)
        ):
            raise RuntimeError("M245 group-3 Q1 differs from M231")
        if group_id in Q1_ANALOG_GROUPS and not np.any(q1_delta != 0.0):
            raise RuntimeError(f"M245 group-{group_id} Q1 scope was not applied")
        diagnostics[str(group_id)] = {
            "q1_changed_rows": int(np.count_nonzero(q1_delta)),
            "q1_mean_absolute_delta_kwh": float(np.mean(np.abs(q1_delta))),
            "q1_max_absolute_delta_kwh": float(np.max(np.abs(q1_delta))),
            "q234_exact_m244": bool(
                np.array_equal(
                    candidate.loc[q234, column].to_numpy(),
                    m244.loc[q234, column].to_numpy(),
                )
            ),
        }
    return diagnostics


def main() -> None:
    if sha256_file(OPEN) != OPEN_SHA or sha256_file(BASELINE) != BASELINE_SHA:
        raise RuntimeError("immutable competition input hash mismatch")
    if sha256_file(M244_CANDIDATE) != M244_CANDIDATE_SHA:
        raise RuntimeError("M244 candidate hash mismatch")
    parent_receipt = _verified_parent()
    m244 = _verified_m244()
    m245 = _verified_m245()
    recipes = {
        int(group_id): Recipe(**raw)
        for group_id, raw in m244["policy"]["recipes"].items()
    }

    surface, numeric = development_surface()
    if surface["forecast_kst_dtm"].ge(DEV_CUTOFF).any():
        raise RuntimeError("lockbox row reached M245 test builder")
    feature_sets = _feature_sets(numeric)
    test = _test_surface(feature_sets["core"])
    train_columns = [
        "forecast_id",
        "forecast_kst_dtm",
        "data_available_kst_dtm",
        "group_id",
        "actual_kwh",
        *feature_sets["core"],
    ]
    analog_surface = pd.concat(
        [surface[train_columns], test[train_columns]],
        ignore_index=True,
    )
    if set(analog_surface["forecast_kst_dtm"].dt.year.unique()) != {
        2022,
        2023,
        2025,
        2026,
    }:
        raise RuntimeError("M245 inference surface crossed the frozen year boundary")

    sample, support, support_diagnostics = _operating_support()
    all_issuances = np.sort(support["issuance"].unique())
    supported_issuances = np.sort(
        support.loc[support["analog_supported"], "issuance"].unique()
    )
    if len(all_issuances) != 365 or len(supported_issuances) != 275:
        raise RuntimeError("M245 test issuance support changed")
    test_parent = _test_parent_long(test)
    replacements: list[pd.DataFrame] = []
    retrieval: dict[str, object] = {}
    for group_id, recipe in recipes.items():
        query_issuances = (
            all_issuances if group_id in Q1_ANALOG_GROUPS else supported_issuances
        )
        corrected_profile, corrected_diagnostics = _rare_event_profile(
            analog_surface,
            feature_sets,
            group_id,
            query_issuances,
            recipe,
        )
        spread_profile, spread_diagnostics = _spread_adjusted_profile(
            analog_surface,
            feature_sets,
            group_id,
            query_issuances,
            recipe,
        )
        profile = _composed_profile(corrected_profile, spread_profile)
        replacements.append(
            _apply_spread_recipe(test_parent, profile, group_id, recipe)
        )
        retrieval[str(group_id)] = {
            "query_issuance_days": len(query_issuances),
            "rare_event": corrected_diagnostics,
            "spread": spread_diagnostics,
        }
    test_output = _combine(test_parent, replacements)
    test_output = _enforce_group3_q1_fallback(test_parent, test_output, support)
    wide = _wide_submission(test_output)

    q1_conservative_delta = min(
        float(score["total_delta"]) for score in m245["scores"].values()
    )
    policy = {
        "architecture": "q1_group12_q234_all_exact_m244_rare_event_analog_mos",
        "q1_group12_action": "apply_exact_M244_after_strict_M245_scope_pass",
        "q1_group3_action": "retain_M231_parent_exactly",
        "q2_q3_q4_action": "retain_M244_candidate_exactly",
        "q1_analog_groups": list(Q1_ANALOG_GROUPS),
        "q1_fallback_groups": list(Q1_FALLBACK_GROUPS),
        "supported_operating_quarters": list(SUPPORTED_QUARTERS),
        "m244_policy_sha256": m244["policy_sha256"],
        "m244_candidate_sha256": M244_CANDIDATE_SHA,
        "m245_scope_policy_sha256": m245["policy_sha256"],
        "m245_scope_receipt_sha256": M245_RECEIPT_SHA,
        "parent_csv_sha256": parent_receipt["submission_receipt"]["csv_sha256"],
        "q1_scope_evidence": "strict_2022_to_2023_q1_group12_two_parent_gate",
        "q1_conservative_combined_total_delta": q1_conservative_delta,
        "group3_q1_analog_exposure": False,
    }
    policy_sha = canonical_sha256(policy)
    candidate_id = f"E0_Q1_GROUP12_RARE_EVENT_ANALOG-{policy_sha[:12]}"
    candidate_path = SUBMISSIONS / f"{candidate_id}.csv"
    csv_sha = build_submission(sample, wide, candidate_path)
    validation = validate_submission(
        candidate_path,
        sample,
        candidate_id=candidate_id,
        source_sha256=OPEN_SHA,
        champion_policy_sha256=policy_sha,
        cap_modes={1: "capacity", 2: "capacity", 3: "capacity"},
    )
    if validation.csv_sha256 != csv_sha:
        raise RuntimeError("M245 build and validation hashes differ")
    difference_diagnostics = _exact_scope_diagnostics(candidate_path, support)
    supported_fraction = (
        support_diagnostics["supported_issuance_days"]
        / support_diagnostics["total_issuance_days"]
    )
    q1_fraction = (
        support_diagnostics["fallback_issuance_days"]
        / support_diagnostics["total_issuance_days"]
    )
    conditional_proxy = (
        ONLINE_BASELINE
        + M231_Q4_DELTA
        + supported_fraction * M244_Q4_INCREMENT
        + q1_fraction * (2.0 / 3.0) * q1_conservative_delta
    )
    receipt = {
        "schema_version": 1,
        "state": "LOCAL_Q1_GROUP12_RARE_EVENT_CHALLENGER_BUILT_NOT_UPLOADED",
        "candidate_id": candidate_id,
        "candidate_path": str(candidate_path.relative_to(ROOT)),
        "policy": policy,
        "policy_sha256": policy_sha,
        "support_diagnostics": support_diagnostics,
        "difference_diagnostics": difference_diagnostics,
        "retrieval_diagnostics": retrieval,
        "submission_receipt": asdict(validation),
        "parent_path": str(PARENT.relative_to(ROOT)),
        "parent_receipt_path": str(PARENT_RECEIPT.relative_to(ROOT)),
        "m244_candidate_path": str(M244_CANDIDATE.relative_to(ROOT)),
        "m245_scope_receipt_path": str(M245_RECEIPT.relative_to(ROOT)),
        "conditional_proxy": {
            "value": conditional_proxy,
            "formula": (
                "0.65971 + M231_Q4_delta + (275/365)*M244_Q4_increment "
                "+ (90/365)*(2/3)*min_M245_combined_delta"
            ),
            "evidence_class": "heuristic_prioritization_only_not_online_or_local_score",
        },
        "online_score": None,
        "new_2024_evaluation": False,
        "lockbox_reopened": False,
        "no_external_upload": True,
        "external_actions": [],
    }
    receipt_path = candidate_path.with_suffix(".receipt.json")
    receipt_path.write_text(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    print(json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
