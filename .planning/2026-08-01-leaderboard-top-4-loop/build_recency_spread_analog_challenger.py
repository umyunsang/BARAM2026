"""Build the no-upload M243 candidate with an exact M231 Q1 fallback."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from build_robust_analog_test_challenger import (
    _test_parent_long,
    _test_surface,
    _wide_submission,
)
from build_spread_shrunk_analog_challenger import (
    _enforce_exact_q1_fallback,
    _fallback_and_difference_diagnostics,
    _verified_parent,
)
from build_supported_season_analog_challenger import _operating_support
from build_v2_transfer_sequence_challenger import (
    BASELINE,
    BASELINE_SHA,
    OPEN,
    OPEN_SHA,
)
from run_conditional_daily_analog_profile import Recipe, _combine, _feature_sets
from run_recency_spread_analog_transfer import _composed_profile
from run_recency_weighted_analog_transfer import _recency_profile
from run_spread_shrunk_analog_transfer import (
    _apply_spread_recipe,
    _spread_adjusted_profile,
)
from strict_dev_surface import DEV_CUTOFF, development_surface

from baram.contracts.hashing import canonical_sha256, sha256_file
from baram.submission.build import build_submission
from baram.submission.validate import validate_submission

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "artifacts" / "backtests" / "metric-aligned-probe"
SUBMISSIONS = ROOT / "artifacts" / "submissions"
PARENT = SUBMISSIONS / "E0_V2_SEQUENCE_TRANSFER-a5111cee22bc.csv"
PARENT_RECEIPT = PARENT.with_suffix(".receipt.json")
M243_RECEIPT = OUTPUT / "M243_RECENCY_SPREAD_ANALOG_Q234.json"
M243_RECEIPT_SHA = (
    "be4a590ec53d74a27e4d9d9e280536a1f1e9af3fd73d333889b92d5e143565c0"
)
SUPPORTED_QUARTERS = (2, 3, 4)
ONLINE_BASELINE = 0.65971
M231_Q4_DELTA = 0.0025467789388804
M243_Q4_INCREMENT = 0.0034924832702925013


def _verified_m243() -> dict[str, object]:
    if sha256_file(M243_RECEIPT) != M243_RECEIPT_SHA:
        raise RuntimeError("M243 development receipt hash mismatch")
    receipt = json.loads(M243_RECEIPT.read_text(encoding="utf-8"))
    if (
        receipt["state"]
        != "LOCAL_RECENCY_SPREAD_COMPOSITION_PROMOTED_FOR_TEST_BUILD"
        or not receipt["promotion"]["promoted"]
        or receipt["policy"]["recency_half_life_days"] != 365.0
        or receipt["policy"]["spread_reference_quantile"] != 0.75
    ):
        raise RuntimeError("M243 development promotion contract changed")
    if (
        receipt.get("online_score") is not None
        or receipt.get("new_2024_evaluation")
        or receipt.get("lockbox_reopened")
        or not receipt.get("no_external_upload")
    ):
        raise RuntimeError("M243 external-evidence boundary changed")
    return receipt


def main() -> None:
    if sha256_file(OPEN) != OPEN_SHA or sha256_file(BASELINE) != BASELINE_SHA:
        raise RuntimeError("immutable competition input hash mismatch")
    parent_receipt = _verified_parent()
    m243 = _verified_m243()
    recipes = {
        int(group_id): Recipe(**raw)
        for group_id, raw in m243["policy"]["recipes"].items()
    }

    surface, numeric = development_surface()
    if surface["forecast_kst_dtm"].ge(DEV_CUTOFF).any():
        raise RuntimeError("lockbox row reached M243 test builder")
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
        raise RuntimeError("M243 inference surface crossed the frozen year boundary")

    sample, support, support_diagnostics = _operating_support()
    supported_issuances = np.sort(
        support.loc[support["analog_supported"], "issuance"].unique()
    )
    if len(supported_issuances) != 275:
        raise RuntimeError("M243 supported issuance count changed")
    test_parent = _test_parent_long(test)
    replacements: list[pd.DataFrame] = []
    retrieval: dict[str, object] = {}
    for group_id, recipe in recipes.items():
        recency_profile, recency_diagnostics = _recency_profile(
            analog_surface,
            feature_sets,
            group_id,
            supported_issuances,
            recipe,
        )
        spread_profile, spread_diagnostics = _spread_adjusted_profile(
            analog_surface,
            feature_sets,
            group_id,
            supported_issuances,
            recipe,
        )
        profile = _composed_profile(recency_profile, spread_profile)
        replacements.append(
            _apply_spread_recipe(test_parent, profile, group_id, recipe)
        )
        retrieval[str(group_id)] = {
            "recency": recency_diagnostics,
            "spread": spread_diagnostics,
        }
    test_output = _combine(test_parent, replacements)
    test_output = _enforce_exact_q1_fallback(test_parent, test_output, support)
    difference_diagnostics = _fallback_and_difference_diagnostics(
        test_parent,
        test_output,
        support,
    )
    wide = _wide_submission(test_output)

    policy = {
        "architecture": "q1_m231_q234_m243_recency_plus_spread",
        "q1_action": "retain_M231_parent_exactly",
        "q2_q3_q4_action": "apply_fixed_M243_recency_spread_composition",
        "supported_operating_quarters": list(SUPPORTED_QUARTERS),
        "operating_day_contract": "minimum forecast timestamp per complete issuance batch",
        "m243_policy_sha256": m243["policy_sha256"],
        "m243_receipt_sha256": M243_RECEIPT_SHA,
        "parent_csv_sha256": parent_receipt["submission_receipt"]["csv_sha256"],
        "analog_target_history": "2022-2023 complete issuance days only",
        "half_life_days": 365.0,
        "spread_reference_quantile": 0.75,
        "q1_analog_exposure": False,
    }
    policy_sha = canonical_sha256(policy)
    candidate_id = f"E0_RECENCY_SPREAD_ANALOG-{policy_sha[:12]}"
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
        raise RuntimeError("M243 build and validation hashes differ")
    supported_fraction = (
        support_diagnostics["supported_issuance_days"]
        / support_diagnostics["total_issuance_days"]
    )
    conditional_proxy = (
        ONLINE_BASELINE
        + M231_Q4_DELTA
        + supported_fraction * M243_Q4_INCREMENT
    )
    receipt = {
        "schema_version": 1,
        "state": "LOCAL_RECENCY_SPREAD_ANALOG_CHALLENGER_BUILT_NOT_UPLOADED",
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
        "development_receipt_path": str(M243_RECEIPT.relative_to(ROOT)),
        "conditional_proxy": {
            "value": conditional_proxy,
            "formula": "0.65971 + M231_Q4_delta + (275/365)*M243_Q4_increment",
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
