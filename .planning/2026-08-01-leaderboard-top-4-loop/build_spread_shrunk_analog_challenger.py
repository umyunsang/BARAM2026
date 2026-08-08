"""Build the no-upload M240 test candidate with an exact M231 Q1 fallback."""

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
from build_supported_season_analog_challenger import _operating_support
from build_v2_transfer_sequence_challenger import (
    BASELINE,
    BASELINE_SHA,
    OPEN,
    OPEN_SHA,
)
from run_conditional_daily_analog_profile import Recipe, _combine, _feature_sets
from run_spread_shrunk_analog_transfer import (
    _apply_spread_recipe,
    _spread_adjusted_profile,
)
from strict_dev_surface import DEV_CUTOFF, development_surface

from baram.contracts.hashing import canonical_sha256, sha256_file
from baram.submission.build import build_submission
from baram.submission.validate import validate_submission

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "artifacts" / "cache" / OPEN_SHA
OUTPUT = ROOT / "artifacts" / "backtests" / "metric-aligned-probe"
SUBMISSIONS = ROOT / "artifacts" / "submissions"
PARENT = SUBMISSIONS / "E0_V2_SEQUENCE_TRANSFER-a5111cee22bc.csv"
PARENT_RECEIPT = PARENT.with_suffix(".receipt.json")
M240_RECEIPT = OUTPUT / "M240_SPREAD_SHRUNK_ANALOG_Q234.json"
M240_RECEIPT_SHA = "5f62fb05e1233e9c35a3dcab2cf011968142edf598b4b4df9ba76b3fe9c1a62d"
SUPPORTED_QUARTERS = (2, 3, 4)
ONLINE_BASELINE = 0.65971
M231_Q4_DELTA = 0.0025467789388804
M240_Q4_INCREMENT = 0.0028486134251533013


def _verified_parent() -> dict[str, object]:
    receipt = json.loads(PARENT_RECEIPT.read_text(encoding="utf-8"))
    if sha256_file(PARENT) != receipt["submission_receipt"]["csv_sha256"]:
        raise RuntimeError("M231 parent CSV hash mismatch")
    if receipt.get("online_score") is not None:
        raise RuntimeError("M231 parent unexpectedly contains an online score")
    return receipt


def _verified_m240() -> dict[str, object]:
    if sha256_file(M240_RECEIPT) != M240_RECEIPT_SHA:
        raise RuntimeError("M240 development receipt hash mismatch")
    receipt = json.loads(M240_RECEIPT.read_text(encoding="utf-8"))
    if (
        receipt["state"] != "LOCAL_SPREAD_SHRINKAGE_PROMOTED_FOR_TEST_BUILD"
        or not receipt["promotion"]["promoted"]
        or receipt["policy"]["reference_quantile"] != 0.75
    ):
        raise RuntimeError("M240 development promotion contract changed")
    if (
        receipt.get("online_score") is not None
        or receipt.get("new_2024_evaluation")
        or receipt.get("lockbox_reopened")
        or not receipt.get("no_external_upload")
    ):
        raise RuntimeError("M240 external-evidence boundary changed")
    return receipt


def _fallback_and_difference_diagnostics(
    parent: pd.DataFrame,
    output: pd.DataFrame,
    support: pd.DataFrame,
) -> dict[str, object]:
    keys = ["forecast_id", "forecast_kst_dtm"]
    merged = output.merge(
        parent.rename(columns={"prediction_kwh": "parent_prediction_kwh"}),
        on=[*keys, "group_id", "data_available_kst_dtm"],
        validate="one_to_one",
    ).merge(
        support[[*keys, "analog_supported"]],
        on=keys,
        validate="many_to_one",
    )
    active = merged["analog_supported"].to_numpy(dtype=bool)
    delta = (
        merged["prediction_kwh"].to_numpy(dtype=float)
        - merged["parent_prediction_kwh"].to_numpy(dtype=float)
    )
    if not np.array_equal(delta[~active], np.zeros(int((~active).sum()))):
        raise RuntimeError("Q1 fallback differs from M231 parent")
    diagnostics: dict[str, object] = {}
    for group_id in (1, 2, 3):
        group = merged["group_id"].eq(group_id).to_numpy()
        supported = group & active
        diagnostics[str(group_id)] = {
            "supported_mean_absolute_delta_kwh": float(
                np.mean(np.abs(delta[supported]))
            ),
            "supported_max_absolute_delta_kwh": float(
                np.max(np.abs(delta[supported]))
            ),
            "fallback_max_absolute_delta_kwh": float(
                np.max(np.abs(delta[group & ~active]))
            ),
        }
    return diagnostics


def _enforce_exact_q1_fallback(
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
    fallback = ~working["analog_supported"].to_numpy(dtype=bool)
    working.loc[fallback, "prediction_kwh"] = working.loc[
        fallback, "parent_prediction_kwh"
    ].to_numpy(dtype=float)
    return working[output.columns]


def main() -> None:
    if sha256_file(OPEN) != OPEN_SHA or sha256_file(BASELINE) != BASELINE_SHA:
        raise RuntimeError("immutable competition input hash mismatch")
    parent_receipt = _verified_parent()
    m240 = _verified_m240()
    recipes = {
        int(group_id): Recipe(**raw)
        for group_id, raw in m240["policy"]["recipes"].items()
    }

    surface, numeric = development_surface()
    if surface["forecast_kst_dtm"].ge(DEV_CUTOFF).any():
        raise RuntimeError("lockbox row reached M240 test builder")
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
        raise RuntimeError("M240 inference surface crossed the frozen year boundary")

    sample, support, support_diagnostics = _operating_support()
    if tuple(SUPPORTED_QUARTERS) != (2, 3, 4):
        raise RuntimeError("supported-quarter contract changed")
    supported_issuances = np.sort(
        support.loc[support["analog_supported"], "issuance"].unique()
    )
    if len(supported_issuances) != 275:
        raise RuntimeError("M240 supported issuance count changed")
    test_parent = _test_parent_long(test)
    replacements: list[pd.DataFrame] = []
    retrieval: dict[str, object] = {}
    for group_id, recipe in recipes.items():
        profile, diagnostics = _spread_adjusted_profile(
            analog_surface,
            feature_sets,
            group_id,
            supported_issuances,
            recipe,
        )
        replacements.append(
            _apply_spread_recipe(test_parent, profile, group_id, recipe)
        )
        retrieval[str(group_id)] = diagnostics
    test_output = _combine(test_parent, replacements)
    test_output = _enforce_exact_q1_fallback(
        test_parent,
        test_output,
        support,
    )
    difference_diagnostics = _fallback_and_difference_diagnostics(
        test_parent,
        test_output,
        support,
    )
    wide = _wide_submission(test_output)

    policy = {
        "architecture": "q1_m231_q234_m240_train_loo_spread_shrinkage",
        "q1_action": "retain_M231_parent_exactly",
        "q2_q3_q4_action": "apply_fixed_M240_spread_shrinkage",
        "supported_operating_quarters": list(SUPPORTED_QUARTERS),
        "operating_day_contract": "minimum forecast timestamp per complete issuance batch",
        "m240_policy_sha256": m240["policy_sha256"],
        "m240_receipt_sha256": M240_RECEIPT_SHA,
        "parent_csv_sha256": parent_receipt["submission_receipt"]["csv_sha256"],
        "analog_target_history": "2022-2023 complete issuance days only",
        "q1_analog_exposure": False,
    }
    policy_sha = canonical_sha256(policy)
    candidate_id = f"E0_SPREAD_SHRUNK_ANALOG-{policy_sha[:12]}"
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
        raise RuntimeError("M240 build and validation hashes differ")
    supported_fraction = (
        support_diagnostics["supported_issuance_days"]
        / support_diagnostics["total_issuance_days"]
    )
    conditional_proxy = (
        ONLINE_BASELINE
        + M231_Q4_DELTA
        + supported_fraction * M240_Q4_INCREMENT
    )
    receipt = {
        "schema_version": 1,
        "state": "LOCAL_SPREAD_SHRUNK_ANALOG_CHALLENGER_BUILT_NOT_UPLOADED",
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
        "development_receipt_path": str(M240_RECEIPT.relative_to(ROOT)),
        "conditional_proxy": {
            "value": conditional_proxy,
            "formula": "0.65971 + M231_Q4_delta + (275/365)*M240_Q4_increment",
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
