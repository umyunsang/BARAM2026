"""Build the frozen M263 full-history test ensemble without scoring or upload."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from baram.constants import CAPACITIES_KWH, SUBMISSION_COLUMNS
from baram.contracts.hashing import canonical_sha256, sha256_file
from baram.submission.build import build_submission
from baram.submission.validate import validate_submission

ROOT = Path(__file__).resolve().parents[2]
SUBMISSIONS = ROOT / "artifacts" / "submissions"
EVIDENCE = ROOT / "artifacts" / "backtests" / "metric-aligned-probe"
OPEN = Path("/Users/um-yunsang/Downloads/open.zip")
BASELINE = Path("/Users/um-yunsang/Downloads/baseline.ipynb")
OPEN_SHA = "920be0c458d820e855bf79dd25723146f52ce1736138aedba5e6bc853f1f720b"
BASELINE_SHA = "712b26f4d2748860c94cff1e0100c23810468c983173f8e9ef8d009fe82df48c"

CANDIDATE_ID = "M263_FULL_HISTORY_INDEPENDENT_LINEAGE_ENSEMBLE"
CANDIDATE_PATH = SUBMISSIONS / "submission_M263.csv"
RECEIPT_PATH = SUBMISSIONS / "submission_M263.receipt.json"
M252_PATH = SUBMISSIONS / "submission_M252.csv"
M252_SHA = "06fe135a22b2eff0b66303b87090f43a2b3c99ec688f4eb84cadac0c874e0730"
M252_RECEIPT = SUBMISSIONS / "E0_FULL_HISTORY_Q1_GROUP12_ANALOG-1741a964e30b.receipt.json"
M252_RECEIPT_SHA = (
    "ac783f2ec153e84dc4db62864f791f502559e4aa75e51bd375a859ac8ddf75f2"
)
M261_PATH = SUBMISSIONS / "submission_M261.csv"
M261_SHA = "fb937f77dfe501b4d7f8e52da098f07f79b1924dace347d3020da405b660773b"
M261_RECEIPT = SUBMISSIONS / "submission_M261.receipt.json"
M261_RECEIPT_SHA = (
    "0bbf2aae103ad1115ed5809d5747e965732b827d01ee4c381352eaec0622961c"
)
M263_RECEIPT = EVIDENCE / "M263_INDEPENDENT_LINEAGE_CONVEX_ENSEMBLE.json"
M263_RECEIPT_SHA = (
    "9169edab2b983dc5c47907295b8524ffba9c01615eda87d8eb288a742848e47e"
)
M263_OOF_SHA = "5b09c1b55621766e09bbd39a25d43f6eaf36d14621577604d2474d3bea2348a8"
SAMPLE_KEYS = (
    ROOT
    / "artifacts"
    / "cache"
    / OPEN_SHA
    / "submission_keys.parquet"
)
SAMPLE_KEYS_SHA = (
    "f675805cb030ce6d401c735194bc48965cd398a1abd0b4bea21ec546a750ddb7"
)
M244_MASSES = {1: 0.0, 2: 0.10, 3: 0.0}
PREDICTION_COLUMNS = ["kpx_group_1", "kpx_group_2", "kpx_group_3"]
KEYS = ["forecast_id", "forecast_kst_dtm"]


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_sources() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    expected_hashes = {
        OPEN: OPEN_SHA,
        BASELINE: BASELINE_SHA,
        M252_PATH: M252_SHA,
        M252_RECEIPT: M252_RECEIPT_SHA,
        M261_PATH: M261_SHA,
        M261_RECEIPT: M261_RECEIPT_SHA,
        M263_RECEIPT: M263_RECEIPT_SHA,
        SAMPLE_KEYS: SAMPLE_KEYS_SHA,
    }
    for path, expected in expected_hashes.items():
        if sha256_file(path) != expected:
            raise RuntimeError(f"M263 source hash mismatch: {path}")
    m252 = _json(M252_RECEIPT)
    m261 = _json(M261_RECEIPT)
    m263 = _json(M263_RECEIPT)
    if (
        m252["state"] != "LOCAL_FROZEN_POLICY_FULL_HISTORY_FINAL_FIT_BUILT_NOT_UPLOADED"
        or m252["submission_receipt"]["csv_sha256"] != M252_SHA
        or m252.get("new_2024_evaluation")
        or m252.get("lockbox_reopened")
        or m252.get("external_actions")
    ):
        raise RuntimeError("M252 frozen deployment evidence changed")
    if (
        m261["candidate_id"] != "M261_FULL_HISTORY_STRICT_TEMPORAL_TOP100"
        or m261["state"]
        != "LOCAL_FROZEN_M107_POLICY_FULL_HISTORY_FINAL_FIT_BUILT_NOT_UPLOADED"
        or m261["submission_receipt"]["csv_sha256"] != M261_SHA
        or m261.get("new_2024_evaluation")
        or m261.get("lockbox_reopened")
        or m261.get("external_actions")
    ):
        raise RuntimeError("M261 frozen deployment evidence changed")
    selected = {int(group): float(mass) for group, mass in m263["selected_m244_masses"].items()}
    if (
        m263["candidate_id"] != "M263_INDEPENDENT_LINEAGE_CONVEX_ENSEMBLE"
        or m263["state"]
        != "LOCAL_INDEPENDENT_LINEAGE_ENSEMBLE_PROMOTED_NO_TEST_BUILD"
        or not m263["promotion"]["promoted"]
        or m263["prediction_sha256"] != M263_OOF_SHA
        or selected != M244_MASSES
        or m263.get("new_2024_evaluation")
        or m263.get("lockbox_reopened")
        or m263.get("external_actions")
    ):
        raise RuntimeError("M263 promoted OOF evidence changed")
    return m252, m261, m263


def _source_frame(path: Path, name: str) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    if tuple(frame.columns) != SUBMISSION_COLUMNS or len(frame) != 8760:
        raise RuntimeError(f"{name} submission schema changed")
    frame["forecast_kst_dtm"] = pd.to_datetime(
        frame["forecast_kst_dtm"], errors="raise"
    )
    if frame[KEYS].duplicated().any():
        raise RuntimeError(f"{name} submission keys are not unique")
    values = frame[PREDICTION_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0.0).any():
        raise RuntimeError(f"{name} submission predictions are invalid")
    return frame


def main() -> None:
    _validate_sources()
    m252 = _source_frame(M252_PATH, "M252")
    m261 = _source_frame(M261_PATH, "M261")
    if not m252[KEYS].equals(m261[KEYS]):
        raise RuntimeError("M252 and M261 test keys/order changed")

    wide = m261.copy()
    for group_id, mass in M244_MASSES.items():
        column = f"kpx_group_{group_id}"
        wide[column] = (
            (1.0 - mass) * m261[column].to_numpy(dtype=float)
            + mass * m252[column].to_numpy(dtype=float)
        )
    for group_id in (1, 3):
        column = f"kpx_group_{group_id}"
        if not np.array_equal(
            wide[column].to_numpy(dtype=float),
            m261[column].to_numpy(dtype=float),
        ):
            raise RuntimeError(f"M263 group {group_id} changed from exact M261")
    expected_group2 = (
        0.90 * m261["kpx_group_2"].to_numpy(dtype=float)
        + 0.10 * m252["kpx_group_2"].to_numpy(dtype=float)
    )
    if not np.array_equal(
        wide["kpx_group_2"].to_numpy(dtype=float), expected_group2
    ):
        raise RuntimeError("M263 group 2 blend algebra changed")

    sample = pd.read_parquet(SAMPLE_KEYS)
    policy = {
        "architecture": "frozen_full_history_independent_lineage_convex_ensemble",
        "selection_source": "M263_strict_Q2_Q3_stability_with_Q4_transfer",
        "m107_deployment": "M261_FULL_HISTORY_STRICT_TEMPORAL_TOP100",
        "m244_deployment": "M252_full_history_rare_event_analog_lineage",
        "m244_masses": {str(group): mass for group, mass in M244_MASSES.items()},
        "group_actions": {
            "1": "exact_M261",
            "2": "0.90_M261_plus_0.10_M252",
            "3": "exact_M261",
        },
        "selection_after_final_fit": False,
        "parameter_search_after_final_fit": False,
        "metric_after_final_fit": False,
        "online_feedback_used_for_weight_selection": False,
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
        raise RuntimeError("M263 build and validation hashes differ")

    roundtrip = pd.read_csv(CANDIDATE_PATH, encoding="utf-8-sig")
    roundtrip_group2_error = np.abs(
        roundtrip["kpx_group_2"].to_numpy(dtype=float) - expected_group2
    )
    roundtrip_group2_max_error = float(roundtrip_group2_error.max())
    if (
        not np.array_equal(
            roundtrip["kpx_group_1"].to_numpy(dtype=float),
            m261["kpx_group_1"].to_numpy(dtype=float),
        )
        or not np.array_equal(
            roundtrip["kpx_group_3"].to_numpy(dtype=float),
            m261["kpx_group_3"].to_numpy(dtype=float),
        )
        or roundtrip_group2_max_error > 1e-10
    ):
        raise RuntimeError("M263 CSV round-trip blend contract changed")

    values = wide[PREDICTION_COLUMNS].to_numpy(dtype=float)
    receipt = {
        "schema_version": 1,
        "state": "LOCAL_FROZEN_M263_FULL_HISTORY_ENSEMBLE_BUILT_NOT_UPLOADED",
        "candidate_id": CANDIDATE_ID,
        "candidate_path": str(CANDIDATE_PATH.relative_to(ROOT)),
        "policy": policy,
        "policy_sha256": policy_sha,
        "submission_receipt": asdict(validation),
        "prediction_diagnostics": {
            "row_count": len(wide),
            "finite": bool(np.isfinite(values).all()),
            "minimum_kwh": float(values.min()),
            "maximum_kwh": float(values.max()),
            "negative_rows": int((values < 0.0).any(axis=1).sum()),
            "capacity_exceed_rows": {
                str(group): int(
                    (
                        wide[f"kpx_group_{group}"].to_numpy(dtype=float)
                        > CAPACITIES_KWH[group]
                    ).sum()
                )
                for group in CAPACITIES_KWH
            },
            "group1_exact_m261": True,
            "group3_exact_m261": True,
            "group2_mean_absolute_change_from_m261_kwh": float(
                np.mean(
                    np.abs(
                        wide["kpx_group_2"].to_numpy(dtype=float)
                        - m261["kpx_group_2"].to_numpy(dtype=float)
                    )
                )
            ),
            "csv_roundtrip_group2_formula_max_abs_kwh": roundtrip_group2_max_error,
            "csv_roundtrip_formula_tolerance_kwh": 1e-10,
        },
        "inputs": {
            "open_zip_sha256": OPEN_SHA,
            "baseline_ipynb_sha256": BASELINE_SHA,
            "sample_keys_sha256": SAMPLE_KEYS_SHA,
            "m252_csv_sha256": M252_SHA,
            "m252_receipt_sha256": M252_RECEIPT_SHA,
            "m261_csv_sha256": M261_SHA,
            "m261_receipt_sha256": M261_RECEIPT_SHA,
            "m263_oof_prediction_sha256": M263_OOF_SHA,
            "m263_oof_receipt_sha256": M263_RECEIPT_SHA,
            "builder_code_sha256": sha256_file(Path(__file__)),
        },
        "evaluation_contract": {
            "score_function_calls": 0,
            "direct_score_calls": [],
            "local_score": None,
            "online_score": None,
            "metrics_computed_on_2024": False,
            "2024_slice_or_comparison_created": False,
            "selection_after_final_fit": False,
            "parameter_search_after_final_fit": False,
            "target_total_strictly_greater_than": 0.66000,
            "target_status": "UNVERIFIED_REQUIRES_EXTERNAL_DACON_RESULT",
        },
        "online_score": None,
        "no_external_upload": True,
        "new_2024_evaluation": False,
        "lockbox_reopened": False,
        "external_actions": [],
    }
    RECEIPT_PATH.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    print(json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
