"""Build a non-uploaded v2 test challenger with fixed within-issuance smoothing."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from baram.contracts.hashing import canonical_sha256, sha256_file
from baram.submission.build import build_submission
from baram.submission.validate import validate_submission

ROOT = Path(__file__).resolve().parents[2]
OPEN = Path("/Users/um-yunsang/Downloads/open.zip")
BASELINE = Path("/Users/um-yunsang/Downloads/baseline.ipynb")
OPEN_SHA = "920be0c458d820e855bf79dd25723146f52ce1736138aedba5e6bc853f1f720b"
BASELINE_SHA = "712b26f4d2748860c94cff1e0100c23810468c983173f8e9ef8d009fe82df48c"
CACHE = ROOT / "artifacts" / "cache" / OPEN_SHA
PARENT = ROOT / "artifacts" / "submissions" / "E0_DECISION_CONTROL-46980d5f798a.csv"
PARENT_RECEIPT = ROOT / "artifacts" / "submissions" / "v2_final_candidate.receipt.json"
CAPACITIES = {1: 21600.0, 2: 21600.0, 3: 21000.0}
RECIPES = {
    1: ("ramp", 0.025),
    2: ("median3", 0.30),
    3: ("gauss5", 0.325),
}


def _smooth(values: np.ndarray, kind: str) -> np.ndarray:
    if kind == "ramp":
        result = values.copy()
        result[1:-1] = (
            0.25 * values[:-2] + 0.50 * values[1:-1] + 0.25 * values[2:]
        )
        return result
    if kind == "median3":
        padded = np.pad(values, (1, 1), mode="edge")
        return np.asarray(
            [np.median(padded[index : index + 3]) for index in range(len(values))]
        )
    if kind == "gauss5":
        return np.convolve(
            np.pad(values, (2, 2), mode="edge"),
            [0.0625, 0.25, 0.375, 0.25, 0.0625],
            mode="valid",
        )
    raise ValueError(f"unknown smoothing recipe: {kind}")


def _prediction_sha(frame: pd.DataFrame) -> str:
    serializable = frame.copy()
    serializable["forecast_kst_dtm"] = pd.to_datetime(
        serializable["forecast_kst_dtm"]
    ).dt.strftime("%Y-%m-%dT%H:%M:%S")
    return canonical_sha256(serializable.to_dict(orient="records"))


def main() -> None:
    if sha256_file(OPEN) != OPEN_SHA or sha256_file(BASELINE) != BASELINE_SHA:
        raise RuntimeError("immutable competition input hash mismatch")
    parent_receipt = json.loads(PARENT_RECEIPT.read_text(encoding="utf-8"))
    expected_parent_sha = parent_receipt["submission_receipt"]["csv_sha256"]
    if sha256_file(PARENT) != expected_parent_sha:
        raise RuntimeError("immutable v2 submission parent hash mismatch")

    sample = pd.read_parquet(CACHE / "submission_keys.parquet")
    test = pd.read_parquet(
        CACHE / "test_features.parquet",
        columns=[
            "forecast_id",
            "forecast_kst_dtm",
            "data_available_kst_dtm",
            "group_id",
        ],
    )
    if len(test) != 3 * len(sample):
        raise RuntimeError("test feature cardinality changed")
    per_forecast = test.groupby(["forecast_id", "forecast_kst_dtm"], sort=False).agg(
        issuance_count=("data_available_kst_dtm", "nunique"),
        group_count=("group_id", "nunique"),
        issuance=("data_available_kst_dtm", "first"),
    )
    if not (
        per_forecast["issuance_count"].eq(1).all()
        and per_forecast["group_count"].eq(3).all()
    ):
        raise RuntimeError("test forecast/group/issuance topology changed")
    issuance_map = per_forecast.reset_index()[
        ["forecast_id", "forecast_kst_dtm", "issuance"]
    ]
    batch_sizes = issuance_map.groupby("issuance", sort=False).size()
    if len(batch_sizes) != 365 or not batch_sizes.eq(24).all():
        raise RuntimeError("test issuance batches are not 365 complete 24-hour blocks")

    parent = pd.read_csv(PARENT, encoding="utf-8-sig")
    parent["forecast_kst_dtm"] = pd.to_datetime(parent["forecast_kst_dtm"])
    working = parent.merge(
        issuance_map,
        on=["forecast_id", "forecast_kst_dtm"],
        how="left",
        validate="one_to_one",
    )
    if working["issuance"].isna().any():
        raise RuntimeError("submission/test issuance join failed")

    for group_id, (kind, weight) in RECIPES.items():
        column = f"kpx_group_{group_id}"
        normalized = working[column].to_numpy(dtype=float) / CAPACITIES[group_id]
        smoothed = normalized.copy()
        for indices in working.groupby("issuance", sort=False).groups.values():
            positions = np.asarray(list(indices), dtype=int)
            order = np.argsort(working.loc[positions, "forecast_kst_dtm"].to_numpy())
            ordered = positions[order]
            if len(ordered) != 24:
                raise RuntimeError("incomplete issuance reached sequence smoother")
            smoothed[ordered] = _smooth(normalized[ordered], kind)
        working[column] = np.clip(
            ((1.0 - weight) * normalized + weight * smoothed)
            * CAPACITIES[group_id],
            0.0,
            CAPACITIES[group_id],
        )

    wide = working[list(parent.columns)]
    policy = {
        "architecture": "fixed_v2_test_within_issuance_sequence_smoothing",
        "parent_csv_sha256": expected_parent_sha,
        "recipes": {
            str(group_id): {"kind": kind, "smoothing_weight": weight}
            for group_id, (kind, weight) in RECIPES.items()
        },
        "topology": "365_complete_24_hour_issuance_batches",
    }
    policy_sha = canonical_sha256(policy)
    candidate_id = f"E0_M229_SEQUENCE_SMOOTH-{policy_sha[:12]}"
    output = ROOT / "artifacts" / "submissions" / f"{candidate_id}.csv"
    csv_sha = build_submission(sample, wide, output)
    validation = validate_submission(
        output,
        sample,
        candidate_id=candidate_id,
        source_sha256=OPEN_SHA,
        champion_policy_sha256=policy_sha,
        cap_modes={1: "capacity", 2: "capacity", 3: "capacity"},
    )
    if validation.csv_sha256 != csv_sha:
        raise RuntimeError("submission build and validation hashes differ")
    receipt = {
        "schema_version": 1,
        "state": "LOCAL_SEQUENCE_CHALLENGER_BUILT_NOT_UPLOADED",
        "candidate_path": str(output.relative_to(ROOT)),
        "candidate_id": candidate_id,
        "policy": policy,
        "policy_sha256": policy_sha,
        "prediction_sha256": _prediction_sha(wide),
        "submission_receipt": asdict(validation),
        "parent_path": str(PARENT.relative_to(ROOT)),
        "parent_receipt_path": str(PARENT_RECEIPT.relative_to(ROOT)),
        "parent_model_lineage_sha256": parent_receipt["model_lineage_sha256"],
        "validation_scope": (
            "format/topology/reproducibility only; no labels or online score; "
            "different parent lineage prevents local M229 promotion inference"
        ),
        "new_2024_evaluation": False,
        "lockbox_reopened": False,
        "no_external_upload": True,
        "external_actions": [],
    }
    receipt_path = output.with_suffix(".receipt.json")
    receipt_path.write_text(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    print(json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
