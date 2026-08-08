"""Freeze a Q3-selected v2 sequence transform, check Q4 transfer, and build test CSV."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from strict_dev_surface import DEV_CUTOFF, development_surface

from baram.contracts.hashing import canonical_sha256, sha256_file
from baram.evaluation.official import evaluate_official
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
OOF = (
    ROOT
    / "artifacts"
    / "backtests"
    / "decision-v2"
    / "baram-v2-20260801-01"
    / "U6_GROUP_WIND-oof.parquet"
)
CAPACITIES = {1: 21600.0, 2: 21600.0, 3: 21000.0}
KINDS = ("ramp", "median3", "mean3", "gauss5", "mean5", "median5")
WEIGHTS = tuple(float(value) for value in np.arange(0.025, 0.501, 0.025))
EXPECTED_SELECTIONS = {
    1: ("mean5", 0.47500000000000003),
    2: ("median5", 0.5),
    3: ("median5", 0.325),
}
METRIC_COLUMNS = [
    "forecast_id",
    "forecast_kst_dtm",
    "group_id",
    "actual_kwh",
    "prediction_kwh",
]


def _smooth(values: np.ndarray, kind: str) -> np.ndarray:
    if kind == "ramp":
        return np.convolve(
            np.pad(values, (1, 1), mode="edge"),
            [0.25, 0.50, 0.25],
            mode="valid",
        )
    if kind == "median3":
        padded = np.pad(values, (1, 1), mode="edge")
        return np.asarray(
            [np.median(padded[index : index + 3]) for index in range(len(values))]
        )
    if kind == "mean3":
        return np.convolve(
            np.pad(values, (1, 1), mode="edge"),
            [1.0 / 3.0] * 3,
            mode="valid",
        )
    if kind == "gauss5":
        return np.convolve(
            np.pad(values, (2, 2), mode="edge"),
            [0.0625, 0.25, 0.375, 0.25, 0.0625],
            mode="valid",
        )
    if kind == "mean5":
        return np.convolve(
            np.pad(values, (2, 2), mode="edge"),
            [0.20] * 5,
            mode="valid",
        )
    if kind == "median5":
        padded = np.pad(values, (2, 2), mode="edge")
        return np.asarray(
            [np.median(padded[index : index + 5]) for index in range(len(values))]
        )
    raise ValueError(f"unknown smoothing recipe: {kind}")


def _apply_long(
    frame: pd.DataFrame,
    selections: dict[int, tuple[str, float]],
) -> pd.DataFrame:
    output = frame.reset_index(drop=True).copy()
    for group_id, (kind, weight) in selections.items():
        mask = output["group_id"].eq(group_id).to_numpy()
        group_positions = np.flatnonzero(mask)
        capacity = CAPACITIES[group_id]
        normalized = output.loc[mask, "prediction_kwh"].to_numpy(dtype=float) / capacity
        smoothed = normalized.copy()
        group = output.loc[mask].reset_index(drop=True)
        for indices in group.groupby("data_available_kst_dtm", sort=False).groups.values():
            positions = np.asarray(list(indices), dtype=int)
            order = np.argsort(group.loc[positions, "forecast_kst_dtm"].to_numpy())
            ordered = positions[order]
            smoothed[ordered] = _smooth(normalized[ordered], kind)
        transformed = (1.0 - weight) * normalized + weight * smoothed
        output.loc[group_positions, "prediction_kwh"] = np.clip(
            transformed * capacity,
            0.0,
            capacity,
        )
    return output


def _group_score(frame: pd.DataFrame, group_id: int) -> dict[str, float]:
    group = frame.loc[frame["group_id"].eq(group_id)]
    capacity = CAPACITIES[group_id]
    eligible = group["actual_kwh"].to_numpy(dtype=float) >= 0.10 * capacity
    actual = group.loc[eligible, "actual_kwh"].to_numpy(dtype=float)
    prediction = group.loc[eligible, "prediction_kwh"].to_numpy(dtype=float)
    error = np.abs(prediction - actual) / capacity
    units = np.select([error <= 0.06, error <= 0.08], [4.0, 3.0], default=0.0)
    one_minus_nmae = 1.0 - float(error.mean())
    ficr = float(np.sum(actual * units) / np.sum(actual * 4.0))
    return {
        "total": 0.5 * (one_minus_nmae + ficr),
        "one_minus_nmae": one_minus_nmae,
        "ficr": ficr,
    }


def _score(frame: pd.DataFrame) -> dict[str, float]:
    result = evaluate_official(frame[METRIC_COLUMNS], CAPACITIES)
    return {
        "total": result.total,
        "one_minus_nmae": result.one_minus_nmae,
        "ficr": result.ficr,
    }


def _score_repeated_rows(frame: pd.DataFrame) -> float:
    """Evaluate official algebra when bootstrap sampling repeats issuance keys."""
    components = [_group_score(frame, group_id) for group_id in CAPACITIES]
    return 0.5 * (
        float(np.mean([item["one_minus_nmae"] for item in components]))
        + float(np.mean([item["ficr"] for item in components]))
    )


def _paired_issuance_bootstrap(
    parent: pd.DataFrame,
    transformed: pd.DataFrame,
    replicates: int = 2_000,
) -> dict[str, float | int | str]:
    issuances = parent["data_available_kst_dtm"].drop_duplicates().to_numpy()
    positions = {
        issuance: np.flatnonzero(
            parent["data_available_kst_dtm"].eq(issuance).to_numpy()
        )
        for issuance in issuances
    }
    random = np.random.default_rng(20260803)
    deltas = np.empty(replicates, dtype=float)
    for index in range(replicates):
        sampled = random.choice(issuances, size=len(issuances), replace=True)
        rows = np.concatenate([positions[issuance] for issuance in sampled])
        deltas[index] = _score_repeated_rows(
            transformed.iloc[rows]
        ) - _score_repeated_rows(parent.iloc[rows])
    return {
        "unit": "issuance_day",
        "seed": 20260803,
        "replicates": replicates,
        "mean": float(deltas.mean()),
        "std": float(deltas.std(ddof=1)),
        "q025": float(np.quantile(deltas, 0.025)),
        "q05": float(np.quantile(deltas, 0.05)),
        "median": float(np.quantile(deltas, 0.50)),
        "q95": float(np.quantile(deltas, 0.95)),
        "q975": float(np.quantile(deltas, 0.975)),
        "positive_fraction": float(np.mean(deltas > 0.0)),
    }


def _select_q3(q3: pd.DataFrame) -> tuple[dict[int, tuple[str, float]], dict[str, object]]:
    selections: dict[int, tuple[str, float]] = {}
    diagnostics: dict[str, object] = {}
    for group_id in CAPACITIES:
        group = q3.loc[q3["group_id"].eq(group_id)].reset_index(drop=True)
        baseline = _group_score(group, group_id)
        best_total = baseline["total"]
        best = ("identity", 0.0)
        for kind in KINDS:
            for weight in WEIGHTS:
                trial = _apply_long(group, {group_id: (kind, weight)})
                total = _group_score(trial, group_id)["total"]
                if total > best_total:
                    best_total = total
                    best = (kind, weight)
        if best[0] == "identity":
            raise RuntimeError(f"group {group_id} did not select a sequence recipe")
        selections[group_id] = best
        selected = _apply_long(group, {group_id: best})
        diagnostics[str(group_id)] = {
            "baseline": baseline,
            "selected": _group_score(selected, group_id),
            "kind": best[0],
            "weight": best[1],
        }
    if selections != EXPECTED_SELECTIONS:
        raise RuntimeError(f"Q3 sequence selections changed: {selections}")
    return selections, diagnostics


def _test_topology() -> tuple[pd.DataFrame, pd.DataFrame]:
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
    grouped = test.groupby(["forecast_id", "forecast_kst_dtm"], sort=False).agg(
        issuance_count=("data_available_kst_dtm", "nunique"),
        group_count=("group_id", "nunique"),
        issuance=("data_available_kst_dtm", "first"),
    )
    if not (
        grouped["issuance_count"].eq(1).all()
        and grouped["group_count"].eq(3).all()
    ):
        raise RuntimeError("test forecast/group/issuance topology changed")
    issuance_map = grouped.reset_index()[
        ["forecast_id", "forecast_kst_dtm", "issuance"]
    ]
    sizes = issuance_map.groupby("issuance", sort=False).size()
    if len(sizes) != 365 or not sizes.eq(24).all():
        raise RuntimeError("test issuance batches are not 365 complete 24-hour blocks")
    return sample, issuance_map


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
    parent_sha = parent_receipt["submission_receipt"]["csv_sha256"]
    if sha256_file(PARENT) != parent_sha:
        raise RuntimeError("immutable v2 submission parent hash mismatch")

    surface, _ = development_surface()
    if surface["forecast_kst_dtm"].ge(DEV_CUTOFF).any():
        raise RuntimeError("lockbox row reached v2 sequence-transfer builder")
    metadata = surface[
        ["forecast_id", "forecast_kst_dtm", "group_id", "data_available_kst_dtm"]
    ]
    oof = pd.read_parquet(OOF).merge(
        metadata,
        on=["forecast_id", "forecast_kst_dtm", "group_id"],
        validate="one_to_one",
    )
    q3 = oof.loc[oof["fold_id"].eq("dev-2023-Q3")].reset_index(drop=True)
    q4 = oof.loc[oof["fold_id"].eq("dev-2023-Q4")].reset_index(drop=True)
    selections, q3_selection = _select_q3(q3)
    q3_output = _apply_long(q3, selections)
    q4_output = _apply_long(q4, selections)
    transfer = {
        "selection_fold": "dev-2023-Q3",
        "check_fold": "dev-2023-Q4",
        "q3_selection": q3_selection,
        "q3_parent_score": _score(q3),
        "q3_transformed_score": _score(q3_output),
        "q4_parent_score": _score(q4),
        "q4_transformed_score": _score(q4_output),
        "q4_total_delta": _score(q4_output)["total"] - _score(q4)["total"],
        "q4_group_parent": {
            str(group_id): _group_score(q4, group_id) for group_id in CAPACITIES
        },
        "q4_group_transformed": {
            str(group_id): _group_score(q4_output, group_id)
            for group_id in CAPACITIES
        },
        "q4_monthly": {
            str(month): {
                "parent": _score(
                    q4.loc[q4["forecast_kst_dtm"].dt.month.eq(month)]
                ),
                "transformed": _score(
                    q4_output.loc[q4_output["forecast_kst_dtm"].dt.month.eq(month)]
                ),
                "total_delta": _score(
                    q4_output.loc[q4_output["forecast_kst_dtm"].dt.month.eq(month)]
                )["total"]
                - _score(q4.loc[q4["forecast_kst_dtm"].dt.month.eq(month)])["total"],
            }
            for month in (10, 11, 12)
        },
        "paired_bootstrap": _paired_issuance_bootstrap(q4, q4_output),
        "scope": (
            "incremental transform transfer check only; v2 parent selection had seen "
            "both folds, but transform candidates and weights were selected on Q3 only"
        ),
    }
    if transfer["q4_total_delta"] <= 0.0:
        raise RuntimeError("Q3-selected sequence transform did not transfer to Q4")

    sample, issuance_map = _test_topology()
    parent = pd.read_csv(PARENT, encoding="utf-8-sig")
    parent["forecast_kst_dtm"] = pd.to_datetime(parent["forecast_kst_dtm"])
    wide = parent.merge(
        issuance_map,
        on=["forecast_id", "forecast_kst_dtm"],
        how="left",
        validate="one_to_one",
    )
    if wide["issuance"].isna().any():
        raise RuntimeError("submission/test issuance join failed")
    for group_id, (kind, weight) in selections.items():
        column = f"kpx_group_{group_id}"
        capacity = CAPACITIES[group_id]
        normalized = wide[column].to_numpy(dtype=float) / capacity
        smoothed = normalized.copy()
        for indices in wide.groupby("issuance", sort=False).groups.values():
            positions = np.asarray(list(indices), dtype=int)
            order = np.argsort(wide.loc[positions, "forecast_kst_dtm"].to_numpy())
            ordered = positions[order]
            smoothed[ordered] = _smooth(normalized[ordered], kind)
        wide[column] = np.clip(
            ((1.0 - weight) * normalized + weight * smoothed) * capacity,
            0.0,
            capacity,
        )
    wide = wide[list(parent.columns)]

    policy = {
        "architecture": "q3_selected_q4_positive_v2_sequence_transfer",
        "candidate_kinds": list(KINDS),
        "candidate_weights": list(WEIGHTS),
        "parent_csv_sha256": parent_sha,
        "parent_oof_sha256": sha256_file(OOF),
        "recipes": {
            str(group_id): {"kind": kind, "smoothing_weight": weight}
            for group_id, (kind, weight) in selections.items()
        },
        "topology": "365_complete_24_hour_test_issuance_batches",
    }
    policy_sha = canonical_sha256(policy)
    candidate_id = f"E0_V2_SEQUENCE_TRANSFER-{policy_sha[:12]}"
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
        "state": "LOCAL_V2_SEQUENCE_TRANSFER_CHALLENGER_BUILT_NOT_UPLOADED",
        "candidate_path": str(output.relative_to(ROOT)),
        "candidate_id": candidate_id,
        "policy": policy,
        "policy_sha256": policy_sha,
        "prediction_sha256": _prediction_sha(wide),
        "submission_receipt": asdict(validation),
        "transfer_check": transfer,
        "parent_path": str(PARENT.relative_to(ROOT)),
        "parent_receipt_path": str(PARENT_RECEIPT.relative_to(ROOT)),
        "parent_model_lineage_sha256": parent_receipt["model_lineage_sha256"],
        "online_score": None,
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
