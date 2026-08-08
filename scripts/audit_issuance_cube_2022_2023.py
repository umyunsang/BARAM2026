"""Scope-corrected 2022-2023 supplied-NWP issuance metadata audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from collections import Counter
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SOURCE_SPECS = {
    "gfs": {"member": "train/gfs_train.csv", "grids": 9, "rows": 157_680},
    "ldaps": {"member": "train/ldaps_train.csv", "grids": 16, "rows": 280_320},
}
KEY_COLUMNS = (
    "forecast_kst_dtm",
    "data_available_kst_dtm",
    "grid_id",
    "latitude",
    "longitude",
)
START_DAY = pd.Timestamp("2022-01-01")
END_DAY = pd.Timestamp("2023-12-31")
EXPECTED_DAYS = pd.date_range(START_DAY, END_DAY, freq="D")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _parse_variable(column: str) -> dict[str, Any]:
    pieces = column.split("_")
    if len(pieces) >= 3:
        level_type = pieces[0]
        try:
            level_value: int | str = int(pieces[1])
        except ValueError:
            level_value = pieces[1]
        variable = "_".join(pieces[2:])
    else:
        level_type = "unknown"
        level_value = "unknown"
        variable = column
    return {
        "column": column,
        "level_type": level_type,
        "level_value": level_value,
        "variable": variable,
        "temporal_support": "unknown_from_csv",
        "bounds": None,
        "quality_flag": None,
    }


def _source_audit(
    archive: zipfile.ZipFile,
    source: str,
    spec: dict[str, Any],
) -> dict[str, Any]:
    # nrows is the core scope guard: no 2024 operating-day weather row is
    # materialized even though the archive member continues past this prefix.
    with archive.open(spec["member"]) as stream:
        frame = pd.read_csv(
            stream,
            encoding="utf-8-sig",
            nrows=int(spec["rows"]),
        )
    if len(frame) != spec["rows"]:
        raise RuntimeError(f"{source}: frozen prefix row count mismatch")
    missing_keys = sorted(set(KEY_COLUMNS) - set(frame.columns))
    if missing_keys:
        raise RuntimeError(f"{source}: missing columns {missing_keys}")
    value_columns = [column for column in frame if column not in KEY_COLUMNS]
    valid = pd.to_datetime(frame["forecast_kst_dtm"], errors="raise")
    available = pd.to_datetime(frame["data_available_kst_dtm"], errors="raise")
    operating_day = (valid - timedelta(hours=1)).dt.normalize()
    if operating_day.min() != START_DAY or operating_day.max() != END_DAY:
        raise RuntimeError(f"{source}: prefix crosses the declared operating-day scope")
    observed_days = pd.DatetimeIndex(sorted(operating_day.unique()))
    if not observed_days.equals(EXPECTED_DAYS):
        raise RuntimeError(f"{source}: 2022-2023 operating-day coverage mismatch")
    basis = operating_day - timedelta(days=1) + timedelta(hours=14)
    lead = (valid - available).dt.total_seconds() / 3600.0
    slack = (basis - available).dt.total_seconds() / 3600.0
    if not np.allclose(lead, np.round(lead)):
        raise RuntimeError(f"{source}: noninteger availability-to-valid offset")
    if not np.allclose(slack, np.round(slack)):
        raise RuntimeError(f"{source}: noninteger basis slack")

    grids = sorted(int(value) for value in frame["grid_id"].unique())
    if len(grids) != spec["grids"]:
        raise RuntimeError(f"{source}: grid count mismatch")
    duplicate_count = int(frame.duplicated(["forecast_kst_dtm", "grid_id"]).sum())
    issue_rows = Counter(available.dt.strftime("%Y-%m-%dT%H:%M:%S"))
    grouped = pd.DataFrame(
        {
            "issue": available.dt.strftime("%Y-%m-%dT%H:%M:%S"),
            "valid": valid,
            "grid": frame["grid_id"].astype(int),
        }
    ).groupby("issue", sort=True)
    expected_per_issue = 24 * len(grids)
    incomplete_issues: dict[str, Any] = {}
    for issue, subset in grouped:
        row = {
            "rows": len(subset),
            "unique_valid_hours": int(subset["valid"].nunique()),
            "unique_grids": int(subset["grid"].nunique()),
        }
        if (
            row["rows"] != expected_per_issue
            or row["unique_valid_hours"] != 24
            or row["unique_grids"] != len(grids)
        ):
            incomplete_issues[str(issue)] = row
    values = frame[value_columns].apply(pd.to_numeric, errors="coerce")
    finite = np.isfinite(values.to_numpy(dtype=float))
    catalog: list[dict[str, Any]] = []
    for offset, column in enumerate(value_columns):
        nonfinite_count = int((~finite[:, offset]).sum())
        record = _parse_variable(column)
        record["nonfinite_count"] = nonfinite_count
        record["nonfinite_rate"] = float(nonfinite_count / len(frame))
        catalog.append(record)
    return {
        "source": source,
        "member": spec["member"],
        "scope_guard": {
            "reader": "read_csv_nrows",
            "nrows": int(spec["rows"]),
            "operating_day_min": operating_day.min().isoformat(),
            "operating_day_max": operating_day.max().isoformat(),
            "contains_2024_operating_day": bool((operating_day.dt.year == 2024).any()),
        },
        "rows": len(frame),
        "grid_count": len(grids),
        "grid_ids": grids,
        "valid_min": valid.min().isoformat(),
        "valid_max": valid.max().isoformat(),
        "availability_min": available.min().isoformat(),
        "availability_max": available.max().isoformat(),
        "availability_hours": sorted(int(value) for value in available.dt.hour.unique()),
        "availability_to_valid_lead_hours": sorted(
            int(value) for value in np.round(lead).astype(int).unique()
        ),
        "basis_minus_availability_hours": sorted(
            int(value) for value in np.round(slack).astype(int).unique()
        ),
        "operating_day_count": len(observed_days),
        "issue_count": len(issue_rows),
        "expected_rows_per_issue": expected_per_issue,
        "incomplete_issues": incomplete_issues,
        "duplicate_valid_grid_rows": duplicate_count,
        "variables": catalog,
        "total_nonfinite_values": int((~finite).sum()),
        "issue_ids": sorted(issue_rows),
    }


def run(repo: Path, predeclaration: Path, output: Path) -> dict[str, Any]:
    frozen = json.loads(predeclaration.read_text())
    observed = {
        relative: _sha256(repo / relative)
        for relative in frozen["input_bundle"]["files"]
    }
    if observed != frozen["input_bundle"]["files"]:
        raise RuntimeError("N10B input hash mismatch")
    if _canonical_hash(observed) != frozen["input_bundle"]["sha256"]:
        raise RuntimeError("N10B input bundle mismatch")

    archive_path = repo / "inputs/competition/open_wind_236727.zip"
    with zipfile.ZipFile(archive_path) as archive:
        sources = {
            source: _source_audit(archive, source, spec)
            for source, spec in SOURCE_SPECS.items()
        }
    issue_sets = {source: set(record["issue_ids"]) for source, record in sources.items()}
    common_issues = set.intersection(*issue_sets.values())
    source_issue_differences = {
        source: sorted(issues - common_issues) for source, issues in issue_sets.items()
    }
    for record in sources.values():
        record.pop("issue_ids")

    basis_safe = all(
        record["basis_minus_availability_hours"] == [1]
        for record in sources.values()
    )
    scope_safe = all(
        not record["scope_guard"]["contains_2024_operating_day"]
        and record["operating_day_count"] == 730
        for record in sources.values()
    )
    complete = all(
        not record["incomplete_issues"]
        and record["duplicate_valid_grid_rows"] == 0
        and record["total_nonfinite_values"] == 0
        for record in sources.values()
    ) and not any(source_issue_differences.values())
    csv_metadata_fields = {
        "source": "implicit_in_member_name_only",
        "forecast_reference_time": None,
        "available_at": "data_available_kst_dtm",
        "cycle": None,
        "valid_time": "forecast_kst_dtm",
        "lead_time": "derived_from_available_at_not_reference_time",
        "temporal_support": None,
        "bounds": None,
        "variable": "encoded_in_column_name",
        "level_type": "encoded_in_column_name",
        "level_value": "encoded_in_column_name",
        "grid_id": "grid_id",
        "quality_flag": None,
    }
    reference_explicit = csv_metadata_fields["forecast_reference_time"] is not None
    temporal_support_explicit = csv_metadata_fields["temporal_support"] is not None
    if not scope_safe or not basis_safe or not complete:
        verdict = "INVALID"
    elif not reference_explicit or not temporal_support_explicit:
        verdict = "AMBIGUOUS"
    else:
        verdict = "VALIDATED"
    result = {
        "schema_version": 1,
        "node_id": "S17-N10B_ISSUANCE_CUBE_2022_2023_CORRECTED_AUDIT",
        "predeclaration_sha256": _sha256(predeclaration),
        "input_bundle_sha256": frozen["input_bundle"]["sha256"],
        "archive_sha256": observed["inputs/competition/open_wind_236727.zip"],
        "members_read": [spec["member"] for spec in SOURCE_SPECS.values()],
        "scope_method": "frozen nrows prefix; aggregate only asserted operating days 2022-2023",
        "members_not_read": [
            "test/gfs_test.csv",
            "test/ldaps_test.csv",
            "train/train_labels.csv",
            "train/scada_unison_train.csv",
            "train/scada_vestas_train.csv",
        ],
        "csv_metadata_fields": csv_metadata_fields,
        "sources": sources,
        "cross_source": {
            "common_issue_count": len(common_issues),
            "source_issue_differences": source_issue_differences,
            "issue_timestamps_identical": not any(source_issue_differences.values()),
        },
        "canonical_contract_audit": {
            "preserves_valid_time": True,
            "preserves_available_at": True,
            "derives_lead_from_available_at_not_reference": True,
            "preserves_source_as_value_column": False,
            "preserves_forecast_reference_time": False,
            "preserves_cycle": False,
            "preserves_temporal_support_or_bounds": False,
            "preserves_quality_flag": False,
        },
        "gates": {
            "scope_safe_2022_2023_only": scope_safe,
            "basis_safe": basis_safe,
            "source_issue_and_value_complete": complete,
            "forecast_reference_explicit": reference_explicit,
            "temporal_support_explicit": temporal_support_explicit,
        },
        "verdict": verdict,
        "handoff": (
            ["S17-N11_FOUR_MAPPING_TRAINING_WIND_DIAGNOSTIC"]
            if verdict == "AMBIGUOUS"
            else []
        ),
        "actions": {
            "model_fits": 0,
            "target_or_scada_access": False,
            "official_score_calls": 0,
            "lockbox_2024_performance_access": False,
            "2024_operating_day_weather_values_materialized": False,
            "test_data_access": False,
            "dacon_actions": [],
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--predeclaration",
        type=Path,
        default=Path("reports/s17_n10b_issuance_cube_corrected_predeclaration.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/audits/s17_n10b_issuance_cube_2022_2023.json"),
    )
    args = parser.parse_args()
    repo = args.repo.resolve()
    predeclaration = args.predeclaration
    if not predeclaration.is_absolute():
        predeclaration = repo / predeclaration
    output = args.output
    if not output.is_absolute():
        output = repo / output
    result = run(repo, predeclaration, output)
    print(
        json.dumps(
            {
                "verdict": result["verdict"],
                "gates": result["gates"],
                "handoff": result["handoff"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
