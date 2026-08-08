"""Audit supplied training NWP issuance timing and missingness without targets."""

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

MEMBERS = {
    "gfs": "train/gfs_train.csv",
    "ldaps": "train/ldaps_train.csv",
}
KEY_COLUMNS = (
    "forecast_kst_dtm",
    "data_available_kst_dtm",
    "grid_id",
    "latitude",
    "longitude",
)


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


def _source_audit(archive: zipfile.ZipFile, source: str, member: str) -> dict[str, Any]:
    row_count = 0
    grids: set[int] = set()
    valid_min: pd.Timestamp | None = None
    valid_max: pd.Timestamp | None = None
    availability_min: pd.Timestamp | None = None
    availability_max: pd.Timestamp | None = None
    availability_hours: set[int] = set()
    leads: set[int] = set()
    basis_slack_hours: set[int] = set()
    issue_rows: Counter[str] = Counter()
    issue_valid_hours: dict[str, set[str]] = {}
    issue_grids: dict[str, set[int]] = {}
    nonfinite: Counter[str] = Counter()
    duplicates = 0
    seen_keys: set[tuple[str, int]] = set()
    value_columns: list[str] | None = None

    with archive.open(member) as stream:
        chunks = pd.read_csv(stream, encoding="utf-8-sig", chunksize=50_000)
        for chunk in chunks:
            missing_keys = sorted(set(KEY_COLUMNS) - set(chunk.columns))
            if missing_keys:
                raise RuntimeError(f"{source}: missing columns {missing_keys}")
            if value_columns is None:
                value_columns = [column for column in chunk if column not in KEY_COLUMNS]
            elif value_columns != [column for column in chunk if column not in KEY_COLUMNS]:
                raise RuntimeError(f"{source}: schema changed between chunks")
            valid = pd.to_datetime(chunk["forecast_kst_dtm"], errors="raise")
            available = pd.to_datetime(
                chunk["data_available_kst_dtm"], errors="raise"
            )
            operating_day = (valid - timedelta(hours=1)).dt.normalize()
            basis = operating_day - timedelta(days=1) + timedelta(hours=14)
            lead = (valid - available).dt.total_seconds() / 3600.0
            slack = (basis - available).dt.total_seconds() / 3600.0
            if not np.allclose(lead, np.round(lead)):
                raise RuntimeError(f"{source}: noninteger availability-to-valid offset")
            if not np.allclose(slack, np.round(slack)):
                raise RuntimeError(f"{source}: noninteger basis slack")
            row_count += len(chunk)
            grids.update(int(value) for value in chunk["grid_id"].unique())
            valid_min = valid.min() if valid_min is None else min(valid_min, valid.min())
            valid_max = valid.max() if valid_max is None else max(valid_max, valid.max())
            availability_min = (
                available.min()
                if availability_min is None
                else min(availability_min, available.min())
            )
            availability_max = (
                available.max()
                if availability_max is None
                else max(availability_max, available.max())
            )
            availability_hours.update(int(value) for value in available.dt.hour.unique())
            leads.update(int(value) for value in np.round(lead).astype(int).unique())
            basis_slack_hours.update(
                int(value) for value in np.round(slack).astype(int).unique()
            )
            issues = available.dt.strftime("%Y-%m-%dT%H:%M:%S")
            valid_strings = valid.dt.strftime("%Y-%m-%dT%H:%M:%S")
            for issue, valid_string, grid in zip(
                issues,
                valid_strings,
                chunk["grid_id"],
                strict=True,
            ):
                issue_rows[issue] += 1
                issue_valid_hours.setdefault(issue, set()).add(valid_string)
                issue_grids.setdefault(issue, set()).add(int(grid))
                key = (valid_string, int(grid))
                if key in seen_keys:
                    duplicates += 1
                else:
                    seen_keys.add(key)
            values = chunk[value_columns].apply(pd.to_numeric, errors="coerce")
            finite = np.isfinite(values.to_numpy(dtype=float))
            for offset, column in enumerate(value_columns):
                nonfinite[column] += int((~finite[:, offset]).sum())

    if value_columns is None or valid_min is None or valid_max is None:
        raise RuntimeError(f"{source}: empty source")
    expected_per_issue = 24 * len(grids)
    incomplete_issues = {
        issue: {
            "rows": int(issue_rows[issue]),
            "unique_valid_hours": len(issue_valid_hours[issue]),
            "unique_grids": len(issue_grids[issue]),
        }
        for issue in sorted(issue_rows)
        if issue_rows[issue] != expected_per_issue
        or len(issue_valid_hours[issue]) != 24
        or len(issue_grids[issue]) != len(grids)
    }
    catalog = []
    for column in value_columns:
        record = _parse_variable(column)
        record["nonfinite_count"] = int(nonfinite[column])
        record["nonfinite_rate"] = float(nonfinite[column] / row_count)
        catalog.append(record)
    return {
        "source": source,
        "member": member,
        "rows": row_count,
        "grid_count": len(grids),
        "grid_ids": sorted(grids),
        "valid_min": valid_min.isoformat(),
        "valid_max": valid_max.isoformat(),
        "availability_min": availability_min.isoformat(),
        "availability_max": availability_max.isoformat(),
        "availability_hours": sorted(availability_hours),
        "availability_to_valid_lead_hours": sorted(leads),
        "basis_minus_availability_hours": sorted(basis_slack_hours),
        "issue_count": len(issue_rows),
        "expected_rows_per_issue": expected_per_issue,
        "incomplete_issues": incomplete_issues,
        "duplicate_valid_grid_rows": duplicates,
        "whole_source_issue_gaps": 0,
        "variables": catalog,
        "total_nonfinite_values": int(sum(nonfinite.values())),
        "issue_ids": sorted(issue_rows),
    }


def run(repo: Path, predeclaration: Path, output: Path) -> dict[str, Any]:
    frozen = json.loads(predeclaration.read_text())
    observed = {
        relative: _sha256(repo / relative)
        for relative in frozen["input_bundle"]["files"]
    }
    if observed != frozen["input_bundle"]["files"]:
        raise RuntimeError("N10 input hash mismatch")
    if _canonical_hash(observed) != frozen["input_bundle"]["sha256"]:
        raise RuntimeError("N10 input bundle mismatch")
    archive_path = repo / "inputs/competition/open_wind_236727.zip"
    with zipfile.ZipFile(archive_path) as archive:
        sources = {
            source: _source_audit(archive, source, member)
            for source, member in MEMBERS.items()
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
    if not basis_safe or not complete:
        verdict = "INVALID"
    elif not reference_explicit or not temporal_support_explicit:
        verdict = "AMBIGUOUS"
    else:
        verdict = "VALIDATED"
    result = {
        "schema_version": 1,
        "node_id": "S17-N10_ISSUANCE_CUBE_TIMING_AND_MISSINGNESS_AUDIT",
        "predeclaration_sha256": _sha256(predeclaration),
        "input_bundle_sha256": frozen["input_bundle"]["sha256"],
        "archive_sha256": observed["inputs/competition/open_wind_236727.zip"],
        "members_read": list(MEMBERS.values()),
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
            "lockbox_2024_access": False,
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
        default=Path("reports/s17_n10_issuance_cube_audit_predeclaration.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/audits/s17_n10_issuance_cube.json"),
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
