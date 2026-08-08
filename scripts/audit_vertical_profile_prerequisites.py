"""Audit static prerequisites for the R2 vertical-profile re-entry."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

WEATHER_MEMBERS = {
    "gfs": "train/gfs_train.csv",
    "ldaps": "train/ldaps_train.csv",
}
COMPARABLE_VECTORS = {
    "gfs": {
        10: ("heightAboveGround_10_10u", "heightAboveGround_10_10v"),
        80: ("heightAboveGround_80_u", "heightAboveGround_80_v"),
        100: ("heightAboveGround_100_100u", "heightAboveGround_100_100v"),
    },
    "ldaps": {
        5: ("heightAboveGround_5_XBLWS", "heightAboveGround_5_YBLWS"),
        10: ("heightAboveGround_10_10u", "heightAboveGround_10_10v"),
    },
}
EXTREMA_NOT_LEVEL_MEANS = {
    "ldaps_50_max": (
        "heightAboveGround_50_50MUmax",
        "heightAboveGround_50_50MVmax",
    ),
    "ldaps_50_min": (
        "heightAboveGround_50_50MUmin",
        "heightAboveGround_50_50MVmin",
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _bytes_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _header(archive: zipfile.ZipFile, member: str) -> list[str]:
    with archive.open(member) as stream:
        return stream.readline().decode("utf-8-sig").rstrip("\r\n").split(",")


def _normalise_static_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float):
        return float(value)
    if isinstance(value, int):
        return int(value)
    return str(value)


def _static_workbook(data: bytes) -> dict[str, Any]:
    sheets = pd.read_excel(
        io.BytesIO(data),
        sheet_name=None,
        header=None,
        engine="openpyxl",
    )
    report: dict[str, Any] = {}
    for sheet_name, raw in sheets.items():
        header_candidates = [
            index
            for index, row in raw.iterrows()
            if any(
                re.search(r"Hub Height|허브", str(value), re.I)
                for value in row
                if not pd.isna(value)
            )
        ]
        if len(header_candidates) != 1:
            raise RuntimeError(f"{sheet_name}: static workbook header is ambiguous")
        header_index = header_candidates[0]
        keep_positions = [
            position
            for position, value in enumerate(raw.iloc[header_index])
            if not pd.isna(value)
        ]
        columns = [str(raw.iloc[header_index, position]) for position in keep_positions]
        frame = raw.iloc[header_index + 1 :, keep_positions].copy()
        frame.columns = columns
        frame = frame.dropna(how="all").reset_index(drop=True)
        records = [
            {str(column): _normalise_static_value(value) for column, value in row.items()}
            for row in frame.to_dict(orient="records")
        ]
        report[str(sheet_name)] = {
            "header_row_zero_based": int(header_index),
            "rows": len(frame),
            "columns": columns,
            "records": records,
        }
    return report


def _metadata_mentions(text: str, workbook: dict[str, Any]) -> dict[str, Any]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    hub_lines = [line for line in lines if re.search(r"hub|허브|117\s*m", line, re.I)]
    elevation_lines = [
        line
        for line in lines
        if re.search(r"elevation|altitude|orograph|표고|고도", line, re.I)
    ]
    rotor_lines = [
        line for line in lines if re.search(r"rotor|diameter|로터|직경|V126|U136", line, re.I)
    ]
    column_candidates = {
        "hub": sorted(
            {
                column
                for sheet in workbook.values()
                for column in sheet["columns"]
                if re.search(r"hub|허브", column, re.I)
            }
        ),
        "elevation": sorted(
            {
                column
                for sheet in workbook.values()
                for column in sheet["columns"]
                if re.search(r"elevation|altitude|표고|고도", column, re.I)
            }
        ),
        "rotor": sorted(
            {
                column
                for sheet in workbook.values()
                for column in sheet["columns"]
                if re.search(r"rotor|diameter|로터|직경", column, re.I)
            }
        ),
    }
    hub_values = sorted(
        {
            float(record[column])
            for sheet in workbook.values()
            for record in sheet["records"]
            for column in column_candidates["hub"]
            if record.get(column) is not None
        }
    )
    rotor_values = sorted(
        {
            float(record[column])
            for sheet in workbook.values()
            for record in sheet["records"]
            for column in column_candidates["rotor"]
            if record.get(column) is not None
        }
    )
    return {
        "hub_117_documented": hub_values == [117.0],
        "hub_height_values_m": hub_values,
        "rotor_diameter_values_m": rotor_values,
        "hub_lines": hub_lines,
        "elevation_lines": elevation_lines,
        "rotor_lines": rotor_lines,
        "static_column_candidates": column_candidates,
    }


def run(repo: Path, predeclaration: Path, output: Path) -> dict[str, Any]:
    frozen = json.loads(predeclaration.read_text())
    observed = {
        relative: _sha256(repo / relative)
        for relative in frozen["input_bundle"]["files"]
    }
    if observed != frozen["input_bundle"]["files"]:
        raise RuntimeError("N12 input hash mismatch")
    if _canonical_hash(observed) != frozen["input_bundle"]["sha256"]:
        raise RuntimeError("N12 input bundle mismatch")

    archive_path = repo / "inputs/competition/open_wind_236727.zip"
    with zipfile.ZipFile(archive_path) as archive:
        description_bytes = archive.read("data_description.md")
        workbook_bytes = archive.read("info.xlsx")
        headers = {
            source: _header(archive, member)
            for source, member in WEATHER_MEMBERS.items()
        }
    description = description_bytes.decode("utf-8")
    workbook = _static_workbook(workbook_bytes)
    metadata = _metadata_mentions(description, workbook)

    source_vectors: dict[str, Any] = {}
    for source, vectors in COMPARABLE_VECTORS.items():
        available = {
            height: list(pair)
            for height, pair in vectors.items()
            if set(pair).issubset(headers[source])
        }
        source_vectors[source] = {
            "comparable_agl_vectors": available,
            "comparable_heights_m": sorted(available),
            "brackets_117m": bool(
                available
                and min(available) <= 117 <= max(available)
                and min(available) < 117 < max(available)
            ),
            "highest_comparable_height_m": max(available) if available else None,
        }
    extrema_presence = {
        name: set(pair).issubset(headers["ldaps"])
        for name, pair in EXTREMA_NOT_LEVEL_MEANS.items()
    }

    prepare = json.loads((repo / "artifacts/manifests/prepare.json").read_text())
    feature_names = prepare["feature_names"]
    raw_vertical_features = sorted(
        feature
        for feature in feature_names
        if any(
            token in feature
            for token in (
                "heightAboveGround_5_",
                "heightAboveGround_10_",
                "heightAboveGround_50_",
                "heightAboveGround_80_",
                "heightAboveGround_100_",
            )
        )
    )
    vertical_pca_features = sorted(
        feature
        for feature in feature_names
        if "pca" in feature.lower() and "vertical" in feature.lower()
    )
    prior_midpoint = json.loads(
        (repo / "reports/m271_cycle6_wind50mid_receipt.json").read_text()
    )
    prior_status = {
        "wind50_midpoint_result": prior_midpoint["result"]["predeclared_check"][
            "verdict"
        ],
        "wind50_midpoint_score_bearing": False,
        "strict_chronology_vertical_pca_receipt_found_in_frozen_inputs": False,
        "research_classification": (
            "S13 marks grid PCA/REWS as legacy-rejected, but no exact S17 strict "
            "chronology score-bearing receipt is in this audit bundle"
        ),
    }

    fixed_height_ready = bool(
        metadata["hub_117_documented"]
        and all(record["brackets_117m"] for record in source_vectors.values())
    )
    pca_ready = bool(
        all(len(record["comparable_heights_m"]) >= 2 for record in source_vectors.values())
        and not prior_status["strict_chronology_vertical_pca_receipt_found_in_frozen_inputs"]
    )
    if fixed_height_ready and pca_ready:
        verdict = "READY_FIXED_AND_PCA"
        handoff = ["S17-N13_R2_FIXED_HEIGHT_PLUS_PCA_STRICT_PREQUENTIAL"]
    elif pca_ready:
        verdict = "READY_PCA_ONLY"
        handoff = ["S17-N13_R2_TWO_COMPONENT_VERTICAL_PCA_STRICT_PREQUENTIAL"]
    else:
        verdict = "NOT_READY_PHYSICAL_SEMANTICS_OR_DUPLICATE"
        handoff = ["S17-N13_R3_HEIGHT_ONLY_TERRAIN_PREREQUISITE_AUDIT"]

    result = {
        "schema_version": 1,
        "node_id": "S17-N12_VERTICAL_PROFILE_TERRAIN_PREREQUISITE_AUDIT",
        "predeclaration_sha256": _sha256(predeclaration),
        "input_bundle_sha256": frozen["input_bundle"]["sha256"],
        "archive_static_members": {
            "data_description.md": {
                "sha256": _bytes_hash(description_bytes),
                "bytes": len(description_bytes),
            },
            "info.xlsx": {
                "sha256": _bytes_hash(workbook_bytes),
                "bytes": len(workbook_bytes),
            },
        },
        "weather_rows_read": 0,
        "weather_headers": headers,
        "static_workbook": workbook,
        "metadata": metadata,
        "source_vectors": source_vectors,
        "ldaps_50_extrema_present_but_not_level_means": extrema_presence,
        "current_feature_surface": {
            "raw_vertical_feature_count": len(raw_vertical_features),
            "raw_vertical_features": raw_vertical_features,
            "vertical_pca_feature_count": len(vertical_pca_features),
            "vertical_pca_features": vertical_pca_features,
        },
        "prior_evidence": prior_status,
        "gates": {
            "fixed_height_ready": fixed_height_ready,
            "pca_ready": pca_ready,
            "same_run_basis_safe_from_n10b": True,
            "ldaps_50_extrema_not_relabelled_as_mean": True,
        },
        "verdict": verdict,
        "handoff": handoff,
        "actions": {
            "model_fits": 0,
            "weather_value_rows": 0,
            "target_or_scada_access": False,
            "official_score_calls": 0,
            "comparison_index": None,
            "lockbox_2024_performance_access": False,
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
        default=Path("reports/s17_n12_vertical_profile_prerequisite_predeclaration.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/audits/s17_n12_vertical_profile_prerequisites.json"),
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
                "gates": result["gates"],
                "verdict": result["verdict"],
                "handoff": result["handoff"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
