"""Audit static prerequisites for an R3 height-only terrain correction."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import urllib.request
import zipfile
from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from pypdf import PdfReader

MEMBER = "train/ldaps_train.csv"
COLUMNS = (
    "forecast_kst_dtm",
    "grid_id",
    "latitude",
    "longitude",
    "surface_0_h",
)
CUTOFF = pd.Timestamp("2023-12-31")


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


def _request(url: str, *, method: str = "GET") -> tuple[bytes, dict[str, Any]]:
    request = urllib.request.Request(
        url,
        method=method,
        headers={"User-Agent": "BARAM2026-static-prerequisite-audit/1.0"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        body = response.read() if method == "GET" else b""
        metadata = {
            "status": int(response.status),
            "headers": {key.lower(): value for key, value in response.headers.items()},
            "final_url": response.geturl(),
        }
    return body, metadata


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _download_sources(
    sources: dict[str, str], source_dir: Path, *, offline: bool
) -> dict[str, Any]:
    source_dir.mkdir(parents=True, exist_ok=True)
    names = {
        "license": "License-COPDEM-30.pdf",
        "product": "cop_dem_product.html",
        "aws_readme": "aws_readme.html",
    }
    receipts: dict[str, Any] = {}
    for key, filename in names.items():
        path = source_dir / filename
        if offline:
            if not path.is_file():
                raise RuntimeError(f"missing frozen source: {path}")
            body = path.read_bytes()
            metadata = json.loads((source_dir / f"{filename}.headers.json").read_text())
        else:
            body, metadata = _request(sources[key])
            if metadata["status"] != 200 or not body:
                raise RuntimeError(f"source retrieval failed: {key}")
            path.write_bytes(body)
            (source_dir / f"{filename}.headers.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
            )
        receipts[key] = {
            "url": sources[key],
            "path": str(path),
            "sha256": _sha256(path),
            "bytes": len(body),
            "http": metadata,
        }
    if offline:
        head_metadata = json.loads((source_dir / "tile_head.json").read_text())
    else:
        body, head_metadata = _request(sources["tile_head_only"], method="HEAD")
        if body:
            raise RuntimeError("HEAD unexpectedly returned a retained body")
        (source_dir / "tile_head.json").write_text(
            json.dumps(head_metadata, ensure_ascii=False, indent=2) + "\n"
        )
    receipts["tile_head_only"] = {
        "url": sources["tile_head_only"],
        "method": "HEAD",
        "body_bytes": 0,
        "http": head_metadata,
    }
    return receipts


def _pdf_text(path: Path) -> str:
    return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)


def _audit_orography(archive_path: Path) -> dict[str, Any]:
    aggregates: dict[int, dict[str, Any]] = {}
    retained_rows = 0
    out_of_scope_rows = 0
    with zipfile.ZipFile(archive_path) as archive:
        with archive.open(MEMBER) as stream:
            chunks = pd.read_csv(stream, usecols=list(COLUMNS), chunksize=250_000)
            for chunk in chunks:
                forecast = pd.to_datetime(chunk["forecast_kst_dtm"], errors="raise")
                operating_day = (forecast - timedelta(hours=1)).dt.normalize()
                keep = operating_day <= CUTOFF
                out_of_scope_rows += int((~keep).sum())
                scoped = chunk.loc[keep].copy()
                retained_rows += len(scoped)
                # Conversion and every value statistic occur only after the frozen filter.
                for column in ("latitude", "longitude", "surface_0_h"):
                    scoped[column] = pd.to_numeric(scoped[column], errors="coerce")
                for grid_id, group in scoped.groupby("grid_id", sort=False):
                    key = int(grid_id)
                    record = aggregates.setdefault(
                        key,
                        {
                            "rows": 0,
                            "latitude_min": math.inf,
                            "latitude_max": -math.inf,
                            "longitude_min": math.inf,
                            "longitude_max": -math.inf,
                            "surface_0_h_min": math.inf,
                            "surface_0_h_max": -math.inf,
                            "nonfinite": 0,
                        },
                    )
                    record["rows"] += len(group)
                    values = group[["latitude", "longitude", "surface_0_h"]]
                    finite = values.notna().all(axis=1)
                    finite &= values.map(math.isfinite).all(axis=1)
                    record["nonfinite"] += int((~finite).sum())
                    valid = values.loc[finite]
                    if valid.empty:
                        continue
                    for column in ("latitude", "longitude", "surface_0_h"):
                        record[f"{column}_min"] = min(
                            record[f"{column}_min"], float(valid[column].min())
                        )
                        record[f"{column}_max"] = max(
                            record[f"{column}_max"], float(valid[column].max())
                        )
    grids = {str(key): value for key, value in sorted(aggregates.items())}
    finite = all(record["nonfinite"] == 0 for record in grids.values())
    constant = all(
        record["latitude_min"] == record["latitude_max"]
        and record["longitude_min"] == record["longitude_max"]
        and record["surface_0_h_min"] == record["surface_0_h_max"]
        for record in grids.values()
    )
    return {
        "member": MEMBER,
        "operating_day_max": str(CUTOFF.date()),
        "filter_before_conversion_and_statistics": True,
        "retained_rows": retained_rows,
        "out_of_scope_rows_skipped": out_of_scope_rows,
        "grid_count": len(grids),
        "all_finite": finite,
        "constant_coordinates_and_orography_per_grid": constant,
        "grids": grids,
    }


def run(
    repo: Path,
    predeclaration: Path,
    output: Path,
    source_dir: Path,
    *,
    offline: bool,
) -> dict[str, Any]:
    frozen = json.loads(predeclaration.read_text())
    observed = {
        relative: _sha256(repo / relative)
        for relative in frozen["input_bundle"]["files"]
    }
    if observed != frozen["input_bundle"]["files"]:
        raise RuntimeError("N14 input hash mismatch")
    if _canonical_hash(observed) != frozen["input_bundle"]["sha256"]:
        raise RuntimeError("N14 input bundle mismatch")
    receipts = _download_sources(
        frozen["official_sources"], source_dir, offline=offline
    )
    licence_path = Path(receipts["license"]["path"])
    product_path = Path(receipts["product"]["path"])
    aws_path = Path(receipts["aws_readme"]["path"])
    if not licence_path.is_absolute():
        licence_path = repo / licence_path
    if not product_path.is_absolute():
        product_path = repo / product_path
    if not aws_path.is_absolute():
        aws_path = repo / aws_path
    licence = _normalise(_pdf_text(licence_path))
    product = _normalise(product_path.read_text(errors="replace"))
    aws = _normalise(aws_path.read_text(errors="replace"))
    rights = {
        "worldwide_unlimited_time": "worldwide and without limitation in time" in licence,
        "reproduction": "(a) reproduction" in licence,
        "distribution": "(b) distribution" in licence,
        "public_communication": "(c) communication to the general public" in licence,
        "adaptation_modification_combination": (
            "(d) adaptation, modification and combination with other data and information"
            in licence
        ),
        "free_of_charge": "free of charge to the user" in licence,
        "noncommercial_restriction_absent": not any(
            phrase in licence
            for phrase in ("non-commercial", "noncommercial", "non commercial")
        ),
        "attribution_obligation_captured": (
            "© dlr e.v. 2010-2014" in licence
            and "© airbus defence and space gmbh 2014-2018" in licence
        ),
    }
    license_gate = all(rights.values())
    time_evidence = {
        "source_acquired_2011_2015": (
            "between 2011 and 2015" in product
            or ("2011" in product and "2015" in product)
        ),
        "available_for_use_2019": "available for use in 2019" in product,
        "glo30_global_30m": "glo-30 offers global coverage at a resolution of 30 metres" in product,
        "aws_free_general_public": (
            "available on a free basis for the general public" in aws
        ),
    }
    time_gate = all(time_evidence.values())
    head = receipts["tile_head_only"]["http"]
    content_length = int(head["headers"].get("content-length", "0"))
    availability_gate = bool(head["status"] == 200 and content_length > 0)
    orography = _audit_orography(repo / "inputs/competition/open_wind_236727.zip")
    model_orography_gate = bool(
        orography["retained_rows"] > 0
        and orography["grid_count"] == 16
        and orography["all_finite"]
        and orography["constant_coordinates_and_orography_per_grid"]
    )
    n12_receipt = json.loads(
        (repo / "reports/s17_n12_vertical_profile_prerequisite_receipt.json").read_text()
    )
    n12_report_path = repo / n12_receipt["artifact"]["path"]
    if _sha256(n12_report_path) != n12_receipt["artifact"]["sha256"]:
        raise RuntimeError("N12 report mutation")
    n12_report = json.loads(n12_report_path.read_text())
    static = n12_report["static_workbook"]["info"]
    records = static["records"]
    static_gate = bool(
        len(records) == 17
        and all(record["좌표(Google)"] for record in records)
        and all(float(record["Hub Height(m)"]) == 117.0 for record in records)
    )
    formula_text = (repo / "research/lanes/S13_S5_preprocessing_deep.md").read_text()
    formula_gate = all(
        marker in formula_text
        for marker in ("u_HC(z)", "exp(", "z_site", "z_model")
    )
    formula_inputs_gate = bool(
        static_gate and model_orography_gate and availability_gate and formula_gate
    )
    gates = {
        "license": license_gate,
        "time": time_gate,
        "availability": availability_gate,
        "model_orography": model_orography_gate,
        "formula_inputs": formula_inputs_gate,
    }
    verdict = "READY_STATIC_DEM_EXTRACTION" if all(gates.values()) else "BLOCKED"
    result = {
        "schema_version": 1,
        "node_id": frozen["node_id"],
        "predeclaration_sha256": _sha256(predeclaration),
        "input_bundle_sha256": frozen["input_bundle"]["sha256"],
        "source_receipts": receipts,
        "license_rights": rights,
        "time_evidence": time_evidence,
        "tile_head": {
            "status": head["status"],
            "content_length": content_length,
            "last_modified": head["headers"].get("last-modified"),
            "etag": head["headers"].get("etag"),
            "body_bytes": 0,
        },
        "ldaps_model_orography": orography,
        "supplied_static_metadata": {
            "turbines": len(records),
            "coordinate_rows": sum(bool(record["좌표(Google)"]) for record in records),
            "hub_height_values_m": sorted(
                {float(record["Hub Height(m)"]) for record in records}
            ),
        },
        "published_formula_markers": {
            "u_hc": "u_HC(z)" in formula_text,
            "exponential": "exp(" in formula_text,
            "site_height": "z_site" in formula_text,
            "model_height": "z_model" in formula_text,
        },
        "gates": gates,
        "verdict": verdict,
        "handoff": (
            frozen["handoff_if_ready"] if verdict == "READY_STATIC_DEM_EXTRACTION" else []
        ),
        "actions": {
            "dem_tile_method": "HEAD",
            "dem_tile_body_bytes": 0,
            "model_fits": 0,
            "official_score_calls": 0,
            "target_or_scada_access": False,
            "test_member_access": False,
            "lockbox_2024_performance_access": False,
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
        default=Path("reports/s17_n14_r3_terrain_prerequisite_predeclaration.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/audits/s17_n14_r3_terrain_prerequisite.json"),
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("artifacts/external/copdem_n14_prerequisite_sources"),
    )
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    repo = args.repo.resolve()
    predeclaration = args.predeclaration
    output = args.output
    source_dir = args.source_dir
    if not predeclaration.is_absolute():
        predeclaration = repo / predeclaration
    if not output.is_absolute():
        output = repo / output
    if not source_dir.is_absolute():
        source_dir = repo / source_dir
    print(
        json.dumps(
            run(repo, predeclaration, output, source_dir, offline=args.offline),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
