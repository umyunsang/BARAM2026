#!/usr/bin/env python3
"""Extract the frozen 2023 ECMWF 18Z four-field development cube."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from baram.constants import BASELINE_NOTEBOOK_SHA256, OPEN_ZIP_SHA256
from baram.data.ecmwf_open import (
    CENTROIDS,
    DECLARED_MISSING_INIT_DATES,
    FIELDS,
    HOURLY_LEADS,
    STEPS,
    extract_present_day,
    init_date_for_operating_day,
    unavailable_day_frame,
)

PREDECLARATION_SHA256 = "306d70bc23986f569aac4bda21da7fd5578410b8e70d3cb9926b75854ffd3d31"
ROOT = Path(__file__).resolve().parents[1]
PREDECLARATION = ROOT / "reports/s17_n5_ecmwf_18z_extraction_predeclaration.json"
OUTPUT_DIR = ROOT / "artifacts/external/ecmwf_18z_2023"
CHECKPOINTS = OUTPUT_DIR / "checkpoints"
FINAL_DATA = OUTPUT_DIR / "ecmwf_18z_q234.parquet"
FINAL_MANIFEST = OUTPUT_DIR / "manifest.json"
OPERATING_START = date(2023, 4, 1)
OPERATING_END = date(2023, 12, 31)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_bytes(
        path,
        (json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n").encode(),
    )


def contract() -> tuple[dict[str, Any], str]:
    payload = {
        "predeclaration_sha256": PREDECLARATION_SHA256,
        "operating_start": OPERATING_START.isoformat(),
        "operating_end": OPERATING_END.isoformat(),
        "steps": list(STEPS),
        "hourly_leads": list(HOURLY_LEADS),
        "fields": [field.__dict__ for field in FIELDS],
        "centroids": CENTROIDS,
        "spatial": "nearest4_idw_power2",
        "temporal": "linear_uv_components",
        "declared_missing_init_dates": sorted(
            value.isoformat() for value in DECLARED_MISSING_INIT_DATES
        ),
        "module_sha256": sha256(ROOT / "src/baram/data/ecmwf_open.py"),
    }
    return payload, canonical_sha(payload)


def verify_inputs() -> dict[str, str]:
    paths = {
        "competition_archive": ROOT / "inputs/competition/open_wind_236727.zip",
        "baseline_notebook": ROOT / "inputs/notebooks/baseline.ipynb",
    }
    hashes = {name: sha256(path) for name, path in paths.items()}
    if hashes["competition_archive"] != OPEN_ZIP_SHA256:
        raise RuntimeError("competition archive hash mismatch")
    if hashes["baseline_notebook"] != BASELINE_NOTEBOOK_SHA256:
        raise RuntimeError("baseline notebook hash mismatch")
    if sha256(PREDECLARATION) != PREDECLARATION_SHA256:
        raise RuntimeError("predeclaration hash mismatch")
    return hashes


def checkpoint_paths(init_date: date) -> tuple[Path, Path, Path]:
    stem = init_date.strftime("%Y%m%d")
    return (
        CHECKPOINTS / f"{stem}.parquet",
        CHECKPOINTS / f"{stem}.receipt.json",
        CHECKPOINTS / f"{stem}.COMMITTED.json",
    )


def verify_checkpoint(init_date: date, contract_sha256: str) -> dict[str, Any] | None:
    data_path, receipt_path, marker_path = checkpoint_paths(init_date)
    if not marker_path.exists():
        for orphan in (data_path, receipt_path):
            if orphan.exists():
                orphan.unlink()
        return None
    marker = json.loads(marker_path.read_text())
    if marker.get("contract_sha256") != contract_sha256:
        raise RuntimeError(f"checkpoint contract conflict: {init_date}")
    if not data_path.is_file() or not receipt_path.is_file():
        raise RuntimeError(f"committed checkpoint incomplete: {init_date}")
    if sha256(data_path) != marker.get("data_sha256"):
        raise RuntimeError(f"checkpoint data hash conflict: {init_date}")
    if sha256(receipt_path) != marker.get("receipt_sha256"):
        raise RuntimeError(f"checkpoint receipt hash conflict: {init_date}")
    frame = pd.read_parquet(data_path)
    if len(frame) != 72 or set(frame["extraction_contract_sha256"]) != {contract_sha256}:
        raise RuntimeError(f"checkpoint frame contract conflict: {init_date}")
    return marker


def write_checkpoint(
    init_date: date,
    frame: pd.DataFrame,
    receipt: dict[str, Any],
    contract_sha256: str,
) -> dict[str, Any]:
    data_path, receipt_path, marker_path = checkpoint_paths(init_date)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_data = data_path.with_suffix(".parquet.tmp")
    frame.to_parquet(temporary_data, index=False, compression="zstd")
    with temporary_data.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary_data, data_path)
    receipt = {
        **receipt,
        "contract_sha256": contract_sha256,
        "predeclaration_sha256": PREDECLARATION_SHA256,
        "data_path": str(data_path.relative_to(ROOT)),
        "data_sha256": sha256(data_path),
        "generated_at": datetime.now().astimezone().isoformat(),
    }
    atomic_json(receipt_path, receipt)
    marker = {
        "init_date": init_date.isoformat(),
        "contract_sha256": contract_sha256,
        "data_sha256": sha256(data_path),
        "receipt_sha256": sha256(receipt_path),
        "rows": 72,
    }
    atomic_json(marker_path, marker)
    return marker


def process_date(
    init_date: date,
    *,
    contract_sha256: str,
    max_workers: int,
) -> dict[str, Any]:
    existing = verify_checkpoint(init_date, contract_sha256)
    if existing is not None:
        print(f"SKIP {init_date} {existing['data_sha256'][:12]}", flush=True)
        return existing
    if init_date in DECLARED_MISSING_INIT_DATES:
        frame = unavailable_day_frame(init_date)
        receipt = {
            "init_date": init_date.isoformat(),
            "operating_day": str(frame["operating_day"].iloc[0].date()),
            "declared_archive_gap": True,
            "index_requests": [],
            "range_requests": [],
            "index_bytes": 0,
            "range_bytes": 0,
            "rows": 72,
            "raw_grib_retained": False,
        }
    else:
        frame, receipt = extract_present_day(init_date, max_workers=max_workers)
        receipt["declared_archive_gap"] = False
    frame["extraction_contract_sha256"] = contract_sha256
    frame["source_license"] = "CC-BY-4.0"
    marker = write_checkpoint(init_date, frame, receipt, contract_sha256)
    print(
        f"DONE {init_date} gap={receipt['declared_archive_gap']} "
        f"range_MB={receipt['range_bytes'] / 1e6:.3f} {marker['data_sha256'][:12]}",
        flush=True,
    )
    return marker


def all_init_dates() -> list[date]:
    values = []
    current = init_date_for_operating_day(OPERATING_START)
    final = init_date_for_operating_day(OPERATING_END)
    while current <= final:
        values.append(current)
        current += timedelta(days=1)
    if len(values) != 275:
        raise RuntimeError("unexpected extraction date count")
    return values


def finalize(
    dates: list[date],
    *,
    contract_payload: dict[str, Any],
    contract_sha256: str,
    input_hashes: dict[str, str],
) -> dict[str, Any]:
    markers = []
    frames = []
    day_receipts = []
    for init_date in dates:
        marker = verify_checkpoint(init_date, contract_sha256)
        if marker is None:
            raise RuntimeError(f"missing checkpoint: {init_date}")
        data_path, receipt_path, _ = checkpoint_paths(init_date)
        markers.append(marker)
        frames.append(pd.read_parquet(data_path))
        day_receipts.append(json.loads(receipt_path.read_text()))
    result = pd.concat(frames, ignore_index=True).sort_values(
        ["forecast_kst_dtm", "group_id"], kind="stable"
    )
    keys = ["fold_id", "group_id", "forecast_kst_dtm"]
    if len(result) != 275 * 72 or result.duplicated(keys).any():
        raise RuntimeError("final cube row/key mismatch")
    if int(result["ecmwf_available"].sum()) != 269 * 72:
        raise RuntimeError("final source availability count mismatch")
    if set(result.loc[~result["ecmwf_available"], "operating_day"].dt.date) != {
        date(2023, 4, 29) + timedelta(days=offset) for offset in range(6)
    }:
        raise RuntimeError("final missing operating-day set mismatch")
    temporary = FINAL_DATA.with_suffix(".parquet.tmp")
    FINAL_DATA.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(temporary, index=False, compression="zstd")
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, FINAL_DATA)
    manifest = {
        "schema_version": 1,
        "node_id": "S17-N5_ECMWF_18Z_2023_EXTRACTION",
        "generated_at": datetime.now().astimezone().isoformat(),
        "contract": contract_payload,
        "contract_sha256": contract_sha256,
        "input_hashes": input_hashes,
        "data_path": str(FINAL_DATA.relative_to(ROOT)),
        "data_sha256": sha256(FINAL_DATA),
        "rows": len(result),
        "available_rows": int(result["ecmwf_available"].sum()),
        "unavailable_rows": int((~result["ecmwf_available"]).sum()),
        "operating_days": int(result["operating_day"].nunique()),
        "available_operating_days": int(
            result.loc[result["ecmwf_available"], "operating_day"].nunique()
        ),
        "index_bytes_transferred": sum(row["index_bytes"] for row in day_receipts),
        "range_bytes_transferred": sum(row["range_bytes"] for row in day_receipts),
        "grib_messages_retained": 0,
        "checkpoints": markers,
    }
    atomic_json(FINAL_MANIFEST, manifest)
    manifest["manifest_sha256"] = sha256(FINAL_MANIFEST)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--smoke-date", type=date.fromisoformat)
    parser.add_argument("--init-start", type=date.fromisoformat)
    parser.add_argument("--init-end", type=date.fromisoformat)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= args.max_workers <= 6:
        raise RuntimeError("max-workers must be 1-6")
    input_hashes = verify_inputs()
    contract_payload, contract_sha256 = contract()
    dates = all_init_dates()
    if args.smoke_date is not None:
        if args.init_start is not None or args.init_end is not None:
            raise RuntimeError("smoke-date cannot be combined with a date range")
        if args.smoke_date not in dates:
            raise RuntimeError("smoke date outside frozen extraction range")
        process_date(
            args.smoke_date,
            contract_sha256=contract_sha256,
            max_workers=args.max_workers,
        )
        return
    if (args.init_start is None) != (args.init_end is None):
        raise RuntimeError("init-start and init-end must be supplied together")
    selected = dates
    partial = args.init_start is not None
    if partial:
        if args.init_start > args.init_end:
            raise RuntimeError("invalid init range")
        selected = [
            value for value in dates if args.init_start <= value <= args.init_end
        ]
        if not selected:
            raise RuntimeError("init range outside frozen extraction surface")
    for init_date in selected:
        process_date(
            init_date,
            contract_sha256=contract_sha256,
            max_workers=args.max_workers,
        )
    if partial:
        print(
            json.dumps(
                {
                    "partial": True,
                    "first": selected[0].isoformat(),
                    "last": selected[-1].isoformat(),
                    "days": len(selected),
                    "contract_sha256": contract_sha256,
                },
                indent=2,
            )
        )
        return
    manifest = finalize(
        dates,
        contract_payload=contract_payload,
        contract_sha256=contract_sha256,
        input_hashes=input_hashes,
    )
    print(json.dumps(manifest, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
