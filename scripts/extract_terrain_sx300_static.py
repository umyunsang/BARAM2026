"""Acquire and extract the frozen S17-N20 Copernicus Sx lookup."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests
from PIL import Image


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def acquire(repo: Path, predeclaration: Path) -> dict[str, Any]:
    frozen = json.loads(predeclaration.read_text())
    contract = frozen["external_object_contract"]
    destination = repo / contract["allowed_destination"]
    receipt_path = destination.parent / "acquisition_receipt.json"
    if destination.exists() or receipt_path.exists():
        raise RuntimeError("N20 acquisition is exactly-once; destination or receipt already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + contract["partial_suffix"])
    if partial.exists():
        raise RuntimeError("N20 stale partial exists")
    expected = contract["expected"]
    digest = hashlib.sha256()
    body_bytes = 0
    retrieved_at = datetime.now(UTC).isoformat()
    try:
        with requests.get(
            contract["url"],
            stream=True,
            timeout=(30, 180),
            allow_redirects=False,
            headers={"User-Agent": "BARAM2026-S17-N20-static-extraction/1.0"},
        ) as response:
            observed = {
                "status": response.status_code,
                "content_length": int(response.headers.get("Content-Length", "0")),
                "etag": response.headers.get("ETag"),
                "last_modified": response.headers.get("Last-Modified"),
                "content_type": response.headers.get("Content-Type"),
                "request_method": response.request.method,
                "redirects": len(response.history),
            }
            required = {
                "status": 200,
                "content_length": expected["content_length"],
                "etag": expected["etag"],
                "last_modified": expected["last_modified"],
                "content_type": expected["content_type"],
                "request_method": "GET",
                "redirects": 0,
            }
            if observed != required:
                raise RuntimeError(f"N20 object identity mismatch: {observed}")
            with partial.open("xb") as stream:
                for block in response.iter_content(chunk_size=1 << 20):
                    if not block:
                        continue
                    body_bytes += len(block)
                    if body_bytes > contract["maximum_body_bytes"]:
                        raise RuntimeError("N20 body byte cap exceeded")
                    digest.update(block)
                    stream.write(block)
                stream.flush()
                os.fsync(stream.fileno())
        if body_bytes != expected["content_length"]:
            raise RuntimeError("N20 streamed byte count mismatch")
        os.replace(partial, destination)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    receipt = {
        "schema_version": 1,
        "node_id": frozen["node_id"],
        "retrieved_at": retrieved_at,
        "url": contract["url"],
        "request_count": 1,
        "request_method": "GET",
        "request_headers": {"Range": None},
        "response": observed,
        "body_bytes": body_bytes,
        "sha256": digest.hexdigest(),
        "destination": str(destination.relative_to(repo)),
        "atomic_rename": True,
        "partial_survives": partial.exists(),
        "other_external_requests": 0,
    }
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    return receipt


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def parse_geokeys(values: tuple[int, ...]) -> dict[int, dict[str, int]]:
    if len(values) < 4 or values[0:3] != (1, 1, 0):
        raise RuntimeError("N20 unsupported GeoKey directory header")
    count = values[3]
    if len(values) != 4 + 4 * count:
        raise RuntimeError("N20 malformed GeoKey directory")
    parsed = {}
    for offset in range(4, len(values), 4):
        key, location, value_count, value_offset = values[offset : offset + 4]
        parsed[key] = {
            "location": location,
            "count": value_count,
            "value_offset": value_offset,
        }
    return parsed


def decode_audit(repo: Path, predeclaration: Path) -> dict[str, Any]:
    frozen = json.loads(predeclaration.read_text())
    observed_inputs = {
        relative: sha256_path(repo / relative)
        for relative in frozen["input_bundle"]["files"]
    }
    if observed_inputs != frozen["input_bundle"]["files"]:
        raise RuntimeError("N20 input hash mismatch")
    if canonical_hash(observed_inputs) != frozen["input_bundle"]["sha256"]:
        raise RuntimeError("N20 input bundle mismatch")
    contract = frozen["external_object_contract"]
    destination = repo / contract["allowed_destination"]
    receipt_path = destination.parent / "acquisition_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    body_sha256 = sha256_path(destination)
    if receipt["sha256"] != body_sha256:
        raise RuntimeError("N20 acquired body hash mismatch")
    if receipt["body_bytes"] != destination.stat().st_size:
        raise RuntimeError("N20 acquired body size mismatch")
    required_tags = frozen["extraction_contract"]["required_geotiff_tags"]
    with Image.open(destination) as image:
        observed_tags = sorted(int(tag) for tag in image.tag_v2)
        missing_tags = sorted(set(required_tags) - set(observed_tags))
        geokey_values = tuple(int(value) for value in image.tag_v2[34735])
        geokeys = parse_geokeys(geokey_values)
        metadata = {
            "format": image.format,
            "mode": image.mode,
            "width": image.width,
            "height": image.height,
            "n_frames": image.n_frames,
            "observed_tags": observed_tags,
            "required_tags": required_tags,
            "missing_tags": missing_tags,
            "pixel_scale": list(image.tag_v2[33550]),
            "tiepoint": list(image.tag_v2[33922]),
            "geokey_directory": list(geokey_values),
            "geokeys": {str(key): value for key, value in sorted(geokeys.items())},
            "model_type": geokeys[1024]["value_offset"],
            "raster_type": geokeys[1025]["value_offset"],
            "geographic_epsg": geokeys[2048]["value_offset"],
            "pixel_values_read": 0,
        }
    if metadata["model_type"] != 2:
        raise RuntimeError("N20 GeoTIFF is not geographic")
    if metadata["raster_type"] != 2:
        raise RuntimeError("N20 GeoTIFF contradicts frozen PixelIsPoint semantics")
    if metadata["geographic_epsg"] != 4326:
        raise RuntimeError("N20 GeoTIFF is not EPSG:4326")
    if missing_tags != [42113]:
        raise RuntimeError(f"N20 unexpected required-tag result: {missing_tags}")
    result = {
        "schema_version": 1,
        "node_id": frozen["node_id"],
        "predeclaration_sha256": sha256_path(predeclaration),
        "input_bundle_sha256": frozen["input_bundle"]["sha256"],
        "acquisition": {
            "receipt_path": str(receipt_path.relative_to(repo)),
            "receipt_sha256": sha256_path(receipt_path),
            "body_path": str(destination.relative_to(repo)),
            "body_sha256": body_sha256,
            "body_bytes": destination.stat().st_size,
            "request_count": receipt["request_count"],
            "response": receipt["response"],
        },
        "decode_metadata": metadata,
        "gates": {
            "object_identity": True,
            "decode_required_tags": False,
            "decode_epsg4326": True,
            "decode_pixel_is_point": True,
            "coverage": None,
            "formula": None,
            "parity": None,
            "lookup": None,
        },
        "lookup_artifacts_written": [],
        "verdict": "BLOCKED_MISSING_GDAL_NODATA_TAG_STATIC_BODY_FROZEN",
        "handoff": ["S17-N20A_TERRAIN_SX300_H8_NO_NODATA_TAG_RECOVERY"],
        "actions": {
            "external_requests_total": 1,
            "external_request_methods": ["GET"],
            "dem_body_bytes": destination.stat().st_size,
            "pixel_values_read": 0,
            "weather_rows": 0,
            "label_target_scada_rows": 0,
            "model_or_optimizer_fits": 0,
            "official_or_component_score_calls": 0,
            "strict_protocol_calls": 0,
            "2024_access": False,
            "test_access": False,
            "rejected_ecmwf_access": False,
            "quarantined_n10_access": False,
            "dependency_changes": False,
            "dacon_actions": [],
        },
    }
    output = destination.parent / "manifest.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--predeclaration",
        type=Path,
        default=Path("reports/s17_n20_terrain_sx300_static_extraction_predeclaration.json"),
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--acquire-only", action="store_true")
    mode.add_argument("--decode-audit", action="store_true")
    args = parser.parse_args()
    repo = args.repo.resolve()
    predeclaration = args.predeclaration
    if not predeclaration.is_absolute():
        predeclaration = repo / predeclaration
    if args.acquire_only:
        result = acquire(repo, predeclaration)
    else:
        result = decode_audit(repo, predeclaration)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
