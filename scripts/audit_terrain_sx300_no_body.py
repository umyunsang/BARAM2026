"""No-body prerequisite audit for the frozen S17-N19 terrain Sx lookup."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
import tempfile
import tomllib
from datetime import datetime
from email.utils import parsedate_to_datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import requests
from PIL import Image, TiffImagePlugin, features

WGS84_A_M = 6_378_137.0
WGS84_E2 = 6.6943799901413165e-3
CUTOFF_UTC = "2026-07-05T00:00:00+00:00"
COPDEM_BASE = "https://copernicus-dem-30m.s3.amazonaws.com"


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


def _verify_inputs(repo: Path, predeclaration: Path) -> dict[str, Any]:
    frozen = json.loads(predeclaration.read_text())
    observed = {
        relative: _sha256(repo / relative)
        for relative in frozen["input_bundle"]["files"]
    }
    if observed != frozen["input_bundle"]["files"]:
        raise RuntimeError("N19 input hash mismatch")
    if _canonical_hash(observed) != frozen["input_bundle"]["sha256"]:
        raise RuntimeError("N19 input bundle mismatch")
    return frozen


def _radii(latitude_deg: float) -> tuple[float, float]:
    phi = math.radians(latitude_deg)
    denominator = math.sqrt(1.0 - WGS84_E2 * math.sin(phi) ** 2)
    prime_vertical = WGS84_A_M / denominator
    meridional = (
        WGS84_A_M
        * (1.0 - WGS84_E2)
        / (1.0 - WGS84_E2 * math.sin(phi) ** 2) ** 1.5
    )
    return meridional, prime_vertical


def _buffer_bounds(
    latitude_deg: float, longitude_deg: float, radius_m: float
) -> tuple[float, float, float, float]:
    meridional, prime_vertical = _radii(latitude_deg)
    latitude_delta = math.degrees(radius_m / meridional)
    longitude_delta = math.degrees(
        radius_m / (prime_vertical * math.cos(math.radians(latitude_deg)))
    )
    return (
        latitude_deg - latitude_delta,
        latitude_deg + latitude_delta,
        longitude_deg - longitude_delta,
        longitude_deg + longitude_delta,
    )


def _tile_ids(
    coordinates: list[tuple[float, float]], radius_m: float
) -> list[str]:
    tiles: set[str] = set()
    for latitude, longitude in coordinates:
        south, north, west, east = _buffer_bounds(latitude, longitude, radius_m)
        for latitude_degree in range(math.floor(south), math.floor(north) + 1):
            for longitude_degree in range(math.floor(west), math.floor(east) + 1):
                north_south = "N" if latitude_degree >= 0 else "S"
                east_west = "E" if longitude_degree >= 0 else "W"
                tiles.add(
                    f"{north_south}{abs(latitude_degree):02d}_"
                    f"{east_west}{abs(longitude_degree):03d}"
                )
    return sorted(tiles)


def _tile_url(tile_id: str) -> str:
    latitude, longitude = tile_id.split("_")
    stem = f"Copernicus_DSM_COG_10_{latitude}_00_{longitude}_00_DEM"
    return f"{COPDEM_BASE}/{stem}/{stem}.tif"


def _head_tile(tile_id: str) -> dict[str, Any]:
    url = _tile_url(tile_id)
    response = requests.head(
        url,
        timeout=30,
        allow_redirects=True,
        headers={"User-Agent": "BARAM2026-S17-N19-no-body-audit/1.0"},
    )
    result = {
        "tile_id": tile_id,
        "method": "HEAD",
        "url": url,
        "status": response.status_code,
        "content_length": int(response.headers.get("Content-Length", "0")),
        "content_type": response.headers.get("Content-Type"),
        "etag": response.headers.get("ETag"),
        "last_modified": response.headers.get("Last-Modified"),
        "body_bytes": 0,
    }
    if result["status"] != 200 or result["content_length"] <= 0:
        raise RuntimeError(f"N19 tile HEAD failed: {result}")
    if not result["etag"] or not result["last_modified"]:
        raise RuntimeError("N19 tile identity headers missing")
    modified = parsedate_to_datetime(result["last_modified"])
    cutoff = datetime.fromisoformat(CUTOFF_UTC)
    result["pre_cutoff"] = modified <= cutoff
    if not result["pre_cutoff"]:
        raise RuntimeError("N19 tile postdates the eligibility cutoff")
    return result


def _cells_at_point(value: float, tolerance: float = 1e-12) -> set[int]:
    rounded = round(value)
    if abs(value - rounded) <= tolerance:
        return {int(rounded) - 1, int(rounded)}
    return {math.floor(value)}


def _supercover_breakpoints(
    start: tuple[float, float], end: tuple[float, float]
) -> set[tuple[int, int]]:
    x0, y0 = start
    x1, y1 = end
    dx = x1 - x0
    dy = y1 - y0
    breakpoints = {0.0, 1.0}
    if abs(dx) > 1e-15:
        for boundary in range(math.floor(min(x0, x1)), math.ceil(max(x0, x1)) + 1):
            position = (boundary - x0) / dx
            if 0.0 <= position <= 1.0:
                breakpoints.add(float(position))
    if abs(dy) > 1e-15:
        for boundary in range(math.floor(min(y0, y1)), math.ceil(max(y0, y1)) + 1):
            position = (boundary - y0) / dy
            if 0.0 <= position <= 1.0:
                breakpoints.add(float(position))
    ordered = sorted(breakpoints)
    probes = set(ordered)
    probes.update((left + right) / 2.0 for left, right in pairwise(ordered))
    cells: set[tuple[int, int]] = set()
    for position in probes:
        x_value = x0 + position * dx
        y_value = y0 + position * dy
        cells.update(
            (x_cell, y_cell)
            for x_cell in _cells_at_point(x_value)
            for y_cell in _cells_at_point(y_value)
        )
    return cells


def _segment_intersects_cell(
    start: tuple[float, float],
    end: tuple[float, float],
    cell: tuple[int, int],
) -> bool:
    x0, y0 = start
    dx = end[0] - x0
    dy = end[1] - y0
    x_cell, y_cell = cell
    lower = 0.0
    upper = 1.0
    for direction, near_value in (
        (-dx, x0 - x_cell),
        (dx, x_cell + 1.0 - x0),
        (-dy, y0 - y_cell),
        (dy, y_cell + 1.0 - y0),
    ):
        value = near_value
        if abs(direction) <= 1e-15:
            if value < 0.0:
                return False
            continue
        ratio = value / direction
        if direction < 0.0:
            if ratio > upper:
                return False
            lower = max(lower, ratio)
        else:
            if ratio < lower:
                return False
            upper = min(upper, ratio)
    return lower <= upper + 1e-12


def _supercover_reference(
    start: tuple[float, float], end: tuple[float, float]
) -> set[tuple[int, int]]:
    x_low = math.floor(min(start[0], end[0])) - 1
    x_high = math.ceil(max(start[0], end[0])) + 1
    y_low = math.floor(min(start[1], end[1])) - 1
    y_high = math.ceil(max(start[1], end[1])) + 1
    return {
        (x_cell, y_cell)
        for x_cell in range(x_low, x_high + 1)
        for y_cell in range(y_low, y_high + 1)
        if _segment_intersects_cell(start, end, (x_cell, y_cell))
    }


def _from_direction(u_value: float, v_value: float) -> float:
    if not math.isfinite(u_value) or not math.isfinite(v_value):
        return math.nan
    if u_value == 0.0 and v_value == 0.0:
        return math.nan
    return math.degrees(math.atan2(-u_value, -v_value)) % 360.0


def _direction_bin(direction: float) -> int:
    if not math.isfinite(direction):
        raise ValueError("finite direction required")
    return int(5 * math.floor((direction + 2.5) / 5.0)) % 360


def _sx_for_cells(
    elevation: np.ndarray,
    query: tuple[float, float],
    cells: set[tuple[int, int]],
    *,
    pixel_m: float,
    height_m: float,
) -> float:
    query_cell = (math.floor(query[0]), math.floor(query[1]))
    query_elevation = float(elevation[query_cell[1], query_cell[0]])
    angles = []
    for x_cell, y_cell in cells:
        if (x_cell, y_cell) == query_cell:
            continue
        if not (
            0 <= y_cell < elevation.shape[0]
            and 0 <= x_cell < elevation.shape[1]
        ):
            continue
        x_delta = (x_cell + 0.5 - query[0]) * pixel_m
        y_delta = (y_cell + 0.5 - query[1]) * pixel_m
        distance = math.hypot(x_delta, y_delta)
        if distance <= 0.0:
            continue
        rise = float(elevation[y_cell, x_cell]) - (query_elevation + height_m)
        angles.append(math.degrees(math.atan2(rise, distance)))
    if not angles:
        raise RuntimeError("synthetic Sx ray resolved no noncentral cells")
    return max(angles)


def _sx_sector(
    elevation: np.ndarray,
    query: tuple[float, float],
    direction_deg: float,
    traversal: Any,
    *,
    pixel_m: float = 30.0,
    height_m: float = 8.0,
    dmax_m: float = 300.0,
) -> tuple[float, list[list[list[int]]]]:
    ray_values = []
    serialized_cells = []
    ray_pixels = dmax_m / pixel_m
    for offset in (-15, -10, -5, 0, 5, 10, 15):
        bearing = math.radians((direction_deg + offset) % 360.0)
        endpoint = (
            query[0] + ray_pixels * math.sin(bearing),
            query[1] + ray_pixels * math.cos(bearing),
        )
        cells = traversal(query, endpoint)
        ray_values.append(
            _sx_for_cells(
                elevation,
                query,
                cells,
                pixel_m=pixel_m,
                height_m=height_m,
            )
        )
        serialized_cells.append(
            [[x_cell, y_cell] for x_cell, y_cell in sorted(cells)]
        )
    return float(np.mean(ray_values)), serialized_cells


def _parity_audit() -> dict[str, Any]:
    shape = (61, 61)
    query = (30.5, 30.5)
    y_index, x_index = np.indices(shape)
    cases: dict[str, np.ndarray] = {
        "flat": np.full(shape, 1000.0),
        "planar_slope": 1000.0 + 1.5 * (y_index - 30.0),
        "single_upwind_obstacle": np.full(shape, 1000.0),
        "all_negative_horizon": 1000.0 - 0.5 * np.hypot(x_index - 30.0, y_index - 30.0),
        "shared_corner_ray": 1000.0 + 0.1 * x_index - 0.2 * y_index,
    }
    cases["single_upwind_obstacle"][38, 30] = 1200.0
    directions = {
        "flat": 0.0,
        "planar_slope": 0.0,
        "single_upwind_obstacle": 0.0,
        "all_negative_horizon": 180.0,
        "shared_corner_ray": 45.0,
    }
    records: dict[str, Any] = {}
    maximum_error = 0.0
    all_cell_sets_equal = True
    for name, elevation in cases.items():
        primary, primary_cells = _sx_sector(
            elevation, query, directions[name], _supercover_breakpoints
        )
        reference, reference_cells = _sx_sector(
            elevation, query, directions[name], _supercover_reference
        )
        error = abs(primary - reference)
        cell_sets_equal = primary_cells == reference_cells
        maximum_error = max(maximum_error, error)
        all_cell_sets_equal = all_cell_sets_equal and cell_sets_equal
        records[name] = {
            "primary_sx_deg": primary,
            "reference_sx_deg": reference,
            "abs_error_deg": error,
            "cell_sets_identical": cell_sets_equal,
            "primary_ray_cell_counts": [len(cells) for cells in primary_cells],
        }
    if not all_cell_sets_equal or maximum_error > 1e-12:
        raise RuntimeError("N19 independent Sx parity failed")
    if records["flat"]["primary_sx_deg"] >= 0.0:
        raise RuntimeError("N19 flat H=8 exposure sign failed")
    if records["single_upwind_obstacle"]["primary_sx_deg"] <= 0.0:
        raise RuntimeError("N19 obstacle shelter sign failed")
    return {
        "cases": records,
        "all_cell_sets_identical": all_cell_sets_equal,
        "max_abs_sx_deg": maximum_error,
    }


def _synthetic_tiff_audit() -> dict[str, Any]:
    values = (np.arange(20, dtype=np.float32).reshape(4, 5) - 3.25).astype(np.float32)
    directory = TiffImagePlugin.ImageFileDirectory_v2()
    directory.tagtype[33550] = 12
    directory[33550] = (1.0 / 3600.0, 1.0 / 3600.0, 0.0)
    directory.tagtype[33922] = 12
    directory[33922] = (0.0, 0.0, 0.0, 128.0, 38.0, 0.0)
    directory.tagtype[34735] = 3
    directory[34735] = (
        1,
        1,
        0,
        3,
        1024,
        0,
        1,
        2,
        1025,
        0,
        1,
        2,
        2048,
        0,
        1,
        4326,
    )
    directory.tagtype[42113] = 2
    directory[42113] = "-9999"
    path_string = ""
    with tempfile.TemporaryDirectory(prefix="baram-s17-n19-") as temporary:
        path = Path(temporary) / "synthetic_float32_geotiff.tif"
        path_string = str(path)
        Image.fromarray(values, mode="F").save(
            path,
            format="TIFF",
            compression="tiff_deflate",
            tiffinfo=directory,
        )
        with Image.open(path) as image:
            decoded = np.asarray(image, dtype=np.float32)
            observed_tags = {
                str(tag): image.tag_v2.get(tag)
                for tag in (33550, 33922, 34735, 42113)
            }
        if not np.array_equal(values, decoded):
            raise RuntimeError("N19 synthetic float TIFF roundtrip failed")
        if any(value is None for value in observed_tags.values()):
            raise RuntimeError("N19 synthetic GeoTIFF tag roundtrip failed")
        file_sha256 = _sha256(path)
        file_bytes = path.stat().st_size
    if Path(path_string).exists():
        raise RuntimeError("N19 synthetic TIFF temporary file survived")
    return {
        "pillow_version": Image.__version__,
        "libtiff": bool(features.check("libtiff")),
        "zlib": bool(features.check("zlib")),
        "dtype": str(decoded.dtype),
        "shape": list(decoded.shape),
        "values_exact": True,
        "observed_tags": observed_tags,
        "temporary_file_sha256": file_sha256,
        "temporary_file_bytes": file_bytes,
        "temporary_file_removed": True,
    }


def _assignment_names(node: ast.AST) -> list[str]:
    names: list[str] = []
    if isinstance(node, ast.Name):
        names.append(node.id)
    elif isinstance(node, (ast.Tuple, ast.List)):
        for element in node.elts:
            names.extend(_assignment_names(element))
    return names


def _lineage_audit(repo: Path, manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    observed = {
        relative: _sha256(repo / relative) for relative in manifest["files"]
    }
    if observed != manifest["files"]:
        raise RuntimeError("N19 lineage universe hash mismatch")
    if _canonical_hash(observed) != manifest["bundle_sha256"]:
        raise RuntimeError("N19 lineage bundle mismatch")
    implementation_hits: list[dict[str, str]] = []
    reference_hits: list[dict[str, Any]] = []
    implementation_pattern = re.compile(
        r"terrain_?sx|sx300|winstral|dem_exposure|wind_shelter", re.IGNORECASE
    )
    reference_pattern = re.compile(
        r"winstral|sx_grid|sx-?300|terrain__sx|dem[_ -]?exposure", re.IGNORECASE
    )
    for relative in sorted(manifest["files"]):
        path = repo / relative
        text = path.read_text(errors="strict")
        if path.suffix == ".py":
            tree = ast.parse(text, filename=relative)
            for node in ast.walk(tree):
                names: list[str] = []
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    names.append(node.name)
                elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    for target in targets:
                        names.extend(_assignment_names(target))
                for name in names:
                    if implementation_pattern.search(name):
                        implementation_hits.append(
                            {"path": relative, "kind": type(node).__name__, "name": name}
                        )
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    if re.search(r"terrain__sx|dem_exposure", node.value, re.IGNORECASE):
                        implementation_hits.append(
                            {
                                "path": relative,
                                "kind": "materialized_feature_string",
                                "name": node.value[:160],
                            }
                        )
        else:
            for line_number, line in enumerate(text.splitlines(), start=1):
                if re.search(r"terrain__sx|dem_exposure", line, re.IGNORECASE):
                    implementation_hits.append(
                        {
                            "path": relative,
                            "kind": "materialized_schema_string",
                            "name": f"line {line_number}: {line[:160]}",
                        }
                    )
        lines = [
            {"line": line_number, "text": line[:240]}
            for line_number, line in enumerate(text.splitlines(), start=1)
            if reference_pattern.search(line)
        ]
        if lines:
            reference_hits.append({"path": relative, "lines": lines})
    if implementation_hits:
        raise RuntimeError(f"N19 materialized Sx duplicate found: {implementation_hits[:3]}")
    return {
        "manifest_sha256": _sha256(manifest_path),
        "bundle_sha256": manifest["bundle_sha256"],
        "files_scanned": len(observed),
        "bytes_scanned": sum((repo / relative).stat().st_size for relative in observed),
        "implementation_hits": implementation_hits,
        "textual_reference_hits": reference_hits,
        "binary_or_generated_bodies_read": 0,
    }


def run(repo: Path, predeclaration: Path, output: Path) -> dict[str, Any]:
    frozen = _verify_inputs(repo, predeclaration)
    formula_path = repo / frozen["frozen_formula_manifest"]["path"]
    lineage_path = repo / frozen["frozen_lineage_universe"]["path"]
    if _sha256(formula_path) != frozen["frozen_formula_manifest"]["sha256"]:
        raise RuntimeError("N19 formula manifest mutation")
    formula = json.loads(formula_path.read_text())
    if formula["parameters"]["dmax_m"] != 300.0:
        raise RuntimeError("N19 dmax changed")
    if formula["parameters"]["height_m"] != 8.0:
        raise RuntimeError("N19 height changed")
    if formula["parameters"]["extra_variants_allowed"]:
        raise RuntimeError("N19 parameter variants admitted")
    n14_path = repo / "artifacts/audits/s17_n14_r3_terrain_prerequisite.json"
    n12_path = repo / "artifacts/audits/s17_n12_vertical_profile_prerequisites.json"
    n14 = json.loads(n14_path.read_text())
    n12 = json.loads(n12_path.read_text())
    grids = n14["ldaps_model_orography"]["grids"]
    if len(grids) != 16:
        raise RuntimeError("N19 LDAPS coordinate count changed")
    coordinates = [
        (float(record["latitude_min"]), float(record["longitude_min"]))
        for _, record in sorted(grids.items(), key=lambda item: int(item[0]))
    ]
    if len(set(coordinates)) != 16 or not np.isfinite(np.asarray(coordinates)).all():
        raise RuntimeError("N19 LDAPS coordinates not unique finite")
    headers = n12["weather_headers"]["ldaps"]
    required_headers = frozen["gates"]["source_and_schema"]["required_header_names"]
    if not set(required_headers).issubset(headers):
        raise RuntimeError("N19 LDAPS direction schema missing")
    if n12["weather_rows_read"] != 0:
        raise RuntimeError("N19 inherited header receipt read weather rows")
    tile_ids = _tile_ids(coordinates, formula["parameters"]["dmax_m"])
    if tile_ids != frozen["gates"]["tile_geometry_and_head"]["expected_tile_ids"]:
        raise RuntimeError(f"N19 tile geometry mismatch: {tile_ids}")
    tile_heads = [_head_tile(tile_id) for tile_id in tile_ids]
    direction_checks = {
        "u0_vminus1": _from_direction(0.0, -1.0),
        "u1_v0": _from_direction(1.0, 0.0),
        "zero_vector_is_nan": math.isnan(_from_direction(0.0, 0.0)),
        "bin_2p5_tie_clockwise": _direction_bin(2.5),
        "bin_357p5_wrap": _direction_bin(357.5),
    }
    if direction_checks != {
        "u0_vminus1": 0.0,
        "u1_v0": 270.0,
        "zero_vector_is_nan": True,
        "bin_2p5_tie_clockwise": 5,
        "bin_357p5_wrap": 0,
    }:
        raise RuntimeError(f"N19 direction convention failed: {direction_checks}")
    parity = _parity_audit()
    synthetic_tiff = _synthetic_tiff_audit()
    lock = tomllib.loads((repo / "uv.lock").read_text())
    locked_pillow_versions = sorted(
        package["version"]
        for package in lock["package"]
        if package.get("name", "").lower() == "pillow"
    )
    synthetic_tiff["uv_lock_pillow_versions"] = locked_pillow_versions
    if synthetic_tiff["pillow_version"] != "12.3.0":
        raise RuntimeError("N19 Pillow version changed")
    if locked_pillow_versions != ["12.3.0"]:
        raise RuntimeError("N19 uv.lock Pillow version changed")
    if not synthetic_tiff["libtiff"] or not synthetic_tiff["zlib"]:
        raise RuntimeError("N19 Pillow TIFF codecs unavailable")
    lineage = _lineage_audit(repo, lineage_path)
    result = {
        "schema_version": 1,
        "node_id": frozen["node_id"],
        "predeclaration_sha256": _sha256(predeclaration),
        "input_bundle_sha256": frozen["input_bundle"]["sha256"],
        "formula_manifest": {
            "path": str(formula_path.relative_to(repo)),
            "sha256": _sha256(formula_path),
            "dmax_m": formula["parameters"]["dmax_m"],
            "height_m": formula["parameters"]["height_m"],
            "direction_bins": len(formula["parameters"]["direction_bins_deg"]),
            "runtime_columns": len(formula["parameters"]["runtime_columns"]),
            "extra_variants": formula["parameters"]["extra_variants_allowed"],
        },
        "source_and_schema": {
            "competition_archive_sha256": _sha256(
                repo / "inputs/competition/open_wind_236727.zip"
            ),
            "ldaps_grid_count": len(coordinates),
            "coordinates": [
                {"grid_id": index, "latitude": latitude, "longitude": longitude}
                for index, (latitude, longitude) in enumerate(coordinates, start=1)
            ],
            "unique_finite_coordinates": True,
            "required_direction_headers": required_headers,
            "weather_rows_read": 0,
        },
        "tile_geometry": {
            "radius_m": formula["parameters"]["dmax_m"],
            "tile_ids": tile_ids,
            "heads": tile_heads,
            "request_methods": ["HEAD"] * len(tile_heads),
            "dem_body_bytes": 0,
        },
        "direction_checks": direction_checks,
        "independent_formula_parity": parity,
        "synthetic_geotiff_decoder": synthetic_tiff,
        "nonduplication": lineage,
        "gates": {
            "source_and_schema": True,
            "tile_head": True,
            "formula_and_direction": True,
            "independent_parity": True,
            "synthetic_decoder": True,
            "nonduplication": True,
        },
        "verdict": "READY_STATIC_LOOKUP_EXTRACTION",
        "handoff": ["S17-N20_TERRAIN_SX300_H8_STATIC_LOOKUP_EXTRACTION"],
        "actions": {
            "external_requests": len(tile_heads),
            "external_request_methods": ["HEAD"] * len(tile_heads),
            "dem_body_bytes": 0,
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
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--predeclaration",
        type=Path,
        default=Path("reports/s17_n19_terrain_sx300_no_body_predeclaration.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/audits/s17_n19_terrain_sx300_no_body.json"),
    )
    args = parser.parse_args()
    repo = args.repo.resolve()
    predeclaration = args.predeclaration
    output = args.output
    if not predeclaration.is_absolute():
        predeclaration = repo / predeclaration
    if not output.is_absolute():
        output = repo / output
    result = run(repo, predeclaration, output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
