"""Local-only typed recovery of the S17-N20 terrain Sx lookup."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

WGS84_A_M = 6_378_137.0
WGS84_E2 = 6.6943799901413165e-3


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def verify_inputs(repo: Path, predeclaration: Path) -> dict[str, Any]:
    frozen = json.loads(predeclaration.read_text())
    observed = {
        relative: sha256_path(repo / relative)
        for relative in frozen["input_bundle"]["files"]
    }
    if observed != frozen["input_bundle"]["files"]:
        raise RuntimeError("N20A input hash mismatch")
    if canonical_hash(observed) != frozen["input_bundle"]["sha256"]:
        raise RuntimeError("N20A input bundle mismatch")
    body = frozen["immutable_body"]
    body_path = repo / body["path"]
    if sha256_path(body_path) != body["sha256"]:
        raise RuntimeError("N20A immutable body hash mismatch")
    if body_path.stat().st_size != body["bytes"]:
        raise RuntimeError("N20A immutable body byte mismatch")
    return frozen


def radii(latitude_deg: float) -> tuple[float, float]:
    phi = math.radians(latitude_deg)
    denominator = math.sqrt(1.0 - WGS84_E2 * math.sin(phi) ** 2)
    prime_vertical = WGS84_A_M / denominator
    meridional = (
        WGS84_A_M
        * (1.0 - WGS84_E2)
        / (1.0 - WGS84_E2 * math.sin(phi) ** 2) ** 1.5
    )
    return meridional, prime_vertical


def endpoint_lon_lat(
    latitude: float,
    longitude: float,
    bearing_deg: float,
    distance_m: float,
) -> tuple[float, float]:
    meridional, prime_vertical = radii(latitude)
    bearing = math.radians(bearing_deg)
    east_m = distance_m * math.sin(bearing)
    north_m = distance_m * math.cos(bearing)
    return (
        longitude
        + math.degrees(
            east_m / (prime_vertical * math.cos(math.radians(latitude)))
        ),
        latitude + math.degrees(north_m / meridional),
    )


def cells_at_point(value: float, tolerance: float = 1e-12) -> set[int]:
    rounded = round(value)
    if abs(value - rounded) <= tolerance:
        return {int(rounded) - 1, int(rounded)}
    return {math.floor(value)}


def supercover_breakpoints(
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
    result: set[tuple[int, int]] = set()
    for position in probes:
        x_value = x0 + position * dx
        y_value = y0 + position * dy
        result.update(
            (x_cell, y_cell)
            for x_cell in cells_at_point(x_value)
            for y_cell in cells_at_point(y_value)
        )
    return result


def segment_intersects_cell(
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
    for direction, value in (
        (-dx, x0 - x_cell),
        (dx, x_cell + 1.0 - x0),
        (-dy, y0 - y_cell),
        (dy, y_cell + 1.0 - y0),
    ):
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


def supercover_reference(
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
        if segment_intersects_cell(start, end, (x_cell, y_cell))
    }


def grid_coordinate(
    latitude: float,
    longitude: float,
    *,
    tie_longitude: float,
    tie_latitude: float,
    scale_x: float,
    scale_y: float,
) -> tuple[float, float]:
    return (
        (longitude - tie_longitude) / scale_x + 0.5,
        (tie_latitude - latitude) / scale_y + 0.5,
    )


def pixel_center_lon_lat(
    x_cell: int,
    y_cell: int,
    *,
    tie_longitude: float,
    tie_latitude: float,
    scale_x: float,
    scale_y: float,
) -> tuple[float, float]:
    return (
        tie_longitude + x_cell * scale_x,
        tie_latitude - y_cell * scale_y,
    )


def local_distance(
    query_latitude: float,
    query_longitude: float,
    pixel_latitude: float,
    pixel_longitude: float,
) -> float:
    meridional, prime_vertical = radii(query_latitude)
    east_m = math.radians(pixel_longitude - query_longitude) * prime_vertical * math.cos(
        math.radians(query_latitude)
    )
    north_m = math.radians(pixel_latitude - query_latitude) * meridional
    return math.hypot(east_m, north_m)


def build_geometry(
    coordinates: list[tuple[float, float]],
    directions: list[int],
    offsets: list[int],
    *,
    tie_longitude: float,
    tie_latitude: float,
    scale_x: float,
    scale_y: float,
    dmax_m: float,
) -> tuple[
    list[list[list[set[tuple[int, int]]]]],
    list[list[list[set[tuple[int, int]]]]],
    list[tuple[float, float]],
]:
    primary_all = []
    reference_all = []
    query_points = []
    for latitude, longitude in coordinates:
        query = grid_coordinate(
            latitude,
            longitude,
            tie_longitude=tie_longitude,
            tie_latitude=tie_latitude,
            scale_x=scale_x,
            scale_y=scale_y,
        )
        query_points.append(query)
        primary_directions = []
        reference_directions = []
        for direction in directions:
            primary_rays = []
            reference_rays = []
            for offset in offsets:
                endpoint_longitude, endpoint_latitude = endpoint_lon_lat(
                    latitude,
                    longitude,
                    (direction + offset) % 360,
                    dmax_m,
                )
                endpoint = grid_coordinate(
                    endpoint_latitude,
                    endpoint_longitude,
                    tie_longitude=tie_longitude,
                    tie_latitude=tie_latitude,
                    scale_x=scale_x,
                    scale_y=scale_y,
                )
                primary_rays.append(supercover_breakpoints(query, endpoint))
                reference_rays.append(supercover_reference(query, endpoint))
            primary_directions.append(primary_rays)
            reference_directions.append(reference_rays)
        primary_all.append(primary_directions)
        reference_all.append(reference_directions)
    return primary_all, reference_all, query_points


def sx_lookup(
    values: np.ndarray,
    coordinates: list[tuple[float, float]],
    query_points: list[tuple[float, float]],
    geometry: list[list[list[set[tuple[int, int]]]]],
    *,
    crop_left: int,
    crop_top: int,
    tie_longitude: float,
    tie_latitude: float,
    scale_x: float,
    scale_y: float,
    height_m: float,
) -> np.ndarray:
    lookup = np.empty((len(coordinates), len(geometry[0])), dtype=np.float64)
    for grid_index, ((latitude, longitude), query) in enumerate(
        zip(coordinates, query_points, strict=True)
    ):
        query_cell = (math.floor(query[0]), math.floor(query[1]))
        query_elevation = float(
            values[query_cell[1] - crop_top, query_cell[0] - crop_left]
        )
        for direction_index, rays in enumerate(geometry[grid_index]):
            ray_maxima = []
            for cells in rays:
                angles = []
                for x_cell, y_cell in cells:
                    if (x_cell, y_cell) == query_cell:
                        continue
                    pixel_longitude, pixel_latitude = pixel_center_lon_lat(
                        x_cell,
                        y_cell,
                        tie_longitude=tie_longitude,
                        tie_latitude=tie_latitude,
                        scale_x=scale_x,
                        scale_y=scale_y,
                    )
                    distance = local_distance(
                        latitude,
                        longitude,
                        pixel_latitude,
                        pixel_longitude,
                    )
                    if distance <= 0.0:
                        continue
                    elevation = float(values[y_cell - crop_top, x_cell - crop_left])
                    rise = elevation - (query_elevation + height_m)
                    angles.append(math.degrees(math.atan2(rise, distance)))
                if not angles:
                    raise RuntimeError("N20A ray has no noncentral candidate")
                ray_maxima.append(max(angles))
            lookup[grid_index, direction_index] = float(np.mean(ray_maxima))
    return lookup


def write_lookup_artifacts(
    outdir: Path, lookup: np.ndarray, directions: list[int]
) -> dict[str, Any]:
    npy_path = outdir / "lookup.npy"
    parquet_path = outdir / "lookup.parquet"
    np.save(npy_path, lookup, allow_pickle=False)
    grid_ids = np.repeat(np.arange(1, lookup.shape[0] + 1, dtype=np.int16), len(directions))
    direction_values = np.tile(np.asarray(directions, dtype=np.int16), lookup.shape[0])
    sx_values = lookup.reshape(-1)
    table = pa.table(
        {
            "grid_id": pa.array(grid_ids, type=pa.int16()),
            "direction_bin_deg": pa.array(direction_values, type=pa.int16()),
            "sx_deg": pa.array(sx_values, type=pa.float64()),
        }
    )
    pq.write_table(
        table,
        parquet_path,
        compression="zstd",
        version="2.6",
        write_statistics=True,
    )
    npy_roundtrip = np.load(npy_path, allow_pickle=False)
    parquet_roundtrip = pq.read_table(parquet_path).column("sx_deg").to_numpy()
    if not np.array_equal(npy_roundtrip, lookup):
        raise RuntimeError("N20A NPY roundtrip mismatch")
    if not np.array_equal(parquet_roundtrip, lookup.reshape(-1)):
        raise RuntimeError("N20A Parquet roundtrip mismatch")
    return {
        "npy": {
            "path": str(npy_path),
            "sha256": sha256_path(npy_path),
            "bytes": npy_path.stat().st_size,
            "roundtrip_exact": True,
        },
        "parquet": {
            "path": str(parquet_path),
            "sha256": sha256_path(parquet_path),
            "bytes": parquet_path.stat().st_size,
            "roundtrip_exact": True,
            "rows": table.num_rows,
            "schema": str(table.schema),
        },
    }


def run(repo: Path, predeclaration: Path) -> dict[str, Any]:
    frozen = verify_inputs(repo, predeclaration)
    body_path = repo / frozen["immutable_body"]["path"]
    formula_path = repo / frozen["extraction_contract"]["formula_path"]
    if sha256_path(formula_path) != frozen["extraction_contract"]["formula_sha256"]:
        raise RuntimeError("N20A formula hash mismatch")
    formula = json.loads(formula_path.read_text())
    n20_manifest = json.loads(
        (body_path.parent / "manifest.json").read_text()
    )
    metadata = n20_manifest["decode_metadata"]
    if metadata["missing_tags"] != [42113]:
        raise RuntimeError("N20A typed recovery premise changed")
    if (
        metadata["width"],
        metadata["height"],
        metadata["model_type"],
        metadata["raster_type"],
        metadata["geographic_epsg"],
    ) != (3600, 3600, 2, 2, 4326):
        raise RuntimeError("N20A raster metadata changed")
    scale_x, scale_y, _ = metadata["pixel_scale"]
    tie = metadata["tiepoint"]
    tie_longitude = float(tie[3])
    tie_latitude = float(tie[4])
    n14 = json.loads(
        (repo / "artifacts/audits/s17_n14_r3_terrain_prerequisite.json").read_text()
    )
    grids = n14["ldaps_model_orography"]["grids"]
    coordinates = [
        (float(record["latitude_min"]), float(record["longitude_min"]))
        for _, record in sorted(grids.items(), key=lambda item: int(item[0]))
    ]
    directions = [int(value) for value in formula["parameters"]["direction_bins_deg"]]
    offsets = [int(value) for value in formula["parameters"]["ray_offsets_deg"]]
    if directions != list(range(0, 360, 5)) or offsets != [-15, -10, -5, 0, 5, 10, 15]:
        raise RuntimeError("N20A formula directions changed")
    primary, reference, query_points = build_geometry(
        coordinates,
        directions,
        offsets,
        tie_longitude=tie_longitude,
        tie_latitude=tie_latitude,
        scale_x=scale_x,
        scale_y=scale_y,
        dmax_m=float(formula["parameters"]["dmax_m"]),
    )
    ray_count = 0
    mismatched_rays = []
    all_cells: set[tuple[int, int]] = set()
    for grid_index in range(len(coordinates)):
        all_cells.add(
            (math.floor(query_points[grid_index][0]), math.floor(query_points[grid_index][1]))
        )
        for direction_index in range(len(directions)):
            for ray_index in range(len(offsets)):
                ray_count += 1
                primary_cells = primary[grid_index][direction_index][ray_index]
                reference_cells = reference[grid_index][direction_index][ray_index]
                if primary_cells != reference_cells:
                    mismatched_rays.append(
                        [grid_index + 1, directions[direction_index], offsets[ray_index]]
                    )
                all_cells.update(primary_cells)
                all_cells.update(reference_cells)
    if ray_count != 8064 or mismatched_rays:
        raise RuntimeError(f"N20A geometry parity failed: {mismatched_rays[:3]}")
    if any(
        x_cell < 0 or y_cell < 0 or x_cell >= metadata["width"] or y_cell >= metadata["height"]
        for x_cell, y_cell in all_cells
    ):
        raise RuntimeError("N20A candidate outside COG coverage")
    crop_left = min(cell[0] for cell in all_cells)
    crop_right = max(cell[0] for cell in all_cells) + 1
    crop_top = min(cell[1] for cell in all_cells)
    crop_bottom = max(cell[1] for cell in all_cells) + 1
    crop_pixels = (crop_right - crop_left) * (crop_bottom - crop_top)
    if crop_pixels > frozen["extraction_contract"]["maximum_loaded_crop_pixels"]:
        raise RuntimeError("N20A crop pixel cap exceeded")
    with Image.open(body_path) as image:
        image.seek(0)
        crop_values = np.asarray(
            image.crop((crop_left, crop_top, crop_right, crop_bottom)),
            dtype=np.float32,
        )
    accessed_values = np.asarray(
        [
            crop_values[y_cell - crop_top, x_cell - crop_left]
            for x_cell, y_cell in sorted(all_cells)
        ],
        dtype=np.float32,
    )
    finite = bool(np.isfinite(accessed_values).all())
    plausible = bool(
        (accessed_values >= -500.0).all() and (accessed_values <= 9000.0).all()
    )
    if not finite or not plausible:
        raise RuntimeError("N20A explicit nodata/value gate failed")
    primary_lookup = sx_lookup(
        crop_values,
        coordinates,
        query_points,
        primary,
        crop_left=crop_left,
        crop_top=crop_top,
        tie_longitude=tie_longitude,
        tie_latitude=tie_latitude,
        scale_x=scale_x,
        scale_y=scale_y,
        height_m=float(formula["parameters"]["height_m"]),
    )
    reference_lookup = sx_lookup(
        crop_values,
        coordinates,
        query_points,
        reference,
        crop_left=crop_left,
        crop_top=crop_top,
        tie_longitude=tie_longitude,
        tie_latitude=tie_latitude,
        scale_x=scale_x,
        scale_y=scale_y,
        height_m=float(formula["parameters"]["height_m"]),
    )
    maximum_error = float(np.max(np.abs(primary_lookup - reference_lookup)))
    if maximum_error > 1e-12 or not np.array_equal(primary_lookup, reference_lookup):
        raise RuntimeError("N20A lookup parity failed")
    if primary_lookup.shape != (16, 72) or not np.isfinite(primary_lookup).all():
        raise RuntimeError("N20A lookup shape/finiteness failed")
    output_artifacts = write_lookup_artifacts(body_path.parent, primary_lookup, directions)
    for artifact in output_artifacts.values():
        artifact["path"] = str(Path(artifact["path"]).relative_to(repo))
    result = {
        "schema_version": 1,
        "node_id": frozen["node_id"],
        "predeclaration_sha256": sha256_path(predeclaration),
        "input_bundle_sha256": frozen["input_bundle"]["sha256"],
        "body": {
            "path": frozen["immutable_body"]["path"],
            "sha256": sha256_path(body_path),
            "bytes": body_path.stat().st_size,
            "new_external_requests": 0,
        },
        "typed_recovery": frozen["typed_recovery"],
        "raster": {
            "frame": 0,
            "size": [metadata["width"], metadata["height"]],
            "epsg": metadata["geographic_epsg"],
            "pixel_is_point": metadata["raster_type"] == 2,
            "pixel_scale": [scale_x, scale_y],
            "tie_longitude": tie_longitude,
            "tie_latitude": tie_latitude,
            "crop": [crop_left, crop_top, crop_right, crop_bottom],
            "crop_shape": list(crop_values.shape),
            "crop_pixels": crop_pixels,
            "unique_accessed_cells": len(all_cells),
            "accessed_min_elevation_m": float(np.min(accessed_values)),
            "accessed_max_elevation_m": float(np.max(accessed_values)),
            "accessed_all_finite": finite,
            "accessed_all_plausible": plausible,
            "fills_or_replacements": 0,
        },
        "geometry_parity": {
            "rays": ray_count,
            "mismatched_cell_sets": len(mismatched_rays),
            "lookup_max_abs_error_deg": maximum_error,
            "lookup_array_equal": bool(np.array_equal(primary_lookup, reference_lookup)),
        },
        "lookup": {
            "shape": list(primary_lookup.shape),
            "dtype": str(primary_lookup.dtype),
            "finite_values": int(np.isfinite(primary_lookup).sum()),
            "min_sx_deg": float(np.min(primary_lookup)),
            "max_sx_deg": float(np.max(primary_lookup)),
            "mean_sx_deg": float(np.mean(primary_lookup)),
            "direction_order_deg": directions,
            "grid_ids": list(range(1, 17)),
            "artifacts": output_artifacts,
        },
        "gates": {
            "inputs": True,
            "metadata_only_42113_missing": True,
            "coverage": True,
            "geometry_parity": True,
            "nodata_finite_plausible": True,
            "lookup": True,
            "roundtrip": True,
        },
        "verdict": "READY_TERRAIN_MODEL_FAMILY_PREREQUISITE_AUDIT",
        "handoff": ["S17-N21_TERRAIN_SX300_H8_MODEL_FAMILY_PREREQUISITE_AUDIT"],
        "actions": {
            "external_requests": 0,
            "dem_body_bytes_new": 0,
            "pixel_values_loaded": crop_pixels,
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
    manifest_path = body_path.parent / "recovery_manifest.json"
    manifest_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--predeclaration",
        type=Path,
        default=Path("reports/s17_n20a_terrain_sx300_no_nodata_recovery_predeclaration.json"),
    )
    args = parser.parse_args()
    repo = args.repo.resolve()
    predeclaration = args.predeclaration
    if not predeclaration.is_absolute():
        predeclaration = repo / predeclaration
    print(json.dumps(run(repo, predeclaration), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
