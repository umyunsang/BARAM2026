"""Screen graph-spectral NWP grid modes with a strict source-rank classifier.

The GFS grid is regular while the supplied LDAPS points form an irregular
diamond, so a rectangular DCT would impose a false topology.  This runner
instead derives a deterministic graph-Laplacian basis from the official grid
coordinates and projects every supplied NWP field onto its six lowest modes.
All supervised operations use only complete issuance batches before Q3.
"""

from __future__ import annotations

import argparse
import csv
import gc
import io
import json
import re
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from run_inner_policy_classifier import _group_total, _policy_values
from run_sequence_classifier import (
    BASELINE,
    BASELINE_SHA,
    CAPACITIES,
    OPEN,
    OPEN_SHA,
    OUTPUT,
    _score,
    _sha256,
)
from run_site_wind_teacher import _strict_preceding_mask, _validation_mask
from run_strict_prequential_source_rank import (
    FROZEN_MIXTURE,
    FROZEN_POLICY,
    _mixture,
    _source_columns,
    _source_probability,
)
from strict_dev_surface import DEV_CUTOFF, development_surface

BASE_COLUMNS = ["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh"]
CLASS_WIDTH = 0.02
GRID_PATTERN = re.compile(r"^(gfs|ldaps)__grid(\d+)__(.+)$")
GRAPH_MODES = 6
PARENT_PATH = OUTPUT / "M189_STRICT_PREQUENTIAL_SOURCE_RANK_Q3-dev-2023-Q3.parquet"


def _grid_coordinates(member: str, expected: int) -> pd.DataFrame:
    records: list[dict[str, float | int]] = []
    with zipfile.ZipFile(OPEN) as archive, archive.open(member) as binary:
        text = io.TextIOWrapper(binary, encoding="utf-8-sig", newline="")
        reader = csv.DictReader(text)
        first_time: str | None = None
        for row in reader:
            timestamp = str(row["forecast_kst_dtm"])
            if first_time is None:
                first_time = timestamp
            elif timestamp != first_time:
                break
            records.append(
                {
                    "grid_id": int(row["grid_id"]),
                    "latitude": float(row["latitude"]),
                    "longitude": float(row["longitude"]),
                }
            )
    frame = pd.DataFrame(records).sort_values("grid_id").reset_index(drop=True)
    if len(frame) != expected or frame["grid_id"].tolist() != list(
        range(1, expected + 1)
    ):
        raise RuntimeError(f"{member} coordinate contract changed")
    return frame


def _graph_basis(coordinates: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    latitude = coordinates["latitude"].to_numpy(dtype=float)
    longitude = coordinates["longitude"].to_numpy(dtype=float)
    center_latitude = float(latitude.mean())
    xy = np.column_stack(
        [
            (longitude - longitude.mean())
            * 111.32
            * np.cos(np.deg2rad(center_latitude)),
            (latitude - latitude.mean()) * 111.32,
        ]
    )
    delta = xy[:, None, :] - xy[None, :, :]
    distance = np.sqrt(np.sum(delta**2, axis=2))
    positive = distance[distance > 0]
    bandwidth = float(np.median(np.min(np.where(distance > 0, distance, np.inf), axis=1)))
    weights = np.exp(-(distance**2) / (2.0 * bandwidth**2))
    weights[distance > 2.6 * bandwidth] = 0.0
    np.fill_diagonal(weights, 0.0)
    degree = weights.sum(axis=1)
    if np.any(degree <= 0.0) or positive.size == 0:
        raise RuntimeError("weather grid graph is disconnected at a vertex")
    inverse = 1.0 / np.sqrt(degree)
    laplacian = np.eye(len(weights)) - inverse[:, None] * weights * inverse[None, :]
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    order = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    for index in range(eigenvectors.shape[1]):
        anchor = int(np.argmax(np.abs(eigenvectors[:, index])))
        if eigenvectors[anchor, index] < 0.0:
            eigenvectors[:, index] *= -1.0
    return eigenvectors.astype("float32"), eigenvalues.astype("float32")


def _spatial_modes(
    surface: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, dict[str, object]]:
    grouped: dict[tuple[str, str], list[tuple[int, str]]] = defaultdict(list)
    for name in feature_columns:
        match = GRID_PATTERN.match(name)
        if match:
            grouped[(match.group(1), match.group(3))].append(
                (int(match.group(2)), name)
            )
    counts = {source: sum(key[0] == source for key in grouped) for source in ("gfs", "ldaps")}
    if counts != {"gfs": 41, "ldaps": 34}:
        raise RuntimeError(f"raw grid variable contract changed: {counts}")
    coordinates = {
        "gfs": _grid_coordinates("train/gfs_train.csv", 9),
        "ldaps": _grid_coordinates("train/ldaps_train.csv", 16),
    }
    bases: dict[str, np.ndarray] = {}
    eigenvalues: dict[str, list[float]] = {}
    for source, frame in coordinates.items():
        basis, spectrum = _graph_basis(frame)
        bases[source] = basis
        eigenvalues[source] = [float(value) for value in spectrum[:GRAPH_MODES]]

    generated: dict[str, np.ndarray] = {}
    for (source, variable), columns in sorted(grouped.items()):
        ordered = sorted(columns)
        expected = 9 if source == "gfs" else 16
        if [grid_id for grid_id, _ in ordered] != list(range(1, expected + 1)):
            raise RuntimeError(f"incomplete {source} grid for {variable}")
        values = surface[[name for _, name in ordered]].to_numpy(dtype="float32")
        row_mean = np.nanmean(values, axis=1)
        missing = np.where(~np.isfinite(values))
        if missing[0].size:
            values[missing] = row_mean[missing[0]]
        if not np.isfinite(values).all():
            raise RuntimeError(f"non-finite spatial field: {source}/{variable}")
        basis = bases[source]
        coefficient = values @ basis[:, :GRAPH_MODES]
        reconstruction = coefficient @ basis[:, :GRAPH_MODES].T
        residual = np.sqrt(np.mean((values - reconstruction) ** 2, axis=1))
        for mode in range(GRAPH_MODES):
            generated[f"graph__{source}__{variable}__mode{mode}"] = coefficient[
                :, mode
            ].astype("float32")
        generated[f"graph__{source}__{variable}__residual"] = residual.astype(
            "float32"
        )
    frame = pd.DataFrame(generated, index=surface.index)
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise RuntimeError("graph-spectral context contains non-finite values")
    diagnostics: dict[str, object] = {
        "variable_counts": counts,
        "mode_count": GRAPH_MODES,
        "generated_feature_count": frame.shape[1],
        "low_eigenvalues": eigenvalues,
    }
    return frame, diagnostics


def _frame(
    surface: pd.DataFrame,
    validation: np.ndarray,
    normalized: np.ndarray,
) -> pd.DataFrame:
    output = surface.loc[validation, BASE_COLUMNS].copy()
    output["prediction_kwh"] = normalized * output["group_id"].map(
        CAPACITIES
    ).to_numpy(dtype=float)
    return output


def _group_scores(output: pd.DataFrame) -> dict[str, dict[str, float]]:
    scores: dict[str, dict[str, float]] = {}
    for group_id, capacity in CAPACITIES.items():
        group = output.loc[output["group_id"].eq(group_id)]
        scores[str(group_id)] = _group_total(
            group["actual_kwh"].to_numpy(dtype=float) / capacity,
            group["prediction_kwh"].to_numpy(dtype=float) / capacity,
        )
    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--fold", choices=("dev-2023-Q3",), required=True)
    parser.add_argument("--iterations", type=int, default=150)
    parser.add_argument("--global-features", type=int, default=120)
    parser.add_argument("--source-features", type=int, default=180)
    args = parser.parse_args()
    if not 100 <= args.iterations <= 240:
        raise ValueError("iterations must be between 100 and 240")
    if not 80 <= args.global_features <= 180:
        raise ValueError("global feature count must be between 80 and 180")
    if not 120 <= args.source_features <= 240:
        raise ValueError("source feature count must be between 120 and 240")
    if _sha256(OPEN) != OPEN_SHA or _sha256(BASELINE) != BASELINE_SHA:
        raise RuntimeError("immutable competition input hash mismatch")

    started = time.perf_counter()
    surface, feature_columns = development_surface()
    if surface["forecast_kst_dtm"].ge(DEV_CUTOFF).any():
        raise RuntimeError("lockbox row reached graph-spectrum runner")
    validation = _validation_mask(surface, args.fold)
    history = _strict_preceding_mask(surface, validation)
    target = surface["actual_kwh"] / surface["group_id"].map(CAPACITIES)
    training = (
        history
        & surface["actual_kwh"].notna().to_numpy()
        & target.ge(0.10).to_numpy()
    )
    spectral, graph_diagnostics = _spatial_modes(surface, feature_columns)
    matrix = pd.concat([surface[feature_columns].astype("float32"), spectral], axis=1)
    del spectral
    gc.collect()

    raw_bins = np.floor(
        (target.clip(0.10, 1.074999) - 0.10) / CLASS_WIDTH
    ).astype("Int64")
    active_bins = [
        int(value)
        for value in sorted(raw_bins.loc[training].dropna().astype(int).unique())
    ]
    bin_to_class = {bin_id: index for index, bin_id in enumerate(active_bins)}
    classes = raw_bins.map(bin_to_class).astype("Int64")
    centers = np.asarray(
        [
            target.loc[training & classes.eq(index)].mean()
            for index in range(len(active_bins))
        ],
        dtype=float,
    )

    candidates = list(matrix.columns)
    source_specs = {
        "global": (candidates, args.global_features),
        "gfs": (_source_columns(candidates, "gfs"), args.source_features),
        "ldaps": (_source_columns(candidates, "ldaps"), args.source_features),
    }
    probabilities: dict[str, np.ndarray] = {}
    selected_features: dict[str, list[str]] = {}
    for source_index, (source, (source_candidates, count)) in enumerate(
        source_specs.items()
    ):
        probability, selected = _source_probability(
            matrix,
            target,
            classes,
            training,
            validation,
            source_candidates,
            count,
            args.iterations,
            len(active_bins),
            20261200 + source_index * 10,
            source,
        )
        probabilities[source] = probability
        selected_features[source] = selected
    del matrix
    gc.collect()

    groups = surface.loc[validation, "group_id"].to_numpy(dtype=int)
    means = {
        group_id: float(
            target.loc[training & surface["group_id"].eq(group_id).to_numpy()].mean()
        )
        for group_id in CAPACITIES
    }
    probability = _mixture(probabilities, *FROZEN_MIXTURE)
    normalized = _policy_values(probability, centers, groups, means)[FROZEN_POLICY]
    graph_output = _frame(surface, validation, normalized)

    parent = pd.read_parquet(PARENT_PATH)
    expected_keys = pd.MultiIndex.from_frame(
        surface.loc[validation, ["forecast_id", "group_id"]]
    )
    parent_keys = pd.MultiIndex.from_frame(parent[["forecast_id", "group_id"]])
    if len(parent) != int(validation.sum()) or not parent_keys.equals(expected_keys):
        raise RuntimeError("M189 parent key contract changed")
    parent_normalized = parent["prediction_kwh"].to_numpy(dtype=float) / parent[
        "group_id"
    ].map(CAPACITIES).to_numpy(dtype=float)
    blend_output = _frame(surface, validation, 0.5 * normalized + 0.5 * parent_normalized)

    output = graph_output.assign(fold_id=args.fold, model_id=args.candidate_id)
    output_path = OUTPUT / f"{args.candidate_id}-{args.fold}.parquet"
    output.to_parquet(output_path, index=False)
    receipt = {
        "schema_version": 1,
        "candidate_id": args.candidate_id,
        "fold_id": args.fold,
        "architecture": "strict_graph_laplacian_spatial_modes_source_rank",
        "scope": (
            "fixed official-data-only graph representation screen; outer Q3 labels "
            "excluded from graph construction, feature selection, and fit"
        ),
        "physical_development_cutoff": str(DEV_CUTOFF),
        "strict_training_rows": int(training.sum()),
        "class_width": CLASS_WIDTH,
        "class_count": len(active_bins),
        "iterations": args.iterations,
        "graph_diagnostics": graph_diagnostics,
        "selected_features": selected_features,
        "selected_graph_feature_counts": {
            source: sum(name.startswith("graph__") for name in names)
            for source, names in selected_features.items()
        },
        "frozen_mixture": {
            "global_weight": FROZEN_MIXTURE[0],
            "gfs_share": FROZEN_MIXTURE[1],
        },
        "frozen_policy": FROZEN_POLICY,
        "fold_score": _score(graph_output),
        "group_scores": _group_scores(graph_output),
        "fixed_half_m189_blend_score": _score(blend_output),
        "fixed_half_m189_group_scores": _group_scores(blend_output),
        "runtime_seconds": round(time.perf_counter() - started, 2),
        "prediction_path": str(output_path.relative_to(Path.cwd())),
        "prediction_sha256": _sha256(output_path),
        "parent_path": str(PARENT_PATH.relative_to(Path.cwd())),
        "parent_sha256": _sha256(PARENT_PATH),
        "observed_scada_feature_count": 0,
        "new_2024_evaluation": False,
        "lockbox_reopened": False,
        "external_actions": [],
    }
    receipt_path = OUTPUT / f"{args.candidate_id}-{args.fold}.json"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "candidate_id": args.candidate_id,
                "fold_score": receipt["fold_score"],
                "group_scores": receipt["group_scores"],
                "fixed_half_m189_blend_score": receipt[
                    "fixed_half_m189_blend_score"
                ],
                "selected_graph_feature_counts": receipt[
                    "selected_graph_feature_counts"
                ],
                "runtime_seconds": receipt["runtime_seconds"],
                "prediction_sha256": receipt["prediction_sha256"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
