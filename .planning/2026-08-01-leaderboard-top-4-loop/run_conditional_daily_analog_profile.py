"""Screen a chronology-safe daily analog-profile correction on the v2 lineage.

The model retrieves complete historical 24-hour target profiles using only the
competition-supplied NWP trajectory that was available at issuance time.  All
candidate choices are made on the v2 Q3 OOF surface and then frozen for a Q4
transfer check.  The consumed 2024 lockbox is never materialized.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from build_v2_transfer_sequence_challenger import (
    CAPACITIES,
    EXPECTED_SELECTIONS,
    METRIC_COLUMNS,
    OOF,
    _apply_long,
    _group_score,
    _paired_issuance_bootstrap,
    _score,
)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from strict_dev_surface import DEV_CUTOFF, development_surface

from baram.contracts.hashing import canonical_sha256, sha256_file

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "artifacts" / "backtests" / "metric-aligned-probe"
OPEN = Path("/Users/um-yunsang/Downloads/open.zip")
BASELINE = Path("/Users/um-yunsang/Downloads/baseline.ipynb")
OPEN_SHA = "920be0c458d820e855bf79dd25723146f52ce1736138aedba5e6bc853f1f720b"
BASELINE_SHA = "712b26f4d2748860c94cff1e0100c23810468c983173f8e9ef8d009fe82df48c"
CANDIDATE_ID = "M232_STRICT_DAILY_ANALOG_PROFILE_Q3"
NEIGHBOR_COUNTS = (5, 10, 20, 40, 80)
KERNELS = ("uniform", "inverse", "exponential")
HEADS = ("mean", "median", "utility")
TRANSFORMS = ("level", "shape", "scaled")
BLEND_WEIGHTS = tuple(float(value) for value in np.arange(0.025, 0.501, 0.025))
ACTIONS = np.arange(0.075, 1.076, 0.005)


@dataclass(frozen=True)
class Representation:
    feature_set: str
    mode: str
    components: int
    season_weight: float

    @property
    def name(self) -> str:
        season = str(self.season_weight).replace(".", "p")
        return (
            f"{self.feature_set}_{self.mode}_pca{self.components}_season{season}"
        )


@dataclass(frozen=True)
class Recipe:
    representation: str
    neighbors: int
    kernel: str
    head: str
    transform: str
    blend_weight: float


REPRESENTATIONS = (
    Representation("core", "raw", 16, 0.0),
    Representation("core", "raw_delta", 24, 2.5),
    Representation("extended", "raw", 24, 2.5),
)


def _feature_sets(columns: list[str]) -> dict[str, list[str]]:
    core = [
        name
        for name in columns
        if (
            "spatial__idw__wind" in name
            or name in {
                "phys__hub117_speed",
                "phys_v2__hub117_speed",
                "source_disagreement__wind10_speed_idw",
                "source_disagreement__wind10_speed_idw__abs",
            }
        )
    ]
    extended = [
        name
        for name in columns
        if (
            name in core
            or (
                name.startswith(("gfs__", "ldaps__"))
                and "wind" in name.lower()
                and name.endswith(("__mean", "__std", "__q50", "__q90"))
            )
        )
    ]
    if len(core) < 20 or len(extended) < 80:
        raise RuntimeError(
            f"daily analog feature contract changed: core={len(core)}, "
            f"extended={len(extended)}"
        )
    return {"core": core, "extended": extended}


def _complete_group_days(
    surface: pd.DataFrame,
    group_id: int,
    feature_names: list[str],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    frame = surface.loc[surface["group_id"].eq(group_id)].copy()
    frame = frame.sort_values(
        ["data_available_kst_dtm", "forecast_kst_dtm"]
    ).reset_index(drop=True)
    sizes = frame.groupby("data_available_kst_dtm", sort=True).size()
    complete_issuances = sizes.loc[sizes.eq(24)].index
    frame = frame.loc[
        frame["data_available_kst_dtm"].isin(complete_issuances)
    ].reset_index(drop=True)
    sizes = frame.groupby("data_available_kst_dtm", sort=True).size()
    if frame.empty or not sizes.eq(24).all():
        raise RuntimeError(f"group {group_id} daily topology is incomplete")
    issuances = frame["data_available_kst_dtm"].drop_duplicates().to_numpy()
    forecasts = frame["forecast_kst_dtm"].to_numpy().reshape(len(issuances), 24)
    if not np.all(forecasts[:, 1:] > forecasts[:, :-1]):
        raise RuntimeError(f"group {group_id} daily forecast order changed")
    x = frame[feature_names].to_numpy(dtype="float32").reshape(
        len(issuances), 24, len(feature_names)
    )
    y = (
        frame["actual_kwh"].to_numpy(dtype=float) / CAPACITIES[group_id]
    ).reshape(len(issuances), 24)
    return frame, issuances, x, y


def _representation_matrix(values: np.ndarray, mode: str) -> np.ndarray:
    if mode == "raw":
        return values.reshape(len(values), -1)
    if mode == "raw_delta":
        delta = np.diff(values, axis=1, prepend=values[:, :1, :])
        return np.concatenate(
            [values.reshape(len(values), -1), delta.reshape(len(values), -1)],
            axis=1,
        )
    raise ValueError(f"unknown representation mode: {mode}")


def _cyclic_doy(issuances: np.ndarray) -> np.ndarray:
    dates = pd.DatetimeIndex(pd.to_datetime(issuances))
    angle = 2.0 * np.pi * (dates.dayofyear.to_numpy(dtype=float) - 1.0) / 365.25
    return np.column_stack([np.sin(angle), np.cos(angle)])


def _distances(
    train_values: np.ndarray,
    query_values: np.ndarray,
    train_issuances: np.ndarray,
    query_issuances: np.ndarray,
    representation: Representation,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    train = _representation_matrix(train_values, representation.mode)
    query = _representation_matrix(query_values, representation.mode)
    median = np.nanmedian(train, axis=0)
    median = np.where(np.isfinite(median), median, 0.0)
    train = np.where(np.isfinite(train), train, median)
    query = np.where(np.isfinite(query), query, median)
    scaler = StandardScaler()
    train = np.clip(scaler.fit_transform(train), -8.0, 8.0)
    query = np.clip(scaler.transform(query), -8.0, 8.0)
    components = min(representation.components, len(train) - 1, train.shape[1])
    if components != representation.components:
        raise RuntimeError(
            f"cannot fit {representation.components} PCA components on {len(train)} days"
        )
    pca = PCA(
        n_components=components,
        whiten=True,
        svd_solver="randomized",
        random_state=20260803,
    )
    train = pca.fit_transform(train)
    query = pca.transform(query)
    if representation.season_weight:
        train = np.column_stack(
            [train, representation.season_weight * _cyclic_doy(train_issuances)]
        )
        query = np.column_stack(
            [query, representation.season_weight * _cyclic_doy(query_issuances)]
        )
    squared = (
        np.sum(query * query, axis=1, keepdims=True)
        + np.sum(train * train, axis=1)[None, :]
        - 2.0 * query @ train.T
    )
    squared = np.maximum(squared, 0.0)
    order = np.argsort(squared, axis=1)
    max_neighbors = min(max(NEIGHBOR_COUNTS), len(train))
    order = order[:, :max_neighbors]
    distance = np.sqrt(np.take_along_axis(squared, order, axis=1))
    diagnostics = {
        "input_dimensions": int(_representation_matrix(train_values, representation.mode).shape[1]),
        "components": components,
        "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
        "median_nearest_distance": float(np.median(distance[:, 0])),
    }
    return order, distance, diagnostics


def _kernel_weights(distance: np.ndarray, kernel: str) -> np.ndarray:
    if kernel == "uniform":
        raw = np.ones_like(distance)
    elif kernel == "inverse":
        raw = 1.0 / np.maximum(distance, 1e-4)
    elif kernel == "exponential":
        bandwidth = np.maximum(np.median(distance, axis=1, keepdims=True), 1e-4)
        raw = np.exp(-distance / bandwidth)
    else:
        raise ValueError(f"unknown kernel: {kernel}")
    return raw / raw.sum(axis=1, keepdims=True)


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    output = np.empty((len(values), values.shape[2]), dtype=float)
    for query_index in range(len(values)):
        for hour in range(values.shape[2]):
            target = values[query_index, :, hour]
            order = np.argsort(target)
            cumulative = np.cumsum(weights[query_index, order])
            output[query_index, hour] = target[order[np.searchsorted(cumulative, 0.5)]]
    return output


def _utility_action(
    values: np.ndarray,
    weights: np.ndarray,
    mean_generation: float,
) -> np.ndarray:
    output = np.empty((len(values), values.shape[2]), dtype=float)
    for query_index in range(len(values)):
        for hour in range(values.shape[2]):
            target = values[query_index, :, hour]
            weight = weights[query_index]
            error = np.abs(ACTIONS[:, None] - target[None, :])
            units = np.select(
                [error <= 0.06, error <= 0.08],
                [4.0, 3.0],
                default=0.0,
            )
            utility = (
                -(error @ weight)
                + ((units * target[None, :]) @ weight)
                / (4.0 * mean_generation)
            )
            output[query_index, hour] = ACTIONS[int(np.argmax(utility))]
    return output


def _profile_heads(
    neighbor_targets: np.ndarray,
    weights: np.ndarray,
    mean_generation: float,
) -> dict[str, np.ndarray]:
    return {
        "mean": np.sum(neighbor_targets * weights[:, :, None], axis=1),
        "median": _weighted_median(neighbor_targets, weights),
        "utility": _utility_action(neighbor_targets, weights, mean_generation),
    }


def _profile_frame(
    frame: pd.DataFrame,
    query_issuances: np.ndarray,
    profile: np.ndarray,
    group_id: int,
) -> pd.DataFrame:
    mask = frame["data_available_kst_dtm"].isin(query_issuances)
    query = frame.loc[
        mask,
        ["forecast_id", "forecast_kst_dtm", "group_id", "data_available_kst_dtm"],
    ].copy()
    order = pd.Categorical(
        query["data_available_kst_dtm"], categories=query_issuances, ordered=True
    )
    query = query.assign(_order=order).sort_values(
        ["_order", "forecast_kst_dtm"]
    )
    if len(query) != profile.size or not query["group_id"].eq(group_id).all():
        raise RuntimeError(f"group {group_id} profile alignment changed")
    query["analog_normalized"] = profile.reshape(-1)
    return query.drop(columns="_order")


def _apply_recipe(
    parent: pd.DataFrame,
    analog: pd.DataFrame,
    group_id: int,
    recipe: Recipe,
) -> pd.DataFrame:
    group = parent.loc[parent["group_id"].eq(group_id)].copy()
    group = group.merge(
        analog[["forecast_id", "group_id", "analog_normalized"]],
        on=["forecast_id", "group_id"],
        how="left",
        validate="one_to_one",
    )
    parent_normalized = group["prediction_kwh"].to_numpy(dtype=float) / CAPACITIES[
        group_id
    ]
    analog_normalized = group["analog_normalized"].to_numpy(dtype=float)
    transformed = parent_normalized.copy()
    for positions in group.groupby("data_available_kst_dtm", sort=False).indices.values():
        positions = np.asarray(positions, dtype=int)
        p = parent_normalized[positions]
        a = analog_normalized[positions]
        if not np.isfinite(a).all():
            continue
        if recipe.transform == "level":
            target = a
        elif recipe.transform == "shape":
            target = p.mean() + (a - a.mean())
        elif recipe.transform == "scaled":
            target = a * p.mean() / max(float(a.mean()), 1e-4)
        else:
            raise ValueError(f"unknown transform: {recipe.transform}")
        transformed[positions] = (
            (1.0 - recipe.blend_weight) * p + recipe.blend_weight * target
        )
    group["prediction_kwh"] = np.clip(
        transformed * CAPACITIES[group_id], 0.0, CAPACITIES[group_id]
    )
    return group[parent.columns]


def _group_total_from_arrays(
    actual_kwh: np.ndarray,
    prediction_kwh: np.ndarray,
    group_id: int,
) -> float:
    capacity = CAPACITIES[group_id]
    eligible = actual_kwh >= 0.10 * capacity
    actual = actual_kwh[eligible]
    prediction = prediction_kwh[eligible]
    error = np.abs(prediction - actual) / capacity
    units = np.select([error <= 0.06, error <= 0.08], [4.0, 3.0], default=0.0)
    return 0.5 * (
        1.0 - float(error.mean())
        + float(np.sum(actual * units) / np.sum(actual * 4.0))
    )


def _screen_group(
    surface: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    parent: pd.DataFrame,
    group_id: int,
    query_issuances: np.ndarray,
) -> tuple[Recipe | None, pd.DataFrame | None, dict[str, object]]:
    parent_group = parent.loc[parent["group_id"].eq(group_id)].copy()
    baseline = _group_score(parent_group, group_id)
    best_total = baseline["total"]
    best_recipe: Recipe | None = None
    best_analog: pd.DataFrame | None = None
    representation_diagnostics: dict[str, object] = {}
    profile_cache: dict[tuple[str, int, str, str], pd.DataFrame] = {}

    for representation in REPRESENTATIONS:
        frame, issuances, values, targets = _complete_group_days(
            surface, group_id, feature_sets[representation.feature_set]
        )
        query_mask = np.isin(issuances, query_issuances)
        cutoff = pd.Timestamp(np.min(query_issuances))
        day_end = frame.groupby("data_available_kst_dtm", sort=True)[
            "forecast_kst_dtm"
        ].max().to_numpy()
        train_mask = (day_end < cutoff) & np.isfinite(targets).all(axis=1)
        if int(train_mask.sum()) < 120 or not query_mask.any():
            raise RuntimeError(
                f"group {group_id} insufficient strict analog days: "
                f"train={int(train_mask.sum())}, query={int(query_mask.sum())}"
            )
        order, distance, diagnostics = _distances(
            values[train_mask],
            values[query_mask],
            issuances[train_mask],
            issuances[query_mask],
            representation,
        )
        representation_diagnostics[representation.name] = {
            **diagnostics,
            "training_days": int(train_mask.sum()),
            "query_days": int(query_mask.sum()),
            "feature_count": len(feature_sets[representation.feature_set]),
        }
        available = min(distance.shape[1], max(NEIGHBOR_COUNTS))
        for neighbors in NEIGHBOR_COUNTS:
            if neighbors > available:
                continue
            neighbor_indices = order[:, :neighbors]
            neighbor_targets = targets[train_mask][neighbor_indices]
            mean_generation = float(np.nanmean(targets[train_mask]))
            for kernel in KERNELS:
                weights = _kernel_weights(distance[:, :neighbors], kernel)
                heads = _profile_heads(neighbor_targets, weights, mean_generation)
                for head, profile in heads.items():
                    analog = _profile_frame(
                        frame,
                        issuances[query_mask],
                        profile,
                        group_id,
                    )
                    profile_cache[(representation.name, neighbors, kernel, head)] = analog
                    merged = parent_group.merge(
                        analog[["forecast_id", "group_id", "analog_normalized"]],
                        on=["forecast_id", "group_id"],
                        how="left",
                        validate="one_to_one",
                    )
                    if merged["analog_normalized"].isna().any():
                        raise RuntimeError("screening analog merge is incomplete")
                    actual = merged["actual_kwh"].to_numpy(dtype=float)
                    p = merged["prediction_kwh"].to_numpy(dtype=float) / CAPACITIES[
                        group_id
                    ]
                    a = merged["analog_normalized"].to_numpy(dtype=float)
                    day_positions = list(
                        merged.groupby("data_available_kst_dtm", sort=False).indices.values()
                    )
                    for transform in TRANSFORMS:
                        target = a.copy()
                        if transform != "level":
                            for positions in day_positions:
                                positions = np.asarray(positions, dtype=int)
                                if transform == "shape":
                                    target[positions] = p[positions].mean() + (
                                        a[positions] - a[positions].mean()
                                    )
                                else:
                                    target[positions] = (
                                        a[positions]
                                        * p[positions].mean()
                                        / max(float(a[positions].mean()), 1e-4)
                                    )
                        for blend_weight in BLEND_WEIGHTS:
                            candidate = np.clip(
                                (1.0 - blend_weight) * p + blend_weight * target,
                                0.0,
                                1.0,
                            )
                            total = _group_total_from_arrays(
                                actual,
                                candidate * CAPACITIES[group_id],
                                group_id,
                            )
                            if total > best_total:
                                best_total = total
                                best_recipe = Recipe(
                                    representation=representation.name,
                                    neighbors=neighbors,
                                    kernel=kernel,
                                    head=head,
                                    transform=transform,
                                    blend_weight=blend_weight,
                                )
                                best_analog = analog

    if best_recipe is None or best_analog is None:
        return None, None, {
            "baseline": baseline,
            "selected": baseline,
            "recipe": None,
            "representation_diagnostics": representation_diagnostics,
        }
    selected = _apply_recipe(parent, best_analog, group_id, best_recipe)
    return best_recipe, best_analog, {
        "baseline": baseline,
        "selected": _group_score(selected, group_id),
        "recipe": asdict(best_recipe),
        "representation_diagnostics": representation_diagnostics,
    }


def _selected_profile(
    surface: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    group_id: int,
    query_issuances: np.ndarray,
    recipe: Recipe,
) -> tuple[pd.DataFrame, dict[str, object]]:
    representation = next(
        item for item in REPRESENTATIONS if item.name == recipe.representation
    )
    frame, issuances, values, targets = _complete_group_days(
        surface, group_id, feature_sets[representation.feature_set]
    )
    query_mask = np.isin(issuances, query_issuances)
    cutoff = pd.Timestamp(np.min(query_issuances))
    day_end = frame.groupby("data_available_kst_dtm", sort=True)[
        "forecast_kst_dtm"
    ].max().to_numpy()
    train_mask = (day_end < cutoff) & np.isfinite(targets).all(axis=1)
    order, distance, diagnostics = _distances(
        values[train_mask],
        values[query_mask],
        issuances[train_mask],
        issuances[query_mask],
        representation,
    )
    neighbor_indices = order[:, : recipe.neighbors]
    weights = _kernel_weights(distance[:, : recipe.neighbors], recipe.kernel)
    heads = _profile_heads(
        targets[train_mask][neighbor_indices],
        weights,
        float(np.nanmean(targets[train_mask])),
    )
    profile = _profile_frame(
        frame,
        issuances[query_mask],
        heads[recipe.head],
        group_id,
    )
    return profile, {
        **diagnostics,
        "training_days": int(train_mask.sum()),
        "query_days": int(query_mask.sum()),
    }


def _combine(parent: pd.DataFrame, replacements: list[pd.DataFrame]) -> pd.DataFrame:
    output = parent.copy()
    for replacement in replacements:
        group_id = int(replacement["group_id"].iloc[0])
        output = pd.concat(
            [output.loc[~output["group_id"].eq(group_id)], replacement],
            ignore_index=True,
        )
    return output.sort_values(["forecast_kst_dtm", "group_id"]).reset_index(drop=True)


def main() -> None:
    if sha256_file(OPEN) != OPEN_SHA or sha256_file(BASELINE) != BASELINE_SHA:
        raise RuntimeError("immutable competition input hash mismatch")
    started = time.perf_counter()
    surface, numeric = development_surface()
    if surface["forecast_kst_dtm"].ge(DEV_CUTOFF).any():
        raise RuntimeError("lockbox row reached daily analog runner")
    feature_sets = _feature_sets(numeric)
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
    q3_parent = _apply_long(q3, EXPECTED_SELECTIONS)
    q4_parent = _apply_long(q4, EXPECTED_SELECTIONS)
    q3_issuances = np.sort(q3["data_available_kst_dtm"].unique())
    q4_issuances = np.sort(q4["data_available_kst_dtm"].unique())

    selections: dict[int, Recipe] = {}
    q3_replacements: list[pd.DataFrame] = []
    q3_diagnostics: dict[str, object] = {}
    for group_id in CAPACITIES:
        recipe, analog, diagnostics = _screen_group(
            surface,
            feature_sets,
            q3_parent,
            group_id,
            q3_issuances,
        )
        q3_diagnostics[str(group_id)] = diagnostics
        if recipe is not None and analog is not None:
            selections[group_id] = recipe
            q3_replacements.append(
                _apply_recipe(q3_parent, analog, group_id, recipe)
            )
    q3_output = _combine(q3_parent, q3_replacements)

    q4_replacements: list[pd.DataFrame] = []
    q4_diagnostics: dict[str, object] = {}
    for group_id, recipe in selections.items():
        profile, diagnostics = _selected_profile(
            surface,
            feature_sets,
            group_id,
            q4_issuances,
            recipe,
        )
        replacement = _apply_recipe(q4_parent, profile, group_id, recipe)
        q4_replacements.append(replacement)
        q4_diagnostics[str(group_id)] = {
            "recipe": asdict(recipe),
            "retrieval": diagnostics,
            "parent": _group_score(q4_parent, group_id),
            "transformed": _group_score(replacement, group_id),
        }
    q4_output = _combine(q4_parent, q4_replacements)

    output = q3_output.assign(
        fold_id="dev-2023-Q3",
        model_id=CANDIDATE_ID,
    )
    output_path = OUTPUT / f"{CANDIDATE_ID}-dev-2023-Q3.parquet"
    output[[*METRIC_COLUMNS, "fold_id", "model_id", "data_available_kst_dtm"]].to_parquet(
        output_path,
        index=False,
    )
    receipt = {
        "schema_version": 1,
        "candidate_id": CANDIDATE_ID,
        "architecture": "strict_daily_nwp_analog_target_profile_v2_sequence_correction",
        "selection_fold": "dev-2023-Q3",
        "frozen_transfer_fold": "dev-2023-Q4",
        "feature_sets": feature_sets,
        "representations": [asdict(item) | {"name": item.name} for item in REPRESENTATIONS],
        "candidate_neighbor_counts": list(NEIGHBOR_COUNTS),
        "candidate_kernels": list(KERNELS),
        "candidate_heads": list(HEADS),
        "candidate_transforms": list(TRANSFORMS),
        "candidate_blend_weights": list(BLEND_WEIGHTS),
        "selected_recipes": {
            str(group_id): asdict(recipe) for group_id, recipe in selections.items()
        },
        "q3_parent_score": _score(q3_parent),
        "q3_selected_score": _score(q3_output),
        "q3_group_diagnostics": q3_diagnostics,
        "q4_parent_score": _score(q4_parent),
        "q4_frozen_score": _score(q4_output),
        "q4_total_delta": _score(q4_output)["total"] - _score(q4_parent)["total"],
        "q4_group_diagnostics": q4_diagnostics,
        "q4_paired_bootstrap": _paired_issuance_bootstrap(q4_parent, q4_output),
        "training_scope": "complete pre-cutoff issuance days from physical pre-2024 surface",
        "prediction_path": str(output_path.relative_to(ROOT)),
        "prediction_sha256": sha256_file(output_path),
        "policy_sha256": canonical_sha256(
            {str(group_id): asdict(recipe) for group_id, recipe in selections.items()}
        ),
        "parent_oof_path": str(OOF.relative_to(ROOT)),
        "parent_oof_sha256": sha256_file(OOF),
        "runtime_seconds": round(time.perf_counter() - started, 2),
        "same_fold_selection": True,
        "new_2024_evaluation": False,
        "lockbox_reopened": False,
        "external_actions": [],
    }
    receipt_path = OUTPUT / f"{CANDIDATE_ID}-dev-2023-Q3.json"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
