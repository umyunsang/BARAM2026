"""Augment strict PLS source-rank heads with sliced inverse regression axes."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from run_correlation_wind_pls_rank import _correlation_union, _probability
from run_group_balanced_pls_rank import _frame, _group_scores
from run_inner_policy_classifier import _policy_values
from run_multioutput_donor_pls_rank import M195_LATENT, M195_RECEIPT
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
)
from sklearn.decomposition import PCA
from strict_dev_surface import DEV_CUTOFF, development_surface

CLASS_WIDTH = 0.02
CORRELATION_FEATURES_PER_GROUP = 80
PCA_COMPONENTS = 48
SIR_COMPONENTS = 8
SIR_SLICES = 12
PARENT_PATH = OUTPUT / "M225_LOCAL_Q3_PROB_SMOOTH_BLEND-dev-2023-Q3.parquet"


def _sir_coordinates(
    base_matrix: pd.DataFrame,
    target: pd.Series,
    training: np.ndarray,
    source_candidates: dict[str, list[str]],
    surface: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object], dict[str, list[str]]]:
    frames: list[pd.DataFrame] = []
    diagnostics: dict[str, object] = {}
    columns_by_source: dict[str, list[str]] = {}
    group = surface["group_id"].to_numpy(dtype=int)
    regimes = {
        "pooled": training,
        "donor12": training & np.isin(group, [1, 2]),
        "group3": training & (group == 3),
    }
    for source_index, (source, candidates) in enumerate(source_candidates.items()):
        candidates = list(dict.fromkeys(candidates))
        fit_values = base_matrix.loc[training, candidates]
        medians = fit_values.median(axis=0)
        means = fit_values.fillna(medians).mean(axis=0)
        scales = fit_values.fillna(medians).std(axis=0).replace(0.0, 1.0)
        values = (
            base_matrix[candidates]
            .fillna(medians)
            .sub(means, axis=1)
            .div(scales, axis=1)
            .replace([np.inf, -np.inf], 0.0)
            .to_numpy(dtype="float32")
        )
        component_count = min(PCA_COMPONENTS, len(candidates), int(training.sum()) - 1)
        pca = PCA(
            n_components=component_count,
            svd_solver="randomized",
            random_state=20263200 + source_index,
        )
        fit_scores = pca.fit_transform(values[training])
        all_scores = pca.transform(values)
        whitening = np.sqrt(np.maximum(pca.explained_variance_, 1e-8))
        fit_scores /= whitening
        all_scores /= whitening
        source_columns: list[str] = []
        source_diagnostics: dict[str, object] = {
            "raw_candidate_count": len(candidates),
            "pca_component_count": component_count,
            "pca_explained_variance_ratio": float(
                pca.explained_variance_ratio_.sum()
            ),
        }
        training_positions = np.flatnonzero(training)
        position_lookup = np.full(len(surface), -1, dtype=int)
        position_lookup[training_positions] = np.arange(len(training_positions))
        for regime, regime_mask in regimes.items():
            global_positions = np.flatnonzero(regime_mask)
            local_positions = position_lookup[global_positions]
            if (local_positions < 0).any():
                raise RuntimeError(f"{source}/{regime} SIR position contract changed")
            regime_target = target.iloc[global_positions]
            slices = pd.qcut(
                regime_target,
                q=SIR_SLICES,
                labels=False,
                duplicates="drop",
            ).to_numpy(dtype=int)
            slice_ids = np.unique(slices)
            between = np.zeros((component_count, component_count), dtype=float)
            for slice_id in slice_ids:
                selected = local_positions[slices == slice_id]
                mean = fit_scores[selected].mean(axis=0)
                between += (len(selected) / len(local_positions)) * np.outer(mean, mean)
            eigenvalues, eigenvectors = np.linalg.eigh(between)
            order = np.argsort(eigenvalues)[::-1][:SIR_COMPONENTS]
            coordinates = all_scores @ eigenvectors[:, order]
            names = [
                f"sir__{source}__{regime}__{index:02d}"
                for index in range(coordinates.shape[1])
            ]
            frames.append(
                pd.DataFrame(
                    coordinates.astype("float32"),
                    columns=names,
                    index=surface.index,
                )
            )
            source_columns.extend(names)
            source_diagnostics[regime] = {
                "training_rows": len(global_positions),
                "slice_count": len(slice_ids),
                "leading_eigenvalues": [
                    float(eigenvalues[index]) for index in order
                ],
            }
        columns_by_source[source] = source_columns
        diagnostics[source] = source_diagnostics
        del pca, values, fit_values, fit_scores, all_scores
        gc.collect()
    return pd.concat(frames, axis=1), diagnostics, columns_by_source


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--fold", choices=("dev-2023-Q3",), required=True)
    args = parser.parse_args()
    if _sha256(OPEN) != OPEN_SHA or _sha256(BASELINE) != BASELINE_SHA:
        raise RuntimeError("immutable competition input hash mismatch")
    if not PARENT_PATH.exists() or not M195_RECEIPT.exists() or not M195_LATENT.exists():
        raise RuntimeError("verified SIR/PLS parent surface is missing")

    started = time.perf_counter()
    surface, feature_columns = development_surface()
    if surface["forecast_kst_dtm"].ge(DEV_CUTOFF).any():
        raise RuntimeError("lockbox row reached SIR runner")
    validation = _validation_mask(surface, args.fold)
    history = _strict_preceding_mask(surface, validation)
    target = surface["actual_kwh"] / surface["group_id"].map(CAPACITIES)
    training = (
        history
        & surface["actual_kwh"].notna().to_numpy()
        & target.ge(0.10).to_numpy()
    )

    parent_receipt = json.loads(M195_RECEIPT.read_text())
    if _sha256(M195_LATENT) != parent_receipt["latent_checkpoint_sha256"]:
        raise RuntimeError("M195 latent checkpoint hash mismatch")
    with np.load(M195_LATENT, allow_pickle=False) as cached:
        latent_columns = [str(value) for value in cached["columns"].tolist()]
        latent_values = np.asarray(cached["values"], dtype="float32")
    if latent_values.shape != (len(surface), len(latent_columns)):
        raise RuntimeError("M195 latent checkpoint shape contract changed")

    base_matrix = surface[feature_columns].astype("float32")
    correlated, correlated_by_group = _correlation_union(
        base_matrix,
        surface,
        target,
        training,
        CORRELATION_FEATURES_PER_GROUP,
    )
    raw_columns = set(feature_columns)
    raw_parent = {
        source: [
            name
            for name in parent_receipt["selected_features"][source]
            if name in raw_columns
        ]
        for source in ("global", "gfs", "ldaps")
    }
    source_candidates = {
        "global": [*raw_parent["global"], *correlated],
        "gfs": [
            *raw_parent["gfs"],
            *[name for name in correlated if "gfs" in name.lower()],
        ],
        "ldaps": [
            *raw_parent["ldaps"],
            *[name for name in correlated if "ldaps" in name.lower()],
        ],
    }
    sir_frame, sir_diagnostics, sir_columns = _sir_coordinates(
        base_matrix,
        target,
        training,
        source_candidates,
        surface,
    )
    matrix = pd.concat(
        [
            base_matrix,
            pd.DataFrame(latent_values, columns=latent_columns, index=surface.index),
            sir_frame,
        ],
        axis=1,
    )
    del base_matrix, latent_values, sir_frame
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
    sample_weight = target.clip(lower=0.10).copy()
    additions = {
        "global": [*correlated, *sir_columns["global"]],
        "gfs": [
            *[name for name in correlated if "gfs" in name.lower()],
            *sir_columns["global"],
            *sir_columns["gfs"],
        ],
        "ldaps": [
            *[name for name in correlated if "ldaps" in name.lower()],
            *sir_columns["global"],
            *sir_columns["ldaps"],
        ],
    }
    selected_features = {
        source: list(
            dict.fromkeys(
                [*parent_receipt["selected_features"][source], *additions[source]]
            )
        )
        for source in ("global", "gfs", "ldaps")
    }
    probabilities = {
        source: _probability(
            matrix,
            sample_weight,
            classes,
            training,
            validation,
            selected_features[source],
            len(active_bins),
            20263300 + 10 * source_index,
            source,
        )
        for source_index, source in enumerate(("global", "gfs", "ldaps"))
    }
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
    raw_output = _frame(surface, validation, normalized)

    parent = pd.read_parquet(PARENT_PATH)
    expected_keys = pd.MultiIndex.from_frame(
        surface.loc[validation, ["forecast_id", "group_id"]]
    )
    parent_keys = pd.MultiIndex.from_frame(parent[["forecast_id", "group_id"]])
    if len(parent) != int(validation.sum()) or not parent_keys.equals(expected_keys):
        raise RuntimeError("M225 parent key contract changed")
    capacity = parent["group_id"].map(CAPACITIES).to_numpy(dtype=float)
    parent_normalized = parent["prediction_kwh"].to_numpy(dtype=float) / capacity
    output = _frame(surface, validation, 0.5 * parent_normalized + 0.5 * normalized)
    output["fold_id"] = args.fold
    output["model_id"] = args.candidate_id
    output_path = OUTPUT / f"{args.candidate_id}-{args.fold}.parquet"
    probability_path = OUTPUT / f"{args.candidate_id}-{args.fold}-probability.npz"
    output.to_parquet(output_path, index=False)
    np.savez(probability_path, probability=probability.astype("float32"))

    receipt = {
        "schema_version": 1,
        "candidate_id": args.candidate_id,
        "fold_id": args.fold,
        "architecture": "strict_sliced_inverse_regression_pls_source_rank_half_m225",
        "scope": (
            "official-data-only PCA-whitened SIR features learned on strict preceding "
            "targets; outer Q3 labels excluded from axes, class heads, policy, and blend"
        ),
        "physical_development_cutoff": str(DEV_CUTOFF),
        "strict_training_rows": int(training.sum()),
        "class_width": CLASS_WIDTH,
        "class_count": len(active_bins),
        "pca_components": PCA_COMPONENTS,
        "sir_components_per_regime": SIR_COMPONENTS,
        "sir_slices": SIR_SLICES,
        "sir_diagnostics": sir_diagnostics,
        "sir_feature_counts": {
            source: len(names) for source, names in sir_columns.items()
        },
        "correlation_features_per_group": CORRELATION_FEATURES_PER_GROUP,
        "correlation_union_count": len(correlated),
        "correlation_features_by_group": correlated_by_group,
        "selected_feature_counts": {
            source: len(names) for source, names in selected_features.items()
        },
        "frozen_mixture": {
            "global_weight": FROZEN_MIXTURE[0],
            "gfs_share": FROZEN_MIXTURE[1],
        },
        "frozen_policy": FROZEN_POLICY,
        "raw_score": _score(raw_output),
        "raw_group_scores": _group_scores(raw_output),
        "fixed_parent_weight": 0.5,
        "fold_score": _score(output),
        "group_scores": _group_scores(output),
        "runtime_seconds": round(time.perf_counter() - started, 2),
        "prediction_path": str(output_path.relative_to(Path.cwd())),
        "prediction_sha256": _sha256(output_path),
        "probability_path": str(probability_path.relative_to(Path.cwd())),
        "probability_sha256": _sha256(probability_path),
        "parent_path": str(PARENT_PATH.relative_to(Path.cwd())),
        "parent_sha256": _sha256(PARENT_PATH),
        "m195_receipt_sha256": _sha256(M195_RECEIPT),
        "m195_latent_sha256": _sha256(M195_LATENT),
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
                "sir_diagnostics": sir_diagnostics,
                "selected_feature_counts": receipt["selected_feature_counts"],
                "raw_score": receipt["raw_score"],
                "raw_group_scores": receipt["raw_group_scores"],
                "fold_score": receipt["fold_score"],
                "group_scores": receipt["group_scores"],
                "runtime_seconds": receipt["runtime_seconds"],
                "prediction_sha256": receipt["prediction_sha256"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
