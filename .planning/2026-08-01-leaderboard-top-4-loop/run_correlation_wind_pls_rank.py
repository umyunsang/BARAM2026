"""Preserve training-correlated raw wind families in the strict PLS source rank."""

from __future__ import annotations

import argparse
import gc
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
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
from strict_dev_surface import DEV_CUTOFF, development_surface
from xgboost import XGBClassifier

CLASS_WIDTH = 0.02
CORRELATION_FEATURES_PER_GROUP = 80
ITERATIONS = 150
PARENT_PATHS = {
    "M197": OUTPUT / "M197_STRICT_FIXED_HALF_PLS-dev-2023-Q3.parquet",
    "M213": OUTPUT / "M213_STRICT_G3_CORRELATION_CHAMPION-dev-2023-Q3.parquet",
    "M215": OUTPUT
    / "M215_STRICT_G3_BALANCED_CORRELATION_CHAMPION-dev-2023-Q3.parquet",
    "M218": OUTPUT / "M218_LOCAL_Q3_CORR_BIN_BLEND-dev-2023-Q3.parquet",
    "M220": OUTPUT / "M220_LOCAL_Q3_CORR_BIN_WEIGHT_BLEND-dev-2023-Q3.parquet",
}


def _correlation_union(
    matrix: pd.DataFrame,
    surface: pd.DataFrame,
    target: pd.Series,
    training: np.ndarray,
    feature_count: int,
) -> tuple[list[str], dict[str, list[str]]]:
    selected_by_group: dict[str, list[str]] = {}
    union: list[str] = []
    for group_id in CAPACITIES:
        fit = training & surface["group_id"].eq(group_id).to_numpy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            correlation = matrix.loc[fit].corrwith(target.loc[fit]).abs()
        correlation = correlation.replace([np.inf, -np.inf], np.nan).dropna()
        selected = [
            str(name)
            for name in correlation.sort_values(ascending=False)
            .head(feature_count)
            .index
        ]
        if len(selected) != feature_count:
            raise RuntimeError(f"group {group_id} correlation support changed")
        selected_by_group[str(group_id)] = selected
        union.extend(selected)
    return list(dict.fromkeys(union)), selected_by_group


def _probability(
    matrix: pd.DataFrame,
    sample_weight: pd.Series,
    classes: pd.Series,
    training: np.ndarray,
    validation: np.ndarray,
    selected: list[str],
    class_count: int,
    seed: int,
    source: str,
) -> np.ndarray:
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=class_count,
        n_estimators=ITERATIONS,
        learning_rate=0.03,
        max_depth=5,
        min_child_weight=20.0,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=5.0,
        max_bin=256,
        tree_method="hist",
        random_state=seed,
        n_jobs=6,
    )
    model.fit(
        matrix.loc[training, selected],
        classes.loc[training].astype(int),
        sample_weight=sample_weight.loc[training],
    )
    raw = np.asarray(model.predict_proba(matrix.loc[validation, selected]), dtype=float)
    learned = np.asarray(model.classes_, dtype=int)
    probability = np.zeros((int(validation.sum()), class_count), dtype=float)
    probability[:, learned] = raw
    probability /= probability.sum(axis=1, keepdims=True)
    print(
        json.dumps(
            {
                "source": source,
                "feature_count": len(selected),
                "training_rows": int(training.sum()),
                "validation_rows": int(validation.sum()),
            }
        ),
        flush=True,
    )
    del model, raw
    gc.collect()
    return probability


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--fold", choices=("dev-2023-Q3",), required=True)
    parser.add_argument("--group-balance", action="store_true")
    parser.add_argument("--group3-interactions", action="store_true")
    parser.add_argument("--parent", choices=tuple(PARENT_PATHS), default="M197")
    parser.add_argument(
        "--class-width", type=float, choices=(0.02, 0.025), default=CLASS_WIDTH
    )
    parser.add_argument(
        "--correlation-features-per-group",
        type=int,
        choices=(80, 160),
        default=CORRELATION_FEATURES_PER_GROUP,
    )
    args = parser.parse_args()
    if _sha256(OPEN) != OPEN_SHA or _sha256(BASELINE) != BASELINE_SHA:
        raise RuntimeError("immutable competition input hash mismatch")
    parent_path = PARENT_PATHS[args.parent]
    if not parent_path.exists() or not M195_RECEIPT.exists() or not M195_LATENT.exists():
        raise RuntimeError("verified PLS parent surface is missing")

    started = time.perf_counter()
    surface, feature_columns = development_surface()
    if surface["forecast_kst_dtm"].ge(DEV_CUTOFF).any():
        raise RuntimeError("lockbox row reached correlation-wind runner")
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
    cached = np.load(M195_LATENT, allow_pickle=False)
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
        args.correlation_features_per_group,
    )
    interaction_columns: list[str] = []
    interaction_frame: pd.DataFrame | None = None
    if args.group3_interactions:
        group3 = surface["group_id"].eq(3).to_numpy(dtype="float32")
        interaction_columns = [f"corrg3__{name}" for name in correlated]
        interaction_frame = pd.DataFrame(
            base_matrix[correlated].to_numpy(dtype="float32") * group3[:, None],
            columns=interaction_columns,
            index=surface.index,
        )
    matrix = pd.concat(
        [
            base_matrix,
            pd.DataFrame(latent_values, columns=latent_columns, index=surface.index),
            *([] if interaction_frame is None else [interaction_frame]),
        ],
        axis=1,
    )
    del base_matrix, latent_values, interaction_frame
    gc.collect()

    raw_bins = np.floor(
        (target.clip(0.10, 1.074999) - 0.10) / args.class_width
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
    group_counts = {
        group_id: int(
            np.sum(training & surface["group_id"].eq(group_id).to_numpy())
        )
        for group_id in CAPACITIES
    }
    total_rows = float(sum(group_counts.values()))
    group_factors = {
        group_id: total_rows / (3.0 * count)
        for group_id, count in group_counts.items()
    }
    sample_weight = target.clip(lower=0.10).copy()
    if args.group_balance:
        sample_weight *= surface["group_id"].map(group_factors)

    additions = {
        "global": [*correlated, *interaction_columns],
        "gfs": [
            name
            for name in [*correlated, *interaction_columns]
            if "gfs" in name.lower()
        ],
        "ldaps": [
            name
            for name in [*correlated, *interaction_columns]
            if "ldaps" in name.lower()
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
            20262120 + 10 * source_index,
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

    parent = pd.read_parquet(parent_path)
    expected_keys = pd.MultiIndex.from_frame(
        surface.loc[validation, ["forecast_id", "group_id"]]
    )
    parent_keys = pd.MultiIndex.from_frame(parent[["forecast_id", "group_id"]])
    if len(parent) != int(validation.sum()) or not parent_keys.equals(expected_keys):
        raise RuntimeError("M197 parent key contract changed")
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
        "architecture": "strict_training_correlation_wind_family_pls_source_rank",
        "scope": (
            "fixed official-data-only training-correlation feature screen; outer "
            "Q3 labels excluded from ranking, classifier fit, policy, and blend"
        ),
        "physical_development_cutoff": str(DEV_CUTOFF),
        "strict_training_rows": int(training.sum()),
        "class_width": args.class_width,
        "class_count": len(active_bins),
        "correlation_features_per_group": args.correlation_features_per_group,
        "correlation_union_count": len(correlated),
        "correlation_source_counts": {
            source: len(names) for source, names in additions.items()
        },
        "group3_interactions": args.group3_interactions,
        "group3_interaction_feature_count": len(interaction_columns),
        "correlation_features_by_group": correlated_by_group,
        "selected_feature_counts": {
            source: len(names) for source, names in selected_features.items()
        },
        "group_balance": args.group_balance,
        "group_training_rows": group_counts,
        "group_weight_factors": group_factors if args.group_balance else None,
        "iterations": ITERATIONS,
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
        "parent_path": str(parent_path.relative_to(Path.cwd())),
        "parent_sha256": _sha256(parent_path),
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
                "correlation_union_count": len(correlated),
                "correlation_source_counts": receipt["correlation_source_counts"],
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
