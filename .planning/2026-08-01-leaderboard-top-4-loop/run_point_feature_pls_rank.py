"""Stack honest raw-weather point predictions into strict PLS source-rank heads."""

from __future__ import annotations

import argparse
import gc
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
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
from sklearn.model_selection import GroupKFold
from strict_dev_surface import DEV_CUTOFF, development_surface

CLASS_WIDTH = 0.02
CORRELATION_FEATURES_PER_GROUP = 80
POINT_FEATURES = 160
POINT_ITERATIONS = 280
POINT_FOLDS = 4
PARENT_PATH = OUTPUT / "M222_LOCAL_Q3_POINT_ANCHOR_BLEND-dev-2023-Q3.parquet"


def _rank_point_features(
    matrix: pd.DataFrame,
    target: pd.Series,
    positions: np.ndarray,
) -> list[str]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        correlation = matrix.iloc[positions].corrwith(target.iloc[positions]).abs()
    correlation = correlation.replace([np.inf, -np.inf], np.nan).dropna()
    selected = [
        str(name)
        for name in correlation.sort_values(ascending=False)
        .head(POINT_FEATURES)
        .index
    ]
    if len(selected) != POINT_FEATURES:
        raise RuntimeError("point feature support changed")
    return selected


def _point_model(seed: int) -> LGBMRegressor:
    return LGBMRegressor(
        objective="l1",
        n_estimators=POINT_ITERATIONS,
        learning_rate=0.025,
        num_leaves=31,
        min_child_samples=60,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=3.0,
        random_state=seed,
        n_jobs=6,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )


def _crossfit_point(
    matrix: pd.DataFrame,
    target: pd.Series,
    surface: pd.DataFrame,
    training: np.ndarray,
    validation: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    point = np.full(len(surface), np.nan, dtype="float32")
    diagnostics: dict[str, object] = {}
    for group_id in CAPACITIES:
        group = surface["group_id"].eq(group_id).to_numpy()
        training_positions = np.flatnonzero(training & group)
        validation_positions = np.flatnonzero(validation & group)
        batches = (
            surface.iloc[training_positions]["data_available_kst_dtm"]
            .astype(str)
            .to_numpy()
        )
        splitter = GroupKFold(n_splits=POINT_FOLDS)
        fold_diagnostics: list[dict[str, object]] = []
        for fold_index, (fit_local, holdout_local) in enumerate(
            splitter.split(training_positions, groups=batches)
        ):
            fit_positions = training_positions[fit_local]
            holdout_positions = training_positions[holdout_local]
            selected = _rank_point_features(matrix, target, fit_positions)
            model = _point_model(20262700 + group_id * 10 + fold_index)
            model.fit(
                matrix.iloc[fit_positions][selected],
                target.iloc[fit_positions],
            )
            point[holdout_positions] = np.clip(
                model.predict(matrix.iloc[holdout_positions][selected]),
                0.075,
                1.075,
            ).astype("float32")
            fold_diagnostics.append(
                {
                    "fold": fold_index,
                    "fit_rows": len(fit_positions),
                    "holdout_rows": len(holdout_positions),
                    "fit_feature_count": len(selected),
                    "holdout_batches": int(np.unique(batches[holdout_local]).size),
                }
            )
            print(
                json.dumps(
                    {"point_group": group_id, "crossfit": fold_diagnostics[-1]}
                ),
                flush=True,
            )
            del model
            gc.collect()

        if not np.isfinite(point[training_positions]).all():
            raise RuntimeError(f"group {group_id} point crossfit is incomplete")
        selected = _rank_point_features(matrix, target, training_positions)
        final_model = _point_model(20262750 + group_id)
        final_model.fit(
            matrix.iloc[training_positions][selected],
            target.iloc[training_positions],
        )
        point[validation_positions] = np.clip(
            final_model.predict(matrix.iloc[validation_positions][selected]),
            0.075,
            1.075,
        ).astype("float32")
        group_oof_error = np.abs(
            point[training_positions] - target.iloc[training_positions].to_numpy()
        )
        diagnostics[str(group_id)] = {
            "training_rows": len(training_positions),
            "validation_rows": len(validation_positions),
            "oof_mae": float(group_oof_error.mean()),
            "final_feature_count": len(selected),
            "crossfit_folds": fold_diagnostics,
        }
        del final_model
        gc.collect()
    if not np.isfinite(point[training | validation]).all():
        raise RuntimeError("point stack contains missing fit/apply values")
    diagnostics["pooled_oof_mae"] = float(
        np.mean(np.abs(point[training] - target.loc[training].to_numpy()))
    )
    return point, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--fold", choices=("dev-2023-Q3",), required=True)
    args = parser.parse_args()
    if _sha256(OPEN) != OPEN_SHA or _sha256(BASELINE) != BASELINE_SHA:
        raise RuntimeError("immutable competition input hash mismatch")
    if not PARENT_PATH.exists() or not M195_RECEIPT.exists() or not M195_LATENT.exists():
        raise RuntimeError("verified PLS/point-anchor parent surface is missing")

    started = time.perf_counter()
    surface, feature_columns = development_surface()
    if surface["forecast_kst_dtm"].ge(DEV_CUTOFF).any():
        raise RuntimeError("lockbox row reached point-feature runner")
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
    point, point_diagnostics = _crossfit_point(
        base_matrix,
        target,
        surface,
        training,
        validation,
    )
    point_columns = [
        "stack_point",
        "stack_point_sq",
        "stack_point_sqrt",
        "stack_point_g1",
        "stack_point_g2",
        "stack_point_g3",
    ]
    point_frame = pd.DataFrame(index=surface.index)
    point_frame["stack_point"] = point
    point_frame["stack_point_sq"] = point**2
    point_frame["stack_point_sqrt"] = np.sqrt(np.clip(point, 0.0, None))
    for group_id in CAPACITIES:
        point_frame[f"stack_point_g{group_id}"] = point * surface[
            "group_id"
        ].eq(group_id).to_numpy(dtype="float32")
    matrix = pd.concat(
        [
            base_matrix,
            pd.DataFrame(latent_values, columns=latent_columns, index=surface.index),
            point_frame,
        ],
        axis=1,
    )
    del base_matrix, latent_values, point_frame
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
        "global": [*correlated, *point_columns],
        "gfs": [
            *[name for name in correlated if "gfs" in name.lower()],
            *point_columns,
        ],
        "ldaps": [
            *[name for name in correlated if "ldaps" in name.lower()],
            *point_columns,
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
            20262780 + 10 * source_index,
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
    point_output = _frame(surface, validation, point[validation])

    parent = pd.read_parquet(PARENT_PATH)
    expected_keys = pd.MultiIndex.from_frame(
        surface.loc[validation, ["forecast_id", "group_id"]]
    )
    parent_keys = pd.MultiIndex.from_frame(parent[["forecast_id", "group_id"]])
    if len(parent) != int(validation.sum()) or not parent_keys.equals(expected_keys):
        raise RuntimeError("M222 parent key contract changed")
    capacity = parent["group_id"].map(CAPACITIES).to_numpy(dtype=float)
    parent_normalized = parent["prediction_kwh"].to_numpy(dtype=float) / capacity
    output = _frame(surface, validation, 0.5 * parent_normalized + 0.5 * normalized)
    output["fold_id"] = args.fold
    output["model_id"] = args.candidate_id
    output_path = OUTPUT / f"{args.candidate_id}-{args.fold}.parquet"
    probability_path = OUTPUT / f"{args.candidate_id}-{args.fold}-probability.npz"
    point_path = OUTPUT / f"{args.candidate_id}-{args.fold}-point.npz"
    output.to_parquet(output_path, index=False)
    np.savez(probability_path, probability=probability.astype("float32"))
    np.savez(point_path, point=point.astype("float32"))

    receipt = {
        "schema_version": 1,
        "candidate_id": args.candidate_id,
        "fold_id": args.fold,
        "architecture": "strict_raw_crossfit_point_feature_pls_source_rank_half_m222",
        "scope": (
            "official-data-only raw-weather batch-grouped cross-fitted point stack; "
            "outer Q3 labels excluded from point, class, policy, and fixed blend"
        ),
        "physical_development_cutoff": str(DEV_CUTOFF),
        "strict_training_rows": int(training.sum()),
        "class_width": CLASS_WIDTH,
        "class_count": len(active_bins),
        "correlation_features_per_group": CORRELATION_FEATURES_PER_GROUP,
        "correlation_union_count": len(correlated),
        "correlation_features_by_group": correlated_by_group,
        "point_feature_count": POINT_FEATURES,
        "point_iterations": POINT_ITERATIONS,
        "point_folds": POINT_FOLDS,
        "point_diagnostics": point_diagnostics,
        "stack_feature_names": point_columns,
        "selected_feature_counts": {
            source: len(names) for source, names in selected_features.items()
        },
        "frozen_mixture": {
            "global_weight": FROZEN_MIXTURE[0],
            "gfs_share": FROZEN_MIXTURE[1],
        },
        "frozen_policy": FROZEN_POLICY,
        "point_score": _score(point_output),
        "point_group_scores": _group_scores(point_output),
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
        "point_path": str(point_path.relative_to(Path.cwd())),
        "point_sha256": _sha256(point_path),
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
                "point_score": receipt["point_score"],
                "point_group_scores": receipt["point_group_scores"],
                "raw_score": receipt["raw_score"],
                "raw_group_scores": receipt["raw_group_scores"],
                "fold_score": receipt["fold_score"],
                "group_scores": receipt["group_scores"],
                "point_diagnostics": point_diagnostics,
                "runtime_seconds": receipt["runtime_seconds"],
                "prediction_sha256": receipt["prediction_sha256"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
