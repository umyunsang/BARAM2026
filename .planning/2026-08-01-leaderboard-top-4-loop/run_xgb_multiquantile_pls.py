"""Fit XGBoost multi-quantile distributions over the strict PLS surface."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from run_group_balanced_pls_rank import _frame, _group_scores
from run_multioutput_donor_pls_rank import (
    M195_LATENT,
    M195_RECEIPT,
    _multioutput_latent,
)
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
from run_strict_prequential_source_rank import FROZEN_MIXTURE, _source_columns
from strict_dev_surface import DEV_CUTOFF, development_surface
from xgboost import XGBRegressor

QUANTILES = np.linspace(0.05, 0.95, 19)
ITERATIONS = 180
GLOBAL_FEATURES = 140
SOURCE_FEATURES = 200
UTILITY_GAMMA = 2.0
ACTIONS = np.arange(0.075, 1.0751, 0.0025)
PARENT_PATH = OUTPUT / "M197_STRICT_FIXED_HALF_PLS-dev-2023-Q3.parquet"


def _screen(
    matrix: pd.DataFrame,
    target: pd.Series,
    training: np.ndarray,
    candidates: list[str],
    count: int,
    seed: int,
) -> list[str]:
    model = LGBMRegressor(
        objective="l1",
        n_estimators=180,
        learning_rate=0.035,
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
    model.fit(matrix.loc[training, candidates], target.loc[training])
    gain = model.booster_.feature_importance(importance_type="gain")
    order = np.argsort(gain)[::-1][:count]
    selected = [candidates[position] for position in order]
    del model
    gc.collect()
    return selected


def _source_quantiles(
    matrix: pd.DataFrame,
    target: pd.Series,
    training: np.ndarray,
    validation: np.ndarray,
    candidates: list[str],
    feature_count: int,
    seed: int,
    source: str,
) -> tuple[np.ndarray, list[str]]:
    selected = _screen(
        matrix,
        target,
        training,
        candidates,
        feature_count,
        seed,
    )
    model = XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=QUANTILES,
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
        random_state=seed + 1,
        n_jobs=6,
    )
    model.fit(matrix.loc[training, selected], target.loc[training])
    prediction = np.asarray(
        model.predict(matrix.loc[validation, selected]), dtype=float
    )
    if prediction.shape != (int(validation.sum()), len(QUANTILES)):
        raise RuntimeError(f"{source} multi-quantile shape contract changed")
    prediction = np.sort(np.clip(prediction, 0.075, 1.075), axis=1)
    print(
        json.dumps(
            {
                "source": source,
                "training_rows": int(training.sum()),
                "validation_rows": int(validation.sum()),
                "feature_count": len(selected),
                "quantile_count": len(QUANTILES),
            }
        ),
        flush=True,
    )
    del model
    gc.collect()
    return prediction, selected


def _mixture(quantiles: dict[str, np.ndarray]) -> np.ndarray:
    global_weight, gfs_share = FROZEN_MIXTURE
    source = gfs_share * quantiles["gfs"] + (1.0 - gfs_share) * quantiles["ldaps"]
    mixed = global_weight * quantiles["global"] + (1.0 - global_weight) * source
    return np.sort(np.clip(mixed, 0.075, 1.075), axis=1)


def _quantile_actions(
    quantile_values: np.ndarray,
    groups: np.ndarray,
    mean_generation: dict[int, float],
) -> np.ndarray:
    chosen = np.empty(len(quantile_values), dtype=float)
    for group_id in CAPACITIES:
        mask = groups == group_id
        samples = quantile_values[mask]
        error = np.abs(ACTIONS[:, None, None] - samples[None, :, :])
        units = np.select(
            [error <= 0.06, error <= 0.08], [4.0, 3.0], default=0.0
        )
        expected_error = error.mean(axis=2).T
        expected_revenue = (samples[None, :, :] * units).mean(axis=2).T / (
            4.0 * mean_generation[group_id]
        )
        utility = -expected_error + UTILITY_GAMMA * expected_revenue
        chosen[mask] = ACTIONS[np.argmax(utility, axis=1)]
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--fold", choices=("dev-2023-Q3",), required=True)
    args = parser.parse_args()
    if _sha256(OPEN) != OPEN_SHA or _sha256(BASELINE) != BASELINE_SHA:
        raise RuntimeError("immutable competition input hash mismatch")
    if not PARENT_PATH.exists() or not M195_RECEIPT.exists() or not M195_LATENT.exists():
        raise RuntimeError("verified PLS parent surface is missing")

    started = time.perf_counter()
    surface, feature_columns = development_surface()
    if surface["forecast_kst_dtm"].ge(DEV_CUTOFF).any():
        raise RuntimeError("lockbox row reached XGBoost multi-quantile runner")
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
    parent_columns = [str(value) for value in cached["columns"].tolist()]
    parent_values = np.asarray(cached["values"], dtype="float32")
    if parent_values.shape != (len(surface), len(parent_columns)):
        raise RuntimeError("M195 latent checkpoint shape contract changed")
    base_matrix = surface[feature_columns].astype("float32")
    multioutput, multioutput_diagnostics = _multioutput_latent(
        surface,
        base_matrix,
        history,
        validation,
        target,
        feature_columns,
    )
    matrix = pd.concat(
        [
            base_matrix,
            pd.DataFrame(parent_values, columns=parent_columns, index=surface.index),
            multioutput,
        ],
        axis=1,
    )
    del base_matrix, parent_values, multioutput
    gc.collect()

    all_candidates = list(matrix.columns)
    global_latent = [
        name
        for name in all_candidates
        if "__global__" in name and name.startswith(("pls__", "mpls__"))
    ]
    source_specs = {
        "global": (all_candidates, GLOBAL_FEATURES),
        "gfs": (
            list(dict.fromkeys([*_source_columns(all_candidates, "gfs"), *global_latent])),
            SOURCE_FEATURES,
        ),
        "ldaps": (
            list(
                dict.fromkeys(
                    [*_source_columns(all_candidates, "ldaps"), *global_latent]
                )
            ),
            SOURCE_FEATURES,
        ),
    }
    quantiles: dict[str, np.ndarray] = {}
    selected_features: dict[str, list[str]] = {}
    for source_index, (source, (candidates, count)) in enumerate(source_specs.items()):
        prediction, selected = _source_quantiles(
            matrix,
            target,
            training,
            validation,
            candidates,
            count,
            20262500 + source_index * 10,
            source,
        )
        quantiles[source] = prediction
        selected_features[source] = selected
    del matrix
    gc.collect()

    quantile_cache_path = OUTPUT / f"{args.candidate_id}-{args.fold}-quantiles.npz"
    np.savez(
        quantile_cache_path,
        quantile_levels=QUANTILES,
        global_quantiles=quantiles["global"].astype("float32"),
        gfs_quantiles=quantiles["gfs"].astype("float32"),
        ldaps_quantiles=quantiles["ldaps"].astype("float32"),
    )
    mixed = _mixture(quantiles)
    groups = surface.loc[validation, "group_id"].to_numpy(dtype=int)
    means = {
        group_id: float(
            target.loc[training & surface["group_id"].eq(group_id).to_numpy()].mean()
        )
        for group_id in CAPACITIES
    }
    normalized = _quantile_actions(mixed, groups, means)
    raw_output = _frame(surface, validation, normalized)
    median_output = _frame(surface, validation, mixed[:, len(QUANTILES) // 2])

    parent = pd.read_parquet(PARENT_PATH)
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
    output.to_parquet(output_path, index=False)

    receipt = {
        "schema_version": 1,
        "candidate_id": args.candidate_id,
        "fold_id": args.fold,
        "architecture": "strict_xgb_multiquantile_m195_m196_half_m197",
        "scope": (
            "fixed official-data-only multi-quantile distribution; outer Q3 labels "
            "excluded from PLS, screen, quantile fit, utility action, and blend"
        ),
        "physical_development_cutoff": str(DEV_CUTOFF),
        "strict_training_rows": int(training.sum()),
        "quantiles": QUANTILES.tolist(),
        "iterations": ITERATIONS,
        "utility_gamma": UTILITY_GAMMA,
        "selected_features": selected_features,
        "selected_latent_feature_counts": {
            source: sum(name.startswith(("pls__", "mpls__")) for name in names)
            for source, names in selected_features.items()
        },
        "multioutput_diagnostics": multioutput_diagnostics,
        "frozen_mixture": {
            "global_weight": FROZEN_MIXTURE[0],
            "gfs_share": FROZEN_MIXTURE[1],
        },
        "median_score": _score(median_output),
        "median_group_scores": _group_scores(median_output),
        "raw_score": _score(raw_output),
        "raw_group_scores": _group_scores(raw_output),
        "fixed_parent_weight": 0.5,
        "fold_score": _score(output),
        "group_scores": _group_scores(output),
        "runtime_seconds": round(time.perf_counter() - started, 2),
        "prediction_path": str(output_path.relative_to(Path.cwd())),
        "prediction_sha256": _sha256(output_path),
        "quantile_cache_path": str(quantile_cache_path.relative_to(Path.cwd())),
        "quantile_cache_sha256": _sha256(quantile_cache_path),
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
                "median_score": receipt["median_score"],
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
