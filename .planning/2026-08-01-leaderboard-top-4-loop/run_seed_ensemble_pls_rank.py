"""Average five XGBoost probability seeds over the strict M195/M196 surface."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from run_group_balanced_pls_rank import _frame, _group_scores
from run_inner_policy_classifier import _policy_values
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
from run_strict_prequential_source_rank import (
    FROZEN_MIXTURE,
    FROZEN_POLICY,
    _mixture,
    _screen,
    _source_columns,
)
from strict_dev_surface import DEV_CUTOFF, development_surface
from xgboost import XGBClassifier

CLASS_WIDTH = 0.02
ITERATIONS = 150
GLOBAL_FEATURES = 120
SOURCE_FEATURES = 180
BASE_SEEDS = (20261601, 20262301, 20262302, 20262303, 20262304)
PARENT_PATH = OUTPUT / "M197_STRICT_FIXED_HALF_PLS-dev-2023-Q3.parquet"


def _seed_probability(
    matrix: pd.DataFrame,
    target: pd.Series,
    classes: pd.Series,
    training: np.ndarray,
    validation: np.ndarray,
    selected: list[str],
    class_count: int,
    source_index: int,
    source: str,
) -> tuple[np.ndarray, dict[str, float]]:
    seed_probabilities: list[np.ndarray] = []
    weights = target.loc[training].clip(lower=0.10)
    for seed_index, base_seed in enumerate(BASE_SEEDS):
        seed = base_seed + source_index * 10
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
            sample_weight=weights,
        )
        raw = np.asarray(
            model.predict_proba(matrix.loc[validation, selected]), dtype=float
        )
        learned = np.asarray(model.classes_, dtype=int)
        probability = np.zeros((int(validation.sum()), class_count), dtype=float)
        probability[:, learned] = raw
        probability /= probability.sum(axis=1, keepdims=True)
        seed_probabilities.append(probability)
        print(
            json.dumps(
                {
                    "source": source,
                    "seed_index": seed_index + 1,
                    "seed_count": len(BASE_SEEDS),
                    "seed": seed,
                    "status": "fit",
                }
            ),
            flush=True,
        )
        del model, raw
        gc.collect()
    stacked = np.stack(seed_probabilities)
    average = stacked.mean(axis=0)
    average /= average.sum(axis=1, keepdims=True)
    diagnostics = {
        "mean_probability_std": float(stacked.std(axis=0).mean()),
        "max_probability_std": float(stacked.std(axis=0).max()),
    }
    del stacked, seed_probabilities
    gc.collect()
    return average, diagnostics


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
        raise RuntimeError("lockbox row reached PLS seed-ensemble runner")
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
    probabilities: dict[str, np.ndarray] = {}
    selected_features: dict[str, list[str]] = {}
    source_diagnostics: dict[str, dict[str, float]] = {}
    for source_index, (source, (candidates, count)) in enumerate(source_specs.items()):
        selected = _screen(
            matrix,
            target,
            training,
            candidates,
            count,
            20261600 + source_index * 10,
        )
        probability, diagnostics = _seed_probability(
            matrix,
            target,
            classes,
            training,
            validation,
            selected,
            len(active_bins),
            source_index,
            source,
        )
        probabilities[source] = probability
        selected_features[source] = selected
        source_diagnostics[source] = diagnostics
    del matrix
    gc.collect()

    probability_cache_path = OUTPUT / f"{args.candidate_id}-{args.fold}-probability.npz"
    np.savez(
        probability_cache_path,
        global_probability=probabilities["global"].astype("float32"),
        gfs_probability=probabilities["gfs"].astype("float32"),
        ldaps_probability=probabilities["ldaps"].astype("float32"),
    )
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
        "architecture": "strict_five_seed_xgb_m195_m196_probability_bag_half_m197",
        "scope": (
            "fixed official-data-only stochastic tree bag; outer Q3 labels excluded "
            "from PLS, feature screen, every seed fit, policy, and blend"
        ),
        "physical_development_cutoff": str(DEV_CUTOFF),
        "strict_training_rows": int(training.sum()),
        "class_width": CLASS_WIDTH,
        "class_count": len(active_bins),
        "iterations": ITERATIONS,
        "base_seeds": list(BASE_SEEDS),
        "selected_features": selected_features,
        "selected_latent_feature_counts": {
            source: sum(name.startswith(("pls__", "mpls__")) for name in names)
            for source, names in selected_features.items()
        },
        "source_diagnostics": source_diagnostics,
        "multioutput_diagnostics": multioutput_diagnostics,
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
        "probability_cache_path": str(probability_cache_path.relative_to(Path.cwd())),
        "probability_cache_sha256": _sha256(probability_cache_path),
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
                "raw_score": receipt["raw_score"],
                "raw_group_scores": receipt["raw_group_scores"],
                "fold_score": receipt["fold_score"],
                "group_scores": receipt["group_scores"],
                "source_diagnostics": source_diagnostics,
                "runtime_seconds": receipt["runtime_seconds"],
                "prediction_sha256": receipt["prediction_sha256"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
