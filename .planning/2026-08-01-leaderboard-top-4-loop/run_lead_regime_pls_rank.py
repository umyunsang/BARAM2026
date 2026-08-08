"""Fit lead-regime source-rank experts on the verified strict M195 PLS surface."""

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
ITERATIONS = 180
LEAD_REGIMES = ((12, 17), (18, 23), (24, 29), (30, 35))
PARENT_PATH = OUTPUT / "M197_STRICT_FIXED_HALF_PLS-dev-2023-Q3.parquet"


def _fit_regime_probability(
    matrix: pd.DataFrame,
    target: pd.Series,
    classes: pd.Series,
    training: np.ndarray,
    validation: np.ndarray,
    lead: pd.Series,
    selected: list[str],
    class_count: int,
    source: str,
    seed: int,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    probability = np.zeros((int(validation.sum()), class_count), dtype=float)
    validation_lead = lead.loc[validation].to_numpy(dtype=int)
    diagnostics: list[dict[str, object]] = []
    for regime_index, (lower, upper) in enumerate(LEAD_REGIMES):
        in_regime = lead.between(lower, upper).to_numpy()
        fit = training & in_regime
        apply = validation & in_regime
        apply_positions = np.flatnonzero(
            (validation_lead >= lower) & (validation_lead <= upper)
        )
        global_labels = classes.loc[fit].astype(int).to_numpy()
        active = np.asarray(sorted(np.unique(global_labels)), dtype=int)
        if len(active) < 30 or int(fit.sum()) < 2500 or int(apply.sum()) == 0:
            raise RuntimeError(
                f"{source} lead regime {lower}-{upper} has insufficient support"
            )
        local_labels = np.searchsorted(active, global_labels)
        model = XGBClassifier(
            objective="multi:softprob",
            num_class=len(active),
            n_estimators=ITERATIONS,
            learning_rate=0.03,
            max_depth=4,
            min_child_weight=12.0,
            subsample=0.9,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=6.0,
            max_bin=256,
            tree_method="hist",
            random_state=seed + regime_index,
            n_jobs=6,
        )
        model.fit(
            matrix.loc[fit, selected],
            local_labels,
            sample_weight=target.loc[fit].clip(lower=0.10),
        )
        raw = np.asarray(model.predict_proba(matrix.loc[apply, selected]), dtype=float)
        probability[np.ix_(apply_positions, active)] = raw
        diagnostics.append(
            {
                "source": source,
                "lead_lower": lower,
                "lead_upper": upper,
                "fit_rows": int(fit.sum()),
                "apply_rows": int(apply.sum()),
                "active_classes": len(active),
            }
        )
        print(json.dumps({"lead_regime": diagnostics[-1]}), flush=True)
        del model, raw
        gc.collect()
    row_sums = probability.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        raise RuntimeError(f"{source} lead-regime probabilities are incomplete")
    return probability, diagnostics


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
        raise RuntimeError("lockbox row reached lead-regime runner")
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
    matrix = pd.concat(
        (
            surface[feature_columns].astype("float32"),
            pd.DataFrame(latent_values, columns=latent_columns, index=surface.index),
        ),
        axis=1,
    )
    del latent_values
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

    selected_features = {
        str(source): [str(name) for name in names]
        for source, names in parent_receipt["selected_features"].items()
    }
    probabilities: dict[str, np.ndarray] = {}
    diagnostics: list[dict[str, object]] = []
    for source_index, source in enumerate(("global", "gfs", "ldaps")):
        selected = selected_features[source]
        missing = sorted(set(selected).difference(matrix.columns))
        if missing:
            raise RuntimeError(f"{source} selected-feature contract changed: {missing[:3]}")
        probability, source_diagnostics = _fit_regime_probability(
            matrix,
            target,
            classes,
            training,
            validation,
            surface["lead_hour"],
            selected,
            len(active_bins),
            source,
            20262090 + 20 * source_index,
        )
        probabilities[source] = probability
        diagnostics.extend(source_diagnostics)
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
        "architecture": "strict_four_lead_regime_m195_source_rank_half_m197",
        "scope": (
            "fixed official-data-only lead-regime source-rank screen; outer Q3 "
            "labels excluded from PLS, classifier fit, policy, and blend"
        ),
        "physical_development_cutoff": str(DEV_CUTOFF),
        "strict_training_rows": int(training.sum()),
        "class_width": CLASS_WIDTH,
        "class_count": len(active_bins),
        "iterations": ITERATIONS,
        "lead_regimes": [list(regime) for regime in LEAD_REGIMES],
        "selected_feature_counts": {
            source: len(names) for source, names in selected_features.items()
        },
        "diagnostics": diagnostics,
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
