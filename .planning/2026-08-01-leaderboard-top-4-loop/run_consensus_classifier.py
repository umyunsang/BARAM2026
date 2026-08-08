"""Screen shared classifiers trained on denoised cross-group power targets."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from run_pseudo_group3_classifier import _pseudo_targets
from run_sequence_classifier import (
    BASELINE,
    BASELINE_SHA,
    CAPACITIES,
    OPEN,
    OPEN_SHA,
    OUTPUT,
    _score,
    _sha256,
    _surface,
)
from run_site_wind_classifier import FOLDS, _add_site_wind_features, _choose_actions
from run_site_wind_teacher import _validation_mask

PARENT_PATH = OUTPUT / "M107_STRICT_TEMPORAL_TOP100-oof.parquet"
BASE_COLUMNS = ["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh"]


def _consensus_targets(
    surface: pd.DataFrame,
    preceding: np.ndarray,
    profile: str,
) -> tuple[pd.Series, np.ndarray, dict[str, object]]:
    normalized = surface["actual_kwh"] / surface["group_id"].map(CAPACITIES)
    history = surface.loc[
        preceding,
        ["forecast_kst_dtm", "group_id"],
    ].copy()
    history["target"] = normalized.loc[preceding].to_numpy(dtype=float)
    wide = history.pivot(
        index="forecast_kst_dtm", columns="group_id", values="target"
    )
    mean12 = wide[[1, 2]].mean(axis=1)
    mapped_mean12 = surface["forecast_kst_dtm"].map(mean12)
    target = normalized.copy()
    pseudo = np.zeros(len(surface), dtype=bool)
    if profile == "consensus12":
        target.loc[preceding] = mapped_mean12.loc[preceding]
        pseudo = (
            preceding
            & surface["group_id"].eq(3).to_numpy()
            & surface["actual_kwh"].isna().to_numpy()
        )
    elif profile == "denoise12":
        group12 = preceding & surface["group_id"].isin((1, 2)).to_numpy()
        target.loc[group12] = mapped_mean12.loc[group12]
        pseudo_values, _ = _pseudo_targets(surface, preceding)
        available = pseudo_values.notna().to_numpy()
        target.loc[available] = pseudo_values.loc[available]
        pseudo = available
    else:
        raise ValueError(f"unknown target profile: {profile}")
    diagnostics = {
        "profile": profile,
        "training_target_rows": int(target.loc[preceding].notna().sum()),
        "pseudo_rows": int(pseudo.sum()),
        "target_mean": float(target.loc[preceding].mean()),
    }
    return target, pseudo, diagnostics


def _group_total(frame: pd.DataFrame, group_id: int) -> float:
    capacity = CAPACITIES[group_id]
    part = frame.loc[frame["group_id"].eq(group_id)]
    actual = part["actual_kwh"].to_numpy(dtype=float) / capacity
    prediction = part["prediction_kwh"].to_numpy(dtype=float) / capacity
    valid = actual >= 0.10
    actual = actual[valid]
    prediction = prediction[valid]
    error = np.abs(prediction - actual)
    units = np.select([error <= 0.06, error <= 0.08], [4.0, 3.0], default=0.0)
    return 0.5 * (1.0 - float(error.mean())) + 0.5 * float(
        np.sum(actual * units) / np.sum(actual * 4.0)
    )


def _screen_blends(
    base: pd.DataFrame,
    policies: pd.DataFrame,
    parent: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    parent_map = parent.set_index(["forecast_id", "group_id"])["prediction_kwh"]
    parent_prediction = pd.MultiIndex.from_frame(base[["forecast_id", "group_id"]]).map(
        parent_map
    ).to_numpy(dtype=float)
    policy_columns = sorted(set(policies.columns).difference(BASE_COLUMNS))
    best_group: dict[int, tuple[float, str, float, np.ndarray]] = {}
    for group_id in CAPACITIES:
        mask = base["group_id"].eq(group_id).to_numpy()
        for policy in policy_columns:
            challenger = policies[policy].to_numpy(dtype=float)
            for parent_weight in np.arange(0.0, 1.01, 0.1):
                prediction = (
                    parent_weight * parent_prediction
                    + (1.0 - parent_weight) * challenger
                )
                trial = base.copy()
                trial["prediction_kwh"] = prediction
                score = _group_total(trial, group_id)
                choice = (score, policy, float(parent_weight), prediction)
                if group_id not in best_group or choice[0] > best_group[group_id][0]:
                    best_group[group_id] = choice
    output = base.copy()
    output["prediction_kwh"] = parent_prediction
    selections: dict[str, object] = {}
    for group_id, (score, policy, parent_weight, prediction) in best_group.items():
        mask = output["group_id"].eq(group_id).to_numpy()
        output.loc[mask, "prediction_kwh"] = prediction[mask]
        selections[str(group_id)] = {
            "policy": policy,
            "parent_weight": parent_weight,
            "oracle_group_score": score,
        }
    return output, selections


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--fold", choices=FOLDS, default=FOLDS[-1])
    parser.add_argument("--sitewind-cache-id", required=True)
    parser.add_argument(
        "--target-profile", choices=("consensus12", "denoise12"), required=True
    )
    parser.add_argument("--pseudo-weight", type=float, default=0.25)
    parser.add_argument("--iterations", nargs="+", type=int, default=[40, 60, 80])
    parser.add_argument("--top-features", type=int, default=100)
    args = parser.parse_args()
    if _sha256(OPEN) != OPEN_SHA or _sha256(BASELINE) != BASELINE_SHA:
        raise RuntimeError("immutable competition input hash mismatch")
    if not 0.0 <= args.pseudo_weight <= 2.0:
        raise ValueError("pseudo weight must be between zero and two")
    started = time.perf_counter()
    surface, base_columns, _ = _surface()
    validation = _validation_mask(surface, args.fold)
    start = surface.loc[validation, "forecast_kst_dtm"].min()
    preceding = surface["forecast_kst_dtm"].lt(start).to_numpy()
    target, pseudo, target_diagnostics = _consensus_targets(
        surface, preceding, args.target_profile
    )
    training = preceding & target.ge(0.10).to_numpy()
    raw_bins = np.floor((target.clip(0.10, 1.074999) - 0.10) / 0.02).astype(
        "Int64"
    )
    active_bins = [
        int(value)
        for value in sorted(raw_bins.loc[training].dropna().astype(int).unique())
    ]
    bin_to_class = {bin_id: class_id for class_id, bin_id in enumerate(active_bins)}
    classes = raw_bins.map(bin_to_class).astype("Int64")
    centers = np.asarray(
        [
            target.loc[training & classes.eq(class_id)].mean()
            for class_id in range(len(active_bins))
        ]
    )
    cached = np.load(
        OUTPUT / f"{args.sitewind_cache_id}-{args.fold}-sitewind-features.npz"
    )
    matrix = surface[base_columns].astype("float32")
    sitewind_columns = _add_site_wind_features(
        matrix, cached["legacy"], cached["allweather"]
    )
    params = {
        "objective": "multiclass",
        "num_class": len(active_bins),
        "n_estimators": max(args.iterations),
        "learning_rate": 0.025,
        "num_leaves": 15,
        "min_child_samples": 80,
        "subsample": 0.9,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 2.0,
        "random_state": 20260802,
        "n_jobs": 6,
        "deterministic": True,
        "force_col_wise": True,
        "verbosity": -1,
    }
    sample_weight = target.loc[training].clip(lower=0.10).to_numpy(dtype=float)
    sample_weight *= np.where(pseudo[training], args.pseudo_weight, 1.0)
    screen = LGBMClassifier(**params)
    screen.fit(
        matrix.loc[training],
        classes.loc[training].astype(int),
        sample_weight=sample_weight,
    )
    gains = screen.booster_.feature_importance(importance_type="gain")
    positions = np.argsort(gains)[::-1][: args.top_features]
    selected_features = [matrix.columns[index] for index in positions]
    matrix = matrix[selected_features]
    model = LGBMClassifier(**params)
    model.fit(
        matrix.loc[training],
        classes.loc[training].astype(int),
        sample_weight=sample_weight,
    )
    base = surface.loc[validation, BASE_COLUMNS].copy()
    parent = pd.read_parquet(PARENT_PATH)
    parent = parent.loc[parent["fold_id"].eq(args.fold)]
    sweep: dict[str, object] = {}
    best: tuple[float, int, pd.DataFrame, pd.DataFrame, dict[str, object]] | None = None
    for iteration in sorted(set(args.iterations)):
        probability = model.predict_proba(
            matrix.loc[validation], num_iteration=iteration
        )
        _, _, _, policies = _choose_actions(
            base,
            probability,
            centers,
            target,
            training,
            surface["group_id"],
        )
        output, selections = _screen_blends(base, policies, parent)
        score = _score(output)
        sweep[str(iteration)] = {"score": score, "selections": selections}
        choice = (score["total"], iteration, output, policies, selections)
        if best is None or choice[0] > best[0]:
            best = choice
        print(
            json.dumps(
                {"iteration": iteration, "score": score, "selections": selections}
            ),
            flush=True,
        )
    assert best is not None
    output = best[2]
    output["fold_id"] = args.fold
    output["model_id"] = args.candidate_id
    output_path = OUTPUT / f"{args.candidate_id}-{args.fold}.parquet"
    output.to_parquet(output_path, index=False)
    receipt = {
        "schema_version": 1,
        "candidate_id": args.candidate_id,
        "fold_id": args.fold,
        "architecture": "cross_group_consensus_multiclass",
        "scope": "Q4 representation screen with oracle per-group parent blends",
        "target_diagnostics": target_diagnostics,
        "pseudo_weight": args.pseudo_weight,
        "selected_iteration": best[1],
        "selected_fold_score": _score(output),
        "selected_oracle_blends": best[4],
        "sweep": sweep,
        "feature_count": len(selected_features),
        "sitewind_feature_count": len(sitewind_columns),
        "selected_feature_names": selected_features,
        "runtime_seconds": round(time.perf_counter() - started, 2),
        "prediction_path": str(output_path.relative_to(Path.cwd())),
        "prediction_sha256": _sha256(output_path),
        "observed_scada_feature_count": 0,
        "new_2024_evaluation": False,
        "lockbox_reopened": False,
        "external_actions": [],
    }
    receipt_path = OUTPUT / f"{args.candidate_id}-{args.fold}.json"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
