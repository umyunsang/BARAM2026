"""Learn weather-conditional hourly phase corrections from Q2 and apply to Q3."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
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
from strict_dev_surface import DEV_CUTOFF, development_surface

BASE_COLUMNS = ["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh"]
KEYS = ["forecast_id", "forecast_kst_dtm", "group_id"]
OFFSETS = np.arange(-3, 4, dtype=int)
Q2_POLICY_PATH = OUTPUT / "M150_SOURCE_RANK_XGB_Q2-dev-2023-Q2-policies.parquet"
Q3_POLICY_PATH = OUTPUT / "M149_SOURCE_RANK_XGB_Q3-dev-2023-Q3-policies.parquet"


def _group_score(
    frame: pd.DataFrame,
    group_id: int,
    prediction_kwh: np.ndarray,
) -> dict[str, float]:
    actual = frame["actual_kwh"].to_numpy(dtype=float) / CAPACITIES[group_id]
    prediction = prediction_kwh / CAPACITIES[group_id]
    eligible = np.isfinite(actual) & np.isfinite(prediction) & (actual >= 0.10)
    actual = actual[eligible]
    prediction = prediction[eligible]
    error = np.abs(prediction - actual)
    units = np.select([error <= 0.06, error <= 0.08], [4.0, 3.0], default=0.0)
    one_minus_nmae = 1.0 - float(error.mean())
    ficr = float(np.sum(actual * units) / np.sum(actual * 4.0))
    return {
        "total": 0.5 * (one_minus_nmae + ficr),
        "one_minus_nmae": one_minus_nmae,
        "ficr": ficr,
    }


def _select_parent_policies(
    q2: pd.DataFrame,
    early: np.ndarray,
    common_columns: list[str],
) -> tuple[dict[int, str], dict[str, dict[str, float]]]:
    selected: dict[int, str] = {}
    scores: dict[str, dict[str, float]] = {}
    for group_id in CAPACITIES:
        group = early & q2["group_id"].eq(group_id).to_numpy()
        part = q2.loc[group]
        candidates = {
            name: _group_score(
                part,
                group_id,
                q2.loc[group, name].to_numpy(dtype=float),
            )
            for name in common_columns
        }
        best = max(candidates, key=lambda name: candidates[name]["total"])
        selected[group_id] = best
        scores[str(group_id)] = candidates[best]
    return selected, scores


def _parent_frame(
    policies: pd.DataFrame,
    metadata: pd.DataFrame,
    selections: dict[int, str],
) -> pd.DataFrame:
    frame = policies[BASE_COLUMNS].copy()
    prediction = np.empty(len(frame), dtype=float)
    for group_id, column in selections.items():
        group = frame["group_id"].eq(group_id).to_numpy()
        prediction[group] = policies.loc[group, column].to_numpy(dtype=float)
    frame["parent_prediction_kwh"] = prediction
    return frame.merge(metadata, on=KEYS, validate="one_to_one")


def _phase_candidates(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    ordered = frame.sort_values(
        ["group_id", "data_available_kst_dtm", "forecast_kst_dtm"]
    ).reset_index(drop=True)
    grouped = ordered.groupby(
        ["group_id", "data_available_kst_dtm"], sort=False
    )["parent_prediction_kwh"]
    capacity = ordered["group_id"].map(CAPACITIES).to_numpy(dtype=float)
    candidates = np.column_stack(
        [
            grouped.shift(-int(offset))
            .fillna(ordered["parent_prediction_kwh"])
            .to_numpy(dtype=float)
            / capacity
            for offset in OFFSETS
        ]
    )
    return ordered, candidates


def _oracle_labels(
    frame: pd.DataFrame,
    candidates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    capacity = frame["group_id"].map(CAPACITIES).to_numpy(dtype=float)
    actual = frame["actual_kwh"].to_numpy(dtype=float) / capacity
    eligible = np.isfinite(actual) & (actual >= 0.10)
    error = np.abs(candidates - actual[:, None])
    tie_penalty = 1e-8 * np.abs(OFFSETS)[None, :]
    labels = np.argmin(error + tie_penalty, axis=1).astype(int)
    return labels, eligible


def _meta_matrix(
    surface: pd.DataFrame,
    feature_columns: list[str],
    frame: pd.DataFrame,
    candidates: np.ndarray,
) -> pd.DataFrame:
    non_key_features = [name for name in feature_columns if name not in KEYS]
    feature_source = surface[[*KEYS, *non_key_features]]
    joined = frame[KEYS].merge(feature_source, on=KEYS, validate="one_to_one")
    additions: dict[str, np.ndarray] = {}
    for position, offset in enumerate(OFFSETS):
        additions[f"phase__candidate_{offset:+d}"] = candidates[:, position]
        additions[f"phase__delta_{offset:+d}"] = (
            candidates[:, position] - candidates[:, OFFSETS.tolist().index(0)]
        )
    additions["phase__range"] = candidates.max(axis=1) - candidates.min(axis=1)
    additions["phase__slope2"] = candidates[:, 4] - candidates[:, 2]
    additions["phase__curvature"] = (
        candidates[:, 4] - 2.0 * candidates[:, 3] + candidates[:, 2]
    )
    matrix = pd.concat(
        [
            joined[feature_columns].reset_index(drop=True).astype("float32"),
            pd.DataFrame(additions, dtype="float32"),
        ],
        axis=1,
    )
    return matrix


def _feature_screen(
    matrix: pd.DataFrame,
    labels: np.ndarray,
    fit: np.ndarray,
    weights: np.ndarray,
    count: int,
) -> list[str]:
    model = LGBMClassifier(
        objective="multiclass",
        num_class=len(OFFSETS),
        n_estimators=100,
        learning_rate=0.035,
        num_leaves=15,
        min_child_samples=60,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.75,
        reg_alpha=0.3,
        reg_lambda=5.0,
        random_state=20260803,
        n_jobs=6,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )
    model.fit(matrix.loc[fit], labels[fit], sample_weight=weights[fit])
    gains = model.booster_.feature_importance(importance_type="gain")
    order = np.argsort(gains)[::-1]
    selected = [matrix.columns[position] for position in order[:count]]
    if len(selected) != count or len(set(selected)) != count:
        raise RuntimeError("phase-selector feature screen contract changed")
    return selected


def _fit_probability(
    matrix: pd.DataFrame,
    labels: np.ndarray,
    fit: np.ndarray,
    apply: np.ndarray,
    weights: np.ndarray,
    selected: list[str],
    architecture: str,
    leaves: int,
    iterations: int,
    groups: np.ndarray,
) -> np.ndarray:
    probability = np.zeros((int(apply.sum()), len(OFFSETS)), dtype=float)
    apply_positions = np.flatnonzero(apply)
    model_groups = (0,) if architecture == "shared" else tuple(CAPACITIES)
    for model_group in model_groups:
        if model_group == 0:
            group_fit = fit
            group_apply = apply
        else:
            group_fit = fit & (groups == model_group)
            group_apply = apply & (groups == model_group)
        active = sorted(np.unique(labels[group_fit]).tolist())
        mapping = {label: position for position, label in enumerate(active)}
        mapped = np.asarray([mapping[value] for value in labels[group_fit]], dtype=int)
        model = LGBMClassifier(
            objective="multiclass",
            num_class=len(active),
            n_estimators=iterations,
            learning_rate=0.035,
            num_leaves=leaves,
            min_child_samples=60,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_alpha=0.3,
            reg_lambda=5.0,
            random_state=20260803 + model_group,
            n_jobs=6,
            deterministic=True,
            force_col_wise=True,
            verbosity=-1,
        )
        model.fit(
            matrix.loc[group_fit, selected],
            mapped,
            sample_weight=weights[group_fit],
        )
        raw = model.predict_proba(matrix.loc[group_apply, selected])
        local_positions = np.searchsorted(apply_positions, np.flatnonzero(group_apply))
        for column, label in enumerate(active):
            probability[local_positions, label] = raw[:, column]
    probability /= probability.sum(axis=1, keepdims=True)
    return probability


def _apply_probability(
    frame: pd.DataFrame,
    candidates: np.ndarray,
    apply: np.ndarray,
    probability: np.ndarray,
    confidence_floor: float,
) -> pd.DataFrame:
    selected = np.argmax(probability, axis=1)
    if confidence_floor > 0:
        selected = np.where(
            probability.max(axis=1) >= confidence_floor,
            selected,
            int(np.flatnonzero(OFFSETS == 0)[0]),
        )
    normalized = candidates[apply][np.arange(int(apply.sum())), selected]
    base = frame.loc[apply, BASE_COLUMNS].copy()
    capacity = base["group_id"].map(CAPACITIES).to_numpy(dtype=float)
    base["prediction_kwh"] = normalized * capacity
    return base


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--feature-count", type=int, default=180)
    args = parser.parse_args()
    if not 80 <= args.feature_count <= 300:
        raise ValueError("feature-count must be between 80 and 300")
    if _sha256(OPEN) != OPEN_SHA or _sha256(BASELINE) != BASELINE_SHA:
        raise RuntimeError("immutable competition input hash mismatch")

    started = time.perf_counter()
    surface, feature_columns = development_surface()
    if surface["forecast_kst_dtm"].ge(DEV_CUTOFF).any():
        raise RuntimeError("lockbox row reached the phase selector")
    q2_policies = pd.read_parquet(Q2_POLICY_PATH)
    q3_policies = pd.read_parquet(Q3_POLICY_PATH)
    common = sorted(
        set(q2_policies.columns)
        .intersection(q3_policies.columns)
        .difference(BASE_COLUMNS)
    )
    if len(common) < 20:
        raise RuntimeError("source-rank policy intersection changed")

    q2_times = q2_policies["forecast_kst_dtm"].drop_duplicates().sort_values()
    policy_cutoff = pd.Timestamp(q2_times.iloc[int(len(q2_times) * 0.50)])
    q2_early_keys = q2_policies["forecast_kst_dtm"].lt(policy_cutoff).to_numpy()
    selections, policy_scores = _select_parent_policies(
        q2_policies, q2_early_keys, common
    )
    metadata = surface[[*KEYS, "data_available_kst_dtm"]]
    q2, q2_candidates = _phase_candidates(
        _parent_frame(q2_policies, metadata, selections)
    )
    q3, q3_candidates = _phase_candidates(
        _parent_frame(q3_policies, metadata, selections)
    )
    q2_labels, q2_eligible = _oracle_labels(q2, q2_candidates)
    q3_labels, q3_eligible = _oracle_labels(q3, q3_candidates)
    q2_matrix = _meta_matrix(surface, feature_columns, q2, q2_candidates)
    q3_matrix = _meta_matrix(surface, feature_columns, q3, q3_candidates)
    q2_groups = q2["group_id"].to_numpy(dtype=int)
    q3_groups = q3["group_id"].to_numpy(dtype=int)
    q2_actual = q2["actual_kwh"].to_numpy(dtype=float) / q2["group_id"].map(
        CAPACITIES
    ).to_numpy(dtype=float)
    weights = np.clip(q2_actual, 0.10, None)
    calibration_cutoff = pd.Timestamp(
        q2.loc[q2_eligible, "forecast_kst_dtm"].sort_values().iloc[
            int(q2_eligible.sum() * 0.60)
        ]
    )
    inner_fit = q2_eligible & q2["forecast_kst_dtm"].lt(
        calibration_cutoff
    ).to_numpy()
    calibration = q2_eligible & ~inner_fit
    selected_features = _feature_screen(
        q2_matrix,
        q2_labels,
        inner_fit,
        weights,
        args.feature_count,
    )

    sweep: dict[str, object] = {}
    best: tuple[float, str, int, int, float] | None = None
    for architecture in ("shared", "group"):
        for leaves in (7, 15, 31):
            for iterations in (40, 80, 160):
                probability = _fit_probability(
                    q2_matrix,
                    q2_labels,
                    inner_fit,
                    calibration,
                    weights,
                    selected_features,
                    architecture,
                    leaves,
                    iterations,
                    q2_groups,
                )
                scores: dict[str, dict[str, float]] = {}
                for floor in (0.0, 0.25, 0.35, 0.45, 0.55):
                    output = _apply_probability(
                        q2,
                        q2_candidates,
                        calibration,
                        probability,
                        floor,
                    )
                    score = _score(output)
                    scores[f"{floor:g}"] = score
                    choice = (score["total"], architecture, leaves, iterations, floor)
                    if best is None or choice[0] > best[0]:
                        best = choice
                tag = f"{architecture}_L{leaves}_I{iterations}"
                sweep[tag] = scores
                print(
                    json.dumps(
                        {
                            "inner": tag,
                            "best": max(
                                scores.items(), key=lambda item: item[1]["total"]
                            ),
                        }
                    ),
                    flush=True,
                )
    assert best is not None

    final_features = _feature_screen(
        q2_matrix,
        q2_labels,
        q2_eligible,
        weights,
        args.feature_count,
    )
    q2_probability = _fit_probability(
        q2_matrix,
        q2_labels,
        q2_eligible,
        np.ones(len(q2), dtype=bool),
        weights,
        final_features,
        best[1],
        best[2],
        best[3],
        q2_groups,
    )
    # Refit above is also used as an in-sample diagnostic.  Fit the same frozen
    # selector on Q2 and apply it to Q3 without reading Q3 labels.
    if best[1] == "shared":
        combined_matrix = pd.concat([q2_matrix, q3_matrix], ignore_index=True)
        fit = np.zeros(len(combined_matrix), dtype=bool)
        fit[: len(q2)] = q2_eligible
        apply = np.zeros(len(combined_matrix), dtype=bool)
        apply[len(q2) :] = True
        labels = np.concatenate([q2_labels, np.zeros(len(q3), dtype=int)])
        combined_weights = np.concatenate([weights, np.ones(len(q3))])
        groups = np.concatenate([q2_groups, q3_groups])
        probability = _fit_probability(
            combined_matrix,
            labels,
            fit,
            apply,
            combined_weights,
            final_features,
            best[1],
            best[2],
            best[3],
            groups,
        )
    else:
        combined_matrix = pd.concat([q2_matrix, q3_matrix], ignore_index=True)
        fit = np.zeros(len(combined_matrix), dtype=bool)
        fit[: len(q2)] = q2_eligible
        apply = np.zeros(len(combined_matrix), dtype=bool)
        apply[len(q2) :] = True
        labels = np.concatenate([q2_labels, np.zeros(len(q3), dtype=int)])
        combined_weights = np.concatenate([weights, np.ones(len(q3))])
        groups = np.concatenate([q2_groups, q3_groups])
        probability = _fit_probability(
            combined_matrix,
            labels,
            fit,
            apply,
            combined_weights,
            final_features,
            best[1],
            best[2],
            best[3],
            groups,
        )
    output = _apply_probability(
        q3,
        q3_candidates,
        np.ones(len(q3), dtype=bool),
        probability,
        best[4],
    )
    output["fold_id"] = "dev-2023-Q3"
    output["model_id"] = args.candidate_id
    base_q3 = q3[BASE_COLUMNS].copy()
    base_q3["prediction_kwh"] = q3["parent_prediction_kwh"]
    oracle_q3 = q3[BASE_COLUMNS].copy()
    oracle_q3["prediction_kwh"] = (
        q3_candidates[np.arange(len(q3)), q3_labels]
        * q3["group_id"].map(CAPACITIES).to_numpy(dtype=float)
    )
    in_sample_q2 = _apply_probability(
        q2,
        q2_candidates,
        np.ones(len(q2), dtype=bool),
        q2_probability,
        best[4],
    )
    output_path = OUTPUT / f"{args.candidate_id}-dev-2023-Q3.parquet"
    output.to_parquet(output_path, index=False)
    receipt = {
        "schema_version": 1,
        "candidate_id": args.candidate_id,
        "architecture": "q2_trained_weather_conditional_phase_selector",
        "scope": "legacy source-rank parent diagnostic; Q3 labels excluded from selection",
        "physical_development_cutoff": str(DEV_CUTOFF),
        "offsets": OFFSETS.tolist(),
        "selected_parent_policies": selections,
        "parent_policy_early_q2_scores": policy_scores,
        "selected_architecture": best[1],
        "selected_leaves": best[2],
        "selected_iterations": best[3],
        "selected_confidence_floor": best[4],
        "selected_inner_score": best[0],
        "inner_sweep": sweep,
        "feature_count": args.feature_count,
        "q2_label_distribution": np.bincount(
            q2_labels[q2_eligible], minlength=len(OFFSETS)
        ).tolist(),
        "q3_label_distribution_diagnostic": np.bincount(
            q3_labels[q3_eligible], minlength=len(OFFSETS)
        ).tolist(),
        "q2_in_sample_score": _score(in_sample_q2),
        "q3_parent_score": _score(base_q3),
        "q3_phase_oracle_score": _score(oracle_q3),
        "fold_score": _score(output),
        "selected_feature_names": final_features,
        "runtime_seconds": round(time.perf_counter() - started, 2),
        "prediction_path": str(output_path.relative_to(Path.cwd())),
        "prediction_sha256": _sha256(output_path),
        "new_2024_evaluation": False,
        "lockbox_reopened": False,
        "external_actions": [],
    }
    receipt_path = OUTPUT / f"{args.candidate_id}-dev-2023-Q3.json"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
