"""Rebuild the strongest source-rank representation under strict chronology.

SCADA wind is streamed only up to the outer-fold cutoff and is never exposed to
the power classifier.  NWP-to-wind auxiliary values for meta-training rows are
prequential: each value is predicted by a model fitted only on complete earlier
issuance batches.  The outer policy is the single frozen M165 diagnostic recipe
and is not reselected on Q3 labels.
"""

from __future__ import annotations

import argparse
import csv
import gc
import io
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
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
from run_site_wind_classifier import _add_site_wind_features
from run_site_wind_teacher import _strict_preceding_mask, _validation_mask
from strict_dev_surface import DEV_CUTOFF, development_surface
from xgboost import XGBClassifier

BASE_COLUMNS = ["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh"]
CLASS_WIDTH = 0.02
MIXTURES = (
    (1.00, 0.50),
    (0.75, 0.25),
    (0.75, 0.50),
    (0.75, 0.75),
    (0.50, 0.25),
    (0.50, 0.50),
    (0.50, 0.75),
    (0.00, 1.00),
    (0.00, 0.00),
)
FROZEN_MIXTURE = (0.50, 0.25)
FROZEN_POLICY = "T0.4_G2"
PREQUENTIAL_FRACTIONS = (0.35, 0.57, 0.79, 1.00)
ALLWEATHER_ITERATIONS = {1: 292, 2: 138, 3: 119}


def _finite_mean(row: dict[str, str], names: list[str]) -> float:
    values: list[float] = []
    for name in names:
        raw = row.get(name, "")
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value) and 0.0 <= value < 50.0:
            values.append(value)
    return float(np.mean(values)) if values else np.nan


def _stream_scada_wind(cutoff: pd.Timestamp) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    specifications = (
        (
            "train/scada_vestas_train.csv",
            "vestas",
            ((1, range(1, 7)), (2, range(7, 13))),
        ),
        (
            "train/scada_unison_train.csv",
            "unison",
            ((3, range(1, 6)),),
        ),
    )
    cutoff_text = cutoff.strftime("%Y-%m-%d %H:%M:%S")
    with zipfile.ZipFile(OPEN) as archive:
        for member, prefix, group_specs in specifications:
            with archive.open(member) as binary:
                text = io.TextIOWrapper(binary, encoding="utf-8-sig", newline="")
                reader = csv.DictReader(text)
                for row in reader:
                    timestamp_text = str(row["kst_dtm"])
                    if timestamp_text >= cutoff_text:
                        break
                    for group_id, turbine_numbers in group_specs:
                        names = [
                            f"{prefix}_wtg{number:02d}_ws"
                            for number in turbine_numbers
                        ]
                        records.append(
                            {
                                "raw_time": timestamp_text,
                                "group_id": group_id,
                                "scada_ws": _finite_mean(row, names),
                                "fleet": prefix,
                            }
                        )
    raw = pd.DataFrame.from_records(records)
    if raw.empty:
        raise RuntimeError("pre-cutoff SCADA stream is empty")
    raw["raw_time"] = pd.to_datetime(raw["raw_time"])
    vestas = raw["fleet"].eq("vestas")
    raw["forecast_kst_dtm"] = raw["raw_time"].dt.floor("h") + pd.Timedelta(
        hours=1
    )
    exact_hour = vestas & raw["raw_time"].dt.minute.eq(0)
    raw.loc[exact_hour, "forecast_kst_dtm"] = raw.loc[exact_hour, "raw_time"]
    raw = raw.loc[raw["forecast_kst_dtm"].lt(cutoff)]
    hourly = (
        raw.groupby(["forecast_kst_dtm", "group_id"], as_index=False)["scada_ws"]
        .mean()
        .sort_values(["forecast_kst_dtm", "group_id"])
        .reset_index(drop=True)
    )
    if hourly["forecast_kst_dtm"].ge(cutoff).any():
        raise RuntimeError("SCADA stream crossed the outer cutoff")
    return hourly


def _legacy_columns(feature_columns: list[str]) -> list[str]:
    tokens = (
        "10_10u",
        "10_10v",
        "80_u",
        "80_v",
        "100_100u",
        "100_100v",
        "50mu",
        "50mv",
        "wind10_",
        "wind50",
        "wind80",
        "wind100",
        "gust",
        "group_",
        "lead_hour",
        "month",
    )
    selected = [
        name for name in feature_columns if any(token in name.lower() for token in tokens)
    ]
    if len(selected) < 300:
        raise RuntimeError(f"legacy NWP contract resolved only {len(selected)} columns")
    return selected


def _wind_model(iterations: int, seed: int, leaves: int) -> LGBMRegressor:
    return LGBMRegressor(
        objective="l2",
        n_estimators=iterations,
        learning_rate=0.025 if leaves > 31 else 0.04,
        num_leaves=leaves,
        min_child_samples=40 if leaves > 31 else 60,
        max_bin=255,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.85 if leaves > 31 else 0.8,
        reg_alpha=0.1,
        reg_lambda=3.0 if leaves > 31 else 2.0,
        random_state=seed,
        n_jobs=6,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )


def _prequential_wind(
    surface: pd.DataFrame,
    matrix: pd.DataFrame,
    history: np.ndarray,
    validation: np.ndarray,
    *,
    per_group: bool,
    iterations_by_group: dict[int, int],
    shared_iterations: int,
    seed: int,
    lane: str,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    prediction = np.full(len(surface), np.nan, dtype="float32")
    diagnostics: list[dict[str, object]] = []
    group_ids: tuple[int | None, ...] = (1, 2, 3) if per_group else (None,)
    for group_id in group_ids:
        group = (
            np.ones(len(surface), dtype=bool)
            if group_id is None
            else surface["group_id"].eq(group_id).to_numpy()
        )
        candidate_history = history & group & surface["scada_ws"].notna().to_numpy()
        batches = np.asarray(
            sorted(surface.loc[candidate_history, "data_available_kst_dtm"].unique())
        )
        if len(batches) < 100:
            raise RuntimeError(f"{lane} group {group_id} has too few history batches")
        boundaries = [
            round(len(batches) * fraction) for fraction in PREQUENTIAL_FRACTIONS
        ]
        boundaries[-1] = len(batches)
        start = boundaries[0]
        for segment_index, stop in enumerate(boundaries[1:]):
            apply_batches = set(batches[start:stop])
            apply = candidate_history & surface["data_available_kst_dtm"].isin(
                apply_batches
            ).to_numpy()
            fit = (
                _strict_preceding_mask(surface, apply)
                & group
                & surface["scada_ws"].notna().to_numpy()
            )
            iterations = (
                shared_iterations
                if group_id is None
                else iterations_by_group[group_id]
            )
            model = _wind_model(
                iterations,
                seed + 10 * (group_id or 0) + segment_index,
                63 if per_group else 31,
            )
            model.fit(matrix.loc[fit], surface.loc[fit, "scada_ws"])
            prediction[apply] = model.predict(matrix.loc[apply])
            diagnostics.append(
                {
                    "lane": lane,
                    "group_id": group_id or "shared",
                    "segment": segment_index,
                    "fit_rows": int(fit.sum()),
                    "apply_rows": int(apply.sum()),
                    "iterations": iterations,
                }
            )
            print(json.dumps({"sitewind": diagnostics[-1]}), flush=True)
            del model
            gc.collect()
            start = stop

        fit = history & group & surface["scada_ws"].notna().to_numpy()
        apply = validation & group
        iterations = (
            shared_iterations if group_id is None else iterations_by_group[group_id]
        )
        model = _wind_model(
            iterations,
            seed + 100 + (group_id or 0),
            63 if per_group else 31,
        )
        model.fit(matrix.loc[fit], surface.loc[fit, "scada_ws"])
        prediction[apply] = model.predict(matrix.loc[apply])
        diagnostics.append(
            {
                "lane": lane,
                "group_id": group_id or "shared",
                "segment": "outer",
                "fit_rows": int(fit.sum()),
                "apply_rows": int(apply.sum()),
                "iterations": iterations,
            }
        )
        print(json.dumps({"sitewind": diagnostics[-1]}), flush=True)
        del model
        gc.collect()
    return prediction, diagnostics


def _source_columns(columns: list[str], source: str) -> list[str]:
    common = (
        "sitewind__",
        "hour",
        "month",
        "lead_hour",
        "cal__",
        "group_",
        "capacity",
        "turbine_count",
        "rotor",
        "latitude_centroid",
        "longitude_centroid",
    )
    selected = [
        name
        for name in columns
        if source in name.lower() or any(token in name for token in common)
    ]
    if len(selected) < 200:
        raise RuntimeError(f"{source} source contract resolved {len(selected)} columns")
    return selected


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
    model.fit(
        matrix.loc[training, candidates],
        target.loc[training],
        sample_weight=target.loc[training].clip(lower=0.10),
    )
    gain = model.booster_.feature_importance(importance_type="gain")
    order = np.argsort(gain)[::-1][:count]
    selected = [candidates[position] for position in order]
    del model
    gc.collect()
    return selected


def _source_probability(
    matrix: pd.DataFrame,
    target: pd.Series,
    classes: pd.Series,
    training: np.ndarray,
    validation: np.ndarray,
    candidates: list[str],
    feature_count: int,
    iterations: int,
    class_count: int,
    seed: int,
    source: str,
) -> tuple[np.ndarray, list[str]]:
    selected = _screen(
        matrix, target, training, candidates, feature_count, seed
    )
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=class_count,
        n_estimators=iterations,
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
    model.fit(
        matrix.loc[training, selected],
        classes.loc[training].astype(int),
        sample_weight=target.loc[training].clip(lower=0.10),
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
                "training_rows": int(training.sum()),
                "validation_rows": int(validation.sum()),
                "feature_count": len(selected),
            }
        ),
        flush=True,
    )
    del model, raw
    gc.collect()
    return probability, selected


def _mixture(
    probabilities: dict[str, np.ndarray],
    global_weight: float,
    gfs_share: float,
) -> np.ndarray:
    source = (
        gfs_share * probabilities["gfs"]
        + (1.0 - gfs_share) * probabilities["ldaps"]
    )
    output = global_weight * probabilities["global"] + (1.0 - global_weight) * source
    output /= output.sum(axis=1, keepdims=True)
    return output


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
    parser.add_argument("--global-features", type=int, default=100)
    parser.add_argument("--source-features", type=int, default=160)
    args = parser.parse_args()
    if not 100 <= args.iterations <= 240:
        raise ValueError("iterations must be between 100 and 240")
    if not 80 <= args.global_features <= 160:
        raise ValueError("global-features must be between 80 and 160")
    if not 120 <= args.source_features <= 220:
        raise ValueError("source-features must be between 120 and 220")
    if _sha256(OPEN) != OPEN_SHA or _sha256(BASELINE) != BASELINE_SHA:
        raise RuntimeError("immutable competition input hash mismatch")

    started = time.perf_counter()
    surface, feature_columns = development_surface()
    if surface["forecast_kst_dtm"].ge(DEV_CUTOFF).any():
        raise RuntimeError("lockbox row reached prequential source-rank runner")
    validation = _validation_mask(surface, args.fold)
    history = _strict_preceding_mask(surface, validation)
    outer_cutoff = pd.Timestamp(surface.loc[validation, "forecast_kst_dtm"].min())
    scada = _stream_scada_wind(outer_cutoff)
    surface = surface.merge(
        scada,
        on=["forecast_kst_dtm", "group_id"],
        how="left",
        validate="one_to_one",
    )
    if surface.loc[validation, "scada_ws"].notna().any():
        raise RuntimeError("outer SCADA proxy was materialized")
    allweather_matrix = surface[feature_columns].astype("float32")
    legacy_columns = _legacy_columns(feature_columns)
    legacy, legacy_diagnostics = _prequential_wind(
        surface,
        allweather_matrix[legacy_columns],
        history,
        validation,
        per_group=False,
        iterations_by_group=ALLWEATHER_ITERATIONS,
        shared_iterations=400,
        seed=20260920,
        lane="legacy-shared",
    )
    allweather, allweather_diagnostics = _prequential_wind(
        surface,
        allweather_matrix,
        history,
        validation,
        per_group=True,
        iterations_by_group=ALLWEATHER_ITERATIONS,
        shared_iterations=400,
        seed=20260940,
        lane="allweather-group",
    )
    del allweather_matrix
    gc.collect()

    matrix = surface[feature_columns].astype("float32")
    sitewind_columns = _add_site_wind_features(matrix, legacy, allweather)
    normalized_target = surface["actual_kwh"] / surface["group_id"].map(CAPACITIES)
    training = (
        history
        & surface["actual_kwh"].notna().to_numpy()
        & normalized_target.ge(0.10).to_numpy()
        & np.isfinite(legacy)
        & np.isfinite(allweather)
    )
    for group_id in CAPACITIES:
        if int((training & surface["group_id"].eq(group_id).to_numpy()).sum()) < 1000:
            raise RuntimeError(f"group {group_id} prequential meta-history is too small")
    raw_bins = np.floor(
        (normalized_target.clip(0.10, 1.074999) - 0.10) / CLASS_WIDTH
    ).astype("Int64")
    active_bins = [
        int(value)
        for value in sorted(raw_bins.loc[training].dropna().astype(int).unique())
    ]
    bin_to_class = {bin_id: index for index, bin_id in enumerate(active_bins)}
    classes = raw_bins.map(bin_to_class).astype("Int64")
    centers = np.asarray(
        [
            normalized_target.loc[training & classes.eq(index)].mean()
            for index in range(len(active_bins))
        ],
        dtype=float,
    )
    candidates = list(matrix.columns)
    probabilities: dict[str, np.ndarray] = {}
    selected_features: dict[str, list[str]] = {}
    source_specs = {
        "global": (candidates, args.global_features),
        "gfs": (_source_columns(candidates, "gfs"), args.source_features),
        "ldaps": (_source_columns(candidates, "ldaps"), args.source_features),
    }
    for source_index, (source, (source_candidates, count)) in enumerate(
        source_specs.items()
    ):
        probability, selected = _source_probability(
            matrix,
            normalized_target,
            classes,
            training,
            validation,
            source_candidates,
            count,
            args.iterations,
            len(active_bins),
            20260960 + source_index * 10,
            source,
        )
        probabilities[source] = probability
        selected_features[source] = selected

    groups = surface.loc[validation, "group_id"].to_numpy(dtype=int)
    means = {
        group_id: float(
            normalized_target.loc[
                training & surface["group_id"].eq(group_id).to_numpy()
            ].mean()
        )
        for group_id in CAPACITIES
    }
    fixed_probability = _mixture(probabilities, *FROZEN_MIXTURE)
    fixed_policies = _policy_values(fixed_probability, centers, groups, means)
    fixed_normalized = fixed_policies[FROZEN_POLICY]
    output = _frame(surface, validation, fixed_normalized)
    output["fold_id"] = args.fold
    output["model_id"] = args.candidate_id

    oracle_trials: list[dict[str, object]] = []
    for global_weight, gfs_share in MIXTURES:
        probability = _mixture(probabilities, global_weight, gfs_share)
        for policy, normalized in _policy_values(
            probability, centers, groups, means
        ).items():
            score = _score(_frame(surface, validation, normalized))
            oracle_trials.append(
                {
                    "global_weight": global_weight,
                    "gfs_share": gfs_share,
                    "policy": policy,
                    "score": score,
                }
            )
    oracle_trials.sort(key=lambda item: item["score"]["total"], reverse=True)

    output_path = OUTPUT / f"{args.candidate_id}-{args.fold}.parquet"
    output.to_parquet(output_path, index=False)
    fold_score = _score(output)
    group_scores = _group_scores(output)
    receipt = {
        "schema_version": 1,
        "candidate_id": args.candidate_id,
        "fold_id": args.fold,
        "architecture": "strict_prequential_sitewind_gfs_ldaps_source_rank",
        "scope": (
            "diagnostic reconstruction with outer labels excluded from feature/model fit; "
            "frozen policy inherited from M165 is not promotion evidence"
        ),
        "physical_development_cutoff": str(DEV_CUTOFF),
        "outer_scada_stream_cutoff": str(outer_cutoff),
        "outer_scada_rows_materialized": 0,
        "prequential_fractions": list(PREQUENTIAL_FRACTIONS),
        "legacy_wind_diagnostics": legacy_diagnostics,
        "allweather_wind_diagnostics": allweather_diagnostics,
        "legacy_feature_count": len(legacy_columns),
        "allweather_feature_count": len(feature_columns),
        "sitewind_feature_count": len(sitewind_columns),
        "meta_training_rows": int(training.sum()),
        "meta_training_rows_by_group": {
            str(group_id): int(
                (training & surface["group_id"].eq(group_id).to_numpy()).sum()
            )
            for group_id in CAPACITIES
        },
        "class_width": CLASS_WIDTH,
        "class_count": len(active_bins),
        "active_bins": active_bins,
        "iterations": args.iterations,
        "selected_features": selected_features,
        "frozen_mixture": {
            "global_weight": FROZEN_MIXTURE[0],
            "gfs_share": FROZEN_MIXTURE[1],
        },
        "frozen_policy": FROZEN_POLICY,
        "fold_score": fold_score,
        "group_scores": group_scores,
        "outer_oracle_best": oracle_trials[0],
        "outer_oracle_top10": oracle_trials[:10],
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
    print(
        json.dumps(
            {
                "candidate_id": args.candidate_id,
                "fold_score": fold_score,
                "group_scores": group_scores,
                "outer_oracle_best": oracle_trials[0],
                "meta_training_rows": receipt["meta_training_rows"],
                "runtime_seconds": receipt["runtime_seconds"],
                "prediction_sha256": receipt["prediction_sha256"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
