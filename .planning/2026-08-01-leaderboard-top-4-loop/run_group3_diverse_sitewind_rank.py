"""Strict group-3 source-rank model with diverse site-wind teachers.

CatBoost and ExtraTrees NWP-to-wind predictions are generated prequentially:
the feature screen and every fitted teacher use only complete earlier issuance
batches.  The outer-Q3 SCADA proxy is never streamed.  Groups 1/2 stay exactly
equal to the strict M189 parent while group 3 receives the fixed distribution.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
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
from run_site_wind_teacher import _strict_preceding_mask, _validation_mask
from run_strict_prequential_source_rank import (
    BASE_COLUMNS,
    CLASS_WIDTH,
    FROZEN_MIXTURE,
    FROZEN_POLICY,
    MIXTURES,
    _mixture,
    _source_columns,
    _source_probability,
    _stream_scada_wind,
)
from sklearn.ensemble import ExtraTreesRegressor
from strict_dev_surface import DEV_CUTOFF, development_surface

PARENT_PATH = OUTPUT / "M189_STRICT_PREQUENTIAL_SOURCE_RANK_Q3-dev-2023-Q3.parquet"
TEACHER_FRACTIONS = (0.25, 0.50, 0.75, 1.00)


def _screen_sitewind(
    surface: pd.DataFrame,
    matrix: pd.DataFrame,
    fit: np.ndarray,
    count: int,
) -> list[str]:
    model = LGBMRegressor(
        objective="l1",
        n_estimators=180,
        learning_rate=0.035,
        num_leaves=31,
        min_child_samples=40,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.75,
        reg_alpha=0.1,
        reg_lambda=3.0,
        random_state=20261020,
        n_jobs=6,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )
    model.fit(matrix.loc[fit], surface.loc[fit, "scada_ws"])
    gain = model.booster_.feature_importance(importance_type="gain")
    selected = [matrix.columns[position] for position in np.argsort(gain)[::-1][:count]]
    del model
    gc.collect()
    return selected


def _fit_teacher(
    family: str,
    x_fit: pd.DataFrame,
    target: pd.Series,
    x_apply: pd.DataFrame,
    seed: int,
) -> np.ndarray:
    medians = x_fit.median(axis=0).fillna(0.0)
    train = x_fit.fillna(medians)
    apply = x_apply.fillna(medians)
    if family == "catboost":
        model = CatBoostRegressor(
            loss_function="MAE",
            iterations=600,
            learning_rate=0.035,
            depth=7,
            l2_leaf_reg=5.0,
            random_strength=0.2,
            random_seed=seed,
            thread_count=6,
            allow_writing_files=False,
            verbose=False,
        )
    elif family == "extra_trees":
        model = ExtraTreesRegressor(
            n_estimators=400,
            min_samples_leaf=5,
            max_features=0.75,
            random_state=seed,
            n_jobs=6,
        )
    else:
        raise ValueError(f"unknown site-wind teacher: {family}")
    model.fit(train, target)
    prediction = np.asarray(model.predict(apply), dtype=float)
    del model, train, apply
    gc.collect()
    return prediction


def _prequential_teachers(
    surface: pd.DataFrame,
    matrix: pd.DataFrame,
    history: np.ndarray,
    validation: np.ndarray,
    selected: list[str],
) -> tuple[dict[str, np.ndarray], list[dict[str, object]]]:
    group = surface["group_id"].eq(3).to_numpy()
    candidate = history & group & surface["scada_ws"].notna().to_numpy()
    batches = np.asarray(
        sorted(surface.loc[candidate, "data_available_kst_dtm"].unique())
    )
    boundaries = [
        round(len(batches) * fraction) for fraction in TEACHER_FRACTIONS
    ]
    boundaries[-1] = len(batches)
    predictions = {
        family: np.full(len(surface), np.nan, dtype="float32")
        for family in ("catboost", "extra_trees")
    }
    diagnostics: list[dict[str, object]] = []
    start = boundaries[0]
    for segment_index, stop in enumerate(boundaries[1:]):
        apply_batches = set(batches[start:stop])
        apply = candidate & surface["data_available_kst_dtm"].isin(
            apply_batches
        ).to_numpy()
        fit = (
            _strict_preceding_mask(surface, apply)
            & group
            & surface["scada_ws"].notna().to_numpy()
        )
        for family_index, family in enumerate(predictions):
            predictions[family][apply] = _fit_teacher(
                family,
                matrix.loc[fit, selected],
                surface.loc[fit, "scada_ws"],
                matrix.loc[apply, selected],
                20261030 + 10 * family_index + segment_index,
            )
        diagnostics.append(
            {
                "segment": segment_index,
                "fit_rows": int(fit.sum()),
                "apply_rows": int(apply.sum()),
            }
        )
        print(json.dumps({"sitewind": diagnostics[-1]}), flush=True)
        start = stop

    fit = candidate
    apply = validation & group
    for family_index, family in enumerate(predictions):
        predictions[family][apply] = _fit_teacher(
            family,
            matrix.loc[fit, selected],
            surface.loc[fit, "scada_ws"],
            matrix.loc[apply, selected],
            20261080 + family_index,
        )
    diagnostics.append(
        {
            "segment": "outer",
            "fit_rows": int(fit.sum()),
            "apply_rows": int(apply.sum()),
        }
    )
    print(json.dumps({"sitewind": diagnostics[-1]}), flush=True)
    return predictions, diagnostics


def _add_teacher_features(
    matrix: pd.DataFrame,
    teachers: dict[str, np.ndarray],
) -> list[str]:
    cat = teachers["catboost"]
    extra = teachers["extra_trees"]
    values = {
        "sitewind__catboost": cat,
        "sitewind__extra_trees": extra,
        "sitewind__diverse_mean": 0.5 * (cat + extra),
        "sitewind__diverse_delta": cat - extra,
        "sitewind__diverse_disagreement": np.abs(cat - extra),
    }
    for name, value in values.items():
        matrix[name] = value
    for source in ("catboost", "extra_trees", "diverse_mean"):
        value = matrix[f"sitewind__{source}"]
        matrix[f"sitewind__{source}2"] = value**2
        matrix[f"sitewind__{source}3"] = value**3
        normalized = np.clip((value - 3.0) / 9.0, 0.0, 1.0)
        matrix[f"sitewind__{source}_powercurve"] = normalized**3
    return [name for name in matrix if name.startswith("sitewind__")]


def _group3_score(
    base: pd.DataFrame,
    normalized: np.ndarray,
) -> dict[str, float]:
    actual = base["actual_kwh"].to_numpy(dtype=float) / CAPACITIES[3]
    return _group_total(actual, normalized)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--fold", choices=("dev-2023-Q3",), required=True)
    parser.add_argument("--sitewind-features", type=int, default=300)
    parser.add_argument("--global-features", type=int, default=120)
    parser.add_argument("--source-features", type=int, default=160)
    parser.add_argument("--iterations", type=int, default=150)
    args = parser.parse_args()
    if not 180 <= args.sitewind_features <= 400:
        raise ValueError("sitewind-features must be between 180 and 400")
    if not 80 <= args.global_features <= 180:
        raise ValueError("global-features must be between 80 and 180")
    if not 120 <= args.source_features <= 220:
        raise ValueError("source-features must be between 120 and 220")
    if not 100 <= args.iterations <= 240:
        raise ValueError("iterations must be between 100 and 240")
    if _sha256(OPEN) != OPEN_SHA or _sha256(BASELINE) != BASELINE_SHA:
        raise RuntimeError("immutable competition input hash mismatch")

    started = time.perf_counter()
    surface, feature_columns = development_surface()
    if surface["forecast_kst_dtm"].ge(DEV_CUTOFF).any():
        raise RuntimeError("lockbox row reached diverse site-wind runner")
    validation = _validation_mask(surface, args.fold)
    history = _strict_preceding_mask(surface, validation)
    outer_cutoff = pd.Timestamp(surface.loc[validation, "forecast_kst_dtm"].min())
    surface = surface.merge(
        _stream_scada_wind(outer_cutoff),
        on=["forecast_kst_dtm", "group_id"],
        how="left",
        validate="one_to_one",
    )
    if surface.loc[validation, "scada_ws"].notna().any():
        raise RuntimeError("outer SCADA proxy was materialized")
    base_matrix = surface[feature_columns].astype("float32")
    group = surface["group_id"].eq(3).to_numpy()
    candidate = history & group & surface["scada_ws"].notna().to_numpy()
    batches = np.asarray(
        sorted(surface.loc[candidate, "data_available_kst_dtm"].unique())
    )
    first_start = round(len(batches) * TEACHER_FRACTIONS[0])
    first_stop = round(len(batches) * TEACHER_FRACTIONS[1])
    first_apply = candidate & surface["data_available_kst_dtm"].isin(
        set(batches[first_start:first_stop])
    ).to_numpy()
    screen_fit = (
        _strict_preceding_mask(surface, first_apply)
        & group
        & surface["scada_ws"].notna().to_numpy()
    )
    sitewind_selected = _screen_sitewind(
        surface, base_matrix, screen_fit, args.sitewind_features
    )
    teachers, teacher_diagnostics = _prequential_teachers(
        surface,
        base_matrix,
        history,
        validation,
        sitewind_selected,
    )
    matrix = base_matrix.copy()
    teacher_columns = _add_teacher_features(matrix, teachers)
    target = surface["actual_kwh"] / surface["group_id"].map(CAPACITIES)
    training = (
        history
        & group
        & surface["actual_kwh"].notna().to_numpy()
        & target.ge(0.10).to_numpy()
        & np.isfinite(teachers["catboost"])
        & np.isfinite(teachers["extra_trees"])
    )
    application = validation & group
    if int(training.sum()) < 1500:
        raise RuntimeError("group-3 diverse-teacher meta-history is too small")

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
        [target.loc[training & classes.eq(index)].mean() for index in range(len(active_bins))]
    )
    candidates = list(matrix.columns)
    source_specs = {
        "global": (candidates, args.global_features),
        "gfs": (_source_columns(candidates, "gfs"), args.source_features),
        "ldaps": (_source_columns(candidates, "ldaps"), args.source_features),
    }
    probabilities: dict[str, np.ndarray] = {}
    selected_features: dict[str, list[str]] = {}
    for source_index, (source, (source_candidates, count)) in enumerate(
        source_specs.items()
    ):
        probability, selected = _source_probability(
            matrix,
            target,
            classes,
            training,
            application,
            source_candidates,
            count,
            args.iterations,
            len(active_bins),
            20261100 + 10 * source_index,
            source,
        )
        probabilities[source] = probability
        selected_features[source] = selected

    mean_generation = float(target.loc[training].mean())
    means = {group_id: mean_generation for group_id in CAPACITIES}
    application_groups = np.full(int(application.sum()), 3, dtype=int)
    fixed_probability = _mixture(probabilities, *FROZEN_MIXTURE)
    policies = _policy_values(
        fixed_probability, centers, application_groups, means
    )
    fixed_normalized = policies[FROZEN_POLICY]
    group3_base = surface.loc[application, BASE_COLUMNS].copy()
    group3_score = _group3_score(group3_base, fixed_normalized)

    oracle_trials: list[dict[str, object]] = []
    for global_weight, gfs_share in MIXTURES:
        probability = _mixture(probabilities, global_weight, gfs_share)
        for policy, normalized in _policy_values(
            probability, centers, application_groups, means
        ).items():
            oracle_trials.append(
                {
                    "global_weight": global_weight,
                    "gfs_share": gfs_share,
                    "policy": policy,
                    "score": _group3_score(group3_base, normalized),
                }
            )
    oracle_trials.sort(key=lambda item: item["score"]["total"], reverse=True)

    parent = pd.read_parquet(PARENT_PATH)
    parent_keys = pd.MultiIndex.from_frame(parent[["forecast_id", "group_id"]])
    expected_keys = pd.MultiIndex.from_frame(
        surface.loc[validation, ["forecast_id", "group_id"]]
    )
    if set(parent_keys) != set(expected_keys):
        raise RuntimeError("M189 parent key contract changed")
    replacement = dict(
        zip(group3_base["forecast_id"], fixed_normalized, strict=True)
    )
    output = parent.copy()
    replace = output["group_id"].eq(3)
    output.loc[replace, "prediction_kwh"] = (
        output.loc[replace, "forecast_id"].map(replacement).to_numpy(dtype=float)
        * CAPACITIES[3]
    )
    output["model_id"] = args.candidate_id
    output_path = OUTPUT / f"{args.candidate_id}-{args.fold}.parquet"
    output.to_parquet(output_path, index=False)
    fold_score = _score(output)
    receipt = {
        "schema_version": 1,
        "candidate_id": args.candidate_id,
        "fold_id": args.fold,
        "architecture": "strict_group3_catboost_extratrees_sitewind_source_rank",
        "scope": (
            "outer SCADA excluded; fixed downstream representation screen; "
            "Q3 teacher-family diagnostic means this is not promotion evidence"
        ),
        "physical_development_cutoff": str(DEV_CUTOFF),
        "outer_scada_stream_cutoff": str(outer_cutoff),
        "outer_scada_rows_materialized": 0,
        "teacher_prequential_fractions": list(TEACHER_FRACTIONS),
        "parent_path": str(PARENT_PATH.relative_to(Path.cwd())),
        "parent_sha256": _sha256(PARENT_PATH),
        "sitewind_selected_feature_count": len(sitewind_selected),
        "sitewind_selected_features": sitewind_selected,
        "teacher_feature_count": len(teacher_columns),
        "teacher_diagnostics": teacher_diagnostics,
        "meta_training_rows": int(training.sum()),
        "selected_features": selected_features,
        "class_width": CLASS_WIDTH,
        "class_count": len(active_bins),
        "frozen_mixture": {
            "global_weight": FROZEN_MIXTURE[0],
            "gfs_share": FROZEN_MIXTURE[1],
        },
        "frozen_policy": FROZEN_POLICY,
        "group3_score": group3_score,
        "group3_outer_oracle_best": oracle_trials[0],
        "fold_score": fold_score,
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
                "group3_score": group3_score,
                "group3_outer_oracle_best": oracle_trials[0],
                "fold_score": fold_score,
                "meta_training_rows": int(training.sum()),
                "runtime_seconds": receipt["runtime_seconds"],
                "prediction_sha256": receipt["prediction_sha256"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
