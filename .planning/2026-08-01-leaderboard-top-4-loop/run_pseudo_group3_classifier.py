"""Screen group-3 classifiers augmented with chronology-safe pseudo labels.

Group 3 has no 2022 target.  A compact mapper learned only from preceding dates
transfers the contemporaneous group-1/2 relationship onto the missing 2022
group-3 rows.  Those pseudo labels are training targets only; validation and
inference features remain official NWP-derived columns.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
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
from run_site_wind_classifier import (
    DECISION_GAMMAS,
    DECISION_TEMPERATURES,
    FOLDS,
    _add_site_wind_features,
)
from run_site_wind_teacher import _validation_mask
from xgboost import XGBClassifier

GROUP_ID = 3
PARENT_PATH = OUTPUT / "M103_STRICT_TOP100-oof.parquet"


def _mapper_features(wide: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(index=wide.index)
    result["y1"] = wide[1]
    result["y2"] = wide[2]
    result["mean12"] = wide[[1, 2]].mean(axis=1)
    result["min12"] = wide[[1, 2]].min(axis=1)
    result["max12"] = wide[[1, 2]].max(axis=1)
    result["diff12"] = wide[2] - wide[1]
    for name in ("y1", "y2", "mean12"):
        result[f"{name}2"] = result[name] ** 2
        result[f"{name}3"] = result[name] ** 3
    hour = result.index.hour
    day = result.index.dayofyear
    result["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    result["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    result["doy_sin"] = np.sin(2.0 * np.pi * day / 365.25)
    result["doy_cos"] = np.cos(2.0 * np.pi * day / 365.25)
    return result.astype("float32")


def _weather_transfer_features(
    surface: pd.DataFrame,
    matrix: pd.DataFrame,
    preceding: np.ndarray,
    wide: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """Add official-weather transfer covariates without exposing them at inference."""
    selected: list[str] = []
    geom_terms = (
        "__vector_speed",
        "__mean_point_speed",
        "__coherence",
        "__vector_spread",
        "__layout_along",
        "__layout_cross",
        "__shear",
    )
    for name in matrix.columns:
        if name.startswith("sitewind__"):
            selected.append(name)
        elif name.startswith(("gfs_spatial__", "ldaps_spatial__")) and any(
            term in name
            for term in ("wind", "10u", "10v", "100u", "100v", "MU", "MV")
        ):
            selected.append(name)
        elif name.startswith(("source_disagreement__", "phys_v2__")):
            selected.append(name)
        elif name.startswith("geom__") and "__batch_" not in name and any(
            term in name for term in geom_terms
        ):
            selected.append(name)
        elif name.startswith("geom__align__") and name.endswith(("__cos", "__sin")):
            selected.append(name)
    mapper_surface = pd.concat(
        [
            surface.loc[preceding, ["forecast_kst_dtm", "group_id"]],
            matrix.loc[preceding, selected],
        ],
        axis=1,
    ).copy()
    weather = mapper_surface.pivot(
        index="forecast_kst_dtm", columns="group_id", values=selected
    ).reindex(wide.index)
    result = _mapper_features(wide)
    additions: dict[str, pd.Series] = {}
    for name in selected:
        values = weather[name]
        mean12 = values[[1, 2]].mean(axis=1)
        additions[f"wx__{name}__g3"] = values[3]
        additions[f"wx__{name}__g3_minus_mean12"] = values[3] - mean12
        additions[f"wx__{name}__g2_minus_g1"] = values[2] - values[1]
    for name in (
        "sitewind__legacy_powercurve",
        "sitewind__allweather_powercurve",
        "sitewind__mean_powercurve",
    ):
        if name not in selected:
            continue
        values = weather[name].clip(lower=0.025)
        transfer1 = wide[1] * values[3] / values[1]
        transfer2 = wide[2] * values[3] / values[2]
        additions[f"transfer__{name}__from_g1"] = transfer1.clip(0.0, 1.20)
        additions[f"transfer__{name}__from_g2"] = transfer2.clip(0.0, 1.20)
        additions[f"transfer__{name}__mean"] = pd.concat(
            [transfer1, transfer2], axis=1
        ).mean(axis=1).clip(0.0, 1.20)
    result = pd.concat([result, pd.DataFrame(additions, index=wide.index)], axis=1)
    result = result.replace([np.inf, -np.inf], np.nan).astype("float32")
    return result, selected


def _mapper_params(objective: str, *, alpha: float | None = None) -> dict[str, object]:
    params: dict[str, object] = {
        "objective": objective,
        "n_estimators": 240,
        "learning_rate": 0.02,
        "num_leaves": 11,
        "min_child_samples": 100,
        "max_bin": 127,
        "subsample": 0.9,
        "subsample_freq": 1,
        "colsample_bytree": 0.65,
        "reg_alpha": 0.4,
        "reg_lambda": 6.0,
        "random_state": 20260802,
        "n_jobs": 6,
        "deterministic": True,
        "force_col_wise": True,
        "verbosity": -1,
    }
    if alpha is not None:
        params["alpha"] = alpha
    return params


def _compact_mapper_params() -> dict[str, object]:
    """Return the frozen compact mapper used by the M131 reference screen."""
    return {
        "objective": "l1",
        "n_estimators": 160,
        "learning_rate": 0.025,
        "num_leaves": 7,
        "min_child_samples": 80,
        "max_bin": 127,
        "subsample": 0.9,
        "subsample_freq": 1,
        "colsample_bytree": 0.9,
        "reg_alpha": 0.2,
        "reg_lambda": 4.0,
        "random_state": 20260802,
        "n_jobs": 6,
        "deterministic": True,
        "force_col_wise": True,
        "verbosity": -1,
    }


def _mapper_holdout_score(
    features: pd.DataFrame,
    target: pd.Series,
    observed: pd.Series,
    *,
    compact: bool,
) -> dict[str, float | int]:
    dates = features.index[observed]
    split = max(1, int(len(dates) * 0.80))
    fit_dates = dates[:split]
    apply_dates = dates[split:]
    if len(apply_dates) == 0:
        return {"train_rows": len(fit_dates), "validation_rows": 0}
    params = _compact_mapper_params() if compact else _mapper_params("l1")
    model = LGBMRegressor(**params)
    model.fit(
        features.loc[fit_dates],
        target.loc[fit_dates],
        sample_weight=target.loc[fit_dates].clip(lower=0.10),
    )
    prediction = np.clip(model.predict(features.loc[apply_dates]), 0.0, 1.075)
    score = _group_score(
        target.loc[apply_dates].to_numpy(dtype=float) * CAPACITIES[GROUP_ID],
        prediction * CAPACITIES[GROUP_ID],
    )
    return {
        "train_rows": len(fit_dates),
        "validation_rows": len(apply_dates),
        **score,
    }


def _pseudo_targets(
    surface: pd.DataFrame,
    matrix: pd.DataFrame,
    preceding: np.ndarray,
    mapper_profile: str,
    mapper_blend: float,
) -> tuple[pd.Series, pd.Series, dict[str, object]]:
    history = surface.loc[
        preceding,
        ["forecast_kst_dtm", "group_id", "actual_kwh"],
    ].copy()
    history["normalized_target"] = history["actual_kwh"] / history["group_id"].map(
        CAPACITIES
    )
    wide = history.pivot(
        index="forecast_kst_dtm", columns="group_id", values="normalized_target"
    )
    compact_features = _mapper_features(wide)
    weather_columns: list[str] = []
    if mapper_profile == "weather_transfer":
        features, weather_columns = _weather_transfer_features(
            surface, matrix, preceding, wide
        )
    else:
        features = compact_features
    core_valid = wide[[1, 2]].notna().all(axis=1)
    observed = wide[3].notna() & core_valid
    missing = wide[3].isna() & core_valid
    mapper_params = (
        _compact_mapper_params()
        if mapper_profile == "compact"
        else _mapper_params("l1")
    )
    mapper = LGBMRegressor(**mapper_params)
    mapper.fit(
        features.loc[observed],
        wide.loc[observed, 3],
        sample_weight=wide.loc[observed, 3].clip(lower=0.10),
    )
    values = pd.Series(np.nan, index=surface.index, dtype=float)
    rich_predictions = np.clip(
        mapper.predict(features.loc[missing]), 0.0, 1.075
    )
    predictions = rich_predictions
    confidence_by_time = np.ones(len(predictions), dtype=float)
    if mapper_profile == "weather_transfer":
        compact_mapper = LGBMRegressor(**_compact_mapper_params())
        compact_mapper.fit(
            compact_features.loc[observed],
            wide.loc[observed, 3],
            sample_weight=wide.loc[observed, 3].clip(lower=0.10),
        )
        compact_predictions = np.clip(
            compact_mapper.predict(compact_features.loc[missing]), 0.0, 1.075
        )
        predictions = (
            mapper_blend * rich_predictions
            + (1.0 - mapper_blend) * compact_predictions
        )
        lower = LGBMRegressor(**_mapper_params("quantile", alpha=0.20))
        upper = LGBMRegressor(**_mapper_params("quantile", alpha=0.80))
        for quantile in (lower, upper):
            quantile.fit(
                features.loc[observed],
                wide.loc[observed, 3],
                sample_weight=wide.loc[observed, 3].clip(lower=0.10),
            )
        width = np.maximum(
            upper.predict(features.loc[missing])
            - lower.predict(features.loc[missing]),
            0.005,
        )
        confidence_by_time = np.exp(-width / 0.12)
        confidence_by_time /= max(float(np.mean(confidence_by_time)), 1e-6)
        confidence_by_time = np.clip(confidence_by_time, 0.25, 1.50)
    by_time = dict(zip(features.index[missing], predictions, strict=True))
    confidence_map = dict(
        zip(features.index[missing], confidence_by_time, strict=True)
    )
    pseudo_rows = (
        preceding
        & surface["group_id"].eq(GROUP_ID).to_numpy()
        & surface["actual_kwh"].isna().to_numpy()
    )
    values.loc[pseudo_rows] = surface.loc[
        pseudo_rows, "forecast_kst_dtm"
    ].map(by_time)
    confidence = pd.Series(1.0, index=surface.index, dtype=float)
    confidence.loc[pseudo_rows] = surface.loc[
        pseudo_rows, "forecast_kst_dtm"
    ].map(confidence_map)
    diagnostics = {
        "mapper_profile": mapper_profile,
        "mapper_blend": mapper_blend,
        "mapper_observed_rows": int(observed.sum()),
        "pseudo_rows": int(values.notna().sum()),
        "pseudo_valid_rows": int(values.ge(0.10).sum()),
        "pseudo_mean": float(values.mean()),
        "rich_pseudo_mean": float(np.mean(rich_predictions)),
        "mapper_feature_count": len(features.columns),
        "mapper_weather_source_count": len(weather_columns),
        "mapper_holdout": _mapper_holdout_score(
            features,
            wide[3],
            observed,
            compact=mapper_profile == "compact",
        ),
        "compact_mapper_holdout": _mapper_holdout_score(
            compact_features,
            wide[3],
            observed,
            compact=True,
        ),
        "pseudo_confidence_min": float(confidence.loc[pseudo_rows].min()),
        "pseudo_confidence_mean": float(confidence.loc[pseudo_rows].mean()),
        "pseudo_confidence_max": float(confidence.loc[pseudo_rows].max()),
    }
    return values, confidence, diagnostics


def _pseudo_season_weights(
    surface: pd.DataFrame,
    eligible: np.ndarray,
    pseudo_mask: np.ndarray,
    validation: np.ndarray,
    bandwidth_days: float,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Concentrate fixed pseudo-label mass around the validation season."""
    weights = np.ones(int(eligible.sum()), dtype=float)
    selected_pseudo = pseudo_mask[eligible]
    diagnostics: dict[str, float | int] = {
        "bandwidth_days": bandwidth_days,
        "pseudo_rows": int(selected_pseudo.sum()),
    }
    if bandwidth_days <= 0.0 or not selected_pseudo.any():
        diagnostics.update(
            {
                "validation_center_day": 0.0,
                "weight_min": 1.0,
                "weight_mean": 1.0,
                "weight_max": 1.0,
            }
        )
        return weights, diagnostics

    validation_dates = surface.loc[validation, "forecast_kst_dtm"].sort_values()
    center = validation_dates.iloc[len(validation_dates) // 2]
    center_day = float(center.dayofyear)
    pseudo_dates = surface.loc[eligible, "forecast_kst_dtm"].loc[
        selected_pseudo
    ]
    day = pseudo_dates.dt.dayofyear.to_numpy(dtype=float)
    distance = np.abs(day - center_day)
    distance = np.minimum(distance, 365.25 - distance)
    pseudo_weights = np.exp(-0.5 * (distance / bandwidth_days) ** 2)
    pseudo_weights /= max(float(np.mean(pseudo_weights)), 1e-12)
    pseudo_weights = np.clip(pseudo_weights, 0.10, 4.00)
    weights[selected_pseudo] = pseudo_weights
    diagnostics.update(
        {
            "validation_center_day": center_day,
            "weight_min": float(pseudo_weights.min()),
            "weight_mean": float(pseudo_weights.mean()),
            "weight_max": float(pseudo_weights.max()),
        }
    )
    return weights, diagnostics


def _group_score(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    actual = actual / CAPACITIES[GROUP_ID]
    prediction = prediction / CAPACITIES[GROUP_ID]
    valid = np.isfinite(actual) & (actual >= 0.10)
    actual = actual[valid]
    prediction = prediction[valid]
    error = np.abs(prediction - actual)
    units = np.select([error <= 0.06, error <= 0.08], [4.0, 3.0], default=0.0)
    one_minus_nmae = 1.0 - float(error.mean())
    ficr = float(np.sum(actual * units) / np.sum(actual * 4.0))
    return {
        "total": 0.5 * (one_minus_nmae + ficr),
        "one_minus_nmae": one_minus_nmae,
        "ficr": ficr,
    }


def _actions(
    probability: np.ndarray,
    centers: np.ndarray,
    mean_generation: float,
) -> dict[str, np.ndarray]:
    actions = np.arange(0.075, 1.076, 0.0025)
    error = np.abs(actions[:, None] - centers[None, :])
    units = np.select([error <= 0.06, error <= 0.08], [4.0, 3.0], default=0.0)
    output: dict[str, np.ndarray] = {}
    for temperature in DECISION_TEMPERATURES:
        calibrated = probability ** (1.0 / temperature)
        calibrated /= calibrated.sum(axis=1, keepdims=True)
        for gamma in DECISION_GAMMAS:
            utility = -(calibrated @ error.T) + gamma * (
                calibrated @ (centers[None, :] * units).T
            ) / (4.0 * mean_generation)
            output[f"T{temperature:g}_G{gamma:g}"] = actions[
                np.argmax(utility, axis=1)
            ]
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--fold", choices=FOLDS, default=FOLDS[-1])
    parser.add_argument("--sitewind-cache-id", required=True)
    parser.add_argument(
        "--mapper-profile",
        choices=("compact", "weather_transfer"),
        default="compact",
    )
    parser.add_argument("--mapper-blend", type=float, default=1.0)
    parser.add_argument("--pseudo-weights", nargs="+", type=float, required=True)
    parser.add_argument("--pseudo-season-bandwidth-days", type=float, default=0.0)
    parser.add_argument("--iterations", nargs="+", type=int, default=[40, 60, 80])
    parser.add_argument("--top-features", type=int, default=100)
    parser.add_argument("--family", choices=("lgbm", "xgboost"), default="lgbm")
    args = parser.parse_args()
    if _sha256(OPEN) != OPEN_SHA or _sha256(BASELINE) != BASELINE_SHA:
        raise RuntimeError("immutable competition input hash mismatch")
    if any(not 0.0 <= value <= 2.0 for value in args.pseudo_weights):
        raise ValueError("pseudo weights must be between zero and two")
    if not 0.0 <= args.mapper_blend <= 1.0:
        raise ValueError("mapper blend must be between zero and one")
    if args.pseudo_season_bandwidth_days < 0.0:
        raise ValueError("pseudo season bandwidth must be nonnegative")
    started = time.perf_counter()
    surface, base_columns, _ = _surface()
    validation = _validation_mask(surface, args.fold)
    start = surface.loc[validation, "forecast_kst_dtm"].min()
    preceding = surface["forecast_kst_dtm"].lt(start).to_numpy()
    normalized_target = surface["actual_kwh"] / surface["group_id"].map(CAPACITIES)
    cached = np.load(
        OUTPUT / f"{args.sitewind_cache_id}-{args.fold}-sitewind-features.npz"
    )
    matrix = surface[base_columns].astype("float32")
    sitewind_columns = _add_site_wind_features(
        matrix, cached["legacy"], cached["allweather"]
    )
    pseudo, pseudo_confidence, pseudo_diagnostics = _pseudo_targets(
        surface, matrix, preceding, args.mapper_profile, args.mapper_blend
    )
    target = normalized_target.copy()
    target.loc[pseudo.notna()] = pseudo.loc[pseudo.notna()]
    pseudo_mask = pseudo.notna().to_numpy()
    observed_mask = surface["actual_kwh"].notna().to_numpy()
    group = surface["group_id"].eq(GROUP_ID).to_numpy()
    eligible = preceding & group & target.ge(0.10).to_numpy()
    pseudo_season_weight, pseudo_season_diagnostics = _pseudo_season_weights(
        surface,
        eligible,
        pseudo_mask,
        validation,
        args.pseudo_season_bandwidth_days,
    )

    raw_bins = np.floor((target.clip(0.10, 1.074999) - 0.10) / 0.02).astype(
        "Int64"
    )
    active_bins = [
        int(value)
        for value in sorted(raw_bins.loc[eligible].dropna().astype(int).unique())
    ]
    bin_to_class = {bin_id: class_id for class_id, bin_id in enumerate(active_bins)}
    classes = raw_bins.map(bin_to_class).astype("Int64")
    observed_training = eligible & observed_mask
    centers = np.asarray(
        [
            target.loc[observed_training & classes.eq(class_id)].mean()
            if (observed_training & classes.eq(class_id)).any()
            else target.loc[eligible & classes.eq(class_id)].mean()
            for class_id in range(len(active_bins))
        ]
    )
    lgbm_params = {
        "objective": "multiclass",
        "num_class": len(active_bins),
        "n_estimators": max(args.iterations),
        "learning_rate": 0.025,
        "num_leaves": 15,
        "min_child_samples": 60,
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
    # Screen once using equal pseudo/observed mass, then keep the feature set fixed
    # across the pseudo-weight sweep.
    screen_weights = target.loc[eligible].clip(lower=0.10)
    screen = LGBMClassifier(**lgbm_params)
    screen.fit(
        matrix.loc[eligible],
        classes.loc[eligible].astype(int),
        sample_weight=screen_weights,
    )
    gains = screen.booster_.feature_importance(importance_type="gain")
    selected_positions = np.argsort(gains)[::-1][: args.top_features]
    selected_features = [matrix.columns[index] for index in selected_positions]
    matrix = matrix[selected_features]

    parent = pd.read_parquet(PARENT_PATH)
    parent_fold = parent.loc[parent["fold_id"].eq(args.fold)].copy()
    parent_group = parent_fold.loc[parent_fold["group_id"].eq(GROUP_ID)].copy()
    apply = validation & group
    base = surface.loc[
        apply, ["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh"]
    ].copy()
    best: tuple[float, float, int, str, float, pd.DataFrame] | None = None
    sweep: dict[str, object] = {}
    for pseudo_weight in args.pseudo_weights:
        sample_weight = target.loc[eligible].clip(lower=0.10).to_numpy(dtype=float)
        sample_weight *= np.where(pseudo_mask[eligible], pseudo_weight, 1.0)
        sample_weight *= pseudo_confidence.loc[eligible].to_numpy(dtype=float)
        sample_weight *= pseudo_season_weight
        if args.family == "lgbm":
            model = LGBMClassifier(**lgbm_params)
        else:
            model = XGBClassifier(
                objective="multi:softprob",
                num_class=len(active_bins),
                n_estimators=max(args.iterations),
                learning_rate=0.03,
                max_depth=5,
                min_child_weight=10.0,
                subsample=0.9,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=5.0,
                max_bin=256,
                tree_method="hist",
                random_state=20260802,
                n_jobs=6,
            )
        model.fit(
            matrix.loc[eligible],
            classes.loc[eligible].astype(int),
            sample_weight=sample_weight,
        )
        weight_results: dict[str, object] = {}
        for iteration in sorted(set(args.iterations)):
            if args.family == "lgbm":
                probability = model.predict_proba(
                    matrix.loc[apply], num_iteration=iteration
                )
            else:
                probability = model.predict_proba(
                    matrix.loc[apply], iteration_range=(0, iteration)
                )
            actions = _actions(
                probability,
                centers,
                float(target.loc[observed_training].mean()),
            )
            iteration_results: dict[str, object] = {}
            for policy, normalized in actions.items():
                challenger = normalized * CAPACITIES[GROUP_ID]
                for parent_weight in np.arange(0.0, 1.01, 0.1):
                    prediction = (
                        parent_weight
                        * parent_group["prediction_kwh"].to_numpy(dtype=float)
                        + (1.0 - parent_weight) * challenger
                    )
                    score = _group_score(
                        base["actual_kwh"].to_numpy(dtype=float), prediction
                    )
                    tag = f"{policy}_P{parent_weight:.1f}"
                    iteration_results[tag] = score
                    candidate = (
                        score["total"],
                        pseudo_weight,
                        iteration,
                        policy,
                        float(parent_weight),
                        base.assign(prediction_kwh=prediction),
                    )
                    if best is None or candidate[0] > best[0]:
                        best = candidate
            best_tag = max(
                iteration_results,
                key=lambda name: iteration_results[name]["total"],
            )
            weight_results[str(iteration)] = {
                "best_policy": best_tag,
                "best_score": iteration_results[best_tag],
            }
            print(
                json.dumps(
                    {
                        "pseudo_weight": pseudo_weight,
                        "iteration": iteration,
                        "best_policy": best_tag,
                        "best_score": iteration_results[best_tag],
                    }
                ),
                flush=True,
            )
        sweep[f"{pseudo_weight:g}"] = weight_results
    assert best is not None
    best_group = best[5]
    replace = parent_fold["group_id"].eq(GROUP_ID)
    mapping = dict(
        zip(best_group["forecast_id"], best_group["prediction_kwh"], strict=True)
    )
    parent_fold.loc[replace, "prediction_kwh"] = parent_fold.loc[
        replace, "forecast_id"
    ].map(mapping)
    parent_fold["model_id"] = args.candidate_id
    output_path = OUTPUT / f"{args.candidate_id}-{args.fold}.parquet"
    parent_fold.to_parquet(output_path, index=False)
    receipt = {
        "schema_version": 1,
        "candidate_id": args.candidate_id,
        "fold_id": args.fold,
        "architecture": "group3_pseudo_label_multiclass",
        "family": args.family,
        "scope": "preceding-only pseudo labels; validation inference uses NWP only",
        "pseudo_diagnostics": pseudo_diagnostics,
        "pseudo_season_diagnostics": pseudo_season_diagnostics,
        "pseudo_weights": args.pseudo_weights,
        "selected_pseudo_weight": best[1],
        "selected_iteration": best[2],
        "selected_policy": best[3],
        "selected_parent_weight": best[4],
        "selected_group3_score": _group_score(
            best_group["actual_kwh"].to_numpy(dtype=float),
            best_group["prediction_kwh"].to_numpy(dtype=float),
        ),
        "fold_score": _score(parent_fold),
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
