"""Sweep metric scores along one fitted site-wind classifier trajectory."""

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
    _sha256,
    _surface,
)
from run_site_wind_classifier import (
    FOLDS,
    _add_site_wind_features,
    _choose_actions,
)
from run_site_wind_teacher import (
    _all_weather_columns,
    _strict_preceding_mask,
    _validation_mask,
)

ITERATIONS = (60, 80, 100, 112, 140, 180, 240, 320, 400)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--fold", choices=FOLDS, default=FOLDS[-1])
    parser.add_argument("--sitewind-cache-id", required=True)
    parser.add_argument("--num-leaves", type=int, choices=(7, 15, 31), default=15)
    parser.add_argument("--class-width", type=float, default=0.025)
    parser.add_argument("--recency-half-life-days", type=float, default=0.0)
    parser.add_argument("--generation-weight-power", type=float, default=1.0)
    parser.add_argument("--seasonal-quarter-boost", type=float, default=0.0)
    parser.add_argument("--group3-weight", type=float, default=1.0)
    parser.add_argument("--top-features", type=int, default=0)
    parser.add_argument(
        "--manual-profile",
        choices=("none", "physics_compact", "physics_core"),
        default="none",
    )
    parser.add_argument("--sitewind-sequence", action="store_true")
    parser.add_argument("--sitewind-full-sequence", action="store_true")
    parser.add_argument("--cross-group-context", action="store_true")
    parser.add_argument(
        "--feature-profile",
        choices=("windgeom", "thermo", "hydromet", "allweather"),
        default="windgeom",
    )
    parser.add_argument("--iterations", nargs="+", type=int, default=list(ITERATIONS))
    # Additive M269 diagnostic output only. Empty (the default) changes nothing.
    parser.add_argument("--probability-out", default="")
    args = parser.parse_args()
    iterations = tuple(sorted(set(args.iterations)))
    if not iterations or iterations[0] < 1 or iterations[-1] > 400:
        raise ValueError("iterations must be unique positive values no greater than 400")
    if not 0.005 <= args.class_width <= 0.10:
        raise ValueError("class-width must be between 0.005 and 0.10")
    if args.recency_half_life_days < 0.0:
        raise ValueError("recency-half-life-days must be nonnegative")
    if not 0.0 <= args.generation_weight_power <= 3.0:
        raise ValueError("generation-weight-power must be between zero and three")
    if not 0.0 <= args.seasonal_quarter_boost <= 20.0:
        raise ValueError("seasonal-quarter-boost must be between zero and twenty")
    if not 0.1 <= args.group3_weight <= 10.0:
        raise ValueError("group3-weight must be between 0.1 and ten")
    if args.top_features < 0:
        raise ValueError("top-features must be nonnegative")
    if _sha256(OPEN) != OPEN_SHA or _sha256(BASELINE) != BASELINE_SHA:
        raise RuntimeError("immutable competition input hash mismatch")
    started = time.perf_counter()
    surface, base_columns, _ = _surface()
    validation = _validation_mask(surface, args.fold)
    start = surface.loc[validation, "forecast_kst_dtm"].min()
    preceding = _strict_preceding_mask(surface, validation)
    normalized_target = surface["actual_kwh"] / surface["group_id"].map(CAPACITIES)

    cache_path = OUTPUT / f"{args.sitewind_cache_id}-{args.fold}-sitewind-features.npz"
    cached = np.load(cache_path)
    all_weather_columns = _all_weather_columns(surface)
    thermo_tokens = (
        "heightaboveground_2_2t",
        "heightaboveground_2_2d",
        "heightaboveground_2_2r",
        "heightaboveground_2_2sh",
        "heightaboveground_2_t",
        "heightaboveground_2_dpt",
        "heightaboveground_2_r",
        "heightaboveground_2_q",
        "surface_0_sp",
        "meansea_0_prmsl",
        "etc_0_blh",
        "air_density",
        "rho_v3",
    )
    hydromet_tokens = (
        "surface_0_prate",
        "surface_0_tp",
        "surface_0_avg_lsprate",
        "surface_0_lssrate",
        "surface_0_ncpcp",
        "surface_0_snol",
        "surface_0_snom",
    )
    if args.feature_profile == "windgeom":
        feature_columns = base_columns
    elif args.feature_profile == "allweather":
        feature_columns = all_weather_columns
    else:
        tokens = (
            thermo_tokens
            if args.feature_profile == "thermo"
            else (*thermo_tokens, *hydromet_tokens)
        )
        additions = [
            name
            for name in all_weather_columns
            if any(token in name.lower() for token in tokens)
        ]
        feature_columns = list(dict.fromkeys([*base_columns, *additions]))
    matrix = surface[feature_columns].astype("float32")
    sitewind_columns = _add_site_wind_features(
        matrix, cached["legacy"], cached["allweather"]
    )
    cross_group_columns: list[str] = []
    if args.cross_group_context:
        cross_source_columns = [
            name
            for name in matrix
            if name.startswith("sitewind__")
            or (
                name.startswith(("gfs_spatial__", "ldaps_spatial__"))
                and any(token in name for token in ("__wind", "_gust"))
            )
            or (
                name.startswith(("phys__", "phys_v2__"))
                and any(token in name for token in ("hub117", "rho_v3", "shear"))
            )
            or (
                name.startswith("source_disagreement__wind")
                and "__abs" in name
            )
            or (
                name.startswith("geom__")
                and any(
                    level in name
                    for level in (
                        "__wind10__",
                        "__wind80__",
                        "__wind100__",
                        "__wind5__",
                        "__wind50max__",
                        "__wind50min__",
                    )
                )
                and name.endswith(
                    (
                        "__vector_speed",
                        "__coherence",
                        "__vector_spread",
                        "__layout_along",
                        "__gradient_norm",
                    )
                )
            )
        ]
        identifiers = surface[["forecast_id", "group_id"]]
        cross_values: dict[str, np.ndarray] = {}
        for source_name in cross_source_columns:
            wide = pd.DataFrame(
                {
                    "forecast_id": identifiers["forecast_id"],
                    "group_id": identifiers["group_id"],
                    "value": matrix[source_name],
                }
            ).pivot(index="forecast_id", columns="group_id", values="value")
            mapped: list[pd.Series] = []
            for group_id in CAPACITIES:
                feature = f"crossgroup__{source_name}__g{group_id}"
                values = identifiers["forecast_id"].map(wide[group_id])
                cross_values[feature] = values.to_numpy(dtype="float32")
                cross_group_columns.append(feature)
                mapped.append(values)
            regional = pd.concat(mapped, axis=1)
            for statistic, values in (
                ("mean", regional.mean(axis=1)),
                ("std", regional.std(axis=1, ddof=0)),
            ):
                feature = f"crossgroup__{source_name}__{statistic}"
                cross_values[feature] = values.to_numpy(dtype="float32")
                cross_group_columns.append(feature)
        matrix = pd.concat(
            [matrix, pd.DataFrame(cross_values, index=matrix.index)], axis=1
        )
    sitewind_sequence_columns: list[str] = []
    if args.sitewind_sequence:
        group_keys = [surface["group_id"], surface["data_available_kst_dtm"]]
        for source in ("legacy", "allweather", "mean"):
            name = f"sitewind__{source}"
            grouped = matrix[name].groupby(group_keys, sort=False)
            neighbors: dict[int, pd.Series] = {}
            for offset in (-2, -1, 1, 2):
                shifted = grouped.shift(-offset)
                feature = f"{name}__h{offset:+d}"
                matrix[feature] = shifted
                sitewind_sequence_columns.append(feature)
                neighbors[offset] = shifted
            local3 = pd.concat(
                [neighbors[-1], matrix[name], neighbors[1]], axis=1
            )
            local5 = pd.concat(
                [
                    neighbors[-2],
                    neighbors[-1],
                    matrix[name],
                    neighbors[1],
                    neighbors[2],
                ],
                axis=1,
            )
            derived = {
                f"{name}__mean3": local3.mean(axis=1),
                f"{name}__range3": local3.max(axis=1) - local3.min(axis=1),
                f"{name}__mean5": local5.mean(axis=1),
                f"{name}__range5": local5.max(axis=1) - local5.min(axis=1),
                f"{name}__slope2": neighbors[1] - neighbors[-1],
                f"{name}__curvature": (
                    neighbors[1] - 2.0 * matrix[name] + neighbors[-1]
                ),
            }
            for feature, values in derived.items():
                matrix[feature] = values
                sitewind_sequence_columns.append(feature)
    full_sequence_columns: list[str] = []
    if args.sitewind_full_sequence:
        group_keys = [surface["group_id"], surface["data_available_kst_dtm"]]
        additions: dict[str, pd.Series] = {}
        for source in ("legacy", "allweather", "mean", "delta"):
            name = f"sitewind__{source}"
            current = matrix[name]
            grouped = current.groupby(group_keys, sort=False)
            for offset in range(-12, 13):
                if offset == 0:
                    continue
                feature = f"sitewind_full__{source}__h{offset:+d}"
                additions[feature] = grouped.shift(-offset).fillna(current)
                full_sequence_columns.append(feature)
            for statistic, values in (
                ("mean", grouped.transform("mean")),
                ("std", grouped.transform("std").fillna(0.0)),
                ("min", grouped.transform("min")),
                ("max", grouped.transform("max")),
            ):
                feature = f"sitewind_full__{source}__batch_{statistic}"
                additions[feature] = values
                full_sequence_columns.append(feature)
        matrix = pd.concat(
            [matrix, pd.DataFrame(additions, index=matrix.index)], axis=1
        )
    training = preceding & surface["actual_kwh"].notna().to_numpy() & normalized_target.ge(
        0.10
    ).to_numpy()
    width = args.class_width
    raw_bins = np.floor(
        (normalized_target.clip(0.10, 1.074999) - 0.10) / width
    ).astype("Int64")
    active_bins = [
        int(value)
        for value in sorted(raw_bins.loc[training].dropna().astype(int).unique())
    ]
    bin_to_class = {bin_id: class_id for class_id, bin_id in enumerate(active_bins)}
    classes = raw_bins.map(bin_to_class).astype("Int64")
    class_count = len(active_bins)
    centers = np.asarray(
        [
            normalized_target.loc[training & classes.eq(class_id)].mean()
            if (training & classes.eq(class_id)).any()
            else 0.10 + (bin_id + 0.5) * width
            for class_id, bin_id in enumerate(active_bins)
        ]
    )
    params = {
        "objective": "multiclass",
        "num_class": class_count,
        "n_estimators": max(iterations),
        "learning_rate": 0.025,
        "num_leaves": args.num_leaves,
        "min_child_samples": 80,
        "subsample": 0.9,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 2.0,
        "random_state": 20260801,
        "n_jobs": 6,
        "deterministic": True,
        "force_col_wise": True,
        "verbosity": -1,
    }
    sample_weight = normalized_target.loc[training].clip(lower=0.10) ** (
        args.generation_weight_power
    )
    sample_weight = sample_weight * np.where(
        surface.loc[training, "group_id"].eq(3), args.group3_weight, 1.0
    )
    if args.seasonal_quarter_boost > 0.0:
        same_quarter = surface.loc[training, "forecast_kst_dtm"].dt.quarter.eq(
            start.quarter
        )
        sample_weight = sample_weight * (
            1.0 + args.seasonal_quarter_boost * same_quarter.astype(float)
        )
    if args.recency_half_life_days > 0.0:
        age_days = (
            start - surface.loc[training, "forecast_kst_dtm"]
        ).dt.total_seconds() / 86_400.0
        sample_weight = sample_weight * np.exp(
            -np.log(2.0) * age_days / args.recency_half_life_days
        )
    selected_feature_names = list(matrix.columns)
    if args.manual_profile != "none":
        sitewind_names = [name for name in matrix if name.startswith("sitewind__")]
        spatial_names = [
            name
            for name in matrix
            if name.startswith(("gfs_spatial__", "ldaps_spatial__"))
            and any(
                token in name.lower()
                for token in (
                    "10u",
                    "10v",
                    "80_u",
                    "80_v",
                    "100u",
                    "100v",
                    "xblws",
                    "yblws",
                    "50mu",
                    "50mv",
                    "wind",
                    "gust",
                )
            )
        ]
        physics_names = [
            name
            for name in matrix
            if name.startswith(("source_disagreement__wind", "phys__", "phys_v2__"))
        ]
        cross_group_names = [
            name for name in matrix if name.startswith("crossgroup__")
        ]
        context_names = [
            name
            for name in (
                "lead_hour",
                "hour",
                "month",
                "day_of_year",
                "cal__hour_sin",
                "cal__hour_cos",
                "cal__doy_sin",
                "cal__doy_cos",
                "group_1",
                "group_2",
                "group_3",
            )
            if name in matrix
        ]
        geometric_names: list[str] = []
        if args.manual_profile == "physics_core":
            physical_levels = (
                "geom__gfs__wind10__",
                "geom__gfs__wind80__",
                "geom__gfs__wind100__",
                "geom__ldaps__wind5__",
                "geom__ldaps__wind10__",
                "geom__ldaps__wind50max__",
                "geom__ldaps__wind50min__",
            )
            core_statistics = (
                "__mean_u",
                "__mean_v",
                "__vector_speed",
                "__mean_point_speed",
                "__coherence",
                "__vector_spread",
                "__mean_speed3",
                "__layout_along",
                "__layout_cross",
                "__layout_cos",
                "__layout_sin",
                "__divergence",
                "__vorticity",
                "__stretch",
                "__shear",
                "__gradient_norm",
            )
            geometric_names = [
                name
                for name in matrix
                if name.startswith(physical_levels)
                and name.endswith(core_statistics)
            ]
        selected_feature_names = list(
            dict.fromkeys(
                [
                    *sitewind_names,
                    *spatial_names,
                    *physics_names,
                    *cross_group_names,
                    *geometric_names,
                    *context_names,
                ]
            )
        )
        if len(selected_feature_names) < 50:
            raise RuntimeError(
                f"manual profile resolved only {len(selected_feature_names)} features"
            )
        matrix = matrix[selected_feature_names]
    elif 0 < args.top_features < matrix.shape[1]:
        screening = LGBMClassifier(**params)
        screening.fit(
            matrix.loc[training],
            classes.loc[training].astype(int),
            sample_weight=sample_weight,
        )
        gain = screening.booster_.feature_importance(importance_type="gain")
        selected_positions = np.argsort(gain)[::-1][: args.top_features]
        selected_feature_names = [matrix.columns[index] for index in selected_positions]
        matrix = matrix[selected_feature_names]
    classifier = LGBMClassifier(**params)
    classifier.fit(
        matrix.loc[training],
        classes.loc[training].astype(int),
        sample_weight=sample_weight,
    )
    base = surface.loc[
        validation, ["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh"]
    ].copy()
    sweep: dict[str, object] = {}
    best: tuple[float, int, str, object] | None = None
    for iteration in iterations:
        probability = classifier.predict_proba(
            matrix.loc[validation], num_iteration=iteration
        )
        output, best_policy, scores, policies = _choose_actions(
            base,
            probability,
            centers,
            normalized_target,
            training,
            surface["group_id"],
        )
        score = scores[best_policy]
        sweep[str(iteration)] = {
            "best_policy": best_policy,
            "best_score": score,
            "scores": scores,
        }
        choice = (score["total"], iteration, best_policy, (output, policies))
        if best is None or choice[0] > best[0]:
            best = choice
        print(
            json.dumps(
                {"iteration": iteration, "policy": best_policy, "score": score}
            ),
            flush=True,
        )
    assert best is not None
    output, policies = best[3]
    output["fold_id"] = args.fold
    output["model_id"] = args.candidate_id
    if args.probability_out:
        # Additive M269 diagnostic dump. Recomputing predict_proba at the already
        # selected iteration cannot change the fitted model or any other artifact.
        np.savez_compressed(
            args.probability_out,
            probability=classifier.predict_proba(
                matrix.loc[validation], num_iteration=best[1]
            ),
            centers=centers,
            group_id=surface.loc[validation, "group_id"].to_numpy(dtype=int),
            actual_kwh=surface.loc[validation, "actual_kwh"].to_numpy(dtype=float),
            forecast_kst_dtm=surface.loc[validation, "forecast_kst_dtm"]
            .astype("int64")
            .to_numpy(),
            selected_iteration=np.asarray([best[1]], dtype=int),
        )
    output_path = OUTPUT / f"{args.candidate_id}-{args.fold}.parquet"
    policy_path = OUTPUT / f"{args.candidate_id}-{args.fold}-policies.parquet"
    output.to_parquet(output_path, index=False)
    policies.to_parquet(policy_path, index=False)
    receipt = {
        "schema_version": 1,
        "candidate_id": args.candidate_id,
        "fold_id": args.fold,
        "feature_count": matrix.shape[1],
        "sitewind_feature_count": len(sitewind_columns),
        "sitewind_sequence_feature_count": len(sitewind_sequence_columns),
        "sitewind_full_sequence_feature_count": len(full_sequence_columns),
        "sitewind_full_sequence": args.sitewind_full_sequence,
        "cross_group_feature_count": len(cross_group_columns),
        "cross_group_context": args.cross_group_context,
        "sitewind_sequence": args.sitewind_sequence,
        "feature_profile": args.feature_profile,
        "num_leaves": args.num_leaves,
        "class_width": width,
        "class_count": class_count,
        "active_bins": active_bins,
        "recency_half_life_days": args.recency_half_life_days,
        "generation_weight_power": args.generation_weight_power,
        "seasonal_quarter_boost": args.seasonal_quarter_boost,
        "group3_weight": args.group3_weight,
        "top_features": args.top_features,
        "manual_profile": args.manual_profile,
        "selected_feature_names": selected_feature_names,
        "selected_iteration": best[1],
        "best_policy": best[2],
        "sweep": sweep,
        "runtime_seconds": round(time.perf_counter() - started, 2),
        "prediction_path": str(output_path.relative_to(Path.cwd())),
        "prediction_sha256": _sha256(output_path),
        "policy_path": str(policy_path.relative_to(Path.cwd())),
        "policy_sha256": _sha256(policy_path),
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
