"""Add nonlinear atmospheric-regime features to the strict PLS source rank."""

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
    _source_columns,
    _source_probability,
)
from strict_dev_surface import DEV_CUTOFF, development_surface

CLASS_WIDTH = 0.02
ITERATIONS = 150
GLOBAL_FEATURES = 160
SOURCE_FEATURES = 220
PARENT_PATH = OUTPUT / "M197_STRICT_FIXED_HALF_PLS-dev-2023-Q3.parquet"


def _ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    safe = denominator.where(denominator.abs().gt(1e-3))
    return (numerator / safe).replace([np.inf, -np.inf], np.nan)


def _magnus_rh(temperature_k: pd.Series, dewpoint_k: pd.Series) -> pd.Series:
    temperature_c = temperature_k - 273.15
    dewpoint_c = dewpoint_k - 273.15
    exponent = (
        17.625 * dewpoint_c / (243.04 + dewpoint_c)
        - 17.625 * temperature_c / (243.04 + temperature_c)
    )
    return (100.0 * np.exp(exponent.clip(-20.0, 5.0))).clip(0.0, 105.0)


def _atmospheric_features(surface: pd.DataFrame) -> pd.DataFrame:
    features: dict[str, pd.Series] = {}
    statistics = ("mean", "std", "min", "max", "q10", "q50", "q90")
    for statistic in statistics:
        suffix = f"__{statistic}"
        gfs = {
            "t2": surface[f"gfs__heightAboveGround_2_2t{suffix}"],
            "td2": surface[f"gfs__heightAboveGround_2_2d{suffix}"],
            "t850": surface[f"gfs__isobaricInhPa_850_t{suffix}"],
            "t700": surface[f"gfs__isobaricInhPa_700_t{suffix}"],
            "t500": surface[f"gfs__isobaricInhPa_500_t{suffix}"],
            "w10": surface[f"gfs__wind10_speed{suffix}"],
            "w80": surface[f"gfs__wind80_speed{suffix}"],
            "w100": surface[f"gfs__wind100_speed{suffix}"],
            "wpbl": surface[f"gfs__pbl_wind_speed{suffix}"],
            "w850": surface[f"gfs__wind850_speed{suffix}"],
            "gust": surface[f"gfs__surface_0_gust{suffix}"],
            "vrate": surface[f"gfs__planetaryBoundaryLayer_0_VRATE{suffix}"],
            "pressure": surface[f"gfs__surface_0_sp{suffix}"],
        }
        ldaps = {
            "t2": surface[f"ldaps__heightAboveGround_2_t{suffix}"],
            "rh2": surface[f"ldaps__heightAboveGround_2_r{suffix}"],
            "w5": surface[f"ldaps__wind5_speed{suffix}"],
            "w10": surface[f"ldaps__wind10_speed{suffix}"],
            "w50max": surface[f"ldaps__wind50max_speed{suffix}"],
            "w50min": surface[f"ldaps__wind50min_speed{suffix}"],
            "pressure": surface[f"ldaps__surface_0_sp{suffix}"],
        }
        theta850 = gfs["t850"] * (1000.0 / 850.0) ** 0.286
        theta700 = gfs["t700"] * (1000.0 / 700.0) ** 0.286
        theta500 = gfs["t500"] * (1000.0 / 500.0) ** 0.286
        wind_shear = gfs["w850"] - gfs["w10"]
        prefix = f"atm__gfs__{statistic}"
        features[f"{prefix}__dewpoint_depression"] = gfs["t2"] - gfs["td2"]
        features[f"{prefix}__rh_magnus"] = _magnus_rh(gfs["t2"], gfs["td2"])
        features[f"{prefix}__theta850_minus_t2"] = theta850 - gfs["t2"]
        features[f"{prefix}__theta700_minus_theta850"] = theta700 - theta850
        features[f"{prefix}__theta500_minus_theta700"] = theta500 - theta700
        features[f"{prefix}__gust_excess"] = gfs["gust"] - gfs["w10"]
        features[f"{prefix}__gust_factor"] = _ratio(gfs["gust"], gfs["w10"])
        features[f"{prefix}__w100_w10_ratio"] = _ratio(gfs["w100"], gfs["w10"])
        features[f"{prefix}__w80_w10_ratio"] = _ratio(gfs["w80"], gfs["w10"])
        features[f"{prefix}__pbl_w10_ratio"] = _ratio(gfs["wpbl"], gfs["w10"])
        features[f"{prefix}__alpha_80_10"] = _ratio(
            np.log(gfs["w80"].clip(lower=0.05) / gfs["w10"].clip(lower=0.05)),
            pd.Series(np.log(8.0), index=surface.index),
        )
        features[f"{prefix}__alpha_100_80"] = _ratio(
            np.log(gfs["w100"].clip(lower=0.05) / gfs["w80"].clip(lower=0.05)),
            pd.Series(np.log(1.25), index=surface.index),
        )
        features[f"{prefix}__bulk_richardson_proxy"] = _ratio(
            theta850 - gfs["t2"], wind_shear * wind_shear + 0.25
        )
        features[f"{prefix}__vrate_per_wind"] = _ratio(gfs["vrate"], gfs["w10"])
        features[f"{prefix}__moist_air_density_proxy"] = _ratio(
            gfs["pressure"],
            287.05 * gfs["t2"] * (1.0 + 0.00061 * features[f"{prefix}__rh_magnus"]),
        )

        prefix = f"atm__ldaps__{statistic}"
        envelope = ldaps["w50max"] - ldaps["w50min"]
        midpoint = 0.5 * (ldaps["w50max"] + ldaps["w50min"])
        features[f"{prefix}__w50_envelope"] = envelope
        features[f"{prefix}__w50_midpoint"] = midpoint
        features[f"{prefix}__w50_asymmetry"] = _ratio(envelope, midpoint)
        features[f"{prefix}__w50max_w10_ratio"] = _ratio(
            ldaps["w50max"], ldaps["w10"]
        )
        features[f"{prefix}__w50min_w10_ratio"] = _ratio(
            ldaps["w50min"], ldaps["w10"]
        )
        features[f"{prefix}__w10_w5_ratio"] = _ratio(ldaps["w10"], ldaps["w5"])
        features[f"{prefix}__moist_air_density_proxy"] = _ratio(
            ldaps["pressure"],
            287.05 * ldaps["t2"] * (1.0 + 0.00061 * ldaps["rh2"]),
        )

        prefix = f"atm__cross__gfs_ldaps__{statistic}"
        features[f"{prefix}__t2_delta"] = gfs["t2"] - ldaps["t2"]
        features[f"{prefix}__pressure_delta"] = gfs["pressure"] - ldaps["pressure"]
        features[f"{prefix}__rh_delta"] = (
            features[f"atm__gfs__{statistic}__rh_magnus"] - ldaps["rh2"]
        )
        features[f"{prefix}__wind10_ratio"] = _ratio(gfs["w10"], ldaps["w10"])

    for interpolation in ("idw", "nearest"):
        gfs_prefix = f"gfs_spatial__{interpolation}"
        ldaps_prefix = f"ldaps_spatial__{interpolation}"
        g10 = surface[f"{gfs_prefix}__wind10_speed"]
        g80 = surface[f"{gfs_prefix}__wind80_speed"]
        g100 = surface[f"{gfs_prefix}__wind100_speed"]
        gust = surface[f"{gfs_prefix}__surface_0_gust"]
        l5 = surface[f"{ldaps_prefix}__wind5_speed"]
        l10 = surface[f"{ldaps_prefix}__wind10_speed"]
        lmax = surface[f"{ldaps_prefix}__wind50max_speed"]
        lmin = surface[f"{ldaps_prefix}__wind50min_speed"]
        prefix = f"atm__gfs__spatial_{interpolation}"
        features[f"{prefix}__gust_factor"] = _ratio(gust, g10)
        features[f"{prefix}__w100_w10_ratio"] = _ratio(g100, g10)
        features[f"{prefix}__w80_w10_ratio"] = _ratio(g80, g10)
        features[f"{prefix}__alpha_100_80"] = np.log(
            g100.clip(lower=0.05) / g80.clip(lower=0.05)
        ) / np.log(1.25)
        prefix = f"atm__ldaps__spatial_{interpolation}"
        features[f"{prefix}__w50_envelope"] = lmax - lmin
        features[f"{prefix}__w50_asymmetry"] = _ratio(lmax - lmin, lmax + lmin)
        features[f"{prefix}__w50max_w10_ratio"] = _ratio(lmax, l10)
        features[f"{prefix}__w10_w5_ratio"] = _ratio(l10, l5)
        features[f"atm__cross__spatial_{interpolation}__wind10_ratio"] = _ratio(
            g10, l10
        )

    output = pd.DataFrame(features, index=surface.index).astype("float32")
    if output.shape[1] < 100:
        raise RuntimeError("atmospheric feature contract resolved too few columns")
    return output.replace([np.inf, -np.inf], np.nan)


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
        raise RuntimeError("lockbox row reached atmospheric-regime runner")
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
    atmospheric = _atmospheric_features(surface)
    atmospheric_columns = list(atmospheric)
    matrix = pd.concat(
        [
            base_matrix,
            atmospheric,
            pd.DataFrame(parent_values, columns=parent_columns, index=surface.index),
            multioutput,
        ],
        axis=1,
    )
    del base_matrix, atmospheric, parent_values, multioutput
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
    cross_atmosphere = [name for name in atmospheric_columns if "__cross__" in name]
    source_specs = {
        "global": (all_candidates, GLOBAL_FEATURES),
        "gfs": (
            list(
                dict.fromkeys(
                    [
                        *_source_columns(all_candidates, "gfs"),
                        *global_latent,
                        *cross_atmosphere,
                    ]
                )
            ),
            SOURCE_FEATURES,
        ),
        "ldaps": (
            list(
                dict.fromkeys(
                    [
                        *_source_columns(all_candidates, "ldaps"),
                        *global_latent,
                        *cross_atmosphere,
                    ]
                )
            ),
            SOURCE_FEATURES,
        ),
    }
    probabilities: dict[str, np.ndarray] = {}
    selected_features: dict[str, list[str]] = {}
    for source_index, (source, (candidates, count)) in enumerate(source_specs.items()):
        probability, selected = _source_probability(
            matrix,
            target,
            classes,
            training,
            validation,
            candidates,
            count,
            ITERATIONS,
            len(active_bins),
            20262400 + source_index * 10,
            source,
        )
        probabilities[source] = probability
        selected_features[source] = selected
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

    selected_atmospheric_counts = {
        source: sum(name.startswith("atm__") for name in names)
        for source, names in selected_features.items()
    }
    receipt = {
        "schema_version": 1,
        "candidate_id": args.candidate_id,
        "fold_id": args.fold,
        "architecture": "strict_atmospheric_regime_m195_m196_source_rank_half_m197",
        "scope": (
            "fixed official-data-only nonlinear atmospheric feature screen; outer Q3 "
            "labels excluded from every feature, PLS, classifier, policy, and blend fit"
        ),
        "physical_development_cutoff": str(DEV_CUTOFF),
        "strict_training_rows": int(training.sum()),
        "atmospheric_feature_count": len(atmospheric_columns),
        "class_width": CLASS_WIDTH,
        "class_count": len(active_bins),
        "selected_features": selected_features,
        "selected_atmospheric_feature_counts": selected_atmospheric_counts,
        "selected_latent_feature_counts": {
            source: sum(name.startswith(("pls__", "mpls__")) for name in names)
            for source, names in selected_features.items()
        },
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
                "selected_atmospheric_feature_counts": (
                    selected_atmospheric_counts
                ),
                "runtime_seconds": receipt["runtime_seconds"],
                "prediction_sha256": receipt["prediction_sha256"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
