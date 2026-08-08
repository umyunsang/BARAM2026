"""Add simultaneous group-1/group-2 donor PLS axes to the M195 surface."""

from __future__ import annotations

import argparse
import gc
import json
import time
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
from run_site_wind_teacher import _strict_preceding_mask, _validation_mask
from run_strict_prequential_source_rank import (
    FROZEN_MIXTURE,
    FROZEN_POLICY,
    _mixture,
    _source_columns,
    _source_probability,
)
from sklearn.cross_decomposition import PLSRegression
from sklearn.impute import SimpleImputer
from strict_dev_surface import DEV_CUTOFF, development_surface

BASE_COLUMNS = ["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh"]
CLASS_WIDTH = 0.02
PLS_COMPONENTS = 12
PLS_FEATURES = 300
M195_ID = "M195_STRICT_ROW_PLS_G3_Q3"
M195_RECEIPT = OUTPUT / f"{M195_ID}-dev-2023-Q3.json"
M195_LATENT = OUTPUT / f"{M195_ID}-dev-2023-Q3-latent.npz"
M195_PARENT = OUTPUT / f"{M195_ID}-dev-2023-Q3.parquet"


def _common_columns(feature_columns: list[str]) -> list[str]:
    excluded_tokens = (
        "geom__",
        "spatial__",
        "group_",
        "capacity",
        "turbine",
        "rotor",
        "latitude_centroid",
        "longitude_centroid",
    )
    selected = [
        name
        for name in feature_columns
        if name != "group_id" and not any(token in name for token in excluded_tokens)
    ]
    if len(selected) < 700:
        raise RuntimeError(f"common hourly feature contract resolved {len(selected)}")
    return selected


def _screen_hourly(
    matrix: pd.DataFrame,
    target: np.ndarray,
    candidates: list[str],
    seed: int,
) -> list[str]:
    model = LGBMRegressor(
        objective="l1",
        n_estimators=180,
        learning_rate=0.035,
        num_leaves=31,
        min_child_samples=50,
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
    model.fit(matrix[candidates], target.mean(axis=1))
    gain = model.booster_.feature_importance(importance_type="gain")
    order = np.argsort(gain)[::-1][:PLS_FEATURES]
    selected = [candidates[position] for position in order]
    del model
    gc.collect()
    return selected


def _multioutput_latent(
    surface: pd.DataFrame,
    base_matrix: pd.DataFrame,
    history: np.ndarray,
    validation: np.ndarray,
    target: pd.Series,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    common = _common_columns(feature_columns)
    group1_application = (
        (history | validation) & surface["group_id"].eq(1).to_numpy()
    )
    hourly = base_matrix.loc[group1_application, common].copy().reset_index(drop=True)
    hourly_times = surface.loc[
        group1_application, "forecast_kst_dtm"
    ].reset_index(drop=True)
    hourly_fit = surface.loc[group1_application, "forecast_kst_dtm"].lt(
        surface.loc[validation, "forecast_kst_dtm"].min()
    ).to_numpy()
    donor = surface.loc[
        history & surface["group_id"].isin((1, 2)).to_numpy(),
        ["forecast_kst_dtm", "group_id"],
    ].copy()
    donor["target"] = target.loc[
        history & surface["group_id"].isin((1, 2)).to_numpy()
    ].to_numpy(dtype=float)
    donor_wide = donor.pivot(
        index="forecast_kst_dtm", columns="group_id", values="target"
    ).sort_index()
    fit_times = hourly_times.loc[hourly_fit]
    aligned_target = donor_wide.reindex(fit_times).to_numpy(dtype=float)
    if aligned_target.shape != (int(hourly_fit.sum()), 2):
        raise RuntimeError("simultaneous donor target alignment failed")
    complete_donor = np.isfinite(aligned_target).all(axis=1)
    pls_fit = np.zeros(len(hourly), dtype=bool)
    pls_fit[np.flatnonzero(hourly_fit)[complete_donor]] = True
    fit_target = aligned_target[complete_donor]
    if len(fit_target) < 10000 or not np.isfinite(fit_target).all():
        raise RuntimeError("simultaneous donor complete-pair history is too small")

    generated: dict[str, np.ndarray] = {}
    diagnostics: list[dict[str, object]] = []
    source_candidates = {
        "global": common,
        "gfs": _source_columns(common, "gfs"),
        "ldaps": _source_columns(common, "ldaps"),
    }
    for source_index, (source, candidates) in enumerate(source_candidates.items()):
        fit_frame = hourly.loc[pls_fit, candidates]
        selected = _screen_hourly(
            fit_frame,
            fit_target,
            candidates,
            20261500 + source_index,
        )
        imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        fit_matrix = imputer.fit_transform(hourly.loc[pls_fit, selected])
        apply_matrix = imputer.transform(hourly[selected])
        model = PLSRegression(
            n_components=PLS_COMPONENTS,
            scale=True,
            max_iter=500,
            tol=1e-6,
        )
        model.fit(fit_matrix, fit_target)
        score = np.asarray(model.transform(apply_matrix), dtype="float32")
        prediction = np.asarray(model.predict(apply_matrix), dtype="float32")
        hourly_generated: dict[str, np.ndarray] = {}
        for component in range(PLS_COMPONENTS):
            hourly_generated[f"mpls__{source}__score{component:02d}"] = score[
                :, component
            ]
        for donor_index, group_id in enumerate((1, 2)):
            for power in (1, 2, 3):
                hourly_generated[
                    f"mpls__{source}__g{group_id}_prediction_p{power}"
                ] = np.power(prediction[:, donor_index], power)
        hourly_generated[f"mpls__{source}__prediction_mean"] = prediction.mean(
            axis=1
        )
        hourly_generated[f"mpls__{source}__prediction_delta"] = (
            prediction[:, 0] - prediction[:, 1]
        )
        hourly_generated[f"mpls__{source}__prediction_product"] = (
            prediction[:, 0] * prediction[:, 1]
        )
        lookup = pd.DataFrame(hourly_generated, index=hourly_times)
        if not lookup.index.is_unique:
            raise RuntimeError("hourly donor latent lookup is not unique")
        mapped = lookup.reindex(surface["forecast_kst_dtm"])
        for name in mapped:
            generated[name] = mapped[name].to_numpy(dtype="float32")
        fit_prediction = prediction[pls_fit]
        diagnostic = {
            "source": source,
            "fit_hours": int(pls_fit.sum()),
            "excluded_incomplete_donor_hours": int(
                hourly_fit.sum() - pls_fit.sum()
            ),
            "apply_hours": len(hourly),
            "selected_feature_count": len(selected),
            "components": PLS_COMPONENTS,
            "component_iterations": [int(value) for value in model.n_iter_],
            "in_sample_mae_by_donor": [
                float(np.mean(np.abs(fit_prediction[:, index] - fit_target[:, index])))
                for index in range(2)
            ],
            "selected_features": selected,
        }
        diagnostics.append(diagnostic)
        print(
            json.dumps(
                {
                    "multioutput_pls": {
                        key: value
                        for key, value in diagnostic.items()
                        if key != "selected_features"
                    }
                }
            ),
            flush=True,
        )
        del model, imputer, fit_matrix, apply_matrix, score, prediction
        gc.collect()
    return pd.DataFrame(generated, index=surface.index), diagnostics


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
    parser.add_argument("--global-features", type=int, default=120)
    parser.add_argument("--source-features", type=int, default=180)
    args = parser.parse_args()
    if not M195_RECEIPT.exists() or not M195_LATENT.exists() or not M195_PARENT.exists():
        raise RuntimeError("verified M195 parent surface is missing")
    if _sha256(OPEN) != OPEN_SHA or _sha256(BASELINE) != BASELINE_SHA:
        raise RuntimeError("immutable competition input hash mismatch")

    started = time.perf_counter()
    surface, feature_columns = development_surface()
    if surface["forecast_kst_dtm"].ge(DEV_CUTOFF).any():
        raise RuntimeError("lockbox row reached multioutput donor runner")
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
        raise RuntimeError("M195 latent shape contract changed")
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
    multioutput_columns = list(multioutput)
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
        "global": (all_candidates, args.global_features),
        "gfs": (
            list(dict.fromkeys([*_source_columns(all_candidates, "gfs"), *global_latent])),
            args.source_features,
        ),
        "ldaps": (
            list(
                dict.fromkeys(
                    [*_source_columns(all_candidates, "ldaps"), *global_latent]
                )
            ),
            args.source_features,
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
            args.iterations,
            len(active_bins),
            20261600 + source_index * 10,
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
    multioutput_result = _frame(surface, validation, normalized)

    parent = pd.read_parquet(M195_PARENT)
    expected_keys = pd.MultiIndex.from_frame(
        surface.loc[validation, ["forecast_id", "group_id"]]
    )
    parent_keys = pd.MultiIndex.from_frame(parent[["forecast_id", "group_id"]])
    if len(parent) != int(validation.sum()) or not parent_keys.equals(expected_keys):
        raise RuntimeError("M195 parent key contract changed")
    parent_normalized = parent["prediction_kwh"].to_numpy(dtype=float) / parent[
        "group_id"
    ].map(CAPACITIES).to_numpy(dtype=float)
    blend_result = _frame(
        surface, validation, 0.5 * normalized + 0.5 * parent_normalized
    )

    output = multioutput_result.assign(fold_id=args.fold, model_id=args.candidate_id)
    output_path = OUTPUT / f"{args.candidate_id}-{args.fold}.parquet"
    output.to_parquet(output_path, index=False)
    receipt = {
        "schema_version": 1,
        "candidate_id": args.candidate_id,
        "fold_id": args.fold,
        "architecture": "strict_m195_plus_simultaneous_group12_multioutput_pls_rank",
        "scope": (
            "fixed official-data-only multi-response donor transfer screen; outer Q3 "
            "labels excluded from donor PLS, feature selection, classifier fit, and policy"
        ),
        "physical_development_cutoff": str(DEV_CUTOFF),
        "strict_training_rows": int(training.sum()),
        "class_width": CLASS_WIDTH,
        "class_count": len(active_bins),
        "multioutput_latent_feature_count": len(multioutput_columns),
        "multioutput_diagnostics": multioutput_diagnostics,
        "selected_features": selected_features,
        "selected_multioutput_feature_counts": {
            source: sum(name.startswith("mpls__") for name in names)
            for source, names in selected_features.items()
        },
        "frozen_mixture": {
            "global_weight": FROZEN_MIXTURE[0],
            "gfs_share": FROZEN_MIXTURE[1],
        },
        "frozen_policy": FROZEN_POLICY,
        "fold_score": _score(multioutput_result),
        "group_scores": _group_scores(multioutput_result),
        "fixed_half_m195_blend_score": _score(blend_result),
        "fixed_half_m195_group_scores": _group_scores(blend_result),
        "runtime_seconds": round(time.perf_counter() - started, 2),
        "prediction_path": str(output_path.relative_to(Path.cwd())),
        "prediction_sha256": _sha256(output_path),
        "m195_parent_sha256": _sha256(M195_PARENT),
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
                "fold_score": receipt["fold_score"],
                "group_scores": receipt["group_scores"],
                "fixed_half_m195_blend_score": receipt[
                    "fixed_half_m195_blend_score"
                ],
                "selected_multioutput_feature_counts": receipt[
                    "selected_multioutput_feature_counts"
                ],
                "runtime_seconds": receipt["runtime_seconds"],
                "prediction_sha256": receipt["prediction_sha256"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
