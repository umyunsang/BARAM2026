"""Add multi-basis PLS-DA axes to the strict PLS source-rank family."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from run_inner_policy_classifier import _group_total, _policy_values
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
    _source_probability,
)
from sklearn.cross_decomposition import PLSRegression
from sklearn.impute import SimpleImputer
from strict_dev_surface import DEV_CUTOFF, development_surface

BASE_COLUMNS = ["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh"]
CLASS_WIDTH = 0.02
PLS_COMPONENTS = 12
PLS_FEATURES = 300
ITERATIONS = 150
GLOBAL_FEATURES = 140
SOURCE_FEATURES = 200
SETTLEMENT_THRESHOLDS = np.linspace(0.15, 1.00, 12)
RBF_CENTERS = np.linspace(0.14, 1.04, 12)
RBF_WIDTH = 0.09
PARENT_PATH = OUTPUT / "M197_STRICT_FIXED_HALF_PLS-dev-2023-Q3.parquet"


def _target_basis(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=float), 0.10, 1.075)
    continuous = np.column_stack(
        (
            clipped,
            np.sqrt(clipped),
            clipped * clipped,
            np.log(np.clip(clipped, 0.02, 0.98) / np.clip(1.0 - clipped, 0.02, None)),
        )
    )
    survival = clipped[:, None] > SETTLEMENT_THRESHOLDS[None, :]
    rbf = np.exp(
        -0.5
        * ((clipped[:, None] - RBF_CENTERS[None, :]) / RBF_WIDTH) ** 2
    )
    return np.column_stack((continuous, survival.astype(float), rbf))


def _basis_lane(
    matrix: pd.DataFrame,
    target: pd.Series,
    fit_mask: np.ndarray,
    application_mask: np.ndarray,
    candidates: list[str],
    source: str,
    lane: str,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    selected = _screen(
        matrix,
        target,
        fit_mask,
        candidates,
        PLS_FEATURES,
        seed,
    )
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    fit_matrix = imputer.fit_transform(matrix.loc[fit_mask, selected])
    application_matrix = imputer.transform(matrix.loc[application_mask, selected])
    fit_target = target.loc[fit_mask].to_numpy(dtype=float)
    basis = _target_basis(fit_target)
    model = PLSRegression(
        n_components=PLS_COMPONENTS,
        scale=True,
        max_iter=500,
        tol=1e-6,
        copy=True,
    )
    model.fit(fit_matrix, basis)
    fit_scores = np.asarray(model.transform(fit_matrix), dtype="float32")
    application_scores = np.asarray(
        model.transform(application_matrix), dtype="float32"
    )
    fit_prediction = np.asarray(model.predict(fit_matrix), dtype="float32")
    application_prediction = np.asarray(
        model.predict(application_matrix), dtype="float32"
    )

    generated: dict[str, np.ndarray] = {}
    for component in range(PLS_COMPONENTS):
        values = np.full(len(matrix), np.nan, dtype="float32")
        values[fit_mask] = fit_scores[:, component]
        values[application_mask] = application_scores[:, component]
        generated[f"basis_pls__{source}__{lane}__score{component:02d}"] = values
    for component in range(basis.shape[1]):
        values = np.full(len(matrix), np.nan, dtype="float32")
        values[fit_mask] = fit_prediction[:, component]
        values[application_mask] = application_prediction[:, component]
        generated[f"basis_pls__{source}__{lane}__pred{component:02d}"] = values
    diagnostics: dict[str, object] = {
        "source": source,
        "lane": lane,
        "fit_rows": int(fit_mask.sum()),
        "application_rows": int(application_mask.sum()),
        "selected_feature_count": len(selected),
        "basis_dimension": basis.shape[1],
        "components": PLS_COMPONENTS,
        "component_iterations": [int(value) for value in model.n_iter_],
        "in_sample_basis_mae": float(np.mean(np.abs(fit_prediction - basis))),
    }
    del (
        model,
        imputer,
        fit_matrix,
        application_matrix,
        fit_scores,
        application_scores,
        fit_prediction,
        application_prediction,
        basis,
    )
    gc.collect()
    return generated, diagnostics


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
    args = parser.parse_args()
    if _sha256(OPEN) != OPEN_SHA or _sha256(BASELINE) != BASELINE_SHA:
        raise RuntimeError("immutable competition input hash mismatch")
    if not PARENT_PATH.exists() or not M195_RECEIPT.exists() or not M195_LATENT.exists():
        raise RuntimeError("verified PLS parent surface is missing")

    started = time.perf_counter()
    surface, feature_columns = development_surface()
    if surface["forecast_kst_dtm"].ge(DEV_CUTOFF).any():
        raise RuntimeError("lockbox row reached target-basis PLS runner")
    validation = _validation_mask(surface, args.fold)
    history = _strict_preceding_mask(surface, validation)
    target = surface["actual_kwh"] / surface["group_id"].map(CAPACITIES)
    training = (
        history
        & surface["actual_kwh"].notna().to_numpy()
        & target.ge(0.10).to_numpy()
    )
    donor_training = training & surface["group_id"].isin((1, 2)).to_numpy()
    application = training | validation

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
    basis_candidates = [
        name
        for name in feature_columns
        if name not in {"group_id", "group_1", "group_2", "group_3"}
    ]
    source_candidates = {
        "global": basis_candidates,
        "gfs": _source_columns(basis_candidates, "gfs"),
        "ldaps": _source_columns(basis_candidates, "ldaps"),
    }
    basis_features: dict[str, np.ndarray] = {}
    basis_diagnostics: list[dict[str, object]] = []
    for source_index, (source, candidates) in enumerate(source_candidates.items()):
        for lane_index, (lane, fit_mask) in enumerate(
            (("shared", training), ("donor12", donor_training))
        ):
            generated, diagnostics = _basis_lane(
                base_matrix,
                target,
                fit_mask,
                application,
                candidates,
                source,
                lane,
                20262000 + source_index * 20 + lane_index,
            )
            basis_features.update(generated)
            basis_diagnostics.append(diagnostics)
            print(json.dumps({"basis_pls": diagnostics}), flush=True)

    basis_columns = list(basis_features)
    matrix = pd.concat(
        [
            base_matrix,
            pd.DataFrame(parent_values, columns=parent_columns, index=surface.index),
            multioutput,
            pd.DataFrame(basis_features, index=surface.index),
        ],
        axis=1,
    )
    del base_matrix, parent_values, multioutput, basis_features
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
        if "__global__" in name and name.startswith(("pls__", "basis_pls__"))
    ]
    rank_specs = {
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
    for source_index, (source, (candidates, count)) in enumerate(rank_specs.items()):
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
            20262100 + source_index * 10,
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
    parent_normalized = parent["prediction_kwh"].to_numpy(dtype=float) / parent[
        "group_id"
    ].map(CAPACITIES).to_numpy(dtype=float)
    output = _frame(surface, validation, 0.5 * parent_normalized + 0.5 * normalized)
    output["fold_id"] = args.fold
    output["model_id"] = args.candidate_id
    output_path = OUTPUT / f"{args.candidate_id}-{args.fold}.parquet"
    output.to_parquet(output_path, index=False)

    receipt = {
        "schema_version": 1,
        "candidate_id": args.candidate_id,
        "fold_id": args.fold,
        "architecture": "strict_target_basis_pls_da_source_rank_half_m197",
        "scope": (
            "fixed official-data-only target-basis supervised latent screen; outer "
            "Q3 labels excluded from every feature, PLS, classifier, policy, and blend fit"
        ),
        "physical_development_cutoff": str(DEV_CUTOFF),
        "strict_training_rows": int(training.sum()),
        "donor_training_rows": int(donor_training.sum()),
        "class_width": CLASS_WIDTH,
        "class_count": len(active_bins),
        "pls_components": PLS_COMPONENTS,
        "pls_screen_features": PLS_FEATURES,
        "basis_dimension": 4 + len(SETTLEMENT_THRESHOLDS) + len(RBF_CENTERS),
        "basis_feature_count": len(basis_columns),
        "basis_diagnostics": basis_diagnostics,
        "multioutput_diagnostics": multioutput_diagnostics,
        "selected_features": selected_features,
        "selected_basis_feature_counts": {
            source: sum(name.startswith("basis_pls__") for name in names)
            for source, names in selected_features.items()
        },
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
                "selected_basis_feature_counts": receipt[
                    "selected_basis_feature_counts"
                ],
                "runtime_seconds": receipt["runtime_seconds"],
                "prediction_sha256": receipt["prediction_sha256"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
