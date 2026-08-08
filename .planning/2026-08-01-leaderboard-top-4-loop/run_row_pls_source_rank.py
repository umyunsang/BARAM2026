"""Inject row-level supervised latent transfer axes into strict source rank."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
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
PARENT_PATH = OUTPUT / "M189_STRICT_PREQUENTIAL_SOURCE_RANK_Q3-dev-2023-Q3.parquet"


def _latent_lane(
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
    model = PLSRegression(
        n_components=PLS_COMPONENTS,
        scale=True,
        max_iter=500,
        tol=1e-6,
        copy=True,
    )
    fit_target = target.loc[fit_mask].to_numpy(dtype=float)
    model.fit(fit_matrix, fit_target)
    fit_scores = np.asarray(model.transform(fit_matrix), dtype="float32")
    application_scores = np.asarray(
        model.transform(application_matrix), dtype="float32"
    )
    fit_prediction = np.asarray(model.predict(fit_matrix), dtype=float).reshape(-1)
    application_prediction = np.asarray(
        model.predict(application_matrix), dtype="float32"
    ).reshape(-1)

    generated: dict[str, np.ndarray] = {}
    for component in range(PLS_COMPONENTS):
        values = np.full(len(matrix), np.nan, dtype="float32")
        values[fit_mask] = fit_scores[:, component]
        values[application_mask] = application_scores[:, component]
        generated[f"pls__{source}__{lane}__score{component:02d}"] = values
    for power in (1, 2, 3):
        values = np.full(len(matrix), np.nan, dtype="float32")
        values[fit_mask] = np.power(fit_prediction, power).astype("float32")
        values[application_mask] = np.power(application_prediction, power).astype(
            "float32"
        )
        generated[f"pls__{source}__{lane}__prediction_p{power}"] = values
    diagnostics: dict[str, object] = {
        "source": source,
        "lane": lane,
        "fit_rows": int(fit_mask.sum()),
        "application_rows": int(application_mask.sum()),
        "selected_feature_count": len(selected),
        "selected_features": selected,
        "components": PLS_COMPONENTS,
        "component_iterations": [int(value) for value in model.n_iter_],
        "in_sample_mae": float(np.mean(np.abs(fit_prediction - fit_target))),
    }
    del model, imputer, fit_matrix, application_matrix, fit_scores, application_scores
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
    parser.add_argument("--iterations", type=int, default=150)
    parser.add_argument("--global-features", type=int, default=120)
    parser.add_argument("--source-features", type=int, default=180)
    parser.add_argument("--include-group3-lane", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 100 <= args.iterations <= 240:
        raise ValueError("iterations must be between 100 and 240")
    if not 80 <= args.global_features <= 180:
        raise ValueError("global feature count must be between 80 and 180")
    if not 120 <= args.source_features <= 240:
        raise ValueError("source feature count must be between 120 and 240")
    if _sha256(OPEN) != OPEN_SHA or _sha256(BASELINE) != BASELINE_SHA:
        raise RuntimeError("immutable competition input hash mismatch")

    started = time.perf_counter()
    surface, feature_columns = development_surface()
    if surface["forecast_kst_dtm"].ge(DEV_CUTOFF).any():
        raise RuntimeError("lockbox row reached row-PLS source-rank runner")
    validation = _validation_mask(surface, args.fold)
    history = _strict_preceding_mask(surface, validation)
    target = surface["actual_kwh"] / surface["group_id"].map(CAPACITIES)
    training = (
        history
        & surface["actual_kwh"].notna().to_numpy()
        & target.ge(0.10).to_numpy()
    )
    donor_training = training & surface["group_id"].isin((1, 2)).to_numpy()
    group3_training = training & surface["group_id"].eq(3).to_numpy()
    if args.include_group3_lane and int(group3_training.sum()) < 1500:
        raise RuntimeError("group-3 PLS history is unexpectedly small")
    application = training | validation
    matrix = surface[feature_columns].astype("float32")
    pls_candidates = [
        name
        for name in feature_columns
        if name not in {"group_id", "group_1", "group_2", "group_3"}
    ]
    source_candidates = {
        "global": pls_candidates,
        "gfs": _source_columns(pls_candidates, "gfs"),
        "ldaps": _source_columns(pls_candidates, "ldaps"),
    }
    latent_cache_path = OUTPUT / f"{args.candidate_id}-{args.fold}-latent.npz"
    latent_meta_path = OUTPUT / f"{args.candidate_id}-{args.fold}-latent.json"
    if args.resume and latent_cache_path.exists() and latent_meta_path.exists():
        cached = np.load(latent_cache_path, allow_pickle=False)
        latent_columns = [str(value) for value in cached["columns"].tolist()]
        latent_values = np.asarray(cached["values"], dtype="float32")
        if latent_values.shape != (len(surface), len(latent_columns)):
            raise RuntimeError("PLS latent checkpoint shape contract changed")
        latent_diagnostics = json.loads(latent_meta_path.read_text())
        matrix = pd.concat(
            [
                matrix,
                pd.DataFrame(latent_values, columns=latent_columns, index=matrix.index),
            ],
            axis=1,
        )
        print(
            json.dumps(
                {
                    "checkpoint": "latent",
                    "status": "reused",
                    "feature_count": len(latent_columns),
                }
            ),
            flush=True,
        )
    else:
        latent: dict[str, np.ndarray] = {}
        latent_diagnostics: list[dict[str, object]] = []
        lane_specs: list[tuple[str, np.ndarray]] = [
            ("shared", training),
            ("donor12", donor_training),
        ]
        if args.include_group3_lane:
            lane_specs.append(("group3", group3_training))
        for source_index, (source, candidates) in enumerate(source_candidates.items()):
            for lane_index, (lane, fit_mask) in enumerate(lane_specs):
                generated, diagnostics = _latent_lane(
                    matrix,
                    target,
                    fit_mask,
                    application,
                    candidates,
                    source,
                    lane,
                    20261300 + source_index * 20 + lane_index,
                )
                latent.update(generated)
                latent_diagnostics.append(diagnostics)
                print(
                    json.dumps(
                        {
                            "pls": {
                                key: value
                                for key, value in diagnostics.items()
                                if key != "selected_features"
                            }
                        }
                    ),
                    flush=True,
                )
        latent_columns = list(latent)
        latent_values = np.column_stack([latent[name] for name in latent_columns])
        np.savez(
            latent_cache_path,
            columns=np.asarray(latent_columns),
            values=latent_values.astype("float32"),
        )
        latent_meta_path.write_text(
            json.dumps(latent_diagnostics, sort_keys=True), encoding="utf-8"
        )
        matrix = pd.concat(
            [
                matrix,
                pd.DataFrame(latent_values, columns=latent_columns, index=matrix.index),
            ],
            axis=1,
        )
        del latent, latent_values
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
    global_latent = [name for name in latent_columns if "__global__" in name]
    rank_specs = {
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
    for source_index, (source, (candidates, count)) in enumerate(rank_specs.items()):
        checkpoint_path = OUTPUT / (
            f"{args.candidate_id}-{args.fold}-{source}-probability.npz"
        )
        checkpoint_meta_path = OUTPUT / (
            f"{args.candidate_id}-{args.fold}-{source}-probability.json"
        )
        if args.resume and checkpoint_path.exists() and checkpoint_meta_path.exists():
            cached = np.load(checkpoint_path, allow_pickle=False)
            probability = np.asarray(cached["probability"], dtype=float)
            selected = json.loads(checkpoint_meta_path.read_text())
            if probability.shape != (int(validation.sum()), len(active_bins)):
                raise RuntimeError(f"{source} probability checkpoint shape changed")
            print(
                json.dumps({"checkpoint": source, "status": "reused"}), flush=True
            )
        else:
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
                20261400 + source_index * 10,
                source,
            )
            np.savez(checkpoint_path, probability=probability.astype("float32"))
            checkpoint_meta_path.write_text(
                json.dumps(selected, sort_keys=True), encoding="utf-8"
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
    pls_output = _frame(surface, validation, normalized)

    parent = pd.read_parquet(PARENT_PATH)
    expected_keys = pd.MultiIndex.from_frame(
        surface.loc[validation, ["forecast_id", "group_id"]]
    )
    parent_keys = pd.MultiIndex.from_frame(parent[["forecast_id", "group_id"]])
    if len(parent) != int(validation.sum()) or not parent_keys.equals(expected_keys):
        raise RuntimeError("M189 parent key contract changed")
    parent_normalized = parent["prediction_kwh"].to_numpy(dtype=float) / parent[
        "group_id"
    ].map(CAPACITIES).to_numpy(dtype=float)
    blend_output = _frame(surface, validation, 0.5 * normalized + 0.5 * parent_normalized)

    output = pls_output.assign(fold_id=args.fold, model_id=args.candidate_id)
    output_path = OUTPUT / f"{args.candidate_id}-{args.fold}.parquet"
    output.to_parquet(output_path, index=False)
    receipt = {
        "schema_version": 1,
        "candidate_id": args.candidate_id,
        "fold_id": args.fold,
        "architecture": (
            "strict_shared_group12_donor_group3_row_pls_source_rank"
            if args.include_group3_lane
            else "strict_shared_and_group12_donor_row_pls_source_rank"
        ),
        "scope": (
            "fixed official-data-only supervised-latent transfer screen; outer Q3 "
            "labels excluded from PLS screening, PLS fit, classifier fit, and policy"
        ),
        "physical_development_cutoff": str(DEV_CUTOFF),
        "strict_training_rows": int(training.sum()),
        "donor_training_rows": int(donor_training.sum()),
        "group3_training_rows": int(group3_training.sum()),
        "include_group3_lane": args.include_group3_lane,
        "class_width": CLASS_WIDTH,
        "class_count": len(active_bins),
        "pls_components": PLS_COMPONENTS,
        "pls_screen_features": PLS_FEATURES,
        "latent_feature_count": len(latent_columns),
        "latent_diagnostics": latent_diagnostics,
        "latent_checkpoint_path": str(latent_cache_path.relative_to(Path.cwd())),
        "latent_checkpoint_sha256": _sha256(latent_cache_path),
        "selected_features": selected_features,
        "selected_latent_feature_counts": {
            source: sum(name.startswith("pls__") for name in names)
            for source, names in selected_features.items()
        },
        "frozen_mixture": {
            "global_weight": FROZEN_MIXTURE[0],
            "gfs_share": FROZEN_MIXTURE[1],
        },
        "frozen_policy": FROZEN_POLICY,
        "fold_score": _score(pls_output),
        "group_scores": _group_scores(pls_output),
        "fixed_half_m189_blend_score": _score(blend_output),
        "fixed_half_m189_group_scores": _group_scores(blend_output),
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
                "fold_score": receipt["fold_score"],
                "group_scores": receipt["group_scores"],
                "fixed_half_m189_blend_score": receipt[
                    "fixed_half_m189_blend_score"
                ],
                "selected_latent_feature_counts": receipt[
                    "selected_latent_feature_counts"
                ],
                "runtime_seconds": receipt["runtime_seconds"],
                "prediction_sha256": receipt["prediction_sha256"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
