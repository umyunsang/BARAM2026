"""Fit fixed group-specific RBF-SVR models on compact strict PLS coordinates."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from run_group_balanced_pls_rank import _frame, _group_scores
from run_multioutput_donor_pls_rank import M195_LATENT, M195_RECEIPT
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
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from strict_dev_surface import DEV_CUTOFF, development_surface

C_VALUE = 10.0
EPSILON = 0.03
PARENT_PATH = OUTPUT / "M197_STRICT_FIXED_HALF_PLS-dev-2023-Q3.parquet"


def _latent_columns(columns: list[str]) -> list[str]:
    selected = [
        name
        for name in columns
        if "prediction_p1" in name
        or any(f"score{index:02d}" in name for index in range(6))
    ]
    if len(selected) != 63:
        raise RuntimeError(f"compact PLS contract resolved {len(selected)} columns")
    return selected


def _control_columns(feature_columns: list[str]) -> list[str]:
    exact = {
        "lead_hour",
        "cal__hour_sin",
        "cal__hour_cos",
        "cal__doy_sin",
        "cal__doy_cos",
        "phys__hub117_speed",
        "phys__speed_shear_100_80",
        "phys__air_density",
        "phys__rho_v3",
        "phys_v2__shear_alpha_100_80",
        "phys_v2__hub117_speed",
        "phys_v2__air_density",
        "phys_v2__rho_v3",
    }
    spatial = {
        "gfs_spatial__idw__wind10_speed",
        "gfs_spatial__idw__wind80_speed",
        "gfs_spatial__idw__wind100_speed",
        "ldaps_spatial__idw__wind10_speed",
        "ldaps_spatial__idw__wind5_speed",
        "ldaps_spatial__idw__wind50max_speed",
        "ldaps_spatial__idw__wind50min_speed",
    }
    selected = [name for name in feature_columns if name in exact | spatial]
    if len(selected) < 18:
        raise RuntimeError(f"SVR control contract resolved {len(selected)} columns")
    return selected


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
        raise RuntimeError("lockbox row reached RBF-SVR runner")
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
    all_latent_columns = [str(value) for value in cached["columns"].tolist()]
    latent_values = np.asarray(cached["values"], dtype="float32")
    if latent_values.shape != (len(surface), len(all_latent_columns)):
        raise RuntimeError("M195 latent checkpoint shape contract changed")
    latent_columns = _latent_columns(all_latent_columns)
    latent_positions = [all_latent_columns.index(name) for name in latent_columns]
    controls = _control_columns(feature_columns)
    matrix = np.column_stack(
        (
            latent_values[:, latent_positions],
            surface[controls].to_numpy(dtype="float32"),
        )
    )
    del latent_values
    gc.collect()
    normalized = np.full(int(validation.sum()), np.nan, dtype=float)
    validation_groups = surface.loc[validation, "group_id"].to_numpy(dtype=int)
    diagnostics: dict[str, dict[str, object]] = {}
    for group_id in CAPACITIES:
        fit = training & surface["group_id"].eq(group_id).to_numpy()
        apply = validation & surface["group_id"].eq(group_id).to_numpy()
        model = make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            SVR(
                kernel="rbf",
                C=C_VALUE,
                epsilon=EPSILON,
                gamma="scale",
                cache_size=4096,
                shrinking=True,
            ),
        )
        model.fit(matrix[fit], target.loc[fit].to_numpy(dtype=float))
        prediction = np.clip(model.predict(matrix[apply]), 0.075, 1.075)
        normalized[validation_groups == group_id] = prediction
        svr = model.named_steps["svr"]
        diagnostics[str(group_id)] = {
            "fit_rows": int(fit.sum()),
            "apply_rows": int(apply.sum()),
                "support_vectors": len(svr.support_),
            "support_fraction": float(len(svr.support_) / fit.sum()),
        }
        print(json.dumps({"svr": {"group_id": group_id, **diagnostics[str(group_id)]}}), flush=True)
        del model, svr
        gc.collect()
    if not np.isfinite(normalized).all():
        raise RuntimeError("RBF-SVR produced incomplete validation predictions")
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

    receipt = {
        "schema_version": 1,
        "candidate_id": args.candidate_id,
        "fold_id": args.fold,
        "architecture": "strict_group_rbf_svr_compact_m195_half_m197",
        "scope": (
            "fixed official-data-only compact supervised kernel screen; outer Q3 "
            "labels excluded from imputation, scaling, SVR fit, and blend"
        ),
        "physical_development_cutoff": str(DEV_CUTOFF),
        "strict_training_rows": int(training.sum()),
        "c": C_VALUE,
        "epsilon": EPSILON,
        "gamma": "scale",
        "latent_feature_count": len(latent_columns),
        "latent_features": latent_columns,
        "control_feature_count": len(controls),
        "control_features": controls,
        "diagnostics": diagnostics,
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
                "diagnostics": diagnostics,
                "runtime_seconds": receipt["runtime_seconds"],
                "prediction_sha256": receipt["prediction_sha256"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
