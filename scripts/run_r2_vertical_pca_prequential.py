"""One-shot R2 vertical-PCA materialization and strict prequential evaluation."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from baram.constants import CAPACITIES_KWH
from baram.evaluation.official import evaluate_official
from baram.evaluation.prequential import run_prequential_protocol
from baram.loop.events import EventStore

KEYS = ("fold_id", "group_id", "forecast_kst_dtm")
FAMILY = ("CHAMPION", "R2_CONTROL_ZERO4", "R2_VERTICAL_PCA2X2")
ARMS = FAMILY[1:]
FIT_MAX = {
    "dev-2023-Q2": pd.Timestamp("2023-04-01 00:00:00"),
    "dev-2023-Q3": pd.Timestamp("2023-07-01 00:00:00"),
    "dev-2023-Q4": pd.Timestamp("2023-10-01 00:00:00"),
}
SELECTION_MAX = {
    "dev-2023-Q3": pd.Timestamp("2023-07-01 00:00:00"),
    "dev-2023-Q4": pd.Timestamp("2023-07-01 00:00:00"),
}
PCA_INPUTS = {
    "gfs": (
        "gfs__heightAboveGround_10_10u__mean",
        "gfs__heightAboveGround_10_10v__mean",
        "gfs__heightAboveGround_80_u__mean",
        "gfs__heightAboveGround_80_v__mean",
        "gfs__heightAboveGround_100_100u__mean",
        "gfs__heightAboveGround_100_100v__mean",
    ),
    "ldaps": (
        "ldaps__heightAboveGround_5_XBLWS__mean",
        "ldaps__heightAboveGround_5_YBLWS__mean",
        "ldaps__heightAboveGround_10_10u__mean",
        "ldaps__heightAboveGround_10_10v__mean",
    ),
}
PCA_COLUMNS = ("vpca_gfs_1", "vpca_gfs_2", "vpca_ldaps_1", "vpca_ldaps_2")
MODEL_PARAMS = {
    "objective": "l2",
    "n_estimators": 900,
    "learning_rate": 0.035,
    "num_leaves": 63,
    "min_child_samples": 40,
    "subsample": 0.85,
    "subsample_freq": 1,
    "colsample_bytree": 0.4,
    "reg_lambda": 3.0,
    "random_state": 20260801,
    "n_jobs": 6,
    "verbose": -1,
    "deterministic": True,
    "force_col_wise": True,
}
ACTIONS_CF = np.arange(0.05, 1.0801, 0.0025)
SOFT_CAPS = {1: 0.985, 2: 0.989, 3: 1.005}
QUANTILE_LEVELS = np.linspace(0.01, 0.99, 81)
CALIBRATION_DAYS = 90
TEMPERATURE = 0.5
DENSITY_BINS = 15
GAMMA = 1.5


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _vector_hash(frame: pd.DataFrame, column: str, fold: str) -> str:
    part = frame.loc[frame["fold_id"].eq(fold)].sort_values(list(KEYS), kind="stable")
    key_bytes = part[list(KEYS)].astype(str).agg("|".join, axis=1).str.cat(sep="\n").encode()
    values = np.ascontiguousarray(part[column].to_numpy(dtype="<f8")).tobytes()
    return hashlib.sha256(key_bytes + b"\n" + values).hexdigest()


def _verify_inputs(repo: Path, predeclaration: Path) -> dict[str, Any]:
    frozen = json.loads(predeclaration.read_text())
    observed = {
        relative: _sha256(repo / relative)
        for relative in frozen["input_bundle"]["files"]
    }
    if observed != frozen["input_bundle"]["files"]:
        raise RuntimeError("N13 input hash mismatch")
    if _canonical_hash(observed) != frozen["input_bundle"]["sha256"]:
        raise RuntimeError("N13 input bundle mismatch")
    if frozen["comparison_index"] != 2:
        raise RuntimeError("N13 comparison index is not frozen at 2")
    return frozen


def _load_development_frame(
    repo: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    cache = (
        repo
        / "artifacts/cache/920be0c458d820e855bf79dd25723146f52ce1736138aedba5e6bc853f1f720b"
    )
    prepare = json.loads((repo / "artifacts/manifests/prepare.json").read_text())
    requested = ["forecast_kst_dtm", "data_available_kst_dtm", *prepare["feature_names"]]
    features = pd.read_parquet(
        cache / "train_features.parquet",
        columns=requested,
        filters=[("forecast_kst_dtm", "<=", pd.Timestamp("2024-01-01 00:00:00"))],
    )
    labels = pd.read_parquet(
        cache / "labels_long.parquet",
        columns=[
            "forecast_kst_dtm",
            "operating_day",
            "operating_year",
            "actual_kwh",
            "group_id",
        ],
        filters=[("operating_year", "<=", 2023)],
    )
    if len(features) != 730 * 24 * 3 or len(labels) != len(features):
        raise RuntimeError("2022-2023 filtered development row count mismatch")
    for column in ("forecast_kst_dtm", "data_available_kst_dtm"):
        features[column] = pd.to_datetime(features[column])
    for column in ("forecast_kst_dtm", "operating_day"):
        labels[column] = pd.to_datetime(labels[column])
    if not np.array_equal(
        features["forecast_kst_dtm"].to_numpy(),
        labels["forecast_kst_dtm"].to_numpy(),
    ):
        raise RuntimeError("feature/label timestamp row alignment mismatch")
    if labels["operating_day"].max() > pd.Timestamp("2023-12-31"):
        raise RuntimeError("2024 operating-day label materialized")
    frame = features[["forecast_kst_dtm", "data_available_kst_dtm"]].copy()
    frame["operating_day"] = labels["operating_day"].to_numpy()
    frame["group_id"] = labels["group_id"].to_numpy(dtype=np.int8)
    frame["actual_kwh"] = labels["actual_kwh"].to_numpy(dtype=float)
    capacities = frame["group_id"].map(CAPACITIES_KWH).to_numpy(dtype=float)
    frame["cf"] = frame["actual_kwh"].to_numpy(dtype=float) / capacities
    numeric_columns = [
        column
        for column in prepare["feature_names"]
        if pd.api.types.is_numeric_dtype(features[column])
    ]
    if not all(column in numeric_columns for columns in PCA_INPUTS.values() for column in columns):
        raise RuntimeError("frozen PCA inputs missing from numeric feature surface")
    base = features[numeric_columns].astype(np.float32)
    base = base.replace([np.inf, -np.inf], np.nan)
    return frame, base, numeric_columns


def _assessment_indices(frame: pd.DataFrame, assessment: pd.DataFrame) -> np.ndarray:
    development_keys = pd.MultiIndex.from_frame(frame[["group_id", "forecast_kst_dtm"]])
    if development_keys.has_duplicates:
        raise RuntimeError("development group/timestamp keys duplicate")
    assessment_keys = pd.MultiIndex.from_frame(assessment[["group_id", "forecast_kst_dtm"]])
    positions = development_keys.get_indexer(assessment_keys)
    if (positions < 0).any():
        raise RuntimeError("assessment key absent from development features")
    observed = frame.iloc[positions]["actual_kwh"].to_numpy(dtype=float)
    expected = assessment["actual_kwh"].to_numpy(dtype=float)
    if float(np.max(np.abs(observed - expected))) > 1e-9:
        raise RuntimeError("assessment actual alignment mismatch")
    return positions


def _fit_pca_additions(
    base: pd.DataFrame,
    frame: pd.DataFrame,
    fit_max: pd.Timestamp,
    candidate: bool,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    if not candidate:
        zeros = np.zeros((len(base), len(PCA_COLUMNS)), dtype=np.float32)
        return pd.DataFrame(zeros, columns=PCA_COLUMNS, index=base.index), []
    representative = frame["group_id"].eq(1) & frame["forecast_kst_dtm"].le(fit_max)
    additions: dict[str, np.ndarray] = {}
    records: list[dict[str, Any]] = []
    for source, columns in PCA_INPUTS.items():
        fit_raw = base.loc[representative, list(columns)].to_numpy(dtype=float)
        full_raw = base[list(columns)].to_numpy(dtype=float)
        medians = np.nanmedian(fit_raw, axis=0)
        fit_filled = np.where(np.isfinite(fit_raw), fit_raw, medians[None, :])
        means = fit_filled.mean(axis=0)
        scales = fit_filled.std(axis=0, ddof=0)
        scales = np.where(scales > 1e-12, scales, 1.0)
        standardized = (fit_filled - means[None, :]) / scales[None, :]
        _, singular_values, components = np.linalg.svd(standardized, full_matrices=False)
        components = components[:2].copy()
        for index in range(len(components)):
            anchor = int(np.argmax(np.abs(components[index])))
            if components[index, anchor] < 0:
                components[index] *= -1.0
        full_filled = np.where(np.isfinite(full_raw), full_raw, medians[None, :])
        scores = ((full_filled - means[None, :]) / scales[None, :]) @ components.T
        for component in range(2):
            additions[f"vpca_{source}_{component + 1}"] = scores[:, component].astype(
                np.float32
            )
        variance = singular_values**2
        explained = variance[:2] / variance.sum()
        records.append(
            {
                "source": source,
                "fit_max_time": fit_max.isoformat(),
                "fit_unique_timestamps": int(representative.sum()),
                "input_columns": list(columns),
                "medians": medians.tolist(),
                "means": means.tolist(),
                "scales": scales.tolist(),
                "components": components.tolist(),
                "explained_variance_ratio": explained.tolist(),
                "sign_rule": "largest_absolute_loading_positive",
            }
        )
    ordered = pd.DataFrame(additions, index=base.index)[list(PCA_COLUMNS)]
    return ordered, records


def _model_matrix(base: pd.DataFrame, additions: pd.DataFrame) -> pd.DataFrame:
    if len(base) != len(additions) or not base.index.equals(additions.index):
        raise RuntimeError("base/PCA matrix alignment mismatch")
    return pd.concat([base, additions], axis=1, copy=False)


def _fit_model(
    matrix: pd.DataFrame,
    target: np.ndarray,
    mask: np.ndarray,
    weights: np.ndarray,
) -> lgb.LGBMRegressor:
    if int(mask.sum()) < 1000:
        raise RuntimeError("insufficient past-only model rows")
    model = lgb.LGBMRegressor(**MODEL_PARAMS)
    model.fit(matrix.loc[mask], target[mask], sample_weight=weights[mask])
    return model


def _sharpen_weights(samples: np.ndarray) -> np.ndarray:
    rows, sample_count = samples.shape
    low = samples.min(axis=1, keepdims=True)
    high = samples.max(axis=1, keepdims=True)
    span = np.where(high > low, high - low, 1.0)
    bins = np.clip(
        ((samples - low) / span * DENSITY_BINS).astype(int),
        0,
        DENSITY_BINS - 1,
    )
    counts = np.zeros((rows, DENSITY_BINS), dtype=float)
    np.add.at(
        counts,
        (np.repeat(np.arange(rows), sample_count), bins.ravel()),
        1.0,
    )
    density = np.take_along_axis(counts, bins, axis=1)
    density = density ** (1.0 / TEMPERATURE)
    return density / density.sum(axis=1, keepdims=True)


def _settlement_action(
    point: np.ndarray,
    residual_quantiles: np.ndarray,
    mean_generation: float,
    cap_high: float,
) -> np.ndarray:
    samples = np.clip(
        point[:, None] + residual_quantiles[None, :],
        0.0,
        cap_high,
    )
    weights = _sharpen_weights(samples)
    best_value = np.full(len(point), -np.inf, dtype=float)
    best_action = np.full(len(point), ACTIONS_CF[0], dtype=float)
    for start in range(0, len(ACTIONS_CF), 64):
        actions = ACTIONS_CF[start : start + 64]
        error = np.abs(actions[None, :, None] - samples[:, None, :])
        nmae_utility = -(error * weights[:, None, :]).sum(axis=2)
        units = np.where(error <= 0.06, 4.0, np.where(error <= 0.08, 3.0, 0.0))
        ficr_utility = (
            (samples[:, None, :] * units * weights[:, None, :]).sum(axis=2)
            / (4.0 * mean_generation)
        )
        value = nmae_utility + GAMMA * ficr_utility
        local_index = np.argmax(value, axis=1)
        local_value = value[np.arange(len(point)), local_index]
        improve = local_value > best_value
        best_value[improve] = local_value[improve]
        best_action[improve] = actions[local_index[improve]]
    return np.clip(best_action, 0.0, cap_high)


def _fit_arm_fold(
    arm: str,
    fold: str,
    frame: pd.DataFrame,
    base: pd.DataFrame,
    target: np.ndarray,
    weights: np.ndarray,
    assessment_positions: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any], list[dict[str, Any]]]:
    candidate = arm == "R2_VERTICAL_PCA2X2"
    fit_max = FIT_MAX[fold]
    calibration_start = fit_max - pd.Timedelta(days=CALIBRATION_DAYS)
    timestamp = frame["forecast_kst_dtm"]
    finite_target = np.isfinite(target)
    preliminary_mask = (
        timestamp.le(calibration_start).to_numpy() & finite_target
    )
    calibration_mask = (
        timestamp.gt(calibration_start).to_numpy()
        & timestamp.le(fit_max).to_numpy()
        & finite_target
    )
    final_mask = timestamp.le(fit_max).to_numpy() & finite_target

    preliminary_additions, preliminary_loadings = _fit_pca_additions(
        base,
        frame,
        calibration_start,
        candidate,
    )
    preliminary_matrix = _model_matrix(base, preliminary_additions)
    preliminary_model = _fit_model(
        preliminary_matrix,
        target,
        preliminary_mask,
        weights,
    )
    calibration_prediction = np.clip(
        preliminary_model.predict(preliminary_matrix.loc[calibration_mask]),
        0.0,
        1.05,
    )
    calibration_target = target[calibration_mask]
    calibration_group = frame.loc[calibration_mask, "group_id"].to_numpy(dtype=int)
    calibration_residual = calibration_target - calibration_prediction
    residual_quantiles: dict[int, np.ndarray] = {}
    residual_counts: dict[str, int] = {}
    for group in (1, 2, 3):
        residual = calibration_residual[calibration_group == group]
        residual = residual[np.isfinite(residual)]
        if len(residual) < 240:
            raise RuntimeError(f"{arm}/{fold}/g{group}: insufficient calibration residuals")
        residual_quantiles[group] = np.quantile(residual, QUANTILE_LEVELS)
        residual_counts[str(group)] = len(residual)
    del preliminary_model, preliminary_matrix, preliminary_additions
    gc.collect()

    final_additions, final_loadings = _fit_pca_additions(base, frame, fit_max, candidate)
    final_matrix = _model_matrix(base, final_additions)
    final_model = _fit_model(final_matrix, target, final_mask, weights)
    assessment_point = np.clip(
        final_model.predict(final_matrix.iloc[assessment_positions]),
        0.0,
        1.05,
    )
    assessment_group = frame.iloc[assessment_positions]["group_id"].to_numpy(dtype=int)
    prediction = np.empty(len(assessment_positions), dtype=float)
    mean_generation: dict[str, float] = {}
    for group in (1, 2, 3):
        train_group = final_mask & frame["group_id"].eq(group).to_numpy()
        mean_cf = float(np.mean(target[train_group]))
        if not np.isfinite(mean_cf) or mean_cf <= 0:
            raise RuntimeError(f"{arm}/{fold}/g{group}: invalid past mean generation")
        mean_generation[str(group)] = mean_cf
        selected = assessment_group == group
        action_cf = _settlement_action(
            assessment_point[selected],
            residual_quantiles[group],
            mean_cf,
            SOFT_CAPS[group],
        )
        prediction[selected] = action_cf * CAPACITIES_KWH[group]
    if not np.isfinite(prediction).all():
        raise RuntimeError(f"{arm}/{fold}: nonfinite assessment prediction")
    details = {
        "arm": arm,
        "fold": fold,
        "fit_max_time": fit_max.isoformat(),
        "calibration_start_time": calibration_start.isoformat(),
        "preliminary_fit_rows": int(preliminary_mask.sum()),
        "calibration_rows": int(calibration_mask.sum()),
        "final_fit_rows": int(final_mask.sum()),
        "assessment_rows": len(assessment_positions),
        "residual_counts": residual_counts,
        "residual_quantiles": {
            str(group): values.tolist() for group, values in residual_quantiles.items()
        },
        "past_mean_generation_cf": mean_generation,
        "model_parameters": MODEL_PARAMS,
        "decision_policy": "FIXED_T0.5_G1.5_DIRECT_CF_RESID81",
        "score_calls": 0,
    }
    loadings = [
        {"arm": arm, "fold": fold, "fit_stage": "preliminary", **record}
        for record in preliminary_loadings
    ] + [
        {"arm": arm, "fold": fold, "fit_stage": "final", **record}
        for record in final_loadings
    ]
    del final_model, final_matrix, final_additions
    gc.collect()
    return prediction, details, loadings


def materialize(repo: Path, predeclaration: Path, output_dir: Path) -> dict[str, Any]:
    frozen = _verify_inputs(repo, predeclaration)
    frame, base, numeric_columns = _load_development_frame(repo)
    n9_predictions = pd.read_parquet(
        repo / "artifacts/backtests/s17_n9_cost5_recovery/predictions.parquet"
    )
    n9_predictions["forecast_kst_dtm"] = pd.to_datetime(
        n9_predictions["forecast_kst_dtm"]
    )
    assessment_positions = _assessment_indices(frame, n9_predictions)
    predictions = n9_predictions[[*KEYS, "actual_kwh", "CHAMPION"]].copy()
    target = frame["cf"].to_numpy(dtype=float)
    weights = np.where(
        np.isfinite(target) & (target >= 0.1),
        np.clip(target, 0.0, 1.2),
        0.05,
    )
    fit_details: list[dict[str, Any]] = []
    loadings: list[dict[str, Any]] = []
    model_fit_count = 0
    for fold in FIT_MAX:
        fold_mask = predictions["fold_id"].eq(fold).to_numpy()
        fold_positions = assessment_positions[fold_mask]
        if len(fold_positions) == 0:
            raise RuntimeError(f"{fold}: empty assessment")
        for arm in ARMS:
            prediction, details, arm_loadings = _fit_arm_fold(
                arm,
                fold,
                frame,
                base,
                target,
                weights,
                fold_positions,
            )
            predictions.loc[fold_mask, arm] = prediction
            fit_details.append(details)
            loadings.extend(arm_loadings)
            model_fit_count += 2
    if model_fit_count != 12:
        raise RuntimeError("N13 did not perform exactly 12 frozen model fits")
    if predictions[list(ARMS)].isna().any().any():
        raise RuntimeError("N13 prediction materialization incomplete")

    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "predictions.parquet"
    predictions.to_parquet(prediction_path, index=False)
    fit_path = output_dir / "fit_details.json"
    fit_payload = {
        "schema_version": 1,
        "node_id": "S17-N13_R2_TWO_COMPONENT_VERTICAL_PCA_STRICT_PREQUENTIAL",
        "base_numeric_feature_count": len(numeric_columns),
        "base_numeric_features_sha256": hashlib.sha256(
            "\n".join(numeric_columns).encode()
        ).hexdigest(),
        "model_fit_count": model_fit_count,
        "score_calls": 0,
        "fits": fit_details,
    }
    fit_path.write_text(json.dumps(fit_payload, ensure_ascii=False, indent=2) + "\n")
    loadings_path = output_dir / "pca_loadings.json"
    loadings_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "candidate": "R2_VERTICAL_PCA2X2",
                "control": "four zero columns",
                "records": loadings,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )

    n9_provenance = pd.read_parquet(
        repo / "artifacts/backtests/s17_n9_cost5_recovery/procedure_provenance.parquet"
    )
    provenance_rows: list[dict[str, Any]] = []
    predeclaration_hash = _sha256(predeclaration)
    for fold in ("dev-2023-Q3", "dev-2023-Q4"):
        champion = n9_provenance.loc[
            n9_provenance["model_id"].eq("CHAMPION")
            & n9_provenance["test_fold"].eq(fold)
        ]
        if len(champion) != 1:
            raise RuntimeError(f"{fold}: missing frozen champion provenance")
        provenance_rows.append(champion.iloc[0].to_dict())
        for arm in ARMS:
            provenance_rows.append(
                {
                    "model_id": arm,
                    "test_fold": fold,
                    "fit_max_time": FIT_MAX[fold],
                    "selection_max_time": SELECTION_MAX[fold],
                    "policy_id": "FIXED_T0.5_G1.5_DIRECT_CF_RESID81",
                    "predeclaration_sha256": predeclaration_hash,
                    "prediction_sha256": _vector_hash(predictions, arm, fold),
                    "weights_fit": "past_only_expanding_with_past_calibration_tail",
                }
            )
    provenance = pd.DataFrame(provenance_rows).sort_values(
        ["test_fold", "model_id"], kind="stable"
    )
    for column in ("fit_max_time", "selection_max_time"):
        provenance[column] = pd.to_datetime(provenance[column])
    provenance_path = output_dir / "procedure_provenance.parquet"
    provenance.to_parquet(provenance_path, index=False)

    output_hashes = {
        path.name: _sha256(path)
        for path in (prediction_path, provenance_path, fit_path, loadings_path)
    }
    family = {
        "schema_version": 1,
        "node_id": "S17-N13_R2_TWO_COMPONENT_VERTICAL_PCA_STRICT_PREQUENTIAL",
        "predeclaration_sha256": predeclaration_hash,
        "input_bundle_sha256": frozen["input_bundle"]["sha256"],
        "code_sha256": _sha256(Path(__file__)),
        "family": list(FAMILY),
        "incumbent": "CHAMPION",
        "control": "R2_CONTROL_ZERO4",
        "candidate": "R2_VERTICAL_PCA2X2",
        "comparison_index": 2,
        "output_hashes": output_hashes,
        "prediction_vectors": {
            model: {
                fold: _vector_hash(predictions, model, fold)
                for fold in ("dev-2023-Q3", "dev-2023-Q4")
            }
            for model in FAMILY
        },
        "materialization": {
            "model_fits": model_fit_count,
            "official_score_calls": 0,
            "assessment_rows": len(predictions),
            "outer_rows": int((~predictions["fold_id"].eq("dev-2023-Q2")).sum()),
            "workers_per_fit": 6,
        },
        "forbidden_access": {
            "2024_operating_day_features_or_labels": False,
            "test_period": False,
            "rejected_ecmwf": False,
            "scada_action_features": False,
            "dacon_actions": [],
        },
    }
    family_path = output_dir / "family_manifest.json"
    family_path.write_text(json.dumps(family, ensure_ascii=False, indent=2) + "\n")
    return {
        "family_manifest_sha256": _sha256(family_path),
        "outputs": output_hashes,
        "model_fits": model_fit_count,
        "official_score_calls": 0,
    }


def _metric_frame(frame: pd.DataFrame, prediction: str) -> pd.DataFrame:
    metric = frame[["forecast_kst_dtm", "group_id", "actual_kwh", prediction]].copy()
    metric.insert(0, "forecast_id", np.arange(len(metric), dtype=np.int64))
    return metric.rename(columns={prediction: "prediction_kwh"})


def _score_json(score: Any) -> dict[str, Any]:
    return {
        "total": float(score.total),
        "one_minus_nmae": float(score.one_minus_nmae),
        "ficr": float(score.ficr),
        "group_nmae": {str(key): float(value) for key, value in score.group_nmae.items()},
        "group_ficr": {str(key): float(value) for key, value in score.group_ficr.items()},
    }


def evaluate(
    repo: Path,
    predeclaration: Path,
    output_dir: Path,
    expected_family_sha256: str,
) -> dict[str, Any]:
    frozen = _verify_inputs(repo, predeclaration)
    family_path = output_dir / "family_manifest.json"
    if _sha256(family_path) != expected_family_sha256:
        raise RuntimeError("N13 family manifest freeze mismatch")
    family_manifest = json.loads(family_path.read_text())
    for name, digest in family_manifest["output_hashes"].items():
        if _sha256(output_dir / name) != digest:
            raise RuntimeError(f"N13 post-freeze artifact mutation: {name}")
    predictions = pd.read_parquet(output_dir / "predictions.parquet")
    predictions["forecast_kst_dtm"] = pd.to_datetime(predictions["forecast_kst_dtm"])
    provenance = pd.read_parquet(output_dir / "procedure_provenance.parquet")
    event_store = EventStore(repo, repo / "artifacts/registry/loop_events_s17.sqlite")
    protocol = run_prequential_protocol(
        predictions,
        prediction_columns=list(FAMILY),
        incumbent="CHAMPION",
        capacities=CAPACITIES_KWH,
        procedure_provenance=provenance,
        family_manifest_sha256=expected_family_sha256,
        comparison_index=2,
        event_store=event_store,
        n_rep=4999,
        seed=20260808,
        block_lengths=(3, 7, 14),
        margin_total=0.001635,
    )
    outer = predictions.loc[~predictions["fold_id"].eq("dev-2023-Q2")].copy()
    scores = {
        model: _score_json(
            evaluate_official(_metric_frame(outer, model), CAPACITIES_KWH)
        )
        for model in FAMILY
    }
    candidate_delta_incumbent = (
        scores["R2_VERTICAL_PCA2X2"]["total"] - scores["CHAMPION"]["total"]
    )
    candidate_delta_control = (
        scores["R2_VERTICAL_PCA2X2"]["total"]
        - scores["R2_CONTROL_ZERO4"]["total"]
    )
    protocol_delta = protocol["blocks"]["7"]["joint_max_t"]["candidates"][
        "R2_VERTICAL_PCA2X2"
    ]["observed_delta_total"]
    if abs(candidate_delta_incumbent - protocol_delta) >= 1e-12:
        raise RuntimeError("N13 point/protocol candidate delta disagreement")
    stable = protocol["promotion_stable_all_blocks"]["R2_VERTICAL_PCA2X2"]
    promotion_supported = bool(
        candidate_delta_incumbent >= 0.001635
        and candidate_delta_control >= 0.00357236259
        and stable
        and protocol["inference"] == "SUPPORTED"
    )
    goal_reached = bool(
        promotion_supported and scores["R2_VERTICAL_PCA2X2"]["total"] >= 0.66
    )
    result = {
        "schema_version": 1,
        "node_id": "S17-N13_R2_TWO_COMPONENT_VERTICAL_PCA_STRICT_PREQUENTIAL",
        "family_manifest_sha256": expected_family_sha256,
        "comparison_index": 2,
        "scores": scores,
        "candidate_delta_incumbent_total": candidate_delta_incumbent,
        "candidate_delta_control_total": candidate_delta_control,
        "promotion_supported": promotion_supported,
        "goal_total_0p66_reached": goal_reached,
        "protocol": protocol,
        "score_calls": {
            "outer_point_official": 3,
            "strict_prequential_protocol": 1,
        },
        "model_fits_during_evaluation": 0,
        "evidence_label": frozen["evaluation"]["evidence_label"],
        "forbidden_access": family_manifest["forbidden_access"],
    }
    evaluation_path = output_dir / "evaluation.json"
    evaluation_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("materialize", "evaluate"))
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--predeclaration",
        type=Path,
        default=Path("reports/s17_n13_r2_vertical_pca_predeclaration.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/backtests/s17_n13_r2_vertical_pca"),
    )
    parser.add_argument("--family-sha256", default="")
    args = parser.parse_args()
    repo = args.repo.resolve()
    predeclaration = args.predeclaration
    if not predeclaration.is_absolute():
        predeclaration = repo / predeclaration
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = repo / output_dir
    if args.phase == "materialize":
        result = materialize(repo, predeclaration, output_dir)
    else:
        if len(args.family_sha256) != 64:
            raise RuntimeError("evaluate requires frozen --family-sha256")
        result = evaluate(repo, predeclaration, output_dir, args.family_sha256)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
