"""Typed, network-free orchestration services for the closed local workflow."""

import json
import os
import platform
import re
import time
from argparse import Namespace
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from baram.config import ProjectConfig, load_config
from baram.constants import CAPACITIES_KWH
from baram.contracts.hashing import canonical_sha256, sha256_dataframe, sha256_file
from baram.contracts.types import BlendPolicy, CalibrationPolicy, FoldSpec, SourceSpec
from baram.data.archive import validate_archive
from baram.data.canonical import load_canonical_tables
from baram.data.quality import audit_quality
from baram.decisions.blend import apply_blend, fit_two_model_blend
from baram.decisions.calibrate import (
    apply_calibration,
    cross_fit_calibration,
    fit_group_calibration,
)
from baram.evaluation.failure_slices import shared_failure_slices
from baram.evaluation.official import evaluate_official
from baram.exceptions import ContractError
from baram.experiments.promotion import (
    decide_challenger_activation,
    decide_development_promotion,
    decide_diversity,
    decide_lockbox,
    decide_reproduction,
)
from baram.experiments.registry import write_json_atomic
from baram.features.climatology import apply_climatology, fit_climatology
from baram.features.physics import add_physics_features
from baram.features.weather import build_weather_features
from baram.models.baselines import (
    fit_physics_proxy,
    fit_supplied_rf_bundle,
    predict_bundle,
    predict_physics_proxy,
)
from baram.models.lightgbm import expand_lgbm_grid, fit_lgbm_bundle
from baram.models.oof import filter_complete_validation_rows, generate_oof
from baram.submission.build import build_submission
from baram.submission.validate import validate_submission
from baram.validation.splits import (
    build_development_folds,
    build_group12_diagnostic_folds,
    build_lockbox,
)

_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_KEYS = ["forecast_id", "forecast_kst_dtm", "group_id"]
_NON_FEATURES = {
    "forecast_id",
    "forecast_kst_dtm",
    "data_available_kst_dtm",
    "issuance_batch",
    "operating_day",
    "operating_year",
    "group_id",
    "capacity_kwh",
}
_RF_DETERMINISTIC_JOBS = 1


@dataclass(frozen=True)
class WorkflowResult:
    receipt_paths: tuple[Path, ...]
    summary_sha256: str


def _config_from_args(args: Namespace) -> ProjectConfig:
    return load_config(Path(args.config))


def _run_id(args: Namespace) -> str:
    value = str(args.run_id)
    if _RUN_ID.fullmatch(value) is None:
        raise ContractError("run ID must be a safe 1-128 character local identifier")
    return value


def _write_parquet_atomic(frame: pd.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, engine="pyarrow", compression="zstd")
    temporary.replace(path)
    return sha256_file(path)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ContractError(f"required receipt does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError(f"cannot read receipt {path}: {error}") from error
    if not isinstance(value, dict):
        raise ContractError(f"receipt root must be an object: {path}")
    return value


def _cache_root(config: ProjectConfig) -> Path:
    return config.repo_root / "artifacts" / "cache" / config.open_zip.sha256


def _prepare_manifest_path(config: ProjectConfig) -> Path:
    return config.repo_root / "artifacts" / "manifests" / "prepare.json"


def _split_manifest_path(config: ProjectConfig) -> Path:
    return config.repo_root / "artifacts" / "manifests" / "splits.json"


def _stage_manifest_path(config: ProjectConfig, stage: str) -> Path:
    return config.repo_root / "artifacts" / "manifests" / f"backtest-{stage}.json"


def _feature_names(frame: pd.DataFrame) -> tuple[str, ...]:
    return tuple(
        name
        for name in frame.columns
        if name not in _NON_FEATURES and pd.api.types.is_numeric_dtype(frame[name])
    )


def _downcast_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for name in result.select_dtypes(include=["float64"]).columns:
        result[name] = result[name].astype("float32")
    return result


def run_audit(args: Namespace) -> WorkflowResult:
    """Verify immutable sources and write a deterministic local P0 receipt."""
    config = _config_from_args(args)
    manifest = validate_archive(config.open_zip.path, config.open_zip.sha256)
    baseline_hash = sha256_file(config.baseline_notebook.path)
    if baseline_hash != config.baseline_notebook.sha256:
        raise ContractError("baseline notebook SHA-256 differs from the frozen configuration")
    payload = {
        "archive": asdict(manifest),
        "baseline_notebook_sha256": baseline_hash,
        "configuration_sha256": canonical_sha256(
            {
                "capacities": dict(config.capacities),
                "seed": config.seed,
                "n_jobs": config.n_jobs,
                "artifact_budget_gib": config.artifact_budget_gib,
                "lockbox_year": config.lockbox_year,
            }
        ),
        "external_actions": [],
    }
    path = config.repo_root / "reports" / "audit_receipt.json"
    digest = write_json_atomic(path, payload)
    return WorkflowResult((path,), digest)


def run_prepare(args: Namespace) -> WorkflowResult:
    """Build quality-audited, content-addressed train/test feature caches."""
    config = _config_from_args(args)
    run_id = _run_id(args)
    validate_archive(config.open_zip.path, config.open_zip.sha256)
    tables = load_canonical_tables(config.open_zip.path)
    quality = audit_quality(tables, config.capacities)

    train_features = _downcast_features(
        add_physics_features(build_weather_features(tables.gfs_train, tables.ldaps_train))
    )
    test_features = _downcast_features(
        add_physics_features(
            build_weather_features(
                tables.gfs_test,
                tables.ldaps_test,
                tables.submission_keys[["forecast_id", "forecast_kst_dtm"]],
            )
        )
    )
    names = _feature_names(train_features)
    if names != _feature_names(test_features):
        raise ContractError("prepared train/test model feature columns differ")

    cache = _cache_root(config)
    paths = {
        "train_features": cache / "train_features.parquet",
        "test_features": cache / "test_features.parquet",
        "labels_long": cache / "labels_long.parquet",
        "submission_keys": cache / "submission_keys.parquet",
    }
    hashes = {
        "train_features": _write_parquet_atomic(train_features, paths["train_features"]),
        "test_features": _write_parquet_atomic(test_features, paths["test_features"]),
        "labels_long": _write_parquet_atomic(tables.labels_long, paths["labels_long"]),
        "submission_keys": _write_parquet_atomic(tables.submission_keys, paths["submission_keys"]),
    }
    total_bytes = sum(path.stat().st_size for path in paths.values())
    if total_bytes > config.artifact_budget_gib * 1024**3:
        raise ContractError("prepared cache exceeds the configured artifact budget")
    payload = {
        "run_id": run_id,
        "source_sha256": config.open_zip.sha256,
        "quality_receipt": asdict(quality.receipt),
        "quality_findings": quality.findings,
        "feature_names": names,
        "rows": {
            "train_features": len(train_features),
            "test_features": len(test_features),
            "labels_long": len(tables.labels_long),
            "submission_keys": len(tables.submission_keys),
        },
        "artifact_hashes": hashes,
        "artifact_bytes": total_bytes,
    }
    manifest_path = _prepare_manifest_path(config)
    digest = write_json_atomic(manifest_path, payload)
    run_path = config.repo_root / "reports" / "runs" / f"{run_id}-prepare.json"
    write_json_atomic(run_path, {"prepare_manifest_sha256": digest, "run_id": run_id})
    return WorkflowResult((manifest_path, run_path), digest)


def _folds_payload(folds: tuple[FoldSpec, ...]) -> list[dict[str, Any]]:
    return [asdict(fold) for fold in folds]


def run_split_build(args: Namespace) -> WorkflowResult:
    """Build development, supplemental, and unopened lockbox split manifests."""
    config = _config_from_args(args)
    run_id = _run_id(args)
    prepare_manifest = _read_json(_prepare_manifest_path(config))
    if prepare_manifest.get("source_sha256") != config.open_zip.sha256:
        raise ContractError("prepare manifest source does not match configuration")
    train = pd.read_parquet(_cache_root(config) / "train_features.parquet")
    split_frame = train.loc[train["group_id"].eq(1)].copy()
    split_frame["grid_id"] = 0
    development = build_development_folds(split_frame)
    diagnostics = build_group12_diagnostic_folds(split_frame)
    lockbox = build_lockbox(split_frame, config.lockbox_year)
    payload = {
        "run_id": run_id,
        "source_sha256": config.open_zip.sha256,
        "prepare_manifest_sha256": canonical_sha256(prepare_manifest),
        "development": _folds_payload(development),
        "group12_diagnostics": _folds_payload(diagnostics),
        "lockbox_unconsumed_spec": asdict(lockbox),
    }
    manifest_path = _split_manifest_path(config)
    digest = write_json_atomic(manifest_path, payload)
    run_path = config.repo_root / "reports" / "runs" / f"{run_id}-splits.json"
    write_json_atomic(run_path, {"run_id": run_id, "split_manifest_sha256": digest})
    return WorkflowResult((manifest_path, run_path), digest)


def _fold_from_payload(raw: dict[str, Any]) -> FoldSpec:
    try:
        return FoldSpec(
            fold_id=str(raw["fold_id"]),
            train_batches=tuple(str(item) for item in raw["train_batches"]),
            validation_batches=tuple(str(item) for item in raw["validation_batches"]),
            eligible_groups=tuple(int(item) for item in raw["eligible_groups"]),  # type: ignore[arg-type]
            official_total_eligible=bool(raw["official_total_eligible"]),
            is_lockbox=bool(raw.get("is_lockbox", False)),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ContractError(f"invalid fold receipt: {error}") from error


def _load_pipeline_inputs(
    config: ProjectConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[str, ...], tuple[FoldSpec, ...]]:
    validate_archive(config.open_zip.path, config.open_zip.sha256)
    prepare = _read_json(_prepare_manifest_path(config))
    splits = _read_json(_split_manifest_path(config))
    if prepare.get("source_sha256") != config.open_zip.sha256:
        raise ContractError("prepared source hash differs from the current configuration")
    if splits.get("source_sha256") != config.open_zip.sha256:
        raise ContractError("split source hash differs from the current configuration")
    cache = _cache_root(config)
    artifacts = {
        "train_features": cache / "train_features.parquet",
        "labels_long": cache / "labels_long.parquet",
    }
    expected_hashes = prepare.get("artifact_hashes")
    if not isinstance(expected_hashes, dict):
        raise ContractError("prepare manifest has no artifact hashes")
    for name, path in artifacts.items():
        if not path.is_file() or sha256_file(path) != expected_hashes.get(name):
            raise ContractError(f"prepared artifact hash mismatch: {name}")
    features = pd.read_parquet(artifacts["train_features"])
    labels = pd.read_parquet(artifacts["labels_long"])
    names = tuple(str(item) for item in prepare.get("feature_names", ()))
    if not names or any(name not in features for name in names):
        raise ContractError("prepare manifest feature contract is invalid")
    raw_folds = splits.get("development")
    if not isinstance(raw_folds, list):
        raise ContractError("split manifest has no development folds")
    folds = tuple(_fold_from_payload(item) for item in raw_folds)
    return features, labels, names, folds


def _score_payload(frame: pd.DataFrame, capacities: dict[int, float]) -> dict[str, Any]:
    metric = frame[[*_KEYS, "actual_kwh", "prediction_kwh"]]
    pooled = evaluate_official(metric, capacities)
    fold_scores = {
        str(fold_id): asdict(
            evaluate_official(part[[*_KEYS, "actual_kwh", "prediction_kwh"]], capacities)
        )
        for fold_id, part in frame.groupby("fold_id", sort=True)
    }
    return {"pooled": asdict(pooled), "folds": fold_scores}


def _manual_control_oof(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    folds: tuple[FoldSpec, ...],
    family: str,
) -> pd.DataFrame:
    merged = features.merge(
        labels[[*_KEYS, "actual_kwh"]],
        on=_KEYS,
        how="left",
        validate="one_to_one",
        sort=False,
    )
    outputs: list[pd.DataFrame] = []
    for fold in folds:
        training = merged.loc[merged["issuance_batch"].isin(fold.train_batches)]
        validation = merged.loc[
            merged["issuance_batch"].isin(fold.validation_batches)
            & merged["group_id"].isin(fold.eligible_groups)
        ].copy()
        validation = filter_complete_validation_rows(validation, fold.eligible_groups)
        if family == "climatology":
            state = fit_climatology(training, fold.fold_id)
            predicted = apply_climatology(state, validation, fold.fold_id)["clim_median"]
            validation["prediction_kwh"] = predicted.to_numpy(dtype=float)
        elif family == "physics":
            validation["prediction_kwh"] = np.nan
            for group_id in fold.eligible_groups:
                train_group = training.loc[training["group_id"].eq(group_id)].dropna(
                    subset=["actual_kwh"]
                )
                valid_mask = validation["group_id"].eq(group_id)
                proxy = fit_physics_proxy(
                    train_group["phys__rho_v3"],
                    train_group["actual_kwh"],
                    CAPACITIES_KWH[group_id],
                )
                validation.loc[valid_mask, "prediction_kwh"] = predict_physics_proxy(
                    proxy, validation.loc[valid_mask, "phys__rho_v3"]
                )
        else:
            raise ContractError(f"unsupported manual control family: {family}")
        part = validation[[*_KEYS, "actual_kwh", "prediction_kwh"]].copy()
        part["fold_id"] = fold.fold_id
        part["model_id"] = f"{family}-{fold.fold_id}"
        outputs.append(part)
    result = pd.concat(outputs, ignore_index=True)
    if not np.isfinite(result["prediction_kwh"].to_numpy(dtype=float)).all():
        raise ContractError(f"{family} control produced non-finite predictions")
    return result.sort_values(["forecast_kst_dtm", "group_id"], kind="stable").reset_index(
        drop=True
    )


def _backtest_controls(config: ProjectConfig, run_id: str) -> WorkflowResult:
    features, labels, feature_names, folds = _load_pipeline_inputs(config)
    predictions: dict[str, pd.DataFrame] = {
        "climatology": _manual_control_oof(features, labels, folds, "climatology"),
        "physics": _manual_control_oof(features, labels, folds, "physics"),
    }
    rf = generate_oof(
        features,
        labels,
        folds,
        feature_names,
        family="random_forest",
        architecture="group_specific",
        params={},
        seed=config.seed,
        n_jobs=_RF_DETERMINISTIC_JOBS,
    ).predictions
    rf_repeat = generate_oof(
        features,
        labels,
        folds,
        feature_names,
        family="random_forest",
        architecture="group_specific",
        params={},
        seed=config.seed,
        n_jobs=_RF_DETERMINISTIC_JOBS,
    ).predictions
    repeat_match = sha256_dataframe(rf) == sha256_dataframe(rf_repeat)
    if not repeat_match:
        raise ContractError("supplied RandomForest control is not reproducible")
    predictions["random_forest"] = rf

    artifact_root = config.repo_root / "artifacts" / "backtests" / "controls" / run_id
    scores: dict[str, dict[str, Any]] = {}
    artifacts: dict[str, dict[str, str]] = {}
    for family, frame in predictions.items():
        path = artifact_root / f"{family}-oof.parquet"
        artifacts[family] = {
            "path": str(path.relative_to(config.repo_root)),
            "sha256": _write_parquet_atomic(frame, path),
        }
        scores[family] = _score_payload(frame, dict(config.capacities))
    champion = max(
        scores,
        key=lambda family: (float(scores[family]["pooled"]["total"]), -len(family), family),
    )
    payload = {
        "run_id": run_id,
        "stage": "controls",
        "source_sha256": config.open_zip.sha256,
        "feature_names": feature_names,
        "scores": scores,
        "prediction_artifacts": artifacts,
        "champion": champion,
        "baseline_repeat_hash_match": repeat_match,
        "configuration_slots_used": 0,
    }
    manifest_path = _stage_manifest_path(config, "controls")
    digest = write_json_atomic(manifest_path, payload)
    report_path = config.repo_root / "reports" / "development_controls.json"
    write_json_atomic(report_path, payload)
    return WorkflowResult((manifest_path, report_path), digest)


def _candidate_id(params: dict[str, object], architecture: str) -> str:
    return f"lightgbm-{architecture}-{canonical_sha256(params)[:16]}"


def _backtest_lightgbm(config: ProjectConfig, run_id: str) -> WorkflowResult:
    features, labels, feature_names, folds = _load_pipeline_inputs(config)
    configs = expand_lgbm_grid(config.repo_root / "configs" / "models" / "lightgbm.yaml")
    first_fold = (folds[0],)
    first_results: dict[str, pd.DataFrame] = {}
    first_records: list[dict[str, Any]] = []
    specifications: dict[str, dict[str, Any]] = {}
    for params in configs:
        for architecture in ("group_specific", "shared"):
            candidate_id = _candidate_id(params, architecture)
            frame = generate_oof(
                features,
                labels,
                first_fold,
                feature_names,
                family="lightgbm",
                architecture=architecture,  # type: ignore[arg-type]
                params=params,
                seed=config.seed,
                n_jobs=config.n_jobs,
            ).predictions
            score = _score_payload(frame, dict(config.capacities))
            first_results[candidate_id] = frame
            specifications[candidate_id] = {
                "candidate_id": candidate_id,
                "family": "lightgbm",
                "architecture": architecture,
                "params": params,
                "feature_names": feature_names,
            }
            first_records.append(
                {
                    "candidate_id": candidate_id,
                    "architecture": architecture,
                    "params_sha256": canonical_sha256(params),
                    "score": score,
                }
            )
    ranked_first = sorted(
        first_records,
        key=lambda item: (-float(item["score"]["pooled"]["total"]), item["candidate_id"]),
    )
    promoted_ids = [str(item["candidate_id"]) for item in ranked_first[:6]]

    artifact_root = config.repo_root / "artifacts" / "backtests" / "lightgbm" / run_id
    full_records: list[dict[str, Any]] = []
    for candidate_id in promoted_ids:
        spec = specifications[candidate_id]
        remaining = generate_oof(
            features,
            labels,
            folds[1:],
            feature_names,
            family="lightgbm",
            architecture=spec["architecture"],  # type: ignore[arg-type]
            params=spec["params"],
            seed=config.seed,
            n_jobs=config.n_jobs,
        ).predictions
        full = (
            pd.concat([first_results[candidate_id], remaining], ignore_index=True)
            .sort_values(["forecast_kst_dtm", "group_id"], kind="stable")
            .reset_index(drop=True)
        )
        path = artifact_root / f"{candidate_id}-oof.parquet"
        prediction_hash = _write_parquet_atomic(full, path)
        full_records.append(
            {
                **spec,
                "score": _score_payload(full, dict(config.capacities)),
                "prediction_path": str(path.relative_to(config.repo_root)),
                "prediction_sha256": prediction_hash,
            }
        )
    ranked_full = sorted(
        full_records,
        key=lambda item: (-float(item["score"]["pooled"]["total"]), item["candidate_id"]),
    )
    finalist_ids = [str(item["candidate_id"]) for item in ranked_full[:3]]
    stability: dict[str, dict[str, Any]] = {}
    full_by_id = {str(item["candidate_id"]): item for item in full_records}
    for candidate_id in finalist_ids:
        spec = specifications[candidate_id]
        seed_scores: dict[str, dict[str, Any]] = {
            str(config.seed): full_by_id[candidate_id]["score"]
        }
        for seed in (config.seed + 1, config.seed + 2):
            frame = generate_oof(
                features,
                labels,
                folds,
                feature_names,
                family="lightgbm",
                architecture=spec["architecture"],  # type: ignore[arg-type]
                params=spec["params"],
                seed=seed,
                n_jobs=config.n_jobs,
            ).predictions
            seed_scores[str(seed)] = _score_payload(frame, dict(config.capacities))
            seed_path = artifact_root / f"{candidate_id}-seed-{seed}-oof.parquet"
            _write_parquet_atomic(frame, seed_path)
        totals = [float(item["pooled"]["total"]) for item in seed_scores.values()]
        stability[candidate_id] = {
            "seed_scores": seed_scores,
            "mean_total": float(np.mean(totals)),
            "min_total": float(np.min(totals)),
            "max_total": float(np.max(totals)),
        }
    champion_id = max(
        finalist_ids,
        key=lambda item: (stability[item]["mean_total"], -len(item), item),
    )
    champion = full_by_id[champion_id]
    payload = {
        "run_id": run_id,
        "stage": "lightgbm",
        "source_sha256": config.open_zip.sha256,
        "configuration_slots_used": len(configs),
        "architectures_per_configuration": 2,
        "first_fold_candidates": first_records,
        "promoted_full_fold_candidates": full_records,
        "seed_stability": stability,
        "champion": champion,
    }
    manifest_path = _stage_manifest_path(config, "lightgbm")
    digest = write_json_atomic(manifest_path, payload)
    report_path = config.repo_root / "reports" / "development_lightgbm.json"
    write_json_atomic(report_path, payload)
    return WorkflowResult((manifest_path, report_path), digest)


def _fold_total_scores(score: dict[str, Any]) -> dict[str, float]:
    return {fold: float(value["total"]) for fold, value in score["folds"].items()}


def _feature_families(feature_names: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    calendar = {
        "operating_quarter",
        "hour",
        "month",
        "day_of_year",
        "lead_hour",
        "cal__hour_sin",
        "cal__hour_cos",
        "cal__doy_sin",
        "cal__doy_cos",
    }
    f0 = tuple(
        name
        for name in feature_names
        if name in calendar
        or name.endswith("__mean")
        or name.endswith("__missing_cell_count")
        or name.endswith("__grid_count")
    )
    f1 = tuple(name for name in feature_names if not name.startswith("phys__"))
    f2 = feature_names
    if not f0 or not f1 or not f2:
        raise ContractError("feature-family routing produced an empty mandatory family")
    return {"F0": f0, "F1": f1, "F2": f2}


def _backtest_ablation(config: ProjectConfig, run_id: str) -> WorkflowResult:
    features, labels, all_feature_names, folds = _load_pipeline_inputs(config)
    lightgbm_manifest = _read_json(_stage_manifest_path(config, "lightgbm"))
    champion = lightgbm_manifest.get("champion")
    if not isinstance(champion, dict):
        raise ContractError("LightGBM manifest has no champion")
    params = champion.get("params")
    architecture = champion.get("architecture")
    if not isinstance(params, dict) or architecture not in {"group_specific", "shared"}:
        raise ContractError("LightGBM champion specification is invalid")
    families = _feature_families(all_feature_names)
    artifact_root = config.repo_root / "artifacts" / "backtests" / "ablation" / run_id
    records: dict[str, dict[str, Any]] = {}
    predictions: dict[str, pd.DataFrame] = {}
    for family_name, names in families.items():
        if family_name == "F2" and tuple(champion.get("feature_names", ())) == names:
            path = config.repo_root / str(champion["prediction_path"])
            if sha256_file(path) != champion.get("prediction_sha256"):
                raise ContractError("LightGBM champion prediction artifact hash mismatch")
            frame = pd.read_parquet(path)
        else:
            frame = generate_oof(
                features,
                labels,
                folds,
                names,
                family="lightgbm",
                architecture=architecture,  # type: ignore[arg-type]
                params=params,
                seed=config.seed,
                n_jobs=config.n_jobs,
            ).predictions
        predictions[family_name] = frame
        path = artifact_root / f"{family_name}-oof.parquet"
        records[family_name] = {
            "feature_names": names,
            "score": _score_payload(frame, dict(config.capacities)),
            "prediction_path": str(path.relative_to(config.repo_root)),
            "prediction_sha256": _write_parquet_atomic(frame, path),
        }

    selected = "F0"
    decisions: dict[str, dict[str, Any]] = {}
    for challenger in ("F1", "F2"):
        control_score = records[selected]["score"]
        challenger_score = records[challenger]["score"]
        control_folds = _fold_total_scores(control_score)
        challenger_folds = _fold_total_scores(challenger_score)
        fold_deltas = [challenger_folds[fold] - control_folds[fold] for fold in control_folds]
        pooled_delta = float(challenger_score["pooled"]["total"]) - float(
            control_score["pooled"]["total"]
        )
        decision = decide_development_promotion(pooled_delta, fold_deltas, True)
        decisions[f"{selected}_to_{challenger}"] = asdict(decision)
        if decision.accepted:
            selected = challenger
    payload = {
        "run_id": run_id,
        "stage": "ablation",
        "source_sha256": config.open_zip.sha256,
        "base_candidate_id": champion.get("candidate_id"),
        "family_records": records,
        "promotion_decisions": decisions,
        "F3": {
            "status": "EVALUATED_AS_SEPARATE_FOLD_SAFE_CONTROL",
            "control_manifest": str(
                _stage_manifest_path(config, "controls").relative_to(config.repo_root)
            ),
            "reason": "target climatology remains an independently fold-fitted parent",
        },
        "selected_family": selected,
        "selected": records[selected],
        "model": {
            "family": "lightgbm",
            "architecture": architecture,
            "params": params,
            "seed": config.seed,
        },
    }
    manifest_path = _stage_manifest_path(config, "ablation")
    digest = write_json_atomic(manifest_path, payload)
    report_path = config.repo_root / "reports" / "feature_ablation.json"
    write_json_atomic(report_path, payload)
    return WorkflowResult((manifest_path, report_path), digest)


def _verified_prediction(
    config: ProjectConfig,
    record: dict[str, Any],
    *,
    path_key: str = "prediction_path",
    hash_key: str = "prediction_sha256",
) -> pd.DataFrame:
    path = config.repo_root / str(record.get(path_key, ""))
    if not path.is_file() or sha256_file(path) != record.get(hash_key):
        raise ContractError(f"prediction artifact hash mismatch: {path}")
    return pd.read_parquet(path)


def _failure_context(features: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    wind_name = "phys__hub117_speed"
    missing_names = ("gfs__missing_cell_count", "ldaps__missing_cell_count")
    required = {*_KEYS, "lead_hour", wind_name, *missing_names}
    if not required.issubset(features):
        missing = sorted(required - set(features))
        raise ContractError(f"failure context features are missing: {missing}")
    context = reference[_KEYS].merge(
        features[[*_KEYS, "lead_hour", wind_name, *missing_names]],
        on=_KEYS,
        how="left",
        validate="one_to_one",
        sort=False,
    )
    context = context.rename(columns={wind_name: "wind_speed"})
    context["nwp_missing_count"] = context[list(missing_names)].fillna(0.0).sum(axis=1)
    return context[[*_KEYS, "lead_hour", "wind_speed", "nwp_missing_count"]]


def _abs_residual_correlation(reference: pd.DataFrame, challenger: pd.DataFrame) -> float:
    columns = [*_KEYS, "actual_kwh", "prediction_kwh", "fold_id"]
    left = reference[columns].sort_values(_KEYS, kind="stable").reset_index(drop=True)
    right = challenger[columns].sort_values(_KEYS, kind="stable").reset_index(drop=True)
    if not left[[*_KEYS, "actual_kwh", "fold_id"]].equals(right[[*_KEYS, "actual_kwh", "fold_id"]]):
        raise ContractError("challenger and reference OOF keys, labels, or folds differ")
    left_residual = left["actual_kwh"].to_numpy(dtype=float) - left["prediction_kwh"].to_numpy(
        dtype=float
    )
    right_residual = right["actual_kwh"].to_numpy(dtype=float) - right["prediction_kwh"].to_numpy(
        dtype=float
    )
    if np.std(left_residual) <= 0.0 or np.std(right_residual) <= 0.0:
        return 1.0
    correlation = float(np.corrcoef(left_residual, right_residual)[0, 1])
    return abs(correlation) if np.isfinite(correlation) else 1.0


def _package_version(name: str) -> str:
    try:
        return package_version(name)
    except PackageNotFoundError:
        return "NOT_INSTALLED"


def _write_deep_tier_decision(
    config: ProjectConfig,
    failure_analysis: dict[str, object],
    challenger_records: list[dict[str, Any]],
    reference_total: float,
) -> Path:
    diagnostic = [
        item
        for item in failure_analysis.get("all_slices", [])
        if isinstance(item, dict)
        and str(item.get("slice_id", "")).startswith(("lead_bin=", "operating_season="))
    ]
    best_total = max(
        (float(item["score"]["pooled"]["total"]) for item in challenger_records),
        default=None,
    )
    payload = {
        "status": "DEFERRED_REQUIRES_AMENDMENT",
        "authority": "IP@v1",
        "reason": "IP@v1 excludes PyTorch, deep tiers, Colab, and GPU execution",
        "residual_structure": {
            "aligned_keys_sha256": failure_analysis.get("aligned_keys_sha256"),
            "lead_and_season_slices": diagnostic,
        },
        "runtime_evidence": {
            "challenger_cpu_seconds": float(
                sum(float(item.get("training_seconds", 0.0)) for item in challenger_records)
            ),
            "machine": platform.machine(),
            "xgboost_version": _package_version("xgboost"),
            "catboost_version": _package_version("catboost"),
            "pytorch_execution": False,
        },
        "gpu_evidence": {
            "status": "NOT_EVALUATED_UNDER_CURRENT_AUTHORITY",
            "gpu_or_colab_used": False,
            "reason": (
                "no GPU runtime inspection or mutation is needed for the approved classical tier"
            ),
        },
        "value_evidence": {
            "lightgbm_reference_total": reference_total,
            "best_challenger_total": best_total,
            "best_challenger_delta": (
                best_total - reference_total if best_total is not None else None
            ),
        },
        "amendment_required_for": [
            "PyTorch dependency installation",
            "TiDE or other deep time-series implementation",
            "GPU or Colab execution",
            "additional configuration budget",
        ],
        "external_actions": [],
        "source_sha256": config.open_zip.sha256,
    }
    path = config.repo_root / "reports" / "deep_tier_decision.json"
    write_json_atomic(path, payload)
    return path


def _backtest_challengers(config: ProjectConfig, run_id: str) -> WorkflowResult:
    features, labels, _, folds = _load_pipeline_inputs(config)
    lightgbm = _read_json(_stage_manifest_path(config, "lightgbm"))
    ablation = _read_json(_stage_manifest_path(config, "ablation"))
    stability = lightgbm.get("seed_stability")
    promoted = lightgbm.get("promoted_full_fold_candidates")
    if not isinstance(stability, dict) or len(stability) != 3 or not isinstance(promoted, list):
        raise ContractError("challenger activation requires exactly three LightGBM finalists")
    promoted_by_id = {
        str(item.get("candidate_id")): item for item in promoted if isinstance(item, dict)
    }
    finalist_predictions: dict[str, pd.DataFrame] = {}
    for candidate_id in sorted(stability):
        record = promoted_by_id.get(candidate_id)
        if record is None:
            raise ContractError(f"LightGBM finalist has no promoted OOF artifact: {candidate_id}")
        finalist_predictions[candidate_id] = _verified_prediction(config, record)
    reference = next(iter(finalist_predictions.values()))
    failure_analysis = shared_failure_slices(
        finalist_predictions,
        _failure_context(features, reference),
        dict(config.capacities),
        threshold=0.25,
    )
    selected_failure = failure_analysis.get("selected_shared_failure_slice")
    selected_mass = (
        float(selected_failure.get("minimum_error_mass_fraction", 0.0))
        if isinstance(selected_failure, dict)
        else 0.0
    )
    lightgbm_slots = int(lightgbm.get("configuration_slots_used", 0))
    total_slots = 24
    remaining_slots = max(0, total_slots - lightgbm_slots)
    lock_path = (
        config.repo_root / "artifacts" / "locks" / f"lockbox-{config.lockbox_year}.consumed.json"
    )
    activation = decide_challenger_activation(
        remaining_slots,
        selected_mass,
        lock_path.exists(),
    )
    selected = ablation.get("selected")
    if not isinstance(selected, dict) or not isinstance(selected.get("score"), dict):
        raise ContractError("challenger search requires the selected ablation OOF")
    reference_oof = _verified_prediction(config, selected)
    reference_score = selected["score"]
    reference_total = float(reference_score["pooled"]["total"])
    records: list[dict[str, Any]] = []

    if activation.accepted:
        from baram.models.challengers import expand_challenger_grid

        grids = expand_challenger_grid(config.repo_root / "configs" / "models" / "challengers.yaml")
        if sum(len(items) for items in grids.values()) != 8:
            raise ContractError("approved challenger grid must consume exactly eight slots")
        feature_names = tuple(str(item) for item in selected.get("feature_names", ()))
        if not feature_names:
            raise ContractError("selected ablation family has no feature names")
        artifact_root = config.repo_root / "artifacts" / "backtests" / "challengers" / run_id
        for family in ("xgboost", "catboost"):
            for params in grids[family]:
                candidate_id = f"{family}-shared-{canonical_sha256(params)[:16]}"
                prediction_path = artifact_root / f"{candidate_id}-oof.parquet"
                artifact_receipt_path = artifact_root / f"{candidate_id}.json"
                lineage_sha = canonical_sha256(
                    {
                        "source": config.open_zip.sha256,
                        "folds": [asdict(fold) for fold in folds],
                        "feature_names": feature_names,
                        "family": family,
                        "architecture": "shared",
                        "params": params,
                        "seed": config.seed,
                        "n_jobs": config.n_jobs,
                    }
                )
                cached = False
                training_seconds = 0.0
                if prediction_path.is_file() and artifact_receipt_path.is_file():
                    artifact_receipt = _read_json(artifact_receipt_path)
                    cached = artifact_receipt.get(
                        "lineage_sha256"
                    ) == lineage_sha and artifact_receipt.get("prediction_sha256") == sha256_file(
                        prediction_path
                    )
                    if cached:
                        prediction = pd.read_parquet(prediction_path)
                        training_seconds = float(artifact_receipt.get("training_seconds", 0.0))
                if not cached:
                    started = time.perf_counter()
                    prediction = generate_oof(
                        features,
                        labels,
                        folds,
                        feature_names,
                        family=family,  # type: ignore[arg-type]
                        architecture="shared",
                        params=params,
                        seed=config.seed,
                        n_jobs=config.n_jobs,
                    ).predictions
                    training_seconds = time.perf_counter() - started
                    prediction_sha = _write_parquet_atomic(prediction, prediction_path)
                    write_json_atomic(
                        artifact_receipt_path,
                        {
                            "candidate_id": candidate_id,
                            "lineage_sha256": lineage_sha,
                            "prediction_sha256": prediction_sha,
                            "training_seconds": training_seconds,
                        },
                    )
                prediction_sha = sha256_file(prediction_path)
                score = _score_payload(prediction, dict(config.capacities))
                reference_folds = _fold_total_scores(reference_score)
                challenger_folds = _fold_total_scores(score)
                fold_deltas = [
                    challenger_folds[fold_id] - reference_folds[fold_id]
                    for fold_id in sorted(reference_folds)
                ]
                pooled_delta = float(score["pooled"]["total"]) - reference_total
                p2 = decide_development_promotion(pooled_delta, fold_deltas, True)
                residual_correlation = _abs_residual_correlation(reference_oof, prediction)
                p3 = decide_diversity(residual_correlation, pooled_delta)
                spec = {
                    "family": family,
                    "architecture": "shared",
                    "feature_names": feature_names,
                    "params": params,
                    "seed": config.seed,
                    "n_jobs": config.n_jobs,
                }
                records.append(
                    {
                        "candidate_id": candidate_id,
                        "spec": spec,
                        "params_sha256": canonical_sha256(params),
                        "score": score,
                        "fold_deltas": fold_deltas,
                        "pooled_delta": pooled_delta,
                        "abs_residual_correlation": residual_correlation,
                        "P2": asdict(p2),
                        "P3": asdict(p3),
                        "accepted": p2.accepted and p3.accepted,
                        "prediction_path": str(prediction_path.relative_to(config.repo_root)),
                        "prediction_sha256": prediction_sha,
                        "training_seconds": training_seconds,
                        "checkpoint_reusable": True,
                    }
                )

    accepted = sorted(
        (item for item in records if item["accepted"]),
        key=lambda item: (-float(item["score"]["pooled"]["total"]), item["candidate_id"]),
    )
    deep_path = _write_deep_tier_decision(
        config,
        failure_analysis,
        records,
        reference_total,
    )
    payload = {
        "run_id": run_id,
        "stage": "challengers",
        "status": "ACTIVATED" if activation.accepted else "NOT_ACTIVATED",
        "source_sha256": config.open_zip.sha256,
        "activation": asdict(activation),
        "configuration_slots_total": total_slots,
        "configuration_slots_used_before": lightgbm_slots,
        "configuration_slots_used": len(records),
        "configuration_slots_used_cumulative": lightgbm_slots + len(records),
        "configuration_slots_remaining": remaining_slots - len(records),
        "failure_analysis": failure_analysis,
        "reference": {
            "candidate_id": f"lightgbm-{ablation.get('selected_family', 'selected')}",
            "score": reference_score,
            "prediction_path": selected.get("prediction_path"),
            "prediction_sha256": selected.get("prediction_sha256"),
        },
        "results": records,
        "accepted_challenger_ids": [item["candidate_id"] for item in accepted],
        "champion": accepted[0] if accepted else None,
        "environment": {
            "machine": platform.machine(),
            "xgboost_version": _package_version("xgboost"),
            "catboost_version": _package_version("catboost"),
            "uv_lock_sha256": sha256_file(config.repo_root / "uv.lock"),
            "worker_cap": min(config.n_jobs, 6),
            "cpu_only": True,
        },
        "lockbox_consumed": lock_path.exists(),
        "deep_tier_decision_path": str(deep_path.relative_to(config.repo_root)),
    }
    manifest_path = _stage_manifest_path(config, "challengers")
    digest = write_json_atomic(manifest_path, payload)
    report_path = config.repo_root / "reports" / "challenger_activation.json"
    write_json_atomic(report_path, payload)
    return WorkflowResult((manifest_path, report_path, deep_path), digest)


def run_backtest(args: Namespace) -> WorkflowResult:
    config = _config_from_args(args)
    run_id = _run_id(args)
    if args.stage == "controls":
        return _backtest_controls(config, run_id)
    if args.stage == "lightgbm":
        return _backtest_lightgbm(config, run_id)
    if args.stage == "ablation":
        return _backtest_ablation(config, run_id)
    if args.stage == "challengers":
        return _backtest_challengers(config, run_id)
    raise ContractError(f"unsupported backtest stage: {args.stage}")


def run_select(args: Namespace) -> WorkflowResult:
    config = _config_from_args(args)
    run_id = _run_id(args)
    _, _, _, folds = _load_pipeline_inputs(config)
    controls = _read_json(_stage_manifest_path(config, "controls"))
    ablation = _read_json(_stage_manifest_path(config, "ablation"))
    lightgbm = _read_json(_stage_manifest_path(config, "lightgbm"))
    control_name = str(controls.get("champion"))
    control_artifacts = controls.get("prediction_artifacts")
    if not isinstance(control_artifacts, dict) or control_name not in control_artifacts:
        raise ContractError("control manifest has no champion prediction artifact")
    control_info = control_artifacts[control_name]
    control_path = config.repo_root / str(control_info["path"])
    if sha256_file(control_path) != control_info.get("sha256"):
        raise ContractError("control champion prediction artifact hash mismatch")
    control_oof = pd.read_parquet(control_path)

    tree_selected = ablation.get("selected")
    model_spec = ablation.get("model")
    if not isinstance(tree_selected, dict) or not isinstance(model_spec, dict):
        raise ContractError("ablation manifest has no selected tree specification")
    tree_path = config.repo_root / str(tree_selected["prediction_path"])
    if sha256_file(tree_path) != tree_selected.get("prediction_sha256"):
        raise ContractError("selected tree prediction artifact hash mismatch")
    tree_oof = pd.read_parquet(tree_path)
    key_columns = [*_KEYS, "fold_id"]
    if not tree_oof[key_columns].equals(control_oof[key_columns]):
        raise ContractError("tree and control OOF keys/folds differ")

    metric_sha = canonical_sha256({"metric": "official-1-nmae-ficr", "version": 1})
    fold_order = tuple(fold.fold_id for fold in folds)
    calibrated_cross = cross_fit_calibration(
        tree_oof,
        fold_order,
        dict(config.capacities),
        metric_sha,
    )
    cross_folds = set(calibrated_cross["fold_id"])
    raw_cross = tree_oof.loc[tree_oof["fold_id"].isin(cross_folds)].copy()
    raw_cross = raw_cross.sort_values(_KEYS, kind="stable").reset_index(drop=True)
    calibrated_cross = calibrated_cross.sort_values(_KEYS, kind="stable").reset_index(drop=True)
    calibration_score = _score_payload(calibrated_cross, dict(config.capacities))
    raw_cross_score = _score_payload(raw_cross, dict(config.capacities))
    calibration_fold_deltas = [
        _fold_total_scores(calibration_score)[fold] - _fold_total_scores(raw_cross_score)[fold]
        for fold in sorted(cross_folds)
    ]
    calibration_decision = decide_development_promotion(
        float(calibration_score["pooled"]["total"]) - float(raw_cross_score["pooled"]["total"]),
        calibration_fold_deltas,
        True,
    )
    final_calibrations = [
        fit_group_calibration(
            tree_oof.loc[tree_oof["group_id"].eq(group_id)],
            group_id,  # type: ignore[arg-type]
            capacity,
            sha256_dataframe(tree_oof.loc[tree_oof["group_id"].eq(group_id), [*_KEYS, "fold_id"]]),
            metric_sha,
        )
        for group_id, capacity in config.capacities.items()
    ]

    labels = tree_oof[[*_KEYS, "actual_kwh", "fold_id"]]
    parent_frames = {
        "control": control_oof[[*_KEYS, "prediction_kwh", "fold_id"]],
        "tree": tree_oof[[*_KEYS, "prediction_kwh", "fold_id"]],
    }
    blended_parts: list[pd.DataFrame] = []
    for position, validation_fold in enumerate(fold_order[1:], start=1):
        training_folds = fold_order[:position]
        train_mask = labels["fold_id"].isin(training_folds)
        valid_mask = labels["fold_id"].eq(validation_fold)
        train_predictions = {
            name: frame.loc[train_mask, [*_KEYS, "prediction_kwh"]].reset_index(drop=True)
            for name, frame in parent_frames.items()
        }
        train_labels = labels.loc[train_mask, [*_KEYS, "actual_kwh"]].reset_index(drop=True)
        blend_policy = fit_two_model_blend(
            train_predictions,
            train_labels,
            dict(config.capacities),
            sha256_dataframe(train_labels[_KEYS]),
            metric_sha,
        )
        valid_predictions = {
            name: frame.loc[valid_mask, [*_KEYS, "prediction_kwh"]].reset_index(drop=True)
            for name, frame in parent_frames.items()
        }
        part = apply_blend(blend_policy, valid_predictions)
        part = part.merge(
            labels.loc[valid_mask, [*_KEYS, "actual_kwh"]],
            on=_KEYS,
            how="left",
            validate="one_to_one",
        )
        part["fold_id"] = validation_fold
        part["model_id"] = "crossfit-control-tree-blend"
        blended_parts.append(part)
    blended_cross = pd.concat(blended_parts, ignore_index=True)
    blend_score = _score_payload(blended_cross, dict(config.capacities))
    blend_fold_deltas = [
        _fold_total_scores(blend_score)[fold] - _fold_total_scores(raw_cross_score)[fold]
        for fold in sorted(cross_folds)
    ]
    blend_decision = decide_development_promotion(
        float(blend_score["pooled"]["total"]) - float(raw_cross_score["pooled"]["total"]),
        blend_fold_deltas,
        True,
    )
    final_blend = fit_two_model_blend(
        {
            name: frame[[*_KEYS, "prediction_kwh"]].reset_index(drop=True)
            for name, frame in parent_frames.items()
        },
        labels[[*_KEYS, "actual_kwh"]].reset_index(drop=True),
        dict(config.capacities),
        sha256_dataframe(labels[_KEYS]),
        metric_sha,
    )

    control_spec: dict[str, Any] = {
        "family": control_name,
        "architecture": "group_specific",
        "feature_names": tuple(controls.get("feature_names", ())),
        "params": {},
        "seed": config.seed,
        "n_jobs": _RF_DETERMINISTIC_JOBS if control_name == "random_forest" else config.n_jobs,
    }
    tree_spec: dict[str, Any] = {
        "family": "lightgbm",
        "architecture": model_spec["architecture"],
        "feature_names": tuple(tree_selected["feature_names"]),
        "params": model_spec["params"],
        "seed": int(model_spec["seed"]),
    }

    def frozen_candidate(candidate_id: str, spec: dict[str, Any]) -> dict[str, Any]:
        return {
            "candidate_id": candidate_id,
            "spec": spec,
            "policy_sha256": canonical_sha256(spec),
        }

    control_candidate = frozen_candidate(f"control-{control_name}", control_spec)
    raw_tree_candidate = frozen_candidate("tree-raw", {"kind": "base", "base": tree_spec})
    decision_options: list[tuple[float, dict[str, Any]]] = []
    if calibration_decision.accepted:
        spec = {
            "kind": "calibrated",
            "base": tree_spec,
            "policies": [asdict(policy) for policy in final_calibrations],
        }
        decision_options.append(
            (
                float(calibration_score["pooled"]["total"]),
                frozen_candidate("tree-calibrated", spec),
            )
        )
    if blend_decision.accepted:
        spec = {
            "kind": "blend",
            "parents": {"control": control_spec, "tree": tree_spec},
            "policy": asdict(final_blend),
        }
        decision_options.append(
            (float(blend_score["pooled"]["total"]), frozen_candidate("tree-control-blend", spec))
        )

    challenger_manifest_path = _stage_manifest_path(config, "challengers")
    challenger_manifest: dict[str, Any] | None = None
    challenger_candidate: dict[str, Any] | None = None
    if challenger_manifest_path.is_file():
        challenger_manifest = _read_json(challenger_manifest_path)
        if challenger_manifest.get("source_sha256") != config.open_zip.sha256:
            raise ContractError("challenger manifest source differs from the configured source")
        champion = challenger_manifest.get("champion")
        if challenger_manifest.get("status") == "ACTIVATED" and champion is not None:
            if not isinstance(champion, dict) or not champion.get("accepted"):
                raise ContractError("challenger manifest champion is not an accepted candidate")
            champion_spec = champion.get("spec")
            if not isinstance(champion_spec, dict):
                raise ContractError("challenger champion has no frozen specification")
            challenger_candidate = frozen_candidate(
                str(champion["candidate_id"]),
                {"kind": "base", "base": champion_spec},
            )

    best_decision_candidate = (
        max(decision_options, key=lambda item: (item[0], item[1]["candidate_id"]))[1]
        if decision_options
        else raw_tree_candidate
    )
    candidates = [control_candidate]
    if challenger_candidate is None:
        candidates.append(raw_tree_candidate)
        if decision_options:
            candidates.append(best_decision_candidate)
    else:
        candidates.extend([best_decision_candidate, challenger_candidate])
    if len(candidates) > 3:
        raise ContractError("candidate freeze may contain at most three candidates")

    split_payload = _read_json(_split_manifest_path(config))
    lineages = {
        "source": config.open_zip.sha256,
        "splits": canonical_sha256(split_payload),
        "controls": canonical_sha256(controls),
        "lightgbm": canonical_sha256(lightgbm),
        "ablation": canonical_sha256(ablation),
    }
    if challenger_manifest is not None:
        lineages["challengers"] = canonical_sha256(challenger_manifest)
    challenger_slots = (
        int(challenger_manifest.get("configuration_slots_used", 0))
        if challenger_manifest is not None
        else 0
    )
    configuration_slots_used = int(lightgbm.get("configuration_slots_used", 0)) + challenger_slots
    freeze_id = canonical_sha256({"candidates": candidates, "lineages": lineages, "run_id": run_id})
    payload = {
        "freeze_id": freeze_id,
        "run_id": run_id,
        "source_sha256": config.open_zip.sha256,
        "lineage_hashes": lineages,
        "configuration_slots_total": 24,
        "configuration_slots_used": configuration_slots_used,
        "configuration_slots_remaining": 24 - configuration_slots_used,
        "finalist_seed_runs": sum(
            len(item.get("seed_scores", {})) for item in lightgbm.get("seed_stability", {}).values()
        ),
        "candidate_policy_hashes": [item["policy_sha256"] for item in candidates],
        "candidates": candidates,
        "decision_evidence": {
            "raw_crossfit_score": raw_cross_score,
            "calibration": {
                "decision": asdict(calibration_decision),
                "score": calibration_score,
            },
            "blend": {"decision": asdict(blend_decision), "score": blend_score},
            "challenger": {
                "status": (
                    challenger_manifest.get("status")
                    if challenger_manifest is not None
                    else "NOT_EVALUATED"
                ),
                "champion_candidate_id": (
                    challenger_candidate["candidate_id"]
                    if challenger_candidate is not None
                    else None
                ),
                "manifest_sha256": (
                    canonical_sha256(challenger_manifest)
                    if challenger_manifest is not None
                    else None
                ),
            },
            "utility": {"status": "NOT_ACTIVATED", "reason": "no quantile distribution parent"},
        },
        "lockbox_consumed": False,
    }
    freeze_path = config.repo_root / "artifacts" / "manifests" / "candidate_freeze.json"
    digest = write_json_atomic(freeze_path, payload)
    report_path = config.repo_root / "reports" / "candidate_freeze_receipt.json"
    write_json_atomic(report_path, payload)
    decision_path = config.repo_root / "reports" / "decision_layer.json"
    write_json_atomic(decision_path, payload["decision_evidence"])
    return WorkflowResult((freeze_path, report_path, decision_path), digest)


def _base_fold_prediction(
    spec: dict[str, Any],
    features: pd.DataFrame,
    labels: pd.DataFrame,
    fold: FoldSpec,
    config: ProjectConfig,
) -> pd.DataFrame:
    family = str(spec.get("family"))
    if family in {"climatology", "physics"}:
        return _manual_control_oof(features, labels, (fold,), family)
    if family not in {"random_forest", "lightgbm", "xgboost", "catboost"}:
        raise ContractError(f"unsupported frozen base family: {family}")
    architecture = str(spec.get("architecture", "group_specific"))
    if architecture not in {"group_specific", "shared"}:
        raise ContractError(f"unsupported frozen architecture: {architecture}")
    names = tuple(str(item) for item in spec.get("feature_names", ()))
    if not names:
        raise ContractError("frozen tree base has no feature names")
    params = spec.get("params", {})
    if not isinstance(params, dict):
        raise ContractError("frozen tree params must be an object")
    return generate_oof(
        features,
        labels,
        (fold,),
        names,
        family=family,  # type: ignore[arg-type]
        architecture=architecture,  # type: ignore[arg-type]
        params=params,
        seed=int(spec.get("seed", config.seed)),
        n_jobs=int(spec.get("n_jobs", config.n_jobs)),
    ).predictions


def _calibration_from_payload(raw: dict[str, Any]) -> CalibrationPolicy:
    return CalibrationPolicy(
        group_id=int(raw["group_id"]),  # type: ignore[arg-type]
        scale=float(raw["scale"]),
        offset_capacity=float(raw["offset_capacity"]),
        cap_mode=str(raw["cap_mode"]),  # type: ignore[arg-type]
        training_rows_sha256=str(raw["training_rows_sha256"]),
        parent_model_ids=tuple(str(item) for item in raw["parent_model_ids"]),
        input_prediction_sha256=str(raw["input_prediction_sha256"]),
        metric_sha256=str(raw["metric_sha256"]),
    )


def _blend_from_payload(raw: dict[str, Any]) -> BlendPolicy:
    return BlendPolicy(
        weights_by_group={
            int(group): {str(parent): float(weight) for parent, weight in weights.items()}
            for group, weights in raw["weights_by_group"].items()
        },  # type: ignore[arg-type]
        training_rows_sha256=str(raw["training_rows_sha256"]),
        input_prediction_hashes={
            str(parent): str(value) for parent, value in raw["input_prediction_hashes"].items()
        },
        metric_sha256=str(raw["metric_sha256"]),
    )


def _candidate_fold_prediction(
    spec: dict[str, Any],
    features: pd.DataFrame,
    labels: pd.DataFrame,
    fold: FoldSpec,
    config: ProjectConfig,
    base_cache: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    kind = str(spec.get("kind", "base"))
    if kind == "base":
        base_spec = spec.get("base", spec)
        if not isinstance(base_spec, dict):
            raise ContractError("frozen base candidate is invalid")
        cache_key = canonical_sha256(base_spec)
        if cache_key not in base_cache:
            base_cache[cache_key] = _base_fold_prediction(base_spec, features, labels, fold, config)
        return base_cache[cache_key].copy()
    if kind == "calibrated":
        base_spec = spec.get("base")
        policies = spec.get("policies")
        if not isinstance(base_spec, dict) or not isinstance(policies, list):
            raise ContractError("frozen calibrated candidate is invalid")
        result = _candidate_fold_prediction(
            {"kind": "base", "base": base_spec},
            features,
            labels,
            fold,
            config,
            base_cache,
        )
        for raw_policy in policies:
            policy = _calibration_from_payload(raw_policy)
            mask = result["group_id"].eq(policy.group_id)
            result.loc[mask, "prediction_kwh"] = apply_calibration(
                result.loc[mask, "prediction_kwh"].to_numpy(dtype=float),
                policy.group_id,
                config.capacities[policy.group_id],
                policy,
            )
        result["model_id"] = "frozen-calibrated"
        return result
    if kind == "blend":
        parents = spec.get("parents")
        policy_raw = spec.get("policy")
        if not isinstance(parents, dict) or not isinstance(policy_raw, dict):
            raise ContractError("frozen blend candidate is invalid")
        parent_outputs = {
            str(parent): _candidate_fold_prediction(
                {"kind": "base", "base": parent_spec},
                features,
                labels,
                fold,
                config,
                base_cache,
            )
            for parent, parent_spec in parents.items()
        }
        blended = apply_blend(
            _blend_from_payload(policy_raw),
            {parent: frame[[*_KEYS, "prediction_kwh"]] for parent, frame in parent_outputs.items()},
        )
        truth = next(iter(parent_outputs.values()))[[*_KEYS, "actual_kwh", "fold_id"]]
        result = blended.merge(truth, on=_KEYS, how="left", validate="one_to_one")
        result["model_id"] = "frozen-blend"
        return result
    raise ContractError(f"unsupported frozen candidate kind: {kind}")


def _acquire_one_use_lock(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise ContractError("the 2024 lockbox has already been consumed") from error
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return sha256_file(path)


def run_lockbox(args: Namespace) -> WorkflowResult:
    config = _config_from_args(args)
    run_id = _run_id(args)
    freeze_path = Path(args.candidate_freeze)
    freeze = _read_json(freeze_path)
    if freeze.get("source_sha256") != config.open_zip.sha256:
        raise ContractError("candidate freeze source differs from the configured source")
    candidates = freeze.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ContractError("candidate freeze has no candidates")
    freeze_file_sha = sha256_file(freeze_path)
    lock_path = (
        config.repo_root / "artifacts" / "locks" / f"lockbox-{config.lockbox_year}.consumed.json"
    )
    consumed_lock_sha = _acquire_one_use_lock(
        lock_path,
        {
            "candidate_freeze_file_sha256": freeze_file_sha,
            "freeze_id": freeze.get("freeze_id"),
            "run_id": run_id,
            "lockbox_year": config.lockbox_year,
            "state": "CONSUMED_BEFORE_SCORING",
        },
    )

    features, labels, _, _ = _load_pipeline_inputs(config)
    splits = _read_json(_split_manifest_path(config))
    raw_lockbox = splits.get("lockbox_unconsumed_spec")
    if not isinstance(raw_lockbox, dict):
        raise ContractError("split manifest has no lockbox specification")
    fold = _fold_from_payload(raw_lockbox)
    base_cache: dict[str, pd.DataFrame] = {}
    scores: dict[str, dict[str, Any]] = {}
    prediction_artifacts: dict[str, dict[str, str]] = {}
    artifact_root = config.repo_root / "artifacts" / "backtests" / "lockbox" / run_id
    candidate_by_id: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        if not isinstance(candidate, dict) or not isinstance(candidate.get("spec"), dict):
            raise ContractError("candidate freeze contains an invalid candidate")
        candidate_id = str(candidate["candidate_id"])
        candidate_by_id[candidate_id] = candidate
        prediction = _candidate_fold_prediction(
            candidate["spec"], features, labels, fold, config, base_cache
        )
        score = evaluate_official(
            prediction[[*_KEYS, "actual_kwh", "prediction_kwh"]],
            dict(config.capacities),
        )
        scores[candidate_id] = asdict(score)
        path = artifact_root / f"{candidate_id}.parquet"
        prediction_artifacts[candidate_id] = {
            "path": str(path.relative_to(config.repo_root)),
            "sha256": _write_parquet_atomic(prediction, path),
        }
    control_id = next(
        (
            str(item["candidate_id"])
            for item in candidates
            if str(item["candidate_id"]).startswith("control-")
        ),
        None,
    )
    if control_id is None:
        raise ContractError("candidate freeze has no frozen control")
    control_total = float(scores[control_id]["total"])
    accepted = [
        candidate_id
        for candidate_id, score in scores.items()
        if candidate_id != control_id
        and decide_lockbox(float(score["total"]), control_total).accepted
    ]
    champion_id = (
        max(accepted, key=lambda item: (float(scores[item]["total"]), item))
        if accepted
        else control_id
    )
    champion = candidate_by_id[champion_id]
    metric_sha = canonical_sha256({"metric": "official-1-nmae-ficr", "version": 1})
    payload = {
        "run_id": run_id,
        "candidate_freeze_file_sha256": freeze_file_sha,
        "candidate_freeze_sha256": canonical_sha256(freeze),
        "consumed_lock_sha256": consumed_lock_sha,
        "source_sha256": config.open_zip.sha256,
        "scores": scores,
        "control_candidate_id": control_id,
        "promotion": {
            candidate_id: asdict(decide_lockbox(float(score["total"]), control_total))
            for candidate_id, score in scores.items()
            if candidate_id != control_id
        },
        "champion_candidate_id": champion_id,
        "champion": champion,
        "champion_policy_sha256": champion["policy_sha256"],
        "metric_sha256": metric_sha,
        "prediction_artifacts": prediction_artifacts,
        "post_lockbox_tuning": False,
    }
    receipt_path = config.repo_root / "reports" / "lockbox_receipt.json"
    digest = write_json_atomic(receipt_path, payload)
    return WorkflowResult((lock_path, receipt_path), digest)


def _fit_full_base(
    spec: dict[str, Any],
    features: pd.DataFrame,
    labels: pd.DataFrame,
    config: ProjectConfig,
) -> dict[str, Any]:
    family = str(spec.get("family"))
    merged = features.merge(
        labels[[*_KEYS, "actual_kwh"]],
        on=_KEYS,
        how="left",
        validate="one_to_one",
        sort=False,
    )
    observed = merged.dropna(subset=["actual_kwh"]).reset_index(drop=True)
    if family == "climatology":
        state = fit_climatology(observed, "final-full")
        return {
            "kind": "base",
            "family": family,
            "spec": spec,
            "models": state,
            "lineage": {"training_rows_sha256": state.training_rows_sha256},
        }
    if family == "physics":
        proxies = {
            group_id: fit_physics_proxy(
                observed.loc[observed["group_id"].eq(group_id), "phys__rho_v3"],
                observed.loc[observed["group_id"].eq(group_id), "actual_kwh"],
                capacity,
            )
            for group_id, capacity in config.capacities.items()
        }
        return {
            "kind": "base",
            "family": family,
            "spec": spec,
            "models": proxies,
            "lineage": {"training_rows_sha256": sha256_dataframe(observed[[*_KEYS, "actual_kwh"]])},
        }
    if family not in {"random_forest", "lightgbm", "xgboost", "catboost"}:
        raise ContractError(f"unsupported final base family: {family}")
    architecture = str(spec.get("architecture", "group_specific"))
    names = tuple(str(item) for item in spec.get("feature_names", ()))
    params = spec.get("params", {})
    if not names or not isinstance(params, dict):
        raise ContractError("final tree base has an invalid feature/parameter contract")
    seed = int(spec.get("seed", config.seed))
    n_jobs = int(spec.get("n_jobs", config.n_jobs))
    if architecture == "group_specific":
        bundles = {}
        for group_id, capacity in config.capacities.items():
            group = observed.loc[observed["group_id"].eq(group_id)].reset_index(drop=True)
            if family == "random_forest":
                bundle = fit_supplied_rf_bundle(
                    group[list(names)],
                    group["actual_kwh"],
                    names,
                    "final-full",
                    group_id,  # type: ignore[arg-type]
                    capacity,
                    seed,
                    n_jobs,
                )
            elif family == "lightgbm":
                bundle = fit_lgbm_bundle(
                    group[list(names)],
                    group["actual_kwh"],
                    group["issuance_batch"],
                    names,
                    "final-full",
                    group_id,  # type: ignore[arg-type]
                    capacity,
                    params,
                    seed,
                    n_jobs,
                )
            else:
                from baram.models.challengers import fit_challenger_bundle

                bundle = fit_challenger_bundle(
                    family,  # type: ignore[arg-type]
                    group[list(names)],
                    group["actual_kwh"],
                    group["issuance_batch"],
                    names,
                    "final-full",
                    group_id,  # type: ignore[arg-type]
                    capacity,
                    params,
                    seed,
                    n_jobs,
                )
            bundles[group_id] = bundle
        lineage = {str(group): asdict(bundle.manifest) for group, bundle in bundles.items()}
        return {
            "kind": "base",
            "family": family,
            "architecture": architecture,
            "spec": spec,
            "models": bundles,
            "lineage": lineage,
        }
    if architecture == "shared" and family in {"lightgbm", "xgboost", "catboost"}:
        shared_names = (*names, "group_id", "capacity_kwh")
        normalized = observed["actual_kwh"] / observed["capacity_kwh"]
        if family == "lightgbm":
            bundle = fit_lgbm_bundle(
                observed[list(shared_names)],
                normalized,
                observed["issuance_batch"],
                shared_names,
                "final-full",
                None,
                1.0,
                params,
                seed,
                n_jobs,
            )
        else:
            from baram.models.challengers import fit_challenger_bundle

            bundle = fit_challenger_bundle(
                family,  # type: ignore[arg-type]
                observed[list(shared_names)],
                normalized,
                observed["issuance_batch"],
                shared_names,
                "final-full",
                None,
                1.0,
                params,
                seed,
                n_jobs,
            )
        return {
            "kind": "base",
            "family": family,
            "architecture": architecture,
            "spec": spec,
            "models": bundle,
            "lineage": {"shared": asdict(bundle.manifest)},
        }
    raise ContractError(f"unsupported final family/architecture: {family}/{architecture}")


def _fit_full_candidate(
    spec: dict[str, Any],
    features: pd.DataFrame,
    labels: pd.DataFrame,
    config: ProjectConfig,
) -> dict[str, Any]:
    kind = str(spec.get("kind", "base"))
    if kind == "base":
        base_spec = spec.get("base", spec)
        if not isinstance(base_spec, dict):
            raise ContractError("final base candidate is invalid")
        return _fit_full_base(base_spec, features, labels, config)
    if kind == "calibrated":
        base_spec = spec.get("base")
        if not isinstance(base_spec, dict) or not isinstance(spec.get("policies"), list):
            raise ContractError("final calibrated candidate is invalid")
        base = _fit_full_base(base_spec, features, labels, config)
        return {
            "kind": kind,
            "base": base,
            "policies": spec["policies"],
            "lineage": {
                "base": base["lineage"],
                "decision_policy_sha256": canonical_sha256(spec["policies"]),
            },
        }
    if kind == "blend":
        parents = spec.get("parents")
        if not isinstance(parents, dict) or not isinstance(spec.get("policy"), dict):
            raise ContractError("final blend candidate is invalid")
        fitted_parents = {
            str(name): _fit_full_base(parent, features, labels, config)
            for name, parent in parents.items()
        }
        return {
            "kind": kind,
            "parents": fitted_parents,
            "policy": spec["policy"],
            "lineage": {
                "parents": {name: parent["lineage"] for name, parent in fitted_parents.items()},
                "decision_policy_sha256": canonical_sha256(spec["policy"]),
            },
        }
    raise ContractError(f"unsupported final candidate kind: {kind}")


def _predict_full_base(model: dict[str, Any], test_features: pd.DataFrame) -> pd.DataFrame:
    family = str(model["family"])
    result = test_features[_KEYS].copy()
    result["prediction_kwh"] = np.nan
    if family == "climatology":
        transformed = apply_climatology(model["models"], test_features, "final-full")
        result["prediction_kwh"] = transformed["clim_median"].to_numpy(dtype=float)
    elif family == "physics":
        for group_id, proxy in model["models"].items():
            mask = test_features["group_id"].eq(group_id)
            result.loc[mask, "prediction_kwh"] = predict_physics_proxy(
                proxy, test_features.loc[mask, "phys__rho_v3"]
            )
    elif model.get("architecture") == "shared":
        bundle = model["models"]
        names = bundle.feature_names
        normalized = predict_bundle(bundle, test_features[list(names)], "final-full")
        result["prediction_kwh"] = normalized * test_features["capacity_kwh"].to_numpy(dtype=float)
    else:
        for group_id, bundle in model["models"].items():
            mask = test_features["group_id"].eq(group_id)
            result.loc[mask, "prediction_kwh"] = predict_bundle(
                bundle,
                test_features.loc[mask, list(bundle.feature_names)].reset_index(drop=True),
                "final-full",
            )
    if (
        result["prediction_kwh"].isna().any()
        or not np.isfinite(result["prediction_kwh"].to_numpy(dtype=float)).all()
    ):
        raise ContractError("final base left non-finite predictions")
    result["model_id"] = f"final-{family}"
    return result


def _predict_full_candidate(model: dict[str, Any], test_features: pd.DataFrame) -> pd.DataFrame:
    kind = str(model.get("kind", "base"))
    if kind == "base":
        return _predict_full_base(model, test_features)
    if kind == "calibrated":
        result = _predict_full_base(model["base"], test_features)
        for raw in model["policies"]:
            policy = _calibration_from_payload(raw)
            mask = result["group_id"].eq(policy.group_id)
            result.loc[mask, "prediction_kwh"] = apply_calibration(
                result.loc[mask, "prediction_kwh"].to_numpy(dtype=float),
                policy.group_id,
                CAPACITIES_KWH[policy.group_id],
                policy,
            )
        result["model_id"] = "final-calibrated"
        return result
    if kind == "blend":
        parents = {
            name: _predict_full_base(parent, test_features)
            for name, parent in model["parents"].items()
        }
        result = apply_blend(
            _blend_from_payload(model["policy"]),
            {name: frame[[*_KEYS, "prediction_kwh"]] for name, frame in parents.items()},
        )
        result["model_id"] = "final-blend"
        return result
    raise ContractError(f"unsupported final prediction kind: {kind}")


def _dump_joblib_atomic(value: Any, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(value, temporary, compress=3)
    temporary.replace(path)
    return sha256_file(path)


def run_fit_final(args: Namespace) -> WorkflowResult:
    config = _config_from_args(args)
    run_id = _run_id(args)
    champion_receipt_path = Path(args.champion_receipt)
    champion_receipt = _read_json(champion_receipt_path)
    if champion_receipt.get("source_sha256") != config.open_zip.sha256:
        raise ContractError("lockbox champion source differs from configuration")
    champion = champion_receipt.get("champion")
    if not isinstance(champion, dict) or not isinstance(champion.get("spec"), dict):
        raise ContractError("lockbox receipt has no frozen champion specification")
    features, labels, _, _ = _load_pipeline_inputs(config)
    fitted = _fit_full_candidate(champion["spec"], features, labels, config)
    model_path = config.repo_root / "artifacts" / "models" / f"{run_id}.joblib"
    artifact_hash = _dump_joblib_atomic(fitted, model_path)
    lineage_hash = canonical_sha256(fitted["lineage"])
    payload = {
        "run_id": run_id,
        "source_sha256": config.open_zip.sha256,
        "champion_receipt_path": str(champion_receipt_path),
        "champion_receipt_sha256": sha256_file(champion_receipt_path),
        "champion_candidate_id": champion_receipt["champion_candidate_id"],
        "champion_policy_sha256": champion_receipt["champion_policy_sha256"],
        "champion": champion,
        "model_artifact_path": str(model_path.relative_to(config.repo_root)),
        "model_artifact_sha256": artifact_hash,
        "model_lineage_sha256": lineage_hash,
        "seed": config.seed,
    }
    receipt_path = config.repo_root / "reports" / "final-model.receipt.json"
    digest = write_json_atomic(receipt_path, payload)
    return WorkflowResult((receipt_path,), digest)


def _load_final_test_inputs(
    config: ProjectConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prepare = _read_json(_prepare_manifest_path(config))
    expected = prepare.get("artifact_hashes")
    if not isinstance(expected, dict):
        raise ContractError("prepare manifest has no final-input hashes")
    cache = _cache_root(config)
    test_path = cache / "test_features.parquet"
    sample_path = cache / "submission_keys.parquet"
    for name, path in (("test_features", test_path), ("submission_keys", sample_path)):
        if sha256_file(path) != expected.get(name):
            raise ContractError(f"final input artifact hash mismatch: {name}")
    return pd.read_parquet(test_path), pd.read_parquet(sample_path)


def _wide_predictions(long: pd.DataFrame) -> pd.DataFrame:
    if long.duplicated(_KEYS).any():
        raise ContractError("final long predictions contain duplicate keys")
    wide = long.pivot(
        index=["forecast_id", "forecast_kst_dtm"],
        columns="group_id",
        values="prediction_kwh",
    ).reset_index()
    wide.columns.name = None
    wide = wide.rename(columns={group: f"kpx_group_{group}" for group in (1, 2, 3)})
    required = ["forecast_id", "forecast_kst_dtm", "kpx_group_1", "kpx_group_2", "kpx_group_3"]
    if not set(required).issubset(wide.columns):
        raise ContractError("final predictions do not cover all three groups")
    return wide[required]


def _cap_modes_for_model(model: dict[str, Any]) -> dict[int, str]:
    if model.get("kind") == "calibrated":
        return {int(raw["group_id"]): str(raw["cap_mode"]) for raw in model.get("policies", [])}
    return {1: "nonnegative_only", 2: "nonnegative_only", 3: "nonnegative_only"}


def run_build_submission(args: Namespace) -> WorkflowResult:
    config = _config_from_args(args)
    run_id = _run_id(args)
    model_receipt_path = Path(args.model_receipt)
    model_receipt = _read_json(model_receipt_path)
    if model_receipt.get("source_sha256") != config.open_zip.sha256:
        raise ContractError("final model source differs from configuration")
    model_path = config.repo_root / str(model_receipt["model_artifact_path"])
    if sha256_file(model_path) != model_receipt.get("model_artifact_sha256"):
        raise ContractError("final model artifact hash mismatch")
    fitted = joblib.load(model_path)
    if canonical_sha256(fitted["lineage"]) != model_receipt.get("model_lineage_sha256"):
        raise ContractError("final model lineage hash mismatch")
    test_features, sample = _load_final_test_inputs(config)
    long = _predict_full_candidate(fitted, test_features)
    wide = _wide_predictions(long)
    candidate_path = config.repo_root / "artifacts" / "submissions" / "final_candidate.csv"
    csv_hash = build_submission(sample, wide, candidate_path)
    candidate_id = (
        f"{model_receipt['champion_candidate_id']}-"
        f"{str(model_receipt['champion_policy_sha256'])[:12]}"
    )
    submission_receipt = validate_submission(
        candidate_path,
        sample,
        candidate_id=candidate_id,
        source_sha256=config.open_zip.sha256,
        champion_policy_sha256=str(model_receipt["champion_policy_sha256"]),
        cap_modes=_cap_modes_for_model(fitted),
    )
    if submission_receipt.csv_sha256 != csv_hash:
        raise ContractError("builder and validator submission hashes differ")
    payload = {
        "run_id": run_id,
        "candidate_path": str(candidate_path.relative_to(config.repo_root)),
        "submission_receipt": asdict(submission_receipt),
        "model_receipt_path": str(model_receipt_path),
        "model_receipt_sha256": sha256_file(model_receipt_path),
        "champion_receipt_path": model_receipt["champion_receipt_path"],
        "model_lineage_sha256": model_receipt["model_lineage_sha256"],
        "no_external_upload": True,
    }
    receipt_path = config.repo_root / "artifacts" / "submissions" / "final_candidate.receipt.json"
    digest = write_json_atomic(receipt_path, payload)
    return WorkflowResult((candidate_path, receipt_path), digest)


def run_reproduce(args: Namespace) -> WorkflowResult:
    config = _config_from_args(args)
    candidate_receipt_path = Path(args.candidate_receipt)
    candidate_receipt = _read_json(candidate_receipt_path)
    submission_receipt = candidate_receipt.get("submission_receipt")
    if not isinstance(submission_receipt, dict):
        raise ContractError("candidate receipt has no submission receipt")
    candidate_path = config.repo_root / str(candidate_receipt["candidate_path"])
    if sha256_file(candidate_path) != submission_receipt.get("csv_sha256"):
        raise ContractError("recorded final candidate bytes do not match their receipt")
    model_receipt_path = Path(str(candidate_receipt["model_receipt_path"]))
    model_receipt = _read_json(model_receipt_path)
    champion = model_receipt.get("champion")
    if not isinstance(champion, dict) or not isinstance(champion.get("spec"), dict):
        raise ContractError("model receipt has no reproducible champion")
    features, labels, _, _ = _load_pipeline_inputs(config)
    reproduced_model = _fit_full_candidate(champion["spec"], features, labels, config)
    reproduced_lineage = canonical_sha256(reproduced_model["lineage"])
    test_features, sample = _load_final_test_inputs(config)
    reproduced_long = _predict_full_candidate(reproduced_model, test_features)
    reproduced_wide = _wide_predictions(reproduced_long)
    reproduced_path = (
        config.repo_root
        / "artifacts"
        / "reproduction"
        / f"{submission_receipt['candidate_id']}.csv"
    )
    reproduced_csv_hash = build_submission(sample, reproduced_wide, reproduced_path)
    expected = {
        "model_lineage_sha256": str(candidate_receipt["model_lineage_sha256"]),
        "submission_sha256": str(submission_receipt["csv_sha256"]),
    }
    actual = {
        "model_lineage_sha256": reproduced_lineage,
        "submission_sha256": reproduced_csv_hash,
    }
    decision = decide_reproduction(expected, actual)
    payload = {
        "candidate_receipt_sha256": sha256_file(candidate_receipt_path),
        "expected_hashes": expected,
        "actual_hashes": actual,
        "decision": asdict(decision),
        "fresh_process_required_by_operator_command": True,
    }
    receipt_path = config.repo_root / "reports" / "reproduction_receipt.json"
    digest = write_json_atomic(receipt_path, payload)
    if not decision.accepted:
        raise ContractError(f"final reproduction hash mismatch: {decision.reasons}")
    return WorkflowResult((receipt_path,), digest)


def _tiny_frames() -> tuple[pd.DataFrame, pd.DataFrame, FoldSpec]:
    feature_rows: list[dict[str, object]] = []
    label_rows: list[dict[str, object]] = []
    batches = tuple(f"batch-{index}" for index in range(8))
    for batch_index, batch in enumerate(batches):
        for slot in range(3):
            timestamp = pd.Timestamp("2023-01-01 01:00") + pd.Timedelta(
                batch_index * 3 + slot, unit="h"
            )
            forecast_id = f"tiny-{batch_index}-{slot}"
            for group_id, capacity in CAPACITIES_KWH.items():
                x_value = float(batch_index + slot / 3.0 + group_id / 10.0)
                feature_rows.append(
                    {
                        "forecast_id": forecast_id,
                        "forecast_kst_dtm": timestamp,
                        "issuance_batch": batch,
                        "group_id": group_id,
                        "capacity_kwh": capacity,
                        "x": x_value,
                        "z": float(group_id),
                    }
                )
                label_rows.append(
                    {
                        "forecast_id": forecast_id,
                        "forecast_kst_dtm": timestamp,
                        "group_id": group_id,
                        "actual_kwh": capacity * (0.2 + 0.01 * x_value),
                    }
                )
    fold = FoldSpec(
        "fixture-dev",
        batches[:5],
        batches[5:],
        (1, 2, 3),
        True,
    )
    return pd.DataFrame(feature_rows), pd.DataFrame(label_rows), fold


def run_tiny_fixture_pipeline(root: Path) -> WorkflowResult:
    """Exercise the complete local contract in a disposable, non-lockbox fixture."""
    features, labels, fold = _tiny_frames()
    manifest_sha = canonical_sha256(
        {
            "feature_sha256": sha256_dataframe(features),
            "label_sha256": sha256_dataframe(labels),
            "fold": asdict(fold),
        }
    )
    oof = generate_oof(
        features,
        labels,
        (fold,),
        ("x", "z"),
        family="random_forest",
        architecture="group_specific",
        params={},
        seed=42,
        n_jobs=1,
    ).predictions
    score = evaluate_official(oof[[*_KEYS, "actual_kwh", "prediction_kwh"]], CAPACITIES_KWH)
    policies = tuple(
        fit_group_calibration(
            oof.loc[oof["group_id"].eq(group_id)],
            group_id,  # type: ignore[arg-type]
            capacity,
            sha256_dataframe(oof.loc[oof["group_id"].eq(group_id), _KEYS]),
            canonical_sha256({"metric": "official-v1"}),
        )
        for group_id, capacity in CAPACITIES_KWH.items()
    )
    policy_sha = canonical_sha256(policies)
    fixture_lock_path = root / "artifacts" / "fixture-locks" / "disposable-lockbox.json"
    write_json_atomic(
        fixture_lock_path,
        {"manifest_sha256": manifest_sha, "policy_sha256": policy_sha, "score": asdict(score)},
    )

    dummy_source = root / "fixture-source.bin"
    dummy_source.parent.mkdir(parents=True, exist_ok=True)
    dummy_source.write_bytes(b"baram-tiny-fixture-v1")
    dummy_sha = sha256_file(dummy_source)
    fixture_config = ProjectConfig(
        repo_root=root,
        open_zip=SourceSpec(dummy_source, dummy_sha),
        baseline_notebook=SourceSpec(dummy_source, dummy_sha),
        capacities=CAPACITIES_KWH,  # type: ignore[arg-type]
        seed=42,
        n_jobs=1,
        artifact_budget_gib=1,
        lockbox_year=2099,
    )
    fitted = _fit_full_candidate(
        {
            "kind": "base",
            "base": {
                "family": "random_forest",
                "architecture": "group_specific",
                "feature_names": ("x", "z"),
                "params": {},
                "seed": 42,
            },
        },
        features,
        labels,
        fixture_config,
    )
    sample = pd.DataFrame(
        {
            "forecast_id": [f"forecast_{index:04d}" for index in range(1, 8761)],
            "forecast_kst_dtm": pd.date_range("2025-01-01 01:00", periods=8760, freq="h"),
        }
    )
    test_rows: list[pd.DataFrame] = []
    test_x = np.arange(8760, dtype=float) % 24 / 24.0
    for group_id, capacity in CAPACITIES_KWH.items():
        test_rows.append(
            sample.assign(
                group_id=group_id,
                capacity_kwh=capacity,
                issuance_batch="fixture-test",
                x=test_x + group_id / 10.0,
                z=float(group_id),
            )
        )
    test_features = (
        pd.concat(test_rows, ignore_index=True)
        .sort_values(["forecast_kst_dtm", "group_id"], kind="stable")
        .reset_index(drop=True)
    )
    wide = _wide_predictions(_predict_full_candidate(fitted, test_features))
    candidate = root / "artifacts" / "submissions" / "tiny-candidate.csv"
    submission_sha = build_submission(sample, wide, candidate)
    validate_submission(
        candidate,
        sample,
        candidate_id="tiny-candidate",
        source_sha256=manifest_sha,
        champion_policy_sha256=policy_sha,
        cap_modes={1: "nonnegative_only", 2: "nonnegative_only", 3: "nonnegative_only"},
    )
    payload = {
        "manifest_sha256": manifest_sha,
        "policy_sha256": policy_sha,
        "submission_sha256": submission_sha,
    }
    receipt = root / "reports" / "tiny-reproduction.json"
    digest = write_json_atomic(receipt, payload)
    return WorkflowResult((receipt,), digest)
