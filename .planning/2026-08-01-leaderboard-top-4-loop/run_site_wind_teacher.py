"""Diagnose NWP-to-site-wind reconstruction on frozen 2023 folds.

This development runner never exposes SCADA observations to a power forecast.
Observed validation SCADA is used only to measure the weather-reconstruction
teacher.  Candidate power models must consume predictions written by this
runner, not the observed values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import lightgbm
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from run_sequence_classifier import (
    BASELINE,
    BASELINE_SHA,
    OPEN,
    OPEN_SHA,
    OUTPUT,
    _scada_wind,
    _sequence_columns,
    _sha256,
    _surface,
)

FOLDS = ("dev-2023-Q2", "dev-2023-Q3", "dev-2023-Q4")


@dataclass(frozen=True)
class Recipe:
    name: str
    profile: str
    objective: str
    per_group: bool
    num_leaves: int
    sequence: bool = False
    exact_legacy: bool = False


RECIPES = {
    recipe.name: recipe
    for recipe in (
        Recipe("legacy_shared_l2", "legacy", "l2", False, 31, exact_legacy=True),
        Recipe("legacy_group_l2", "legacy", "l2", True, 31),
        Recipe("windgeom_group_l2", "windgeom", "l2", True, 63),
        Recipe("allweather_group_l2", "allweather", "l2", True, 63),
        Recipe("allweather_group_l1", "allweather", "l1", True, 63),
        Recipe("allweather_group_huber", "allweather", "huber", True, 63),
        Recipe("windgeom_seq_group_l2", "windgeom", "l2", True, 63, sequence=True),
        Recipe("allweather_seq_group_l2", "allweather", "l2", True, 63, sequence=True),
    )
}


def _hash_frame(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validation_mask(surface: pd.DataFrame, fold_id: str) -> np.ndarray:
    reference = pd.read_parquet(OUTPUT / "M50_GEOM_CLASS_TEMP075_G05-oof.parquet")
    reference = reference.loc[reference["fold_id"].eq(fold_id)]
    keys = pd.MultiIndex.from_frame(reference[["forecast_id", "group_id"]])
    surface_keys = pd.MultiIndex.from_frame(surface[["forecast_id", "group_id"]])
    mask = surface_keys.isin(keys)
    if int(mask.sum()) != len(reference):
        raise RuntimeError("validation-key contract changed")
    return np.asarray(mask)


def _strict_preceding_mask(
    surface: pd.DataFrame,
    validation: np.ndarray,
) -> np.ndarray:
    """Select only issuance batches fully observable before validation issuance."""
    cutoff = pd.Timestamp(
        surface.loc[validation, "data_available_kst_dtm"].min()
    )
    batch_last_forecast = surface.groupby(
        "data_available_kst_dtm", sort=False
    )["forecast_kst_dtm"].transform("max")
    preceding = batch_last_forecast.lt(cutoff).to_numpy()
    batch_membership = pd.DataFrame(
        {
            "data_available_kst_dtm": surface["data_available_kst_dtm"],
            "preceding": preceding,
        }
    ).groupby("data_available_kst_dtm", sort=False)["preceding"].nunique()
    if not batch_membership.eq(1).all():
        raise RuntimeError("strict training mask split an issuance batch")
    if preceding.any() and not surface.loc[
        preceding, "forecast_kst_dtm"
    ].lt(cutoff).all():
        raise RuntimeError("strict training mask includes an unobservable label")
    return preceding


def _all_weather_columns(surface: pd.DataFrame) -> list[str]:
    excluded = {
        "forecast_id",
        "forecast_kst_dtm",
        "data_available_kst_dtm",
        "issuance_batch",
        "actual_kwh",
        "scada_ws",
    }
    return [
        name
        for name in surface
        if name not in excluded and pd.api.types.is_numeric_dtype(surface[name])
    ]


def _feature_columns(
    surface: pd.DataFrame,
    base_columns: list[str],
    auxiliary_columns: list[str],
    profile: str,
) -> list[str]:
    if profile == "legacy":
        return auxiliary_columns
    if profile == "windgeom":
        return base_columns
    if profile == "allweather":
        return _all_weather_columns(surface)
    raise ValueError(f"unknown profile: {profile}")


def _add_sequence(surface: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    from baram.features.sequence import add_issuance_sequence_context

    inputs = _sequence_columns(surface)
    context = add_issuance_sequence_context(
        surface[["forecast_kst_dtm", "data_available_kst_dtm", "group_id", *inputs]],
        inputs,
    )
    columns = [name for name in context if name.startswith("seq__")]
    return pd.concat([surface, context[columns]], axis=1), columns


def _metrics(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    finite = np.isfinite(actual) & np.isfinite(prediction)
    actual = actual[finite]
    prediction = prediction[finite]
    error = prediction - actual
    return {
        "count": len(actual),
        "correlation": float(np.corrcoef(actual, prediction)[0, 1]),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "bias": float(np.mean(error)),
        "p90_absolute_error": float(np.quantile(np.abs(error), 0.90)),
    }


def _params(recipe: Recipe, n_estimators: int) -> dict[str, object]:
    if recipe.exact_legacy:
        return {
            "objective": "l2",
            "n_estimators": 400,
            "learning_rate": 0.04,
            "num_leaves": 31,
            "min_child_samples": 60,
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
    return {
        "objective": recipe.objective,
        "n_estimators": n_estimators,
        "learning_rate": 0.025,
        "num_leaves": recipe.num_leaves,
        "min_child_samples": 40,
        "max_bin": 255,
        "subsample": 0.9,
        "subsample_freq": 1,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.1,
        "reg_lambda": 3.0,
        "random_state": 20260802,
        "n_jobs": 6,
        "deterministic": True,
        "force_col_wise": True,
        "verbosity": -1,
    }


def _fit_predict(
    surface: pd.DataFrame,
    feature_columns: list[str],
    preceding: np.ndarray,
    validation: np.ndarray,
    recipe: Recipe,
) -> tuple[np.ndarray, dict[str, int]]:
    matrix = surface[feature_columns].astype("float32")
    prediction = np.full(len(surface), np.nan, dtype="float32")
    selected_iterations: dict[str, int] = {}
    group_ids: tuple[int | None, ...] = (1, 2, 3) if recipe.per_group else (None,)
    for group_id in group_ids:
        group_mask = (
            np.ones(len(surface), dtype=bool)
            if group_id is None
            else surface["group_id"].eq(group_id).to_numpy()
        )
        train_mask = preceding & group_mask & surface["scada_ws"].notna().to_numpy()
        valid_mask = validation & group_mask
        if recipe.exact_legacy:
            iterations = 400
        else:
            batches = (
                surface.loc[train_mask, "data_available_kst_dtm"]
                .drop_duplicates()
                .sort_values()
            )
            cutoff = batches.iloc[int(len(batches) * 0.80)]
            stop_mask = train_mask & surface["data_available_kst_dtm"].ge(cutoff).to_numpy()
            fit_mask = train_mask & ~stop_mask
            probe = LGBMRegressor(**_params(recipe, 1800))
            probe.fit(
                matrix.loc[fit_mask],
                surface.loc[fit_mask, "scada_ws"],
                eval_set=[(matrix.loc[stop_mask], surface.loc[stop_mask, "scada_ws"])],
                eval_metric="l1",
                callbacks=[lightgbm.early_stopping(100, verbose=False)],
            )
            iterations = max(1, int(probe.best_iteration_ or 1800))
        model = LGBMRegressor(**_params(recipe, iterations))
        model.fit(matrix.loc[train_mask], surface.loc[train_mask, "scada_ws"])
        prediction[valid_mask] = model.predict(matrix.loc[valid_mask])
        selected_iterations[str(group_id or "shared")] = iterations
    return prediction, selected_iterations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--fold", choices=FOLDS, default=FOLDS[-1])
    parser.add_argument("--recipes", nargs="+", choices=tuple(RECIPES), required=True)
    args = parser.parse_args()
    if _sha256(OPEN) != OPEN_SHA or _sha256(BASELINE) != BASELINE_SHA:
        raise RuntimeError("immutable competition input hash mismatch")

    started = time.perf_counter()
    surface, base_columns, auxiliary_columns = _surface()
    if not surface["scada_ws"].equals(
        surface.drop(columns="scada_ws")
        .merge(
            _scada_wind(),
            on=["forecast_kst_dtm", "group_id"],
            how="left",
            validate="one_to_one",
        )["scada_ws"]
    ):
        raise RuntimeError("SCADA wind reconstruction is not deterministic")
    validation = _validation_mask(surface, args.fold)
    preceding = _strict_preceding_mask(surface, validation)

    outputs = surface.loc[
        validation,
        ["forecast_id", "forecast_kst_dtm", "group_id", "scada_ws"],
    ].copy()
    receipts: dict[str, object] = {}
    sequence_columns: list[str] = []
    for name in args.recipes:
        recipe = RECIPES[name]
        recipe_started = time.perf_counter()
        if recipe.sequence and not sequence_columns:
            surface, sequence_columns = _add_sequence(surface)
        feature_columns = _feature_columns(
            surface, base_columns, auxiliary_columns, recipe.profile
        )
        if recipe.sequence:
            feature_columns = list(dict.fromkeys([*feature_columns, *sequence_columns]))
        prediction, iterations = _fit_predict(
            surface, feature_columns, preceding, validation, recipe
        )
        output_prediction = prediction[validation]
        outputs[name] = output_prediction
        metrics = {
            "pooled": _metrics(outputs["scada_ws"].to_numpy(), output_prediction),
            "groups": {
                str(group_id): _metrics(
                    outputs.loc[outputs["group_id"].eq(group_id), "scada_ws"].to_numpy(),
                    outputs.loc[outputs["group_id"].eq(group_id), name].to_numpy(),
                )
                for group_id in (1, 2, 3)
            },
        }
        receipts[name] = {
            "recipe": recipe.__dict__,
            "feature_count": len(feature_columns),
            "selected_iterations": iterations,
            "metrics": metrics,
            "runtime_seconds": round(time.perf_counter() - recipe_started, 2),
        }
        print(json.dumps({"recipe": name, **receipts[name]}), flush=True)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT / f"{args.candidate_id}-{args.fold}-site-wind.parquet"
    outputs.to_parquet(output_path, index=False)
    receipt = {
        "schema_version": 1,
        "candidate_id": args.candidate_id,
        "fold_id": args.fold,
        "scope": "2023 development-fold site-wind diagnostic only",
        "recipes": receipts,
        "runtime_seconds": round(time.perf_counter() - started, 2),
        "prediction_path": str(output_path.relative_to(Path.cwd())),
        "prediction_sha256": _hash_frame(output_path),
        "observed_validation_scada_used_for_power_prediction": False,
        "new_2024_evaluation": False,
        "lockbox_reopened": False,
        "external_actions": [],
    }
    receipt_path = OUTPUT / f"{args.candidate_id}-{args.fold}-site-wind.json"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
