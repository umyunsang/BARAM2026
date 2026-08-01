"""Genuine issuance-batch out-of-fold prediction engine."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from baram.contracts.types import FoldSpec
from baram.exceptions import ModelError
from baram.models.baselines import fit_supplied_rf_bundle, predict_bundle
from baram.models.lightgbm import fit_lgbm_bundle


@dataclass(frozen=True)
class OOFResult:
    predictions: pd.DataFrame
    training_forecast_ids: Mapping[str, frozenset[str]]


def filter_complete_validation_rows(
    frame: pd.DataFrame,
    eligible_groups: tuple[int, ...],
) -> pd.DataFrame:
    """Keep only timestamps with one observed label for every eligible group."""
    required = {"forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh"}
    if not required.issubset(frame.columns):
        raise ModelError(
            f"validation completeness requires: {sorted(required - set(frame.columns))}"
        )
    eligible = frame.loc[frame["group_id"].isin(eligible_groups)].copy()
    summary = eligible.groupby(["forecast_id", "forecast_kst_dtm"], sort=False).agg(
        row_count=("group_id", "size"),
        group_count=("group_id", "nunique"),
        observed_count=("actual_kwh", "count"),
    )
    complete = summary.loc[
        summary["row_count"].eq(len(eligible_groups))
        & summary["group_count"].eq(len(eligible_groups))
        & summary["observed_count"].eq(len(eligible_groups))
    ].index
    result = eligible.set_index(["forecast_id", "forecast_kst_dtm"]).loc[
        lambda indexed: indexed.index.isin(complete)
    ]
    result = result.reset_index()
    if result.empty:
        raise ModelError("validation fold has no complete all-group label timestamps")
    return result


def _fit_bundle(
    family: str,
    features: pd.DataFrame,
    target: pd.Series,
    batches: pd.Series,
    feature_names: tuple[str, ...],
    fold_id: str,
    group_id: int | None,
    capacity: float,
    params: dict[str, object],
    seed: int,
    n_jobs: int,
):
    if family == "random_forest":
        if group_id not in (1, 2, 3):
            raise ModelError("RandomForest OOF requires a concrete group")
        return fit_supplied_rf_bundle(
            features,
            target,
            feature_names,
            fold_id,
            group_id,
            capacity,
            seed,
            n_jobs,
        )
    if family == "lightgbm":
        return fit_lgbm_bundle(
            features,
            target,
            batches,
            feature_names,
            fold_id,
            group_id,
            capacity,
            params,
            seed,
            n_jobs,
        )
    if family in {"xgboost", "catboost"}:
        from baram.models.challengers import fit_challenger_bundle

        return fit_challenger_bundle(
            family,  # type: ignore[arg-type]
            features,
            target,
            batches,
            feature_names,
            fold_id,
            group_id,  # type: ignore[arg-type]
            capacity,
            params,
            seed,
            n_jobs,
        )
    raise ModelError(f"unsupported OOF model family: {family}")


def generate_oof(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    folds: tuple[FoldSpec, ...],
    feature_names: tuple[str, ...],
    family: Literal["random_forest", "lightgbm", "xgboost", "catboost"],
    architecture: Literal["group_specific", "shared"],
    params: dict[str, object],
    seed: int,
    n_jobs: int,
) -> OOFResult:
    keys = ["forecast_id", "forecast_kst_dtm", "group_id"]
    required_features = {*keys, "issuance_batch", "capacity_kwh", *feature_names}
    if not required_features.issubset(features.columns):
        raise ModelError(f"OOF features are missing: {sorted(required_features - set(features))}")
    if not set([*keys, "actual_kwh"]).issubset(labels.columns):
        raise ModelError("OOF labels are missing key or target columns")
    merged = features.merge(
        labels[[*keys, "actual_kwh"]],
        on=keys,
        how="left",
        validate="one_to_one",
        sort=False,
    )
    predictions: list[pd.DataFrame] = []
    training_ids: dict[str, frozenset[str]] = {}
    for fold in folds:
        train = merged.loc[merged["issuance_batch"].isin(fold.train_batches)].copy()
        valid = merged.loc[merged["issuance_batch"].isin(fold.validation_batches)].copy()
        if train.empty or valid.empty:
            raise ModelError(f"OOF fold has empty train or validation rows: {fold.fold_id}")
        training_ids[fold.fold_id] = frozenset(train["forecast_id"].unique())
        if set(train["forecast_id"]) & set(valid["forecast_id"]):
            raise ModelError("OOF validation key appears in its training set")
        valid = filter_complete_validation_rows(valid, fold.eligible_groups)

        if architecture == "group_specific":
            for group in fold.eligible_groups:
                group_train = train.loc[train["group_id"].eq(group)].dropna(subset=["actual_kwh"])
                group_valid = valid.loc[valid["group_id"].eq(group)]
                capacity = float(group_train["capacity_kwh"].iloc[0])
                bundle = _fit_bundle(
                    family,
                    group_train[list(feature_names)].reset_index(drop=True),
                    group_train["actual_kwh"].reset_index(drop=True),
                    group_train["issuance_batch"].reset_index(drop=True),
                    feature_names,
                    fold.fold_id,
                    group,
                    capacity,
                    params,
                    seed,
                    n_jobs,
                )
                part = group_valid[[*keys, "actual_kwh"]].copy()
                part["prediction_kwh"] = predict_bundle(
                    bundle,
                    group_valid[list(feature_names)].reset_index(drop=True),
                    fold.fold_id,
                )
                part["fold_id"] = fold.fold_id
                part["model_id"] = bundle.manifest.model_id
                predictions.append(part)
        elif architecture == "shared":
            shared_train = train.loc[train["group_id"].isin(fold.eligible_groups)].dropna(
                subset=["actual_kwh"]
            )
            shared_features = (*feature_names, "group_id", "capacity_kwh")
            normalized_target = shared_train["actual_kwh"] / shared_train["capacity_kwh"]
            bundle = _fit_bundle(
                family,
                shared_train[list(shared_features)].reset_index(drop=True),
                normalized_target.reset_index(drop=True),
                shared_train["issuance_batch"].reset_index(drop=True),
                shared_features,
                fold.fold_id,
                None,
                1.0,
                params,
                seed,
                n_jobs,
            )
            part = valid[[*keys, "actual_kwh"]].copy()
            normalized_prediction = predict_bundle(
                bundle,
                valid[list(shared_features)].reset_index(drop=True),
                fold.fold_id,
            )
            part["prediction_kwh"] = np.maximum(
                normalized_prediction * valid["capacity_kwh"].to_numpy(dtype=float), 0.0
            )
            part["fold_id"] = fold.fold_id
            part["model_id"] = bundle.manifest.model_id
            predictions.append(part)
        else:
            raise ModelError(f"unsupported OOF architecture: {architecture}")
    combined = pd.concat(predictions, ignore_index=True).sort_values(
        ["forecast_kst_dtm", "group_id", "model_id"], kind="stable"
    )
    if combined.duplicated(["forecast_id", "group_id", "model_id"]).any():
        raise ModelError("OOF output contains duplicate model/key rows")
    if not np.isfinite(combined["prediction_kwh"]).all():
        raise ModelError("OOF output contains non-finite predictions")
    return OOFResult(
        predictions=combined.reset_index(drop=True),
        training_forecast_ids=training_ids,
    )
