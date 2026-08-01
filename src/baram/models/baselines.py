"""Deterministic climatology, physics, and supplied RandomForest controls."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.isotonic import IsotonicRegression

from baram.contracts.hashing import canonical_sha256
from baram.contracts.types import GroupId, ModelManifest
from baram.exceptions import ModelError
from baram.features.pipeline import (
    FeaturePipelineState,
    fit_feature_pipeline,
    transform_features,
)


@dataclass
class ModelBundle:
    estimator: object
    manifest: ModelManifest
    feature_names: tuple[str, ...]
    feature_state: FeaturePipelineState
    capacity: float
    group_id: GroupId | None
    target_is_normalized: bool
    cap_mode: Literal["capacity", "1.01_capacity", "nonnegative_only"]


def make_supplied_rf(seed: int, n_jobs: int) -> RandomForestRegressor:
    if n_jobs < 1 or n_jobs > 6:
        raise ModelError("RandomForest n_jobs must be between 1 and 6")
    return RandomForestRegressor(
        n_estimators=120,
        max_depth=14,
        min_samples_leaf=8,
        max_features="sqrt",
        random_state=seed,
        n_jobs=n_jobs,
    )


def _model_manifest(
    family: str,
    fold_id: str,
    group_id: GroupId | None,
    feature_state: FeaturePipelineState,
    params: dict[str, object],
    seed: int,
) -> ModelManifest:
    params_sha = canonical_sha256(params)
    identity = canonical_sha256(
        {"family": family, "fold_id": fold_id, "group_id": group_id, "params": params, "seed": seed}
    )
    return ModelManifest(
        model_id=f"{family}-{identity[:16]}",
        family=family,
        fold_id=fold_id,
        feature_manifest_sha256=canonical_sha256(feature_state),
        training_rows_sha256=feature_state.training_rows_sha256,
        params_sha256=params_sha,
        seed=seed,
    )


def fit_supplied_rf_bundle(
    features: pd.DataFrame,
    target: pd.Series,
    feature_names: tuple[str, ...],
    fold_id: str,
    group_id: GroupId,
    capacity: float,
    seed: int,
    n_jobs: int,
) -> ModelBundle:
    if len(features) != len(target) or target.isna().any() or not np.isfinite(target).all():
        raise ModelError("RandomForest training target must be aligned, finite, and observed")
    if capacity <= 0.0:
        raise ModelError("model capacity must be positive")
    state = fit_feature_pipeline(features, feature_names, fold_id)
    transformed = transform_features(state, features, fold_id)
    estimator = make_supplied_rf(seed, n_jobs)
    estimator.fit(transformed, target.to_numpy(dtype=float) / capacity)
    params = {
        "n_estimators": 120,
        "max_depth": 14,
        "min_samples_leaf": 8,
        "max_features": "sqrt",
        "n_jobs": n_jobs,
    }
    return ModelBundle(
        estimator=estimator,
        manifest=_model_manifest("supplied_rf", fold_id, group_id, state, params, seed),
        feature_names=feature_names,
        feature_state=state,
        capacity=capacity,
        group_id=group_id,
        target_is_normalized=True,
        cap_mode="nonnegative_only",
    )


def predict_bundle(bundle: ModelBundle, features: pd.DataFrame, fold_id: str) -> np.ndarray:
    transformed = transform_features(bundle.feature_state, features, fold_id)
    estimator = bundle.estimator
    if not hasattr(estimator, "predict"):
        raise ModelError("model bundle estimator has no predict method")
    prediction = np.asarray(estimator.predict(transformed), dtype=float)
    if bundle.target_is_normalized:
        prediction = prediction * bundle.capacity
    prediction = np.maximum(prediction, 0.0)
    if bundle.cap_mode == "capacity":
        prediction = np.minimum(prediction, bundle.capacity)
    elif bundle.cap_mode == "1.01_capacity":
        prediction = np.minimum(prediction, 1.01 * bundle.capacity)
    if not np.isfinite(prediction).all():
        raise ModelError("model predictions contain non-finite values")
    return prediction


@dataclass
class PhysicsProxy:
    estimator: IsotonicRegression
    capacity: float


def fit_physics_proxy(proxy: pd.Series, target: pd.Series, capacity: float) -> PhysicsProxy:
    valid = proxy.notna() & target.notna() & np.isfinite(proxy) & np.isfinite(target)
    if valid.sum() < 2:
        raise ModelError("physics proxy requires at least two observed rows")
    estimator = IsotonicRegression(y_min=0.0, y_max=1.01, out_of_bounds="clip")
    estimator.fit(proxy.loc[valid].to_numpy(dtype=float), target.loc[valid].to_numpy() / capacity)
    return PhysicsProxy(estimator=estimator, capacity=capacity)


def predict_physics_proxy(model: PhysicsProxy, proxy: pd.Series) -> np.ndarray:
    filled = proxy.fillna(float(proxy.median())).to_numpy(dtype=float)
    return np.maximum(model.estimator.predict(filled) * model.capacity, 0.0)
