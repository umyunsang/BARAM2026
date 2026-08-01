"""Deterministic capacity-normalized LightGBM fitting with inner stopping."""

from itertools import product
from pathlib import Path

import lightgbm
import numpy as np
import pandas as pd
import yaml
from lightgbm import LGBMRegressor

from baram.contracts.types import GroupId
from baram.exceptions import ContractError, ModelError
from baram.features.pipeline import fit_feature_pipeline, transform_features
from baram.models.baselines import ModelBundle, _model_manifest


def expand_lgbm_grid(path: Path) -> list[dict[str, object]]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        fixed = dict(raw["fixed"])
        grid = dict(raw["grid"])
    except (OSError, TypeError, KeyError, yaml.YAMLError) as error:
        raise ContractError(f"cannot read LightGBM search space: {error}") from error
    names = list(grid)
    configs = [
        {**fixed, **dict(zip(names, values, strict=True))}
        for values in product(*(grid[name] for name in names))
    ]
    if len(configs) != 16:
        raise ContractError(f"LightGBM grid must contain exactly 16 configs, got {len(configs)}")
    return configs


def make_lgbm(params: dict[str, object], seed: int, n_jobs: int) -> LGBMRegressor:
    if n_jobs < 1:
        raise ModelError("LightGBM n_jobs must be positive")
    workers = min(n_jobs, 6)
    return LGBMRegressor(
        objective=str(params.get("objective", "l1")),
        n_estimators=int(params["n_estimators"]),
        learning_rate=float(params["learning_rate"]),
        num_leaves=int(params["num_leaves"]),
        min_child_samples=int(params["min_child_samples"]),
        subsample=float(params["subsample"]),
        colsample_bytree=float(params["colsample_bytree"]),
        reg_alpha=float(params["reg_alpha"]),
        reg_lambda=float(params["reg_lambda"]),
        random_state=seed,
        n_jobs=workers,
        deterministic=True,
        force_col_wise=True,
        subsample_freq=int(params.get("subsample_freq", 1)),
        verbosity=-1,
    )


def fit_lgbm_bundle(
    features: pd.DataFrame,
    target: pd.Series,
    issuance_batches: pd.Series,
    feature_names: tuple[str, ...],
    fold_id: str,
    group_id: GroupId | None,
    capacity: float,
    params: dict[str, object],
    seed: int,
    n_jobs: int,
) -> ModelBundle:
    if len(features) != len(target) or len(features) != len(issuance_batches):
        raise ModelError("LightGBM training arrays are not aligned")
    if target.isna().any() or not np.isfinite(target).all() or capacity <= 0.0:
        raise ModelError("LightGBM target/capacity contract is invalid")
    ordered_batches = list(dict.fromkeys(issuance_batches.astype(str)))
    if len(ordered_batches) < 2:
        raise ModelError("LightGBM inner stopping requires at least two issuance batches")
    stop_count = max(1, int(np.ceil(len(ordered_batches) * 0.2)))
    stop_batches = set(ordered_batches[-stop_count:])
    inner_fit_mask = ~issuance_batches.astype(str).isin(stop_batches)
    inner_stop_mask = ~inner_fit_mask
    inner_state = fit_feature_pipeline(
        features.loc[inner_fit_mask].reset_index(drop=True), feature_names, f"{fold_id}-inner"
    )
    x_inner_fit = transform_features(
        inner_state,
        features.loc[inner_fit_mask].reset_index(drop=True),
        f"{fold_id}-inner",
    )
    x_inner_stop = transform_features(
        inner_state,
        features.loc[inner_stop_mask].reset_index(drop=True),
        f"{fold_id}-inner",
    )
    y_normalized = target.reset_index(drop=True).to_numpy(dtype=float) / capacity
    inner_fit_positions = np.flatnonzero(inner_fit_mask.to_numpy())
    inner_stop_positions = np.flatnonzero(inner_stop_mask.to_numpy())
    stop_model = make_lgbm(params, seed, n_jobs)
    stop_model.fit(
        x_inner_fit,
        y_normalized[inner_fit_positions],
        eval_X=x_inner_stop,
        eval_y=y_normalized[inner_stop_positions],
        callbacks=[
            lightgbm.early_stopping(int(params.get("early_stopping_rounds", 100)), verbose=False)
        ],
    )
    best_iteration = max(1, int(stop_model.best_iteration_ or params["n_estimators"]))
    refit_params = {**params, "n_estimators": best_iteration}
    final_state = fit_feature_pipeline(features.reset_index(drop=True), feature_names, fold_id)
    x_final = transform_features(final_state, features.reset_index(drop=True), fold_id)
    final_model = make_lgbm(refit_params, seed, n_jobs)
    final_model.fit(x_final, y_normalized)
    manifest_params = {**refit_params, "selected_iteration": best_iteration}
    return ModelBundle(
        estimator=final_model,
        manifest=_model_manifest("lightgbm", fold_id, group_id, final_state, manifest_params, seed),
        feature_names=feature_names,
        feature_state=final_state,
        capacity=capacity,
        group_id=group_id,
        target_is_normalized=True,
        cap_mode="nonnegative_only",
    )
