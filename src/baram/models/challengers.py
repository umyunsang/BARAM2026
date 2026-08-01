"""Bounded deterministic XGBoost/CatBoost diversity challengers."""

from itertools import product
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import yaml
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from baram.contracts.hashing import canonical_sha256
from baram.contracts.types import GroupId
from baram.exceptions import ContractError, ModelError
from baram.features.pipeline import fit_feature_pipeline, transform_features
from baram.models.baselines import ModelBundle, _model_manifest

ChallengerFamily = Literal["xgboost", "catboost"]


def expand_challenger_grid(path: Path) -> dict[ChallengerFamily, list[dict[str, object]]]:
    """Expand the approved four-config grid for each challenger family."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ContractError(f"cannot read challenger search space: {error}") from error
    if not isinstance(raw, dict) or set(raw) != {"xgboost", "catboost"}:
        raise ContractError("challenger search space must contain only xgboost and catboost")
    result: dict[ChallengerFamily, list[dict[str, object]]] = {}
    for raw_family in ("xgboost", "catboost"):
        family: ChallengerFamily = raw_family  # type: ignore[assignment]
        try:
            fixed = dict(raw[family]["fixed"])
            grid = dict(raw[family]["grid"])
        except (TypeError, KeyError, ValueError) as error:
            raise ContractError(f"invalid {family} search space: {error}") from error
        names = list(grid)
        configs = [
            {**fixed, **dict(zip(names, values, strict=True))}
            for values in product(*(grid[name] for name in names))
        ]
        if len(configs) != 4 or len({canonical_sha256(item) for item in configs}) != 4:
            raise ContractError(f"{family} grid must contain exactly four unique configs")
        result[family] = configs
    return result


def _worker_count(n_jobs: int) -> int:
    if n_jobs < 1:
        raise ModelError("challenger n_jobs must be positive")
    return min(n_jobs, 6)


def make_xgb(params: dict[str, object], seed: int, n_jobs: int) -> XGBRegressor:
    """Create a CPU-only, worker-capped XGBoost regressor."""
    return XGBRegressor(
        **params,
        random_state=seed,
        n_jobs=_worker_count(n_jobs),
        device="cpu",
        verbosity=0,
    )


def make_catboost(params: dict[str, object], seed: int, n_jobs: int) -> CatBoostRegressor:
    """Create a CPU-only CatBoost regressor that writes no training directory."""
    return CatBoostRegressor(
        **params,
        random_seed=seed,
        thread_count=_worker_count(n_jobs),
        task_type="CPU",
    )


def split_inner_stopping_batches(batches: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Reserve the most recent whole 20% of issuance batches for early stopping."""
    ordered = list(dict.fromkeys(batches.astype(str)))
    if len(ordered) < 2:
        raise ModelError("challenger inner stopping requires at least two issuance batches")
    stop_count = max(1, int(np.ceil(len(ordered) * 0.2)))
    stop_batches = set(ordered[-stop_count:])
    stop_mask = batches.astype(str).isin(stop_batches)
    fit_mask = ~stop_mask
    if not fit_mask.any() or not stop_mask.any():
        raise ModelError("challenger inner stopping produced an empty partition")
    return fit_mask, stop_mask


def _selected_iteration(family: ChallengerFamily, estimator: object) -> int:
    if family == "xgboost":
        value = getattr(estimator, "best_iteration", None)
    else:
        getter = getattr(estimator, "get_best_iteration", None)
        value = getter() if callable(getter) else None
    if value is None or int(value) < 0:
        raise ModelError(f"{family} did not expose a valid early-stopping iteration")
    return int(value) + 1


def fit_challenger_bundle(
    family: ChallengerFamily,
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
    """Select iterations on inner batches, then refit on the full outer-training fold."""
    if family not in {"xgboost", "catboost"}:
        raise ModelError(f"unsupported challenger family: {family}")
    if len(features) != len(target) or len(features) != len(issuance_batches):
        raise ModelError("challenger training arrays are not aligned")
    if target.isna().any() or not np.isfinite(target).all() or capacity <= 0.0:
        raise ModelError("challenger target/capacity contract is invalid")
    fit_mask, stop_mask = split_inner_stopping_batches(issuance_batches)
    inner_fold = f"{fold_id}-inner"
    inner_state = fit_feature_pipeline(
        features.loc[fit_mask].reset_index(drop=True), feature_names, inner_fold
    )
    x_fit = transform_features(
        inner_state, features.loc[fit_mask].reset_index(drop=True), inner_fold
    )
    x_stop = transform_features(
        inner_state, features.loc[stop_mask].reset_index(drop=True), inner_fold
    )
    normalized = target.reset_index(drop=True).to_numpy(dtype=float) / capacity
    fit_positions = np.flatnonzero(fit_mask.to_numpy())
    stop_positions = np.flatnonzero(stop_mask.to_numpy())
    if family == "xgboost":
        stop_model = make_xgb(params, seed, n_jobs)
        stop_model.fit(
            x_fit,
            normalized[fit_positions],
            eval_set=[(x_stop, normalized[stop_positions])],
            verbose=False,
        )
    else:
        stop_model = make_catboost(params, seed, n_jobs)
        stop_model.fit(
            x_fit,
            normalized[fit_positions],
            eval_set=(x_stop, normalized[stop_positions]),
            use_best_model=True,
        )
    selected = _selected_iteration(family, stop_model)
    refit_params = dict(params)
    refit_params.pop("early_stopping_rounds", None)
    if family == "xgboost":
        refit_params["n_estimators"] = selected
    else:
        refit_params["iterations"] = selected
    final_state = fit_feature_pipeline(features.reset_index(drop=True), feature_names, fold_id)
    x_final = transform_features(final_state, features.reset_index(drop=True), fold_id)
    if family == "xgboost":
        final_model = make_xgb(refit_params, seed, n_jobs)
    else:
        final_model = make_catboost(refit_params, seed, n_jobs)
    final_model.fit(x_final, normalized)
    manifest_params = {**refit_params, "selected_iteration": selected}
    return ModelBundle(
        estimator=final_model,
        manifest=_model_manifest(family, fold_id, group_id, final_state, manifest_params, seed),
        feature_names=feature_names,
        feature_state=final_state,
        capacity=capacity,
        group_id=group_id,
        target_is_normalized=True,
        cap_mode="nonnegative_only",
    )
