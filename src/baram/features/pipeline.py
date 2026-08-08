"""Fold-fitted feature imputation and immutable manifest lineage."""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from baram.contracts.hashing import canonical_sha256, sha256_dataframe
from baram.exceptions import LeakageError, ModelError


@dataclass(frozen=True)
class FeaturePipelineState:
    fold_id: str
    feature_names: tuple[str, ...]
    medians: Mapping[str, float]
    training_rows_sha256: str
    spatial_mode: str = "global_only"
    variable_allowlist_sha256: str = ""
    grid_weight_sha256: str = ""


def fit_feature_pipeline(
    train: pd.DataFrame,
    feature_names: tuple[str, ...],
    fold_id: str,
    *,
    spatial_mode: str = "global_only",
    variable_allowlist_sha256: str | None = None,
    grid_weight_sha256: str | None = None,
) -> FeaturePipelineState:
    if tuple(train.columns) != feature_names:
        raise ModelError("training feature columns/order do not match the declared contract")
    if not all(pd.api.types.is_numeric_dtype(train[name]) for name in feature_names):
        raise ModelError("all model features must be numeric")
    medians = {
        name: float(train[name].median()) if train[name].notna().any() else 0.0
        for name in feature_names
    }
    return FeaturePipelineState(
        fold_id=fold_id,
        feature_names=feature_names,
        medians=medians,
        training_rows_sha256=sha256_dataframe(train),
        spatial_mode=spatial_mode,
        variable_allowlist_sha256=(
            variable_allowlist_sha256 or canonical_sha256(("global_only",))
        ),
        grid_weight_sha256=grid_weight_sha256 or canonical_sha256(("global_only",)),
    )


def transform_features(
    state: FeaturePipelineState,
    frame: pd.DataFrame,
    fold_id: str,
) -> pd.DataFrame:
    if state.fold_id != fold_id:
        raise LeakageError("feature state belongs to a different fold")
    if tuple(frame.columns) != state.feature_names:
        raise ModelError("feature columns/order differ from fitted state")
    result = frame.copy()
    for name in state.feature_names:
        result[name] = result[name].fillna(state.medians[name])
    if not np.isfinite(result.to_numpy(dtype=float)).all():
        raise ModelError("transformed features contain non-finite values")
    return result
