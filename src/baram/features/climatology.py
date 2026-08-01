"""Fold-safe target climatology with deterministic backoff."""

from dataclasses import dataclass

import pandas as pd

from baram.contracts.hashing import sha256_dataframe
from baram.exceptions import DataQualityError, LeakageError


@dataclass(frozen=True)
class ClimatologyState:
    fold_id: str
    training_rows_sha256: str
    by_group_month_hour: pd.DataFrame
    by_group_month: pd.DataFrame
    by_group: pd.DataFrame


def fit_climatology(train: pd.DataFrame, fold_id: str) -> ClimatologyState:
    required = ["forecast_id", "group_id", "month", "hour", "actual_kwh"]
    missing = sorted(set(required) - set(train.columns))
    if missing:
        raise DataQualityError(f"climatology inputs are missing: {missing}")
    clean = train[required].dropna(subset=["actual_kwh"]).copy()
    if clean.empty:
        raise DataQualityError("climatology has no observed training targets")
    row_hash = sha256_dataframe(
        clean.sort_values(["forecast_id", "group_id"], kind="stable").reset_index(drop=True)
    )
    return ClimatologyState(
        fold_id=fold_id,
        training_rows_sha256=row_hash,
        by_group_month_hour=clean.groupby(["group_id", "month", "hour"], as_index=False)[
            "actual_kwh"
        ]
        .median()
        .rename(columns={"actual_kwh": "clim_gmh"}),
        by_group_month=clean.groupby(["group_id", "month"], as_index=False)["actual_kwh"]
        .median()
        .rename(columns={"actual_kwh": "clim_gm"}),
        by_group=clean.groupby(["group_id"], as_index=False)["actual_kwh"]
        .median()
        .rename(columns={"actual_kwh": "clim_g"}),
    )


def apply_climatology(
    state: ClimatologyState,
    frame: pd.DataFrame,
    fold_id: str,
) -> pd.DataFrame:
    if state.fold_id != fold_id:
        raise LeakageError("climatology state belongs to a different fold")
    required = {"group_id", "month", "hour"}
    if not required.issubset(frame.columns):
        raise DataQualityError(f"climatology transform is missing: {sorted(required - set(frame))}")
    result = frame.merge(
        state.by_group_month_hour,
        on=["group_id", "month", "hour"],
        how="left",
        validate="many_to_one",
    )
    result = result.merge(
        state.by_group_month,
        on=["group_id", "month"],
        how="left",
        validate="many_to_one",
    )
    result = result.merge(state.by_group, on=["group_id"], how="left", validate="many_to_one")
    result["clim_median"] = result["clim_gmh"].fillna(result["clim_gm"]).fillna(result["clim_g"])
    if result["clim_median"].isna().any():
        raise DataQualityError("climatology has no group-level fallback")
    return result.drop(columns=["clim_gmh", "clim_gm", "clim_g"])
