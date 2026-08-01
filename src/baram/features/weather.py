"""Deterministic calendar, vector, and global spatial weather features."""

from collections.abc import Mapping

import numpy as np
import pandas as pd

from baram.constants import CAPACITIES_KWH
from baram.data.canonical import add_operating_period
from baram.exceptions import ContractError, DataQualityError

_GFS_VECTOR_PAIRS: Mapping[str, tuple[str, str]] = {
    "wind10": ("heightAboveGround_10_10u", "heightAboveGround_10_10v"),
    "wind80": ("heightAboveGround_80_u", "heightAboveGround_80_v"),
    "wind100": ("heightAboveGround_100_100u", "heightAboveGround_100_100v"),
    "pbl_wind": ("planetaryBoundaryLayer_0_u", "planetaryBoundaryLayer_0_v"),
    "wind850": ("isobaricInhPa_850_u", "isobaricInhPa_850_v"),
    "wind700": ("isobaricInhPa_700_u", "isobaricInhPa_700_v"),
    "wind500": ("isobaricInhPa_500_u", "isobaricInhPa_500_v"),
}
_LDAPS_VECTOR_PAIRS: Mapping[str, tuple[str, str]] = {
    "wind10": ("heightAboveGround_10_10u", "heightAboveGround_10_10v"),
    "wind5": ("heightAboveGround_5_XBLWS", "heightAboveGround_5_YBLWS"),
    "wind50max": ("heightAboveGround_50_50MUmax", "heightAboveGround_50_50MVmax"),
    "wind50min": ("heightAboveGround_50_50MUmin", "heightAboveGround_50_50MVmin"),
}
_IDENTIFIERS = {
    "grid_id",
    "latitude",
    "longitude",
    "operating_year",
    "operating_quarter",
    "lead_hour",
}


def add_uv_features(frame: pd.DataFrame, u_col: str, v_col: str, prefix: str) -> pd.DataFrame:
    if u_col not in frame or v_col not in frame:
        raise ContractError(f"vector columns are missing: {u_col}, {v_col}")
    result = frame.copy()
    u = result[u_col].to_numpy(dtype=float)
    v = result[v_col].to_numpy(dtype=float)
    speed = np.hypot(u, v)
    nonzero = speed > 0.0
    result[f"{prefix}_speed"] = speed
    result[f"{prefix}_dir_sin"] = np.divide(v, speed, out=np.zeros_like(v), where=nonzero)
    result[f"{prefix}_dir_cos"] = np.divide(u, speed, out=np.zeros_like(u), where=nonzero)
    return result


def _with_known_vectors(
    frame: pd.DataFrame,
    pairs: Mapping[str, tuple[str, str]],
) -> pd.DataFrame:
    result = frame.copy()
    for prefix, (u_col, v_col) in pairs.items():
        if u_col in result and v_col in result:
            result = add_uv_features(result, u_col, v_col, prefix)
    return result


def aggregate_weather(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Aggregate all numeric grid fields without averaging raw angular fields."""
    keys = ["forecast_kst_dtm", "data_available_kst_dtm"]
    if not set(keys).issubset(frame.columns):
        raise ContractError(f"weather aggregation requires keys: {keys}")
    numeric = [
        name
        for name in frame.select_dtypes(include=[np.number]).columns
        if name not in _IDENTIFIERS and not name.lower().endswith(("_wd", "_direction", "_angle"))
    ]
    if not numeric:
        raise ContractError("weather aggregation has no numeric value columns")
    grouped = frame.groupby(keys, sort=True, dropna=False)
    basic = grouped[numeric].agg(["mean", "std", "min", "max"])
    quantiles = grouped[numeric].quantile([0.1, 0.5, 0.9]).unstack(level=-1)
    quantile_names = {0.1: "q10", 0.5: "q50", 0.9: "q90"}
    quantiles.columns = pd.MultiIndex.from_tuples(
        [(column, quantile_names[float(level)]) for column, level in quantiles.columns]
    )
    aggregated = pd.concat([basic, quantiles], axis=1).reindex(
        columns=pd.MultiIndex.from_tuples(
            [
                (column, statistic)
                for column in numeric
                for statistic in ("mean", "std", "min", "max", "q10", "q50", "q90")
            ]
        )
    )
    aggregated.columns = [
        f"{prefix}__{column}__{statistic}" for column, statistic in aggregated.columns
    ]
    result = aggregated.reset_index()
    std_columns = [name for name in result if name.endswith("__std")]
    result[std_columns] = result[std_columns].fillna(0.0)
    grid_count = grouped.size()
    observed_cells = grouped[numeric].count().sum(axis=1)
    missing_cells = grid_count * len(numeric) - observed_cells
    metadata = pd.DataFrame(
        {
            f"{prefix}__missing_cell_count": missing_cells.to_numpy(dtype="int32"),
            f"{prefix}__grid_count": grid_count.to_numpy(dtype="int16"),
        }
    )
    return pd.concat([result, metadata], axis=1).copy()


def _calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = add_operating_period(frame, "forecast_kst_dtm")
    hour = result["forecast_kst_dtm"].dt.hour
    day_of_year = result["operating_day"].dt.dayofyear
    result["hour"] = hour.astype("int8")
    result["month"] = result["operating_day"].dt.month.astype("int8")
    result["day_of_year"] = day_of_year.astype("int16")
    result["cal__hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    result["cal__hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    result["cal__doy_sin"] = np.sin(2.0 * np.pi * day_of_year / 365.25)
    result["cal__doy_cos"] = np.cos(2.0 * np.pi * day_of_year / 365.25)
    result["lead_hour"] = (
        (result["forecast_kst_dtm"] - result["data_available_kst_dtm"])
        .dt.total_seconds()
        .div(3600.0)
        .astype("int16")
    )
    if (result["lead_hour"] <= 0).any():
        raise DataQualityError("feature lead hours must be positive")
    result["issuance_batch"] = result["data_available_kst_dtm"].dt.strftime("%Y%m%d%H")
    return result


def build_weather_features(
    gfs: pd.DataFrame,
    ldaps: pd.DataFrame,
    forecast_keys: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build one globally aggregated feature row per timestamp and group."""
    gfs_agg = aggregate_weather(_with_known_vectors(gfs, _GFS_VECTOR_PAIRS), "gfs")
    ldaps_agg = aggregate_weather(_with_known_vectors(ldaps, _LDAPS_VECTOR_PAIRS), "ldaps")
    merged = gfs_agg.merge(
        ldaps_agg,
        on=["forecast_kst_dtm", "data_available_kst_dtm"],
        how="inner",
        validate="one_to_one",
        sort=True,
    )
    if (
        len(merged) != gfs["forecast_kst_dtm"].nunique()
        or len(merged) != ldaps["forecast_kst_dtm"].nunique()
    ):
        raise ContractError("GFS and LDAPS forecast/availability keys do not align")
    merged = _calendar_features(merged)
    if forecast_keys is None:
        merged["forecast_id"] = "train-" + merged["forecast_kst_dtm"].dt.strftime("%Y%m%d%H")
    else:
        required = ["forecast_id", "forecast_kst_dtm"]
        merged = forecast_keys[required].merge(
            merged,
            on="forecast_kst_dtm",
            how="left",
            validate="one_to_one",
            sort=False,
        )
        if merged.drop(columns=required).isna().all(axis=1).any():
            raise ContractError("forecast keys are missing weather features")
    group_frames: list[pd.DataFrame] = []
    for group, capacity in CAPACITIES_KWH.items():
        part = merged.copy()
        part["group_id"] = np.int8(group)
        part["capacity_kwh"] = np.float32(capacity)
        group_frames.append(part)
    return (
        pd.concat(group_frames, ignore_index=True)
        .sort_values(["forecast_kst_dtm", "group_id"], kind="stable")
        .reset_index(drop=True)
    )
