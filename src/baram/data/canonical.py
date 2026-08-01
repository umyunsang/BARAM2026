"""Canonical hourly/group tables derived without mutating supplied data."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from baram.data.archive import read_csv_member
from baram.exceptions import ContractError, DataQualityError


@dataclass(frozen=True)
class CanonicalTables:
    labels_long: pd.DataFrame
    submission_keys: pd.DataFrame
    gfs_train: pd.DataFrame
    gfs_test: pd.DataFrame
    ldaps_train: pd.DataFrame
    ldaps_test: pd.DataFrame
    scada_vestas: pd.DataFrame
    scada_unison: pd.DataFrame


def _parse_timestamp(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    if column not in frame:
        raise ContractError(f"missing timestamp column: {column}")
    result = frame.copy()
    try:
        parsed = pd.to_datetime(result[column], errors="raise")
    except (TypeError, ValueError) as error:
        raise DataQualityError(f"invalid timestamp in {column}: {error}") from error
    if parsed.dt.tz is not None:
        raise ContractError(f"{column} must be KST-naive with timezone recorded in the manifest")
    result[column] = parsed
    return result


def add_operating_period(frame: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    """Attach the shifted day/year/quarter shared by splits and calendar features."""
    result = frame.copy()
    if timestamp_col not in result:
        raise ContractError(f"missing operating timestamp column: {timestamp_col}")
    if not pd.api.types.is_datetime64_any_dtype(result[timestamp_col]):
        result = _parse_timestamp(result, timestamp_col)
    operating_day = (result[timestamp_col] - np.timedelta64(1, "h")).dt.normalize()
    result["operating_day"] = operating_day
    result["operating_year"] = operating_day.dt.year.astype("int16")
    result["operating_quarter"] = operating_day.dt.quarter.astype("int8")
    return result


def labels_to_long(labels: pd.DataFrame) -> pd.DataFrame:
    expected = {"kst_dtm", "kpx_group_1", "kpx_group_2", "kpx_group_3"}
    if set(labels.columns) != expected:
        raise ContractError(f"label columns must equal {sorted(expected)}")
    renamed = labels.rename(columns={"kst_dtm": "forecast_kst_dtm"})
    renamed = _parse_timestamp(renamed, "forecast_kst_dtm")
    renamed = add_operating_period(renamed, "forecast_kst_dtm")
    renamed["forecast_id"] = "train-" + renamed["forecast_kst_dtm"].dt.strftime("%Y%m%d%H")
    long = renamed.melt(
        id_vars=[
            "forecast_id",
            "forecast_kst_dtm",
            "operating_day",
            "operating_year",
            "operating_quarter",
        ],
        value_vars=["kpx_group_1", "kpx_group_2", "kpx_group_3"],
        var_name="group_name",
        value_name="actual_kwh",
    )
    long["group_id"] = long["group_name"].str.extract(r"(\d)$").astype("int8")
    return (
        long.drop(columns="group_name")
        .sort_values(["forecast_kst_dtm", "group_id"], kind="stable")
        .reset_index(drop=True)
    )


def submission_to_keys(sample: pd.DataFrame) -> pd.DataFrame:
    expected = {
        "forecast_id",
        "forecast_kst_dtm",
        "kpx_group_1",
        "kpx_group_2",
        "kpx_group_3",
    }
    if set(sample.columns) != expected:
        raise ContractError(f"sample columns must equal {sorted(expected)}")
    keys = _parse_timestamp(sample[["forecast_id", "forecast_kst_dtm"]], "forecast_kst_dtm")
    if keys["forecast_id"].duplicated().any() or keys["forecast_kst_dtm"].duplicated().any():
        raise ContractError("sample submission contains duplicate keys")
    return add_operating_period(keys, "forecast_kst_dtm")


def canonicalize_weather(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "forecast_kst_dtm",
        "data_available_kst_dtm",
        "grid_id",
        "latitude",
        "longitude",
    }
    if not required.issubset(frame.columns):
        raise ContractError(f"weather is missing columns: {sorted(required - set(frame.columns))}")
    result = _parse_timestamp(frame, "forecast_kst_dtm")
    result = _parse_timestamp(result, "data_available_kst_dtm")
    result = add_operating_period(result, "forecast_kst_dtm")
    delta_hours = (
        result["forecast_kst_dtm"] - result["data_available_kst_dtm"]
    ).dt.total_seconds() / 3600.0
    if delta_hours.isna().any() or (delta_hours <= 0).any():
        raise DataQualityError("weather availability must precede every forecast timestamp")
    result["lead_hour"] = delta_hours.astype("int16")
    result["issuance_batch"] = result["data_available_kst_dtm"].dt.strftime("%Y%m%d%H")
    if result.duplicated(["forecast_kst_dtm", "grid_id"]).any():
        raise DataQualityError("weather contains duplicate forecast/grid keys")
    return result


def _canonicalize_scada(frame: pd.DataFrame) -> pd.DataFrame:
    result = _parse_timestamp(frame, "kst_dtm")
    if result["kst_dtm"].duplicated().any():
        raise DataQualityError("SCADA contains duplicate timestamps")
    return result


def load_canonical_tables(path: Path) -> CanonicalTables:
    """Read all allowlisted CSV members into their canonical in-memory contracts."""
    labels = labels_to_long(read_csv_member(path, "train/train_labels.csv"))
    sample = submission_to_keys(read_csv_member(path, "sample_submission.csv"))
    gfs_train = canonicalize_weather(read_csv_member(path, "train/gfs_train.csv"))
    gfs_test = canonicalize_weather(read_csv_member(path, "test/gfs_test.csv"))
    ldaps_train = canonicalize_weather(read_csv_member(path, "train/ldaps_train.csv"))
    ldaps_test = canonicalize_weather(read_csv_member(path, "test/ldaps_test.csv"))
    if list(gfs_train.columns) != list(gfs_test.columns):
        raise ContractError("GFS train/test schemas differ")
    if list(ldaps_train.columns) != list(ldaps_test.columns):
        raise ContractError("LDAPS train/test schemas differ")
    scada_vestas = _canonicalize_scada(read_csv_member(path, "train/scada_vestas_train.csv"))
    scada_unison = _canonicalize_scada(read_csv_member(path, "train/scada_unison_train.csv"))
    return CanonicalTables(
        labels_long=labels,
        submission_keys=sample,
        gfs_train=gfs_train,
        gfs_test=gfs_test,
        ldaps_train=ldaps_train,
        ldaps_test=ldaps_test,
        scada_vestas=scada_vestas,
        scada_unison=scada_unison,
    )
