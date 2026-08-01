"""Atomic, keyed, content-addressed prediction artifacts."""

from pathlib import Path

import numpy as np
import pandas as pd

from baram.contracts.hashing import sha256_file
from baram.exceptions import ContractError

_KEYS = ["forecast_id", "forecast_kst_dtm", "group_id"]
_COLUMNS = [*_KEYS, "prediction_kwh", "fold_id", "model_id"]


def _canonical_prediction_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if len(frame.columns) != len(_COLUMNS) or set(frame.columns) != set(_COLUMNS):
        raise ContractError(f"prediction columns must equal {_COLUMNS}")
    result = frame[_COLUMNS].copy()
    result["forecast_kst_dtm"] = pd.to_datetime(result["forecast_kst_dtm"], errors="raise")
    if result.duplicated(_KEYS).any():
        raise ContractError("prediction artifact contains duplicate keys")
    if (
        result["prediction_kwh"].isna().any()
        or not np.isfinite(result["prediction_kwh"].to_numpy(dtype=float)).all()
    ):
        raise ContractError("prediction artifact contains non-finite values")
    return result.sort_values(_KEYS, kind="stable").reset_index(drop=True)


def write_prediction_artifact(
    frame: pd.DataFrame,
    path: Path,
    expected_keys: pd.DataFrame,
    model_id: str,
    fold_id: str,
) -> str:
    canonical = _canonical_prediction_frame(frame)
    if set(canonical["model_id"].unique()) != {model_id}:
        raise ContractError("prediction model hash/ID does not match its parent")
    if set(canonical["fold_id"].unique()) != {fold_id}:
        raise ContractError("prediction fold hash/ID does not match its parent")
    expected = expected_keys[_KEYS].copy()
    expected["forecast_kst_dtm"] = pd.to_datetime(expected["forecast_kst_dtm"], errors="raise")
    expected = expected.sort_values(_KEYS, kind="stable").reset_index(drop=True)
    if not canonical[_KEYS].equals(expected):
        raise ContractError("prediction keys differ from expected validation keys")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    canonical.to_parquet(temporary, index=False, engine="pyarrow", compression="zstd")
    temporary.replace(path)
    return sha256_file(path)


def read_prediction_artifact(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise ContractError(f"prediction artifact does not exist: {path}")
    return _canonical_prediction_frame(pd.read_parquet(path, engine="pyarrow"))
