"""Sample-preserving construction of a local submission candidate."""

from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import MergeError

from baram.constants import SUBMISSION_COLUMNS
from baram.contracts.hashing import sha256_file
from baram.exceptions import SubmissionError

_KEYS = ["forecast_id", "forecast_kst_dtm"]
_PREDICTIONS = ["kpx_group_1", "kpx_group_2", "kpx_group_3"]


def _canonical_keys(frame: pd.DataFrame, name: str, expected_rows: int) -> pd.DataFrame:
    missing = set(_KEYS) - set(frame.columns)
    if missing:
        raise SubmissionError(f"{name} is missing submission keys: {sorted(missing)}")
    result = frame[_KEYS].copy()
    try:
        result["forecast_kst_dtm"] = pd.to_datetime(result["forecast_kst_dtm"], errors="raise")
    except (TypeError, ValueError) as error:
        raise SubmissionError(f"{name} has invalid timestamps: {error}") from error
    if len(result) != expected_rows:
        raise SubmissionError(f"{name} must contain exactly {expected_rows} rows")
    if result.duplicated(_KEYS).any():
        raise SubmissionError(f"{name} contains duplicate keys")
    return result


def build_submission(
    sample: pd.DataFrame,
    wide_predictions: pd.DataFrame,
    output: Path,
    *,
    expected_rows: int = 8760,
) -> str:
    """Write deterministic UTF-8-BOM CSV bytes in immutable sample key order."""
    sample_keys = _canonical_keys(sample, "sample", expected_rows)
    if tuple(wide_predictions.columns) != SUBMISSION_COLUMNS:
        raise SubmissionError(f"wide prediction columns must equal {list(SUBMISSION_COLUMNS)}")
    prediction_keys = _canonical_keys(wide_predictions, "predictions", expected_rows)
    if prediction_keys.duplicated(_KEYS).any():
        raise SubmissionError("predictions contain duplicate keys")
    numeric = wide_predictions[_PREDICTIONS].apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0.0).any():
        raise SubmissionError("submission predictions must be finite and nonnegative")
    prepared = prediction_keys.assign(**{name: numeric[name] for name in _PREDICTIONS})
    ordered = sample_keys.assign(_sample_order=np.arange(expected_rows, dtype=np.int64))
    try:
        merged = ordered.merge(
            prepared,
            on=_KEYS,
            how="left",
            validate="one_to_one",
            sort=False,
            indicator=True,
        )
    except MergeError as error:
        raise SubmissionError(f"submission key merge failed: {error}") from error
    if not merged["_merge"].eq("both").all() or merged[_PREDICTIONS].isna().any().any():
        raise SubmissionError("submission has missing or mismatched prediction keys")
    merged = merged.sort_values("_sample_order", kind="stable")
    candidate = merged[list(SUBMISSION_COLUMNS)]
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    candidate.to_csv(
        temporary,
        index=False,
        encoding="utf-8-sig",
        lineterminator="\n",
        date_format="%Y-%m-%d %H:%M:%S",
    )
    temporary.replace(output)
    return sha256_file(output)
