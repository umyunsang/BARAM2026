"""Byte-, schema-, key-, numeric-, and policy-level candidate validation."""

import re
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from baram.constants import CAPACITIES_KWH, SUBMISSION_COLUMNS
from baram.contracts.hashing import canonical_sha256, sha256_file
from baram.contracts.types import SubmissionReceipt
from baram.exceptions import SubmissionError

_KEYS = ["forecast_id", "forecast_kst_dtm"]
_PREDICTIONS = ["kpx_group_1", "kpx_group_2", "kpx_group_3"]
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _normalized_keys(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    missing = set(_KEYS) - set(frame.columns)
    if missing:
        raise SubmissionError(f"{name} is missing keys: {sorted(missing)}")
    result = frame[_KEYS].copy()
    try:
        result["forecast_kst_dtm"] = pd.to_datetime(result["forecast_kst_dtm"], errors="raise")
    except (TypeError, ValueError) as error:
        raise SubmissionError(f"{name} contains invalid timestamps: {error}") from error
    if result.duplicated(_KEYS).any():
        raise SubmissionError(f"{name} contains duplicate keys")
    return result.reset_index(drop=True)


def _sample_keys_hash(keys: pd.DataFrame) -> str:
    serializable = keys.copy()
    serializable["forecast_kst_dtm"] = serializable["forecast_kst_dtm"].dt.strftime(
        "%Y-%m-%dT%H:%M:%S"
    )
    return canonical_sha256(serializable.to_dict(orient="records"))


def validate_submission(
    path: Path,
    sample: pd.DataFrame,
    *,
    candidate_id: str,
    source_sha256: str,
    champion_policy_sha256: str,
    cap_modes: Mapping[int, str],
    expected_rows: int = 8760,
) -> SubmissionReceipt:
    """Return a provenance receipt only after the complete P6 contract passes."""
    if not path.is_file():
        raise SubmissionError(f"submission candidate does not exist: {path}")
    if not path.read_bytes().startswith(b"\xef\xbb\xbf"):
        raise SubmissionError("submission must begin with the UTF-8 BOM")
    if not candidate_id:
        raise SubmissionError("submission candidate ID cannot be empty")
    if (
        _SHA256.fullmatch(source_sha256) is None
        or _SHA256.fullmatch(champion_policy_sha256) is None
    ):
        raise SubmissionError("submission lineage requires lowercase SHA-256 values")
    if set(cap_modes) != {1, 2, 3}:
        raise SubmissionError("submission cap policies must cover groups 1, 2, and 3")
    try:
        frame = pd.read_csv(path, encoding="utf-8-sig")
    except (OSError, UnicodeError, pd.errors.ParserError) as error:
        raise SubmissionError(f"cannot read submission candidate: {error}") from error
    if tuple(frame.columns) != SUBMISSION_COLUMNS:
        raise SubmissionError(f"submission columns/order must equal {list(SUBMISSION_COLUMNS)}")
    if len(frame) != expected_rows:
        raise SubmissionError(f"submission must contain exactly {expected_rows} rows")

    candidate_keys = _normalized_keys(frame, "submission")
    sample_keys = _normalized_keys(sample, "sample")
    if len(sample_keys) != expected_rows:
        raise SubmissionError(f"sample must contain exactly {expected_rows} rows")
    if not candidate_keys.equals(sample_keys):
        raise SubmissionError("submission keys/order differ from the immutable sample")
    numeric = frame[_PREDICTIONS].apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise SubmissionError("submission predictions must be finite")
    if (values < 0.0).any():
        raise SubmissionError("submission predictions must be nonnegative")

    allowed_modes = {"capacity", "1.01_capacity", "nonnegative_only"}
    for group_id, column in enumerate(_PREDICTIONS, start=1):
        mode = cap_modes[group_id]
        if mode not in allowed_modes:
            raise SubmissionError(f"unsupported cap policy for group {group_id}: {mode}")
        if mode == "capacity" and numeric[column].gt(CAPACITIES_KWH[group_id]).any():
            raise SubmissionError(f"group {group_id} exceeds its declared capacity cap")
        if mode == "1.01_capacity" and numeric[column].gt(1.01 * CAPACITIES_KWH[group_id]).any():
            raise SubmissionError(f"group {group_id} exceeds its declared 1.01 capacity cap")

    return SubmissionReceipt(
        candidate_id=candidate_id,
        csv_sha256=sha256_file(path),
        row_count=len(frame),
        source_sha256=source_sha256,
        champion_policy_sha256=champion_policy_sha256,
        sample_keys_sha256=_sample_keys_hash(sample_keys),
        encoding="utf-8-sig",
    )
