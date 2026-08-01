from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from baram.exceptions import SubmissionError
from baram.submission.build import build_submission
from baram.submission.validate import validate_submission


def _sample(rows: int = 8760) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "forecast_id": [f"forecast_{index:04d}" for index in range(1, rows + 1)],
            "forecast_kst_dtm": pd.date_range("2025-01-01 01:00:00", periods=rows, freq="h"),
        }
    )


def _write_candidate(path: Path, sample: pd.DataFrame) -> None:
    predictions = sample.assign(kpx_group_1=100.0, kpx_group_2=200.0, kpx_group_3=300.0)
    build_submission(sample, predictions, path)


def test_validate_submission_returns_complete_receipt(tmp_path: Path) -> None:
    """Catches a candidate passing without immutable provenance or policy bounds."""
    sample = _sample()
    path = tmp_path / "submission.csv"
    _write_candidate(path, sample)
    receipt = validate_submission(
        path,
        sample,
        candidate_id="candidate-a",
        source_sha256="1" * 64,
        champion_policy_sha256="2" * 64,
        cap_modes={1: "capacity", 2: "1.01_capacity", 3: "nonnegative_only"},
    )
    assert receipt.row_count == 8760
    assert receipt.encoding == "utf-8-sig"
    assert receipt.csv_sha256
    assert receipt.sample_keys_sha256


@pytest.mark.parametrize("mutation", ["reordered", "missing", "nan", "inf", "negative", "cap"])
def test_validate_submission_rejects_contract_failure(tmp_path: Path, mutation: str) -> None:
    """Catches file-level key, numeric, or declared-cap policy violations."""
    sample = _sample()
    path = tmp_path / "submission.csv"
    _write_candidate(path, sample)
    frame = pd.read_csv(path, encoding="utf-8-sig")
    if mutation == "reordered":
        frame = frame.iloc[::-1]
    elif mutation == "missing":
        frame = frame.iloc[:-1]
    elif mutation == "nan":
        frame.loc[0, "kpx_group_1"] = np.nan
    elif mutation == "inf":
        frame.loc[0, "kpx_group_2"] = np.inf
    elif mutation == "negative":
        frame.loc[0, "kpx_group_3"] = -1.0
    else:
        frame.loc[0, "kpx_group_1"] = 21600.01
    frame.to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\n")
    with pytest.raises(SubmissionError):
        validate_submission(
            path,
            sample,
            candidate_id="candidate-a",
            source_sha256="1" * 64,
            champion_policy_sha256="2" * 64,
            cap_modes={1: "capacity", 2: "1.01_capacity", 3: "nonnegative_only"},
        )


def test_validate_submission_rejects_missing_bom(tmp_path: Path) -> None:
    """Catches an otherwise readable CSV written with the wrong encoding contract."""
    sample = _sample()
    path = tmp_path / "submission.csv"
    predictions = sample.assign(kpx_group_1=1.0, kpx_group_2=2.0, kpx_group_3=3.0)
    predictions.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
    with pytest.raises(SubmissionError, match="BOM"):
        validate_submission(
            path,
            sample,
            candidate_id="candidate-a",
            source_sha256="1" * 64,
            champion_policy_sha256="2" * 64,
            cap_modes={1: "capacity", 2: "capacity", 3: "capacity"},
        )
