from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from baram.contracts.hashing import sha256_file
from baram.exceptions import SubmissionError
from baram.submission.build import build_submission


def _sample(rows: int = 8760) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "forecast_id": [f"forecast_{index:04d}" for index in range(1, rows + 1)],
            "forecast_kst_dtm": pd.date_range("2025-01-01 01:00:00", periods=rows, freq="h"),
        }
    )


def _predictions(sample: pd.DataFrame) -> pd.DataFrame:
    return sample.assign(kpx_group_1=1.0, kpx_group_2=2.0, kpx_group_3=3.0)


def test_build_submission_is_sample_ordered_bom_and_hash_stable(tmp_path: Path) -> None:
    """Catches row sorting, index export, encoding drift, or unstable candidate bytes."""
    sample = _sample()
    predictions = _predictions(sample).iloc[::-1]
    output = tmp_path / "submission.csv"
    first = build_submission(sample, predictions, output)
    second = build_submission(sample, predictions, output)
    assert first == second == sha256_file(output)
    assert output.read_bytes().startswith(b"\xef\xbb\xbf")
    result = pd.read_csv(output, encoding="utf-8-sig")
    assert len(result) == 8760
    assert list(result) == [
        "forecast_id",
        "forecast_kst_dtm",
        "kpx_group_1",
        "kpx_group_2",
        "kpx_group_3",
    ]
    assert result["forecast_id"].tolist() == sample["forecast_id"].tolist()
    assert not any(name.startswith("Unnamed") for name in result)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "nan", "inf", "negative"])
def test_build_submission_rejects_invalid_predictions(tmp_path: Path, mutation: str) -> None:
    """Catches incomplete, duplicate, or physically invalid local predictions."""
    sample = _sample()
    predictions = _predictions(sample)
    if mutation == "missing":
        predictions = predictions.iloc[:-1]
    elif mutation == "duplicate":
        predictions = pd.concat([predictions, predictions.iloc[[0]]], ignore_index=True)
    elif mutation == "nan":
        predictions.loc[0, "kpx_group_1"] = np.nan
    elif mutation == "inf":
        predictions.loc[0, "kpx_group_2"] = np.inf
    else:
        predictions.loc[0, "kpx_group_3"] = -1.0
    with pytest.raises(SubmissionError):
        build_submission(sample, predictions, tmp_path / "invalid.csv")
