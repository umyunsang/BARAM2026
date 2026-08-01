from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from baram.exceptions import ContractError
from baram.predictions.registry import read_prediction_artifact, write_prediction_artifact


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "forecast_id": ["a", "b"],
            "forecast_kst_dtm": pd.to_datetime(["2023-01-01 01:00", "2023-01-01 02:00"]),
            "group_id": [1, 1],
            "prediction_kwh": [100.0, 200.0],
            "fold_id": ["fold-a", "fold-a"],
            "model_id": ["model-a", "model-a"],
        }
    )


def test_prediction_artifact_roundtrip_is_atomic_and_stable(tmp_path: Path) -> None:
    """Catches partial writes or nondeterministic row order in keyed predictions."""
    path = tmp_path / "prediction.parquet"
    expected = _predictions()[["forecast_id", "forecast_kst_dtm", "group_id"]]
    first = write_prediction_artifact(
        _predictions().iloc[::-1], path, expected, "model-a", "fold-a"
    )
    second = write_prediction_artifact(_predictions(), path, expected, "model-a", "fold-a")
    assert first == second
    assert not path.with_suffix(".parquet.tmp").exists()
    pd.testing.assert_frame_equal(read_prediction_artifact(path), _predictions())


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "nonfinite", "wrong_parent"])
def test_prediction_artifact_rejects_contract_mismatch(tmp_path: Path, mutation: str) -> None:
    """Catches duplicate/missing/nonfinite rows or lineage substitution."""
    frame = _predictions()
    expected = frame[["forecast_id", "forecast_kst_dtm", "group_id"]]
    model_id, fold_id = "model-a", "fold-a"
    if mutation == "duplicate":
        frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    elif mutation == "missing":
        frame = frame.iloc[[0]]
    elif mutation == "nonfinite":
        frame.loc[0, "prediction_kwh"] = np.inf
    else:
        model_id = "model-b"
    with pytest.raises(ContractError):
        write_prediction_artifact(frame, tmp_path / "x.parquet", expected, model_id, fold_id)
