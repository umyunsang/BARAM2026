import numpy as np
import pandas as pd

from baram.contracts.hashing import sha256_dataframe


def test_dataframe_hash_is_stable_and_order_sensitive() -> None:
    """Catches lineage hashing that materializes records or ignores schema/row order."""
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-01-01", "2025-01-02"]),
            "value": [1.0, np.nan],
            "name": ["a", "b"],
        }
    )
    first = sha256_dataframe(frame)
    assert first == sha256_dataframe(frame.copy())
    assert first != sha256_dataframe(frame.iloc[::-1].reset_index(drop=True))
    assert first != sha256_dataframe(frame[["name", "value", "timestamp"]])
