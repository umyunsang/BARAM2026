"""Physically load only the pre-2024 development surface.

The historical experiment helpers materialize the complete training parquet and
mask rows afterwards.  That is no longer acceptable after the 2024 lockbox was
consumed.  This module applies parquet predicates before materialization and is
the only loader new development runners should use.
"""

from __future__ import annotations

import pandas as pd
from run_sequence_classifier import CACHE

DEV_CUTOFF = pd.Timestamp("2024-01-01 00:00:00")


def _read_pre2024(name: str) -> pd.DataFrame:
    frame = pd.read_parquet(
        CACHE / name,
        filters=[("forecast_kst_dtm", "<", DEV_CUTOFF)],
    )
    frame["forecast_kst_dtm"] = pd.to_datetime(frame["forecast_kst_dtm"])
    if frame.empty or not frame["forecast_kst_dtm"].lt(DEV_CUTOFF).all():
        raise RuntimeError(f"{name} violated the physical pre-2024 predicate")
    return frame


def development_surface() -> tuple[pd.DataFrame, list[str]]:
    """Return pre-2024 rows and every inference-available numeric feature."""
    features = _read_pre2024("train_features.parquet")
    grid = _read_pre2024("train_grid_pivot.parquet")
    geometric = _read_pre2024("train_geometric.parquet")
    labels = _read_pre2024("labels_long.parquet")

    keys = ["forecast_kst_dtm", "group_id"]
    surface = (
        features.merge(grid, on="forecast_kst_dtm", validate="many_to_one")
        .merge(
            geometric,
            on=["forecast_kst_dtm", "data_available_kst_dtm", "group_id"],
            validate="one_to_one",
        )
        .merge(
            labels[[*keys, "actual_kwh"]],
            on=keys,
            validate="one_to_one",
        )
    )
    if len(surface) != len(labels):
        raise RuntimeError("pre-2024 development merge changed the row contract")
    if not surface["forecast_kst_dtm"].lt(DEV_CUTOFF).all():
        raise RuntimeError("development surface contains a lockbox timestamp")
    if set(surface["forecast_kst_dtm"].dt.year.unique()) != {2022, 2023}:
        raise RuntimeError("development surface year contract changed")

    for group_id in (1, 2, 3):
        surface[f"group_{group_id}"] = (
            surface["group_id"].eq(group_id).astype("int8")
        )
    excluded = {
        "forecast_id",
        "forecast_kst_dtm",
        "data_available_kst_dtm",
        "issuance_batch",
        "actual_kwh",
    }
    numeric = [
        name
        for name in surface.columns
        if name not in excluded and pd.api.types.is_numeric_dtype(surface[name])
    ]
    if len(numeric) < 700:
        raise RuntimeError(
            f"development feature contract unexpectedly resolved {len(numeric)} columns"
        )
    return surface, numeric
