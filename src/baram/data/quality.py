"""Non-mutating data-quality audit and quarantine receipts."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd

from baram.contracts.hashing import canonical_sha256
from baram.contracts.types import GroupId, QualityReceipt
from baram.data.canonical import CanonicalTables
from baram.exceptions import DataQualityError


@dataclass(frozen=True)
class QualityAudit:
    receipt: QualityReceipt
    findings: Mapping[str, Any]


def _internal_gap_count(frame: pd.DataFrame) -> int:
    timestamps = frame["kst_dtm"].sort_values(kind="stable")
    return int(timestamps.diff().dropna().dt.total_seconds().ne(600.0).sum())


def _assert_grid_contract(frame: pd.DataFrame, grids: int, label: str) -> None:
    counts = frame.groupby("forecast_kst_dtm", sort=True)["grid_id"].nunique()
    if not counts.eq(grids).all():
        raise DataQualityError(f"{label} grid cardinality mismatch")


def audit_quality(
    tables: CanonicalTables,
    capacities: Mapping[GroupId, float],
) -> QualityAudit:
    """Record raw anomalies and fail only on primary weather/key contract breaks."""
    for frame, grids, label in (
        (tables.gfs_train, 9, "gfs_train"),
        (tables.gfs_test, 9, "gfs_test"),
        (tables.ldaps_train, 16, "ldaps_train"),
        (tables.ldaps_test, 16, "ldaps_test"),
    ):
        _assert_grid_contract(frame, grids, label)

    labels = tables.labels_long
    null_counts: dict[int, int] = {}
    negative_counts: dict[int, int] = {}
    over_capacity_counts: dict[int, int] = {}
    for group in (1, 2, 3):
        values = labels.loc[labels["group_id"].eq(group), "actual_kwh"]
        null_counts[group] = int(values.isna().sum())
        negative_counts[group] = int(values.lt(0).sum())
        over_capacity_counts[group] = int(values.gt(float(capacities[group])).sum())

    ldaps_missing_mask = tables.ldaps_test.isna().any(axis=1)
    missing_forecasts = (
        tables.ldaps_test.loc[ldaps_missing_mask, "forecast_kst_dtm"]
        .drop_duplicates()
        .sort_values()
        .dt.strftime("%Y-%m-%d %H:%M:%S")
        .tolist()
    )
    findings: dict[str, Any] = {
        "label_null_counts": null_counts,
        "label_negative_counts": negative_counts,
        "label_over_capacity_counts": over_capacity_counts,
        "gfs_blank_cells": {
            "train": int(tables.gfs_train.isna().sum().sum()),
            "test": int(tables.gfs_test.isna().sum().sum()),
        },
        "ldaps_train_blank_cells": int(tables.ldaps_train.isna().sum().sum()),
        "ldaps_test_blank_cells": int(tables.ldaps_test.isna().sum().sum()),
        "ldaps_test_missing_rows": int(ldaps_missing_mask.sum()),
        "ldaps_test_missing_forecasts": missing_forecasts,
        "scada_internal_gap_counts": {
            "unison": _internal_gap_count(tables.scada_unison),
            "vestas": _internal_gap_count(tables.scada_vestas),
        },
        "scada_rows": {
            "unison": len(tables.scada_unison),
            "vestas": len(tables.scada_vestas),
        },
    }
    critical_ok = (
        findings["gfs_blank_cells"] == {"train": 0, "test": 0}
        and findings["ldaps_train_blank_cells"] == 0
        and negative_counts == {1: 0, 2: 0, 3: 0}
    )
    if not critical_ok:
        raise DataQualityError("critical supplied-data fixture mismatch")
    receipt = QualityReceipt(
        findings_sha256=canonical_sha256(findings),
        quarantined=("unison_power_units", "vestas_power"),
        can_build_primary_features=True,
    )
    return QualityAudit(receipt=receipt, findings=findings)
