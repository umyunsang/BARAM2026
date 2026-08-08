from baram.config import ProjectConfig
from baram.data.canonical import CanonicalTables
from baram.data.quality import audit_quality


def test_actual_quality_receipt_matches_frozen_fixture(
    canonical_tables: CanonicalTables,
    config: ProjectConfig,
) -> None:
    """Catches weakening or silent repair of the frozen data-quality fixture."""
    audit = audit_quality(canonical_tables, config.capacities)
    assert audit.receipt.can_build_primary_features is True
    assert audit.receipt.quarantined == ("unison_power_units", "vestas_power")
    assert audit.findings["label_null_counts"] == {1: 104, 2: 103, 3: 8766}
    assert audit.findings["label_negative_counts"] == {1: 0, 2: 0, 3: 0}
    assert audit.findings["label_over_capacity_counts"] == {1: 0, 2: 0, 3: 38}
    assert audit.findings["gfs_blank_cells"] == {"train": 0, "test": 0}
    assert audit.findings["ldaps_test_missing_rows"] == 48
    assert audit.findings["ldaps_test_missing_forecasts"] == [
        "2025-04-08 17:00:00",
        "2025-06-18 18:00:00",
        "2025-07-18 06:00:00",
    ]
    assert audit.findings["scada_internal_gap_counts"] == {"unison": 0, "vestas": 0}
    assert audit.findings["turbine_topology"]["turbine_count"] == 17
    assert audit.findings["turbine_topology"]["group_counts"] == {1: 6, 2: 6, 3: 5}


def test_quality_audit_preserves_raw_ldaps_missing_values(
    canonical_tables: CanonicalTables,
    config: ProjectConfig,
) -> None:
    """Catches a quality audit mutating raw missing weather values."""
    before = int(canonical_tables.ldaps_test.isna().sum().sum())
    assert before == 752
    audit_quality(canonical_tables, config.capacities)
    after = int(canonical_tables.ldaps_test.isna().sum().sum())
    assert after == before
    affected = canonical_tables.ldaps_test.loc[canonical_tables.ldaps_test.isna().any(axis=1)]
    assert affected["forecast_kst_dtm"].nunique() == 3
    assert affected.groupby("forecast_kst_dtm")["grid_id"].nunique().eq(16).all()


def test_scada_timestamps_have_ten_minute_internal_cadence(
    canonical_tables: CanonicalTables,
) -> None:
    """Catches a false quality PASS when an observed SCADA interval is missing."""
    for frame in (canonical_tables.scada_vestas, canonical_tables.scada_unison):
        seconds = frame["kst_dtm"].sort_values().diff().dropna().dt.total_seconds()
        assert seconds.eq(600.0).all()
