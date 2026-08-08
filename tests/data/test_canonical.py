import pandas as pd

from baram.data.canonical import CanonicalTables, add_operating_period


def test_operating_day_keeps_midnight_with_prior_batch() -> None:
    """Catches calendar boundaries splitting the final hour from its issuance batch."""
    frame = pd.DataFrame({"forecast_kst_dtm": pd.to_datetime(["2023-07-01 00:00:00"])})
    result = add_operating_period(frame, "forecast_kst_dtm")
    assert result.loc[0, "operating_day"] == pd.Timestamp("2023-06-30")
    assert result.loc[0, "operating_year"] == 2023
    assert result.loc[0, "operating_quarter"] == 2
    assert result.loc[0, "forecast_kst_dtm"].hour == 0


def test_actual_canonical_label_and_submission_contract(canonical_tables: CanonicalTables) -> None:
    """Catches row, key, period, or ordering drift in the canonical targets."""
    labels = canonical_tables.labels_long
    sample = canonical_tables.submission_keys
    assert len(labels) == 26304 * 3
    assert labels["forecast_kst_dtm"].nunique() == 26304
    assert labels["forecast_kst_dtm"].min() == pd.Timestamp("2022-01-01 01:00:00")
    assert labels["forecast_kst_dtm"].max() == pd.Timestamp("2025-01-01 00:00:00")
    assert not labels.duplicated(["forecast_id", "forecast_kst_dtm", "group_id"]).any()
    assert labels.iloc[0]["forecast_id"] == "train-2022010101"

    operating_counts = (
        labels.drop_duplicates("forecast_kst_dtm")["operating_year"].value_counts().to_dict()
    )
    assert operating_counts == {2022: 8760, 2023: 8760, 2024: 8784}
    assert len(sample) == 8760
    assert sample.iloc[0].to_dict() == {
        "forecast_id": "forecast_0001",
        "forecast_kst_dtm": pd.Timestamp("2025-01-01 01:00:00"),
        "operating_day": pd.Timestamp("2025-01-01 00:00:00"),
        "operating_year": 2025,
        "operating_quarter": 1,
    }
    assert sample.iloc[-1]["forecast_id"] == "forecast_8760"
    assert sample.iloc[-1]["forecast_kst_dtm"] == pd.Timestamp("2026-01-01 00:00:00")
    assert sample["operating_year"].value_counts().to_dict() == {2025: 8760}


def test_actual_weather_contract(canonical_tables: CanonicalTables) -> None:
    """Catches train/test schema drift, duplicate grids, or wrong grid cardinality."""
    for train, test, grids in (
        (canonical_tables.gfs_train, canonical_tables.gfs_test, 9),
        (canonical_tables.ldaps_train, canonical_tables.ldaps_test, 16),
    ):
        assert list(train.columns) == list(test.columns)
        assert not train.duplicated(["forecast_kst_dtm", "grid_id"]).any()
        assert not test.duplicated(["forecast_kst_dtm", "grid_id"]).any()
        assert train.groupby("forecast_kst_dtm")["grid_id"].nunique().eq(grids).all()
        assert test.groupby("forecast_kst_dtm")["grid_id"].nunique().eq(grids).all()
        assert (train["data_available_kst_dtm"] < train["forecast_kst_dtm"]).all()
        assert (test["data_available_kst_dtm"] < test["forecast_kst_dtm"]).all()

    assert len(canonical_tables.gfs_train) == 236736
    assert len(canonical_tables.gfs_test) == 78840
    assert len(canonical_tables.ldaps_train) == 420864
    assert len(canonical_tables.ldaps_test) == 140160


def test_actual_canonical_turbine_contract(canonical_tables: CanonicalTables) -> None:
    turbines = canonical_tables.turbines
    assert len(turbines) == 17
    assert turbines["turbine_id"].is_unique
    assert turbines.groupby("group_id").size().to_dict() == {1: 6, 2: 6, 3: 5}
    assert turbines["hub_height_m"].eq(117.0).all()
