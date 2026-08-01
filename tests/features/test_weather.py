import numpy as np
import pandas as pd
import pytest

from baram.features.weather import add_uv_features, aggregate_weather, build_weather_features


def test_uv_to_speed_and_direction_components() -> None:
    """Catches magnitude or direction components computed from the wrong axes."""
    frame = pd.DataFrame({"u": [3.0, 0.0], "v": [4.0, 0.0]})
    out = add_uv_features(frame, "u", "v", "wind10")
    assert out.loc[0, "wind10_speed"] == pytest.approx(5.0)
    assert out.loc[0, "wind10_dir_sin"] == pytest.approx(0.8)
    assert out.loc[0, "wind10_dir_cos"] == pytest.approx(0.6)
    assert out.loc[1, ["wind10_dir_sin", "wind10_dir_cos"]].tolist() == [0.0, 0.0]


def test_weather_aggregation_uses_literal_spatial_statistics() -> None:
    """Catches missing quantiles or accidental raw-angle averaging."""
    timestamp = pd.Timestamp("2023-01-01 01:00:00")
    available = pd.Timestamp("2022-12-31 13:00:00")
    frame = pd.DataFrame(
        {
            "forecast_kst_dtm": [timestamp] * 4,
            "data_available_kst_dtm": [available] * 4,
            "grid_id": [1, 2, 3, 4],
            "latitude": [1.0, 1.0, 2.0, 2.0],
            "longitude": [10.0, 11.0, 10.0, 11.0],
            "u": [3.0, 0.0, -3.0, 0.0],
            "v": [4.0, 5.0, -4.0, -5.0],
            "scalar": [1.0, 2.0, 3.0, 4.0],
        }
    )
    vector = add_uv_features(frame, "u", "v", "wind")
    result = aggregate_weather(vector, "toy")
    assert result.loc[0, "toy__scalar__mean"] == pytest.approx(2.5)
    assert result.loc[0, "toy__scalar__min"] == 1.0
    assert result.loc[0, "toy__scalar__max"] == 4.0
    assert result.loc[0, "toy__scalar__q10"] == pytest.approx(1.3)
    assert result.loc[0, "toy__scalar__q50"] == pytest.approx(2.5)
    assert result.loc[0, "toy__scalar__q90"] == pytest.approx(3.7)
    assert "toy__wind_speed__q90" in result
    assert all("angle" not in name for name in result.columns)


def test_weather_aggregation_counts_missing_cells_without_fragmentation() -> None:
    """Catches vectorized aggregation losing missingness or grid cardinality metadata."""
    frame = pd.DataFrame(
        {
            "forecast_kst_dtm": pd.to_datetime(["2023-01-01 01:00"] * 2),
            "data_available_kst_dtm": pd.to_datetime(["2022-12-31 13:00"] * 2),
            "grid_id": [1, 2],
            "value_a": [1.0, np.nan],
            "value_b": [np.nan, 2.0],
        }
    )
    result = aggregate_weather(frame, "toy")
    assert result.loc[0, "toy__missing_cell_count"] == 2
    assert result.loc[0, "toy__grid_count"] == 2


def test_build_weather_features_replicates_groups_and_uses_operating_calendar() -> None:
    """Catches missing group rows or midnight assigned to the next seasonal day."""
    times = pd.to_datetime(["2023-06-30 23:00:00", "2023-07-01 00:00:00"])
    available = pd.to_datetime(["2023-06-29 13:00:00"] * 2)
    base = pd.DataFrame(
        {
            "forecast_kst_dtm": times,
            "data_available_kst_dtm": available,
            "grid_id": [1, 1],
            "latitude": [37.0, 37.0],
            "longitude": [129.0, 129.0],
            "heightAboveGround_10_10u": [1.0, 2.0],
            "heightAboveGround_10_10v": [2.0, 1.0],
        }
    )
    features = build_weather_features(base, base)
    assert len(features) == 6
    assert features.groupby("forecast_kst_dtm")["group_id"].nunique().eq(3).all()
    midnight = features.loc[features["forecast_kst_dtm"].eq(times[1])]
    assert midnight["hour"].eq(0).all()
    assert midnight["month"].eq(6).all()
    assert midnight["operating_quarter"].eq(2).all()
    assert np.isfinite(features.filter(regex="sin|cos").to_numpy()).all()
