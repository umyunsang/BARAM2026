import numpy as np
import pandas as pd
import pytest

from baram.exceptions import DataQualityError
from baram.features.physics import add_physics_features, dry_air_density, hub_height_speed


def test_power_law_hub_speed() -> None:
    """Catches a wrong height ratio or exponent."""
    assert hub_height_speed(10.0, 100.0, 117.0, 0.2) == pytest.approx(10.0 * (117.0 / 100.0) ** 0.2)


def test_air_density_standard_conditions() -> None:
    """Catches wrong units in the ideal-gas density proxy."""
    assert dry_air_density(101325.0, 288.15) == pytest.approx(1.225, rel=0.01)


@pytest.mark.parametrize(
    "args",
    [(-1.0, 100.0, 117.0, 0.2), (1.0, 0.0, 117.0, 0.2)],
)
def test_hub_speed_rejects_nonphysical_inputs(args: tuple[float, ...]) -> None:
    """Catches a nonphysical speed or height reaching power proxies."""
    with pytest.raises(DataQualityError):
        hub_height_speed(*args)


def test_add_physics_features_emits_finite_rho_v3_without_label_clipping() -> None:
    """Catches unstable physics proxies or accidental target mutation."""
    frame = pd.DataFrame(
        {
            "gfs__wind100_speed__mean": [10.0],
            "gfs__wind80_speed__mean": [9.0],
            "gfs__surface_0_sp__mean": [101325.0],
            "gfs__heightAboveGround_2_2t__mean": [288.15],
            "actual_kwh": [21130.0],
        }
    )
    result = add_physics_features(frame)
    assert np.isfinite(result["phys__rho_v3"]).all()
    assert result.loc[0, "phys__speed_shear_100_80"] == pytest.approx(1.0)
    assert result.loc[0, "actual_kwh"] == 21130.0


def test_spatial_physics_uses_bounded_shear_and_fleet_proxy() -> None:
    frame = pd.DataFrame(
        {
            "gfs__wind100_speed__mean": [10.0, 10.0],
            "gfs__wind80_speed__mean": [9.0, 9.0],
            "gfs__surface_0_sp__mean": [101325.0, 101325.0],
            "gfs__heightAboveGround_2_2t__mean": [288.15, 288.15],
            "gfs_spatial__idw__wind100_speed": [10.0, 0.0],
            "gfs_spatial__idw__wind80_speed": [9.0, 0.0],
            "gfs_spatial__idw__surface_0_sp": [101325.0, 101325.0],
            "gfs_spatial__idw__heightAboveGround_2_2t": [288.15, 288.15],
            "fleet_swept_area_m2": [1000.0, 1000.0],
        }
    )
    result = add_physics_features(frame)
    assert result.loc[0, "phys_v2__shear_alpha_100_80"] == pytest.approx(
        np.log(10.0 / 9.0) / np.log(100.0 / 80.0)
    )
    assert result.loc[1, "phys_v2__shear_fallback"] == 1
    assert result.loc[1, "phys_v2__shear_alpha_100_80"] == pytest.approx(0.2)
    assert result.loc[0, "phys_v2__fleet_power_proxy_w"] > 0.0
