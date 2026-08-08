"""Site-agnostic physical proxy features with explicit units and guards."""

import numpy as np
import pandas as pd

from baram.exceptions import DataQualityError


def hub_height_speed(
    v_ref: float,
    z_ref: float,
    z_hub: float,
    alpha: float,
) -> float:
    if (
        v_ref < 0.0
        or min(z_ref, z_hub) <= 0.0
        or not np.isfinite([v_ref, z_ref, z_hub, alpha]).all()
    ):
        raise DataQualityError("wind speed and heights must be physically valid")
    return v_ref * (z_hub / z_ref) ** alpha


def dry_air_density(pressure_pa: float, temperature_k: float) -> float:
    if (
        pressure_pa <= 0.0
        or temperature_k <= 0.0
        or not np.isfinite([pressure_pa, temperature_k]).all()
    ):
        raise DataQualityError("pressure and temperature must be positive and finite")
    return pressure_pa / (287.05 * temperature_k)


def add_physics_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    required = {
        "gfs__wind100_speed__mean",
        "gfs__wind80_speed__mean",
        "gfs__surface_0_sp__mean",
        "gfs__heightAboveGround_2_2t__mean",
    }
    missing = sorted(required - set(result.columns))
    if missing:
        raise DataQualityError(f"physics feature inputs are missing: {missing}")
    speed100 = result["gfs__wind100_speed__mean"].astype(float)
    speed80 = result["gfs__wind80_speed__mean"].astype(float)
    pressure = result["gfs__surface_0_sp__mean"].astype(float)
    temperature = result["gfs__heightAboveGround_2_2t__mean"].astype(float)
    valid = (speed100 >= 0) & (speed80 >= 0) & (pressure > 0) & (temperature > 0)
    result["phys__input_missing"] = (~valid | speed100.isna() | pressure.isna()).astype("int8")
    result["phys__hub117_speed"] = speed100 * (117.0 / 100.0) ** 0.2
    result["phys__speed_shear_100_80"] = speed100 - speed80
    result["phys__air_density"] = pressure / (287.05 * temperature)
    result["phys__rho_v3"] = result["phys__air_density"] * result["phys__hub117_speed"].pow(3)
    finite_inputs = result[list(required)].dropna().to_numpy(dtype=float)
    if finite_inputs.size and not np.isfinite(finite_inputs).all():
        raise DataQualityError("physics inputs contain non-finite values")
    spatial_required = {
        "gfs_spatial__idw__wind100_speed",
        "gfs_spatial__idw__wind80_speed",
        "gfs_spatial__idw__surface_0_sp",
        "gfs_spatial__idw__heightAboveGround_2_2t",
        "fleet_swept_area_m2",
    }
    if spatial_required.issubset(result.columns):
        spatial100 = result["gfs_spatial__idw__wind100_speed"].astype(float)
        spatial80 = result["gfs_spatial__idw__wind80_speed"].astype(float)
        spatial_pressure = result["gfs_spatial__idw__surface_0_sp"].astype(float)
        spatial_temperature = result[
            "gfs_spatial__idw__heightAboveGround_2_2t"
        ].astype(float)
        shear_valid = spatial100.gt(0.1) & spatial80.gt(0.1)
        raw_alpha = np.log(spatial100 / spatial80) / np.log(100.0 / 80.0)
        alpha = raw_alpha.where(shear_valid, 0.2).clip(-0.2, 0.6).fillna(0.2)
        physical_valid = (
            shear_valid
            & spatial_pressure.gt(0.0)
            & spatial_temperature.gt(0.0)
            & np.isfinite(
                np.column_stack(
                    [spatial100, spatial80, spatial_pressure, spatial_temperature]
                )
            ).all(axis=1)
        )
        result["phys_v2__shear_fallback"] = (~shear_valid).astype("int8")
        result["phys_v2__input_invalid"] = (~physical_valid).astype("int8")
        result["phys_v2__shear_alpha_100_80"] = alpha
        result["phys_v2__hub117_speed"] = spatial100 * (117.0 / 100.0) ** alpha
        result["phys_v2__air_density"] = spatial_pressure / (
            287.05 * spatial_temperature
        )
        result["phys_v2__rho_v3"] = result["phys_v2__air_density"] * result[
            "phys_v2__hub117_speed"
        ].pow(3)
        result["phys_v2__fleet_power_proxy_w"] = (
            0.5
            * result["phys_v2__rho_v3"]
            * result["fleet_swept_area_m2"].astype(float)
        )
    return result
