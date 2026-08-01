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
    return result
