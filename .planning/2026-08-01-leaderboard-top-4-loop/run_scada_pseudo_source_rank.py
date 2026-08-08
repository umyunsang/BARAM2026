"""Run the source-rank screen with direct supplied-SCADA group-3 pseudo labels.

The official group-3 target is absent in 2022, while the supplied Unison SCADA
contains the five turbine power measurements.  This runner aggregates only
complete six-sample turbine hours, calibrates the aggregate against official
group-3 targets available strictly before the validation fold, and uses the
result as a training target.  SCADA never enters the inference feature matrix.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import run_pseudo_source_rank as runner
from run_sequence_classifier import CAPACITIES
from run_turbine_decomposition import _hourly_turbine_targets

GROUP_ID = 3
TURBINE_COUNT = 5


def _scada_pseudo_targets(
    surface: pd.DataFrame,
    _matrix: pd.DataFrame,
    preceding: np.ndarray,
    _mapper_profile: str,
    _mapper_blend: float,
) -> tuple[pd.Series, pd.Series, dict[str, object]]:
    turbine = _hourly_turbine_targets()
    group = turbine.loc[turbine["group_id"].eq(GROUP_ID)]
    hourly = group.groupby("forecast_kst_dtm", sort=True).agg(
        scada_kwh=("turbine_kwh", "sum"),
        turbine_count=("turbine_kwh", "count"),
    )
    hourly = hourly.loc[hourly["turbine_count"].eq(TURBINE_COUNT)]

    row_group = surface["group_id"].eq(GROUP_ID).to_numpy()
    row_scada = surface["forecast_kst_dtm"].map(hourly["scada_kwh"])
    observed = (
        preceding
        & row_group
        & surface["actual_kwh"].notna().to_numpy()
        & row_scada.notna().to_numpy()
        & row_scada.gt(0.0).to_numpy()
    )
    if int(observed.sum()) < 500:
        raise RuntimeError("too few preceding SCADA-to-label calibration rows")

    actual = surface.loc[observed, "actual_kwh"].to_numpy(dtype=float)
    scada = row_scada.loc[observed].to_numpy(dtype=float)
    ratio = actual / scada
    central = ratio[(ratio >= 0.85) & (ratio <= 1.15)]
    if len(central) < 0.90 * len(ratio):
        raise RuntimeError("SCADA-to-label calibration has too many outliers")
    scale = float(np.median(central))
    calibrated_observed = np.clip(scale * scada, 0.0, CAPACITIES[GROUP_ID] * 1.075)
    normalized_error = np.abs(calibrated_observed - actual) / CAPACITIES[GROUP_ID]

    pseudo_rows = (
        preceding
        & row_group
        & surface["actual_kwh"].isna().to_numpy()
        & row_scada.notna().to_numpy()
    )
    values = pd.Series(np.nan, index=surface.index, dtype=float)
    values.loc[pseudo_rows] = np.clip(
        scale
        * row_scada.loc[pseudo_rows].to_numpy(dtype=float)
        / CAPACITIES[GROUP_ID],
        0.0,
        1.075,
    )
    if int(values.notna().sum()) < 500:
        raise RuntimeError(
            "supplied Unison SCADA does not overlap enough missing group-3 labels"
        )
    confidence = pd.Series(1.0, index=surface.index, dtype=float)
    confidence.loc[pseudo_rows] = float(
        np.clip(1.0 - np.median(normalized_error) / 0.08, 0.50, 1.0)
    )
    correlation = float(np.corrcoef(actual, scada)[0, 1])
    return values, confidence, {
        "mapper_profile": "direct_supplied_scada_aggregate",
        "mapper_blend": 1.0,
        "mapper_feature_count": 0,
        "mapper_weather_source_count": 0,
        "mapper_observed_rows": int(observed.sum()),
        "pseudo_rows": int(values.notna().sum()),
        "pseudo_valid_rows": int(values.ge(0.10).sum()),
        "pseudo_mean": float(values.mean()),
        "rich_pseudo_mean": float(values.mean()),
        "scada_scale": scale,
        "scada_label_correlation": correlation,
        "scada_label_mae_capacity": float(np.mean(normalized_error)),
        "scada_label_median_ae_capacity": float(np.median(normalized_error)),
        "scada_complete_turbine_count": TURBINE_COUNT,
        "pseudo_confidence_min": float(confidence.loc[pseudo_rows].min()),
        "pseudo_confidence_mean": float(confidence.loc[pseudo_rows].mean()),
        "pseudo_confidence_max": float(confidence.loc[pseudo_rows].max()),
        "chronology_contract": (
            "calibration and pseudo targets use only timestamps preceding validation"
        ),
        "inference_scada_feature_count": 0,
    }


if __name__ == "__main__":
    runner._pseudo_targets = _scada_pseudo_targets
    runner.main()
