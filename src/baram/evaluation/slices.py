"""Diagnostic-only error slices backed by the exact group evaluator."""

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from baram.data.canonical import add_operating_period
from baram.evaluation.official import evaluate_group_component
from baram.exceptions import ContractError

_METRIC_COLUMNS = [
    "forecast_id",
    "forecast_kst_dtm",
    "group_id",
    "actual_kwh",
    "prediction_kwh",
]


def attach_diagnostic_bins(
    frame: pd.DataFrame,
    capacities: Mapping[int, float],
) -> pd.DataFrame:
    required = set(_METRIC_COLUMNS)
    if not required.issubset(frame.columns):
        raise ContractError(f"slice frame is missing: {sorted(required - set(frame))}")
    result = frame.copy()
    result["forecast_kst_dtm"] = pd.to_datetime(result["forecast_kst_dtm"], errors="raise")
    if "operating_day" not in result:
        result = add_operating_period(result, "forecast_kst_dtm")
    else:
        result["operating_day"] = pd.to_datetime(result["operating_day"], errors="raise")
    result["all"] = "all"
    result["operating_month"] = result["operating_day"].dt.month.astype("int8")
    result["operating_season"] = result["operating_month"].map(
        {
            12: "winter",
            1: "winter",
            2: "winter",
            3: "spring",
            4: "spring",
            5: "spring",
            6: "summer",
            7: "summer",
            8: "summer",
            9: "autumn",
            10: "autumn",
            11: "autumn",
        }
    )
    result["target_hour"] = result["forecast_kst_dtm"].dt.hour.astype("int8")
    if "lead_hour" in result:
        result["lead_bin"] = pd.cut(
            result["lead_hour"],
            bins=[-np.inf, 17, 23, 29, np.inf],
            labels=["12-17", "18-23", "24-29", "30-35"],
        ).astype(str)
    if "nwp_missing_count" in result:
        result["nwp_missing_state"] = np.where(
            result["nwp_missing_count"].gt(0), "missing", "complete"
        )
    if "wind_speed" in result:
        result["wind_speed_bin"] = pd.cut(
            result["wind_speed"],
            bins=[-np.inf, 3, 8, 12, 20, np.inf],
            labels=["calm", "low", "medium", "high", "extreme"],
        ).astype(str)
    capacity = result["group_id"].map(capacities).astype(float)
    generation_fraction = result["actual_kwh"] / capacity
    result["actual_generation_bin"] = pd.cut(
        generation_fraction,
        bins=[-np.inf, 0.1, 0.3, 0.6, 0.9, np.inf],
        labels=["below_eval", "low", "medium", "high", "near_cap"],
        right=False,
    ).astype(str)
    ordered = result.sort_values(["group_id", "forecast_kst_dtm"], kind="stable")
    ramp_fraction = ordered.groupby("group_id", sort=False)["actual_kwh"].diff().abs() / ordered[
        "group_id"
    ].map(capacities).astype(float)
    ordered["ramp_flag"] = ramp_fraction.ge(0.1).fillna(False)
    error_rate = (ordered["prediction_kwh"] - ordered["actual_kwh"]).abs() / ordered[
        "group_id"
    ].map(capacities).astype(float)
    ordered["settlement_tier"] = np.select(
        [error_rate <= 0.06, error_rate <= 0.08],
        ["unit_4", "unit_3"],
        default="unit_0",
    )
    return ordered.sort_index()


def score_diagnostic_slices(
    frame: pd.DataFrame,
    capacities: Mapping[int, float],
    slice_columns: Sequence[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for slice_name in slice_columns:
        if slice_name not in frame:
            raise ContractError(f"slice column does not exist: {slice_name}")
        for slice_value, sliced in frame.groupby(slice_name, sort=True, dropna=False):
            for group in (1, 2, 3):
                part = sliced.loc[sliced["group_id"].eq(group), _METRIC_COLUMNS]
                if part.empty or part["actual_kwh"].lt(0.1 * capacities[group]).all():
                    continue
                score = evaluate_group_component(part, group, capacities[group])
                rows.append(
                    {
                        "diagnostic_only": True,
                        "slice_name": slice_name,
                        "slice_value": str(slice_value),
                        "group_id": group,
                        "nmae": score.nmae,
                        "ficr": score.ficr,
                        "valid_rows": score.valid_rows,
                    }
                )
    return (
        pd.DataFrame(rows)
        .sort_values(["slice_name", "slice_value", "group_id"], kind="stable")
        .reset_index(drop=True)
    )
