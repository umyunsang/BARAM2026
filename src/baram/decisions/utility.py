"""Conservative residual-utility policies with a hard support threshold."""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from baram.contracts.hashing import canonical_sha256
from baram.contracts.types import ResidualUtilityPolicy
from baram.exceptions import ContractError

_ALLOWED_SHIFTS = np.array([-0.02, -0.01, 0.0, 0.01, 0.02], dtype=float)


@dataclass(frozen=True)
class UtilityFitDecision:
    status: str
    policy: ResidualUtilityPolicy | None
    reasons: tuple[str, ...]


def _frame_hash(frame: pd.DataFrame) -> str:
    columns = sorted(frame.columns)
    serializable = frame[columns].copy()
    for column in serializable.select_dtypes(include=["datetime", "datetimetz"]):
        serializable[column] = serializable[column].dt.strftime("%Y-%m-%dT%H:%M:%S.%f")
    serializable = serializable.sort_values(columns, kind="stable").reset_index(drop=True)
    return canonical_sha256(serializable.to_dict(orient="records"))


def fit_residual_utility(
    frame: pd.DataFrame,
    capacities: Mapping[int, float],
    training_rows_sha256: str,
    metric_sha256: str | None = None,
    *,
    min_residuals_per_group: int = 500,
) -> UtilityFitDecision:
    """Activate discrete point shifts only with enough OOF support per group."""
    required = {"group_id", "actual_kwh", "prediction_kwh", "lead_bin", "wind_speed_bin"}
    missing = required - set(frame.columns)
    if missing:
        raise ContractError(f"residual utility frame is missing: {sorted(missing)}")
    if min_residuals_per_group < 500:
        raise ContractError("residual utility support floor cannot be below 500")
    if set(capacities) != {1, 2, 3} or any(
        not np.isfinite(value) or value <= 0.0 for value in capacities.values()
    ):
        raise ContractError("residual utility requires positive capacities for all groups")
    values = frame[["actual_kwh", "prediction_kwh"]].to_numpy(dtype=float)
    if frame.empty or not np.isfinite(values).all():
        raise ContractError("residual utility inputs must be nonempty and finite")

    counts = frame.groupby("group_id", observed=True).size().to_dict()
    unsupported = tuple(
        f"group_{group}_rows={int(counts.get(group, 0))}<{min_residuals_per_group}"
        for group in (1, 2, 3)
        if counts.get(group, 0) < min_residuals_per_group
    )
    if unsupported:
        return UtilityFitDecision("NOT_ACTIVATED", None, unsupported)

    shifts: dict[str, float] = {}
    for (group_id, lead_bin, wind_speed_bin), part in frame.groupby(
        ["group_id", "lead_bin", "wind_speed_bin"], observed=True, sort=True
    ):
        normalized_residual = np.median(
            (part["actual_kwh"] - part["prediction_kwh"]).to_numpy(dtype=float)
            / capacities[int(group_id)]
        )
        shift = float(_ALLOWED_SHIFTS[np.abs(_ALLOWED_SHIFTS - normalized_residual).argmin()])
        shifts[f"group={int(group_id)}|lead={lead_bin}|wind={wind_speed_bin}"] = shift

    input_hash = _frame_hash(frame)
    policy = ResidualUtilityPolicy(
        shifts_by_state=shifts,
        min_residuals_per_group=min_residuals_per_group,
        training_rows_sha256=training_rows_sha256,
        input_prediction_hashes={"residual_oof": input_hash},
        metric_sha256=metric_sha256
        or canonical_sha256({"metric": "official_1_nmae_ficr", "version": 1}),
    )
    return UtilityFitDecision("ACTIVATED", policy, ())
