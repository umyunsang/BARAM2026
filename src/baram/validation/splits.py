"""Deterministic issuance-batch development and one-use lockbox split contracts."""

import re

import pandas as pd

from baram.contracts.types import FoldSpec, GroupId
from baram.data.canonical import add_operating_period
from baram.exceptions import LeakageError

_OPERATING_COLUMNS = ("operating_day", "operating_year", "operating_quarter")


def _with_canonical_period(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"forecast_kst_dtm", "data_available_kst_dtm", "grid_id"}
    if not required.issubset(frame.columns):
        missing = sorted(required - set(frame.columns))
        raise LeakageError(f"split input is missing columns: {missing}")
    base = frame.drop(columns=[name for name in _OPERATING_COLUMNS if name in frame]).copy()
    canonical = add_operating_period(base, "forecast_kst_dtm")
    for column in _OPERATING_COLUMNS:
        if column not in frame:
            continue
        observed = frame[column].reset_index(drop=True)
        expected = canonical[column].reset_index(drop=True)
        if column == "operating_day":
            observed = pd.to_datetime(observed, errors="coerce")
        if not observed.equals(expected):
            raise LeakageError(f"conflicting operating period field: {column}")
    return canonical


def _batch_table(frame: pd.DataFrame) -> pd.DataFrame:
    canonical = _with_canonical_period(frame)
    grouped = canonical.groupby("data_available_kst_dtm", sort=True)
    summary = grouped.agg(
        first_forecast=("forecast_kst_dtm", "min"),
        last_forecast=("forecast_kst_dtm", "max"),
        forecast_count=("forecast_kst_dtm", "nunique"),
        operating_day=("operating_day", "first"),
        operating_day_count=("operating_day", "nunique"),
        operating_year=("operating_year", "first"),
        operating_year_count=("operating_year", "nunique"),
        operating_quarter=("operating_quarter", "first"),
        operating_quarter_count=("operating_quarter", "nunique"),
    ).reset_index()
    if (
        summary[["operating_day_count", "operating_year_count", "operating_quarter_count"]]
        .ne(1)
        .any()
        .any()
    ):
        raise LeakageError("issuance batch crosses conflicting operating periods")
    if not summary["forecast_count"].eq(24).all():
        raise LeakageError("every issuance batch must contain exactly 24 forecast timestamps")
    summary["batch_id"] = summary["data_available_kst_dtm"].dt.strftime("%Y%m%d%H")
    return summary.sort_values("data_available_kst_dtm", kind="stable").reset_index(drop=True)


def _fold_for_period(
    batches: pd.DataFrame,
    year: int,
    quarter: int | None,
    groups: tuple[GroupId, ...],
    official_total_eligible: bool,
    is_lockbox: bool = False,
) -> FoldSpec:
    valid_mask = batches["operating_year"].eq(year)
    if quarter is not None:
        valid_mask &= batches["operating_quarter"].eq(quarter)
    valid = batches.loc[valid_mask]
    if valid.empty:
        raise LeakageError(f"split period has no issuance batches: year={year}, quarter={quarter}")
    cutoff = valid["data_available_kst_dtm"].min()
    train = batches.loc[batches["last_forecast"].lt(cutoff)]
    if train.empty:
        raise LeakageError("split period has no fully observable training batches")
    suffix = str(year) if quarter is None else f"{year}-Q{quarter}"
    fold_id = f"lockbox-{suffix}" if is_lockbox else f"dev-{suffix}"
    fold = FoldSpec(
        fold_id=fold_id,
        train_batches=tuple(train["batch_id"]),
        validation_batches=tuple(valid["batch_id"]),
        eligible_groups=groups,
        official_total_eligible=official_total_eligible,
        is_lockbox=is_lockbox,
    )
    validate_fold_spec(fold)
    return fold


def validate_fold_spec(fold: FoldSpec) -> None:
    if not fold.train_batches or not fold.validation_batches:
        raise LeakageError("fold must have nonempty train and validation batches")
    if set(fold.train_batches) & set(fold.validation_batches):
        raise LeakageError("fold has overlapping train and validation batches")
    if len(set(fold.validation_batches)) != len(fold.validation_batches):
        raise LeakageError("fold validation batches are duplicated")
    if fold.official_total_eligible and fold.eligible_groups != (1, 2, 3):
        raise LeakageError("official-total fold requires all three eligible groups")
    match = re.search(r"(20\d{2})", fold.fold_id)
    if fold.official_total_eligible and match and int(match.group(1)) == 2022:
        raise LeakageError("2022 folds cannot be official-total eligible")
    if fold.is_lockbox and not fold.fold_id.startswith("lockbox-"):
        raise LeakageError("lockbox flag requires a lockbox fold ID")


def build_development_folds(frame: pd.DataFrame) -> tuple[FoldSpec, ...]:
    batches = _batch_table(frame)
    return tuple(_fold_for_period(batches, 2023, quarter, (1, 2, 3), True) for quarter in (2, 3, 4))


def build_group12_diagnostic_folds(frame: pd.DataFrame) -> tuple[FoldSpec, ...]:
    batches = _batch_table(frame)
    return tuple(_fold_for_period(batches, 2022, quarter, (1, 2), False) for quarter in (2, 3, 4))


def build_lockbox(frame: pd.DataFrame, year: int = 2024) -> FoldSpec:
    return _fold_for_period(_batch_table(frame), year, None, (1, 2, 3), True, is_lockbox=True)
