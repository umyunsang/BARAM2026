import pandas as pd
import pytest

from baram.contracts.hashing import canonical_sha256
from baram.contracts.types import FoldSpec
from baram.data.canonical import CanonicalTables
from baram.exceptions import LeakageError
from baram.validation.splits import (
    build_development_folds,
    build_group12_diagnostic_folds,
    build_lockbox,
    validate_fold_spec,
)


def test_actual_development_folds_use_whole_operating_quarters(
    canonical_tables: CanonicalTables,
) -> None:
    """Catches wrong quarter selection or fragmented issuance batches."""
    folds = build_development_folds(canonical_tables.gfs_train)
    assert [fold.fold_id for fold in folds] == ["dev-2023-Q2", "dev-2023-Q3", "dev-2023-Q4"]
    assert [len(fold.validation_batches) for fold in folds] == [91, 92, 92]
    assert all(fold.eligible_groups == (1, 2, 3) for fold in folds)
    assert all(fold.official_total_eligible and not fold.is_lockbox for fold in folds)
    assert all(set(fold.train_batches).isdisjoint(fold.validation_batches) for fold in folds)


def test_actual_group12_folds_are_diagnostic_only(canonical_tables: CanonicalTables) -> None:
    """Catches a two-group fold being mislabeled as an official total."""
    folds = build_group12_diagnostic_folds(canonical_tables.gfs_train)
    assert [len(fold.validation_batches) for fold in folds] == [91, 92, 92]
    assert all(fold.eligible_groups == (1, 2) for fold in folds)
    assert all(not fold.official_total_eligible for fold in folds)


def test_actual_lockbox_is_one_operating_year(canonical_tables: CanonicalTables) -> None:
    """Catches a calendar-year split dropping the 2025-01-01 midnight row."""
    fold = build_lockbox(canonical_tables.gfs_train, 2024)
    assert fold.fold_id == "lockbox-2024"
    assert len(fold.validation_batches) == 366
    assert fold.is_lockbox and fold.official_total_eligible
    lock_rows = canonical_tables.gfs_train.loc[
        canonical_tables.gfs_train["issuance_batch"].isin(fold.validation_batches)
    ]
    assert lock_rows["forecast_kst_dtm"].nunique() == 8784
    assert pd.Timestamp("2025-01-01 00:00:00") in set(lock_rows["forecast_kst_dtm"])


def test_split_manifest_hash_is_stable(canonical_tables: CanonicalTables) -> None:
    """Catches nondeterministic split ordering across identical runs."""
    first = build_development_folds(canonical_tables.gfs_train)
    second = build_development_folds(canonical_tables.gfs_train.sample(frac=1, random_state=7))
    assert canonical_sha256(first) == canonical_sha256(second)


def test_conflicting_unshifted_period_fields_fail() -> None:
    """Catches deriving a midnight quarter from its calendar timestamp."""
    frame = pd.DataFrame(
        {
            "forecast_kst_dtm": pd.to_datetime(["2023-07-01 00:00:00"]),
            "data_available_kst_dtm": pd.to_datetime(["2023-06-29 13:00:00"]),
            "grid_id": [1],
            "operating_day": pd.to_datetime(["2023-07-01"]),
            "operating_year": [2023],
            "operating_quarter": [3],
        }
    )
    with pytest.raises(LeakageError, match="conflicting operating"):
        build_development_folds(frame)


def test_fold_spec_rejects_invalid_official_scope() -> None:
    """Catches promotion of 2022 or a partial group set to official total."""
    partial = FoldSpec("dev-2023-Q2", ("a",), ("b",), (1, 2), True)
    with pytest.raises(LeakageError, match="all three"):
        validate_fold_spec(partial)
    year_2022 = FoldSpec("dev-2022-Q2", ("a",), ("b",), (1, 2, 3), True)
    with pytest.raises(LeakageError, match="2022"):
        validate_fold_spec(year_2022)
