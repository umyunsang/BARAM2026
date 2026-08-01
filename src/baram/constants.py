"""Competition-wide immutable constants."""

from typing import Final

CAPACITIES_KWH: Final[dict[int, float]] = {1: 21600.0, 2: 21600.0, 3: 21000.0}
GROUP_IDS: Final[tuple[int, ...]] = (1, 2, 3)
SUBMISSION_COLUMNS: Final[tuple[str, ...]] = (
    "forecast_id",
    "forecast_kst_dtm",
    "kpx_group_1",
    "kpx_group_2",
    "kpx_group_3",
)
METRIC_COLUMNS: Final[frozenset[str]] = frozenset(
    {"forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh", "prediction_kwh"}
)
EXPECTED_ARCHIVE_MEMBERS: Final[frozenset[str]] = frozenset(
    {
        "data_description.md",
        "info.xlsx",
        "sample_submission.csv",
        "test/",
        "test/gfs_test.csv",
        "test/ldaps_test.csv",
        "train/",
        "train/gfs_train.csv",
        "train/ldaps_train.csv",
        "train/scada_unison_train.csv",
        "train/scada_vestas_train.csv",
        "train/train_labels.csv",
    }
)
CSV_MEMBERS: Final[frozenset[str]] = frozenset(
    member for member in EXPECTED_ARCHIVE_MEMBERS if member.endswith(".csv")
)
OPEN_ZIP_SHA256: Final[str] = "920be0c458d820e855bf79dd25723146f52ce1736138aedba5e6bc853f1f720b"
BASELINE_NOTEBOOK_SHA256: Final[str] = (
    "712b26f4d2748860c94cff1e0100c23810468c983173f8e9ef8d009fe82df48c"
)
TIMEZONE: Final[str] = "Asia/Seoul"
