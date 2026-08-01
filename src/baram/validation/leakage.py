"""Fail-closed chronological, availability, and feature-schema guards."""

from collections.abc import Iterable

import pandas as pd

from baram.exceptions import LeakageError


def assert_batch_isolation(train: pd.DataFrame, valid: pd.DataFrame) -> None:
    required = "data_available_kst_dtm"
    if required not in train or required not in valid:
        raise LeakageError(f"both frames require {required}")
    overlap = set(train[required]) & set(valid[required])
    if overlap:
        raise LeakageError(f"batch appears in both train and validation: {sorted(overlap)[:3]}")


def assert_label_cutoff(train_labels: pd.DataFrame, valid_weather: pd.DataFrame) -> None:
    if "forecast_kst_dtm" not in train_labels or "data_available_kst_dtm" not in valid_weather:
        raise LeakageError("label cutoff frames are missing required timestamps")
    if train_labels.empty or valid_weather.empty:
        raise LeakageError("label cutoff cannot be checked on an empty frame")
    cutoff = valid_weather["data_available_kst_dtm"].min()
    if train_labels["forecast_kst_dtm"].max() >= cutoff:
        raise LeakageError("training label is not available before validation issuance")


def assert_feature_availability(frame: pd.DataFrame) -> None:
    required = {"forecast_kst_dtm", "data_available_kst_dtm"}
    if not required.issubset(frame.columns):
        raise LeakageError(f"weather availability requires {sorted(required)}")
    if frame[list(required)].isna().any().any():
        raise LeakageError("weather availability timestamps contain missing values")
    if (frame["data_available_kst_dtm"] >= frame["forecast_kst_dtm"]).any():
        raise LeakageError("weather must be available strictly before forecast time")
    publication_columns = [name for name in frame if name.endswith("_published_kst_dtm")]
    for column in publication_columns:
        if (frame[column] > frame["data_available_kst_dtm"]).any():
            raise LeakageError(f"feature was published after the issuance cutoff: {column}")


def assert_inference_schema(columns: Iterable[str]) -> None:
    forbidden_tokens = ("actual", "target", "lag", "scada")
    forbidden = sorted(
        column for column in columns if any(token in column.lower() for token in forbidden_tokens)
    )
    if forbidden:
        raise LeakageError(f"inference schema contains forbidden columns: {forbidden}")


def assert_artifact_fold(expected_fold_id: str, artifact_fold_id: str) -> None:
    if expected_fold_id != artifact_fold_id:
        raise LeakageError(
            f"train-fitted artifact belongs to a different fold: "
            f"expected={expected_fold_id}, observed={artifact_fold_id}"
        )
