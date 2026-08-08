"""Screen daily 24-hour by three-group multi-output models on one later fold."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from run_sequence_classifier import (
    BASELINE,
    BASELINE_SHA,
    CAPACITIES,
    OPEN,
    OPEN_SHA,
    OUTPUT,
    _score,
    _sha256,
    _surface,
)
from run_site_wind_classifier import FOLDS, _add_site_wind_features
from run_site_wind_teacher import _validation_mask
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PARENT_PATH = OUTPUT / "M107_STRICT_TEMPORAL_TOP100-oof.parquet"
SITEWIND_ID = "M64B_ALLWEATHER_SITEWIND_CLASS"
BASE_COLUMNS = ["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh"]
GROUPS = tuple(CAPACITIES)
LEADS = tuple(range(12, 36))
PARENT_WEIGHT_GRID = tuple(round(value, 2) for value in np.linspace(0.0, 1.0, 21))


@dataclass(frozen=True)
class Spec:
    family: str
    feature_count: int
    parameter: float

    @property
    def name(self) -> str:
        return f"{self.family}_f{self.feature_count}_p{self.parameter:g}"


SPECS = (
    *(Spec("ridge", count, alpha) for count in (10, 20, 40) for alpha in (10.0, 100.0, 1000.0)),
    Spec("pls", 10, 8.0),
    Spec("pls", 20, 8.0),
    Spec("pls", 20, 16.0),
    Spec("extra", 20, 2.0),
    Spec("extra", 20, 4.0),
    Spec("extra", 40, 4.0),
    Spec("forest", 20, 4.0),
)


def _selected_features(fold_id: str, count: int) -> list[str]:
    receipt = json.loads((OUTPUT / f"M102_TOP100-{fold_id}.json").read_text())
    names = receipt["selected_feature_names"][:count]
    if len(names) != count or len(set(names)) != count:
        raise RuntimeError(f"M102 feature contract changed for {fold_id}")
    return names


def _model(spec: Spec) -> object:
    if spec.family == "ridge":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            Ridge(alpha=spec.parameter),
        )
    if spec.family == "pls":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            PLSRegression(n_components=int(spec.parameter), scale=False, max_iter=500),
        )
    if spec.family == "extra":
        return ExtraTreesRegressor(
            n_estimators=400,
            min_samples_leaf=int(spec.parameter),
            max_features=0.7,
            random_state=20260802,
            n_jobs=6,
        )
    if spec.family == "forest":
        return RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=int(spec.parameter),
            max_features=0.7,
            max_samples=0.9,
            random_state=20260802,
            n_jobs=6,
        )
    raise ValueError(f"unknown family: {spec.family}")


def _daily_arrays(
    surface: pd.DataFrame,
    matrix: pd.DataFrame,
    features: list[str],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    ordered = surface[
        [
            "forecast_id",
            "forecast_kst_dtm",
            "data_available_kst_dtm",
            "group_id",
            "lead_hour",
            "actual_kwh",
        ]
    ].join(matrix[features])
    ordered = ordered.sort_values(
        ["data_available_kst_dtm", "group_id", "lead_hour"]
    ).reset_index(drop=True)
    expected_rows = len(GROUPS) * len(LEADS)
    sizes = ordered.groupby("data_available_kst_dtm", sort=True).size()
    if not sizes.eq(expected_rows).all():
        raise RuntimeError("daily issuance row contract changed")
    pattern = ordered.groupby("data_available_kst_dtm", sort=True)[
        ["group_id", "lead_hour"]
    ].apply(lambda part: tuple(map(tuple, part.to_numpy())))
    expected_pattern = tuple((group_id, lead) for group_id in GROUPS for lead in LEADS)
    if not pattern.map(lambda value: value == expected_pattern).all():
        raise RuntimeError("daily group/lead order contract changed")

    days = ordered["data_available_kst_dtm"].drop_duplicates().reset_index(drop=True)
    x = ordered[features].to_numpy(dtype="float32").reshape(len(days), -1)
    capacity = ordered["group_id"].map(CAPACITIES).to_numpy(dtype=float)
    y = (ordered["actual_kwh"].to_numpy(dtype=float) / capacity).reshape(len(days), -1)
    return ordered, x, y


def _prediction_frame(
    ordered: pd.DataFrame,
    day_mask: np.ndarray,
    normalized: np.ndarray,
) -> pd.DataFrame:
    rows_per_day = len(GROUPS) * len(LEADS)
    row_mask = np.repeat(day_mask, rows_per_day)
    base = ordered.loc[row_mask, BASE_COLUMNS].copy()
    flat = np.asarray(normalized, dtype=float).reshape(-1)
    if len(flat) != len(base):
        raise RuntimeError("multi-output prediction shape mismatch")
    base["prediction_kwh"] = (
        np.clip(flat, 0.0, 1.075)
        * base["group_id"].map(CAPACITIES).to_numpy(dtype=float)
    )
    return base


def _group_total(frame: pd.DataFrame, group_id: int) -> float:
    part = frame.loc[frame["group_id"].eq(group_id)]
    capacity = CAPACITIES[group_id]
    actual = part["actual_kwh"].to_numpy(dtype=float) / capacity
    prediction = part["prediction_kwh"].to_numpy(dtype=float) / capacity
    valid = np.isfinite(actual) & (actual >= 0.10)
    actual = actual[valid]
    error = np.abs(prediction[valid] - actual)
    units = np.select([error <= 0.06, error <= 0.08], [4.0, 3.0], default=0.0)
    return 0.5 * (
        1.0 - float(error.mean())
        + float((actual * units).sum() / (4.0 * actual.sum()))
    )


def _screen_blend(
    direct: pd.DataFrame,
    parent: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    keys = ["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh"]
    work = direct.merge(
        parent[[*keys, "prediction_kwh"]].rename(
            columns={"prediction_kwh": "parent_prediction_kwh"}
        ),
        on=keys,
        validate="one_to_one",
    )
    best_policies: dict[int, tuple[float, bool]] = {}
    diagnostics: dict[str, object] = {}
    for group_id in GROUPS:
        group_best: tuple[float, float, bool] | None = None
        for parent_weight in PARENT_WEIGHT_GRID:
            normalized = (
                parent_weight * work["parent_prediction_kwh"]
                + (1.0 - parent_weight) * work["prediction_kwh"]
            ) / work["group_id"].map(CAPACITIES)
            for snap in (False, True):
                candidate = work[keys].copy()
                value = normalized
                if snap:
                    value = np.round(value / 0.0025) * 0.0025
                candidate["prediction_kwh"] = (
                    np.clip(value, 0.0, 1.075)
                    * candidate["group_id"].map(CAPACITIES).to_numpy(dtype=float)
                )
                total = _group_total(candidate, group_id)
                choice = (total, parent_weight, snap)
                if group_best is None or choice[0] > group_best[0]:
                    group_best = choice
        assert group_best is not None
        best_policies[group_id] = (group_best[1], group_best[2])
        diagnostics[str(group_id)] = {
            "group_total": group_best[0],
            "parent_weight": group_best[1],
            "snap": group_best[2],
        }

    output = work[keys].copy()
    normalized = np.empty(len(work), dtype=float)
    for group_id, (parent_weight, snap) in best_policies.items():
        apply = work["group_id"].eq(group_id).to_numpy()
        capacity = CAPACITIES[group_id]
        value = (
            parent_weight * work.loc[apply, "parent_prediction_kwh"].to_numpy(dtype=float)
            + (1.0 - parent_weight)
            * work.loc[apply, "prediction_kwh"].to_numpy(dtype=float)
        ) / capacity
        if snap:
            value = np.round(value / 0.0025) * 0.0025
        normalized[apply] = value
    output["prediction_kwh"] = (
        np.clip(normalized, 0.0, 1.075)
        * output["group_id"].map(CAPACITIES).to_numpy(dtype=float)
    )
    return output, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-id", default="M121_DAILY_MULTIOUTPUT")
    parser.add_argument("--fold", choices=FOLDS, default=FOLDS[-1])
    args = parser.parse_args()
    if _sha256(OPEN) != OPEN_SHA or _sha256(BASELINE) != BASELINE_SHA:
        raise RuntimeError("immutable competition input hash mismatch")
    started = time.perf_counter()
    surface, _, _ = _surface()
    cached = np.load(OUTPUT / f"{SITEWIND_ID}-{args.fold}-sitewind-features.npz")
    all_features = _selected_features(args.fold, 100)
    base_features = [name for name in all_features if not name.startswith("sitewind__")]
    matrix = surface[base_features].copy()
    _add_site_wind_features(matrix, cached["legacy"], cached["allweather"])
    missing = set(all_features).difference(matrix.columns)
    if missing:
        raise RuntimeError(f"missing daily features: {sorted(missing)}")

    parent = pd.read_parquet(PARENT_PATH)
    parent = parent.loc[parent["fold_id"].eq(args.fold)]
    validation_rows = _validation_mask(surface, args.fold)
    validation_start = pd.Timestamp(
        surface.loc[validation_rows, "forecast_kst_dtm"].min()
    )
    validation_issuance_start = validation_start - pd.Timedelta(hours=12)
    validation_issuance_end = pd.Timestamp(
        parent["forecast_kst_dtm"].max()
    ) - pd.Timedelta(hours=11)
    best: tuple[float, Spec, pd.DataFrame, dict[str, object]] | None = None
    sweep: dict[str, object] = {}
    for spec in SPECS:
        features = _selected_features(args.fold, spec.feature_count)
        ordered, x, y = _daily_arrays(surface, matrix, features)
        days = ordered["data_available_kst_dtm"].drop_duplicates().reset_index(drop=True)
        train_days = days.lt(validation_issuance_start).to_numpy()
        valid_days = days.ge(validation_issuance_start).to_numpy() & days.lt(
            validation_issuance_end
        ).to_numpy()
        if int(valid_days.sum()) not in {91, 92}:
            raise RuntimeError(f"unexpected validation day count: {int(valid_days.sum())}")
        prediction = np.empty((int(valid_days.sum()), len(GROUPS) * len(LEADS)))
        training_day_counts: dict[str, int] = {}
        for group_position, group_id in enumerate(GROUPS):
            target_slice = slice(
                group_position * len(LEADS),
                (group_position + 1) * len(LEADS),
            )
            group_target = y[:, target_slice]
            group_train = train_days & np.isfinite(group_target).all(axis=1)
            training_day_counts[str(group_id)] = int(group_train.sum())
            model = _model(spec)
            daily_weight = np.clip(group_target[group_train].mean(axis=1), 0.10, None)
            if spec.family in {"ridge", "extra", "forest"}:
                fit_parameters = (
                    {"ridge__sample_weight": daily_weight}
                    if spec.family == "ridge"
                    else {"sample_weight": daily_weight}
                )
                model.fit(
                    x[group_train],
                    group_target[group_train],
                    **fit_parameters,
                )
            else:
                model.fit(x[group_train], group_target[group_train])
            prediction[:, target_slice] = model.predict(x[valid_days])
        prediction = np.clip(prediction, 0.0, 1.075)
        direct = _prediction_frame(ordered, valid_days, prediction)
        blended, policies = _screen_blend(direct, parent)
        raw_score = _score(direct)
        blend_score = _score(blended)
        sweep[spec.name] = {
            "spec": spec.__dict__,
            "training_day_counts": training_day_counts,
            "raw_score": raw_score,
            "oracle_blend_score": blend_score,
            "oracle_group_policies": policies,
        }
        choice = (blend_score["total"], spec, blended, policies)
        if best is None or choice[0] > best[0]:
            best = choice
        print(json.dumps({"spec": spec.name, **sweep[spec.name]}), flush=True)
    assert best is not None
    output = best[2]
    output["fold_id"] = args.fold
    output["model_id"] = args.candidate_id
    output_path = OUTPUT / f"{args.candidate_id}-{args.fold}.parquet"
    output.to_parquet(output_path, index=False)
    receipt = {
        "schema_version": 1,
        "candidate_id": args.candidate_id,
        "fold_id": args.fold,
        "scope": "daily multi-output representation screen; oracle blend not promoted",
        "selected_spec": best[1].__dict__,
        "selected_oracle_blend_score": _score(output),
        "selected_oracle_group_policies": best[3],
        "sweep": sweep,
        "runtime_seconds": round(time.perf_counter() - started, 2),
        "prediction_path": str(output_path.relative_to(Path.cwd())),
        "prediction_sha256": _sha256(output_path),
        "observed_scada_feature_count": 0,
        "new_2024_evaluation": False,
        "lockbox_reopened": False,
        "external_actions": [],
    }
    receipt_path = OUTPUT / f"{args.candidate_id}-{args.fold}.json"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
