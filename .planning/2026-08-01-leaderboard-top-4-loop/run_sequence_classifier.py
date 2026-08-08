"""Screen a metric-aware classifier with within-issuance NWP sequence context."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import zipfile
from pathlib import Path

import lightgbm
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import KFold

from baram.config import load_config
from baram.evaluation.official import evaluate_official
from baram.features.sequence import add_issuance_sequence_context

REPO = Path(__file__).resolve().parents[2]

# 원본 경로를 하드코딩하지 않는다. 2026-08-05 22:19 다른 대회 다운로드가 `open.zip`
# 이름을 덮어써(Dacon 은 모든 대회를 open.zip 으로 내려준다) 이 스크립트가 죽었고,
# 같은 리터럴이 계획 스크립트 11 개에 흩어져 있었다. 이제 `configs/default.yaml` 이
# 단일 출처이고, 경로가 또 바뀌면 설정 한 줄만 고치면 된다.
#
# **동결 해시는 그대로다.** 아래 assert 가 그것을 강제하므로 캐시 키
# (`artifacts/cache/<OPEN_SHA>`) 도 바뀌지 않는다 — 무결성 계약은 유지되고
# 달라지는 것은 파일이 어디 있느냐뿐이다.
_CONFIG = load_config(REPO / "configs" / "default.yaml")
OPEN = _CONFIG.open_zip.path
BASELINE = _CONFIG.baseline_notebook.path
OPEN_SHA = "920be0c458d820e855bf79dd25723146f52ce1736138aedba5e6bc853f1f720b"
BASELINE_SHA = "712b26f4d2748860c94cff1e0100c23810468c983173f8e9ef8d009fe82df48c"
assert _CONFIG.open_zip.sha256 == OPEN_SHA, (
    f"설정의 open_zip sha256 이 동결값과 다르다: {_CONFIG.open_zip.sha256}"
)
assert _CONFIG.baseline_notebook.sha256 == BASELINE_SHA, (
    f"설정의 baseline sha256 이 동결값과 다르다: {_CONFIG.baseline_notebook.sha256}"
)
CACHE = REPO / "artifacts/cache" / OPEN_SHA
OUTPUT = REPO / "artifacts/backtests/metric-aligned-probe"
CAPACITIES = {1: 21_600.0, 2: 21_600.0, 3: 21_000.0}
METRIC_COLUMNS = [
    "forecast_id",
    "forecast_kst_dtm",
    "group_id",
    "actual_kwh",
    "prediction_kwh",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _scada_wind() -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    specifications = (
        ("vestas", "train/scada_vestas_train.csv", ((1, range(1, 7)), (2, range(7, 13)))),
        ("unison", "train/scada_unison_train.csv", ((3, range(1, 6)),)),
    )
    with zipfile.ZipFile(OPEN) as archive:
        for prefix, member, groups in specifications:
            with archive.open(member) as stream:
                raw = pd.read_csv(stream, parse_dates=["kst_dtm"])
            for group_id, turbine_numbers in groups:
                columns = [f"{prefix}_wtg{number:02d}_ws" for number in turbine_numbers]
                wind = raw[columns].where(lambda values: (values >= 0.0) & (values < 50.0))
                if group_id < 3:
                    timestamp = raw["kst_dtm"].dt.ceil("h")
                else:
                    timestamp = raw["kst_dtm"].dt.floor("h") + pd.Timedelta(hours=1)
                part = pd.DataFrame(
                    {
                        "forecast_kst_dtm": timestamp,
                        "group_id": group_id,
                        "scada_ws": wind.mean(axis=1),
                    }
                )
                parts.append(
                    part.groupby(["forecast_kst_dtm", "group_id"], as_index=False)[
                        "scada_ws"
                    ].mean()
                )
    return pd.concat(parts, ignore_index=True)


def _surface() -> tuple[pd.DataFrame, list[str], list[str]]:
    features = pd.read_parquet(CACHE / "train_features.parquet")
    grid = pd.read_parquet(CACHE / "train_grid_pivot.parquet")
    geometric = pd.read_parquet(CACHE / "train_geometric.parquet")
    labels = pd.read_parquet(CACHE / "labels_long.parquet")
    for frame in (features, grid, geometric, labels):
        frame["forecast_kst_dtm"] = pd.to_datetime(frame["forecast_kst_dtm"])
    raw_columns = [name for name in grid if name != "forecast_kst_dtm"]
    support_columns = [
        name
        for name in features
        if name.startswith(
            ("gfs_spatial__", "ldaps_spatial__", "source_disagreement__", "phys__", "phys_v2__")
        )
    ]
    calendar_columns = [
        name
        for name in (
            "hour",
            "month",
            "day_of_year",
            "lead_hour",
            "cal__hour_sin",
            "cal__hour_cos",
            "cal__doy_sin",
            "cal__doy_cos",
            "turbine_count",
            "latitude_centroid",
            "longitude_centroid",
            "hub_height_m",
            "rotor_diameter_m",
            "turbine_capacity_mw",
            "group_capacity_mw",
            "rotor_swept_area_m2",
            "fleet_swept_area_m2",
        )
        if name in features
    ]
    surface = (
        features[
            [
                "forecast_id",
                "forecast_kst_dtm",
                "data_available_kst_dtm",
                "issuance_batch",
                "group_id",
                *support_columns,
                *calendar_columns,
            ]
        ]
        .merge(grid, on="forecast_kst_dtm", validate="many_to_one")
        .merge(
            geometric,
            on=["forecast_kst_dtm", "data_available_kst_dtm", "group_id"],
            validate="one_to_one",
        )
        .merge(
            labels[["forecast_kst_dtm", "group_id", "actual_kwh"]],
            on=["forecast_kst_dtm", "group_id"],
            validate="one_to_one",
        )
        .merge(
            _scada_wind(),
            on=["forecast_kst_dtm", "group_id"],
            how="left",
            validate="one_to_one",
        )
    )
    for group_id in CAPACITIES:
        surface[f"group_{group_id}"] = surface["group_id"].eq(group_id).astype("int8")
    all_columns = list(
        dict.fromkeys(
            [
                *raw_columns,
                *support_columns,
                *calendar_columns,
                "group_1",
                "group_2",
                "group_3",
            ]
        )
    )
    tokens = (
        "10_10u",
        "10_10v",
        "80_u",
        "80_v",
        "100_100u",
        "100_100v",
        "50mu",
        "50mv",
        "wind10_",
        "wind50",
        "wind80",
        "wind100",
        "gust",
        "hub117",
        "rho_v3",
        "power_proxy",
        "shear",
        "group_",
        "lead_hour",
        "month",
        "hour",
        "doy",
    )
    wind_columns = [
        name for name in all_columns if any(token in name.lower() for token in tokens)
    ]
    geometric_columns = [name for name in geometric if name.startswith("geom__")]
    auxiliary_columns = [
        name
        for name in all_columns
        if any(
            token in name
            for token in (
                "10_10u",
                "10_10v",
                "80_u",
                "80_v",
                "100_100u",
                "100_100v",
                "50MU",
                "50MV",
                "wind10_",
                "wind50",
                "wind80",
                "wind100",
                "gust",
                "group_",
                "lead_hour",
                "month",
            )
        )
    ]
    return surface, wind_columns + geometric_columns, auxiliary_columns


def _sequence_columns(surface: pd.DataFrame) -> list[str]:
    preferred = [
        "gfs_spatial__idw__wind10_speed",
        "gfs_spatial__idw__wind80_speed",
        "gfs_spatial__idw__wind100_speed",
        "gfs_spatial__nearest__wind10_speed",
        "gfs_spatial__nearest__wind80_speed",
        "gfs_spatial__nearest__wind100_speed",
        "ldaps_spatial__idw__wind10_speed",
        "ldaps_spatial__idw__wind5_speed",
        "ldaps_spatial__idw__wind50max_speed",
        "ldaps_spatial__idw__wind50min_speed",
        "ldaps_spatial__nearest__wind10_speed",
        "ldaps_spatial__nearest__wind5_speed",
        "ldaps_spatial__nearest__wind50max_speed",
        "ldaps_spatial__nearest__wind50min_speed",
        "phys__hub117_speed",
        "phys__rho_v3",
        "phys_v2__hub117_speed",
        "phys_v2__rho_v3",
    ]
    for source, levels in (
        ("gfs", ("wind80", "wind100")),
        ("ldaps", ("wind10", "wind5", "wind50max", "wind50min")),
    ):
        for level in levels:
            for statistic in ("vector_speed", "mean_speed3", "layout_along"):
                preferred.append(f"geom__{source}__{level}__{statistic}")
    selected = [name for name in preferred if name in surface]
    if len(selected) < 30:
        raise RuntimeError(f"sequence feature contract resolved only {len(selected)} columns")
    return selected


def _score(frame: pd.DataFrame) -> dict[str, float]:
    score = evaluate_official(frame[METRIC_COLUMNS], CAPACITIES)
    return {
        "total": score.total,
        "one_minus_nmae": score.one_minus_nmae,
        "ficr": score.ficr,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--fold", default="dev-2023-Q4")
    parser.add_argument("--num-leaves", type=int, default=15)
    args = parser.parse_args()
    if args.fold not in {"dev-2023-Q2", "dev-2023-Q3", "dev-2023-Q4"}:
        raise ValueError("only frozen 2023 development folds are permitted")
    assert _sha256(OPEN) == OPEN_SHA and _sha256(BASELINE) == BASELINE_SHA
    started = time.perf_counter()
    surface, base_columns, auxiliary_columns = _surface()
    sequence_inputs = _sequence_columns(surface)
    contextual = add_issuance_sequence_context(
        surface[
            ["forecast_kst_dtm", "data_available_kst_dtm", "group_id", *sequence_inputs]
        ],
        sequence_inputs,
    )
    sequence_columns = [name for name in contextual if name.startswith("seq__")]
    surface = pd.concat([surface, contextual[sequence_columns]], axis=1)

    reference = pd.read_parquet(OUTPUT / "M50_GEOM_CLASS_TEMP075_G05-oof.parquet")
    reference = reference.loc[reference["fold_id"].eq(args.fold)]
    valid_keys = set(zip(reference["forecast_id"], reference["group_id"], strict=True))
    valid_mask = np.asarray(
        [
            (forecast_id, group_id) in valid_keys
            for forecast_id, group_id in zip(
                surface["forecast_id"], surface["group_id"], strict=True
            )
        ]
    )
    start = reference["forecast_kst_dtm"].min()
    preceding = surface["forecast_kst_dtm"].lt(start)
    normalized_target = surface["actual_kwh"] / surface["group_id"].map(CAPACITIES)

    auxiliary_matrix = surface[auxiliary_columns].astype("float32")
    auxiliary_train = preceding & surface["scada_ws"].notna()
    auxiliary_ids = np.flatnonzero(auxiliary_train.to_numpy())
    auxiliary_prediction = np.full(len(surface), np.nan, dtype="float32")
    auxiliary_params = {
        "objective": "l2",
        "n_estimators": 400,
        "learning_rate": 0.04,
        "num_leaves": 31,
        "min_child_samples": 60,
        "subsample": 0.9,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 2.0,
        "random_state": 20260801,
        "n_jobs": 6,
        "deterministic": True,
        "force_col_wise": True,
        "verbosity": -1,
    }
    splitter = KFold(3, shuffle=True, random_state=20260801)
    for fit_index, holdout_index in splitter.split(auxiliary_ids):
        model = LGBMRegressor(**auxiliary_params)
        model.fit(
            auxiliary_matrix.iloc[auxiliary_ids[fit_index]],
            surface["scada_ws"].iloc[auxiliary_ids[fit_index]],
        )
        auxiliary_prediction[auxiliary_ids[holdout_index]] = model.predict(
            auxiliary_matrix.iloc[auxiliary_ids[holdout_index]]
        )
    model = LGBMRegressor(**auxiliary_params)
    model.fit(
        auxiliary_matrix.loc[auxiliary_train], surface.loc[auxiliary_train, "scada_ws"]
    )
    auxiliary_prediction[valid_mask] = model.predict(auxiliary_matrix.loc[valid_mask])

    matrix = surface[[*base_columns, *sequence_columns]].astype("float32")
    matrix["aux_scada_ws"] = auxiliary_prediction
    matrix["aux_scada_ws2"] = auxiliary_prediction**2
    matrix["aux_scada_ws3"] = auxiliary_prediction**3
    training = preceding & surface["actual_kwh"].notna() & normalized_target.ge(0.10)
    width = 0.025
    classes = np.floor((normalized_target.clip(0.10, 1.074999) - 0.10) / width).astype(
        "Int64"
    )
    class_count = int(classes.loc[training].max()) + 1
    centers = np.asarray(
        [
            normalized_target.loc[training & classes.eq(class_id)].mean()
            if (training & classes.eq(class_id)).any()
            else 0.10 + (class_id + 0.5) * width
            for class_id in range(class_count)
        ]
    )
    batches = surface.loc[training, "data_available_kst_dtm"].drop_duplicates().sort_values()
    cutoff = batches.iloc[int(len(batches) * 0.80)]
    inner_fit = training & surface["data_available_kst_dtm"].lt(cutoff)
    inner_stop = training & ~inner_fit
    params = {
        "objective": "multiclass",
        "num_class": class_count,
        "n_estimators": 800,
        "learning_rate": 0.025,
        "num_leaves": args.num_leaves,
        "min_child_samples": 80,
        "subsample": 0.9,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 2.0,
        "random_state": 20260801,
        "n_jobs": 6,
        "deterministic": True,
        "force_col_wise": True,
        "verbosity": -1,
    }
    classifier = LGBMClassifier(**params)
    classifier.fit(
        matrix.loc[inner_fit],
        classes.loc[inner_fit].astype(int),
        sample_weight=normalized_target.loc[inner_fit].clip(lower=0.10),
        eval_set=[(matrix.loc[inner_stop], classes.loc[inner_stop].astype(int))],
        eval_sample_weight=[normalized_target.loc[inner_stop].clip(lower=0.10)],
        callbacks=[lightgbm.early_stopping(50, verbose=False)],
    )
    best_iteration = max(1, int(classifier.best_iteration_ or params["n_estimators"]))
    classifier = LGBMClassifier(**{**params, "n_estimators": best_iteration})
    classifier.fit(
        matrix.loc[training],
        classes.loc[training].astype(int),
        sample_weight=normalized_target.loc[training].clip(lower=0.10),
    )
    probability = classifier.predict_proba(matrix.loc[valid_mask])
    base = surface.loc[
        valid_mask, ["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh"]
    ].copy()
    actions = np.arange(0.075, 1.076, 0.0025)
    error = np.abs(actions[:, None] - centers[None, :])
    units = np.select([error <= 0.06, error <= 0.08], [4.0, 3.0], default=0.0)
    mean_generation = {
        group_id: float(normalized_target.loc[training & surface["group_id"].eq(group_id)].mean())
        for group_id in CAPACITIES
    }
    results: dict[str, dict[str, float]] = {}
    for temperature in (1.0, 0.75):
        calibrated = probability ** (1.0 / temperature)
        calibrated /= calibrated.sum(axis=1, keepdims=True)
        for gamma in (0.5, 0.75, 1.0):
            chosen = np.empty(len(base), dtype=float)
            for group_id in CAPACITIES:
                mask = base["group_id"].eq(group_id).to_numpy()
                group_probability = calibrated[mask]
                utility = -(group_probability @ error.T) + gamma * (
                    group_probability @ (centers[None, :] * units).T
                ) / (4.0 * mean_generation[group_id])
                chosen[mask] = actions[np.argmax(utility, axis=1)]
            candidate = base.copy()
            candidate["prediction_kwh"] = (
                chosen * candidate["group_id"].map(CAPACITIES).to_numpy(dtype=float)
            )
            tag = f"T{temperature:g}_G{gamma:g}"
            results[tag] = _score(candidate)
            print(json.dumps({"policy": tag, "score": results[tag]}), flush=True)

    best_policy = max(results, key=lambda name: results[name]["total"])
    temperature = float(best_policy.split("_")[0][1:])
    gamma = float(best_policy.split("_")[1][1:])
    calibrated = probability ** (1.0 / temperature)
    calibrated /= calibrated.sum(axis=1, keepdims=True)
    chosen = np.empty(len(base), dtype=float)
    for group_id in CAPACITIES:
        mask = base["group_id"].eq(group_id).to_numpy()
        group_probability = calibrated[mask]
        utility = -(group_probability @ error.T) + gamma * (
            group_probability @ (centers[None, :] * units).T
        ) / (4.0 * mean_generation[group_id])
        chosen[mask] = actions[np.argmax(utility, axis=1)]
    output = base.copy()
    output["prediction_kwh"] = chosen * output["group_id"].map(CAPACITIES).to_numpy()
    output["fold_id"] = args.fold
    output["model_id"] = args.candidate_id
    path = OUTPUT / f"{args.candidate_id}-q4.parquet"
    output.to_parquet(path, index=False)
    receipt = {
        "candidate_id": args.candidate_id,
        "fold_id": args.fold,
        "feature_count": matrix.shape[1],
        "sequence_input_count": len(sequence_inputs),
        "sequence_feature_count": len(sequence_columns),
        "class_count": class_count,
        "selected_iteration": best_iteration,
        "best_policy": best_policy,
        "scores": results,
        "runtime_seconds": round(time.perf_counter() - started, 2),
        "prediction_path": str(path.relative_to(REPO)),
        "prediction_sha256": _sha256(path),
        "new_2024_evaluation": False,
        "lockbox_reopened": False,
        "external_actions": [],
    }
    receipt_path = OUTPUT / f"{args.candidate_id}-q4.json"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
