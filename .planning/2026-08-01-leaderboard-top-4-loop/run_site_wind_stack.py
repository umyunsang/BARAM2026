"""Sequentially stack inference-legal site-wind forecasts onto M50.

The stacker is trained only on earlier development-fold predictions.  The
observed SCADA column present in the diagnostic artifacts is intentionally
dropped before any power-model matrix is built.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from run_sequence_classifier import (
    BASELINE,
    BASELINE_SHA,
    CAPACITIES,
    METRIC_COLUMNS,
    OPEN,
    OPEN_SHA,
    OUTPUT,
    _sha256,
    _surface,
)
from sklearn.isotonic import IsotonicRegression

from baram.evaluation.official import evaluate_group_component, evaluate_official

FOLDS = ("dev-2023-Q2", "dev-2023-Q3", "dev-2023-Q4")
SITE_WIND_PATHS = {
    "dev-2023-Q2": OUTPUT
    / "M62_SITE_WIND_L2-dev-2023-Q2-site-wind.parquet",
    "dev-2023-Q3": OUTPUT
    / "M62_SITE_WIND_L2-dev-2023-Q3-site-wind.parquet",
    "dev-2023-Q4": OUTPUT
    / "M62_SITE_WIND_LGBM_SCREEN-dev-2023-Q4-site-wind.parquet",
}


@dataclass(frozen=True)
class Candidate:
    name: str
    mode: str
    objective: str = "l1"
    num_leaves: int = 7
    weight_power: float = 0.0


CANDIDATES = (
    Candidate("isotonic_sitewind", "isotonic"),
    Candidate("direct_l1_l7", "direct", "l1", 7, 0.0),
    Candidate("direct_l1_l15", "direct", "l1", 15, 0.0),
    Candidate("direct_l1_l15_w1", "direct", "l1", 15, 1.0),
    Candidate("residual_l1_l7", "residual", "l1", 7, 0.0),
    Candidate("residual_l1_l15", "residual", "l1", 15, 0.0),
    Candidate("residual_l2_l7", "residual", "l2", 7, 0.0),
    Candidate("residual_l2_l15", "residual", "l2", 15, 0.0),
)


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _score(frame: pd.DataFrame) -> dict[str, float]:
    result = evaluate_official(frame[METRIC_COLUMNS], CAPACITIES)
    return {
        "total": result.total,
        "one_minus_nmae": result.one_minus_nmae,
        "ficr": result.ficr,
    }


def _group_score(frame: pd.DataFrame, group_id: int) -> float:
    part = frame.loc[frame["group_id"].eq(group_id)]
    result = evaluate_group_component(
        part[METRIC_COLUMNS], group_id, CAPACITIES[group_id]
    )
    return 0.5 * (1.0 - result.nmae) + 0.5 * result.ficr


def _load_frame() -> tuple[pd.DataFrame, list[str]]:
    reference = pd.read_parquet(OUTPUT / "M50_GEOM_CLASS_TEMP075_G05-oof.parquet")
    parts: list[pd.DataFrame] = []
    for fold_id in FOLDS:
        site = pd.read_parquet(SITE_WIND_PATHS[fold_id]).drop(columns="scada_ws")
        site = site.rename(
            columns={
                "legacy_shared_l2": "legacy_ws",
                "allweather_group_l2": "allweather_ws",
            }
        )
        part = reference.loc[reference["fold_id"].eq(fold_id)].merge(
            site,
            on=["forecast_id", "forecast_kst_dtm", "group_id"],
            how="inner",
            validate="one_to_one",
        )
        if len(part) != int(reference["fold_id"].eq(fold_id).sum()):
            raise RuntimeError(f"site-wind key contract changed for {fold_id}")
        parts.append(part)
    frame = pd.concat(parts, ignore_index=True)
    frame["capacity"] = frame["group_id"].map(CAPACITIES).astype(float)
    frame["m50"] = frame["prediction_kwh"] / frame["capacity"]
    frame["target"] = frame["actual_kwh"] / frame["capacity"]
    frame["ws_mean"] = (frame["legacy_ws"] + frame["allweather_ws"]) / 2.0
    frame["ws_delta"] = frame["allweather_ws"] - frame["legacy_ws"]
    frame["ws_disagreement"] = frame["ws_delta"].abs()
    for name in ("legacy_ws", "allweather_ws", "ws_mean"):
        frame[f"{name}2"] = frame[name] ** 2
        frame[f"{name}3"] = frame[name] ** 3

    surface, _, _ = _surface()
    preferred = [
        "forecast_id",
        "group_id",
        "hour",
        "month",
        "day_of_year",
        "lead_hour",
        "cal__hour_sin",
        "cal__hour_cos",
        "cal__doy_sin",
        "cal__doy_cos",
    ]
    weather_tokens = (
        "spatial__idw__wind",
        "spatial__nearest__wind",
        "phys__hub117_speed",
        "phys_v2__hub117_speed",
        "rho_v3",
        "power_proxy",
        "source_disagreement__",
    )
    compact_weather = [
        name
        for name in surface
        if any(token in name for token in weather_tokens)
        and pd.api.types.is_numeric_dtype(surface[name])
    ]
    allowed = list(dict.fromkeys([*preferred, *compact_weather]))
    compact = surface[allowed].drop_duplicates(["forecast_id", "group_id"])
    frame = frame.merge(
        compact,
        on=["forecast_id", "group_id"],
        how="left",
        validate="one_to_one",
    )
    for group_id in CAPACITIES:
        frame[f"group_{group_id}"] = frame["group_id"].eq(group_id).astype("int8")
    feature_columns = [
        "m50",
        "legacy_ws",
        "allweather_ws",
        "ws_mean",
        "ws_delta",
        "ws_disagreement",
        "legacy_ws2",
        "legacy_ws3",
        "allweather_ws2",
        "allweather_ws3",
        "ws_mean2",
        "ws_mean3",
        *[name for name in allowed if name not in {"forecast_id", "group_id"}],
        "group_1",
        "group_2",
        "group_3",
    ]
    feature_columns = list(dict.fromkeys(feature_columns))
    forbidden = {"actual_kwh", "target", "scada_ws"}
    if forbidden.intersection(feature_columns):
        raise RuntimeError("observed target entered the stack feature matrix")
    return frame, feature_columns


def _fit_predict(
    candidate: Candidate,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    feature_columns: list[str],
) -> np.ndarray:
    eligible = train["target"].ge(0.10)
    train = train.loc[eligible]
    if candidate.mode == "isotonic":
        prediction = np.empty(len(valid), dtype=float)
        for group_id in CAPACITIES:
            fit = train["group_id"].eq(group_id)
            apply = valid["group_id"].eq(group_id).to_numpy()
            model = IsotonicRegression(y_min=0.0, y_max=1.075, out_of_bounds="clip")
            model.fit(train.loc[fit, "allweather_ws"], train.loc[fit, "target"])
            prediction[apply] = model.predict(valid.loc[apply, "allweather_ws"])
        return prediction
    target = train["target"].to_numpy(dtype=float)
    if candidate.mode == "residual":
        target = target - train["m50"].to_numpy(dtype=float)
    model = LGBMRegressor(
        objective=candidate.objective,
        n_estimators=260,
        learning_rate=0.025,
        num_leaves=candidate.num_leaves,
        min_child_samples=100,
        max_bin=127,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        reg_alpha=0.3,
        reg_lambda=5.0,
        random_state=20260802,
        n_jobs=6,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )
    weights = np.power(train["target"].clip(lower=0.10), candidate.weight_power)
    model.fit(train[feature_columns].astype("float32"), target, sample_weight=weights)
    prediction = model.predict(valid[feature_columns].astype("float32"))
    if candidate.mode == "residual":
        prediction += valid["m50"].to_numpy(dtype=float)
    return np.clip(prediction, 0.0, 1.075)


def _prediction_frame(base: pd.DataFrame, normalized: np.ndarray) -> pd.DataFrame:
    result = base[
        ["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh"]
    ].copy()
    result["prediction_kwh"] = (
        normalized * result["group_id"].map(CAPACITIES).to_numpy(dtype=float)
    )
    return result


def _apply_policy(
    base: pd.DataFrame,
    candidate_prediction: np.ndarray,
    weights: dict[int, float],
    snap: bool,
) -> pd.DataFrame:
    group_weights = base["group_id"].map(weights).to_numpy(dtype=float)
    normalized = group_weights * base["m50"].to_numpy(dtype=float) + (
        1.0 - group_weights
    ) * candidate_prediction
    if snap:
        normalized = np.round(normalized / 0.0025) * 0.0025
    return _prediction_frame(base, np.clip(normalized, 0.0, 1.075))


def _select(
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[Candidate, dict[int, float], bool, dict[str, object]]:
    weight_grid = np.round(np.linspace(0.0, 1.0, 21), 2)
    diagnostics: dict[str, object] = {}
    best: tuple[float, Candidate, dict[int, float], bool] | None = None
    for candidate in CANDIDATES:
        prediction = _fit_predict(candidate, train, calibration, feature_columns)
        candidate_results: dict[str, object] = {}
        for snap in (False, True):
            weights: dict[int, float] = {}
            group_scores: dict[str, float] = {}
            for group_id in CAPACITIES:
                best_group: tuple[float, float] | None = None
                for weight in weight_grid:
                    trial = _apply_policy(
                        calibration,
                        prediction,
                        {key: float(weight) for key in CAPACITIES},
                        snap,
                    )
                    score = _group_score(trial, group_id)
                    if best_group is None or score > best_group[0]:
                        best_group = (score, float(weight))
                assert best_group is not None
                group_scores[str(group_id)] = best_group[0]
                weights[group_id] = best_group[1]
            trial = _apply_policy(calibration, prediction, weights, snap)
            score = _score(trial)
            candidate_results[str(snap)] = {
                "weights": weights,
                "group_totals": group_scores,
                "score": score,
            }
            choice = (score["total"], candidate, weights, snap)
            if best is None or choice[0] > best[0]:
                best = choice
        diagnostics[candidate.name] = candidate_results
    assert best is not None
    return best[1], best[2], best[3], diagnostics


def _chronology_stage(
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    application: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, dict[str, object]]:
    candidate, weights, snap, diagnostics = _select(
        train, calibration, feature_columns
    )
    prediction = _fit_predict(
        candidate,
        pd.concat([train, calibration]),
        application,
        feature_columns,
    )
    output = _apply_policy(application, prediction, weights, snap)
    receipt = {
        "selected_candidate": candidate.__dict__,
        "selected_m50_weights": weights,
        "snap_to_0p0025": snap,
        "calibration_diagnostics": diagnostics,
        "application_score": _score(output),
    }
    return output, receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-id", required=True)
    args = parser.parse_args()
    if _sha256(OPEN) != OPEN_SHA or _sha256(BASELINE) != BASELINE_SHA:
        raise RuntimeError("immutable competition input hash mismatch")
    started = time.perf_counter()
    frame, feature_columns = _load_frame()
    q2 = frame.loc[frame["fold_id"].eq(FOLDS[0])].sort_values("forecast_kst_dtm")
    q3 = frame.loc[frame["fold_id"].eq(FOLDS[1])].sort_values("forecast_kst_dtm")
    q4 = frame.loc[frame["fold_id"].eq(FOLDS[2])].sort_values("forecast_kst_dtm")

    q2_times = q2["forecast_kst_dtm"].drop_duplicates().sort_values()
    cutoff = q2_times.iloc[int(len(q2_times) * 0.60)]
    q2_early = q2.loc[q2["forecast_kst_dtm"].lt(cutoff)]
    q2_late = q2.loc[q2["forecast_kst_dtm"].ge(cutoff)]
    q3_output, q3_receipt = _chronology_stage(
        q2_early, q2_late, q3, feature_columns
    )
    q4_output, q4_receipt = _chronology_stage(q2, q3, q4, feature_columns)
    q2_output = _prediction_frame(q2, q2["m50"].to_numpy(dtype=float))
    q2_output["fold_id"] = FOLDS[0]
    q3_output["fold_id"] = FOLDS[1]
    q4_output["fold_id"] = FOLDS[2]
    combined = pd.concat([q2_output, q3_output, q4_output], ignore_index=True)
    combined["model_id"] = args.candidate_id

    output_path = OUTPUT / f"{args.candidate_id}-oof.parquet"
    combined.to_parquet(output_path, index=False)
    receipt = {
        "schema_version": 1,
        "candidate_id": args.candidate_id,
        "scope": "2023 sequential site-wind stack; Q2 raw, Q3 Q2-tuned, Q4 Q3-tuned",
        "feature_count": len(feature_columns),
        "q3_stage": q3_receipt,
        "q4_stage": q4_receipt,
        "fold_scores": {
            fold_id: _score(combined.loc[combined["fold_id"].eq(fold_id)])
            for fold_id in FOLDS
        },
        "pooled": _score(combined),
        "runtime_seconds": round(time.perf_counter() - started, 2),
        "prediction_path": str(output_path.relative_to(Path.cwd())),
        "prediction_sha256": _file_sha(output_path),
        "observed_scada_feature_count": 0,
        "new_2024_evaluation": False,
        "lockbox_reopened": False,
        "external_actions": [],
    }
    receipt_path = OUTPUT / f"{args.candidate_id}-oof.json"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
