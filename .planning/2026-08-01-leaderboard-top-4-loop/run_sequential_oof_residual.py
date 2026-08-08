"""Correct M107 with residual models trained only on earlier OOF quarters."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
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
from run_site_wind_classifier import _add_site_wind_features

FOLDS = ("dev-2023-Q2", "dev-2023-Q3", "dev-2023-Q4")
PARENT_ID = "M107_STRICT_TEMPORAL_TOP100"
PARENT_PATH = OUTPUT / f"{PARENT_ID}-oof.parquet"
SITEWIND_ID = "M64B_ALLWEATHER_SITEWIND_CLASS"
KEYS = ["forecast_id", "forecast_kst_dtm", "group_id"]
BASE_COLUMNS = [*KEYS, "actual_kwh"]
WEIGHT_GRID = tuple(round(value, 2) for value in np.linspace(0.0, 1.0, 21))


@dataclass(frozen=True)
class Spec:
    mode: str
    objective: str
    leaves: int
    iterations: int
    target_weight_power: float

    @property
    def name(self) -> str:
        return (
            f"{self.mode}_{self.objective}_l{self.leaves}_i{self.iterations}"
            f"_w{self.target_weight_power:g}"
        )


SPECS = (
    Spec("shared", "l1", 7, 80, 0.0),
    Spec("shared", "l1", 15, 80, 0.0),
    Spec("shared", "huber", 7, 80, 0.0),
    Spec("shared", "huber", 15, 80, 0.0),
    Spec("shared", "l1", 7, 160, 1.0),
    Spec("shared", "l1", 15, 160, 1.0),
    Spec("group", "l1", 7, 80, 0.0),
    Spec("group", "l1", 15, 80, 1.0),
)


def _stable_features() -> list[str]:
    selected: list[set[str]] = []
    for fold_id in FOLDS:
        receipt = json.loads((OUTPUT / f"M102_TOP100-{fold_id}.json").read_text())
        names = receipt["selected_feature_names"]
        if len(names) != 100 or len(set(names)) != 100:
            raise RuntimeError(f"M102 feature contract changed for {fold_id}")
        selected.append(set(names))
    stable = sorted(set.intersection(*selected))
    if len(stable) != 73:
        raise RuntimeError(f"expected 73 stable M102 features, received {len(stable)}")
    return stable


def _fold_surface(
    surface: pd.DataFrame,
    parent: pd.DataFrame,
    fold_id: str,
    stable_features: list[str],
) -> pd.DataFrame:
    cached = np.load(OUTPUT / f"{SITEWIND_ID}-{fold_id}-sitewind-features.npz")
    has_sitewind = any(name.startswith("sitewind__") for name in stable_features)
    if has_sitewind:
        missing_sitewind = [
            name for name in stable_features if name.startswith("sitewind__")
        ]
        base_features = [name for name in stable_features if not name.startswith("sitewind__")]
        matrix = surface[base_features].copy()
        _add_site_wind_features(matrix, cached["legacy"], cached["allweather"])
        if set(missing_sitewind).difference(matrix.columns):
            raise RuntimeError(f"missing site-wind features for {fold_id}")
        matrix = matrix[stable_features]
    else:
        matrix = surface[stable_features].copy()

    start, end = {
        "dev-2023-Q2": (pd.Timestamp("2023-04-01 01:00:00"), pd.Timestamp("2023-07-01 01:00:00")),
        "dev-2023-Q3": (pd.Timestamp("2023-07-01 01:00:00"), pd.Timestamp("2023-10-01 01:00:00")),
        "dev-2023-Q4": (pd.Timestamp("2023-10-01 01:00:00"), pd.Timestamp("2024-01-01 01:00:00")),
    }[fold_id]
    mask = surface["forecast_kst_dtm"].ge(start) & surface["forecast_kst_dtm"].lt(end)
    meta = surface.loc[
        mask,
        [
            *BASE_COLUMNS,
            "data_available_kst_dtm",
            "lead_hour",
        ],
    ].copy()
    meta = meta.join(matrix.loc[mask])
    fold_parent = parent.loc[parent["fold_id"].eq(fold_id), [*KEYS, "prediction_kwh"]]
    meta = meta.merge(fold_parent, on=KEYS, validate="one_to_one")
    capacity = meta["group_id"].map(CAPACITIES).astype(float)
    meta["parent"] = meta["prediction_kwh"] / capacity
    meta["target"] = meta["actual_kwh"] / capacity
    meta["residual"] = meta["target"] - meta["parent"]
    meta["fold_id"] = fold_id

    parent_wide = meta.pivot(
        index="forecast_kst_dtm", columns="group_id", values="parent"
    ).rename(columns={1: "parent_g1", 2: "parent_g2", 3: "parent_g3"})
    parent_wide["parent_group_mean"] = parent_wide.mean(axis=1)
    parent_wide["parent_group_range"] = parent_wide.max(axis=1) - parent_wide.min(axis=1)
    meta = meta.join(parent_wide, on="forecast_kst_dtm")

    group_keys = [meta["group_id"], meta["data_available_kst_dtm"]]
    grouped_parent = meta["parent"].groupby(group_keys, sort=False)
    grouped_wind = meta["sitewind__mean"].groupby(group_keys, sort=False)
    for offset in (-2, -1, 1, 2):
        meta[f"parent_h{offset:+d}"] = grouped_parent.shift(-offset).fillna(meta["parent"])
        meta[f"sitewind_mean_h{offset:+d}"] = grouped_wind.shift(-offset).fillna(
            meta["sitewind__mean"]
        )
    meta["parent_slope2"] = meta["parent_h+1"] - meta["parent_h-1"]
    meta["parent_curvature"] = (
        meta["parent_h+1"] - 2.0 * meta["parent"] + meta["parent_h-1"]
    )
    meta["sitewind_slope2"] = (
        meta["sitewind_mean_h+1"] - meta["sitewind_mean_h-1"]
    )
    for group_id in CAPACITIES:
        meta[f"group_{group_id}"] = meta["group_id"].eq(group_id).astype("int8")
    return meta.sort_values(["forecast_kst_dtm", "group_id"]).reset_index(drop=True)


def _features(frame: pd.DataFrame, stable_features: list[str]) -> list[str]:
    additions = [
        "parent",
        "parent_g1",
        "parent_g2",
        "parent_g3",
        "parent_group_mean",
        "parent_group_range",
        "parent_h-2",
        "parent_h-1",
        "parent_h+1",
        "parent_h+2",
        "parent_slope2",
        "parent_curvature",
        "sitewind_mean_h-2",
        "sitewind_mean_h-1",
        "sitewind_mean_h+1",
        "sitewind_mean_h+2",
        "sitewind_slope2",
        "lead_hour",
        "group_1",
        "group_2",
        "group_3",
    ]
    names = [*stable_features, *additions]
    if len(names) != len(set(names)) or set(names).difference(frame.columns):
        raise RuntimeError("residual feature contract changed")
    return names


def _model(spec: Spec) -> LGBMRegressor:
    return LGBMRegressor(
        objective=spec.objective,
        n_estimators=spec.iterations,
        learning_rate=0.025,
        num_leaves=spec.leaves,
        min_child_samples=80,
        max_bin=127,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.3,
        reg_lambda=5.0,
        random_state=20260802,
        n_jobs=6,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )


def _fit_predict(
    spec: Spec,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    features: list[str],
) -> np.ndarray:
    eligible = train["target"].ge(0.10) & train["target"].notna()
    train = train.loc[eligible]
    prediction = np.empty(len(valid), dtype=float)
    if spec.mode == "shared":
        model = _model(spec)
        weight = train["target"].clip(lower=0.10) ** spec.target_weight_power
        model.fit(
            train[features].astype("float32"),
            train["residual"],
            sample_weight=weight,
        )
        prediction[:] = model.predict(valid[features].astype("float32"))
    else:
        for group_id in CAPACITIES:
            fit = train["group_id"].eq(group_id)
            apply = valid["group_id"].eq(group_id).to_numpy()
            model = _model(spec)
            weight = train.loc[fit, "target"].clip(lower=0.10) ** spec.target_weight_power
            model.fit(
                train.loc[fit, features].astype("float32"),
                train.loc[fit, "residual"],
                sample_weight=weight,
            )
            prediction[apply] = model.predict(valid.loc[apply, features].astype("float32"))
    return np.clip(valid["parent"].to_numpy(dtype=float) + prediction, 0.0, 1.075)


def _prediction_frame(base: pd.DataFrame, normalized: np.ndarray) -> pd.DataFrame:
    result = base[BASE_COLUMNS].copy()
    result["prediction_kwh"] = normalized * result["group_id"].map(CAPACITIES).to_numpy(dtype=float)
    return result


def _group_total(frame: pd.DataFrame, group_id: int) -> float:
    part = frame.loc[frame["group_id"].eq(group_id)]
    capacity = CAPACITIES[group_id]
    actual = part["actual_kwh"].to_numpy(dtype=float) / capacity
    prediction = part["prediction_kwh"].to_numpy(dtype=float) / capacity
    valid = np.isfinite(actual) & actual.__ge__(0.10)
    actual = actual[valid]
    error = np.abs(prediction[valid] - actual)
    units = np.select([error <= 0.06, error <= 0.08], [4.0, 3.0], default=0.0)
    one_minus_nmae = 1.0 - float(error.mean())
    ficr = float((actual * units).sum() / (4.0 * actual.sum()))
    return 0.5 * (one_minus_nmae + ficr)


def _apply_policy(
    base: pd.DataFrame,
    corrected: np.ndarray,
    parent_weights: dict[int, float],
    snap: bool,
) -> pd.DataFrame:
    weights = base["group_id"].map(parent_weights).to_numpy(dtype=float)
    normalized = weights * base["parent"].to_numpy(dtype=float) + (1.0 - weights) * corrected
    if snap:
        normalized = np.round(normalized / 0.0025) * 0.0025
    return _prediction_frame(base, np.clip(normalized, 0.0, 1.075))


def _select(
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    features: list[str],
) -> tuple[Spec, dict[int, float], bool, dict[str, object]]:
    diagnostics: dict[str, object] = {}
    best: tuple[float, Spec, dict[int, float], bool] | None = None
    for spec in SPECS:
        corrected = _fit_predict(spec, train, calibration, features)
        spec_results: dict[str, object] = {}
        for snap in (False, True):
            weights: dict[int, float] = {}
            group_scores: dict[str, float] = {}
            for group_id in CAPACITIES:
                best_group: tuple[float, float] | None = None
                for weight in WEIGHT_GRID:
                    uniform = {key: weight for key in CAPACITIES}
                    trial = _apply_policy(calibration, corrected, uniform, snap)
                    total = _group_total(trial, group_id)
                    if best_group is None or total > best_group[0]:
                        best_group = (total, weight)
                assert best_group is not None
                group_scores[str(group_id)] = best_group[0]
                weights[group_id] = best_group[1]
            trial = _apply_policy(calibration, corrected, weights, snap)
            score = _score(trial)
            spec_results[str(snap)] = {
                "parent_weights": weights,
                "group_totals": group_scores,
                "score": score,
            }
            choice = (score["total"], spec, weights, snap)
            if best is None or choice[0] > best[0]:
                best = choice
        diagnostics[spec.name] = spec_results
    assert best is not None
    return best[1], best[2], best[3], diagnostics


def _stage(
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    application: pd.DataFrame,
    features: list[str],
) -> tuple[pd.DataFrame, dict[str, object]]:
    spec, weights, snap, diagnostics = _select(train, calibration, features)
    corrected = _fit_predict(
        spec,
        pd.concat([train, calibration], ignore_index=True),
        application,
        features,
    )
    output = _apply_policy(application, corrected, weights, snap)
    return output, {
        "selected_spec": asdict(spec),
        "selected_parent_weights": weights,
        "snap_to_0p0025": snap,
        "calibration_diagnostics": diagnostics,
        "application_score": _score(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-id", default="M120_SEQUENTIAL_OOF_RESIDUAL")
    args = parser.parse_args()
    if _sha256(OPEN) != OPEN_SHA or _sha256(BASELINE) != BASELINE_SHA:
        raise RuntimeError("immutable competition input hash mismatch")
    started = time.perf_counter()
    surface, _, _ = _surface()
    parent = pd.read_parquet(PARENT_PATH)
    stable = _stable_features()
    folds = {
        fold_id: _fold_surface(surface, parent, fold_id, stable) for fold_id in FOLDS
    }
    features = _features(folds[FOLDS[0]], stable)

    q2 = folds[FOLDS[0]]
    q2_times = q2["forecast_kst_dtm"].drop_duplicates().sort_values()
    cutoff = q2_times.iloc[int(len(q2_times) * 0.60)]
    q2_early = q2.loc[q2["forecast_kst_dtm"].lt(cutoff)]
    q2_late = q2.loc[q2["forecast_kst_dtm"].ge(cutoff)]
    q3_output, q3_receipt = _stage(q2_early, q2_late, folds[FOLDS[1]], features)
    q4_output, q4_receipt = _stage(q2, folds[FOLDS[1]], folds[FOLDS[2]], features)
    q2_output = _prediction_frame(q2, q2["parent"].to_numpy(dtype=float))
    q2_output["fold_id"] = FOLDS[0]
    q3_output["fold_id"] = FOLDS[1]
    q4_output["fold_id"] = FOLDS[2]
    output = pd.concat([q2_output, q3_output, q4_output], ignore_index=True)
    output["model_id"] = args.candidate_id

    output_path = OUTPUT / f"{args.candidate_id}-oof.parquet"
    output.to_parquet(output_path, index=False)
    receipt = {
        "schema_version": 1,
        "candidate_id": args.candidate_id,
        "scope": "Q2 parent control; Q3 selected inside Q2; Q4 selected on Q3",
        "parent_candidate_id": PARENT_ID,
        "parent_prediction_sha256": _sha256(PARENT_PATH),
        "stable_m102_feature_count": len(stable),
        "feature_count": len(features),
        "q3_stage": q3_receipt,
        "q4_stage": q4_receipt,
        "fold_scores": {
            fold_id: _score(output.loc[output["fold_id"].eq(fold_id)])
            for fold_id in FOLDS
        },
        "pooled": _score(output),
        "runtime_seconds": round(time.perf_counter() - started, 2),
        "prediction_path": str(output_path.relative_to(Path.cwd())),
        "prediction_sha256": _sha256(output_path),
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
