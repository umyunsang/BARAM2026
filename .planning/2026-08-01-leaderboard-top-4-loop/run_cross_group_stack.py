"""Chronology-safe cross-group calibration of the group-3 power forecast."""

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
)
from sklearn.linear_model import Ridge

FOLDS = ("dev-2023-Q2", "dev-2023-Q3", "dev-2023-Q4")


@dataclass(frozen=True)
class Spec:
    family: str
    objective: str = "l1"
    num_leaves: int = 7
    iterations: int = 120
    alpha: float = 1.0

    @property
    def name(self) -> str:
        if self.family == "ridge":
            return f"ridge_a{self.alpha:g}"
        return f"lgb_{self.objective}_l{self.num_leaves}_i{self.iterations}"


SPECS = (
    Spec("ridge", alpha=0.1),
    Spec("ridge", alpha=1.0),
    Spec("ridge", alpha=10.0),
    *(
        Spec("lgbm", objective, leaves, iterations)
        for objective in ("l1", "l2", "huber")
        for leaves in (3, 7, 15)
        for iterations in (60, 120, 240)
    ),
)


def _wide(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    capacity = work["group_id"].map(CAPACITIES).astype(float)
    work["normalized_prediction"] = work["prediction_kwh"] / capacity
    work["normalized_actual"] = work["actual_kwh"] / capacity
    predictions = work.pivot(
        index="forecast_kst_dtm",
        columns="group_id",
        values="normalized_prediction",
    ).rename(columns={1: "p1", 2: "p2", 3: "p3"})
    target = (
        work.loc[work["group_id"].eq(3)]
        .set_index("forecast_kst_dtm")[
            ["forecast_id", "fold_id", "normalized_actual"]
        ]
        .rename(columns={"normalized_actual": "target"})
    )
    output = target.join(predictions, how="inner").reset_index()
    output["p12_mean"] = (output["p1"] + output["p2"]) / 2.0
    output["p12_min"] = output[["p1", "p2"]].min(axis=1)
    output["p12_max"] = output[["p1", "p2"]].max(axis=1)
    output["p12_diff"] = output["p2"] - output["p1"]
    output["p3_minus_p12"] = output["p3"] - output["p12_mean"]
    output["p12_disagreement"] = output["p12_diff"].abs()
    for name in ("p1", "p2", "p3", "p12_mean"):
        output[f"{name}_2"] = output[name] ** 2
        output[f"{name}_3"] = output[name] ** 3
    hour = output["forecast_kst_dtm"].dt.hour
    day = output["forecast_kst_dtm"].dt.dayofyear
    output["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    output["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    output["doy_sin"] = np.sin(2.0 * np.pi * day / 365.25)
    output["doy_cos"] = np.cos(2.0 * np.pi * day / 365.25)
    return output.sort_values("forecast_kst_dtm").reset_index(drop=True)


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {
        "forecast_id",
        "forecast_kst_dtm",
        "fold_id",
        "target",
    }
    return [name for name in frame if name not in excluded]


def _fit_predict(
    spec: Spec,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    features: list[str],
) -> np.ndarray:
    eligible = train["target"].ge(0.10) & train["target"].notna()
    x_train = train.loc[eligible, features].astype("float32")
    y_train = train.loc[eligible, "target"].astype(float)
    x_valid = valid[features].astype("float32")
    weights = y_train.clip(lower=0.10)
    if spec.family == "ridge":
        model = Ridge(alpha=spec.alpha)
        model.fit(x_train, y_train, sample_weight=weights)
    else:
        model = LGBMRegressor(
            objective=spec.objective,
            n_estimators=spec.iterations,
            learning_rate=0.025,
            num_leaves=spec.num_leaves,
            min_child_samples=80,
            max_bin=127,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.9,
            reg_alpha=0.2,
            reg_lambda=4.0,
            random_state=20260802,
            n_jobs=6,
            deterministic=True,
            force_col_wise=True,
            verbosity=-1,
        )
        model.fit(x_train, y_train, sample_weight=weights)
    return np.clip(model.predict(x_valid), 0.0, 1.075)


def _group_score(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(actual) & (actual >= 0.10)
    actual = actual[valid]
    prediction = prediction[valid]
    error = np.abs(prediction - actual)
    unit = np.select([error <= 0.06, error <= 0.08], [4.0, 3.0], default=0.0)
    one_minus_nmae = 1.0 - float(error.mean())
    ficr = float((actual * unit).sum() / (4.0 * actual.sum()))
    return {
        "total": 0.5 * (one_minus_nmae + ficr),
        "one_minus_nmae": one_minus_nmae,
        "ficr": ficr,
    }


def _apply_policy(
    frame: pd.DataFrame,
    model_prediction: np.ndarray,
    policy: tuple[float, float, float, bool],
) -> np.ndarray:
    base_weight, scale, offset, snap = policy
    prediction = base_weight * frame["p3"].to_numpy(dtype=float) + (
        1.0 - base_weight
    ) * model_prediction
    prediction = prediction * scale + offset
    if snap:
        prediction = np.round(prediction / 0.0025) * 0.0025
    return np.clip(prediction, 0.0, 1.075)


def _select(
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    features: list[str],
) -> tuple[Spec, tuple[float, float, float, bool], dict[str, object]]:
    diagnostics: dict[str, object] = {}
    best: tuple[float, Spec, tuple[float, float, float, bool]] | None = None
    for spec in SPECS:
        model_prediction = _fit_predict(spec, train, calibration, features)
        spec_best: tuple[float, tuple[float, float, float, bool], dict[str, float]] | None = (
            None
        )
        for base_weight in (0.0, 0.25, 0.5, 0.75, 1.0):
            for scale in (0.98, 1.0, 1.02):
                for offset in (-0.01, 0.0, 0.01):
                    for snap in (False, True):
                        policy = (base_weight, scale, offset, snap)
                        prediction = _apply_policy(
                            calibration, model_prediction, policy
                        )
                        score = _group_score(
                            calibration["target"].to_numpy(dtype=float), prediction
                        )
                        choice = (score["total"], policy, score)
                        if spec_best is None or choice[0] > spec_best[0]:
                            spec_best = choice
        assert spec_best is not None
        diagnostics[spec.name] = {
            "policy": spec_best[1],
            "score": spec_best[2],
        }
        choice = (spec_best[0], spec, spec_best[1])
        if best is None or choice[0] > best[0]:
            best = choice
    assert best is not None
    return best[1], best[2], diagnostics


def _stage(
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    application: pd.DataFrame,
    features: list[str],
) -> tuple[np.ndarray, dict[str, object]]:
    spec, policy, diagnostics = _select(train, calibration, features)
    prediction = _fit_predict(
        spec,
        pd.concat([train, calibration], ignore_index=True),
        application,
        features,
    )
    output = _apply_policy(application, prediction, policy)
    return output, {
        "selected_spec": asdict(spec),
        "selected_policy": policy,
        "calibration_diagnostics": diagnostics,
        "application_group3_score": _group_score(
            application["target"].to_numpy(dtype=float), output
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-id", default="M105_CROSS_GROUP_STACK")
    args = parser.parse_args()
    if _sha256(OPEN) != OPEN_SHA or _sha256(BASELINE) != BASELINE_SHA:
        raise RuntimeError("immutable competition input hash mismatch")
    started = time.perf_counter()
    parent = pd.read_parquet(OUTPUT / "M103_STRICT_TOP100-oof.parquet")
    wide = _wide(parent)
    features = _feature_columns(wide)
    quarters = {
        fold: wide.loc[wide["fold_id"].eq(fold)].copy() for fold in FOLDS
    }
    q2_times = quarters[FOLDS[0]]["forecast_kst_dtm"].drop_duplicates().sort_values()
    cutoff = q2_times.iloc[int(len(q2_times) * 0.60)]
    q2_early = quarters[FOLDS[0]].loc[
        quarters[FOLDS[0]]["forecast_kst_dtm"].lt(cutoff)
    ]
    q2_late = quarters[FOLDS[0]].loc[
        quarters[FOLDS[0]]["forecast_kst_dtm"].ge(cutoff)
    ]
    q3_prediction, q3_receipt = _stage(
        q2_early, q2_late, quarters[FOLDS[1]], features
    )
    q4_prediction, q4_receipt = _stage(
        quarters[FOLDS[0]], quarters[FOLDS[1]], quarters[FOLDS[2]], features
    )
    replacements = {
        FOLDS[1]: dict(
            zip(quarters[FOLDS[1]]["forecast_id"], q3_prediction, strict=True)
        ),
        FOLDS[2]: dict(
            zip(quarters[FOLDS[2]]["forecast_id"], q4_prediction, strict=True)
        ),
    }
    output = parent.copy()
    for fold, mapping in replacements.items():
        replace = output["fold_id"].eq(fold) & output["group_id"].eq(3)
        normalized = output.loc[replace, "forecast_id"].map(mapping).to_numpy(dtype=float)
        output.loc[replace, "prediction_kwh"] = normalized * CAPACITIES[3]
    output["model_id"] = args.candidate_id
    output_path = OUTPUT / f"{args.candidate_id}-oof.parquet"
    output.to_parquet(output_path, index=False)
    receipt = {
        "schema_version": 1,
        "candidate_id": args.candidate_id,
        "scope": "Q2 control; Q3 selected within Q2; Q4 selected on Q3",
        "parent_candidate_id": "M103_STRICT_TOP100",
        "feature_count": len(features),
        "q3_stage": q3_receipt,
        "q4_stage": q4_receipt,
        "fold_scores": {
            fold: _score(output.loc[output["fold_id"].eq(fold)]) for fold in FOLDS
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
