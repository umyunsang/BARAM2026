"""Screen validation-weighted daily/hourly reconciliation on exact M107 OOF.

M257 adds one independent daily-mean forecast to the 24 hourly M107 leaves.
Q2 is retained exactly and supplies the first validation-error block. Q3 uses
weights learned only from Q2; Q4 uses weights learned only from Q2+Q3. The
consumed 2024 lockbox is never materialized and no test submission is built.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from build_v2_transfer_sequence_challenger import (
    CAPACITIES,
    METRIC_COLUMNS,
    _paired_issuance_bootstrap,
    _score,
)
from run_conditional_daily_analog_profile import (
    BASELINE,
    BASELINE_SHA,
    OPEN,
    OPEN_SHA,
    _complete_group_days,
    _feature_sets,
)
from run_strict_parent_analog_transfer import (
    FOLD_MAP,
    M107_PREDICTION,
    M107_PREDICTION_SHA,
    M107_RECEIPT,
    M107_RECEIPT_SHA,
    _pooled,
    _strict_parents,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from strict_dev_surface import DEV_CUTOFF, development_surface

from baram.contracts.hashing import canonical_sha256, sha256_file

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "artifacts" / "backtests" / "metric-aligned-probe"
MODEL_ID = "M257_VALIDATION_WEIGHTED_TEMPORAL_RECONCILIATION"
RIDGE_ALPHA = 1000.0
LOWER_BOUND = 0.0
UPPER_BOUND = 1.075
MIN_TRAIN_DAYS = 60
KEYS = ["forecast_id", "forecast_kst_dtm", "group_id"]


def _verified_m107_receipt() -> dict[str, object]:
    if sha256_file(M107_PREDICTION) != M107_PREDICTION_SHA:
        raise RuntimeError("M107 strict prediction hash mismatch")
    if sha256_file(M107_RECEIPT) != M107_RECEIPT_SHA:
        raise RuntimeError("M107 strict receipt hash mismatch")
    receipt = json.loads(M107_RECEIPT.read_text(encoding="utf-8"))
    if (
        receipt["candidate_id"] != "M107_STRICT_TEMPORAL_TOP100"
        or receipt["prediction_sha256"] != M107_PREDICTION_SHA
        or receipt.get("new_2024_evaluation")
        or receipt.get("lockbox_reopened")
        or receipt.get("external_actions")
    ):
        raise RuntimeError("M107 strict evidence boundary changed")
    return receipt


def _project_bounded_mean(
    profile: np.ndarray,
    target_mean: float,
) -> np.ndarray:
    values = np.asarray(profile, dtype=float)
    if values.shape != (24,) or not np.isfinite(values).all():
        raise ValueError("coherence projection requires one finite 24-hour profile")
    target = float(np.clip(target_mean, LOWER_BOUND, UPPER_BOUND))
    if (
        np.min(values) >= LOWER_BOUND
        and np.max(values) <= UPPER_BOUND
        and abs(float(values.mean()) - target) <= 1e-14
    ):
        return values.copy()

    low = LOWER_BOUND - float(np.max(values)) - 1.0
    high = UPPER_BOUND - float(np.min(values)) + 1.0
    for _ in range(100):
        midpoint = 0.5 * (low + high)
        candidate_mean = float(
            np.clip(values + midpoint, LOWER_BOUND, UPPER_BOUND).mean()
        )
        if candidate_mean < target:
            low = midpoint
        else:
            high = midpoint
    projected = np.clip(
        values + 0.5 * (low + high),
        LOWER_BOUND,
        UPPER_BOUND,
    )
    if abs(float(projected.mean()) - target) > 1e-12:
        raise RuntimeError("bounded daily/hourly coherence projection failed")
    return projected


def _self_test_projection() -> None:
    profile = np.linspace(0.05, 0.95, 24)
    identity = _project_bounded_mean(profile, float(profile.mean()))
    if not np.allclose(identity, profile, rtol=0.0, atol=1e-13):
        raise RuntimeError("M257 identity projection self-test failed")
    for target in (0.0, 0.2, 0.8, UPPER_BOUND):
        projected = _project_bounded_mean(profile, target)
        if (
            np.min(projected) < LOWER_BOUND
            or np.max(projected) > UPPER_BOUND
            or abs(float(projected.mean()) - target) > 1e-12
        ):
            raise RuntimeError("M257 bounded projection self-test failed")


def _complete_parent_issuances(
    parent_group: pd.DataFrame,
    surface_issuances: np.ndarray,
) -> list[pd.Timestamp]:
    counts = parent_group.groupby("data_available_kst_dtm", sort=True).size()
    parent_complete = set(pd.to_datetime(counts.loc[counts.eq(24)].index))
    surface_complete = set(pd.to_datetime(surface_issuances))
    return sorted(parent_complete.intersection(surface_complete))


def _daily_model() -> object:
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        Ridge(alpha=RIDGE_ALPHA),
    )


def _daily_predictions(
    surface: pd.DataFrame,
    core_features: list[str],
    parents: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, dict[str, object]]:
    records: list[dict[str, object]] = []
    diagnostics: dict[str, object] = {}
    for group_id in CAPACITIES:
        frame, issuances, values, targets = _complete_group_days(
            surface,
            group_id,
            core_features,
        )
        issuance_index = {
            pd.Timestamp(value): position for position, value in enumerate(issuances)
        }
        day_end = (
            frame.groupby("data_available_kst_dtm", sort=True)["forecast_kst_dtm"]
            .max()
            .to_numpy()
        )
        group_diagnostics: dict[str, object] = {}
        for fold in FOLD_MAP:
            parent_group = parents[fold].loc[
                parents[fold]["group_id"].eq(group_id)
            ].copy()
            query_issuances = _complete_parent_issuances(parent_group, issuances)
            if not query_issuances:
                raise RuntimeError(f"M257 {fold} group {group_id} has no complete days")
            query_mask = np.isin(
                pd.to_datetime(issuances),
                np.asarray(query_issuances, dtype="datetime64[ns]"),
            )
            cutoff = pd.Timestamp(query_issuances[0])
            train_mask = (
                (day_end < cutoff)
                & np.isfinite(targets).all(axis=1)
                & np.isfinite(values).any(axis=(1, 2))
            )
            if int(train_mask.sum()) < MIN_TRAIN_DAYS:
                raise RuntimeError(
                    f"M257 {fold} group {group_id} has only "
                    f"{int(train_mask.sum())} strict training days"
                )
            train_x = values[train_mask].reshape(int(train_mask.sum()), -1)
            train_y = targets[train_mask].mean(axis=1)
            model = _daily_model()
            model.fit(
                train_x,
                train_y,
                ridge__sample_weight=np.clip(train_y, 0.10, None),
            )
            query_positions = np.flatnonzero(query_mask)
            query_x = values[query_mask].reshape(int(query_mask.sum()), -1)
            daily_prediction = np.clip(
                np.asarray(model.predict(query_x), dtype=float),
                LOWER_BOUND,
                UPPER_BOUND,
            )
            if len(query_positions) != len(query_issuances):
                raise RuntimeError("M257 query-day alignment changed")

            parent_means: list[float] = []
            actual_means: list[float] = []
            for local_index, issuance in enumerate(query_issuances):
                position = issuance_index[issuance]
                block = parent_group.loc[
                    parent_group["data_available_kst_dtm"].eq(issuance)
                ].sort_values("forecast_kst_dtm")
                if len(block) != 24:
                    raise RuntimeError("M257 complete parent block changed")
                parent_normalized = (
                    block["prediction_kwh"].to_numpy(dtype=float)
                    / CAPACITIES[group_id]
                )
                actual_normalized = (
                    block["actual_kwh"].to_numpy(dtype=float) / CAPACITIES[group_id]
                )
                if not np.allclose(
                    actual_normalized,
                    targets[position],
                    rtol=0.0,
                    atol=1e-12,
                ):
                    raise RuntimeError("M257 parent/surface target alignment changed")
                parent_mean = float(parent_normalized.mean())
                actual_mean = float(actual_normalized.mean())
                parent_means.append(parent_mean)
                actual_means.append(actual_mean)
                records.append(
                    {
                        "fold": fold,
                        "group_id": int(group_id),
                        "data_available_kst_dtm": issuance,
                        "daily_prediction": float(daily_prediction[local_index]),
                        "parent_mean": parent_mean,
                        "actual_mean": actual_mean,
                    }
                )
            parent_array = np.asarray(parent_means, dtype=float)
            actual_array = np.asarray(actual_means, dtype=float)
            group_diagnostics[fold] = {
                "training_days": int(train_mask.sum()),
                "query_days": len(query_issuances),
                "parent_daily_mae": float(np.mean(np.abs(parent_array - actual_array))),
                "daily_model_mae": float(
                    np.mean(np.abs(daily_prediction - actual_array))
                ),
                "daily_model_mean": float(np.mean(daily_prediction)),
                "actual_daily_mean": float(np.mean(actual_array)),
            }
        diagnostics[str(group_id)] = group_diagnostics
    daily = pd.DataFrame.from_records(records)
    if daily.duplicated(["fold", "group_id", "data_available_kst_dtm"]).any():
        raise RuntimeError("M257 daily prediction keys are not unique")
    return daily, diagnostics


def _validation_weight(
    daily: pd.DataFrame,
    group_id: int,
    calibration_folds: tuple[str, ...],
) -> dict[str, object]:
    calibration = daily.loc[
        daily["group_id"].eq(group_id) & daily["fold"].isin(calibration_folds)
    ]
    delta = (
        calibration["daily_prediction"].to_numpy(dtype=float)
        - calibration["parent_mean"].to_numpy(dtype=float)
    )
    residual = (
        calibration["actual_mean"].to_numpy(dtype=float)
        - calibration["parent_mean"].to_numpy(dtype=float)
    )
    denominator = float(np.dot(delta, delta))
    numerator = float(np.dot(delta, residual))
    raw_weight = 0.0 if denominator <= 1e-15 else numerator / denominator
    weight = float(np.clip(raw_weight, 0.0, 1.0))
    reconciled = (
        calibration["parent_mean"].to_numpy(dtype=float) + weight * delta
    )
    actual = calibration["actual_mean"].to_numpy(dtype=float)
    return {
        "calibration_folds": list(calibration_folds),
        "calibration_days": len(calibration),
        "numerator": numerator,
        "denominator": denominator,
        "raw_weight": raw_weight,
        "weight": weight,
        "parent_daily_mae": float(
            np.mean(
                np.abs(
                    calibration["parent_mean"].to_numpy(dtype=float) - actual
                )
            )
        ),
        "daily_model_mae": float(
            np.mean(
                np.abs(
                    calibration["daily_prediction"].to_numpy(dtype=float)
                    - actual
                )
            )
        ),
        "reconciled_daily_mae": float(np.mean(np.abs(reconciled - actual))),
    }


def _apply_reconciliation(
    parent: pd.DataFrame,
    daily: pd.DataFrame,
    fold: str,
    weights: dict[int, dict[str, object]],
) -> tuple[pd.DataFrame, dict[str, object]]:
    output = parent.reset_index(drop=True).copy()
    original = output["prediction_kwh"].to_numpy(dtype=float).copy()
    eligible = np.zeros(len(output), dtype=bool)
    coherence_errors: list[float] = []
    target_shifts: list[float] = []
    group_diagnostics: dict[str, object] = {}
    for group_id in CAPACITIES:
        group_records = daily.loc[
            daily["fold"].eq(fold) & daily["group_id"].eq(group_id)
        ].sort_values("data_available_kst_dtm")
        weight = float(weights[group_id]["weight"])
        changed_blocks = 0
        for record in group_records.itertuples(index=False):
            positions = np.flatnonzero(
                output["group_id"].eq(group_id).to_numpy()
                & output["data_available_kst_dtm"]
                .eq(record.data_available_kst_dtm)
                .to_numpy()
            )
            if len(positions) != 24:
                raise RuntimeError("M257 application block is not 24 hours")
            eligible[positions] = True
            if weight <= 0.0:
                continue
            order = np.argsort(
                output.iloc[positions]["forecast_kst_dtm"].to_numpy()
            )
            ordered_positions = positions[order]
            parent_profile = (
                output.iloc[ordered_positions]["prediction_kwh"].to_numpy(dtype=float)
                / CAPACITIES[group_id]
            )
            reconciled_mean = float(
                record.parent_mean
                + weight * (record.daily_prediction - record.parent_mean)
            )
            reconciled_profile = _project_bounded_mean(
                parent_profile,
                reconciled_mean,
            )
            output.loc[ordered_positions, "prediction_kwh"] = (
                reconciled_profile * CAPACITIES[group_id]
            )
            coherence_errors.append(
                abs(float(reconciled_profile.mean()) - reconciled_mean)
            )
            target_shifts.append(reconciled_mean - float(record.parent_mean))
            changed_blocks += int(
                not np.array_equal(reconciled_profile, parent_profile)
            )
        group_diagnostics[str(group_id)] = {
            "weight": weight,
            "eligible_complete_blocks": len(group_records),
            "changed_complete_blocks": changed_blocks,
        }
    untouched = ~eligible
    if not np.array_equal(
        output.loc[untouched, "prediction_kwh"].to_numpy(dtype=float),
        original[untouched],
    ):
        raise RuntimeError("M257 changed a partial or unsupported boundary row")
    return output, {
        "groups": group_diagnostics,
        "eligible_rows": int(eligible.sum()),
        "untouched_rows": int(untouched.sum()),
        "changed_rows": int(
            np.count_nonzero(
                output["prediction_kwh"].to_numpy(dtype=float) != original
            )
        ),
        "max_coherence_error": max(coherence_errors, default=0.0),
        "mean_daily_target_shift": float(np.mean(target_shifts))
        if target_shifts
        else 0.0,
        "mean_absolute_daily_target_shift": float(np.mean(np.abs(target_shifts)))
        if target_shifts
        else 0.0,
    }


def main() -> None:
    _self_test_projection()
    if sha256_file(OPEN) != OPEN_SHA or sha256_file(BASELINE) != BASELINE_SHA:
        raise RuntimeError("immutable competition input hash mismatch")
    m107_receipt = _verified_m107_receipt()
    surface, numeric = development_surface()
    if surface["forecast_kst_dtm"].ge(DEV_CUTOFF).any():
        raise RuntimeError("lockbox row reached M257 reconciliation runner")
    core_features = _feature_sets(numeric)["core"]
    if len(core_features) != 25:
        raise RuntimeError("M257 frozen 25-core feature contract changed")
    parents, boundary_fallback = _strict_parents(surface)
    daily, daily_diagnostics = _daily_predictions(
        surface,
        core_features,
        parents,
    )

    weights: dict[str, dict[int, dict[str, object]]] = {
        "q2": {
            group_id: {
                "calibration_folds": [],
                "calibration_days": 0,
                "raw_weight": 0.0,
                "weight": 0.0,
                "policy": "exact_parent_control",
            }
            for group_id in CAPACITIES
        },
        "q3": {
            group_id: _validation_weight(daily, group_id, ("q2",))
            for group_id in CAPACITIES
        },
        "q4": {
            group_id: _validation_weight(daily, group_id, ("q2", "q3"))
            for group_id in CAPACITIES
        },
    }
    outputs: dict[str, pd.DataFrame] = {}
    application_diagnostics: dict[str, object] = {}
    for fold in FOLD_MAP:
        outputs[fold], application_diagnostics[fold] = _apply_reconciliation(
            parents[fold],
            daily,
            fold,
            weights[fold],
        )
    if not np.array_equal(
        outputs["q2"]["prediction_kwh"].to_numpy(dtype=float),
        parents["q2"]["prediction_kwh"].to_numpy(dtype=float),
    ):
        raise RuntimeError("M257 Q2 control changed")

    fold_scores: dict[str, object] = {}
    fold_deltas: dict[str, float] = {}
    for fold in FOLD_MAP:
        parent_score = _score(parents[fold])
        reconciled_score = _score(outputs[fold])
        expected_parent = m107_receipt["fold_scores"][FOLD_MAP[fold]]
        for metric in ("total", "one_minus_nmae", "ficr"):
            if abs(parent_score[metric] - expected_parent[metric]) > 1e-12:
                raise RuntimeError(f"M107 {fold} {metric} reproduction changed")
        fold_deltas[fold] = reconciled_score["total"] - parent_score["total"]
        fold_scores[fold] = {
            "parent": parent_score,
            "reconciled": reconciled_score,
            "deltas": {
                metric: reconciled_score[metric] - parent_score[metric]
                for metric in ("total", "one_minus_nmae", "ficr")
            },
        }

    pooled_parent_frame = _pooled(parents)
    pooled_output_frame = _pooled(outputs)
    pooled_parent = _score(pooled_parent_frame)
    pooled_output = _score(pooled_output_frame)
    for metric in ("total", "one_minus_nmae", "ficr"):
        if abs(pooled_parent[metric] - m107_receipt["pooled"][metric]) > 1e-12:
            raise RuntimeError(f"M107 pooled {metric} reproduction changed")
    pooled_deltas = {
        metric: pooled_output[metric] - pooled_parent[metric]
        for metric in ("total", "one_minus_nmae", "ficr")
    }
    q4_bootstrap = _paired_issuance_bootstrap(parents["q4"], outputs["q4"])
    affected_folds_positive = fold_deltas["q3"] > 0.0 and fold_deltas["q4"] > 0.0
    pooled_total_positive = pooled_deltas["total"] > 0.0
    q4_bootstrap_positive = q4_bootstrap["positive_fraction"] > 0.50
    promoted = (
        affected_folds_positive
        and pooled_total_positive
        and q4_bootstrap_positive
    )
    promotion = {
        "promoted": promoted,
        "q2_exact_parent_control": fold_deltas["q2"] == 0.0,
        "q3_q4_total_deltas_positive": affected_folds_positive,
        "pooled_total_delta_positive": pooled_total_positive,
        "q4_bootstrap_positive_fraction_above_half": q4_bootstrap_positive,
        "rule": (
            "Q2 remains exact M107; Q3 and Q4 Total deltas are positive; "
            "pooled Q2-Q4 Total delta is positive; and Q4 paired issuance "
            "bootstrap positive fraction exceeds 0.50"
        ),
    }

    prediction_path = OUTPUT / f"{MODEL_ID}-oof.parquet"
    prediction_output = pooled_output_frame.copy()
    prediction_output["model_id"] = MODEL_ID
    prediction_output[
        [*METRIC_COLUMNS, "fold_id", "model_id", "data_available_kst_dtm"]
    ].to_parquet(prediction_path, index=False)
    policy = {
        "architecture": "validation_weighted_daily_hourly_temporal_reconciliation",
        "parent": "M107_STRICT_TEMPORAL_TOP100",
        "daily_features": "flattened_exact_M234_25_core_features_over_24_hours",
        "daily_model": {
            "family": "Ridge",
            "alpha": RIDGE_ALPHA,
            "imputation": "per_feature_training_median",
            "standardization": "per_feature_training_mean_and_population_std",
            "target": "normalized_24_hour_mean_generation",
            "sample_weight": "training_daily_mean_clipped_below_at_0.10",
        },
        "sequential_weight_rule": (
            "clipped_no_intercept_OLS_of_actual_minus_parent_mean_on_"
            "daily_model_minus_parent_mean"
        ),
        "q2_policy": "exact_parent_control",
        "q3_calibration_folds": ["q2"],
        "q4_calibration_folds": ["q2", "q3"],
        "projection": "additive_euclidean_projection_to_bounded_fixed_mean",
        "bounds": [LOWER_BOUND, UPPER_BOUND],
        "feature_search": False,
        "model_search": False,
        "alpha_search": False,
        "weight_search": False,
        "group_exception_search": False,
        "quarter_exception_search": False,
        "test_build": False,
    }
    receipt = {
        "schema_version": 1,
        "candidate_id": MODEL_ID,
        "state": (
            "LOCAL_TEMPORAL_RECONCILIATION_PROMOTED_NO_TEST_BUILD"
            if promoted
            else "LOCAL_TEMPORAL_RECONCILIATION_REJECTED_NO_TEST_BUILD"
        ),
        "policy": policy,
        "policy_sha256": canonical_sha256(policy),
        "weights": {
            fold: {str(group): value for group, value in fold_weights.items()}
            for fold, fold_weights in weights.items()
        },
        "fold_scores": fold_scores,
        "pooled": {
            "parent": pooled_parent,
            "reconciled": pooled_output,
            "deltas": pooled_deltas,
        },
        "q4_paired_bootstrap": q4_bootstrap,
        "promotion": promotion,
        "daily_model_diagnostics": daily_diagnostics,
        "application_diagnostics": application_diagnostics,
        "boundary_fallback": boundary_fallback,
        "source_receipts": {
            "open_zip_sha256": OPEN_SHA,
            "baseline_ipynb_sha256": BASELINE_SHA,
            "m107_prediction_sha256": M107_PREDICTION_SHA,
            "m107_receipt_sha256": M107_RECEIPT_SHA,
        },
        "prediction_path": str(prediction_path.relative_to(ROOT)),
        "prediction_sha256": sha256_file(prediction_path),
        "completion_target": {
            "metric": "Dacon Total",
            "strict_threshold": 0.66000,
            "status": "UNVERIFIED_REQUIRES_EXTERNAL_DACON_RESULT",
        },
        "online_score": None,
        "no_external_upload": True,
        "new_2024_evaluation": False,
        "lockbox_reopened": False,
        "external_actions": [],
    }
    receipt_path = OUTPUT / f"{MODEL_ID}.json"
    receipt_path.write_text(
        json.dumps(receipt, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
