"""Screen one fixed Tweedie point model as a small M107 diversity blend.

The Q2-selected M102 feature list and a bounded blend grid are calibration inputs.
Q2 output remains exact M107; selected group weights are frozen for Q3 and Q4.
Only the physically filtered pre-2024 surface is materialized, and no test build
or external action is permitted by this runner.
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
from lightgbm import LGBMRegressor
from run_conditional_daily_analog_profile import BASELINE, BASELINE_SHA, OPEN, OPEN_SHA
from run_site_wind_classifier import _add_site_wind_features
from run_strict_parent_analog_transfer import FOLD_MAP, _pooled, _strict_parents
from run_temporal_reconciliation import _verified_m107_receipt
from strict_dev_surface import DEV_CUTOFF, development_surface

from baram.contracts.hashing import canonical_sha256, sha256_file

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "artifacts" / "backtests" / "metric-aligned-probe"
MODEL_ID = "M258_TWEEDIE_TRUST_BLEND"
FEATURE_RECEIPT = OUTPUT / "M102_TOP100-dev-2023-Q2.json"
FEATURE_RECEIPT_SHA = (
    "17f7ea69f4eca3e1c9500ae2df6f6c2361a6eeca9d8a953551825f476d5b06eb"
)
SITEWIND_PATHS = {
    fold: OUTPUT / f"M64B_ALLWEATHER_SITEWIND_CLASS-{long_fold}-sitewind-features.npz"
    for fold, long_fold in FOLD_MAP.items()
}
SITEWIND_SHAS = {
    "q2": "ef37ea3bf4d59f855e5938bf2ba6319fe9d549fc40da4bbb24f9c26f2d55bd9a",
    "q3": "e8571f126bd0aa4748b5531efc3eb9a943a4dc3e6d8324d07eb1351738686c0d",
    "q4": "e2a55b5eb87b4573ca707c0bde4b39ff9297048950d391b25cd6484bb39a0f57",
}
N_ESTIMATORS = 60
VARIANCE_POWER = 1.5
NEW_MODEL_WEIGHTS = (0.0, 0.05, 0.10, 0.15, 0.20)
LOWER_BOUND = 0.0
UPPER_BOUND = 1.075
KEYS = ["forecast_id", "forecast_kst_dtm", "group_id"]


def _feature_names() -> list[str]:
    if sha256_file(FEATURE_RECEIPT) != FEATURE_RECEIPT_SHA:
        raise RuntimeError("M258 Q2 feature receipt hash mismatch")
    receipt = json.loads(FEATURE_RECEIPT.read_text(encoding="utf-8"))
    names = list(receipt["selected_feature_names"])
    if (
        receipt["candidate_id"] != "M102_TOP100"
        or receipt["fold_id"] != FOLD_MAP["q2"]
        or receipt["selected_iteration"] != N_ESTIMATORS
        or receipt["top_features"] != 100
        or len(names) != 100
        or len(set(names)) != 100
        or receipt.get("new_2024_evaluation")
        or receipt.get("lockbox_reopened")
        or receipt.get("external_actions")
    ):
        raise RuntimeError("M258 Q2 feature contract changed")
    return names


def _matrix_for_fold(
    surface: pd.DataFrame,
    numeric: list[str],
    fold: str,
    feature_names: list[str],
) -> tuple[pd.DataFrame, dict[str, object]]:
    path = SITEWIND_PATHS[fold]
    if sha256_file(path) != SITEWIND_SHAS[fold]:
        raise RuntimeError(f"M258 {fold} site-wind cache hash mismatch")
    with np.load(path) as cached:
        if set(cached.files) != {"legacy", "allweather", "iterations"}:
            raise RuntimeError(f"M258 {fold} site-wind cache schema changed")
        if len(cached["legacy"]) < len(surface) or len(cached["allweather"]) < len(
            surface
        ):
            raise RuntimeError(f"M258 {fold} site-wind cache is too short")
        legacy = np.asarray(cached["legacy"][: len(surface)], dtype=float)
        allweather = np.asarray(cached["allweather"][: len(surface)], dtype=float)
        iterations = np.asarray(cached["iterations"], dtype=int).tolist()
    matrix = surface[numeric].astype("float32").copy()
    added = _add_site_wind_features(matrix, legacy, allweather)
    missing = sorted(set(feature_names).difference(matrix.columns))
    if missing or len(added) != 14:
        raise RuntimeError(f"M258 feature surface changed: missing={missing}")
    return matrix[feature_names], {
        "cache_path": str(path.relative_to(ROOT)),
        "cache_sha256": SITEWIND_SHAS[fold],
        "sitewind_columns": added,
        "sitewind_selected_iterations": iterations,
    }


def _model(seed: int) -> LGBMRegressor:
    return LGBMRegressor(
        objective="tweedie",
        tweedie_variance_power=VARIANCE_POWER,
        n_estimators=N_ESTIMATORS,
        learning_rate=0.025,
        num_leaves=15,
        min_child_samples=80,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=2.0,
        random_state=seed,
        n_jobs=6,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )


def _raw_fold_prediction(
    surface: pd.DataFrame,
    matrix: pd.DataFrame,
    parent: pd.DataFrame,
    fold: str,
    boundary_issuances: set[pd.Timestamp],
) -> tuple[pd.DataFrame, dict[str, object]]:
    output = parent.reset_index(drop=True).copy()
    output["tweedie_prediction_kwh"] = output["prediction_kwh"].to_numpy(dtype=float)
    output["tweedie_eligible"] = False
    lookup = surface.reset_index(names="surface_position")[[*KEYS, "surface_position"]]
    parent_keys = output.reset_index(names="parent_position")[[
        "parent_position",
        *KEYS,
    ]]
    aligned = parent_keys.merge(lookup, on=KEYS, how="left", validate="one_to_one")
    if not np.array_equal(
        aligned["parent_position"].to_numpy(dtype=int),
        np.arange(len(output), dtype=int),
    ):
        raise RuntimeError(f"M258 {fold} parent/surface merge reordered rows")
    positions = aligned["surface_position"].to_numpy(dtype=float)
    has_surface = np.isfinite(positions)
    boundary = output["data_available_kst_dtm"].isin(boundary_issuances).to_numpy()
    eligible = has_surface & ~boundary
    cutoff = pd.Timestamp(output["data_available_kst_dtm"].min())
    if cutoff >= DEV_CUTOFF:
        raise RuntimeError("M258 fold cutoff reached the lockbox")

    diagnostics: dict[str, object] = {
        "training_cutoff": str(cutoff),
        "eligible_prediction_rows": int(eligible.sum()),
        "parent_fallback_rows": int((~eligible).sum()),
        "groups": {},
    }
    normalized_target = surface["actual_kwh"] / surface["group_id"].map(CAPACITIES)
    for group_offset, group_id in enumerate(CAPACITIES, start=1):
        group = surface["group_id"].eq(group_id).to_numpy()
        training = (
            group
            & surface["forecast_kst_dtm"].lt(cutoff).to_numpy()
            & np.isfinite(normalized_target.to_numpy(dtype=float))
        )
        if int(training.sum()) < 1000:
            raise RuntimeError(f"M258 {fold} group {group_id} training rows changed")
        group_apply = eligible & output["group_id"].eq(group_id).to_numpy()
        apply_surface_positions = positions[group_apply].astype(int)
        if not np.isfinite(matrix.iloc[apply_surface_positions].to_numpy(dtype=float)).any(
            axis=1
        ).all():
            raise RuntimeError(f"M258 {fold} group {group_id} has empty feature rows")
        model = _model(20260803 + 10 * list(FOLD_MAP).index(fold) + group_offset)
        model.fit(matrix.loc[training], normalized_target.loc[training])
        normalized_prediction = np.clip(
            np.asarray(model.predict(matrix.iloc[apply_surface_positions]), dtype=float),
            LOWER_BOUND,
            UPPER_BOUND,
        )
        if not np.isfinite(normalized_prediction).all():
            raise RuntimeError(f"M258 {fold} group {group_id} prediction is non-finite")
        output.loc[group_apply, "tweedie_prediction_kwh"] = (
            normalized_prediction * CAPACITIES[group_id]
        )
        output.loc[group_apply, "tweedie_eligible"] = True
        diagnostics["groups"][str(group_id)] = {
            "training_rows": int(training.sum()),
            "training_target_mean": float(normalized_target.loc[training].mean()),
            "training_zero_fraction": float(normalized_target.loc[training].eq(0.0).mean()),
            "prediction_rows": int(group_apply.sum()),
            "prediction_mean": float(np.mean(normalized_prediction)),
            "prediction_min": float(np.min(normalized_prediction)),
            "prediction_max": float(np.max(normalized_prediction)),
        }
    if not output.loc[~eligible, "tweedie_prediction_kwh"].equals(
        output.loc[~eligible, "prediction_kwh"]
    ):
        raise RuntimeError("M258 changed a boundary fallback row")
    return output, diagnostics


def _group_score(frame: pd.DataFrame, group_id: int, prediction: str) -> dict[str, float]:
    capacity = CAPACITIES[group_id]
    group = frame.loc[frame["group_id"].eq(group_id)]
    valid = group["actual_kwh"].to_numpy(dtype=float) >= 0.10 * capacity
    actual = group.loc[valid, "actual_kwh"].to_numpy(dtype=float)
    predicted = group.loc[valid, prediction].to_numpy(dtype=float)
    error = np.abs(predicted - actual) / capacity
    units = np.select([error <= 0.06, error <= 0.08], [4.0, 3.0], default=0.0)
    one_minus_nmae = 1.0 - float(error.mean())
    ficr = float(np.sum(actual * units) / np.sum(actual * 4.0))
    return {
        "total": 0.5 * (one_minus_nmae + ficr),
        "one_minus_nmae": one_minus_nmae,
        "ficr": ficr,
    }


def _select_q2_weights(raw_q2: pd.DataFrame) -> dict[int, dict[str, object]]:
    selections: dict[int, dict[str, object]] = {}
    for group_id in CAPACITIES:
        group = raw_q2.loc[raw_q2["group_id"].eq(group_id)].copy()
        if not group["tweedie_eligible"].all():
            raise RuntimeError(f"M258 Q2 group {group_id} has fallback rows")
        parent = group["prediction_kwh"].to_numpy(dtype=float)
        tweedie = group["tweedie_prediction_kwh"].to_numpy(dtype=float)
        sweep: dict[str, dict[str, float]] = {}
        best: tuple[float, float, dict[str, float]] | None = None
        for weight in NEW_MODEL_WEIGHTS:
            group["candidate_prediction_kwh"] = np.clip(
                (1.0 - weight) * parent + weight * tweedie,
                LOWER_BOUND,
                UPPER_BOUND * CAPACITIES[group_id],
            )
            score = _group_score(group, group_id, "candidate_prediction_kwh")
            sweep[f"{weight:.2f}"] = score
            choice = (score["total"], -weight, score)
            if best is None or choice[:2] > best[:2]:
                best = choice
        assert best is not None
        selected_weight = -best[1]
        selections[group_id] = {
            "new_model_weight": selected_weight,
            "parent_weight": 1.0 - selected_weight,
            "q2_group_score": best[2],
            "sweep": sweep,
        }
    return selections


def _apply_weights(
    parent: pd.DataFrame,
    raw: pd.DataFrame,
    fold: str,
    selections: dict[int, dict[str, object]],
) -> tuple[pd.DataFrame, dict[str, object]]:
    output = parent.reset_index(drop=True).copy()
    original = output["prediction_kwh"].to_numpy(dtype=float).copy()
    if not output[KEYS].equals(raw[KEYS]):
        raise RuntimeError(f"M258 {fold} raw/parent key alignment changed")
    if fold == "q2":
        return output, {"policy": "exact_parent_control", "changed_rows": 0}
    group_diagnostics: dict[str, object] = {}
    for group_id in CAPACITIES:
        weight = float(selections[group_id]["new_model_weight"])
        mask = (
            output["group_id"].eq(group_id).to_numpy()
            & raw["tweedie_eligible"].to_numpy(dtype=bool)
        )
        blended = (1.0 - weight) * original[mask] + weight * raw.loc[
            mask, "tweedie_prediction_kwh"
        ].to_numpy(dtype=float)
        output.loc[mask, "prediction_kwh"] = np.clip(
            blended,
            LOWER_BOUND,
            UPPER_BOUND * CAPACITIES[group_id],
        )
        group_diagnostics[str(group_id)] = {
            "new_model_weight": weight,
            "eligible_rows": int(mask.sum()),
            "changed_rows": int(
                np.count_nonzero(
                    output.loc[mask, "prediction_kwh"].to_numpy(dtype=float)
                    != original[mask]
                )
            ),
        }
    fallback = ~raw["tweedie_eligible"].to_numpy(dtype=bool)
    if not np.array_equal(
        output.loc[fallback, "prediction_kwh"].to_numpy(dtype=float),
        original[fallback],
    ):
        raise RuntimeError(f"M258 {fold} changed a fallback row")
    return output, {
        "policy": "q2_selected_group_trust_weights",
        "groups": group_diagnostics,
        "changed_rows": int(
            np.count_nonzero(output["prediction_kwh"].to_numpy(dtype=float) != original)
        ),
        "fallback_rows": int(fallback.sum()),
    }


def main() -> None:
    if sha256_file(OPEN) != OPEN_SHA or sha256_file(BASELINE) != BASELINE_SHA:
        raise RuntimeError("immutable competition input hash mismatch")
    m107_receipt = _verified_m107_receipt()
    surface, numeric = development_surface()
    if surface["forecast_kst_dtm"].ge(DEV_CUTOFF).any():
        raise RuntimeError("lockbox row reached M258")
    feature_names = _feature_names()
    parents, boundary_fallback = _strict_parents(surface)
    boundary_issuances = {
        pd.Timestamp(value) for value in boundary_fallback.get("issuances", [])
    }

    raw: dict[str, pd.DataFrame] = {}
    model_diagnostics: dict[str, object] = {}
    cache_diagnostics: dict[str, object] = {}
    raw_scores: dict[str, dict[str, float]] = {}
    for fold in FOLD_MAP:
        matrix, cache_diagnostics[fold] = _matrix_for_fold(
            surface,
            numeric,
            fold,
            feature_names,
        )
        raw[fold], model_diagnostics[fold] = _raw_fold_prediction(
            surface,
            matrix,
            parents[fold],
            fold,
            boundary_issuances,
        )
        raw_frame = raw[fold].copy()
        raw_frame["prediction_kwh"] = raw_frame["tweedie_prediction_kwh"]
        raw_scores[fold] = _score(raw_frame)

    selections = _select_q2_weights(raw["q2"])
    outputs: dict[str, pd.DataFrame] = {}
    application_diagnostics: dict[str, object] = {}
    for fold in FOLD_MAP:
        outputs[fold], application_diagnostics[fold] = _apply_weights(
            parents[fold],
            raw[fold],
            fold,
            selections,
        )
    if not np.array_equal(
        outputs["q2"]["prediction_kwh"].to_numpy(dtype=float),
        parents["q2"]["prediction_kwh"].to_numpy(dtype=float),
    ):
        raise RuntimeError("M258 Q2 control changed")

    fold_scores: dict[str, object] = {}
    fold_deltas: dict[str, float] = {}
    for fold in FOLD_MAP:
        parent_score = _score(parents[fold])
        output_score = _score(outputs[fold])
        expected = m107_receipt["fold_scores"][FOLD_MAP[fold]]
        for metric in ("total", "one_minus_nmae", "ficr"):
            if abs(parent_score[metric] - expected[metric]) > 1e-12:
                raise RuntimeError(f"M107 {fold} {metric} reproduction changed")
        fold_deltas[fold] = output_score["total"] - parent_score["total"]
        fold_scores[fold] = {
            "parent": parent_score,
            "tweedie_raw": raw_scores[fold],
            "blended": output_score,
            "deltas": {
                metric: output_score[metric] - parent_score[metric]
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
        "architecture": "fixed_group_lightgbm_tweedie_small_trust_blend",
        "parent": "M107_STRICT_TEMPORAL_TOP100",
        "feature_source": "M102_Q2_selected_top100_frozen_for_all_folds",
        "sitewind_source": "fold_specific_M64B_strict_preceding_crossfit_cache",
        "objective": "tweedie_log_link",
        "tweedie_variance_power": VARIANCE_POWER,
        "n_estimators": N_ESTIMATORS,
        "learning_rate": 0.025,
        "num_leaves": 15,
        "min_child_samples": 80,
        "training_target": "all_preceding_finite_capacity_normalized_labels",
        "sample_weight": None,
        "q2_new_model_weight_grid": list(NEW_MODEL_WEIGHTS),
        "q2_policy": "exact_parent_control_and_weight_selection_only",
        "q3_q4_policy": "same_q2_selected_group_weights",
        "boundary_policy": "exact_M107_parent_for_final_partial_metadata_issuance",
        "feature_search": False,
        "tree_search": False,
        "variance_power_search": False,
        "sample_weight_search": False,
        "scale_offset_search": False,
        "group_exception_search": False,
        "quarter_exception_search": False,
        "test_build": False,
    }
    receipt = {
        "schema_version": 1,
        "candidate_id": MODEL_ID,
        "state": (
            "LOCAL_TWEEDIE_TRUST_BLEND_PROMOTED_NO_TEST_BUILD"
            if promoted
            else "LOCAL_TWEEDIE_TRUST_BLEND_REJECTED_NO_TEST_BUILD"
        ),
        "policy": policy,
        "policy_sha256": canonical_sha256(policy),
        "selected_q2_weights": {
            str(group): value for group, value in selections.items()
        },
        "fold_scores": fold_scores,
        "pooled": {
            "parent": pooled_parent,
            "blended": pooled_output,
            "deltas": pooled_deltas,
        },
        "q4_paired_bootstrap": q4_bootstrap,
        "promotion": promotion,
        "model_diagnostics": model_diagnostics,
        "cache_diagnostics": cache_diagnostics,
        "application_diagnostics": application_diagnostics,
        "boundary_fallback": boundary_fallback,
        "source_receipts": {
            "open_zip_sha256": OPEN_SHA,
            "baseline_ipynb_sha256": BASELINE_SHA,
            "m107_prediction_sha256": m107_receipt["prediction_sha256"],
            "feature_receipt_sha256": FEATURE_RECEIPT_SHA,
            "sitewind_cache_sha256": SITEWIND_SHAS,
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
