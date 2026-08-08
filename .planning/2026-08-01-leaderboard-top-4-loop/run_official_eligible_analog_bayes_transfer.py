"""Evaluate one official-eligibility-aware Bayes head over exact M244 members."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from build_v2_transfer_sequence_challenger import (
    METRIC_COLUMNS,
    OOF,
    _group_score,
    _paired_issuance_bootstrap,
    _score,
)
from run_conditional_daily_analog_profile import (
    ACTIONS,
    BASELINE,
    BASELINE_SHA,
    OPEN,
    OPEN_SHA,
    REPRESENTATIONS,
    Recipe,
    _combine,
    _complete_group_days,
    _distances,
    _feature_sets,
    _kernel_weights,
    _profile_frame,
    _profile_heads,
)
from run_rare_event_corrected_analog_transfer import (
    HUB_FEATURE,
    SLOPE_CAP,
    TAIL_QUANTILE,
    _ols_slope,
    _rare_event_profile,
)
from run_recency_spread_analog_transfer import _composed_profile, _parents
from run_recency_weighted_analog_transfer import HALF_LIFE_DAYS
from run_spread_shrunk_analog_transfer import (
    REFERENCE_QUANTILE,
    _apply_spread_recipe,
    _spread_adjusted_profile,
)
from run_stable_daily_analog_profile import DISTRIBUTION
from strict_dev_surface import DEV_CUTOFF, development_surface

from baram.contracts.hashing import canonical_sha256, sha256_file

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "artifacts" / "backtests" / "metric-aligned-probe"
M244_RECEIPT = OUTPUT / "M244_RARE_EVENT_CORRECTED_ANALOG_Q234.json"
M244_RECEIPT_SHA = (
    "8489f2759ce39fe4abb7c5179386cde0e1a39d22a5d07673a782b4163410a371"
)
MODEL_ID = "M248_OFFICIAL_ELIGIBLE_ANALOG_BAYES_Q234"
FOLDS = ("q2", "q3", "q4")
ELIGIBILITY_THRESHOLD = 0.10
ELIGIBLE_MASS_EPSILON = 1e-12


def _official_utility_action(
    values: np.ndarray,
    weights: np.ndarray,
    mean_eligible_generation: float,
    legacy_action: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    if not np.isfinite(mean_eligible_generation) or mean_eligible_generation <= 0.0:
        raise RuntimeError("M248 eligible-generation denominator is invalid")
    output = legacy_action.copy()
    eligible_mass_values: list[float] = []
    fallback_count = 0
    for query_index in range(len(values)):
        for hour in range(values.shape[2]):
            target = values[query_index, :, hour]
            weight = weights[query_index]
            eligible = target >= ELIGIBILITY_THRESHOLD
            eligible_mass = float(np.sum(weight[eligible]))
            eligible_mass_values.append(eligible_mass)
            if eligible_mass <= ELIGIBLE_MASS_EPSILON:
                fallback_count += 1
                continue
            conditional_weight = np.where(eligible, weight / eligible_mass, 0.0)
            error = np.abs(ACTIONS[:, None] - target[None, :])
            expected_error = error @ conditional_weight
            units = np.select(
                [error <= 0.06, error <= 0.08],
                [4.0, 3.0],
                default=0.0,
            )
            expected_settlement_generation = (
                units * target[None, :]
            ) @ conditional_weight
            expected_ficr = expected_settlement_generation / (
                4.0 * mean_eligible_generation
            )
            expected_total = 0.5 * (1.0 - expected_error) + 0.5 * expected_ficr
            output[query_index, hour] = ACTIONS[int(np.argmax(expected_total))]
    eligible_mass_array = np.asarray(eligible_mass_values, dtype=float)
    return output, {
        "official_eligibility_threshold": ELIGIBILITY_THRESHOLD,
        "mean_eligible_generation": mean_eligible_generation,
        "eligible_mass": {
            "min": float(np.min(eligible_mass_array)),
            "mean": float(np.mean(eligible_mass_array)),
            "median": float(np.median(eligible_mass_array)),
            "max": float(np.max(eligible_mass_array)),
        },
        "legacy_fallback_count": fallback_count,
        "legacy_fallback_fraction": float(fallback_count / output.size),
        "action_changed_count": int(np.count_nonzero(output != legacy_action)),
        "action_changed_fraction": float(np.mean(output != legacy_action)),
    }


def _eligible_rare_event_profiles(
    surface: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    group_id: int,
    query_issuances: np.ndarray,
    recipe: Recipe,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    representation = next(
        item for item in REPRESENTATIONS if item.name == recipe.representation
    )
    features = feature_sets[representation.feature_set]
    if HUB_FEATURE not in features:
        raise RuntimeError("M248 frozen M244 hub-speed feature is absent")
    hub_index = features.index(HUB_FEATURE)
    frame, issuances, values, targets = _complete_group_days(
        surface,
        group_id,
        features,
    )
    query_mask = np.isin(issuances, query_issuances)
    cutoff = pd.Timestamp(np.min(query_issuances))
    day_end = (
        frame.groupby("data_available_kst_dtm", sort=True)["forecast_kst_dtm"]
        .max()
        .reindex(pd.to_datetime(issuances))
        .to_numpy()
    )
    train_mask = (day_end < cutoff) & np.isfinite(targets).all(axis=1)
    if int(train_mask.sum()) < 80 or not query_mask.any():
        raise RuntimeError(
            f"group {group_id} M248 days changed: "
            f"train={int(train_mask.sum())}, query={int(query_mask.sum())}"
        )
    train_values = values[train_mask]
    query_values = values[query_mask]
    train_targets = targets[train_mask]
    train_issuances = issuances[train_mask]
    query_days = issuances[query_mask]
    order, distance, retrieval = _distances(
        train_values,
        query_values,
        train_issuances,
        query_days,
        representation,
    )
    neighbor_indices = order[:, : recipe.neighbors]
    neighbor_distance = distance[:, : recipe.neighbors]
    base_weights = _kernel_weights(neighbor_distance, recipe.kernel)
    age_days = (
        query_days[:, None] - train_issuances[neighbor_indices]
    ) / np.timedelta64(1, "D")
    age_days = np.asarray(age_days, dtype=float)
    if np.any(age_days <= 0.0):
        raise RuntimeError("M248 found a non-historical analog neighbor")
    weights = base_weights * np.exp2(-age_days / HALF_LIFE_DAYS)
    weights /= weights.sum(axis=1, keepdims=True)

    train_hub = train_values[:, :, hub_index].copy()
    query_hub = query_values[:, :, hub_index].copy()
    hub_median = float(np.nanmedian(train_hub))
    train_hub = np.where(np.isfinite(train_hub), train_hub, hub_median)
    query_hub = np.where(np.isfinite(query_hub), query_hub, hub_median)
    raw_slope = _ols_slope(train_hub.reshape(-1), train_targets.reshape(-1))
    slope = float(np.clip(raw_slope, 0.0, SLOPE_CAP))
    threshold = float(np.quantile(train_hub, TAIL_QUANTILE))
    high = query_hub > threshold
    neighbor_hub = train_hub[neighbor_indices]
    neighbor_targets = train_targets[neighbor_indices]
    adjustment = slope * (query_hub[:, None, :] - neighbor_hub)
    corrected_targets = np.where(
        high[:, None, :],
        neighbor_targets + adjustment,
        neighbor_targets,
    )
    corrected_targets = np.clip(corrected_targets, 0.0, 1.0)

    mean_generation = float(np.nanmean(train_targets))
    legacy_heads = _profile_heads(corrected_targets, weights, mean_generation)
    legacy_profile = legacy_heads[recipe.head]
    eligible_targets = train_targets[train_targets >= ELIGIBILITY_THRESHOLD]
    if len(eligible_targets) < 100:
        raise RuntimeError("M248 eligible target support changed")
    mean_eligible_generation = float(np.mean(eligible_targets))
    if recipe.head == "utility":
        corrected_profile, head_diagnostics = _official_utility_action(
            corrected_targets,
            weights,
            mean_eligible_generation,
            legacy_profile,
        )
    elif recipe.head == "median":
        corrected_profile = legacy_profile.copy()
        head_diagnostics = {
            "official_eligibility_threshold": ELIGIBILITY_THRESHOLD,
            "mean_eligible_generation": mean_eligible_generation,
            "legacy_fallback_count": 0,
            "legacy_fallback_fraction": 0.0,
            "action_changed_count": 0,
            "action_changed_fraction": 0.0,
            "head_unchanged": True,
        }
    else:
        raise RuntimeError(f"M248 unexpected frozen head: {recipe.head}")
    return (
        _profile_frame(frame, query_days, legacy_profile, group_id),
        _profile_frame(frame, query_days, corrected_profile, group_id),
        {
            **retrieval,
            "training_days": int(train_mask.sum()),
            "query_days": int(query_mask.sum()),
            "head": recipe.head,
            "head_diagnostics": head_diagnostics,
            "hub_feature": HUB_FEATURE,
            "tail_quantile": TAIL_QUANTILE,
            "hub_threshold": threshold,
            "raw_ols_slope": raw_slope,
            "applied_slope": slope,
            "slope_cap": SLOPE_CAP,
            "high_query_fraction": float(np.mean(high)),
        },
    )


def main() -> None:
    if sha256_file(OPEN) != OPEN_SHA or sha256_file(BASELINE) != BASELINE_SHA:
        raise RuntimeError("immutable competition input hash mismatch")
    if sha256_file(M244_RECEIPT) != M244_RECEIPT_SHA:
        raise RuntimeError("M244 promoted receipt hash mismatch")
    m244_receipt = json.loads(M244_RECEIPT.read_text(encoding="utf-8"))
    recipes = {
        int(group_id): Recipe(**raw)
        for group_id, raw in m244_receipt["policy"]["recipes"].items()
    }

    surface, numeric = development_surface()
    if surface["forecast_kst_dtm"].ge(DEV_CUTOFF).any():
        raise RuntimeError("lockbox row reached M248 Bayes-head runner")
    feature_sets = _feature_sets(numeric)
    parents = _parents(surface)
    query_issuances = {
        fold: np.sort(parent["data_available_kst_dtm"].unique())
        for fold, parent in parents.items()
    }

    m244_replacements: dict[str, list[pd.DataFrame]] = {
        fold: [] for fold in FOLDS
    }
    corrected_replacements: dict[str, list[pd.DataFrame]] = {
        fold: [] for fold in FOLDS
    }
    diagnostics: dict[str, object] = {}
    for group_id, recipe in recipes.items():
        group_diagnostics: dict[str, object] = {"recipe": asdict(recipe)}
        for fold in FOLDS:
            parent = parents[fold]
            exact_m244_profile, exact_m244_retrieval = _rare_event_profile(
                surface,
                feature_sets,
                group_id,
                query_issuances[fold],
                recipe,
            )
            legacy_profile, corrected_profile, corrected_retrieval = (
                _eligible_rare_event_profiles(
                    surface,
                    feature_sets,
                    group_id,
                    query_issuances[fold],
                    recipe,
                )
            )
            legacy_values = legacy_profile["analog_normalized"].to_numpy(dtype=float)
            exact_values = exact_m244_profile["analog_normalized"].to_numpy(dtype=float)
            if not np.array_equal(legacy_values, exact_values):
                raise RuntimeError(f"M248 failed exact M244 head reproduction: {fold}/g{group_id}")
            spread_profile, spread_retrieval = _spread_adjusted_profile(
                surface,
                feature_sets,
                group_id,
                query_issuances[fold],
                recipe,
            )
            exact_m244_profile = _composed_profile(
                exact_m244_profile,
                spread_profile,
            )
            corrected_profile = _composed_profile(
                corrected_profile,
                spread_profile,
            )
            exact_m244 = _apply_spread_recipe(
                parent,
                exact_m244_profile,
                group_id,
                recipe,
            )
            corrected = _apply_spread_recipe(
                parent,
                corrected_profile,
                group_id,
                recipe,
            )
            m244_replacements[fold].append(exact_m244)
            corrected_replacements[fold].append(corrected)
            group_diagnostics[fold] = {
                "parent": _group_score(parent, group_id),
                "m244": _group_score(exact_m244, group_id),
                "official_eligible_bayes": _group_score(corrected, group_id),
                "exact_m244_retrieval": exact_m244_retrieval,
                "corrected_retrieval": corrected_retrieval,
                "spread_retrieval": spread_retrieval,
            }
        diagnostics[str(group_id)] = group_diagnostics

    m244_outputs = {
        fold: _combine(parents[fold], m244_replacements[fold]) for fold in FOLDS
    }
    corrected_outputs = {
        fold: _combine(parents[fold], corrected_replacements[fold]) for fold in FOLDS
    }
    scores: dict[str, object] = {}
    m244_deltas: dict[str, float] = {}
    corrected_deltas: dict[str, float] = {}
    for fold in FOLDS:
        parent_score = _score(parents[fold])
        m244_score = _score(m244_outputs[fold])
        corrected_score = _score(corrected_outputs[fold])
        expected = m244_receipt["scores"][fold]["rare_event_corrected"]
        if abs(m244_score["total"] - expected["total"]) > 1e-12:
            raise RuntimeError(f"M244 {fold} reproduction score changed")
        m244_deltas[fold] = m244_score["total"] - parent_score["total"]
        corrected_deltas[fold] = corrected_score["total"] - parent_score["total"]
        scores[fold] = {
            "parent": parent_score,
            "m244": m244_score,
            "official_eligible_bayes": corrected_score,
            "m244_total_delta": m244_deltas[fold],
            "corrected_total_delta": corrected_deltas[fold],
            "corrected_minus_m244_total": (
                corrected_score["total"] - m244_score["total"]
            ),
        }

    m244_q4_bootstrap = _paired_issuance_bootstrap(
        parents["q4"], m244_outputs["q4"]
    )
    corrected_q4_bootstrap = _paired_issuance_bootstrap(
        parents["q4"], corrected_outputs["q4"]
    )
    expected_bootstrap = m244_receipt["corrected_q4_paired_bootstrap"]
    if (
        abs(m244_q4_bootstrap["mean"] - expected_bootstrap["mean"]) > 1e-12
        or m244_q4_bootstrap["positive_fraction"]
        != expected_bootstrap["positive_fraction"]
    ):
        raise RuntimeError("M244 Q4 bootstrap reproduction changed")
    all_positive = all(delta > 0.0 for delta in corrected_deltas.values())
    improves_worst_fold = min(corrected_deltas.values()) > min(m244_deltas.values())
    improves_q4_robustness = (
        corrected_deltas["q4"] > m244_deltas["q4"]
        and corrected_q4_bootstrap["positive_fraction"]
        > m244_q4_bootstrap["positive_fraction"]
    )
    promoted = all_positive and (improves_worst_fold or improves_q4_robustness)
    promotion = {
        "promoted": promoted,
        "all_q2_q3_q4_deltas_positive": all_positive,
        "m244_worst_fold_delta": min(m244_deltas.values()),
        "corrected_worst_fold_delta": min(corrected_deltas.values()),
        "improves_worst_fold": improves_worst_fold,
        "improves_q4_robustness": improves_q4_robustness,
        "rule": (
            "all corrected fold deltas positive and either corrected worst-fold "
            "delta strictly exceeds M244 or both Q4 delta and paired-bootstrap "
            "positive fraction strictly exceed M244"
        ),
    }

    prediction_path = OUTPUT / f"{MODEL_ID}-dev-2023-Q3.parquet"
    corrected_outputs["q3"].assign(
        fold_id="dev-2023-Q3",
        model_id=MODEL_ID,
    )[[*METRIC_COLUMNS, "fold_id", "model_id", "data_available_kst_dtm"]].to_parquet(
        prediction_path,
        index=False,
    )
    policy = {
        "architecture": "exact_m244_with_official_eligibility_aware_Bayes_head",
        "recipes": {
            str(group_id): asdict(recipe) for group_id, recipe in recipes.items()
        },
        "eligibility_threshold_capacity_fraction": ELIGIBILITY_THRESHOLD,
        "expected_error": "weighted_mean_conditioned_on_eligible_member_mass",
        "settlement_reward": (
            "weighted_generation_times_units_conditioned_on_eligible_member_mass"
        ),
        "settlement_denominator": (
            "four_times_strict_history_group_mean_eligible_normalized_generation"
        ),
        "zero_eligible_mass_fallback": "exact_M244_legacy_action",
        "action_grid": [float(value) for value in ACTIONS],
        "recency_half_life_days": HALF_LIFE_DAYS,
        "spread_reference_quantile": REFERENCE_QUANTILE,
        "retrieval_search": False,
        "eligibility_search": False,
        "denominator_search": False,
        "action_search": False,
        "head_assignment_search": False,
        "blend_search": False,
        "group_exception_search": False,
        "q1_policy_if_promoted": "unconditional M231 fallback pending scope audit",
    }
    receipt = {
        "schema_version": 1,
        "candidate_id": MODEL_ID,
        "state": (
            "LOCAL_OFFICIAL_ELIGIBLE_ANALOG_BAYES_PROMOTED_FOR_SCOPE_AUDIT"
            if promoted
            else "LOCAL_OFFICIAL_ELIGIBLE_ANALOG_BAYES_REJECTED_NO_TEST_BUILD"
        ),
        "policy": policy,
        "policy_sha256": canonical_sha256(policy),
        "scores": scores,
        "diagnostics": diagnostics,
        "m244_q4_paired_bootstrap": m244_q4_bootstrap,
        "corrected_q4_paired_bootstrap": corrected_q4_bootstrap,
        "promotion": promotion,
        "prediction_path": str(prediction_path.relative_to(ROOT)),
        "prediction_sha256": sha256_file(prediction_path),
        "inputs": {
            "open_zip_sha256": OPEN_SHA,
            "baseline_ipynb_sha256": BASELINE_SHA,
            "distribution_parent_sha256": sha256_file(DISTRIBUTION),
            "v2_parent_oof_sha256": sha256_file(OOF),
            "m244_receipt_sha256": sha256_file(M244_RECEIPT),
        },
        "online_score": None,
        "new_2024_evaluation": False,
        "lockbox_reopened": False,
        "no_external_upload": True,
        "external_actions": [],
    }
    receipt_path = OUTPUT / f"{MODEL_ID}.json"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    print(json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
