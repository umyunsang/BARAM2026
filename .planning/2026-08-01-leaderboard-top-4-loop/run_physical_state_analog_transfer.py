"""Evaluate one fixed three-variable physical-state augmentation over M244."""

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
    BASELINE,
    BASELINE_SHA,
    OPEN,
    OPEN_SHA,
    Recipe,
    _combine,
    _feature_sets,
)
from run_rare_event_corrected_analog_transfer import (
    HUB_FEATURE,
    SLOPE_CAP,
    TAIL_QUANTILE,
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
MODEL_ID = "M249_PHYSICAL_STATE_ANALOG_Q234"
FOLDS = ("q2", "q3", "q4")
PHYSICAL_FEATURES = (
    "phys_v2__shear_alpha_100_80",
    "phys_v2__air_density",
    "phys_v2__rho_v3",
)


def _augmented_feature_sets(columns: list[str]) -> dict[str, list[str]]:
    feature_sets = _feature_sets(columns)
    missing = [name for name in PHYSICAL_FEATURES if name not in columns]
    if missing:
        raise RuntimeError(f"M249 physical features are absent: {missing}")
    duplicate = [name for name in PHYSICAL_FEATURES if name in feature_sets["core"]]
    if duplicate:
        raise RuntimeError(f"M249 physical features entered the frozen core: {duplicate}")
    return {
        **feature_sets,
        "core": [*feature_sets["core"], *PHYSICAL_FEATURES],
    }


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
        raise RuntimeError("lockbox row reached M249 physical-state runner")
    original_feature_sets = _feature_sets(numeric)
    augmented_feature_sets = _augmented_feature_sets(numeric)
    if len(original_feature_sets["core"]) != 25:
        raise RuntimeError("M249 frozen M244 core feature count changed")
    if len(augmented_feature_sets["core"]) != 28:
        raise RuntimeError("M249 augmented physical feature count changed")
    parents = _parents(surface)
    query_issuances = {
        fold: np.sort(parent["data_available_kst_dtm"].unique())
        for fold, parent in parents.items()
    }

    m244_replacements: dict[str, list[pd.DataFrame]] = {
        fold: [] for fold in FOLDS
    }
    augmented_replacements: dict[str, list[pd.DataFrame]] = {
        fold: [] for fold in FOLDS
    }
    diagnostics: dict[str, object] = {}
    for group_id, recipe in recipes.items():
        group_diagnostics: dict[str, object] = {"recipe": asdict(recipe)}
        for fold in FOLDS:
            parent = parents[fold]
            m244_profile, m244_retrieval = _rare_event_profile(
                surface,
                original_feature_sets,
                group_id,
                query_issuances[fold],
                recipe,
            )
            m244_spread_profile, m244_spread_retrieval = _spread_adjusted_profile(
                surface,
                original_feature_sets,
                group_id,
                query_issuances[fold],
                recipe,
            )
            augmented_profile, augmented_retrieval = _rare_event_profile(
                surface,
                augmented_feature_sets,
                group_id,
                query_issuances[fold],
                recipe,
            )
            augmented_spread_profile, augmented_spread_retrieval = (
                _spread_adjusted_profile(
                    surface,
                    augmented_feature_sets,
                    group_id,
                    query_issuances[fold],
                    recipe,
                )
            )
            profile_difference = (
                augmented_profile["analog_normalized"].to_numpy(dtype=float)
                - m244_profile["analog_normalized"].to_numpy(dtype=float)
            )
            multiplier_difference = (
                augmented_spread_profile["analog_blend_multiplier"].to_numpy(
                    dtype=float
                )
                - m244_spread_profile["analog_blend_multiplier"].to_numpy(
                    dtype=float
                )
            )
            m244_profile = _composed_profile(m244_profile, m244_spread_profile)
            augmented_profile = _composed_profile(
                augmented_profile,
                augmented_spread_profile,
            )
            m244 = _apply_spread_recipe(parent, m244_profile, group_id, recipe)
            augmented = _apply_spread_recipe(
                parent,
                augmented_profile,
                group_id,
                recipe,
            )
            m244_replacements[fold].append(m244)
            augmented_replacements[fold].append(augmented)
            group_diagnostics[fold] = {
                "parent": _group_score(parent, group_id),
                "m244": _group_score(m244, group_id),
                "physical_state_augmented": _group_score(augmented, group_id),
                "m244_retrieval": m244_retrieval,
                "m244_spread_retrieval": m244_spread_retrieval,
                "augmented_retrieval": augmented_retrieval,
                "augmented_spread_retrieval": augmented_spread_retrieval,
                "analog_profile_difference": {
                    "mean": float(np.mean(profile_difference)),
                    "mean_absolute": float(np.mean(np.abs(profile_difference))),
                    "max_absolute": float(np.max(np.abs(profile_difference))),
                    "changed_fraction": float(np.mean(profile_difference != 0.0)),
                },
                "spread_multiplier_difference": {
                    "mean": float(np.mean(multiplier_difference)),
                    "mean_absolute": float(np.mean(np.abs(multiplier_difference))),
                    "max_absolute": float(np.max(np.abs(multiplier_difference))),
                    "changed_fraction": float(np.mean(multiplier_difference != 0.0)),
                },
            }
        diagnostics[str(group_id)] = group_diagnostics

    m244_outputs = {
        fold: _combine(parents[fold], m244_replacements[fold]) for fold in FOLDS
    }
    augmented_outputs = {
        fold: _combine(parents[fold], augmented_replacements[fold]) for fold in FOLDS
    }
    scores: dict[str, object] = {}
    m244_deltas: dict[str, float] = {}
    augmented_deltas: dict[str, float] = {}
    for fold in FOLDS:
        parent_score = _score(parents[fold])
        m244_score = _score(m244_outputs[fold])
        augmented_score = _score(augmented_outputs[fold])
        expected = m244_receipt["scores"][fold]["rare_event_corrected"]
        if abs(m244_score["total"] - expected["total"]) > 1e-12:
            raise RuntimeError(f"M244 {fold} reproduction score changed")
        m244_deltas[fold] = m244_score["total"] - parent_score["total"]
        augmented_deltas[fold] = augmented_score["total"] - parent_score["total"]
        scores[fold] = {
            "parent": parent_score,
            "m244": m244_score,
            "physical_state_augmented": augmented_score,
            "m244_total_delta": m244_deltas[fold],
            "augmented_total_delta": augmented_deltas[fold],
            "augmented_minus_m244_total": (
                augmented_score["total"] - m244_score["total"]
            ),
        }

    m244_q4_bootstrap = _paired_issuance_bootstrap(
        parents["q4"], m244_outputs["q4"]
    )
    augmented_q4_bootstrap = _paired_issuance_bootstrap(
        parents["q4"], augmented_outputs["q4"]
    )
    expected_bootstrap = m244_receipt["corrected_q4_paired_bootstrap"]
    if (
        abs(m244_q4_bootstrap["mean"] - expected_bootstrap["mean"]) > 1e-12
        or m244_q4_bootstrap["positive_fraction"]
        != expected_bootstrap["positive_fraction"]
    ):
        raise RuntimeError("M244 Q4 bootstrap reproduction changed")
    all_positive = all(delta > 0.0 for delta in augmented_deltas.values())
    improves_worst_fold = min(augmented_deltas.values()) > min(m244_deltas.values())
    improves_q4_robustness = (
        augmented_deltas["q4"] > m244_deltas["q4"]
        and augmented_q4_bootstrap["positive_fraction"]
        > m244_q4_bootstrap["positive_fraction"]
    )
    promoted = all_positive and (improves_worst_fold or improves_q4_robustness)
    promotion = {
        "promoted": promoted,
        "all_q2_q3_q4_deltas_positive": all_positive,
        "m244_worst_fold_delta": min(m244_deltas.values()),
        "augmented_worst_fold_delta": min(augmented_deltas.values()),
        "improves_worst_fold": improves_worst_fold,
        "improves_q4_robustness": improves_q4_robustness,
        "rule": (
            "all augmented fold deltas positive and either augmented worst-fold "
            "delta strictly exceeds M244 or both Q4 delta and paired-bootstrap "
            "positive fraction strictly exceed M244"
        ),
    }

    prediction_path = OUTPUT / f"{MODEL_ID}-dev-2023-Q3.parquet"
    augmented_outputs["q3"].assign(
        fold_id="dev-2023-Q3",
        model_id=MODEL_ID,
    )[[*METRIC_COLUMNS, "fold_id", "model_id", "data_available_kst_dtm"]].to_parquet(
        prediction_path,
        index=False,
    )
    policy = {
        "architecture": "exact_m244_with_three_feature_physical_state_retrieval",
        "recipes": {
            str(group_id): asdict(recipe) for group_id, recipe in recipes.items()
        },
        "base_core_feature_count": len(original_feature_sets["core"]),
        "augmented_core_feature_count": len(augmented_feature_sets["core"]),
        "physical_features": list(PHYSICAL_FEATURES),
        "representation": "frozen_raw_delta_PCA24_season2p5_refit_on_strict_history",
        "rare_event_hub_feature": HUB_FEATURE,
        "rare_event_tail_quantile": TAIL_QUANTILE,
        "rare_event_slope_bounds": [0.0, SLOPE_CAP],
        "recency_half_life_days": HALF_LIFE_DAYS,
        "spread_reference_quantile": REFERENCE_QUANTILE,
        "physical_feature_search": False,
        "feature_weight_search": False,
        "feature_transform_search": False,
        "pca_search": False,
        "neighbor_search": False,
        "recipe_search": False,
        "group_exception_search": False,
        "q1_policy_if_promoted": "unconditional M231 fallback pending scope audit",
    }
    receipt = {
        "schema_version": 1,
        "candidate_id": MODEL_ID,
        "state": (
            "LOCAL_PHYSICAL_STATE_ANALOG_PROMOTED_FOR_SCOPE_AUDIT"
            if promoted
            else "LOCAL_PHYSICAL_STATE_ANALOG_REJECTED_NO_TEST_BUILD"
        ),
        "policy": policy,
        "policy_sha256": canonical_sha256(policy),
        "scores": scores,
        "diagnostics": diagnostics,
        "m244_q4_paired_bootstrap": m244_q4_bootstrap,
        "augmented_q4_paired_bootstrap": augmented_q4_bootstrap,
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
