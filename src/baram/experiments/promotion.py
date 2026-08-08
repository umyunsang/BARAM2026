"""Pure P0-P6 candidate promotion decisions."""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from baram.contracts.types import PromotionDecision
from baram.exceptions import ContractError


@dataclass(frozen=True)
class V2PromotionThresholds:
    material_total_delta_min: float
    finalist_seed_total_range_max: float
    pooled_one_minus_nmae_delta_min: float
    per_group_component_total_delta_min: float
    group3_component_total_delta_min_for_decision_ensemble: float
    decision_ficr_delta_min: float
    absolute_residual_correlation_max_exclusive: float


def load_v2_promotion_thresholds(path: Path) -> V2PromotionThresholds:
    try:
        raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
        if raw.get("all_later_fold_total_deltas_positive") is not True:
            raise ContractError("v2 promotion requires every later fold delta to be positive")
        return V2PromotionThresholds(
            material_total_delta_min=float(raw["material_total_delta_min"]),
            finalist_seed_total_range_max=float(raw["finalist_seed_total_range_max"]),
            pooled_one_minus_nmae_delta_min=float(raw["pooled_one_minus_nmae_delta_min"]),
            per_group_component_total_delta_min=float(
                raw["per_group_component_total_delta_min"]
            ),
            group3_component_total_delta_min_for_decision_ensemble=float(
                raw["group3_component_total_delta_min_for_decision_ensemble"]
            ),
            decision_ficr_delta_min=float(raw["decision_ficr_delta_min"]),
            absolute_residual_correlation_max_exclusive=float(
                raw["absolute_residual_correlation_max_exclusive"]
            ),
        )
    except (OSError, TypeError, KeyError, ValueError, yaml.YAMLError) as error:
        raise ContractError(f"cannot read v2 promotion thresholds: {error}") from error


def decide_v2_candidate(
    pooled_total_delta: float,
    fold_total_deltas: Sequence[float],
    one_minus_nmae_delta: float,
    group_component_total_deltas: Mapping[int, float],
    seed_total_range: float,
    thresholds: V2PromotionThresholds,
) -> PromotionDecision:
    checks = {
        "material_total": pooled_total_delta >= thresholds.material_total_delta_min,
        "all_later_folds_positive": bool(fold_total_deltas)
        and all(value > 0.0 for value in fold_total_deltas),
        "nmae_guardrail": one_minus_nmae_delta
        >= thresholds.pooled_one_minus_nmae_delta_min,
        "group_guardrails": set(group_component_total_deltas) == {1, 2, 3}
        and all(
            value >= thresholds.per_group_component_total_delta_min
            for value in group_component_total_deltas.values()
        ),
        "seed_range": seed_total_range <= thresholds.finalist_seed_total_range_max,
    }
    failed = tuple(name for name, passed in checks.items() if not passed)
    return PromotionDecision(
        not failed,
        "V2_CANDIDATE",
        failed,
        {
            "total": pooled_total_delta,
            "one_minus_nmae": one_minus_nmae_delta,
            "seed_range": seed_total_range,
        },
    )


def decide_v2_decision(
    pooled_total_delta: float,
    fold_total_deltas: Sequence[float],
    one_minus_nmae_delta: float,
    ficr_delta: float,
    group_component_total_deltas: Mapping[int, float],
    thresholds: V2PromotionThresholds,
) -> PromotionDecision:
    base = decide_v2_candidate(
        pooled_total_delta,
        fold_total_deltas,
        one_minus_nmae_delta,
        group_component_total_deltas,
        0.0,
        thresholds,
    )
    failed = list(base.reasons)
    if ficr_delta < thresholds.decision_ficr_delta_min:
        failed.append("ficr_materiality")
    if group_component_total_deltas.get(3, -math.inf) < (
        thresholds.group3_component_total_delta_min_for_decision_ensemble
    ):
        failed.append("group3_guardrail")
    return PromotionDecision(
        not failed,
        "V2_DECISION",
        tuple(failed),
        {**base.deltas, "ficr": ficr_delta},
    )


def decide_v2_ensemble(
    pooled_total_delta: float,
    fold_total_deltas: Sequence[float],
    one_minus_nmae_delta: float,
    group_component_total_deltas: Mapping[int, float],
    abs_residual_correlation: float,
    thresholds: V2PromotionThresholds,
) -> PromotionDecision:
    base = decide_v2_candidate(
        pooled_total_delta,
        fold_total_deltas,
        one_minus_nmae_delta,
        group_component_total_deltas,
        0.0,
        thresholds,
    )
    failed = list(base.reasons)
    if group_component_total_deltas.get(3, -math.inf) < (
        thresholds.group3_component_total_delta_min_for_decision_ensemble
    ):
        failed.append("group3_guardrail")
    if not abs_residual_correlation < (
        thresholds.absolute_residual_correlation_max_exclusive
    ):
        failed.append("residual_diversity")
    return PromotionDecision(
        not failed,
        "V2_ENSEMBLE",
        tuple(failed),
        {**base.deltas, "abs_residual_correlation": abs_residual_correlation},
    )


def decide_challenger_activation(
    configuration_slots_remaining: int,
    shared_failure_mass: float,
    lockbox_consumed: bool,
) -> PromotionDecision:
    """Apply the approved conditional-search gate without inspecting the lockbox."""
    enough_slots = configuration_slots_remaining >= 8
    named_failure = math.isfinite(shared_failure_mass) and shared_failure_mass >= 0.25
    accepted = enough_slots and named_failure and not lockbox_consumed
    reasons = (
        f"configuration_slots_remaining={configuration_slots_remaining}",
        f"shared_failure_mass={shared_failure_mass:.12f}",
        f"lockbox_consumed={lockbox_consumed}",
    )
    return PromotionDecision(
        accepted,
        "ACTIVATION",
        reasons,
        {"shared_failure_mass": shared_failure_mass},
    )


def decide_contract(checks: Mapping[str, bool]) -> PromotionDecision:
    failed = tuple(sorted(name for name, passed in checks.items() if not passed))
    return PromotionDecision(not failed, "P0", failed, {})


def decide_baseline(parity_pass: bool, repeated_hash_match: bool) -> PromotionDecision:
    accepted = parity_pass and repeated_hash_match
    return PromotionDecision(
        accepted,
        "P1",
        (f"parity_pass={parity_pass}", f"repeated_hash_match={repeated_hash_match}"),
        {},
    )


def decide_development_promotion(
    pooled_delta: float,
    fold_deltas: Sequence[float],
    contracts_pass: bool,
) -> PromotionDecision:
    nonnegative = sum(delta >= 0.0 for delta in fold_deltas)
    majority = bool(fold_deltas) and nonnegative > len(fold_deltas) / 2
    accepted = contracts_pass and pooled_delta > 1e-9 and majority
    reasons = (
        f"contracts_pass={contracts_pass}",
        f"pooled_delta={pooled_delta:.12f}",
        f"nonnegative_folds={nonnegative}/{len(fold_deltas)}",
    )
    return PromotionDecision(accepted, "P2", reasons, {"pooled": pooled_delta})


def decide_diversity(
    abs_residual_correlation: float,
    score_delta: float,
) -> PromotionDecision:
    accepted = abs_residual_correlation < 0.995 and score_delta > 1e-9
    return PromotionDecision(
        accepted,
        "P3",
        (f"abs_residual_correlation={abs_residual_correlation:.12f}",),
        {"pooled": score_delta},
    )


def decide_lockbox(candidate_total: float, control_total: float) -> PromotionDecision:
    delta = candidate_total - control_total
    return PromotionDecision(
        delta > 1e-9,
        "P4",
        (f"total_delta={delta:.12f}",),
        {"total": delta},
    )


def decide_reproduction(
    expected_hashes: Mapping[str, str],
    actual_hashes: Mapping[str, str],
) -> PromotionDecision:
    all_keys = set(expected_hashes) | set(actual_hashes)
    mismatches = tuple(
        sorted(key for key in all_keys if actual_hashes.get(key) != expected_hashes.get(key))
    )
    return PromotionDecision(not mismatches, "P5", mismatches, {})


def decide_submission(contract_checks: Mapping[str, bool]) -> PromotionDecision:
    failed = tuple(sorted(name for name, passed in contract_checks.items() if not passed))
    return PromotionDecision(not failed, "P6", failed, {})
