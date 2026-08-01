"""Pure P0-P6 candidate promotion decisions."""

from collections.abc import Mapping, Sequence

from baram.contracts.types import PromotionDecision


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
