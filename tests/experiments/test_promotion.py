from baram.experiments.promotion import (
    V2PromotionThresholds,
    decide_baseline,
    decide_challenger_activation,
    decide_contract,
    decide_development_promotion,
    decide_diversity,
    decide_lockbox,
    decide_reproduction,
    decide_submission,
    decide_v2_candidate,
    decide_v2_decision,
    decide_v2_ensemble,
)

V2 = V2PromotionThresholds(0.0035, 0.0035, -0.001, -0.001, 0.0, 0.007, 0.995)


def test_challenger_activation_requires_slots_shared_failure_and_unopened_lockbox() -> None:
    """Catches an optional search running outside its approved three-part gate."""
    accepted = decide_challenger_activation(
        configuration_slots_remaining=8,
        shared_failure_mass=0.25,
        lockbox_consumed=False,
    )
    assert accepted.accepted is True
    assert accepted.gate == "ACTIVATION"
    assert not decide_challenger_activation(7, 0.25, False).accepted
    assert not decide_challenger_activation(8, 0.249999, False).accepted
    assert not decide_challenger_activation(8, 0.25, True).accepted


def test_development_promotion_requires_pooled_and_majority_gain() -> None:
    """Catches promotion from one lucky fold or a nonpositive pooled delta."""
    accepted = decide_development_promotion(0.001, [0.002, 0.001, -0.0005], True)
    rejected = decide_development_promotion(0.001, [0.002, -0.001, -0.0005], True)
    assert accepted.accepted is True
    assert rejected.accepted is False
    assert accepted.gate == "P2"


def test_contract_failure_overrides_score_gain() -> None:
    """Catches metric gain bypassing a failed source or leakage gate."""
    decision = decide_development_promotion(0.01, [0.01, 0.01], contracts_pass=False)
    assert decision.accepted is False


def test_closed_promotion_gates_report_failures() -> None:
    """Catches missing P0/P1/P3-P6 rejection branches."""
    assert decide_contract({"hash": True, "schema": False}).reasons == ("schema",)
    assert not decide_baseline(True, False).accepted
    assert decide_diversity(0.9, 0.001).accepted
    assert not decide_diversity(0.999, 0.001).accepted
    assert decide_lockbox(0.91, 0.90).accepted
    assert not decide_reproduction({"a": "1"}, {"a": "2"}).accepted
    assert not decide_submission({"encoding": True, "keys": False}).accepted


def test_v2_candidate_requires_material_all_fold_and_component_stability() -> None:
    accepted = decide_v2_candidate(
        0.0035, [0.001, 0.002], -0.001, {1: 0.0, 2: -0.001, 3: 0.001}, 0.0035, V2
    )
    assert accepted.accepted
    assert not decide_v2_candidate(
        0.0034, [0.001, 0.002], 0.0, {1: 0.0, 2: 0.0, 3: 0.0}, 0.0, V2
    ).accepted
    assert not decide_v2_candidate(
        0.01, [0.001, 0.0], 0.0, {1: 0.0, 2: 0.0, 3: 0.0}, 0.0, V2
    ).accepted


def test_v2_decision_and_ensemble_enforce_ficr_group3_and_diversity() -> None:
    common = (0.004, [0.002, 0.002], 0.0, {1: 0.001, 2: 0.001, 3: 0.001})
    assert decide_v2_decision(common[0], common[1], common[2], 0.007, common[3], V2).accepted
    assert not decide_v2_decision(
        common[0], common[1], common[2], 0.0069, common[3], V2
    ).accepted
    assert decide_v2_ensemble(*common, 0.994, V2).accepted
    assert not decide_v2_ensemble(*common, 0.995, V2).accepted
