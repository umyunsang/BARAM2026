from baram.experiments.promotion import (
    decide_baseline,
    decide_contract,
    decide_development_promotion,
    decide_diversity,
    decide_lockbox,
    decide_reproduction,
    decide_submission,
)


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
