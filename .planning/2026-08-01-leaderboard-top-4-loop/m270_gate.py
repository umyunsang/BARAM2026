"""M270 frozen promotion gate on monthly blocks.

WHY THIS REPLACES THE QUARTERLY GATE
The previous gate required positive deltas on Q3 and Q4 plus a Q4 paired bootstrap above
0.50. That is two paired observations, so it cannot separate structural instability from
two folds happening to disagree, and its bootstrap addresses within-fold uncertainty
rather than across-fold transfer stability. M269 Stage B produced `+0.009074` on Q3 and
`-0.006229` on Q4 and the gate had no way to read it.

FROZEN 2026-08-04, BEFORE ANY CANDIDATE WAS EVALUATED UNDER IT.

  G1  sign test on positive months, one-sided p <= 0.10
      Stated as a p-value rather than a count so it adapts when months are excluded.
      At n = 9 this means at least 7 positive months (p = 0.0898).
  G2  median monthly Total delta > 0
  G3  block-bootstrap 5% quantile of the mean monthly Total delta > 0
  G4  worst monthly Total delta >= -0.010

G1 tests consistency, G3 tests effect size, and requiring both means a candidate must be
both reliable and materially positive. G4 blocks a candidate that wins on average while
collapsing in one regime, which matters because the official score is an annual aggregate.

STATUS OF EACH THRESHOLD
G1-G3 introduce no fitted constant; they are standard tests at a declared level. G4 does
introduce one and is therefore a DECLARED CONVENTION, not a calibrated threshold. Its
rationale: the remaining gap to target is about 0.032 Total, so a single month losing more
than 0.010 consumes roughly a third of the gap in one twelfth of the year, and observed
monthly Total already spans 0.083 naturally, so a 0.010 regression reads as a regime
failure rather than noise. No calibration exists to justify a precise value and none is
claimed.

Component metrics (`1-NMAE`, per-group FICR, settlement tier counts) remain mandatory
diagnostics on every receipt but are NOT independent gate conditions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from m270_monthly_validation import paired_monthly_delta

SIGN_TEST_ALPHA = 0.10
WORST_MONTH_FLOOR = -0.010
GATE_VERSION = "M270_MONTHLY_GATE_v1_frozen_2026-08-04"


@dataclass(frozen=True)
class GateResult:
    passed: bool
    conditions: dict[str, bool]
    evidence: dict[str, object] = field(default_factory=dict)

    def explain(self) -> str:
        lines = [f"gate={GATE_VERSION}", f"PASSED={self.passed}"]
        for name, ok in self.conditions.items():
            lines.append(f"  {'PASS' if ok else 'FAIL'}  {name}")
        return "\n".join(lines)


def evaluate_gate(candidate: pd.DataFrame, parent: pd.DataFrame) -> GateResult:
    """Apply the frozen monthly gate to a candidate against its declared parent."""
    stats = paired_monthly_delta(candidate, parent)
    conditions = {
        f"G1 sign-test p <= {SIGN_TEST_ALPHA} "
        f"(p={stats['sign_test_p_greater']:.4f}, "
        f"{stats['positive_months']}/{stats['months_scored']})":
            bool(stats["sign_test_p_greater"] <= SIGN_TEST_ALPHA),
        f"G2 median delta > 0 (median={stats['median_total_delta']:+.6f})":
            bool(stats["median_total_delta"] > 0.0),
        f"G3 bootstrap q05 > 0 (q05={stats['block_bootstrap_q05']:+.6f})":
            bool(stats["block_bootstrap_q05"] > 0.0),
        f"G4 worst month >= {WORST_MONTH_FLOOR:+.3f} "
        f"(worst={stats['min_total_delta']:+.6f})":
            bool(stats["min_total_delta"] >= WORST_MONTH_FLOOR),
    }
    return GateResult(
        passed=all(conditions.values()), conditions=conditions, evidence=stats
    )
