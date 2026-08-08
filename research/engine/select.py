
"""Loop engineering: dynamic node selection.

The previous workflow ran nodes in the order a human wrote them down.  That is why eighteen
consecutive S12 nodes were all siblings on one rung: nothing in the process could notice that the
family had stopped paying and move.

Selection here is expected-value-per-minute with an explicit family-level posterior.  Each node
declares a prior effect and a cost; each family (the axis it belongs to) carries a Beta posterior
over "this family produces a champion-taking result", updated by every arbitration.  A node's
score is

    EV = P_family_success * prior_effect / cost_minutes        (Thompson-sampled)

so a family that has failed k times in a row is sampled down automatically, and an unexplored
family with a modest prior still gets tried.  This is the standard bandit remedy for the failure
mode the ledger recorded: 30 consecutive failures inside two families.
"""
from __future__ import annotations
import numpy as np
from collections import defaultdict


class FamilyBandit:
    def __init__(self, prior_a: float = 1.0, prior_b: float = 1.0):
        self.a = defaultdict(lambda: prior_a)
        self.b = defaultdict(lambda: prior_b)

    def update(self, family: str, success: bool, partial: float = 0.0):
        """A rejected node still informs: `partial` in [0,1] credits a near-miss."""
        if success:
            self.a[family] += 1.0
        else:
            self.a[family] += partial
            self.b[family] += 1.0 - partial

    def sample(self, family: str, rng) -> float:
        return float(rng.beta(self.a[family], self.b[family]))

    def mean(self, family: str) -> float:
        return self.a[family] / (self.a[family] + self.b[family])


def select(nodes, bandit: FamilyBandit, budget_min: float, seed: int = 0, k: int = 1):
    """Thompson-sample a family posterior per candidate, rank by EV per minute, respect budget."""
    rng = np.random.default_rng(seed)
    scored = []
    for n in nodes:
        if n.status not in ('proposed', 'selected'):
            continue
        fam = n.prereg.get('axis', n.stage)
        p = bandit.sample(fam, rng)
        ev = p * float(n.prior_effect) / max(float(n.cost_min), 1.0)
        scored.append((ev, p, n))
    scored.sort(key=lambda t: -t[0])
    out, spent = [], 0.0
    for ev, p, n in scored:
        if len(out) >= k or spent + n.cost_min > budget_min:
            continue
        out.append((n, {'ev_per_min': ev, 'family_p': p, 'family': n.prereg.get('axis', n.stage)}))
        spent += n.cost_min
    return out
