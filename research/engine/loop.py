
"""The loop driver: propose -> admit -> select -> script -> run -> arbitrate -> update -> repeat.

Escape condition: the champion's local Total reaches 0.66.
Stopping discipline (contract R6): the engine also reports, every cycle, the paired distance to
the escape condition in units of the paired bootstrap sd, so "how far are we really" is never
again answered in marginal units.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/engine')
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/nodes')
from graph import ExcavationGraph                     # noqa: E402
from contract import Node, MIN_EFFECT                 # noqa: E402
from arbiter import arbitrate, KEY                    # noqa: E402
from select import FamilyBandit, select               # noqa: E402

TARGET = 0.66


def status(g: ExcavationGraph) -> str:
    ch = g.champion_record.get('score', {}).get('total', 0.0)
    gap = TARGET - ch
    return (f'champion total={ch:.6f}  gap to {TARGET}={gap:+.6f}  '
            f'= {gap/0.00075:.0f} paired-sd (paired sd ~0.00075) '
            f'/ {gap/0.0096:.1f} marginal-sd')


def cycle(g: ExcavationGraph, bandit: FamilyBandit, runner, budget_min=60.0, seed=0):
    """runner(node) -> DataFrame with KEY + actual_kwh + 'cand' + 'champ' columns, or None."""
    picks = select(list(g.nodes.values()), bandit, budget_min, seed=seed, k=1)
    if not picks:
        return None
    node, info = picks[0]
    g.set_status(node.id, 'selected')
    print(f'[select] {node.id} {node.title}  family={info["family"]} '
          f'p={info["family_p"]:.3f} ev/min={info["ev_per_min"]:.2e}', flush=True)
    df = runner(node)
    if df is None:
        g.set_status(node.id, 'rejected', notes=node.notes + ' | runner returned nothing')
        bandit.update(info['family'], False)
        return node
    took, arb = arbitrate(df, 'cand', 'champ', g.comparisons + 1,
                          metric=node.prereg.get('primary_metric', 'total'))
    g.record_result(node.id, {'total': arb['point_cand']})
    g.record_arbitration(node.id, arb, took)
    partial = float(np.clip(arb['p_better'], 0, 1)) ** 2
    bandit.update(info['family'], took, partial=partial)
    print(f'[arbitrate] {node.id} delta={arb["point_delta"]:+.6f} '
          f'paired_sd={arb["paired_sd"]:.6f} P(better)={arb["p_better"]:.3f} '
          f'required={arb["p_required_adjusted"]:.3f} -> {"CHAMPION" if took else "rejected"}',
          flush=True)
    g.save()
    return node
