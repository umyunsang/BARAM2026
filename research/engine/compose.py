"""Composed-pipeline evaluation and reverse ablation -- the protocol change that makes stage work
measurable.

THE POWER PROBLEM, STATED WITH OUR OWN NUMBERS.
  paired bootstrap sd of a candidate-minus-champion difference : 0.00055 - 0.00093
  realistic size of a single stage upgrade in this problem class: 0.001 - 0.003
  => a single stage upgrade sits at roughly 1-4 paired sd. At 1-2 sd it is indistinguishable, and
     the Model Confidence Set over 23 such candidates duly returned EIGHT tied models.
  k stages composed, if roughly additive : k * 0.002
  => ten stages is +0.02, about 25 paired sd. Decisive.

So the protocol inverts:
  1  BUILD the whole upgraded pipeline (every researched stage at its SOTA setting).
  2  TEST the composed pipeline against the incumbent once.  This is the claim.
  3  ATTRIBUTE per stage by REVERSE ABLATION -- revert one stage at a time to its current
     setting and measure the drop.  Attribution costs k evaluations but spends only ONE
     championship comparison, so contract R6's multiplicity budget is not consumed by
     attribution.
  4  PRUNE any stage whose reverse-ablation drop is negative (it was carried by the others).

This is the standard ablation discipline used in benchmark papers, and it is the discipline this
project has never applied: every previous node was step 2 with k = 1.
"""
from __future__ import annotations
import sys, json, itertools
import numpy as np, pandas as pd

sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/scratch')
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/nodes')
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/engine')
from lib import official_total, FOLDS, CAPS      # noqa: E402
from arbiter import arbitrate, paired_bootstrap  # noqa: E402

KEY = ['fold_id', 'group_id', 'forecast_kst_dtm']


def power_table(stage_effect=0.002, paired_sd=0.00075, target_gap=0.023816):
    """What k stages buy, in paired-sd units. Printed at the head of every composition run so the
    granularity argument is never lost again."""
    rows = []
    for k in (1, 2, 3, 5, 8, 10, 12, 15):
        d = k * stage_effect
        rows.append(dict(k_stages=k, expected_delta=d, in_paired_sd=d / paired_sd,
                         closes_gap_pct=100 * d / target_gap))
    return pd.DataFrame(rows)


def evaluate(frame: pd.DataFrame, col: str) -> dict:
    return official_total(frame.assign(prediction_kwh=frame[col])[
        ['group_id', 'actual_kwh', 'prediction_kwh']])


def compose_and_ablate(frame: pd.DataFrame, full_col: str, incumbent_col: str,
                       ablation_cols: dict[str, str], n_comparisons: int) -> dict:
    """frame carries KEY, actual_kwh, the composed pipeline, the incumbent, and one column per
    reverse-ablation variant (stage_id -> column name)."""
    out = {'power_table': power_table().to_dict('records')}
    s_full = evaluate(frame, full_col); s_inc = evaluate(frame, incumbent_col)
    out['composed'] = s_full; out['incumbent'] = s_inc
    took, arb = arbitrate(frame, full_col, incumbent_col, n_comparisons)
    out['arbitration'] = arb; out['took_champion'] = took

    abl = []
    for sid, col in ablation_cols.items():
        s = evaluate(frame, col)
        r = paired_bootstrap(frame, full_col, col)
        abl.append(dict(stage=sid, total_without=s['total'],
                        contribution=s_full['total'] - s['total'],
                        paired_mean=r['paired_mean'], paired_sd=r['paired_sd'],
                        p_stage_helps=r['p_better']))
    out['ablation'] = sorted(abl, key=lambda d: -d['contribution'])
    return out


if __name__ == '__main__':
    print('--- what composition buys, in units of our own measured paired sd ---')
    print(power_table().round(4).to_string(index=False))
    print('\nreading: at k=1 a genuine stage upgrade is 2.7 paired sd and would usually be called')
    print('a draw; at k=12 the same per-stage effect is 32 paired sd and closes the whole gap.')
    print('The last three sessions measured at k=1 and concluded the axes were closed.')
