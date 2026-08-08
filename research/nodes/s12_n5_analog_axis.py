"""S12-N5 * bring the ANALOG lineage into the local research pool.

AGENTS.md records that member error correlation against M115 is 0.984-0.994 for every
classifier-family member but 0.944 for the analog M244, i.e. the analog lineage is the one
genuinely decorrelated family in this repository.  The whole S9/S10 local pool search
(15 of my members + the 3 deployed classifier stems) never contained an analog member --
M244_RARE_EVENT_CORRECTED_ANALOG_Q234 is the only analog stem with all three dev-2023 folds
saved, and it was never joined into the local pool.

PROVENANCE / TRAP NOTE (AGENTS.md standing rule):
  (a) policy provenance -- M244 has NO `-policies` grid file, only a single `prediction_kwh`
      column per fold.  The documented mixed-policy trap applies to metric-aligned-probe
      classifier stems whose `prediction_kwh` was assembled post-hoc from several policies;
      it cannot be *verified* absent for M244 because no policy grid exists to check against.
      This node therefore reports M244 results as PROVENANCE-UNVERIFIED and never promotes
      them without that caveat.
  (b) weight provenance -- all blend weights are fold-outside (chosen on the other two folds).
  (c) row-alignment key -- ['fold_id','group_id','forecast_kst_dtm'].
"""
import sys, json, itertools
import numpy as np, pandas as pd
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/nodes')
from loop_lib import *

AB = '/Users/um-yunsang/BARAM2026/artifacts/backtests/metric-aligned-probe/'
N = '/Users/um-yunsang/BARAM2026/research/nodes/'


def load_plain(stem, col):
    parts = []
    for f in FOLDS:
        d = pd.read_parquet(AB + f'{stem}-{f}.parquet').copy()
        d['fold_id'] = f
        parts.append(d[KEY + ['actual_kwh', 'prediction_kwh']].rename(columns={'prediction_kwh': col}))
    return pd.concat(parts, ignore_index=True)


def fo_blend_ndof(J, cols, step=0.05):
    """N-dof simplex blend, weights chosen fold-outside."""
    k = len(cols)
    grid = []
    def rec(pre, rem, left):
        if rem == 1:
            grid.append(tuple(pre + [round(left, 4)])); return
        w = 0.0
        while w <= left + 1e-9:
            rec(pre + [round(w, 4)], rem - 1, left - w); w += step
    rec([], k, 1.0)
    rows = []; picks = {}
    for f in FOLDS:
        oth = J[J.fold_id != f]; held = J[J.fold_id == f]; best = None
        A_o = oth[cols].to_numpy(); base_o = oth[['group_id', 'actual_kwh']].copy()
        for wv in grid:
            p = A_o @ np.array(wv)
            t = official_total(base_o.assign(prediction_kwh=p))['total']
            if best is None or t > best[0]:
                best = (t, wv)
        picks[f] = best[1]
        rows.append(held.assign(prediction_kwh=held[cols].to_numpy() @ np.array(best[1])))
    D = pd.concat(rows, ignore_index=True)
    return official_total(D[['group_id', 'actual_kwh', 'prediction_kwh']]), picks


if __name__ == '__main__':
    R = canonical_keys()
    Dm, dsolo, dpicks = fo_policy(utility_frames(align_prob('D', R), R), R)
    dep = load_depavg()
    ANA = load_plain('M244_RARE_EVENT_CORRECTED_ANALOG_Q234', 'ANALOG')
    FT = load_plain('M129_GROUP_FINETUNE', 'FINETUNE')

    J = (Dm[KEY + ['actual_kwh', 'prediction_kwh']].rename(columns={'prediction_kwh': 'D'})
         .merge(dep, on=KEY)
         .merge(ANA.drop(columns=['actual_kwh']), on=KEY)
         .merge(FT.drop(columns=['actual_kwh']), on=KEY))
    print('joined rows', len(J))
    tot = lambda c: official_total(J.assign(prediction_kwh=J[c])[
        ['group_id', 'actual_kwh', 'prediction_kwh']])
    for c in ['D', 'DEPAVG', 'ANALOG', 'FINETUNE']:
        s = tot(c)
        print(f'  solo {c:9s} {s["total"]:.6f}  (1-NMAE={s["one_minus_nmae"]:.6f} FICR={s["ficr"]:.6f})')

    cap = J.group_id.map(CAPS)
    E = pd.DataFrame({c: (J[c] - J.actual_kwh) / cap for c in ['D', 'DEPAVG', 'ANALOG', 'FINETUNE']})
    print('\n--- error correlation ---')
    print(E.corr().round(4).to_string())

    out = {'solos': {c: tot(c)['total'] for c in ['D', 'DEPAVG', 'ANALOG', 'FINETUNE']},
           'error_corr': E.corr().round(6).to_dict(), 'blends': {}}
    print('\n--- fold-outside blends (weights chosen on the other two folds) ---')
    combos = [('D', 'DEPAVG'), ('D', 'ANALOG'), ('DEPAVG', 'ANALOG'), ('D', 'FINETUNE'),
              ('D', 'DEPAVG', 'ANALOG'), ('D', 'DEPAVG', 'FINETUNE'),
              ('D', 'DEPAVG', 'ANALOG', 'FINETUNE')]
    for c in combos:
        r, pk = fo_blend_ndof(J, list(c))
        out['blends']['+'.join(c)] = {'total': r['total'], 'one_minus_nmae': r['one_minus_nmae'],
                                      'ficr': r['ficr'], 'picks': {k: list(v) for k, v in pk.items()}}
        print(f'  {"+".join(c):34s} {r["total"]:.6f}  d_vs_best={r["total"]-0.6361842493883538:+.6f}  '
              f'picks={list(pk.values())}', flush=True)
    json.dump(out, open(N + 'S12-N5_analog_axis.json', 'w'), indent=1, default=str)
    J.to_parquet(N + 'S12-N5_pool.parquet', index=False)
