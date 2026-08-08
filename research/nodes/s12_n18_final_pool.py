"""S12-N18 * final pooling of everything S12 built, under the fold-outside gate.

Members available to the pool after S12: the incumbent D, the valid-conditional DV
(the best point accuracy of any member: 1-NMAE 0.866061), the analog-with-aligned-decision-layer
KNN120/KNN60, the source-separated DL/DG (corr(DL,DG) = 0.7924, the most decorrelated pair ever
measured here), the source-stacked DSTK if built, and the deployed DEPAVG.

S12-N17 measured the paired bootstrap sd of a candidate-minus-incumbent difference at
0.00055-0.00093, so a genuine gain of ~0.001 is detectable; anything smaller is not.  All
weights are chosen fold-outside (on the other two folds), all policies are chosen fold-outside,
and the row-alignment key is ['fold_id','group_id','forecast_kst_dtm'].
"""
import sys, json, itertools
import numpy as np, pandas as pd
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/nodes')
from loop_lib import *
from s12_n5_analog_axis import fo_blend_ndof

N = '/Users/um-yunsang/BARAM2026/research/nodes/'
TAGS = ['D', 'DV', 'DVT', 'KNN120', 'KNN60', 'KNN300', 'DL', 'DG']

if __name__ == '__main__':
    R = canonical_keys(); dep = load_depavg()
    J = None
    solos = {}
    for t in TAGS:
        try:
            P = align_prob(t, R)
        except FileNotFoundError:
            continue
        Dm, s, _ = fo_policy(utility_frames(P, R), R)
        solos[t] = s['total']
        d = Dm[KEY + ['actual_kwh', 'prediction_kwh']].rename(columns={'prediction_kwh': t})
        J = d if J is None else J.merge(d.drop(columns=['actual_kwh']), on=KEY)
    J = J.merge(dep, on=KEY)
    have = [t for t in TAGS if t in J.columns]
    print('solos:', {k: round(v, 6) for k, v in solos.items()})

    cap = J.group_id.map(CAPS)
    E = pd.DataFrame({c: (J[c] - J.actual_kwh) / cap for c in have + ['DEPAVG']})
    print('\n--- error correlation matrix ---')
    print(E.corr().round(3).to_string())

    out = {'solos': solos, 'error_corr': E.corr().round(6).to_dict(), 'blends': {}}
    print('\n--- 1-dof fold-outside blends against DEPAVG ---')
    for t in have:
        r, pk = fo_blend_ndof(J, [t, 'DEPAVG'])
        out['blends'][f'{t}+DEPAVG'] = {'total': r['total'], 'picks': {k: list(v) for k, v in pk.items()}}
        print(f'  {t:8s}+DEPAVG  {r["total"]:.6f}  d={r["total"]-0.6361842493883538:+.6f}  '
              f'w={[round(v[0],2) for v in pk.values()]}')

    print('\n--- 2-dof fold-outside simplex: D + DEPAVG + one more ---')
    for t in [x for x in have if x != 'D']:
        r, pk = fo_blend_ndof(J, ['D', 'DEPAVG', t])
        out['blends'][f'D+DEPAVG+{t}'] = {'total': r['total'], 'picks': {k: list(v) for k, v in pk.items()}}
        print(f'  D+DEPAVG+{t:8s} {r["total"]:.6f}  d={r["total"]-0.6361842493883538:+.6f}  '
              f'picks={[list(v) for v in pk.values()]}')

    print('\n--- 0-dof uniform member averages, then 1-dof against DEPAVG ---')
    for nm, sub in [('D_DV', ['D', 'DV']), ('D_KNN', ['D', 'KNN120']),
                    ('D_DV_KNN', ['D', 'DV', 'KNN120']),
                    ('D_DL_DG', ['D', 'DL', 'DG']), ('ALL', have)]:
        sub = [s for s in sub if s in J.columns]
        if len(sub) < 2:
            continue
        col = 'U_' + nm
        J[col] = J[sub].mean(axis=1)
        r, pk = fo_blend_ndof(J, [col, 'DEPAVG'])
        out['blends'][f'{nm}+DEPAVG'] = {'total': r['total'], 'members': sub,
                                         'picks': {k: list(v) for k, v in pk.items()}}
        print(f'  {nm:10s}+DEPAVG {r["total"]:.6f}  d={r["total"]-0.6361842493883538:+.6f}  '
              f'w={[round(v[0],2) for v in pk.values()]}')

    best = max(out['blends'], key=lambda k: out['blends'][k]['total'])
    print(f'\nBEST of pool: {best} = {out["blends"][best]["total"]:.6f}  '
          f'(incumbent 0.636184, paired bootstrap sd ~0.0006-0.0009)')
    out['best'] = [best, out['blends'][best]['total']]
    json.dump(out, open(N + 'S12-N18_final_pool.json', 'w'), indent=1, default=str)
    J.to_parquet(N + 'S12-N18_pool.parquet', index=False)
