"""S12-N2 · loss anatomy of current_best (read-only diagnostic, 0 dof)."""
import sys, json
import numpy as np, pandas as pd
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/nodes')
from loop_lib import *

R = canonical_keys(); dep = load_depavg()
PD = align_prob('D', R)
frames = utility_frames(PD, R)
Dm, solo, picks = fo_policy(frames, R)
J = Dm[KEY + ['actual_kwh', 'prediction_kwh']].rename(columns={'prediction_kwh': 'MINE'}).merge(dep, on=KEY)
bl, wp, B = fo_blend_1dof(J, 'MINE', 'DEPAVG')
print('current_best', bl['total'], bl['one_minus_nmae'], bl['ficr'])
print('group nmae', {k: round(v, 5) for k, v in bl['group_nmae'].items()})
print('group ficr', {k: round(v, 5) for k, v in bl['group_ficr'].items()})

cap = B.group_id.map(CAPS)
B['cf_act'] = B.actual_kwh / cap
B['cf_pred'] = B.prediction_kwh / cap
B['err'] = (B.cf_pred - B.cf_act)
V = B[B.cf_act >= 0.10].copy()
V['aerr'] = V.err.abs()
print(f'\nvalid rows {len(V)} / {len(B)}')
print('hit<=0.06 (unweighted)', round(float((V.aerr <= 0.06).mean()), 4))
print('hit<=0.08 (unweighted)', round(float((V.aerr <= 0.08).mean()), 4))
print('mean |err|', round(float(V.aerr.mean()), 5), ' median', round(float(V.aerr.median()), 5))
print('mean signed err', round(float(V.err.mean()), 5))

# where is FICR lost?  actual-weighted miss mass, by actual cf bucket
V['bkt'] = pd.cut(V.cf_act, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.05])
rows = []
for g in (1, 2, 3):
    sub = V[V.group_id == g]
    denom = (sub.cf_act * 4).sum()
    for b, s in sub.groupby('bkt', observed=True):
        u = np.select([s.aerr <= 0.06, s.aerr <= 0.08], [4.0, 3.0], 0.0)
        got = float((s.cf_act * u).sum()); mx = float((s.cf_act * 4).sum())
        rows.append(dict(g=g, bkt=str(b), n=len(s), share_of_denom=mx / denom,
                         ficr_realised=got / denom, ficr_lost=(mx - got) / denom,
                         hit6=float((s.aerr <= 0.06).mean()), bias=float(s.err.mean())))
T = pd.DataFrame(rows)
print('\n--- FICR loss by group x actual-cf bucket (share of that group\'s denominator) ---')
print(T.to_string(index=False, float_format=lambda x: f'{x:8.4f}'))
print('\ntotal ficr_lost per group:', T.groupby('g').ficr_lost.sum().round(4).to_dict())

# oracle: what if we could apply the best CONSTANT additive shift per (group, predicted-cf bucket)?
V2 = B.copy(); V2['pb'] = pd.cut(V2.cf_pred, np.arange(0, 1.11, 0.1))
best_shift = {}
for (g, pb), s in V2.groupby(['group_id', 'pb'], observed=True):
    grid = np.arange(-0.10, 0.1001, 0.005)
    sv = s[s.cf_act >= 0.10]
    if len(sv) < 30:
        best_shift[(g, str(pb))] = 0.0; continue
    sc = []
    for d in grid:
        a = np.clip(sv.cf_pred + d, 0, 1.1); e = np.abs(a - sv.cf_act)
        u = np.select([e <= 0.06, e <= 0.08], [4.0, 3.0], 0.0)
        sc.append(-e.mean() + (sv.cf_act * u).sum() / (4 * sv.cf_act.sum()))
    best_shift[(g, str(pb))] = float(grid[int(np.argmax(sc))])
print('\n--- in-sample oracle per (group, pred-bucket) additive shift ---')
print({f'g{k[0]}_{k[1]}': v for k, v in best_shift.items() if abs(v) > 1e-9})
V2['sh'] = [best_shift.get((g, str(pb)), 0.0) for g, pb in zip(V2.group_id, V2.pb)]
V2['prediction_kwh'] = np.clip(V2.cf_pred + V2.sh, 0, 1.1) * V2.group_id.map(CAPS)
o = official_total(V2[['group_id', 'actual_kwh', 'prediction_kwh']])
print('in-sample oracle-shift total', round(o['total'], 6), '(vs', round(bl['total'], 6), ')')
json.dump({'anatomy': T.to_dict('records'), 'oracle_shift_insample': o['total'],
           'current_best': bl['total']},
          open('/Users/um-yunsang/BARAM2026/research/nodes/S12-N2_diag.json', 'w'), indent=1, default=str)
