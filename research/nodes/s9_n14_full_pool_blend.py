
"""S9-N14 · re-search the blend space using EVERY saved member in research/nodes/
(P,L,G,Q,D,M,X,M2,R,R1,R2,LV,S1,W2,XG) plus DEPAVG -- all already-fit, no retraining
needed, so this is cheap. member() itself is unaffected by DEF-1 (it has its own
correctly-implemented temperature scaling on real probabilities, not harness.py's
buggy uniform-vector pattern), so every saved probability array here is already valid
evidence, no re-fit required.

Goal: current_best is DEPAVG+D (1 dof, 0.636184) chosen in research/nodes/s10_final3.py
before D's sibling members (X, M2, and the s7_members2.py cohort: R,R1,R2,LV,S1,W2,XG)
existed or were fully explored together. Re-run the SAME fold-outside blend search over
the now-complete pool to see whether a member the 2026-08-06/07 sessions didn't have
access to changes the answer.
"""
import sys, json, itertools
import numpy as np, pandas as pd
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/scratch')
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/nodes')
from lib import official_total, FOLDS, CAPS

N = '/Users/um-yunsang/BARAM2026/research/nodes/'
AB = '/Users/um-yunsang/BARAM2026/artifacts/backtests/metric-aligned-probe/'
KEY = ['fold_id', 'group_id', 'forecast_kst_dtm']
DEP = {'M102_TOP100': 'T0.5_G1.5', 'M113_LGBM_DART': 'T0.5_G0.5', 'M115_XGBOOST': 'T0.6_G0.35'}
W = 0.04
ACT = np.arange(0.02, 1.0801, 0.0025)
SC = {1: 0.985, 2: 0.989, 3: 1.005}
TEMPS = [0.6, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
GAMMAS = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0]

MEMBERS = ['P', 'L', 'G', 'Q', 'D', 'M', 'X', 'M2', 'R', 'R1', 'R2', 'LV', 'S1', 'W2', 'XG']


def member(name):
    kp = N + f'S7-N8_{name}_keys.parquet'; pp = N + f'S7-N8_{name}_prob.npy'
    import os
    if not (os.path.exists(kp) and os.path.exists(pp)):
        return None, None
    R = pd.read_parquet(kp); P = np.load(pp)
    NC = P.shape[1]; C = (np.arange(NC) + 0.5) * W
    err = np.abs(ACT[:, None] - C[None, :]); units = np.where(err <= 0.06, 4., np.where(err <= 0.08, 3., 0.))
    g = R.group_id.to_numpy(); mg = R.mean_gen_g.to_numpy()
    capv = np.array([CAPS[x] for x in g]); hi = np.array([SC[x] for x in g]); act = R.cf.to_numpy() * capv
    mask = (C >= 0.10).astype(float); frames = {}
    for tp in TEMPS:
        q = P ** (1.0 / tp); q = q / np.maximum(q.sum(axis=1, keepdims=True), 1e-12)
        q = q * mask[None, :]; q = q / np.maximum(q.sum(axis=1, keepdims=True), 1e-12)
        nm = -(q @ err.T); fic = (q @ ((C[None, :] * units).T))
        for gm in GAMMAS:
            frames[(tp, gm)] = np.minimum(ACT[np.argmax(nm + gm * fic / (4.0 * mg[:, None]), axis=1)], hi) * capv
    Dm = pd.DataFrame({'fold_id': R.fold_id, 'group_id': g, 'forecast_kst_dtm': R.forecast_kst_dtm, 'actual_kwh': act})
    out = np.empty(len(Dm))
    for f in FOLDS:
        sel = (Dm.fold_id == f).to_numpy()
        s2 = {k: official_total(Dm[~sel].assign(prediction_kwh=v[~sel])[['group_id', 'actual_kwh', 'prediction_kwh']])['total']
              for k, v in frames.items()}
        out[sel] = frames[max(s2, key=s2.get)][sel]
    Dm['prediction_kwh'] = out
    return Dm, official_total(Dm[['group_id', 'actual_kwh', 'prediction_kwh']])['total']


def load_depavg():
    fr = []
    for stem, pol in DEP.items():
        parts = []
        for f in FOLDS:
            d = pd.read_parquet(AB + f'{stem}-{f}-policies.parquet')
            d = d.copy(); d['fold_id'] = f
            parts.append(d[KEY + [pol]].rename(columns={pol: stem}))
        fr.append(pd.concat(parts, ignore_index=True))
    J = fr[0]
    for x in fr[1:]:
        J = J.merge(x, on=KEY)
    J['DEPAVG'] = J[list(DEP)].mean(axis=1)
    return J[KEY + ['DEPAVG']]


def fo_blend(J, cols, grid, actual='actual_kwh'):
    rows = []; picks = {}
    for f in FOLDS:
        oth = J[J.fold_id != f]; held = J[J.fold_id == f]
        best = None
        for wv in grid:
            pred = sum(w * oth[c] for w, c in zip(wv, cols))
            t = official_total(oth.assign(prediction_kwh=pred)[['group_id', actual, 'prediction_kwh']]
                                .rename(columns={actual: 'actual_kwh'}))['total']
            if best is None or t > best[0]:
                best = (t, wv)
        picks[f] = best[1]
        pred_h = sum(w * held[c] for w, c in zip(best[1], cols))
        rows.append(held.assign(prediction_kwh=pred_h))
    D = pd.concat(rows, ignore_index=True)
    if actual != 'actual_kwh':
        D = D.rename(columns={actual: 'actual_kwh'})
    return official_total(D[['group_id', 'actual_kwh', 'prediction_kwh']]), picks


if __name__ == '__main__':
    dep = load_depavg()
    solos = {}
    J = None
    for name in MEMBERS:
        Dm, solo = member(name)
        if Dm is None:
            print(f'{name}: MISSING (no saved prob file)', flush=True)
            continue
        solos[name] = solo
        print(f'{name}: solo fold-outside = {solo:.6f}', flush=True)
        col = Dm[KEY + ['actual_kwh', 'prediction_kwh']].rename(columns={'prediction_kwh': name})
        J = col if J is None else J.merge(col.drop(columns=['actual_kwh']), on=KEY)
    J = J.merge(dep, on=KEY)

    print('\n--- 1-dof blends (member + DEPAVG) ---', flush=True)
    g2w = [(w, 1 - w) for w in np.arange(0, 1.001, 0.05)]
    best1 = None
    for name in solos:
        r, picks = fo_blend(J, [name, 'DEPAVG'], g2w)
        print(f'  DEPAVG+{name}: {r["total"]:.6f}', flush=True)
        if best1 is None or r['total'] > best1[0]:
            best1 = (r['total'], name, r)
    print(f'\nBEST 1-dof: DEPAVG+{best1[1]} = {best1[0]:.6f}  (current_best reference: 0.636184)', flush=True)

    print('\n--- top pairwise 2-dof blends (member1+member2, DEPAVG excluded to search diversity) ---', flush=True)
    g3w = [(x, y, 1 - x - y) for x in np.arange(0, 1.01, 0.1) for y in np.arange(0, 1.01 - x + 1e-9, 0.1)]
    top_solo = sorted(solos, key=solos.get, reverse=True)[:5]
    best3 = None
    tried = []
    for a, b in itertools.combinations(top_solo, 2):
        r, picks = fo_blend(J, ['DEPAVG', a, b], g3w)
        tried.append((r['total'], f'DEPAVG+{a}+{b}'))
        if best3 is None or r['total'] > best3[0]:
            best3 = (r['total'], f'DEPAVG+{a}+{b}', r)
    for t, name in sorted(tried, reverse=True)[:8]:
        print(f'  {name}: {t:.6f}', flush=True)
    print(f'\nBEST 2-dof (of DEPAVG+top5): {best3[1]} = {best3[0]:.6f}', flush=True)

    json.dump({'solos': solos, 'best_1dof': {'name': best1[1], 'total': best1[0]},
               'best_2dof_of_top5': {'name': best3[1], 'total': best3[0]},
               'current_best_reference': 0.636184}, open(N + 'S9-N14_full_pool_blend.json', 'w'), indent=1, default=str)
