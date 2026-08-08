"""Decisive B4 test: is ANY of the ~30 candidates distinguishable from the champion?"""
import sys, json
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/engine')
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/nodes')
import numpy as np, pandas as pd
from loop_lib import (canonical_keys, load_depavg, align_prob, utility_frames, fo_policy,
                      KEY, CAPS, official_total)
from mcs import run_mcs, loss_series

R = canonical_keys(); dep = load_depavg()
cands = {}
members = ['D', 'DV', 'DVT', 'DL', 'DG', 'KNN120', 'KNN60', 'KNN300', 'DW1', 'DW2', 'DW3']
acts = {}
for m in members:
    try:
        Dm, _, _ = fo_policy(utility_frames(align_prob(m, R), R), R)
    except FileNotFoundError:
        continue
    acts[m] = Dm[KEY + ['actual_kwh', 'prediction_kwh']].rename(columns={'prediction_kwh': m})
J = None
for m, d in acts.items():
    J = d if J is None else J.merge(d.drop(columns=['actual_kwh']), on=KEY)
J = J.merge(dep, on=KEY)
have = [m for m in members if m in J.columns]
# candidate set: the champion, each member solo, and each member blended 0.3/0.7 with DEPAVG
J['CHAMPION'] = 0.30 * J['D'] + 0.70 * J['DEPAVG']
cols = ['CHAMPION', 'DEPAVG']
for m in have:
    J[f'BL_{m}'] = 0.30 * J[m] + 0.70 * J['DEPAVG']
    cols += [f'BL_{m}']
cols += have
cols = list(dict.fromkeys(cols))
print(f'{len(cols)} candidates, {len(J)} rows')

tot = {c: official_total(J.assign(prediction_kwh=J[c])[
    ['group_id', 'actual_kwh', 'prediction_kwh']])['total'] for c in cols}
for c, v in sorted(tot.items(), key=lambda t: -t[1]):
    print(f'  {c:14s} Total={v:.6f}')

# sanity: the linearised loss must reproduce Total exactly
for c in cols[:3]:
    s = 0.5 - loss_series(J, c).sum()
    assert abs(s - tot[c]) < 1e-9, (c, s, tot[c])
print('\nlinearised per-row loss reproduces Total exactly (max err < 1e-9)')

# MCS requires distinct loss series: CHAMPION and BL_D are the same column by construction,
# and a zero-variance difference makes the R-method's elimination step ill-defined.
Lall = pd.DataFrame({c: __import__('mcs').loss_series(J, c) for c in cols})
keep, seen = [], []
for c in cols:
    v = Lall[c].to_numpy()
    if any(np.allclose(v, Lall[s].to_numpy(), atol=1e-12) for s in seen):
        print(f'  dropping {c}: loss series identical to an earlier candidate')
        continue
    seen.append(c); keep.append(c)
cols = keep
print(f'{len(cols)} distinct candidates after de-duplication')
mcs, L = run_mcs(J, cols, size=0.10, reps=2000, block=168)
print('\n=== MODEL CONFIDENCE SET (size=0.10, stationary bootstrap, block=168 rows) ===')
pv = mcs.pvalues.sort_values('Pvalue', ascending=False)
print(pv.to_string())
inc = list(mcs.included)
exc = list(mcs.excluded)
print(f'\nINCLUDED ({len(inc)}): {inc}')
print(f'EXCLUDED ({len(exc)}): {exc}')
print('\nreading: if the champion is the ONLY included model, every one of our ~30 comparisons')
print('was a genuine loss, not noise, and the search family is exhausted. If many models are')
print('included, they are statistically indistinguishable and "champion" is a coin flip.')
json.dump({'totals': tot, 'included': inc, 'excluded': exc,
           'pvalues': mcs.pvalues.to_dict()},
          open('/Users/um-yunsang/BARAM2026/research/engine/mcs_result.json', 'w'),
          indent=1, default=str)
