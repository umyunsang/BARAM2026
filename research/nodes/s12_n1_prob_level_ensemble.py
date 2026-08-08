"""S12-N1 · probability-level ensembling + sub-bin utility refinement.

Untried axis. Every blend in S10/S9-N14 combined members in ACTION space
(w*kwh_a + (1-w)*kwh_b). Under a step reward the action is an argmax of an expected
utility, and argmax does not commute with averaging: the principled ensemble averages the
PREDICTIVE DISTRIBUTIONS and takes one argmax of the ensemble utility. All 15 saved
members expose a 26-class probability matrix on the same key set, so this is testable
with zero re-fitting.

Treatments, each 0 fitted dof beyond the existing (T,G) fold-outside policy pick:
  N1a  sub-bin refinement of the utility integral on D alone (control for N1b/N1c)
  N1b  uniform probability-average over member subsets
  N1c  log-opinion-pool (geometric mean) over the same subsets
"""
import sys, json, itertools
import numpy as np
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/nodes')
from loop_lib import *

R = canonical_keys(); dep = load_depavg()
out = {}

print('--- N1a: sub-bin refinement of the expected-utility integral (member D) ---', flush=True)
PD = align_prob('D', R)
for sb in (1, 2, 4, 8):
    r, _, _ = evaluate_prob(PD, R, tag=f'D subbin={sb}', subbin=sb, dep=dep)
    out[f'N1a_subbin_{sb}'] = r

print('\n--- member solos (control, subbin=1) ---', flush=True)
PROB = {m: align_prob(m, R) for m in ALL_MEMBERS}
solos = {}
for m in ALL_MEMBERS:
    fr = utility_frames(PROB[m], R)
    _, s, _ = fo_policy(fr, R)
    solos[m] = s['total']
    print(f'  {m:4s} solo={s["total"]:.6f}', flush=True)
out['member_solos'] = solos

print('\n--- N1b: uniform probability-average ensembles ---', flush=True)
SUBSETS = {
    'ALL15': ALL_MEMBERS,
    'TOP5': sorted(solos, key=solos.get, reverse=True)[:5],
    'TOP8': sorted(solos, key=solos.get, reverse=True)[:8],
    'FAMILY_DIVERSE': ['D', 'X', 'M2', 'XG', 'R2', 'LV'],
    'D_X': ['D', 'X'],
    'D_X_R2_LV': ['D', 'X', 'R2', 'LV'],
}
for nm, subset in SUBSETS.items():
    Pm = np.mean([PROB[m] for m in subset], axis=0)
    for sb in (1, 4):
        r, _, _ = evaluate_prob(Pm, R, tag=f'PAVG[{nm}] sb={sb}', subbin=sb, dep=dep)
        out[f'N1b_{nm}_sb{sb}'] = r | {'members': subset}

print('\n--- N1c: log-opinion pool (geometric mean) ---', flush=True)
for nm, subset in SUBSETS.items():
    L = np.mean([np.log(np.clip(PROB[m], 1e-9, None)) for m in subset], axis=0)
    Pg = np.exp(L); Pg = Pg / Pg.sum(axis=1, keepdims=True)
    r, _, _ = evaluate_prob(Pg, R, tag=f'LOP[{nm}] sb=4', subbin=4, dep=dep)
    out[f'N1c_{nm}'] = r | {'members': subset}

json.dump(out, open('/Users/um-yunsang/BARAM2026/research/nodes/S12-N1_prob_ensemble.json', 'w'),
          indent=1, default=str)
best = max((k for k in out if k.startswith('N1')), key=lambda k: out[k]['blend'])
print(f'\nBEST: {best} blend={out[best]["blend"]:.6f} '
      f'delta_vs_current_best={out[best]["delta_vs_current_best"]:+.6f}', flush=True)
