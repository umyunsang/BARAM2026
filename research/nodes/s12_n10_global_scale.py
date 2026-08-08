"""S12-N10 * global affine recalibration of the deployed action (alpha / delta / both).

Origin: research/lanes/S12_ext_dacon_solutions.md.  That lane pulled the public leaderboard's
top-100 rows and found that our online 1-NMAE (0.858775) is BELOW every one of the top 100
(min 0.86777, median 0.87425) and below the organiser's own RF baseline (0.86371), while our
FICR is competitive.  It also reports a competitor measurement that a single global scale
alpha ~ 1.0275 improved 1-NMAE and FICR SIMULTANEOUSLY, attributed to systematic
under-prediction at high output (LDAPS smooths the 1078 m Gadeoksan ridge, so its near-surface
wind is biased low over the summit).

This is NOT the treatment S12-N7 rejected.  N7 rescaled about each group's mean,
a' = c + k(a - c) with c ~ 0.5, which is a pure dispersion change with no net shift; it found
k = 1.00 optimal even in-sample.  A multiplicative alpha about ZERO carries an implicit shift
of c*(alpha-1), so the two families intersect only at identity.  This node separates the two
components explicitly and gates each.

Our own S12-N2 anatomy independently supports the sign: bias by ACTUAL capacity-factor bucket
runs +0.10..+0.13 at cf 0.1-0.3 and -0.09..-0.14 at cf 0.8-1.05.  Because FICR weights by
actual production while NMAE weights valid rows equally, the high-cf under-prediction costs
FICR far more than the low-cf over-prediction costs NMAE, so a positive shift can be
FICR-accretive at little NMAE cost -- exactly the asymmetry the lane reports.

All three variants are nested at identity, so the fold-outside gate can reject to baseline.
"""
import sys, json
import numpy as np, pandas as pd
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/nodes')
from loop_lib import *
from s12_n7_dispersion import build_blend

N = '/Users/um-yunsang/BARAM2026/research/nodes/'
HI = {1: 0.985, 2: 0.989, 3: 1.005}


def apply_affine(df, col, alpha, delta):
    cap = df.group_id.map(CAPS).to_numpy()
    hi = df.group_id.map(HI).to_numpy() * cap
    return np.clip(alpha * df[col].to_numpy() + delta * cap, 0.0, hi)


def score(df, col, alpha, delta):
    return official_total(df.assign(prediction_kwh=apply_affine(df, col, alpha, delta))[
        ['group_id', 'actual_kwh', 'prediction_kwh']])


def fo_gate(B, col, grid, label):
    rows = []; picks = {}
    for f in FOLDS:
        oth = B[B.fold_id != f]; held = B[B.fold_id == f]; best = None
        for (al, de) in grid:
            t = score(oth, col, al, de)['total']
            if best is None or t > best[0]:
                best = (t, (al, de))
        picks[f] = best[1]
        rows.append(held.assign(prediction_kwh=apply_affine(held, col, *best[1])))
    D = pd.concat(rows, ignore_index=True)
    s = official_total(D[['group_id', 'actual_kwh', 'prediction_kwh']])
    print(f'  {label:26s} FO total={s["total"]:.6f}  d={s["total"]-0.6361842493883538:+.6f}  '
          f'1-NMAE={s["one_minus_nmae"]:.6f} FICR={s["ficr"]:.6f}  picks={list(picks.values())}', flush=True)
    return {'total': s['total'], 'one_minus_nmae': s['one_minus_nmae'], 'ficr': s['ficr'],
            'picks': {k: list(v) for k, v in picks.items()}}, D


if __name__ == '__main__':
    B = build_blend()
    base = score(B, 'BLEND', 1.0, 0.0)
    print(f'baseline {base["total"]:.6f} (1-NMAE={base["one_minus_nmae"]:.6f} FICR={base["ficr"]:.6f})\n')
    out = {'baseline': base['total'], 'insample_alpha': {}, 'insample_delta': {}}

    print('--- in-sample pure multiplicative alpha (diagnostic) ---')
    for al in [1.0, 1.01, 1.0175, 1.0275, 1.035, 1.05, 1.07, 1.10]:
        s = score(B, 'BLEND', al, 0.0)
        out['insample_alpha'][al] = s['total']
        print(f'  alpha={al:6.4f} total={s["total"]:.6f}  1-NMAE={s["one_minus_nmae"]:.6f}  FICR={s["ficr"]:.6f}')

    print('\n--- in-sample pure additive delta (fraction of capacity) ---')
    for de in [0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04]:
        s = score(B, 'BLEND', 1.0, de)
        out['insample_delta'][de] = s['total']
        print(f'  delta={de:6.4f} total={s["total"]:.6f}  1-NMAE={s["one_minus_nmae"]:.6f}  FICR={s["ficr"]:.6f}')

    print('\n--- fold-outside gates ---')
    ag = [(a, 0.0) for a in np.arange(0.96, 1.1201, 0.0025)]
    dg = [(1.0, d) for d in np.arange(-0.03, 0.0501, 0.0025)]
    bg = [(a, d) for a in np.arange(0.96, 1.1001, 0.005) for d in np.arange(-0.03, 0.0401, 0.005)]
    out['fo_alpha'], _ = fo_gate(B, 'BLEND', ag, 'alpha only (1 dof)')
    out['fo_delta'], _ = fo_gate(B, 'BLEND', dg, 'delta only (1 dof)')
    out['fo_affine'], Daff = fo_gate(B, 'BLEND', bg, 'affine alpha+delta (2 dof)')

    # same gate applied to the raw D member and to DEPAVG separately, to locate the bias
    R = canonical_keys(); dep = load_depavg()
    Dm, _, _ = fo_policy(utility_frames(align_prob('D', R), R), R)
    J = Dm[KEY + ['actual_kwh', 'prediction_kwh']].rename(columns={'prediction_kwh': 'D'}).merge(dep, on=KEY)
    print('\n--- same gate on each component separately ---')
    for c in ['D', 'DEPAVG']:
        out[f'fo_alpha_{c}'], _ = fo_gate(J, c, ag, f'alpha on {c}')
    json.dump(out, open(N + 'S12-N10_global_scale.json', 'w'), indent=1, default=str)
