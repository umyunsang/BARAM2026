"""S14-N2 * the noise floor from DESIGNED REPLICATES, not from a k-NN probe.

Why this supersedes our existing floor estimate.  research/nodes/s11_floor.py estimated the
irreducible conditional spread with a k-nearest-neighbour probe and reported conditional MAD
~0.110 at k=3.  The S14 foundation lane flags that estimate as contaminated: the mean neighbour
radius at k=3 is 1.22 standard deviations in a 15-dimensional standardised space, which is not
local at all (CS4780 L2 / ESL 2.5 on the curse of dimensionality), so the probe mixes genuine
irreducible noise with real state differences and is only an upper bound of unknown tightness.

But this site hands us something far better, which nobody has used: g1 and g2 are DESIGNED
REPLICATES.  Both are VESTAS V126, six turbines each, hub 117 m, on the same 2.5 km ridge, inside
the SAME LDAPS 4x4 box and the SAME GFS 3x3 box.  At a given hour they receive the same NWP
information; the only differences are array orientation and position along the ridge.  So
(y1, y2) at the same hour is a pair of replicate outcomes of "what did this weather produce",
and the classical replicate decomposition (STAT462 3.7 lack-of-fit vs pure error) applies:

    E|y1 - y2|   estimates the PURE ERROR -- the part no function of the NWP can remove
    E|y1 - yhat1| is our achieved error
    the difference is LACK OF FIT -- the only part any modelling work can still take.

For two i.i.d. draws from the same conditional law, the Gini mean difference E|y1-y2| relates to
the best achievable absolute loss E|y - median| by a shape factor: 1.414 for a Gaussian, 1.5 for
a Laplace.  So min achievable MAE ~ E|y1-y2| / 1.4-1.5, and we can compare that directly with our
measured 0.13858.

This is the measurement that decides whether Total 0.66 is reachable at all, so it is reported
with its assumptions and their violations stated explicitly rather than as a single number.
"""
import sys, json
import numpy as np, pandas as pd
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/scratch')
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/nodes')
from lib import CAPS, official_total
from loop_lib import canonical_keys, align_prob, utility_frames, fo_policy, load_depavg, KEY, W

N = '/Users/um-yunsang/BARAM2026/research/nodes/'
S = '/Users/um-yunsang/BARAM2026/research/scratch/'

if __name__ == '__main__':
    LAB = pd.read_parquet(S + 'labels.parquet').set_index('kst_dtm')
    T = pd.read_parquet(S + 'teacher_targets.parquet')
    cf = pd.DataFrame({g: LAB[f'kpx_group_{g}'] / CAPS[g] for g in (1, 2, 3)})
    dev = (cf.index >= '2023-04-01 01:00') & (cf.index <= '2024-01-01')

    print('=== replicate pair g1 vs g2 (both VESTAS V126 x6, same ridge, same NWP box) ===')
    for name, m in [('dev-2023 folds', dev), ('all labelled hours', np.ones(len(cf), bool))]:
        sub = cf[m].dropna(subset=[1, 2])
        both = sub[(sub[1] >= 0.1) | (sub[2] >= 0.1)]
        d = (both[1] - both[2]).abs()
        print(f'  {name:20s} n={len(both):6d}  E|y1-y2| = {d.mean():.5f}   '
              f'median={d.median():.5f}  sd={d.std():.5f}  corr={both[1].corr(both[2]):.4f}')
        for shape, fac in [('Gaussian', 1.414), ('Laplace', 1.500)]:
            print(f'      -> implied floor on MAE ({shape} shape factor {fac}): '
                  f'{d.mean()/fac:.5f}')

    print('\n=== the same pair, restricted to hours BOTH groups are scored (y>=0.1) ===')
    sub = cf[dev].dropna(subset=[1, 2])
    both = sub[(sub[1] >= 0.1) & (sub[2] >= 0.1)]
    d = (both[1] - both[2]).abs()
    print(f'  n={len(both)}  E|y1-y2|={d.mean():.5f}  floor(Gauss)={d.mean()/1.414:.5f}  '
          f'floor(Laplace)={d.mean()/1.5:.5f}')

    print('\n=== how much of the replicate difference is WIND vs AVAILABILITY? ===')
    v1 = T['g1_v_mean'].reindex(both.index); v2 = T['g2_v_mean'].reindex(both.index)
    p1 = T['g1_pc'].reindex(both.index); p2 = T['g2_pc'].reindex(both.index)
    ok = np.isfinite(v1) & np.isfinite(v2) & np.isfinite(p1) & np.isfinite(p2)
    print(f'  n with measured wind for both: {int(ok.sum())}')
    print(f'  E|v1-v2| (measured hub wind, m/s)      = {np.abs(v1[ok]-v2[ok]).mean():.4f}')
    print(f'  E|pc1-pc2| (physics power, both clean) = {np.abs(p1[ok]-p2[ok]).mean():.5f}')
    print(f'  E|y1-y2|   (metered)                   = {np.abs(both[1][ok]-both[2][ok]).mean():.5f}')
    print('  -> the metered gap ABOVE the physics gap is availability/outage, which no NWP model')
    print('     can predict; the physics gap is genuine micro-siting the NWP box cannot resolve.')

    print('\n=== our achieved error against these floors ===')
    R = canonical_keys(); dep = load_depavg()
    Dm, _, _ = fo_policy(utility_frames(align_prob('D', R), R), R)
    J = Dm[KEY + ['actual_kwh', 'prediction_kwh']].rename(columns={'prediction_kwh': 'D'}).merge(dep, on=KEY)
    J['pred'] = 0.30 * J.D + 0.70 * J.DEPAVG
    capv = J.group_id.map(CAPS)
    v = J[J.actual_kwh >= 0.1 * capv]
    mae = ((v.pred - v.actual_kwh).abs() / v.group_id.map(CAPS)).mean()
    print(f'  champion MAE (scored rows)            = {mae:.5f}')
    print(f'  best point forecast MAE               = {1-0.866147:.5f}')
    tgt = 1 - (0.876705)
    print(f'  MAE required for Total 0.66           = {tgt:.5f}')

    out = {}
    d_all = (cf.dropna(subset=[1, 2]).pipe(lambda s: (s[1] - s[2]).abs()))
    out['E_abs_y1_y2_all'] = float(d_all.mean())
    out['floor_gauss'] = float(d_all.mean() / 1.414)
    out['floor_laplace'] = float(d_all.mean() / 1.5)
    out['champion_mae'] = float(mae)
    out['required_mae'] = float(tgt)
    json.dump(out, open(N + 'S14-N2_pure_error.json', 'w'), indent=1, default=str)
    print('\n--- VERDICT INPUTS -------------------------------------------------')
    print(f'  pure-error floor on MAE : {out["floor_laplace"]:.5f} .. {out["floor_gauss"]:.5f}')
    print(f'  we are at               : {out["champion_mae"]:.5f}')
    print(f'  we would need           : {out["required_mae"]:.5f}')
