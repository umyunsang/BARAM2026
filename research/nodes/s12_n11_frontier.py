"""S12-N11 * where do we actually sit on the (1-NMAE, FICR) frontier?

research/lanes/S12_ext_dacon_solutions.md established from the public leaderboard's top-100
rows that our online 1-NMAE (0.858775) is below ALL of them (min 0.86777, median 0.87425) and
below the organiser's own RandomForest baseline (0.86371), while our FICR is competitive.  Two
mutually exclusive explanations:

  H1 FRONTIER-POSITION.  Our raw point accuracy is fine; the decision layer deliberately trades
     NMAE for FICR, so a pure point forecast from the same models would already reach
     1-NMAE ~ 0.874 and the field's numbers say nothing about our accuracy.
  H2 ACCURACY-DEFICIT.  Our raw point accuracy really is below the field, and the decision
     layer is compensating for a weak point forecast rather than exploiting a strong one.

They are separated by one measurement that needs no fitting: score the pure point forecasts we
already have (teacher pc_hat; the conditional mean and median of member D's 26-class
distribution; the deployed M93_POWER_QUANTILE MEDIAN column) under NMAE alone, and trace the
whole gamma frontier of D's decision layer.  Under H1 the gamma=0 end of the frontier reaches
~0.874; under H2 it does not, and no amount of decision-layer work can close the gap.
"""
import sys, json
import numpy as np, pandas as pd
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/nodes')
from loop_lib import *

N = '/Users/um-yunsang/BARAM2026/research/nodes/'
AB = '/Users/um-yunsang/BARAM2026/artifacts/backtests/metric-aligned-probe/'


def sc(df, pred):
    return official_total(df.assign(prediction_kwh=pred)[['group_id', 'actual_kwh', 'prediction_kwh']])


if __name__ == '__main__':
    R = canonical_keys()
    out = {}
    base = pd.DataFrame({'group_id': R.group_id.to_numpy(),
                         'actual_kwh': R.cf.to_numpy() * R.group_id.map(CAPS).to_numpy(),
                         'fold_id': R.fold_id.to_numpy()})
    capv = R.group_id.map(CAPS).to_numpy()

    print('--- pure point forecasts (no decision layer) ---')
    C = (np.arange(26) + 0.5) * W
    for tag in ['D', 'DL', 'DG', 'KNN120', 'KNN300', 'KNN60']:
        try:
            P = align_prob(tag, R)
        except FileNotFoundError:
            continue
        mean = (P * C[None, :]).sum(axis=1)
        cdf = np.cumsum(P, axis=1)
        med = C[np.argmax(cdf >= 0.5, axis=1)]
        for nm, v in [('mean', mean), ('median', med)]:
            s = sc(base, v * capv)
            out[f'{tag}_{nm}'] = s['total']
            print(f'  {tag:8s} {nm:7s} 1-NMAE={s["one_minus_nmae"]:.6f}  FICR={s["ficr"]:.6f}  total={s["total"]:.6f}')

    # deployed quantile member's own median column
    parts = []
    for f in FOLDS:
        d = pd.read_parquet(AB + f'M93_POWER_QUANTILE-{f}-policies.parquet').copy()
        d['fold_id'] = f
        parts.append(d[KEY + ['actual_kwh', 'MEDIAN']])
    M93 = pd.concat(parts, ignore_index=True)
    s = sc(M93, M93.MEDIAN)
    out['M93_MEDIAN'] = s['total']
    print(f'  {"M93":8s} {"MEDIAN":7s} 1-NMAE={s["one_minus_nmae"]:.6f}  FICR={s["ficr"]:.6f}  total={s["total"]:.6f}')

    print('\n--- gamma frontier of member D (temperature fixed at the fold-outside pick T=1.0) ---')
    fr = utility_frames(align_prob('D', R), R, temps=[1.0],
                        gammas=[0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 40.0])
    rows = []
    for (tp, gm), v in sorted(fr.items(), key=lambda x: x[0][1]):
        s = sc(base, v)
        rows.append(dict(gamma=gm, one_minus_nmae=s['one_minus_nmae'], ficr=s['ficr'], total=s['total']))
        print(f'  gamma={gm:5.2f}  1-NMAE={s["one_minus_nmae"]:.6f}  FICR={s["ficr"]:.6f}  total={s["total"]:.6f}')
    out['gamma_frontier_D'] = rows

    print('\n--- gamma frontier of the DEPLOYED average (its own policy columns) ---')
    fr2 = []
    for gm in ['G0', 'G0.2', 'G0.35', 'G0.5', 'G0.75', 'G1', 'G1.25', 'G1.5', 'G2']:
        cols = []
        for stem, tpre in [('M102_TOP100', 'T0.5'), ('M113_LGBM_DART', 'T0.5'), ('M115_XGBOOST', 'T0.6')]:
            ps = []
            for f in FOLDS:
                d = pd.read_parquet(AB + f'{stem}-{f}-policies.parquet').copy()
                d['fold_id'] = f
                ps.append(d[KEY + ['actual_kwh', f'{tpre}_{gm}']].rename(columns={f'{tpre}_{gm}': stem}))
            cols.append(pd.concat(ps, ignore_index=True))
        Jd = cols[0]
        for x in cols[1:]:
            Jd = Jd.merge(x.drop(columns=['actual_kwh']), on=KEY)
        v = Jd[['M102_TOP100', 'M113_LGBM_DART', 'M115_XGBOOST']].mean(axis=1)
        s = sc(Jd, v)
        fr2.append(dict(gamma=gm, one_minus_nmae=s['one_minus_nmae'], ficr=s['ficr'], total=s['total']))
        print(f'  {gm:6s}  1-NMAE={s["one_minus_nmae"]:.6f}  FICR={s["ficr"]:.6f}  total={s["total"]:.6f}')
    out['gamma_frontier_DEPAVG'] = fr2

    print('\n--- what 1-NMAE would we need at our current FICR to reach 0.66? ---')
    print(f'  need (1-NMAE) = 1.32 - FICR = {1.32 - 0.410503:.6f}   (we have 0.861866)')
    print(f'  need FICR      = 1.32 - (1-NMAE) = {1.32 - 0.861866:.6f}   (we have 0.410503)')
    json.dump(out, open(N + 'S12-N11_frontier.json', 'w'), indent=1, default=str)
