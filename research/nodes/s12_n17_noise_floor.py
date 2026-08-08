"""S12-N17 * how big is a real difference on this local metric?

Every S12 treatment -- probability-level ensembling, the 12-stem deployed pool, the analog
lineage, source-separated members, source stacking, dispersion rescaling, ordinal smoothing,
global affine recalibration, valid-conditional training, distribution recentring -- landed in
[0.6311, 0.6362] and none exceeded the incumbent 0.6361842.  A one-sided cloud strictly below
an incumbent that was itself chosen as the maximum of a large prior search is the signature of
a selection maximum, not of a genuinely superior configuration.  That hypothesis is only worth
asserting if the metric's own sampling noise is measured, so this node measures it.

Method: moving-block bootstrap over calendar DAYS (block length 7 days) within each fold,
resampling whole days so that the strong within-day and day-to-day autocorrelation of wind is
preserved; the official score is recomputed from scratch on every replicate, including its
per-group validity gate and its actual-weighted FICR denominator.  Paired replicates are used
for candidate-minus-incumbent differences so the comparison is not inflated by the common
component.
"""
import sys, json
import numpy as np, pandas as pd
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/nodes')
from loop_lib import *

N = '/Users/um-yunsang/BARAM2026/research/nodes/'
RNG = np.random.default_rng(20260807)
B = 400
BLOCK = 7


def blocks_for(fold_days, block=BLOCK):
    n = len(fold_days)
    return [fold_days[i:i + block] for i in range(0, n, block)]


def bootstrap(df, cols, n_rep=B):
    """df must carry fold_id, group_id, actual_kwh and one column per candidate."""
    df = df.copy()
    df['day'] = pd.to_datetime(df.forecast_kst_dtm).dt.normalize()
    by_fold = {f: np.sort(d.day.unique()) for f, d in df.groupby('fold_id')}
    idx_by_day = {(f, d): g.index.to_numpy() for (f, d), g in df.groupby(['fold_id', 'day'])}
    reps = {c: [] for c in cols}
    for _ in range(n_rep):
        take = []
        for f, days in by_fold.items():
            bl = blocks_for(list(days))
            need = len(days)
            got = 0
            while got < need:
                b = bl[RNG.integers(len(bl))]
                for d in b:
                    take.append(idx_by_day[(f, d)])
                    got += 1
                    if got >= need:
                        break
        sel = np.concatenate(take)
        sub = df.loc[sel]
        for c in cols:
            reps[c].append(official_total(sub.assign(prediction_kwh=sub[c])[
                ['group_id', 'actual_kwh', 'prediction_kwh']])['total'])
    return {c: np.asarray(v) for c, v in reps.items()}


if __name__ == '__main__':
    R = canonical_keys(); dep = load_depavg()
    frames = {}
    for tag in ['D', 'DV', 'KNN120', 'DL', 'DG']:
        try:
            Dm, _, _ = fo_policy(utility_frames(align_prob(tag, R), R), R)
        except FileNotFoundError:
            continue
        frames[tag] = Dm[KEY + ['actual_kwh', 'prediction_kwh']].rename(columns={'prediction_kwh': tag})
    J = frames['D']
    for t, fr in frames.items():
        if t != 'D':
            J = J.merge(fr.drop(columns=['actual_kwh']), on=KEY)
    J = J.merge(dep, on=KEY)
    # incumbent blend at its fold-outside weight (0.30 on every fold)
    J['INCUMBENT'] = 0.30 * J['D'] + 0.70 * J['DEPAVG']
    J['DV_BLEND'] = 0.30 * J['DV'] + 0.70 * J['DEPAVG']
    J['KNN_BLEND'] = 0.30 * J['KNN120'] + 0.70 * J['DEPAVG']

    cols = ['INCUMBENT', 'DV_BLEND', 'KNN_BLEND', 'DEPAVG', 'D']
    pt = {c: official_total(J.assign(prediction_kwh=J[c])[
        ['group_id', 'actual_kwh', 'prediction_kwh']])['total'] for c in cols}
    print('point estimates:', {k: round(v, 6) for k, v in pt.items()})

    print(f'\nrunning {B} block-bootstrap replicates (block = {BLOCK} days)...', flush=True)
    reps = bootstrap(J, cols)
    out = {'point': pt, 'block_days': BLOCK, 'n_rep': B, 'se': {}, 'paired': {}}
    print('\n--- marginal sampling distribution of the local Total ---')
    for c in cols:
        v = reps[c]
        out['se'][c] = {'mean': float(v.mean()), 'sd': float(v.std(ddof=1)),
                        'q025': float(np.quantile(v, 0.025)), 'q975': float(np.quantile(v, 0.975))}
        print(f'  {c:11s} mean={v.mean():.6f}  sd={v.std(ddof=1):.6f}  '
              f'95% CI [{np.quantile(v,0.025):.6f}, {np.quantile(v,0.975):.6f}]')

    print('\n--- PAIRED differences against the incumbent ---')
    for c in cols:
        if c == 'INCUMBENT':
            continue
        d = reps[c] - reps['INCUMBENT']
        out['paired'][c] = {'mean': float(d.mean()), 'sd': float(d.std(ddof=1)),
                            'p_better': float((d > 0).mean())}
        print(f'  {c:11s} delta={d.mean():+.6f}  sd={d.std(ddof=1):.6f}  '
              f'P(candidate > incumbent) = {(d>0).mean():.3f}')

    sd = out['se']['INCUMBENT']['sd']
    gap = 0.66 - pt['INCUMBENT']
    print(f'\n  local Total sd  = {sd:.6f}')
    print(f'  gap to 0.66     = {gap:.6f}  =  {gap/sd:.1f} bootstrap SD')
    out['gap_to_target'] = {'gap': gap, 'in_sd': gap / sd}
    json.dump(out, open(N + 'S12-N17_noise_floor.json', 'w'), indent=1, default=str)
