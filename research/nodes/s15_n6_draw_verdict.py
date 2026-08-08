"""S15-N6 * the verdict: where does `current_best` sit in its own configuration's seed law?

S15-N5 refitted the DART classifier under six sibling seeds with the physics teacher held at the
champion's own seed (20260801), i.e. inside exactly the configuration family the champion belongs
to.  This node pushes each through the identical champion pipeline -- fold-outside (T,G), then the
1-dof fold-outside blend against DEPAVG -- and reports the champion's position in that law.
"""
import sys, json
import numpy as np, pandas as pd
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/scratch')
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/nodes')
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/engine')
from lib import CAPS, FOLDS, official_total
from loop_lib import canonical_keys, align_prob, load_depavg, utility_frames, fo_policy, KEY, W
from arbiter import arbitrate

N = '/Users/um-yunsang/BARAM2026/research/nodes/'
SEEDS = (20260804, 20260805, 20260806, 20260807, 20260808, 20260809)


def pipeline(P, R, dep):
    Dm, solo, _ = fo_policy(utility_frames(P, R), R)
    J = Dm[KEY + ['actual_kwh', 'prediction_kwh']].rename(columns={'prediction_kwh': 'M'}).merge(dep, on=KEY)
    rows = []
    for f in FOLDS:
        oth = J[J.fold_id != f]; held = J[J.fold_id == f]; best = None
        for w in np.arange(0, 1.001, 0.05):
            t = official_total(oth.assign(prediction_kwh=w * oth.M + (1 - w) * oth.DEPAVG)[
                ['group_id', 'actual_kwh', 'prediction_kwh']])['total']
            if best is None or t > best[0]:
                best = (t, w)
        rows.append(held.assign(pred=best[1] * held.M + (1 - best[1]) * held.DEPAVG, w=best[1]))
    B = pd.concat(rows, ignore_index=True)
    s = official_total(B.assign(prediction_kwh=B.pred)[['group_id', 'actual_kwh', 'prediction_kwh']])
    return B, solo['total'], s


if __name__ == '__main__':
    R = canonical_keys(); dep = load_depavg()
    Rk = pd.read_parquet(N + 'S15-N5_keys.parquet')
    assert Rk[KEY].equals(R[KEY])
    print('=== champion configuration family: teacher seed 20260801 fixed, DART seed varied ===')
    rec = []
    for s in SEEDS:
        P = np.load(N + f'S15-N5_prob_{s}.npy')
        B, solo, sc = pipeline(P, R, dep)
        rec.append(dict(seed=s, solo=solo, blend=sc['total'],
                        one_minus_nmae=sc['one_minus_nmae'], ficr=sc['ficr'],
                        w=float(B.w.iloc[0])))
        print(f'  dart seed {s}: solo={solo:.6f}  blend={sc["total"]:.6f}  '
              f'(1-NMAE={sc["one_minus_nmae"]:.6f} FICR={sc["ficr"]:.6f})')
    Bc, solo_c, sc_c = pipeline(align_prob('D', R), R, dep)
    print(f'\n  CHAMPION  (dart seed 20260803): solo={solo_c:.6f}  blend={sc_c["total"]:.6f}')

    b = np.array([r['blend'] for r in rec]); so = np.array([r['solo'] for r in rec])
    z = (sc_c['total'] - b.mean()) / b.std(ddof=1)
    print(f'\n=== the champion inside its own seed law ===')
    print(f'  sibling blends: mean={b.mean():.6f}  sd={b.std(ddof=1):.6f}  '
          f'min={b.min():.6f}  max={b.max():.6f}')
    print(f'  champion       = {sc_c["total"]:.6f}')
    print(f'  z-score        = {z:+.2f} sd above the mean of its own configuration')
    print(f'  siblings above the champion: {int((b >= sc_c["total"]).sum())} of {len(b)}')
    dep_only = official_total(Bc.assign(prediction_kwh=Bc.DEPAVG)[
        ['group_id', 'actual_kwh', 'prediction_kwh']])['total']
    print(f'\n  plain DEPAVG                       = {dep_only:.6f}')
    print(f'  champion advantage over DEPAVG     = {sc_c["total"]-dep_only:+.6f}')
    print(f'  seed-mean advantage over DEPAVG    = {b.mean()-dep_only:+.6f}')
    print(f'  => {100*(1-(b.mean()-dep_only)/(sc_c["total"]-dep_only)):.0f}% of the champion\'s '
          f'measured advantage does not survive a seed redraw')
    print(f'\n  E[max of {len(b)+1} draws] under normality = mean + {1.27:.2f} sd = '
          f'{b.mean()+1.27*b.std(ddof=1):.6f}  (champion is {sc_c["total"]:.6f})')

    # what an honest seed-averaged member gives
    Pavg = np.mean([np.load(N + f'S15-N5_prob_{s}.npy') for s in SEEDS] +
                   [align_prob('D', R)], axis=0)
    Ba, solo_a, sc_a = pipeline(Pavg, R, dep)
    print(f'\n  7-seed AVERAGED member (the honest estimator): solo={solo_a:.6f}  '
          f'blend={sc_a["total"]:.6f}')
    cmp = Bc[KEY + ['actual_kwh']].copy(); cmp['champ'] = Bc.pred.to_numpy()
    cmp = cmp.merge(Ba[KEY + ['pred']].rename(columns={'pred': 'cand'}), on=KEY)
    took, arb = arbitrate(cmp, 'cand', 'champ', n_comparisons=1)
    print(f'  ARBITER seed-avg vs champion: delta={arb["point_delta"]:+.6f} '
          f'sd={arb["paired_sd"]:.6f} P={arb["p_better"]:.3f} -> '
          f'{"CHAMPION" if took else "rejected"}')
    json.dump({'siblings': rec, 'champion': sc_c['total'], 'z': float(z),
               'depavg': dep_only, 'seed_mean': float(b.mean()), 'seed_sd': float(b.std(ddof=1)),
               'seed_avg_blend': sc_a['total'], 'arb': arb},
              open(N + 'S15-N6_draw_verdict.json', 'w'), indent=1, default=str)
