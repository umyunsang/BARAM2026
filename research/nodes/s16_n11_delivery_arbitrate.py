"""S16-N11 * connect the DELIVERED lineage to the evaluation apparatus, at last.

The F3 audit surfaced the largest unquantified risk in this project: for six sessions the harness
has been arbitrating candidates that differ by fractions of the seed floor, while the artifact that
would actually be submitted -- submission_M268.csv, the M231/M267 sequence-transfer lineage -- has
`local_score: null` in every receipt and has never been scored by it.  The deployed stems
M102/M113/M115 exist only as dev-2023 backtests with no 2025 predictions, so the local champion
0.30*D + 0.70*DEPAVG has never been materialised on the graded year either.  We optimised one
object and would deliver another.

The connecting artifact turns out to exist: M267_FIXED_M231_SEQUENCE_TRANSFER-oof.parquet carries
19,785 out-of-fold rows on exactly our canonical key set, which is the same lineage M268 was built
from.  So the comparison that the F3 document called the only remaining high-value measurement can
be made now, with no fit at all.

What is being decided: whether to keep delivering the M268 lineage or to rebuild delivery around
the local champion.  The prior from F18 is to keep M268 -- the champion's +0.003592 over plain
DEPAVG is inside a 10% Model Confidence Set containing eight models, roughly half of it evaporates
under a seed redraw, and it does not exist on 2025 -- but that prior was formed WITHOUT knowing
where M268 actually scores.  This node supplies the missing number.
"""
import sys, json
import numpy as np, pandas as pd
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/scratch')
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/nodes')
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/engine')
from lib import CAPS, FOLDS, official_total
from loop_lib import canonical_keys, align_prob, load_depavg, utility_frames, fo_policy, KEY
from arbiter import arbitrate

AB = '/Users/um-yunsang/BARAM2026/artifacts/backtests/metric-aligned-probe/'
N = '/Users/um-yunsang/BARAM2026/research/nodes/'

if __name__ == '__main__':
    R = canonical_keys(); dep = load_depavg()
    Dm, solo_D, _ = fo_policy(utility_frames(align_prob('D', R), R), R)
    J = Dm[KEY + ['actual_kwh', 'prediction_kwh']].rename(columns={'prediction_kwh': 'D'}).merge(dep, on=KEY)
    J['CHAMPION'] = 0.30 * J.D + 0.70 * J.DEPAVG
    o = pd.read_parquet(AB + 'M267_FIXED_M231_SEQUENCE_TRANSFER-oof.parquet')
    J = J.merge(o[KEY + ['prediction_kwh']].rename(columns={'prediction_kwh': 'DELIVERED'}), on=KEY)
    assert len(J) == 19785, len(J)
    base = J[['group_id', 'actual_kwh']]
    cap = J.group_id.map(CAPS).to_numpy(); y = J.actual_kwh.to_numpy()
    sc = y >= 0.10 * cap

    print('=== the three objects, on identical rows and the identical metric ===')
    rows = {}
    for nm in ('DELIVERED', 'CHAMPION', 'DEPAVG', 'D'):
        s = official_total(base.assign(prediction_kwh=J[nm]))
        e = np.abs(J[nm].to_numpy() - y) / cap
        rows[nm] = dict(total=s['total'], nmae=s['one_minus_nmae'], ficr=s['ficr'],
                        u4=float((e <= 0.06)[sc].mean()))
        print(f'  {nm:10s} Total={s["total"]:.6f}  1-NMAE={s["one_minus_nmae"]:.6f}  '
              f'FICR={s["ficr"]:.6f}  u=4 rate={(e <= 0.06)[sc].mean():.4f}')

    print('\n=== paired arbitration, moving-block bootstrap ===')
    for cand, champ in (('DELIVERED', 'CHAMPION'), ('DELIVERED', 'DEPAVG')):
        cmp = J[KEY + ['actual_kwh']].copy()
        cmp['cand'] = J[cand]; cmp['champ'] = J[champ]
        took, arb = arbitrate(cmp, 'cand', 'champ', n_comparisons=1)
        print(f'  {cand} vs {champ}: delta={arb["point_delta"]:+.6f}  '
              f'paired sd={arb["paired_sd"]:.6f}  P(better)={arb["p_better"]:.3f}  '
              f'-> {"DELIVERED WINS" if took else "not distinguishable / worse"}')
        rows[f'{cand}_vs_{champ}'] = arb

    print('\n=== per-fold, since the delivery lineage was built for full history ===')
    for f in FOLDS:
        sub = J[J.fold_id == f]
        b = sub[['group_id', 'actual_kwh']]
        print(f'  {f}: DELIVERED={official_total(b.assign(prediction_kwh=sub.DELIVERED))["total"]:.6f}  '
              f'CHAMPION={official_total(b.assign(prediction_kwh=sub.CHAMPION))["total"]:.6f}  '
              f'DEPAVG={official_total(b.assign(prediction_kwh=sub.DEPAVG))["total"]:.6f}')

    rI = (J.DELIVERED.to_numpy() - y) / cap
    rC = (J.CHAMPION.to_numpy() - y) / cap
    print(f'\n  error correlation DELIVERED vs CHAMPION = {np.corrcoef(rI, rC)[0,1]:.4f}')
    print(f'  band-hit correlation                     = '
          f'{np.corrcoef((np.abs(rI)<=0.06)[sc].astype(float), (np.abs(rC)<=0.06)[sc].astype(float))[0,1]:.4f}')
    bl = []
    for f in FOLDS:
        oth = J[J.fold_id != f]; held = J[J.fold_id == f]; best = None
        for w in np.arange(0, 1.001, 0.05):
            t = official_total(oth.assign(prediction_kwh=w * oth.DELIVERED + (1 - w) * oth.CHAMPION)[
                ['group_id', 'actual_kwh', 'prediction_kwh']])['total']
            if best is None or t > best[0]:
                best = (t, w)
        bl.append(held.assign(pred=best[1] * held.DELIVERED + (1 - best[1]) * held.CHAMPION, w=best[1]))
    B = pd.concat(bl, ignore_index=True)
    sb = official_total(B.assign(prediction_kwh=B.pred)[['group_id', 'actual_kwh', 'prediction_kwh']])
    print(f'  fold-outside blend of the two: {sb["total"]:.6f}  (w on DELIVERED = '
          f'{sorted(set(B.w.round(2)))})')
    rows['blend'] = sb['total']
    json.dump(rows, open(N + 'S16-N11_delivery.json', 'w'), indent=1, default=str)
