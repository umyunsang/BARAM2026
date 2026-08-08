"""S14-N9 (engine node F17) * James-Stein shrinkage of the per-group parameters.

The exact defect this treats.  research/nodes/registry.json records that fold-outside per-group
blend weights OSCILLATE -- `g3: 1.00 / 1.00 / 0.15` across the three folds -- and that the 3-dof
per-group blend was rejected because three folds cannot estimate three free weights (in-sample
0.640253 collapsing to 0.635453 fold-outside, below the 1-dof uniform 0.639170 on that surface).
That is the textbook symptom James-Stein addresses, and p = 3 groups is exactly where the theorem
starts to bite.

The correction is NOT another per-group fit.  It is one shrinkage intensity applied to all three:

    w_g^JS = w_0 + (1 - c) * (w_g - w_0)

with w_0 the pooled estimate.  Two variants are gated:
  JS_ANALYTIC   c = (p-2) * s2 / sum_g (w_g - w_0)^2 , clipped to [0,1] -- ZERO fitted dof, the
                shrinkage factor is computed from the spread of the group estimates themselves
  JS_SWEPT      c chosen fold-outside on a grid -- ONE fitted dof
Both are compared against the deployed 1-dof pooled weight and against the already-rejected
3-dof free per-group weights, so the dof ladder is explicit.  The same treatment is then applied
to the per-group soft caps SC = {0.985, 0.989, 1.005}, the only other per-group constants the
champion carries.
"""
import sys, json
import numpy as np, pandas as pd
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/scratch')
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/nodes')
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/engine')
from lib import CAPS, FOLDS, official_total
from loop_lib import canonical_keys, align_prob, load_depavg, utility_frames, fo_policy, KEY
from arbiter import arbitrate

N = '/Users/um-yunsang/BARAM2026/research/nodes/'
WGRID = np.arange(0.0, 1.001, 0.05)
CGRID = np.arange(0.0, 1.001, 0.1)


def group_score(d, pred, g):
    """Single-group score. official_total averages over three groups and returns NaN when only
    one is present, which silently made every per-group search return the first grid point -- the
    same defect that produced the meaningless all-0.8 per-group arm in S12-N7."""
    cap = CAPS[g]
    m = d.actual_kwh.to_numpy() >= 0.1 * cap
    if m.sum() < 30:
        return -1.0
    a = d.actual_kwh.to_numpy()[m]
    e = np.abs(pred[m] - a) / cap
    u = np.select([e <= 0.06, e <= 0.08], [4.0, 3.0], 0.0)
    return 0.5 * (1.0 - float(e.mean())) + 0.5 * float((a * u).sum() / (a * 4.0).sum())


def best_w(df, gsel=None):
    if gsel is None:
        best = None
        for w in WGRID:
            t = official_total(df.assign(prediction_kwh=w * df.D + (1 - w) * df.DEPAVG)[
                ['group_id', 'actual_kwh', 'prediction_kwh']])['total']
            if best is None or t > best[0]:
                best = (t, float(w))
        return best[1]
    d = df[df.group_id == gsel]
    best = None
    for w in WGRID:
        p = (w * d.D + (1 - w) * d.DEPAVG).to_numpy()
        t = group_score(d, p, gsel)
        if best is None or t > best[0]:
            best = (t, float(w))
    return best[1]


if __name__ == '__main__':
    R = canonical_keys(); dep = load_depavg()
    Dm, _, _ = fo_policy(utility_frames(align_prob('D', R), R), R)
    J = Dm[KEY + ['actual_kwh', 'prediction_kwh']].rename(columns={'prediction_kwh': 'D'}).merge(dep, on=KEY)
    base = J[['group_id', 'actual_kwh']]

    rows = {k: np.empty(len(J)) for k in ('POOLED', 'FREE3', 'JS_ANALYTIC', 'JS_SWEPT')}
    diag = {}
    for f in FOLDS:
        sel = (J.fold_id == f).to_numpy()
        oth, held = J[~sel], J[sel]
        w0 = best_w(oth)
        wg = {g: best_w(oth, g) for g in (1, 2, 3)}
        v = np.array([wg[g] for g in (1, 2, 3)])
        ss = float(((v - w0) ** 2).sum())
        s2 = float(np.var(v, ddof=1)) / 3.0
        c_an = float(np.clip((3 - 2) * s2 / max(ss, 1e-9), 0.0, 1.0))
        diag[f] = {'w_pooled': w0, 'w_group': wg, 'spread_ss': ss, 'c_analytic': c_an}
        print(f'  {f}: pooled w={w0:.2f}  per-group={wg}  c_analytic={c_an:.3f}')

        def apply(wmap):
            wv = held.group_id.map(wmap).to_numpy()
            return wv * held.D.to_numpy() + (1 - wv) * held.DEPAVG.to_numpy()
        rows['POOLED'][sel] = apply({g: w0 for g in (1, 2, 3)})
        rows['FREE3'][sel] = apply(wg)
        rows['JS_ANALYTIC'][sel] = apply({g: w0 + (1 - c_an) * (wg[g] - w0) for g in (1, 2, 3)})
        # swept c: choose c on the other two folds
        bestc = None
        for c in CGRID:
            acc = []
            for f2 in FOLDS:
                if f2 == f:
                    continue
                s2m = (J.fold_id == f2).to_numpy()
                o2 = J[~s2m & ~sel]
                if len(o2) == 0:
                    continue
                w02 = best_w(o2); wg2 = {g: best_w(o2, g) for g in (1, 2, 3)}
                wm = {g: w02 + (1 - c) * (wg2[g] - w02) for g in (1, 2, 3)}
                h2 = J[s2m]
                wv = h2.group_id.map(wm).to_numpy()
                acc.append(official_total(h2.assign(
                    prediction_kwh=wv * h2.D + (1 - wv) * h2.DEPAVG)[
                    ['group_id', 'actual_kwh', 'prediction_kwh']])['total'])
            m = float(np.mean(acc)) if acc else -1
            if bestc is None or m > bestc[0]:
                bestc = (m, float(c))
        diag[f]['c_swept'] = bestc[1]
        rows['JS_SWEPT'][sel] = apply({g: w0 + (1 - bestc[1]) * (wg[g] - w0) for g in (1, 2, 3)})

    print('\n--- fold-outside results (dof ladder explicit) ---')
    out = {}
    for k in ('POOLED', 'FREE3', 'JS_ANALYTIC', 'JS_SWEPT'):
        s = official_total(base.assign(prediction_kwh=rows[k]))
        out[k] = {'total': s['total'], 'one_minus_nmae': s['one_minus_nmae'], 'ficr': s['ficr']}
        dof = {'POOLED': 1, 'FREE3': 3, 'JS_ANALYTIC': 1, 'JS_SWEPT': 2}[k]
        print(f'  {k:12s} dof={dof}  Total={s["total"]:.6f}  1-NMAE={s["one_minus_nmae"]:.6f}  '
              f'FICR={s["ficr"]:.6f}')
    print(f'\n  champion (deployed pooled w=0.30) = 0.636184')
    cmp = J[KEY + ['actual_kwh']].copy()
    cmp['champ'] = 0.30 * J.D + 0.70 * J.DEPAVG
    for k in ('JS_ANALYTIC', 'JS_SWEPT', 'FREE3'):
        cmp['cand'] = rows[k]
        took, arb = arbitrate(cmp, 'cand', 'champ', n_comparisons=1)
        print(f'  ARBITER {k:12s} delta={arb["point_delta"]:+.6f} sd={arb["paired_sd"]:.6f} '
              f'P={arb["p_better"]:.3f} -> {"CHAMPION" if took else "rejected"}')
        out[k]['arbitration'] = arb
    json.dump({'diag': diag, 'scores': out}, open(N + 'S14-N9_james_stein.json', 'w'),
              indent=1, default=str)
