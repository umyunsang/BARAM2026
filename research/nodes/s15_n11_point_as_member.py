"""S15-N11 * use the composed pipeline where it is strong, without touching the champion's density.

What we have.  The composed pipeline (B1 + B2 + A3 + D1, D4 pruned by reverse ablation) is the
most ACCURATE point forecaster this project has produced: 3-seed mean 1-NMAE 0.866484 against the
champion blend's 0.861866, and Total 0.605922 on the point axis (+0.003623 over base).  What it is
not is a good ACTION: its FICR is 0.344 because a point forecast does not seek settlement bands.

What we learned, eight times over.  Transplanting accuracy treatments INTO the member destroys
them -- S15-N9 (all four stages, -0.002546) and S15-N10 (the same without the density regulariser,
-0.003020) both landed well below the honest champion baseline of 0.634573, because the champion's
teacher, its top-150 selection and its DART classifier are co-adapted and any upstream change
perturbs the feature set the classifier was built on.

So this node stops transplanting and starts COMPOSING AT THE ACTION LEVEL: the composed point
forecast enters as a THIRD member alongside D and DEPAVG, where its accuracy can pull 1-NMAE up
while the existing members carry FICR.  The champion's density is untouched.  Weights are chosen
fold-outside; the 2-dof simplex is reported next to the 1-dof addition so the degrees of freedom
are explicit, and the honest seed-mean champion is the comparator.
"""
import sys, json
import numpy as np, pandas as pd
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/scratch')
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/nodes')
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/engine')
from lib import CAPS, FOLDS, official_total
from loop_lib import canonical_keys, load_depavg, align_prob, utility_frames, fo_policy, KEY
from arbiter import arbitrate

N = '/Users/um-yunsang/BARAM2026/research/nodes/'
SEEDS = (20260801, 20260802, 20260803)
ARMS = ['FULL', 'no_B1', 'no_B2', 'no_A3', 'no_D4', 'no_D1']


def fo_simplex(J, cols, step=0.05):
    grid = []

    def rec(pre, rem, left):
        if rem == 1:
            grid.append(tuple(pre + [round(left, 4)])); return
        w = 0.0
        while w <= left + 1e-9:
            rec(pre + [round(w, 4)], rem - 1, left - w); w += step
    rec([], len(cols), 1.0)
    rows = []; picks = {}
    for f in FOLDS:
        oth = J[J.fold_id != f]; held = J[J.fold_id == f]
        Ao = oth[cols].to_numpy(); base = oth[['group_id', 'actual_kwh']]
        best = None
        for wv in grid:
            t = official_total(base.assign(prediction_kwh=Ao @ np.array(wv)))['total']
            if best is None or t > best[0]:
                best = (t, wv)
        picks[f] = best[1]
        rows.append(held.assign(pred=held[cols].to_numpy() @ np.array(best[1])))
    B = pd.concat(rows, ignore_index=True)
    return B, official_total(B.assign(prediction_kwh=B.pred)[
        ['group_id', 'actual_kwh', 'prediction_kwh']]), picks


if __name__ == '__main__':
    R = canonical_keys(); dep = load_depavg()
    Dm, _, _ = fo_policy(utility_frames(align_prob('D', R), R), R)
    J = Dm[KEY + ['actual_kwh', 'prediction_kwh']].rename(columns={'prediction_kwh': 'D'}).merge(dep, on=KEY)

    # S15-N8 persisted only its json, so the composed predictions come from S15-N7, whose
    # COMPOSED arm still carries D4.  Reverse ablation put D4 at -0.000563, so this arm is
    # 0.605359 against the pruned 0.605922 -- a conservative stand-in, not a favourable one.
    K = pd.read_parquet(N + 'S15-N7_keys.parquet')
    P = np.load(N + 'S15-N7_preds.npy')
    order = sorted([f'{a}_s{s}' for a in ('BASE', 'COMPOSED') for s in SEEDS])
    pos = {k: i for i, k in enumerate(order)}
    comp = np.mean([P[pos[f'COMPOSED_s{s}']] for s in SEEDS], axis=0)
    Kc = K.copy(); Kc['COMP'] = comp * Kc.group_id.map(CAPS).to_numpy()
    J = J.merge(Kc[KEY + ['COMP']], on=KEY)
    base = J[['group_id', 'actual_kwh']]
    J['CHAMP'] = 0.30 * J.D + 0.70 * J.DEPAVG
    for c in ('D', 'DEPAVG', 'COMP', 'CHAMP'):
        s = official_total(base.assign(prediction_kwh=J[c]))
        print(f'  {c:7s} Total={s["total"]:.6f}  1-NMAE={s["one_minus_nmae"]:.6f}  FICR={s["ficr"]:.6f}')

    cap = J.group_id.map(CAPS)
    E = pd.DataFrame({c: (J[c] - J.actual_kwh) / cap for c in ('D', 'DEPAVG', 'COMP')})
    print('\n  error correlation (the composed point member is the new information):')
    print(E.corr().round(4).to_string())

    print('\n=== fold-outside blends, honest champion comparator = 0.634573 (sd 0.000849) ===')
    out = {}
    for nm, cols in [('D+DEPAVG (champion, 1 dof)', ['D', 'DEPAVG']),
                     ('COMP+DEPAVG (1 dof)', ['COMP', 'DEPAVG']),
                     ('D+DEPAVG+COMP (2 dof)', ['D', 'DEPAVG', 'COMP'])]:
        B, s, picks = fo_simplex(J, cols)
        out[nm] = {'total': s['total'], 'one_minus_nmae': s['one_minus_nmae'], 'ficr': s['ficr'],
                   'picks': {k: list(v) for k, v in picks.items()}}
        print(f'  {nm:28s} Total={s["total"]:.6f}  1-NMAE={s["one_minus_nmae"]:.6f}  '
              f'FICR={s["ficr"]:.6f}  picks={[list(v) for v in picks.values()]}')
        if 'COMP' in cols and len(cols) == 3:
            cmp = J[KEY + ['actual_kwh']].copy(); cmp['champ'] = J.CHAMP
            cmp = cmp.merge(B[KEY + ['pred']].rename(columns={'pred': 'cand'}), on=KEY)
            took, arb = arbitrate(cmp, 'cand', 'champ', n_comparisons=1)
            print(f'    ARBITER vs DEPLOYED champion: delta={arb["point_delta"]:+.6f} '
                  f'sd={arb["paired_sd"]:.6f} P={arb["p_better"]:.3f} -> '
                  f'{"CHAMPION" if took else "rejected"}')
            print(f'    vs HONEST champion mean 0.634573: {s["total"]-0.634573:+.6f} '
                  f'({(s["total"]-0.634573)/0.000849:+.1f} seed-sd)')
            out[nm]['arb'] = arb
    json.dump(out, open(N + 'S15-N11_point_as_member.json', 'w'), indent=1, default=str)
