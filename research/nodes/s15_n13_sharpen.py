"""S15-N13 * the composed density is flat, and the temperature grid stops before it can be fixed.

S15-N12 settled the diagnosis. The composed member is BETTER than the champion's D on every
accuracy measure -- point 1-NMAE 0.865320 vs 0.862828 (+0.002492) and solo 0.627190 vs 0.625669
(+0.001522) -- and much FLATTER: mass inside the +-0.06 settlement window around its own mode is
0.567 against D's 0.759, mean sd 0.1198 against 0.0767, entropy 1.941 against 1.404.  Flatness is
why it blends worse: a flat member resembles DEPAVG (itself an average of actions) and stops being
complementary.

The decision layer already owns the instrument that fixes this -- q^(1/T) with T < 1 sharpens --
but its grid is TEMPS = [0.6, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0], i.e. 0.6 is the floor.  D selects
1.0 and never needed more; a density with 1.9 nats of entropy does.  The registry already records
this exact move once before: "extend the T/gamma policy grid (optimum was on the corner)",
accepted at foldout_delta +0.003205 in S7.

Treatment: extend TEMPS downward to 0.15 and re-run the identical fold-outside gate.  This adds no
degrees of freedom -- T is already a fold-outside-selected parameter -- it only stops truncating
its range.  Both the composed member and D are run on the extended grid so the comparison is fair
and so we can see whether the champion was also sitting on a boundary.
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
SEEDS = (20260803, 20260804, 20260805)
WIDE_T = [0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0]
GAM = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0]


def run(P, R, dep, temps, tag):
    fr = utility_frames(P, R, temps=temps, gammas=GAM)
    Dm, solo, picks = fo_policy(fr, R)
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
    print(f'  {tag:34s} solo={solo["total"]:.6f}  blend={s["total"]:.6f}  '
          f'(1-NMAE={s["one_minus_nmae"]:.6f} FICR={s["ficr"]:.6f})  '
          f'T,G picks={[str(v) for v in picks.values()]}  w={[round(float(x),2) for x in B.groupby("fold_id").w.first()]}')
    return B, solo['total'], s


if __name__ == '__main__':
    R = canonical_keys(); dep = load_depavg()
    PD = align_prob('D', R)
    PC = {s: np.load(N + f'S15-N9_prob_{s}.npy') for s in SEEDS}
    PCavg = np.mean(list(PC.values()), axis=0)
    NARROW = [0.6, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]

    print('=== narrow temperature grid (as deployed) ===')
    Bd_n, sd_n, sc_d_n = run(PD, R, dep, NARROW, 'D (champion)')
    Bc_n, sc_n, sc_c_n = run(PCavg, R, dep, NARROW, 'COMPOSED seed-avg')
    print('\n=== extended temperature grid, floor 0.6 -> 0.15 ===')
    Bd_w, sd_w, sc_d_w = run(PD, R, dep, WIDE_T, 'D (champion)')
    Bc_w, sc_w, sc_c_w = run(PCavg, R, dep, WIDE_T, 'COMPOSED seed-avg')
    per = []
    for s in SEEDS:
        _, _, sc_i = run(PC[s], R, dep, WIDE_T, f'COMPOSED seed {s}')
        per.append(sc_i['total'])
    per = np.array(per)

    print(f'\n=== verdict ===')
    print(f'  champion on the extended grid : {sc_d_w["total"]:.6f} '
          f'(was {sc_d_n["total"]:.6f}; the grid floor was NOT binding for D)')
    print(f'  composed on the extended grid : {sc_c_w["total"]:.6f} '
          f'(was {sc_c_n["total"]:.6f})')
    print(f'  composed per-seed mean        : {per.mean():.6f}  sd={per.std(ddof=1):.6f}')
    print(f'  honest champion baseline      : 0.634573  sd 0.000849')
    print(f'  composed mean - honest champ  : {per.mean()-0.634573:+.6f} '
          f'({(per.mean()-0.634573)/0.000849:+.1f} seed-sd)')
    cmp = Bd_n[KEY + ['actual_kwh']].copy(); cmp['champ'] = Bd_n.pred.to_numpy()
    cmp = cmp.merge(Bc_w[KEY + ['pred']].rename(columns={'pred': 'cand'}), on=KEY)
    took, arb = arbitrate(cmp, 'cand', 'champ', n_comparisons=1)
    print(f'  ARBITER (composed+wide-T vs deployed champion): delta={arb["point_delta"]:+.6f} '
          f'sd={arb["paired_sd"]:.6f} P={arb["p_better"]:.3f} -> {"CHAMPION" if took else "rejected"}')
    json.dump({'champion_narrow': sc_d_n['total'], 'champion_wide': sc_d_w['total'],
               'composed_narrow': sc_c_n['total'], 'composed_wide': sc_c_w['total'],
               'composed_per_seed_wide': per.tolist(), 'arb': arb},
              open(N + 'S15-N13_sharpen.json', 'w'), indent=1, default=str)
