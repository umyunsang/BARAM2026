"""S12-N7 * dispersion recalibration of the blended action.

Mechanism.  current_best is 0.30*D + 0.70*DEPAVG, and DEPAVG is itself the mean of three
deployed actions -- so the deployed value is an average of four correlated actions.  Averaging
contracts dispersion.  Under a squared/absolute loss that contraction is exactly what you want,
but FICR is a step reward on |pred-actual| <= 0.06*cap: it does not reward being closer on
average, it rewards landing inside a fixed-width band.  A contracted predictor systematically
sits between the band around the low outcome and the band around the high outcome, hitting
neither.  Every S12 result so far has shown the same signature -- treatments that raise the
blend's 1-NMAE lower its FICR -- which is the fingerprint of over-contraction.

Treatment: rescale the deployed action about a fixed centre, a' = c + k*(a - c), and choose k
fold-outside (k on the other two folds, applied to the held-out fold).  1 fitted dof.
Centres tested: the group's training-period mean capacity factor (fixed, 0 dof) and the
fold-outside median of the action itself.  Also tested per-group k (3 dof) purely to show
whether the fold-outside gate rejects it, as it rejected every previous per-group weighting.
"""
import sys, json
import numpy as np, pandas as pd
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/nodes')
from loop_lib import *

N = '/Users/um-yunsang/BARAM2026/research/nodes/'
KS = np.arange(0.80, 1.4001, 0.02)


def build_blend():
    R = canonical_keys(); dep = load_depavg()
    Dm, solo, _ = fo_policy(utility_frames(align_prob('D', R), R), R)
    J = Dm[KEY + ['actual_kwh', 'prediction_kwh']].rename(columns={'prediction_kwh': 'D'}).merge(dep, on=KEY)
    rows = []
    for f in FOLDS:
        oth = J[J.fold_id != f]; held = J[J.fold_id == f]; best = None
        for w in np.arange(0, 1.001, 0.05):
            p = w * oth.D + (1 - w) * oth.DEPAVG
            t = official_total(oth.assign(prediction_kwh=p)[['group_id', 'actual_kwh', 'prediction_kwh']])['total']
            if best is None or t > best[0]:
                best = (t, w)
        rows.append(held.assign(BLEND=best[1] * held.D + (1 - best[1]) * held.DEPAVG))
    return pd.concat(rows, ignore_index=True)


def rescale(df, col, k, centres):
    cap = df.group_id.map(CAPS).to_numpy()
    c = df.group_id.map(centres).to_numpy() * cap
    hi = df.group_id.map({1: 0.985, 2: 0.989, 3: 1.005}).to_numpy() * cap
    return np.clip(c + k * (df[col].to_numpy() - c), 0.0, hi)


if __name__ == '__main__':
    B = build_blend()
    base = official_total(B.assign(prediction_kwh=B.BLEND)[['group_id', 'actual_kwh', 'prediction_kwh']])
    print(f'baseline blend {base["total"]:.6f} (1-NMAE={base["one_minus_nmae"]:.6f} FICR={base["ficr"]:.6f})')

    # dispersion diagnostic: sd(pred)/sd(actual) on scored rows, per group
    print('\n--- dispersion diagnostic (scored rows only) ---')
    cap = B.group_id.map(CAPS)
    V = B[B.actual_kwh >= 0.1 * cap]
    centres = {}
    for g in (1, 2, 3):
        s = V[V.group_id == g]
        sp = float((s.BLEND / CAPS[g]).std()); sa = float((s.actual_kwh / CAPS[g]).std())
        centres[g] = float((s.BLEND / CAPS[g]).mean())
        print(f'  g{g}: sd(pred)={sp:.4f} sd(actual)={sa:.4f} ratio={sp/sa:.4f} '
              f'corr={np.corrcoef(s.BLEND, s.actual_kwh)[0,1]:.4f} centre={centres[g]:.4f}')

    out = {'baseline': base['total'], 'dispersion': {}}

    print('\n--- in-sample k sweep (diagnostic only, NOT a claim) ---')
    for k in [0.9, 1.0, 1.05, 1.1, 1.15, 1.2, 1.3]:
        p = rescale(B, 'BLEND', k, centres)
        s = official_total(B.assign(prediction_kwh=p)[['group_id', 'actual_kwh', 'prediction_kwh']])
        print(f'  k={k:4.2f}  total={s["total"]:.6f}  1-NMAE={s["one_minus_nmae"]:.6f}  FICR={s["ficr"]:.6f}')
        out['dispersion'][f'insample_k{k}'] = s['total']

    print('\n--- fold-outside k (1 dof) ---')
    rows = []; picks = {}
    for f in FOLDS:
        oth = B[B.fold_id != f]; held = B[B.fold_id == f]; best = None
        cen = {g: float((oth[(oth.group_id == g) & (oth.actual_kwh >= 0.1 * oth.group_id.map(CAPS))].BLEND / CAPS[g]).mean())
               for g in (1, 2, 3)}
        for k in KS:
            p = rescale(oth, 'BLEND', k, cen)
            t = official_total(oth.assign(prediction_kwh=p)[['group_id', 'actual_kwh', 'prediction_kwh']])['total']
            if best is None or t > best[0]:
                best = (t, k)
        picks[f] = float(best[1])
        rows.append(held.assign(prediction_kwh=rescale(held, 'BLEND', best[1], cen)))
    Dd = pd.concat(rows, ignore_index=True)
    s = official_total(Dd[['group_id', 'actual_kwh', 'prediction_kwh']])
    print(f'  FO k picks={list(picks.values())}  total={s["total"]:.6f}  '
          f'd_vs_current_best={s["total"]-0.6361842493883538:+.6f}')
    out['foldoutside_k'] = {'total': s['total'], 'picks': picks,
                            'one_minus_nmae': s['one_minus_nmae'], 'ficr': s['ficr']}

    print('\n--- fold-outside per-group k (3 dof, expected to be rejected) ---')
    rows = []; picks3 = {}
    for f in FOLDS:
        oth = B[B.fold_id != f]; held = B[B.fold_id == f]
        cen = {g: float((oth[(oth.group_id == g) & (oth.actual_kwh >= 0.1 * oth.group_id.map(CAPS))].BLEND / CAPS[g]).mean())
               for g in (1, 2, 3)}
        kg = {}
        for g in (1, 2, 3):
            sub = oth[oth.group_id == g]; best = None
            for k in KS:
                p = rescale(sub, 'BLEND', k, cen)
                t = official_total(sub.assign(prediction_kwh=p)[['group_id', 'actual_kwh', 'prediction_kwh']])['total']
                if best is None or t > best[0]:
                    best = (t, k)
            kg[g] = float(best[1])
        picks3[f] = kg
        p = np.concatenate([rescale(held[held.group_id == g], 'BLEND', kg[g], cen) for g in (1, 2, 3)])
        hh = pd.concat([held[held.group_id == g] for g in (1, 2, 3)], ignore_index=True)
        rows.append(hh.assign(prediction_kwh=p))
    Dd3 = pd.concat(rows, ignore_index=True)
    s3 = official_total(Dd3[['group_id', 'actual_kwh', 'prediction_kwh']])
    print(f'  FO per-group k picks={picks3}  total={s3["total"]:.6f}  '
          f'd_vs_current_best={s3["total"]-0.6361842493883538:+.6f}')
    out['foldoutside_k_pergroup'] = {'total': s3['total'], 'picks': picks3}

    json.dump(out, open(N + 'S12-N7_dispersion.json', 'w'), indent=1, default=str)
