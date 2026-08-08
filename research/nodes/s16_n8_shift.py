"""S16-N8 * break the trade by construction: D's SHAPE at the composed pipeline's LOCATION.

S16-N7 sampled both ends of the barrier and found the trade monotone:

    accuracy-targeted GBDT members   sharp (band mass 0.759) but correlated (0.92-0.99)
    IDR / EasyUQ                     decorrelated (0.7818, our best ever) but flat (0.516)

The decorrelation came precisely FROM discarding conditioning information, and discarding it
flattens the density.  So no single estimator gives both.  But the two properties live in
different parts of a predictive distribution -- sharpness is its SHAPE, accuracy is its LOCATION --
and nothing forces one estimator to supply both.

This node takes the sharp shape from D and moves it to the more accurate location the composed
pipeline found.  The composed member holds this project's best point accuracy (1-NMAE 0.867079
against the deployed lineage's 0.858) and lost the blend only by being flat (0.567).  D holds the
sharpest density we have (0.759) and is the champion's working member.  Shifting D's PMF along the
capacity-factor axis by lambda * (composed_mean - D_mean) preserves the shape EXACTLY -- band mass
and entropy are invariant to a translation -- while importing the location.

lambda = 0 recovers D unchanged and is the plumbing control; a custom transform that silently
does nothing has already cost this project two nodes, so the control must reproduce D's solo score
to the digit.  The shift is done on the 26-bin grid by fractional-index interpolation, mass is
renormalised, and the existing fold-outside decision layer and blend are unchanged.
"""
import sys, json
import numpy as np, pandas as pd
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/scratch')
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/nodes')
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/engine')
from lib import CAPS, FOLDS, official_total
from loop_lib import (canonical_keys, align_prob, load_depavg, utility_frames, fo_policy,
                      KEY, W, DEP, AB)
from arbiter import arbitrate

N = '/Users/um-yunsang/BARAM2026/research/nodes/'
NC = 26
SEEDS = (20260803, 20260804, 20260805)
C = (np.arange(NC) + 0.5) * W


def shift_pmf(P, d):
    """Translate each row's PMF by d (in capacity-factor units) on the bin grid, linearly
    splitting mass between the two neighbouring bins.  A translation leaves the shape -- hence
    band mass and entropy -- invariant except at the clipped edges."""
    s = d / W
    f = np.floor(s).astype(int); r = (s - f)[:, None]
    out = np.zeros_like(P)
    idx = np.arange(NC)[None, :]
    for shift, wgt in ((f, 1 - r), (f + 1, r)):
        t = idx + shift[:, None]
        np.clip(t, 0, NC - 1, out=t)
        np.add.at(out, (np.arange(len(P))[:, None], t), P * wgt)
    return out / out.sum(1, keepdims=True)


def sharp(P):
    md = C[np.argmax(P, 1)]
    return float((P * (np.abs(C[None, :] - md[:, None]) <= 0.06)).sum(1).mean()), \
           float((-P * np.log(P + 1e-12)).sum(1).mean())


if __name__ == '__main__':
    R = canonical_keys(); dep = load_depavg()
    PD = align_prob('D', R)
    PCmp = np.mean([np.load(N + f'S15-N9_prob_{s}.npy') for s in SEEDS], axis=0)
    mD = (PD * C[None, :]).sum(1); mC = (PCmp * C[None, :]).sum(1)
    Dm, solo_D, _ = fo_policy(utility_frames(PD, R), R)
    J = Dm[KEY + ['actual_kwh', 'prediction_kwh']].rename(columns={'prediction_kwh': 'D'}).merge(dep, on=KEY)
    J['CHAMPION'] = 0.30 * J.D + 0.70 * J.DEPAVG
    base = J[['group_id', 'actual_kwh']]
    champ = official_total(base.assign(prediction_kwh=J.CHAMPION))['total']
    bmD, enD = sharp(PD)
    print(f'D solo {solo_D["total"]:.6f}  shape {bmD:.3f}/{enD:.3f}   champion {champ:.6f}')
    print(f'mean |composed - D| location difference = {np.abs(mC-mD).mean():.4f} cf\n')

    res = {}
    for lam in (0.0, 0.25, 0.5, 0.75, 1.0):
        P = shift_pmf(PD, lam * (mC - mD))
        bm, en = sharp(P)
        M, solo, _ = fo_policy(utility_frames(P, R), R)
        if lam == 0.0:
            ok = abs(solo['total'] - solo_D['total']) < 1e-9
            print(f'  PLUMBING CONTROL lam=0 reproduces D solo: {ok} '
                  f'({solo["total"]:.9f} vs {solo_D["total"]:.9f})')
            if not ok:
                raise SystemExit('control failed - the shift is not identity at lam=0')
        Jm = J.merge(M[KEY + ['prediction_kwh']].rename(columns={'prediction_kwh': 'S'}), on=KEY)
        capJ = Jm.group_id.map(CAPS).to_numpy(); yJ = Jm.actual_kwh.to_numpy()
        rS = (Jm.S.to_numpy() - yJ) / capJ; rD = (Jm.D.to_numpy() - yJ) / capJ
        rP = (Jm.DEPAVG.to_numpy() - yJ) / capJ
        sc = yJ >= 0.10 * capJ
        rows = []
        for f in FOLDS:
            oth = Jm[Jm.fold_id != f]; held = Jm[Jm.fold_id == f]; best = None
            for w in np.arange(0, 1.001, 0.05):
                t = official_total(oth.assign(prediction_kwh=w * oth.S + (1 - w) * oth.DEPAVG)[
                    ['group_id', 'actual_kwh', 'prediction_kwh']])['total']
                if best is None or t > best[0]:
                    best = (t, w)
            rows.append(held.assign(pred=best[1] * held.S + (1 - best[1]) * held.DEPAVG, w=best[1]))
        B = pd.concat(rows, ignore_index=True)
        sb = official_total(B.assign(prediction_kwh=B.pred)[['group_id', 'actual_kwh', 'prediction_kwh']])
        print(f'  lam={lam:.2f}: solo {solo["total"]:.6f}  shape {bm:.3f}/{en:.3f}  '
              f'u=4 {(np.abs(rS)<=0.06)[sc].mean():.4f}  corr(D) {np.corrcoef(rS,rD)[0,1]:.4f}  '
              f'corr(DEP) {np.corrcoef(rS,rP)[0,1]:.4f}  ->  BLEND {sb["total"]:.6f} '
              f'w={sorted(set(B.w.round(2)))}')
        res[lam] = dict(solo=solo['total'], band_mass=bm, entropy=en, blend=sb['total'],
                        corr_D=float(np.corrcoef(rS, rD)[0, 1]))
        if lam > 0:
            cmp = Jm[KEY + ['actual_kwh']].copy(); cmp['champ'] = Jm.CHAMPION
            cmp = cmp.merge(B[KEY + ['pred']].rename(columns={'pred': 'cand'}), on=KEY)
            took, arb = arbitrate(cmp, 'cand', 'champ', n_comparisons=4)
            print(f'         ARBITER vs champion: delta={arb["point_delta"]:+.6f} '
                  f'sd={arb["paired_sd"]:.6f} P={arb["p_better"]:.3f} '
                  f'-> {"CHAMPION" if took else "rejected"}')
            res[lam]['arb'] = arb
    bl = max(res, key=lambda k: res[k]['blend'])
    print(f'\n  best lambda={bl} blend {res[bl]["blend"]:.6f} vs champion {champ:.6f} '
          f'(honest 0.634573, seed floor 0.001635)')
    json.dump(res, open(N + 'S16-N8_shift.json', 'w'), indent=1, default=str)
