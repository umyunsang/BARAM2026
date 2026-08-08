"""S16-N7 * IDR / EasyUQ -- the one construction the lane could not rule out as sharp AND decorrelated.

Where this stands.  S16 established that the ensembling axis was closed on the wrong statistic
(continuous-error correlation 0.947, when the metric pays on band hits whose correlation is only
0.6245), and that a per-row oracle over members we already own scores 0.723333 at four members and
0.807182 at fourteen.  Three gates were then built and all three failed:

    S16-N3  FFORMA instance-conditional soft weights   top-1 0.5363   best rule 0.635832
    S16-N4  four binary band-hit models                recoverable-set accuracy 0.4205   0.636247
    S16-N6  the same over all fourteen members         recoverable-set accuracy 0.3021   0.635821

The fourteen-member result is the decisive one: widening the pool cut the needle fraction from
0.707 to 0.314 and raised the oracle to 0.807, yet the gate's realised hit rate stayed at 0.3509
against the champion's 0.3503.  Identification power on the recoverable set is invariant to pool
size.  Selection is closed.

That leaves exactly one named construction from the lane.  Every member this project owns is a
GBDT trained for accuracy, and in Schulz & Lerch's wind-gust postprocessing benchmark IDR is the
outlier: the NARROWEST prediction interval of all methods (4.72 against EMOS 5.94 and BQN 4.94)
with the WORST postprocessing CRPS (0.98 against 0.84) and 84% coverage at a nominal 90.5%.  It is
sharp, deliberately not most accurate, and order-restricted rather than gradient-boosted -- the
only candidate whose failure mode is the OPPOSITE of ours.  Our composed member failed the blend
precisely by being more accurate and flatter (band mass 0.567 against D's 0.759).

EasyUQ construction: take a point forecast x, and estimate P(cf <= c | x) by isotonic regression of
the indicator on x, decreasing in nothing and increasing in the threshold, fold-outside.  The
covariate here is the composed pipeline's point forecast, the most accurate this project has
produced (1-NMAE 0.867079).  What is measured is not only Total but SHARPNESS and the error
correlation with D and DEPAVG, because a member that lands near 0.85 while holding its solo score
is the thing nine replications have said we lack.
"""
import sys, json
import numpy as np, pandas as pd
from sklearn.isotonic import IsotonicRegression
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
EDGES = np.arange(1, NC) * W          # 25 interior thresholds of the 26 bins


def sharpness(P):
    """band mass within +-0.06 of the mode, and entropy -- the two numbers that separated our
    composed member (0.567 / 1.941) from the incumbent D (0.759 / 1.404)."""
    C = (np.arange(NC) + 0.5) * W
    md = C[np.argmax(P, 1)]
    m = (np.abs(C[None, :] - md[:, None]) <= 0.06)
    return float((P * m).sum(1).mean()), float((-P * np.log(P + 1e-12)).sum(1).mean())


if __name__ == '__main__':
    R = canonical_keys(); dep = load_depavg()
    cap = R.group_id.map(CAPS).to_numpy(); y = R.cf.to_numpy()
    Dm, solo_D, _ = fo_policy(utility_frames(align_prob('D', R), R), R)
    J = Dm[KEY + ['actual_kwh', 'prediction_kwh']].rename(columns={'prediction_kwh': 'D'}).merge(dep, on=KEY)
    J['CHAMPION'] = 0.30 * J.D + 0.70 * J.DEPAVG
    base = J[['group_id', 'actual_kwh']]
    champ = official_total(base.assign(prediction_kwh=J.CHAMPION))['total']

    PD = align_prob('D', R)
    PCmp = np.mean([np.load(N + f'S15-N9_prob_{s}.npy') for s in SEEDS], axis=0)
    C = (np.arange(NC) + 0.5) * W
    x_cmp = (PCmp * C[None, :]).sum(1)          # composed pipeline point forecast
    x_D = (PD * C[None, :]).sum(1)
    print(f'covariates: composed mean in [{x_cmp.min():.3f},{x_cmp.max():.3f}], '
          f'D mean in [{x_D.min():.3f},{x_D.max():.3f}]')
    bm_D, en_D = sharpness(PD); bm_C, en_C = sharpness(PCmp)
    print(f'sharpness  D: band mass {bm_D:.3f} entropy {en_D:.3f}   '
          f'composed: {bm_C:.3f} / {en_C:.3f}')

    out = {}
    for nm, x in (('IDR_composed', x_cmp), ('IDR_D', x_D)):
        P = np.zeros((len(R), NC))
        for f in FOLDS:
            sel = (R.fold_id == f).to_numpy(); trn = (~sel) & np.isfinite(y)
            cdf = np.zeros((int(sel.sum()), NC - 1))
            for j, c in enumerate(EDGES):
                ir = IsotonicRegression(increasing=False, out_of_bounds='clip', y_min=0, y_max=1)
                ir.fit(x[trn], (y[trn] <= c).astype(float))
                cdf[:, j] = ir.predict(x[sel])
            cdf = np.maximum.accumulate(cdf, axis=1)          # enforce monotone CDF in threshold
            F = np.concatenate([np.zeros((len(cdf), 1)), cdf, np.ones((len(cdf), 1))], axis=1)
            pm = np.diff(F, axis=1)
            pm = np.clip(pm, 1e-9, None); pm /= pm.sum(1, keepdims=True)
            P[sel] = pm
        bm, en = sharpness(P)
        M, solo, _ = fo_policy(utility_frames(P, R), R)
        Jm = J.merge(M[KEY + ['prediction_kwh']].rename(columns={'prediction_kwh': 'IDR'}), on=KEY)
        capJ = Jm.group_id.map(CAPS).to_numpy(); yJ = Jm.actual_kwh.to_numpy()
        rI = (Jm.IDR.to_numpy() - yJ) / capJ
        rD = (Jm.D.to_numpy() - yJ) / capJ
        rP = (Jm.DEPAVG.to_numpy() - yJ) / capJ
        cD = float(np.corrcoef(rI, rD)[0, 1]); cP = float(np.corrcoef(rI, rP)[0, 1])
        hI = np.abs(rI) <= 0.06; hD = np.abs(rD) <= 0.06
        sc = yJ >= 0.10 * capJ
        cb = float(np.corrcoef(hI[sc].astype(float), hD[sc].astype(float))[0, 1])
        print(f'\n--- {nm} ---')
        print(f'  solo Total {solo["total"]:.6f}   band mass {bm:.3f} entropy {en:.3f}  '
              f'(D: {bm_D:.3f}/{en_D:.3f})')
        print(f'  error corr with D {cD:.4f}  with DEPAVG {cP:.4f}   '
              f'(D-DEPAVG is 0.9217; our composed member was 0.9643)')
        print(f'  band-hit corr with D {cb:.4f}   u=4 rate {hI[sc].mean():.4f} '
              f'(D {hD[sc].mean():.4f})')
        # blend against the champion pair, fold-outside weight
        rows = []
        for f in FOLDS:
            oth = Jm[Jm.fold_id != f]; held = Jm[Jm.fold_id == f]; best = None
            for w in np.arange(0, 1.001, 0.05):
                t = official_total(oth.assign(prediction_kwh=w * oth.IDR + (1 - w) * oth.CHAMPION)[
                    ['group_id', 'actual_kwh', 'prediction_kwh']])['total']
                if best is None or t > best[0]:
                    best = (t, w)
            rows.append(held.assign(pred=best[1] * held.IDR + (1 - best[1]) * held.CHAMPION,
                                    w=best[1]))
        B = pd.concat(rows, ignore_index=True)
        sb = official_total(B.assign(prediction_kwh=B.pred)[['group_id', 'actual_kwh', 'prediction_kwh']])
        print(f'  blended with champion (fold-outside w={sorted(set(B.w.round(2)))}): '
              f'{sb["total"]:.6f}   champion {champ:.6f}')
        cmp = Jm[KEY + ['actual_kwh']].copy(); cmp['champ'] = Jm.CHAMPION
        cmp = cmp.merge(B[KEY + ['pred']].rename(columns={'pred': 'cand'}), on=KEY)
        took, arb = arbitrate(cmp, 'cand', 'champ', n_comparisons=1)
        print(f'  ARBITER: delta={arb["point_delta"]:+.6f} sd={arb["paired_sd"]:.6f} '
              f'P={arb["p_better"]:.3f} -> {"CHAMPION" if took else "rejected"}')
        out[nm] = dict(solo=solo['total'], band_mass=bm, entropy=en, corr_D=cD, corr_DEP=cP,
                       bandhit_corr_D=cb, hit=float(hI[sc].mean()), blend=sb['total'], arb=arb)
        np.save(N + f'S16-N7_{nm}_prob.npy', P)
    json.dump(out, open(N + 'S16-N7_idr.json', 'w'), indent=1, default=str)
