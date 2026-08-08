"""S16-N9 * select features for the PEAK, not the tails.

S16-N8 isolated the mechanism behind nine replications of the point->action barrier, with sharpness
held exactly constant by construction: MAE improvement comes from the TAILS of the error
distribution, band hits come from its PEAK.  Translating D's sharp density to the composed
pipeline's more accurate location kept band mass at 0.758 and still drove the u=4 rate down
monotonically, 0.3424 -> 0.3303.

That mechanism indicts a step this project has never questioned.  The deployed architecture picks
its 150 features by taking the importances of `mu`, an L2-trained regressor fitted to the physics
teacher `pc_true`.  Importance under a squared/absolute loss measures how much a feature reduces
LARGE errors -- exactly the tail quantity S16-N8 says is the wrong target.  The features that place
the peak inside +-0.06 need not be the features that shorten the tails, and nothing in this
pipeline has ever checked whether they are the same set.

So: keep the entire architecture fixed and change only the selector's objective.  The band selector
is a binary classifier for the event |pc_true - cf| <= 0.06 -- the settlement event itself, at the
teacher stage -- and its top-150 importances replace the regressor's.  Everything downstream is
untouched: same teacher, same DART, same decision layer, same fold-outside blend.

Three arms, three seeds each per contract R10, all scored as means:
    MAE     the incumbent selector, which is also the plumbing control -- it must reproduce the
            deployed lineage, or the harness is not what we think it is
    BAND    top-150 by band-hit importance
    UNION   the union of both top-100s, in case the two objectives are complementary rather than
            competing

Reported per arm: solo, blend, the u=4 rate, band mass and entropy, error correlation with D, and
the overlap between the two selected feature sets -- because if the overlap is near 1.0 the
question answers itself and the axis closes without a single further fit.
"""
import sys, json, time
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/scratch')
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/nodes')
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/engine')
from harness import surface, MU
from lib import CAPS, FOLDS, official_total
from loop_lib import (canonical_keys, align_prob, load_depavg, utility_frames, fo_policy,
                      KEY, W, DEP, AB)
from arbiter import arbitrate

N = '/Users/um-yunsang/BARAM2026/research/nodes/'
NC = 26
SEEDS = (20260803, 20260804, 20260805)
C = (np.arange(NC) + 0.5) * W
DART = dict(objective='multiclass', boosting_type='dart', n_estimators=400, learning_rate=0.08,
            num_leaves=31, min_child_samples=60, subsample=0.85, subsample_freq=1,
            colsample_bytree=0.4, reg_lambda=3.0, n_jobs=6, verbose=-1)
SEL = dict(objective='binary', n_estimators=300, learning_rate=0.05, num_leaves=63,
           min_child_samples=100, colsample_bytree=0.6, subsample=0.8, subsample_freq=1,
           reg_lambda=5.0, n_jobs=6, verbose=-1)


def blend_dep(P, R, dep):
    M, solo, _ = fo_policy(utility_frames(P, R), R)
    J = M[KEY + ['actual_kwh', 'prediction_kwh']].rename(columns={'prediction_kwh': 'M'}).merge(dep, on=KEY)
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
    return B, solo['total'], official_total(
        B.assign(prediction_kwh=B.pred)[['group_id', 'actual_kwh', 'prediction_kwh']])['total']


if __name__ == '__main__':
    A, FR, COLS = surface(('G2', 'DROP:grid__'))
    R = canonical_keys(); dep = load_depavg()
    grp = A['grp'].to_numpy(); idx = A.index
    cf = A['cf'].to_numpy(); pct = A['pc_true'].to_numpy()
    valid = np.isfinite(cf) & (cf >= 0.1)
    w_prod = np.where(valid, np.clip(cf, 0, 1.2), 0.05)
    w_valid = np.where(valid, 1.0, 0.15)
    gapv = pct - cf
    cls = np.clip(np.nan_to_num(cf, nan=0.0) / W, 0, NC - 1).astype(int)
    band = (np.abs(pct - cf) <= 0.06).astype(int)
    print(f'teacher band-hit base rate on labelled rows: '
          f'{band[np.isfinite(cf) & np.isfinite(pct)].mean():.4f}')

    probs = {f'{a}_{s}': [] for a in ('MAE', 'BAND', 'UNION') for s in SEEDS}
    overlaps = []
    for f, (a_, b_) in FOLDS.items():
        a_ = pd.Timestamp(a_); b_ = pd.Timestamp(b_)
        tr = np.asarray(idx < a_); va = np.asarray((idx >= a_) & (idx <= b_))
        keep = np.isfinite(cf[va])
        cm = tr & np.isfinite(cf) & (~(gapv >= 0.05))
        for sd in SEEDS:
            t0 = time.time()
            mp = dict(MU); mp['random_state'] = sd
            m = tr & np.isfinite(pct)
            mu = lgb.LGBMRegressor(**mp)
            mu.fit(A.loc[m, COLS], pct[m], sample_weight=w_prod[m])
            pc = np.clip(mu.predict(A[COLS]), 0, 1)
            imp_mae = pd.Series(mu.feature_importances_, index=COLS).sort_values(ascending=False)
            mb = tr & np.isfinite(cf) & np.isfinite(pct)
            sc_ = lgb.LGBMClassifier(**dict(SEL, random_state=sd))
            sc_.fit(A.loc[mb, COLS], band[mb], sample_weight=w_prod[mb])
            imp_bnd = pd.Series(sc_.feature_importances_, index=COLS).sort_values(ascending=False)
            s_mae = list(imp_mae.head(150).index); s_bnd = list(imp_bnd.head(150).index)
            ov = len(set(s_mae) & set(s_bnd)) / 150
            overlaps.append(ov)
            s_uni = list(dict.fromkeys(list(imp_mae.head(100).index) + list(imp_bnd.head(100).index)))
            for arm, sel in (('MAE', s_mae), ('BAND', s_bnd), ('UNION', s_uni)):
                B0 = A[sel].copy(); B0['pc_hat'] = pc
                for k in (1, 2, 3):
                    B0[f'ig{k}'] = (grp == k).astype('float32')
                d = lgb.LGBMClassifier(**dict(DART, random_state=sd))
                d.fit(B0[cm], cls[cm], sample_weight=w_valid[cm])
                raw = d.predict_proba(B0[va])
                P = np.zeros((raw.shape[0], NC)); P[:, np.asarray(d.classes_, int)] = raw
                probs[f'{arm}_{sd}'].append(P[keep])
            print(f'  [{f}] seed {sd} overlap(top150 MAE vs BAND)={ov:.3f} '
                  f'|union|={len(s_uni)} {round(time.time()-t0,1)}s', flush=True)

    print(f'\n=== feature-set overlap: mean {np.mean(overlaps):.3f} '
          f'(1.000 would close this axis outright) ===')
    Ps = {k: np.vstack(v) for k, v in probs.items()}
    Bch, solo_D, _ = fo_policy(utility_frames(align_prob('D', R), R), R)
    Jc = Bch[KEY + ['actual_kwh', 'prediction_kwh']].rename(columns={'prediction_kwh': 'D'}).merge(dep, on=KEY)
    Jc['CHAMPION'] = 0.30 * Jc.D + 0.70 * Jc.DEPAVG
    champ = official_total(Jc[['group_id', 'actual_kwh']].assign(prediction_kwh=Jc.CHAMPION))['total']
    capJ = Jc.group_id.map(CAPS).to_numpy(); yJ = Jc.actual_kwh.to_numpy(); scm = yJ >= 0.10 * capJ
    out = {'overlap': float(np.mean(overlaps))}
    for arm in ('MAE', 'BAND', 'UNION'):
        bs, ss = [], []
        for sd in SEEDS:
            _, so, bl = blend_dep(Ps[f'{arm}_{sd}'], R, dep); bs.append(bl); ss.append(so)
        Pav = np.mean([Ps[f'{arm}_{s}'] for s in SEEDS], axis=0)
        Bv, so_a, bl_a = blend_dep(Pav, R, dep)
        Mv, _, _ = fo_policy(utility_frames(Pav, R), R)
        Jm = Jc.merge(Mv[KEY + ['prediction_kwh']].rename(columns={'prediction_kwh': 'S'}), on=KEY)
        r = (Jm.S.to_numpy() - yJ) / capJ
        md = C[np.argmax(Pav, 1)]
        bm = float((Pav * (np.abs(C[None, :] - md[:, None]) <= 0.06)).sum(1).mean())
        print(f'\n  {arm}: blend mean={np.mean(bs):.6f} sd={np.std(bs, ddof=1):.6f}  '
              f'seed-avg={bl_a:.6f}  solo={np.mean(ss):.6f}')
        print(f'        u=4 {(np.abs(r) <= 0.06)[scm].mean():.4f}  band mass {bm:.3f}  '
              f'corr(D) {np.corrcoef(r, (Jm.D.to_numpy()-yJ)/capJ)[0,1]:.4f}')
        out[arm] = dict(per_seed=bs, mean=float(np.mean(bs)), sd=float(np.std(bs, ddof=1)),
                        seed_avg=bl_a, solo=float(np.mean(ss)), band_mass=bm)
        if arm != 'MAE':
            cmp = Jc[KEY + ['actual_kwh']].copy(); cmp['champ'] = Jc.CHAMPION
            cmp = cmp.merge(Bv[KEY + ['pred']].rename(columns={'pred': 'cand'}), on=KEY)
            took, arb = arbitrate(cmp, 'cand', 'champ', n_comparisons=2)
            print(f'        ARBITER vs champion {champ:.6f}: delta={arb["point_delta"]:+.6f} '
                  f'sd={arb["paired_sd"]:.6f} P={arb["p_better"]:.3f} '
                  f'-> {"CHAMPION" if took else "rejected"}')
            out[arm]['arb'] = arb
    json.dump(out, open(N + 'S16-N9_bandsel.json', 'w'), indent=1, default=str)
