"""S15-N4 * seed ensembling applied to the champion's own member D.

Why this is a live axis and not a repeat of the closed ensembling axis.  The axis was closed on a
measured minimum pairwise error correlation of 0.934 across the 12 deployed stems and 15 own
members -- but every one of those pairs is a different MODEL CLASS or feature set.  Nobody ever
measured the correlation between refits of the SAME configuration under different seeds, and ESL
15.1's variance algebra says that is exactly where the remaining reducible variance lives:
Var = rho*s^2 + (1-rho)*s^2/B, so the gain is governed by rho, and seed-to-seed rho is far below
0.934 by construction.

S15-N3 measured the effect on the point pipeline: three-seed averaging moves Total
0.602325 -> 0.603454 (+0.001129) and 1-NMAE 0.865850 -> 0.866588, against a seed spread of
0.001635.  This node applies the same operation to member D itself -- the DART classifier that the
champion blends at weight 0.30 -- by averaging the 26-class PROBABILITY matrices across seeds
before the decision layer, which is the correct place to average for a distributional member.
"""
import sys, json, time
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/scratch')
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/nodes')
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/engine')
from harness import surface, MU
from lib import FOLDS, CAPS, official_total

N = '/Users/um-yunsang/BARAM2026/research/nodes/'
W = 0.04; NC = 26
DART = dict(objective='multiclass', boosting_type='dart', n_estimators=400, learning_rate=0.08,
            num_leaves=31, min_child_samples=60, subsample=0.85, subsample_freq=1,
            colsample_bytree=0.4, reg_lambda=3.0, n_jobs=6, verbose=-1)
SEEDS = (20260803, 20260811, 20260819, 20260827, 20260835)

if __name__ == '__main__':
    A, FR, COLS = surface(('G2', 'DROP:grid__'))
    cf = A['cf'].to_numpy(); grp = A['grp'].to_numpy(); idx = A.index
    valid = np.isfinite(cf) & (cf >= 0.1)
    w_prod = np.where(valid, np.clip(cf, 0, 1.2), 0.05)
    w_valid = np.where(valid, 1.0, 0.15)
    gapv = A['pc_true'].to_numpy() - cf
    rows = []; probs = {s: [] for s in SEEDS}
    for f, (a, b) in FOLDS.items():
        a = pd.Timestamp(a); b = pd.Timestamp(b)
        tr = np.asarray(idx < a); va = np.asarray((idx >= a) & (idx <= b)); keep = np.isfinite(cf[va])
        m = tr & np.isfinite(A['pc_true'].to_numpy())
        cls = np.clip(np.nan_to_num(cf, nan=0.0) / W, 0, NC - 1).astype(int)
        cm = tr & np.isfinite(cf) & (~(gapv >= 0.05))
        rows.append(pd.DataFrame({'fold_id': f, 'group_id': grp[va][keep],
                                  'forecast_kst_dtm': idx[va][keep], 'cf': cf[va][keep],
                                  'mean_gen_g': [float(np.nanmean(cf[tr & (grp == g)])) for g in grp[va][keep]]}))
        for sd in SEEDS:
            t0 = time.time()
            mp = dict(MU); mp['random_state'] = sd
            mu = lgb.LGBMRegressor(**mp)
            mu.fit(A.loc[m, COLS], A.loc[m, 'pc_true'], sample_weight=w_prod[m])
            pc = np.clip(mu.predict(A[COLS]), 0, 1)
            sel = list(pd.Series(mu.feature_importances_, index=COLS).sort_values(ascending=False).head(150).index)
            B = A[sel].copy(); B['pc_hat'] = pc
            for k in (1, 2, 3):
                B[f'ig{k}'] = (grp == k).astype('float32')
            dp = dict(DART); dp['random_state'] = sd
            d = lgb.LGBMClassifier(**dp)
            d.fit(B[cm], cls[cm], sample_weight=w_valid[cm])
            raw = d.predict_proba(B[va])
            P = np.zeros((raw.shape[0], NC)); P[:, np.asarray(d.classes_, int)] = raw
            probs[sd].append(P[keep])
            print(f'  [{f}] seed {sd} {round(time.time()-t0,1)}s', flush=True)
    R = pd.concat(rows, ignore_index=True)
    R.to_parquet(N + 'S7-N8_DSEED_keys.parquet', index=False)
    Ps = {s: np.vstack(v) for s, v in probs.items()}
    for s, P in Ps.items():
        np.save(N + f'S7-N8_DSEED{s}_prob.npy', P)
    np.save(N + 'S7-N8_DSEED_prob.npy', np.mean(list(Ps.values()), axis=0))
    # error correlation BETWEEN SEEDS, the quantity nobody measured
    C26 = (np.arange(NC) + 0.5) * W
    mu_ = {s: (P * C26[None, :]).sum(1) for s, P in Ps.items()}
    E = pd.DataFrame({str(s): mu_[s] - R.cf.to_numpy() for s in SEEDS})
    print('\n--- error correlation BETWEEN SEED REFITS of the same configuration ---')
    print(E.corr().round(4).to_string())
    off = E.corr().where(~np.eye(len(SEEDS), dtype=bool)).stack()
    print(f'  mean off-diagonal rho = {off.mean():.4f}   (the deployed pool minimum was 0.934)')
    json.dump({'seed_rho_mean': float(off.mean()), 'seeds': list(SEEDS)},
              open(N + 'S15-N4_seed_ensemble.json', 'w'), indent=1)
    print('DONE', flush=True)
