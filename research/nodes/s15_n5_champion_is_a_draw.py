"""S15-N5 * is `current_best = 0.636184` a lucky draw from its own configuration?

The observation that forces this node.  S15-N4 refitted member D under five seeds and blended each
through the identical champion pipeline.  The five blends span 0.632652 - 0.634317 with mean
0.633167, while the deployed champion sits at 0.636184 -- outside the range entirely, about 4.5
seed-sd above the mean of its own configuration.  And the distance from that seed-mean to the
champion (0.003017) is almost exactly the champion's entire measured advantage over plain DEPAVG
(0.003592), which the Model Confidence Set had already declared indistinguishable.

CONFOUND IN S15-N4, DECLARED AND NOW REMOVED.  The original generator research/nodes/s7_more.py
seeds the physics teacher with harness.MU's random_state = 20260801 and the DART classifier with
20260803.  S15-N4 set BOTH to the same value, so it varied two seeds at once and its runs are not
members of the champion's configuration family.  This node holds the teacher seed at 20260801 --
exactly as the champion has it -- and varies ONLY the DART classifier seed, which is the family
the champion is drawn from.

If the champion sits in the upper tail of that distribution, then `current_best` is a lucky draw,
its advantage over DEPAVG is a selection artefact, and every treatment this project has rejected
over three sessions was competing against noise rather than against a real incumbent.
"""
import sys, json, time
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/scratch')
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/nodes')
from harness import surface, MU
from lib import FOLDS, CAPS

N = '/Users/um-yunsang/BARAM2026/research/nodes/'
W = 0.04; NC = 26
DART = dict(objective='multiclass', boosting_type='dart', n_estimators=400, learning_rate=0.08,
            num_leaves=31, min_child_samples=60, subsample=0.85, subsample_freq=1,
            colsample_bytree=0.4, reg_lambda=3.0, n_jobs=6, verbose=-1)
# the champion's own DART seed is 20260803; these are siblings from the same family
DART_SEEDS = (20260804, 20260805, 20260806, 20260807, 20260808, 20260809)

if __name__ == '__main__':
    A, FR, COLS = surface(('G2', 'DROP:grid__'))
    cf = A['cf'].to_numpy(); grp = A['grp'].to_numpy(); idx = A.index
    valid = np.isfinite(cf) & (cf >= 0.1)
    w_prod = np.where(valid, np.clip(cf, 0, 1.2), 0.05)
    w_valid = np.where(valid, 1.0, 0.15)
    gapv = A['pc_true'].to_numpy() - cf
    rows = []; probs = {s: [] for s in DART_SEEDS}
    for f, (a, b) in FOLDS.items():
        a = pd.Timestamp(a); b = pd.Timestamp(b)
        tr = np.asarray(idx < a); va = np.asarray((idx >= a) & (idx <= b)); keep = np.isfinite(cf[va])
        rows.append(pd.DataFrame({'fold_id': f, 'group_id': grp[va][keep],
                                  'forecast_kst_dtm': idx[va][keep], 'cf': cf[va][keep],
                                  'mean_gen_g': [float(np.nanmean(cf[tr & (grp == g)])) for g in grp[va][keep]]}))
        t0 = time.time()
        # teacher held at the champion's seed, exactly as s7_more.py has it
        m = tr & np.isfinite(A['pc_true'].to_numpy())
        mu = lgb.LGBMRegressor(**MU)                      # random_state = 20260801
        mu.fit(A.loc[m, COLS], A.loc[m, 'pc_true'], sample_weight=w_prod[m])
        pc = np.clip(mu.predict(A[COLS]), 0, 1)
        sel = list(pd.Series(mu.feature_importances_, index=COLS).sort_values(ascending=False).head(150).index)
        B = A[sel].copy(); B['pc_hat'] = pc
        for k in (1, 2, 3):
            B[f'ig{k}'] = (grp == k).astype('float32')
        cls = np.clip(np.nan_to_num(cf, nan=0.0) / W, 0, NC - 1).astype(int)
        cm = tr & np.isfinite(cf) & (~(gapv >= 0.05))
        print(f'  [{f}] teacher (seed 20260801, as the champion) {round(time.time()-t0,1)}s', flush=True)
        for sd in DART_SEEDS:
            t1 = time.time()
            dp = dict(DART); dp['random_state'] = sd
            d = lgb.LGBMClassifier(**dp)
            d.fit(B[cm], cls[cm], sample_weight=w_valid[cm])
            raw = d.predict_proba(B[va])
            P = np.zeros((raw.shape[0], NC)); P[:, np.asarray(d.classes_, int)] = raw
            probs[sd].append(P[keep])
            print(f'    dart seed {sd} {round(time.time()-t1,1)}s', flush=True)
    R = pd.concat(rows, ignore_index=True)
    R.to_parquet(N + 'S15-N5_keys.parquet', index=False)
    for s, v in probs.items():
        np.save(N + f'S15-N5_prob_{s}.npy', np.vstack(v))
    json.dump({'dart_seeds': list(DART_SEEDS), 'teacher_seed': 20260801},
              open(N + 'S15-N5_meta.json', 'w'), indent=1)
    print('DONE', flush=True)
