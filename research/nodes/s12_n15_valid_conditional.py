"""S12-N15 * train the classifier on the conditional the decision layer actually uses.

Finding being applied.  S12-N12 measured, on the point-forecast axis, that fitting only the
rows the metric scores (cf >= 0.1) is worth +0.006161 in 1-NMAE over fitting all rows
(0.863750 vs 0.857589) -- the single largest point-accuracy move found this session, and larger
than the entire gap between the best existing artifact (0.864617) and the best configuration
found in the bake-off (0.866147).

Why it should transfer to member D, and why it is not what D already does.  D's classifier
trains on `cm = tr & isfinite(cf) & ~(gapv >= 0.05)` with sample weight 0.15 on rows with
cf < 0.1, so it estimates p(cf | x) over the full support.  The decision layer then multiplies
by the indicator C >= 0.10 and renormalises, i.e. it *wants* p(cf | cf >= 0.1, x).  Estimating
the unconditional law and conditioning afterwards spends model capacity on a region the metric
never scores and then throws that region away.  Fitting the conditional directly is the same
correction the bake-off measured, applied to the real architecture.

Variants (everything else byte-identical to research/nodes/s7_more.py's D):
  DV   classifier trained on scored rows only (cf >= 0.1), availability gate kept
  DVT  DV, and the physics teacher also fitted on scored rows only
  DVP  DV, plus the best point forecast from S12-N12 (LightGBM L1 on scored rows, with the
       physics teacher as an input) spliced into B as an extra column after top-150 selection
"""
import sys, json, time
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/scratch')
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/nodes')
from harness import surface, MU
from lib import FOLDS, CAPS

N = '/Users/um-yunsang/BARAM2026/research/nodes/'
W = 0.04; NC = 26
DART_CLF = dict(objective='multiclass', boosting_type='dart', n_estimators=400,
                learning_rate=0.08, num_leaves=31, min_child_samples=60, subsample=0.85,
                subsample_freq=1, colsample_bytree=0.4, reg_lambda=3.0,
                random_state=20260803, n_jobs=6, verbose=-1)
L1P = dict(objective='l1', n_estimators=900, learning_rate=0.035, num_leaves=63,
           min_child_samples=40, subsample=0.85, subsample_freq=1, colsample_bytree=0.4,
           reg_lambda=3.0, random_state=20260801, n_jobs=6, verbose=-1)


def build(tag, valid_clf=True, valid_teacher=False, add_point=False):
    A, FR, COLS = surface(('G2', 'DROP:grid__'))
    cf = A['cf'].to_numpy(); grp = A['grp'].to_numpy(); idx = A.index
    valid = np.isfinite(cf) & (cf >= 0.1)
    w_prod = np.where(valid, np.clip(cf, 0, 1.2), 0.05)
    w_valid = np.where(valid, 1.0, 0.15)
    gapv = A['pc_true'].to_numpy() - cf
    ok = np.isfinite(A['pc_true'].to_numpy())

    rows = []; probs = []
    for f, (a, b) in FOLDS.items():
        a = pd.Timestamp(a); b = pd.Timestamp(b)
        tr = np.asarray(idx < a); va = np.asarray((idx >= a) & (idx <= b)); keep = np.isfinite(cf[va])
        t0 = time.time()
        m = tr & ok & (valid if valid_teacher else np.ones(len(A), bool))
        mu = lgb.LGBMRegressor(**MU)
        mu.fit(A.loc[m, COLS], A.loc[m, 'pc_true'], sample_weight=w_prod[m])
        pc = np.clip(mu.predict(A[COLS]), 0, 1)
        sel = list(pd.Series(mu.feature_importances_, index=COLS).sort_values(ascending=False).head(150).index)
        B = A[sel].copy(); B['pc_hat'] = pc
        for k in (1, 2, 3):
            B[f'ig{k}'] = (grp == k).astype('float32')
        if add_point:
            F = A[COLS].copy(); F['pc_hat'] = pc
            pm = lgb.LGBMRegressor(**L1P)
            pm.fit(F[tr & valid], cf[tr & valid])
            B['pt__l1'] = np.clip(pm.predict(F), 0, 1.1)
            B['pt__gap'] = B['pt__l1'] - pc

        cls = np.clip(np.nan_to_num(cf, nan=0.0) / W, 0, NC - 1).astype(int)
        cm = tr & np.isfinite(cf) & (~(gapv >= 0.05))
        if valid_clf:
            cm = cm & valid
        d = lgb.LGBMClassifier(**DART_CLF)
        d.fit(B[cm], cls[cm], sample_weight=w_valid[cm])
        raw = d.predict_proba(B[va])
        P = np.zeros((raw.shape[0], NC)); P[:, np.asarray(d.classes_, int)] = raw
        rows.append(pd.DataFrame({'fold_id': f, 'group_id': grp[va][keep],
                                  'forecast_kst_dtm': idx[va][keep], 'cf': cf[va][keep],
                                  'mean_gen_g': [float(np.nanmean(cf[tr & (grp == g)])) for g in grp[va][keep]]}))
        probs.append(P[keep])
        print(f'  [{tag}] {f} {round(time.time()-t0,1)}s  (train rows {int(cm.sum())})', flush=True)
    R = pd.concat(rows, ignore_index=True)
    Pf = np.vstack(probs)
    R.to_parquet(N + f'S7-N8_{tag}_keys.parquet', index=False)
    np.save(N + f'S7-N8_{tag}_prob.npy', Pf)
    return R, Pf


if __name__ == '__main__':
    build('DV', valid_clf=True)
    build('DVP', valid_clf=True, add_point=True)
    build('DVT', valid_clf=True, valid_teacher=True)
    print('DONE', flush=True)
