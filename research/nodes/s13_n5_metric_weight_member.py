"""S13-N5 * (S5 rung) put the metric's own row weight into the real architecture.

Two independent results converge on the same defect.

(1) MEASURED, S13-N4.  On the point-forecast axis, group-balanced weights w_i = 1/n_{g(i)}
    gave 1-NMAE +0.000897, and combining them with block-aware outage removal gave
    Total 0.625665 versus the control's 0.603068 (+0.022597), almost entirely through FICR
    (0.340190 -> 0.385018).

(2) DERIVED, research/lanes/S13_S5_preprocessing_deep.md.  Because the metric is
        Total = 0.5*(1 - (1/3)SUM_g NMAE_g) + 0.5*(1/3)SUM_g FICR_g
    with NMAE_g an unweighted mean over that group's scored rows and FICR_g an
    actual-weighted step reward over the same rows, the sensitivity of Total to row i is

        dTotal/d(row i)  ~  (1/n_g) * [ a + b * y_i ]

    i.e. an inverse-group-size factor multiplying an affine function of the row's own
    production.  The repository currently uses w_prod ~ clip(cf,0,1.2) for the teacher (the
    pure b-extreme, a=0) and w_valid ~ 1.0/0.15 for the classifier (the pure a-extreme, b=0),
    and NEITHER carries the 1/n_g factor.  Group 3 supplies only 9.0-15.4% of the scored
    training rows across the three folds while carrying exactly 1/3 of the score.

This node applies that weight to the verified D architecture (research/nodes/s7_more.py),
changing nothing else -- same features, same teacher, same top-150 selection, same DART
classifier, same discretisation, same decision layer, same fold-outside gate.

  DW0   control: reproduce D exactly (sanity anchor, must return 0.625669 solo)
  DW1   group-balanced only:      w = (1/n_g) * w_valid
  DW2   metric affine, beta=1:    w = (1/n_g) * (1 + y/mean_y_g) on scored rows
  DW3   metric affine, beta=2:    w = (1/n_g) * (1 + 2*y/mean_y_g) on scored rows
The teacher keeps its own production weighting but gains the 1/n_g factor in DW1-DW3.
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


def build(tag, group_balance=False, beta=None):
    A, FR, COLS = surface(('G2', 'DROP:grid__'))
    cf = A['cf'].to_numpy(); grp = A['grp'].to_numpy(); idx = A.index
    valid = np.isfinite(cf) & (cf >= 0.1)
    w_prod = np.where(valid, np.clip(cf, 0, 1.2), 0.05)
    w_valid = np.where(valid, 1.0, 0.15)
    gapv = A['pc_true'].to_numpy() - cf

    rows = []; probs = []
    for f, (a, b) in FOLDS.items():
        a = pd.Timestamp(a); b = pd.Timestamp(b)
        tr = np.asarray(idx < a); va = np.asarray((idx >= a) & (idx <= b)); keep = np.isfinite(cf[va])
        t0 = time.time()
        cm = tr & np.isfinite(cf) & (~(gapv >= 0.05))

        ng = {g: max(int((tr & valid & (grp == g)).sum()), 1) for g in (1, 2, 3)}
        my = {g: float(np.nanmean(cf[tr & valid & (grp == g)])) for g in (1, 2, 3)}
        gfac = np.array([1.0 / ng[g] for g in grp]) if group_balance else np.ones(len(A))
        gfac = gfac / gfac[cm].mean()

        if beta is None:
            wc = w_valid * gfac
        else:
            yr = np.where(valid, cf / np.array([my[g] for g in grp]), 0.0)
            wc = np.where(valid, 1.0 + beta * yr, 0.15) * gfac
        wt = w_prod * gfac

        m = tr & np.isfinite(A['pc_true'].to_numpy())
        mu = lgb.LGBMRegressor(**MU)
        mu.fit(A.loc[m, COLS], A.loc[m, 'pc_true'], sample_weight=wt[m])
        pc = np.clip(mu.predict(A[COLS]), 0, 1)
        sel = list(pd.Series(mu.feature_importances_, index=COLS).sort_values(ascending=False).head(150).index)
        B = A[sel].copy(); B['pc_hat'] = pc
        for k in (1, 2, 3):
            B[f'ig{k}'] = (grp == k).astype('float32')
        cls = np.clip(np.nan_to_num(cf, nan=0.0) / W, 0, NC - 1).astype(int)
        d = lgb.LGBMClassifier(**DART_CLF)
        d.fit(B[cm], cls[cm], sample_weight=wc[cm])
        raw = d.predict_proba(B[va])
        P = np.zeros((raw.shape[0], NC)); P[:, np.asarray(d.classes_, int)] = raw
        rows.append(pd.DataFrame({'fold_id': f, 'group_id': grp[va][keep],
                                  'forecast_kst_dtm': idx[va][keep], 'cf': cf[va][keep],
                                  'mean_gen_g': [float(np.nanmean(cf[tr & (grp == g)])) for g in grp[va][keep]]}))
        probs.append(P[keep])
        print(f'  [{tag}] {f} {round(time.time()-t0,1)}s', flush=True)
    R = pd.concat(rows, ignore_index=True)
    Pf = np.vstack(probs)
    R.to_parquet(N + f'S7-N8_{tag}_keys.parquet', index=False)
    np.save(N + f'S7-N8_{tag}_prob.npy', Pf)


if __name__ == '__main__':
    build('DW1', group_balance=True)
    build('DW2', group_balance=True, beta=1.0)
    build('DW3', group_balance=True, beta=2.0)
    print('DONE', flush=True)
