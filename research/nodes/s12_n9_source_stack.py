"""S12-N9 * per-source teachers as STACKED FEATURES (the actual HEFTCom2024 mechanism).

S12-N3 tested only half of the principle recorded in
research/lanes/S6_ext_A_competitions.md sec 0.2 -- it trained one member per NWP source and then
AVERAGED them.  It measured corr(DL,DG) = 0.7924, by far the lowest member-pair error
correlation ever recorded in this project (every concatenated-matrix member sits at
0.98-0.99), which confirms the diversity mechanism is real; but each half-blind member scored
only 0.6088 / 0.6122 solo, too weak for the average to beat D's 0.6257.

The lane's actual finding was STACKING, not averaging: "NWP 소스별로 따로 학습한 뒤 스태킹...
소스를 열로 합치지 않는다".  A stack keeps the full-information model and adds each source's
own compressed opinion plus their disagreement as extra inputs, letting the meta-model learn
*when to trust which source* -- something neither the concatenated matrix (which sees raw
columns, not calibrated per-source opinions) nor the average (which uses a fixed weight) can
express.

Leakage control.  A stacked feature must be out-of-fold on the rows the meta-model trains on,
otherwise the meta-model sees an over-fit base prediction in training and an honest one at
validation.  The base source-teachers are therefore fitted with an INNER 3-fold split over the
training window only, and their training-row values are inner-out-of-fold; validation-row
values come from a base model fitted on the whole training window.  The pre-existing `pc_hat`
is left exactly as `s7_more.py` computes it, so the only change is the added columns.

Added columns: pcL (LDAPS-only teacher), pcG (GFS-only teacher), pcL-pcG, |pcL-pcG|,
pc_hat-pcL, pc_hat-pcG.  Spliced into B AFTER the top-150 selection, so they cannot be
dropped by unrelated feature competition (same convention as s9_n12_true_d_plus_rews.py).
"""
import sys, json, time
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/scratch')
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/nodes')
from harness import surface, MU
from lib import FOLDS, CAPS
from s12_n3_source_split import split_columns

N = '/Users/um-yunsang/BARAM2026/research/nodes/'
W = 0.04; NC = 26
DART_CLF = dict(objective='multiclass', boosting_type='dart', n_estimators=400,
                learning_rate=0.08, num_leaves=31, min_child_samples=60, subsample=0.85,
                subsample_freq=1, colsample_bytree=0.4, reg_lambda=3.0,
                random_state=20260803, n_jobs=6, verbose=-1)
INNER = 3


def base_teacher_oof(A, cols, tr, target_ok, w_prod, idx):
    """Inner-OOF base prediction on training rows; full-train fit for the rest."""
    pred = np.full(len(A), np.nan)
    tr_idx = np.where(tr & target_ok)[0]
    order = np.argsort(idx.values[tr_idx])
    tr_sorted = tr_idx[order]
    chunks = np.array_split(tr_sorted, INNER)
    for c in chunks:
        hold = np.zeros(len(A), bool); hold[c] = True
        fit = np.zeros(len(A), bool); fit[np.setdiff1d(tr_sorted, c)] = True
        m = lgb.LGBMRegressor(**MU)
        m.fit(A.loc[fit, cols], A.loc[fit, 'pc_true'], sample_weight=w_prod[fit])
        pred[c] = np.clip(m.predict(A.loc[hold, cols]), 0, 1)
    full = lgb.LGBMRegressor(**MU)
    full.fit(A.loc[tr & target_ok, cols], A.loc[tr & target_ok, 'pc_true'],
             sample_weight=w_prod[tr & target_ok])
    rest = ~np.isfinite(pred)
    pred[rest] = np.clip(full.predict(A.loc[rest, cols]), 0, 1)
    return pred


def build(tag='DSTK'):
    A, FR, COLS = surface(('G2', 'DROP:grid__'))
    cld, cgf, _ = split_columns(COLS)
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
        m = tr & ok
        mu = lgb.LGBMRegressor(**MU)
        mu.fit(A.loc[m, COLS], A.loc[m, 'pc_true'], sample_weight=w_prod[m])
        pc = np.clip(mu.predict(A[COLS]), 0, 1)
        sel = list(pd.Series(mu.feature_importances_, index=COLS).sort_values(ascending=False).head(150).index)

        pcL = base_teacher_oof(A, cld, tr, ok, w_prod, idx)
        pcG = base_teacher_oof(A, cgf, tr, ok, w_prod, idx)
        print(f'  [{tag}] {f} base teachers done {round(time.time()-t0,1)}s', flush=True)

        B = A[sel].copy(); B['pc_hat'] = pc
        for k in (1, 2, 3):
            B[f'ig{k}'] = (grp == k).astype('float32')
        B['stk__pcL'] = pcL; B['stk__pcG'] = pcG
        B['stk__dLG'] = pcL - pcG; B['stk__adLG'] = np.abs(pcL - pcG)
        B['stk__dFL'] = pc - pcL; B['stk__dFG'] = pc - pcG

        cls = np.clip(np.nan_to_num(cf, nan=0.0) / W, 0, NC - 1).astype(int)
        cm = tr & np.isfinite(cf) & (~(gapv >= 0.05))
        d = lgb.LGBMClassifier(**DART_CLF)
        d.fit(B[cm], cls[cm], sample_weight=w_valid[cm])
        raw = d.predict_proba(B[va])
        P = np.zeros((raw.shape[0], NC)); P[:, np.asarray(d.classes_, int)] = raw
        rows.append(pd.DataFrame({'fold_id': f, 'group_id': grp[va][keep],
                                  'forecast_kst_dtm': idx[va][keep], 'cf': cf[va][keep],
                                  'mean_gen_g': [float(np.nanmean(cf[tr & (grp == g)])) for g in grp[va][keep]]}))
        probs.append(P[keep])
        print(f'  [{tag}] {f} total {round(time.time()-t0,1)}s', flush=True)
    R = pd.concat(rows, ignore_index=True)
    Pf = np.vstack(probs)
    R.to_parquet(N + f'S7-N8_{tag}_keys.parquet', index=False)
    np.save(N + f'S7-N8_{tag}_prob.npy', Pf)
    return R, Pf


if __name__ == '__main__':
    build('DSTK')
    print('DONE', flush=True)
