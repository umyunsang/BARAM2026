"""S12-N12 * point-accuracy bake-off.  The binding constraint, isolated.

S12-N11 settled the frontier question: member D's gamma sweep moves 1-NMAE from 0.863925 to
0.864617 and FICR from 0.377378 to 0.387189 -- both rise together, so D's decision layer has NO
NMAE/FICR trade-off to exploit and 1-NMAE ~ 0.8646 is a hard ceiling set by the point forecast,
not by the action rule.  research/lanes/S12_ext_dacon_solutions.md independently puts the
public top-100's minimum 1-NMAE at 0.86777 and its median at 0.87425, with the organiser's own
RandomForest baseline at 0.86371.  We are therefore sitting at roughly baseline point accuracy
and the whole remaining gap is accuracy, not action placement.

This node isolates that one quantity.  Every variant is scored by 1-NMAE on the identical
3-fold expanding window and the identical row set; no decision layer, no blending, nothing that
can launder a point-accuracy deficit into a policy artefact.

Variants (all on the frozen G2/DROP:grid__ surface):
  L2        LightGBM squared loss on cf                        (what the teacher uses)
  L1        LightGBM absolute loss on cf                       (matches NMAE exactly)
  L1V       L1, trained ONLY on rows the metric actually scores (cf >= 0.1) -- never tested;
            NMAE ignores cf<0.1 rows entirely, so including them is an unforced objective
            mismatch
  L1W       L1 on all rows, weight 1.0 valid / 0.15 invalid    (current weighting convention)
  L1P       L1V plus the physics teacher pc_hat as a feature   (current two-stage architecture)
  HUB       L1V on the physics target pc_true, then used directly as the cf point forecast
  CBQ50     CatBoost quantile alpha=0.50
  CBQ575    CatBoost quantile alpha=0.575  (the lane reports top solutions using q~0.575-0.60,
            i.e. deliberately above the median, consistent with the high-output under-prediction
            we measured independently in S12-N2)
  XGB       XGBoost pseudo-Huber / absolute
  AVG       equal-weight mean of the above (0 dof)
"""
import sys, json, time
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/scratch')
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/nodes')
from harness import surface, MU
from lib import FOLDS, CAPS, official_total

N = '/Users/um-yunsang/BARAM2026/research/nodes/'
BASE = dict(n_estimators=900, learning_rate=0.035, num_leaves=63, min_child_samples=40,
            subsample=0.85, subsample_freq=1, colsample_bytree=0.4, reg_lambda=3.0,
            random_state=20260801, n_jobs=6, verbose=-1)


def run():
    A, FR, COLS = surface(('G2', 'DROP:grid__'))
    cf = A['cf'].to_numpy(); grp = A['grp'].to_numpy(); idx = A.index
    pct = A['pc_true'].to_numpy()
    valid = np.isfinite(cf) & (cf >= 0.1)
    w_prod = np.where(valid, np.clip(cf, 0, 1.2), 0.05)
    w_valid = np.where(valid, 1.0, 0.15)

    preds = {}
    keys = []
    for f, (a, b) in FOLDS.items():
        a = pd.Timestamp(a); b = pd.Timestamp(b)
        tr = np.asarray(idx < a); va = np.asarray((idx >= a) & (idx <= b)); keep = np.isfinite(cf[va])
        Xva = A.loc[va, COLS].iloc[keep.nonzero()[0]] if False else A[COLS][va][keep]
        keys.append(pd.DataFrame({'fold_id': f, 'group_id': grp[va][keep],
                                  'forecast_kst_dtm': idx[va][keep],
                                  'actual_kwh': cf[va][keep] * np.array([CAPS[g] for g in grp[va][keep]])}))
        fin = tr & np.isfinite(cf)
        finv = fin & valid
        fint = tr & np.isfinite(pct)

        def fit_lgb(name, obj, rows, target, weight=None):
            t0 = time.time()
            m = lgb.LGBMRegressor(objective=obj, **BASE)
            m.fit(A.loc[rows, COLS], target[rows], sample_weight=None if weight is None else weight[rows])
            p = np.clip(m.predict(Xva), 0, 1.1)
            preds.setdefault(name, []).append(p)
            print(f'  [{f}] {name:8s} {round(time.time()-t0,1)}s', flush=True)
            return m

        fit_lgb('L2', 'l2', fin, cf)
        fit_lgb('L1', 'l1', fin, cf)
        fit_lgb('L1V', 'l1', finv, cf)
        fit_lgb('L1W', 'l1', fin, cf, w_valid)
        mu = lgb.LGBMRegressor(**MU)
        mu.fit(A.loc[fint, COLS], pct[fint], sample_weight=w_prod[fint])
        pch = np.clip(mu.predict(A[COLS]), 0, 1)
        preds.setdefault('HUB', []).append(pch[va][keep])
        A2 = A[COLS].copy(); A2['pc_hat'] = pch
        t0 = time.time()
        m = lgb.LGBMRegressor(objective='l1', **BASE)
        m.fit(A2[finv], cf[finv])
        preds.setdefault('L1P', []).append(np.clip(m.predict(A2[va][keep]), 0, 1.1))
        print(f'  [{f}] {"L1P":8s} {round(time.time()-t0,1)}s', flush=True)

        from catboost import CatBoostRegressor
        for al, nm in [(0.50, 'CBQ50'), (0.575, 'CBQ575')]:
            t0 = time.time()
            c = CatBoostRegressor(loss_function=f'Quantile:alpha={al}', iterations=1200, depth=8,
                                  learning_rate=0.05, l2_leaf_reg=3.0, random_seed=20260808,
                                  verbose=0, thread_count=6)
            c.fit(A.loc[finv, COLS].to_numpy('float32'), cf[finv])
            preds.setdefault(nm, []).append(np.clip(c.predict(Xva.to_numpy('float32')), 0, 1.1))
            print(f'  [{f}] {nm:8s} {round(time.time()-t0,1)}s', flush=True)

        import xgboost as xgb
        t0 = time.time()
        xm = xgb.XGBRegressor(objective='reg:absoluteerror', n_estimators=900, max_depth=7,
                              learning_rate=0.05, subsample=0.85, colsample_bytree=0.4,
                              reg_lambda=3.0, random_state=20260808, n_jobs=6, tree_method='hist')
        xm.fit(A.loc[finv, COLS], cf[finv])
        preds.setdefault('XGB', []).append(np.clip(xm.predict(Xva), 0, 1.1))
        print(f'  [{f}] {"XGB":8s} {round(time.time()-t0,1)}s', flush=True)

    K = pd.concat(keys, ignore_index=True)
    capv = K.group_id.map(CAPS).to_numpy()
    P = {k: np.concatenate(v) for k, v in preds.items()}
    P['AVG'] = np.mean([P[k] for k in ['L1', 'L1V', 'L1W', 'L1P', 'CBQ50', 'XGB']], axis=0)
    P['AVGQ'] = np.mean([P[k] for k in ['L1V', 'L1P', 'CBQ575', 'XGB']], axis=0)
    out = {}
    print('\n--- pooled 3-fold point scores (1-NMAE is the number that matters) ---')
    for k in sorted(P):
        s = official_total(K.assign(prediction_kwh=P[k] * capv)[['group_id', 'actual_kwh', 'prediction_kwh']])
        out[k] = {'one_minus_nmae': s['one_minus_nmae'], 'ficr': s['ficr'], 'total': s['total']}
        print(f'  {k:8s} 1-NMAE={s["one_minus_nmae"]:.6f}  FICR={s["ficr"]:.6f}  total={s["total"]:.6f}')
    print(f'\n  reference: best 1-NMAE from any existing artifact = 0.864617 (member D, gamma=20)')
    np.save(N + 'S12-N12_point_preds.npy', np.vstack([P[k] for k in sorted(P)]))
    json.dump({'scores': out, 'order': sorted(P)}, open(N + 'S12-N12_point_bakeoff.json', 'w'), indent=1)
    K.to_parquet(N + 'S12-N12_keys.parquet', index=False)


if __name__ == '__main__':
    run()
