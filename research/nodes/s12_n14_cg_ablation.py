"""S12-N14 * marginal value of the component-grid block on POINT accuracy.

S12-N11 established that 1-NMAE is the binding constraint (D's gamma sweep raises 1-NMAE and
FICR together, so there is no action-placement trade left to harvest) and S12-N12 found the
best point configuration available from the current surface:

    L1  on all rows                     1-NMAE 0.857589
    L1V on scored rows only (cf>=0.1)   1-NMAE 0.863750   (+0.006161 -- the metric ignores
                                                            cf<0.1 rows, so training on them
                                                            was an unforced objective mismatch)
    L1P = L1V + physics teacher pc_hat  1-NMAE 0.865946
    AVGQ (L1V,L1P,CBQ575,XGB)           1-NMAE 0.866147   <- previous best of any artifact:
                                                            0.864617 (member D at gamma=20)

This node asks one question with one treatment: does the S12-N13 component-grid block move
that number?  Control and treatment share fold structure, row set, loss, weights, seed and
hyper-parameters; the only difference is 168 extra columns.

Reported quantity is 1-NMAE, pooled over the three folds, with the group decomposition, plus
the LightGBM gain-importance mass that lands on the new block (a block that earns no
importance cannot be responsible for a score move).
"""
import sys, json, time
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/scratch')
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/nodes')
from harness import surface, MU
from lib import FOLDS, CAPS, official_total

N = '/Users/um-yunsang/BARAM2026/research/nodes/'
CG = '/Users/um-yunsang/BARAM2026/research/scratch/component_grid.parquet'
BASE = dict(n_estimators=900, learning_rate=0.035, num_leaves=63, min_child_samples=40,
            subsample=0.85, subsample_freq=1, colsample_bytree=0.4, reg_lambda=3.0,
            random_state=20260801, n_jobs=6, verbose=-1)


def run():
    A, FR, COLS = surface(('G2', 'DROP:grid__'))
    X = pd.read_parquet(CG)
    cg_cols = list(X.columns)
    A2 = A.copy()
    for c in cg_cols:
        A2[c] = X[c].reindex(A.index).to_numpy()
    COLS_CG = COLS + cg_cols

    cf = A['cf'].to_numpy(); grp = A['grp'].to_numpy(); idx = A.index
    pct = A['pc_true'].to_numpy()
    valid = np.isfinite(cf) & (cf >= 0.1)
    w_prod = np.where(valid, np.clip(cf, 0, 1.2), 0.05)

    keys = []; preds = {}; imp_share = []
    for f, (a, b) in FOLDS.items():
        a = pd.Timestamp(a); b = pd.Timestamp(b)
        tr = np.asarray(idx < a); va = np.asarray((idx >= a) & (idx <= b)); keep = np.isfinite(cf[va])
        keys.append(pd.DataFrame({'fold_id': f, 'group_id': grp[va][keep],
                                  'forecast_kst_dtm': idx[va][keep],
                                  'actual_kwh': cf[va][keep] * np.array([CAPS[g] for g in grp[va][keep]])}))
        finv = tr & valid
        fint = tr & np.isfinite(pct)
        for tag, cols, frame in [('BASE', COLS, A), ('CG', COLS_CG, A2)]:
            t0 = time.time()
            mu = lgb.LGBMRegressor(**MU)
            mu.fit(frame.loc[fint, cols], pct[fint], sample_weight=w_prod[fint])
            pch = np.clip(mu.predict(frame[cols]), 0, 1)
            F = frame[cols].copy(); F['pc_hat'] = pch
            m = lgb.LGBMRegressor(objective='l1', **BASE)
            m.fit(F[finv], cf[finv])
            preds.setdefault(f'L1P_{tag}', []).append(np.clip(m.predict(F[va][keep]), 0, 1.1))
            if tag == 'CG':
                gi = pd.Series(m.booster_.feature_importance('gain'), index=F.columns)
                imp_share.append(float(gi[cg_cols].sum() / gi.sum()))
                tgi = pd.Series(mu.booster_.feature_importance('gain'), index=cols)
                print(f'    teacher gain share on new block: {tgi[cg_cols].sum()/tgi.sum():.4f}; '
                      f'top new: {list(tgi[cg_cols].sort_values(ascending=False).head(5).index)}', flush=True)
            print(f'  [{f}] L1P_{tag} {round(time.time()-t0,1)}s', flush=True)

    K = pd.concat(keys, ignore_index=True)
    capv = K.group_id.map(CAPS).to_numpy()
    out = {}
    print('\n--- pooled 3-fold point scores ---')
    for k in sorted(preds):
        v = np.concatenate(preds[k])
        s = official_total(K.assign(prediction_kwh=v * capv)[['group_id', 'actual_kwh', 'prediction_kwh']])
        out[k] = {'one_minus_nmae': s['one_minus_nmae'], 'ficr': s['ficr'], 'total': s['total'],
                  'group_nmae': s['group_nmae']}
        print(f'  {k:10s} 1-NMAE={s["one_minus_nmae"]:.6f}  FICR={s["ficr"]:.6f}  '
              f'group NMAE={ {g: round(x,5) for g,x in s["group_nmae"].items()} }')
    d = out['L1P_CG']['one_minus_nmae'] - out['L1P_BASE']['one_minus_nmae']
    print(f'\n  component-grid delta on 1-NMAE: {d:+.6f}')
    print(f'  classifier gain share on the new block per fold: {[round(x,4) for x in imp_share]}')
    out['delta_1mnmae'] = d
    out['cg_gain_share'] = imp_share
    np.save(N + 'S12-N14_preds.npy', np.vstack([np.concatenate(preds[k]) for k in sorted(preds)]))
    json.dump(out, open(N + 'S12-N14_cg_ablation.json', 'w'), indent=1, default=str)


if __name__ == '__main__':
    run()
