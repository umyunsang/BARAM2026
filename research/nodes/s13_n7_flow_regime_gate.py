"""S13-N7 * screen S1 then the fold-outside gate for the S13-N6 flow-regime block.

Screen S1 (imposed by research/lanes/S13_S6_features_deep.md, and by S12-N14 where 168
informative-but-redundant columns took 6-11% of gain and LOST 0.000728 of 1-NMAE): if every new
column is >= 0.85 correlated with an existing stability/shear/regime column, the block is a
repackaging of what the surface already has and the gate is not spent.

Gate: 1-NMAE of the best point configuration (S12-N12's L1P: LightGBM absolute loss on the
scored rows with the physics teacher as an input), identical folds, rows, loss, seed and
hyper-parameters; the only change is the added columns.  1-NMAE is the reported quantity
because S12-N11 established it as the binding constraint.
"""
import sys, json, time
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/scratch')
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/nodes')
from harness import surface, MU
from lib import FOLDS, CAPS, official_total

N = '/Users/um-yunsang/BARAM2026/research/nodes/'
FR = '/Users/um-yunsang/BARAM2026/research/scratch/flow_regime.parquet'
L1P = dict(objective='l1', n_estimators=900, learning_rate=0.035, num_leaves=63,
           min_child_samples=40, subsample=0.85, subsample_freq=1, colsample_bytree=0.4,
           reg_lambda=3.0, random_state=20260801, n_jobs=6, verbose=-1)

INCUMBENT_REF = ['atm__alpha_100_80', 'atm__alpha_80_10', 'atm__alpha_50_10',
                 'atm__bulk_richardson_proxy', 'atm__theta850_minus_t2',
                 'atm__theta700_minus_theta850', 'atm__theta500_minus_theta700',
                 'atm__lapse_2m_850', 'atm__blh_norm', 'atm__blh_below_hub',
                 'atm__w100_w10_ratio', 'atm__w80_w10_ratio', 'atm__pbl_w10_ratio',
                 'atm__gust_factor', 'atm__hub_consensus', 'atm__hub_disagree',
                 'ldaps_spatial__idw__etc_0_blh', 'ldaps_spatial__idw__wind50max_speed',
                 'gfs_spatial__idw__wind100_speed', 'ldaps_spatial__idw__heightAboveGround_2_t',
                 'ldaps__meanSea_0_prmsl__mean', 'ldaps__meanSea_0_prmsl__std',
                 'ldaps__surface_0_sp__mean', 'ldaps__surface_0_h__mean',
                 'geom__ldaps__wind50max__layout_along', 'geom__ldaps__wind50max__layout_cross',
                 'g2__l50x__slope_along', 'g2__l50x__slope_cross']


def per_group_columns(X, g):
    """The block is group-indexed: give each group its own columns under a common name."""
    out = {}
    for c in X.columns:
        if c.startswith('fr__g'):
            gg = int(c[5])
            if gg != g:
                continue
            out[c.replace(f'fr__g{g}__', 'fr__grp__')] = X[c]
        else:
            out[c] = X[c]
    return pd.DataFrame(out, index=X.index)


def run():
    A, FRM, COLS = surface(('G2', 'DROP:grid__'))
    X = pd.read_parquet(FR)
    grp = A['grp'].to_numpy(); idx = A.index
    cf = A['cf'].to_numpy(); pct = A['pc_true'].to_numpy()
    valid = np.isfinite(cf) & (cf >= 0.1)
    w_prod = np.where(valid, np.clip(cf, 0, 1.2), 0.05)

    # assemble the group-indexed block aligned to A's stacked (group, time) index
    blocks = []
    for g in (1, 2, 3):
        m = grp == g
        Xi = per_group_columns(X, g).reindex(idx[m])
        Xi.index = np.where(m)[0]
        blocks.append(Xi)
    NEW = pd.concat(blocks).sort_index()
    NEW.index = A.index
    new_cols = list(NEW.columns)
    print(f'new block: {len(new_cols)} columns')

    # ---------- screen S1 ----------------------------------------------------------------
    ref = [c for c in INCUMBENT_REF if c in A.columns]
    R = pd.concat([NEW.reset_index(drop=True), A[ref].reset_index(drop=True)], axis=1)
    C = R.corr().loc[new_cols, ref].abs()
    mx = C.max(axis=1)
    passed = mx[mx < 0.85].index.tolist()
    print(f'\n--- screen S1: |corr| against {len(ref)} incumbent stability/shear/geometry columns ---')
    print(f'  columns with max|corr| < 0.85 : {len(passed)} / {len(new_cols)}')
    print('  most redundant (top 8):')
    for c, v in mx.sort_values(ascending=False).head(8).items():
        print(f'    {c:34s} {v:.3f}  vs {C.loc[c].idxmax()}')
    print('  most novel (top 12):')
    for c, v in mx.sort_values().head(12).items():
        print(f'    {c:34s} {v:.3f}')
    if len(passed) < 5:
        print('\nSCREEN S1 FAILED -- block is a repackaging. Gate NOT spent.')
        json.dump({'screen': 'FAIL', 'n_pass': len(passed)},
                  open(N + 'S13-N7_flow_regime_gate.json', 'w'), indent=1)
        return

    # ---------- gate --------------------------------------------------------------------
    A2 = A.copy()
    for c in passed:
        A2[c] = NEW[c].to_numpy()
    COLS2 = COLS + passed
    keys = []; preds = {}
    for f, (a, b) in FOLDS.items():
        a = pd.Timestamp(a); b = pd.Timestamp(b)
        tr = np.asarray(idx < a); va = np.asarray((idx >= a) & (idx <= b)); keep = np.isfinite(cf[va])
        keys.append(pd.DataFrame({'fold_id': f, 'group_id': grp[va][keep],
                                  'forecast_kst_dtm': idx[va][keep],
                                  'actual_kwh': cf[va][keep] * np.array([CAPS[g] for g in grp[va][keep]])}))
        for tag, cols, frame in [('BASE', COLS, A), ('FLOW', COLS2, A2)]:
            t0 = time.time()
            m = tr & np.isfinite(pct)
            mu = lgb.LGBMRegressor(**MU)
            mu.fit(frame.loc[m, cols], pct[m], sample_weight=w_prod[m])
            pch = np.clip(mu.predict(frame[cols]), 0, 1)
            F = frame[cols].copy(); F['pc_hat'] = pch
            pm = lgb.LGBMRegressor(**L1P)
            pm.fit(F[tr & valid], cf[tr & valid])
            preds.setdefault(tag, []).append(np.clip(pm.predict(F[va][keep]), 0, 1.1))
            if tag == 'FLOW':
                gi = pd.Series(pm.booster_.feature_importance('gain'), index=F.columns)
                top = gi[passed].sort_values(ascending=False).head(5)
                print(f'    new-block gain share {gi[passed].sum()/gi.sum():.4f}; top: {list(top.index)}',
                      flush=True)
            print(f'  [{f}] {tag} {round(time.time()-t0,1)}s', flush=True)

    K = pd.concat(keys, ignore_index=True)
    capv = K.group_id.map(CAPS).to_numpy()
    out = {'screen': 'PASS', 'n_pass': len(passed), 'passed_cols': passed, 'scores': {}}
    print('\n--- gate: pooled 3-fold point scores ---')
    for k in ['BASE', 'FLOW']:
        v = np.concatenate(preds[k])
        s = official_total(K.assign(prediction_kwh=v * capv)[['group_id', 'actual_kwh', 'prediction_kwh']])
        out['scores'][k] = {'one_minus_nmae': s['one_minus_nmae'], 'ficr': s['ficr'],
                            'total': s['total'], 'group_nmae': s['group_nmae']}
        print(f'  {k:5s} 1-NMAE={s["one_minus_nmae"]:.6f}  FICR={s["ficr"]:.6f}  Total={s["total"]:.6f}  '
              f'gNMAE={ {g: round(x,5) for g,x in s["group_nmae"].items()} }')
    d = out['scores']['FLOW']['one_minus_nmae'] - out['scores']['BASE']['one_minus_nmae']
    print(f'\n  flow-regime delta on 1-NMAE: {d:+.6f}')
    out['delta_1mnmae'] = d
    np.save(N + 'S13-N7_preds.npy', np.vstack([np.concatenate(preds[k]) for k in ['BASE', 'FLOW']]))
    json.dump(out, open(N + 'S13-N7_flow_regime_gate.json', 'w'), indent=1, default=str)
    K.to_parquet(N + 'S13-N7_keys.parquet', index=False)


if __name__ == '__main__':
    run()
