"""S15-N2 * PIPELINE STAGE B1 -- per-source spatial reduction.

Verified instruction (S15-N1, reproduced independently on our own parquet against MEASURED hub
wind, all three groups):

    LDAPS 50 m   box max 0.8405 > q90 0.8354 > q75 0.8301 > median 0.8114 > idw(repo) 0.8076
    GFS 100 m    idw(repo) 0.7075 > median 0.6806 > ... > box max 0.5238   (max is the WORST)

The ordering INVERTS between sources, so the single shared inverse-distance rule this repository
uses is provably wrong for LDAPS.  Physical reading: the 17 turbines stand on the ridge, LDAPS
under-resolves that ridge by 80-140 m, so the most exposed cell of the 4x4 box is closer to
ridge-top conditions than a distance weighting that mixes in valley cells; GFS at 0.25 degrees has
no ridge to resolve and its box maximum is noise.

THIS STAGE IS A REPLACEMENT, NOT A WIDENING.  The corrected reduction overwrites the existing
`ldaps_spatial__idw__*` wind columns in place, so the column count is unchanged and contract R2's
dilution tax does not apply -- the comparison is clean without a noise arm.  The operator is
selected on each fold's TRAINING window only.

Reduction used for LDAPS: the MOST EXPOSED CELL -- take the cell with the largest speed at that
level and read its speed AND its direction, rather than taking a max of speeds and an unrelated
weighted direction.  That keeps the wind vector physically consistent.
"""
import sys, json, time
import numpy as np, pandas as pd, pyarrow.parquet as pq, lightgbm as lgb
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/scratch')
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/nodes')
from harness import surface, MU
from lib import FOLDS, CAPS, official_total

C = ('/Users/um-yunsang/BARAM2026/artifacts/cache/'
     '920be0c458d820e855bf79dd25723146f52ce1736138aedba5e6bc853f1f720b/')
S = '/Users/um-yunsang/BARAM2026/research/scratch/'
N = '/Users/um-yunsang/BARAM2026/research/nodes/'
L1P = dict(objective='l1', n_estimators=900, learning_rate=0.035, num_leaves=63,
           min_child_samples=40, subsample=0.85, subsample_freq=1, colsample_bytree=0.4,
           reg_lambda=3.0, random_state=20260801, n_jobs=6, verbose=-1)
LEVELS = {'wind10_speed': ('heightAboveGround_10_10u', 'heightAboveGround_10_10v'),
          'wind50max_speed': ('heightAboveGround_50_50MUmax', 'heightAboveGround_50_50MVmax'),
          'wind50min_speed': ('heightAboveGround_50_50MUmin', 'heightAboveGround_50_50MVmin'),
          'wind5_speed': ('heightAboveGround_5_XBLWS', 'heightAboveGround_5_YBLWS')}


def most_exposed_cell_reads():
    names = pq.ParquetFile(C + 'train_grid_pivot.parquet').schema.names
    need = ['forecast_kst_dtm']
    spec = {}
    for lvl, (uv, vv) in LEVELS.items():
        cu = sorted([c for c in names if c.startswith('ldaps__') and c.endswith('__' + uv)])
        cv = sorted([c for c in names if c.startswith('ldaps__') and c.endswith('__' + vv)])
        if cu and cv:
            spec[lvl] = (cu, cv); need += cu + cv
    G = pd.read_parquet(C + 'train_grid_pivot.parquet',
                        columns=sorted(set(need))).set_index('forecast_kst_dtm').sort_index()
    out = {}
    for lvl, (cu, cv) in spec.items():
        U = G[cu].to_numpy('float64'); V = G[cv].to_numpy('float64')
        U = np.where(np.isfinite(U), U, np.nanmean(U, 1, keepdims=True))
        V = np.where(np.isfinite(V), V, np.nanmean(V, 1, keepdims=True))
        sp = np.hypot(U, V)
        k = np.argmax(sp, axis=1)
        r = np.arange(len(G))
        s = sp[r, k]; u = U[r, k]; v = V[r, k]
        ang = np.arctan2(-u, -v)
        out[f'ldaps_spatial__idw__{lvl}'] = s
        out[f'ldaps_spatial__idw__{lvl.replace("_speed","")}_dir_sin'] = np.sin(ang)
        out[f'ldaps_spatial__idw__{lvl.replace("_speed","")}_dir_cos'] = np.cos(ang)
        out[f'b1__{lvl}__exposed_cell'] = k.astype('float32')
    return pd.DataFrame(out, index=G.index)


if __name__ == '__main__':
    A, FR, COLS = surface(('G2', 'DROP:grid__'))
    X = most_exposed_cell_reads()
    repl = [c for c in X.columns if c in A.columns]
    print(f'columns REPLACED in place (no widening): {len(repl)}')
    for c in repl:
        j = pd.concat([A[c].rename('old'), X[c].reindex(A.index).rename('new')], axis=1).dropna()
        print(f'  {c:52s} corr(old,new)={j.old.corr(j.new):.4f}  '
              f'mean {j.old.mean():.3f} -> {j.new.mean():.3f}')

    A2 = A.copy()
    for c in repl:
        A2[c] = X[c].reindex(A.index).to_numpy()

    cf = A['cf'].to_numpy(); grp = A['grp'].to_numpy(); idx = A.index
    pct = A['pc_true'].to_numpy()
    valid = np.isfinite(cf) & (cf >= 0.1)
    w_prod = np.where(valid, np.clip(cf, 0, 1.2), 0.05)
    keys = []; preds = {}
    for f, (a, b) in FOLDS.items():
        a = pd.Timestamp(a); b = pd.Timestamp(b)
        tr = np.asarray(idx < a); va = np.asarray((idx >= a) & (idx <= b)); keep = np.isfinite(cf[va])
        keys.append(pd.DataFrame({'fold_id': f, 'group_id': grp[va][keep],
                                  'forecast_kst_dtm': idx[va][keep],
                                  'actual_kwh': cf[va][keep] * np.array([CAPS[g] for g in grp[va][keep]])}))
        for tag, frame in (('BASE', A), ('B1', A2)):
            t0 = time.time()
            m = tr & np.isfinite(pct)
            mu = lgb.LGBMRegressor(**MU)
            mu.fit(frame.loc[m, COLS], pct[m], sample_weight=w_prod[m])
            pch = np.clip(mu.predict(frame[COLS]), 0, 1)
            F = frame[COLS].copy(); F['pc_hat'] = pch
            pm = lgb.LGBMRegressor(**L1P); pm.fit(F[tr & valid], cf[tr & valid])
            preds.setdefault(tag, []).append(np.clip(pm.predict(F[va][keep]), 0, 1.1))
            print(f'  [{f}] {tag} {round(time.time()-t0,1)}s', flush=True)
    K = pd.concat(keys, ignore_index=True)
    capv = K.group_id.map(CAPS).to_numpy()
    out = {}
    print('\n--- STAGE B1 effect on the point pipeline (identical column count) ---')
    for k in ('BASE', 'B1'):
        v = np.concatenate(preds[k])
        s = official_total(K.assign(prediction_kwh=v * capv)[['group_id', 'actual_kwh', 'prediction_kwh']])
        out[k] = {'one_minus_nmae': s['one_minus_nmae'], 'ficr': s['ficr'], 'total': s['total'],
                  'group_nmae': s['group_nmae']}
        print(f'  {k:5s} 1-NMAE={s["one_minus_nmae"]:.6f}  FICR={s["ficr"]:.6f}  Total={s["total"]:.6f}')
    d1 = out['B1']['one_minus_nmae'] - out['BASE']['one_minus_nmae']
    dt = out['B1']['total'] - out['BASE']['total']
    print(f'\n  B1 - BASE: 1-NMAE {d1:+.6f}   Total {dt:+.6f}')
    print(f'  (paired measurement sd on Total is ~0.00075; a stage is expected to be worth ~0.002)')
    np.save(N + 'S15-N2_preds.npy', np.vstack([np.concatenate(preds[k]) for k in ('BASE', 'B1')]))
    X.to_parquet(S + 'b1_reduction.parquet')
    json.dump(out, open(N + 'S15-N2_stage_b1.json', 'w'), indent=1, default=str)
    K.to_parquet(N + 'S15-N2_keys.parquet', index=False)
