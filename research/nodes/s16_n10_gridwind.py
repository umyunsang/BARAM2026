"""S16-N10 * attack the dominant error channel with the field we throw away.

Where the error actually is.  S13's decomposition put 0.13022 of the scored-row MAE 0.13858 in the
NWP->hub-wind channel and only 0.04804 in label/availability; with perfect wind the metric reaches
0.869922.  Every decision-layer construction is now closed -- selection (N3/N4/N6), complementarity
(N7), recentring (N8), representation (N9) -- and the single largest real effect this project has
ever measured lives in that channel: B2's supervised hub-wind transfer, +0.004214 by reverse
ablation, nearly three times the seed floor.

And the feature surface DROPS the raw field.  `surface(('G2','DROP:grid__'))` discards
train_grid_pivot's 914 columns, leaving only hand-built reductions -- idw, max, q90, per-source
spatial summaries.  S15-N1 already showed those reductions are not interchangeable and that their
ranking INVERTS between sources (LDAPS: max 0.8405 > q90 0.8354 > idw 0.8076; GFS: idw 0.7075 >
... > max 0.5238), which is direct evidence that a fixed hand reduction is throwing information
away -- the right combination is source-dependent and we guessed it.

So this node asks the cheap version of the question first.  Not "does it raise Total" -- that costs
half an hour and lands inside the seed floor -- but "does the raw field predict MEASURED hub wind
better than our reductions do?"  scada_vestas / scada_unison carry hub-height ws on 26,304 hourly
joins, so this is a clean supervised problem with a real target, evaluated by MAE and by the
correlation that B2's downstream gain actually rode on.  Only if the field wins here is the full
pipeline worth refitting.

Three arms, fold-outside, three seeds:
    REDUCED  the current surface's wind columns -- the incumbent and the plumbing control
    GRID     the 914-column raw pivot alone
    BOTH     the union
"""
import sys, json, time
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/scratch')
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/nodes')
from harness import surface
from lib import FOLDS

S = '/Users/um-yunsang/BARAM2026/research/scratch/'
CACHE = ('/Users/um-yunsang/BARAM2026/artifacts/cache/'
         '920be0c458d820e855bf79dd25723146f52ce1736138aedba5e6bc853f1f720b/')
N = '/Users/um-yunsang/BARAM2026/research/nodes/'
SEEDS = (20260803, 20260804, 20260805)
REG = dict(objective='l1', n_estimators=600, learning_rate=0.04, num_leaves=63,
           min_child_samples=40, colsample_bytree=0.5, subsample=0.8, subsample_freq=1,
           reg_lambda=5.0, n_jobs=6, verbose=-1)

if __name__ == '__main__':
    # the pivot carries its timestamp as a COLUMN, not an index
    G = pd.read_parquet(CACHE + 'train_grid_pivot.parquet')
    G = G.set_index(pd.to_datetime(G['forecast_kst_dtm'])).drop(columns=['forecast_kst_dtm'])
    G.index.name = None
    print(f'grid pivot {G.shape}  {G.index.min()} -> {G.index.max()}')
    # SCADA is 10-MINUTE data (157,819 rows over three years); the target is the hourly mean of
    # hub-height ws across the six Vestas turbines, which is what the forecast hour settles on.
    dV = pd.read_parquet(S + 'scada_vestas.parquet')
    wcV = [c for c in dV.columns if c.endswith('_ws')]
    hub = (dV.set_index(pd.to_datetime(dV['kst_dtm']))[wcV].mean(axis=1)
             .resample('1h').mean().rename('hub_ws').dropna())
    print(f'measured hub wind: {len(wcV)} turbines, 10-min -> {len(hub)} hourly values, '
          f'{hub.index.min()} -> {hub.index.max()}, mean {hub.mean():.3f} m/s')

    A, FR, COLS = surface(('G2', 'DROP:grid__'))
    A1 = A[A['grp'] == 1]
    red = [c for c in COLS if any(k in c for k in ('wind', 'ws', 'spatial', 'atm__'))]
    X_red = A1[red]
    X_red.index = pd.to_datetime(X_red.index)
    Xg = G.reindex(X_red.index)
    keep = Xg.notna().mean() > 0.9
    Xg = Xg.loc[:, keep].astype('float32')
    y = hub.reindex(X_red.index)
    ok = y.notna().to_numpy() & np.isfinite(Xg.to_numpy()).any(1)
    print(f'joined rows {int(ok.sum())} of {len(X_red)}   reduced cols {len(red)}   '
          f'grid cols kept {Xg.shape[1]} of {G.shape[1]}')

    ARMS = {'REDUCED': X_red, 'GRID': Xg,
            'BOTH': pd.concat([X_red.reset_index(drop=True),
                               Xg.reset_index(drop=True)], axis=1).set_index(X_red.index)}
    out = {}
    for nm, X in ARMS.items():
        maes, cors = [], []
        for f, (a_, b_) in FOLDS.items():
            a_ = pd.Timestamp(a_); b_ = pd.Timestamp(b_)
            tr = (X.index < a_) & ok
            va = (X.index >= a_) & (X.index <= b_) & ok
            if tr.sum() < 500 or va.sum() < 100:
                continue
            for sd in SEEDS:
                m = lgb.LGBMRegressor(**dict(REG, random_state=sd))
                m.fit(X[tr], y[tr])
                p = m.predict(X[va])
                maes.append(float(np.mean(np.abs(p - y[va]))))
                cors.append(float(np.corrcoef(p, y[va])[0, 1]))
        out[nm] = dict(mae=float(np.mean(maes)), mae_sd=float(np.std(maes, ddof=1)),
                       corr=float(np.mean(cors)), n_cols=int(X.shape[1]))
        print(f'  {nm:8s} cols={X.shape[1]:4d}  hub-wind MAE={np.mean(maes):.4f} '
              f'(sd {np.std(maes, ddof=1):.4f})  corr={np.mean(cors):.4f}')
    b = min(out, key=lambda k: out[k]['mae'])
    imp = (out['REDUCED']['mae'] - out[b]['mae']) / out['REDUCED']['mae']
    print(f'\n  best {b}: {imp:+.2%} hub-wind MAE vs the incumbent reduction')
    print('  S13 attributes 0.13022 of 0.13858 scored MAE to this channel; a k% cut here is')
    print('  worth roughly 0.94*k% of the point error, before any decision-layer loss.')
    json.dump(out, open(N + 'S16-N10_gridwind.json', 'w'), indent=1, default=str)
