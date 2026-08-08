
import sys, json, time, numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0,'/Users/um-yunsang/BARAM2026/research/scratch')
import featbuild

SPLIT=pd.Timestamp('2023-04-01 01:00:00'); END=pd.Timestamp('2024-01-01 00:00:00')
D = pd.read_parquet('/Users/um-yunsang/BARAM2026/research/scratch/scada_group_aggs.parquet')

PARAMS=dict(n_estimators=2000, learning_rate=0.025, num_leaves=63, min_child_samples=40,
            subsample=0.85, subsample_freq=1, colsample_bytree=0.5, reg_lambda=3.0,
            random_state=20260801, n_jobs=6, verbose=-1)

def variant_cols(X, kind):
    if kind=='base':   return [c for c in X.columns if not c.startswith('atm__') and '__lag' not in c and '__d-' not in c and not c.endswith(('__d1','__d2','__d3')) and '__batch_' not in c]
    if kind=='atm':    return [c for c in X.columns if '__lag' not in c and '__d-' not in c and not c.endswith(('__d1','__d2','__d3')) and '__batch_' not in c]
    if kind=='full':   return list(X.columns)
    raise ValueError(kind)

def run(kind):
    res={}
    for g in (1,2,3):
        X = featbuild.build(g)
        cols = variant_cols(X, kind)
        F = X[cols]
        y = D[f'g{g}_ws_mean'].reindex(F.index)
        tr = F.index<SPLIT; va=(F.index>=SPLIT)&(F.index<=END)
        ytr=y[tr]; m=ytr.notna().to_numpy()
        t0=time.time()
        mdl=lgb.LGBMRegressor(**PARAMS); mdl.fit(F[tr][m], ytr[m])
        p=mdl.predict(F[va]); yv=y[va].to_numpy(); ok=np.isfinite(yv)
        e=p[ok]-yv[ok]
        res[g]=dict(rmse=float(np.sqrt((e**2).mean())), mae=float(np.abs(e).mean()),
                    nfeat=len(cols), secs=round(time.time()-t0,1))
        print(f'{kind} g{g}: rmse={res[g]["rmse"]:.4f} mae={res[g]["mae"]:.4f} n={len(cols)} {res[g]["secs"]}s', flush=True)
        pd.DataFrame({'kst_dtm':F.index[va],'pred':p,'true':yv}).to_parquet(
            f'/Users/um-yunsang/BARAM2026/research/scratch/A2_{kind}_g{g}.parquet', index=False)
        if kind=='full':
            imp=pd.Series(mdl.feature_importances_, index=cols).sort_values(ascending=False)
            imp.head(60).to_csv(f'/Users/um-yunsang/BARAM2026/research/scratch/A2_imp_g{g}.csv')
        del X, F, mdl
    mr=float(np.mean([res[g]['rmse'] for g in res]))
    print(f'{kind}: MEAN RMSE={mr:.4f}', flush=True)
    return res, mr

if __name__=='__main__':
    out={}
    for kind in sys.argv[1:]:
        out[kind]=run(kind)
    json.dump(out, open('/Users/um-yunsang/BARAM2026/research/scratch/A2_'+'_'.join(sys.argv[1:])+'.json','w'), indent=1)
