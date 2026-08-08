
"""Fast feature-surface sweep measured directly in the metric's own units:
MAE of the curve-integrated capacity factor `pc`."""
import sys, json, time, numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0,'/Users/um-yunsang/BARAM2026/research/scratch')
import featbuild
S='/Users/um-yunsang/BARAM2026/research/scratch/'
T=pd.read_parquet(S+'teacher_targets.parquet')
SPLIT=pd.Timestamp('2023-04-01 01:00:00'); END=pd.Timestamp('2024-01-01 00:00:00')
P=dict(objective='l2', n_estimators=900, learning_rate=0.035, num_leaves=63,
       min_child_samples=40, subsample=0.85, subsample_freq=1, colsample_bytree=0.4,
       reg_lambda=3.0, random_state=20260801, n_jobs=6, verbose=-1)

def run(tag, geom, grid):
    mae=[]; rms=[]
    for g in (1,2,3):
        X=featbuild.build2(g, geom=geom, grid=grid)
        y=T[f'g{g}_pc'].reindex(X.index)
        tr=np.asarray(X.index<SPLIT); va=np.asarray((X.index>=SPLIT)&(X.index<=END))
        m=y[tr].notna().to_numpy()
        t0=time.time(); mdl=lgb.LGBMRegressor(**P); mdl.fit(X[tr][m], y[tr][m])
        p=np.clip(mdl.predict(X[va]),0,1); yt=y[va].to_numpy(); ok=np.isfinite(yt)
        mae.append(float(np.abs(p[ok]-yt[ok]).mean())); rms.append(float(np.sqrt(((p[ok]-yt[ok])**2).mean())))
        print(f'  {tag} g{g} mae={mae[-1]:.5f} rmse={rms[-1]:.5f} n={X.shape[1]} {round(time.time()-t0,1)}s', flush=True)
        pd.Series(mdl.feature_importances_, index=X.columns).sort_values(ascending=False)\
          .head(80).to_csv(S+f'sweep_imp_{tag}_g{g}.csv')
        del X, mdl
    print(f'{tag:16s} MEAN pc-MAE={np.mean(mae):.5f}  MEAN pc-RMSE={np.mean(rms):.5f}', flush=True)
    return float(np.mean(mae))

if __name__=='__main__':
    out={}
    out['base']=run('base', False, False)
    out['geom']=run('geom', True, False)
    out['grid']=run('grid', False, True)
    out['both']=run('both', True, True)
    json.dump(out, open(S+'sweep_pc.json','w'), indent=1); print(out)
