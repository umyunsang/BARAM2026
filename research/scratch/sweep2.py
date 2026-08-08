
"""S7 point-skill sweep, measured in the metric's own unit: MAE of curve-integrated cf.
Fast protocol: train < 2023-04-01, validate 2023-04-01..2023-12-31.
Variants test the two untested information levers plus data pooling and estimator bagging."""
import sys, json, time, numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0,'/Users/um-yunsang/BARAM2026/research/scratch')
import featbuild
S='/Users/um-yunsang/BARAM2026/research/scratch/'
T=pd.read_parquet(S+'teacher_targets.parquet')
SPLIT=pd.Timestamp('2023-04-01 01:00:00'); END=pd.Timestamp('2024-01-01 00:00:00')
BASE=dict(objective='l2', n_estimators=900, learning_rate=0.035, num_leaves=63,
          min_child_samples=40, subsample=0.85, subsample_freq=1, colsample_bytree=0.4,
          reg_lambda=3.0, random_state=20260801, n_jobs=6, verbose=-1)

def cache_X(geom, grid):
    return {g: featbuild.build2(g, geom=geom, grid=grid) for g in (1,2,3)}

def score(preds, truths):
    mae=[float(np.abs(p[np.isfinite(t)]-t[np.isfinite(t)]).mean()) for p,t in zip(preds,truths)]
    return mae

def run_single(Xs, tag, params_list=(BASE,)):
    maes=[]
    for g in (1,2,3):
        X=Xs[g]; y=T[f'g{g}_pc'].reindex(X.index)
        tr=np.asarray(X.index<SPLIT); va=np.asarray((X.index>=SPLIT)&(X.index<=END))
        m=y[tr].notna().to_numpy()
        t0=time.time(); acc=[]
        for pr in params_list:
            mdl=lgb.LGBMRegressor(**pr); mdl.fit(X[tr][m], y[tr][m])
            acc.append(np.clip(mdl.predict(X[va]),0,1))
        p=np.mean(acc,axis=0); yt=y[va].to_numpy(); ok=np.isfinite(yt)
        maes.append(float(np.abs(p[ok]-yt[ok]).mean()))
        print(f'  {tag} g{g} mae={maes[-1]:.5f} n={X.shape[1]} {round(time.time()-t0,1)}s', flush=True)
    print(f'{tag:18s} MEAN pc-MAE={np.mean(maes):.5f}', flush=True)
    return float(np.mean(maes))

def run_pooled(Xs, tag, params=BASE):
    frames=[]
    for g in (1,2,3):
        X=Xs[g].copy()
        X['grp']=g
        for k in (1,2,3): X[f'is_g{k}']=float(k==g)
        X['y']=T[f'g{g}_pc'].reindex(X.index).to_numpy()
        frames.append(X)
    A=pd.concat(frames)
    tr=np.asarray(A.index<SPLIT)
    cols=[c for c in A.columns if c not in ('y','grp')]
    m=np.isfinite(A['y'].to_numpy())&tr
    t0=time.time(); mdl=lgb.LGBMRegressor(**params); mdl.fit(A.loc[m,cols], A.loc[m,'y'])
    print(f'  {tag} pooled fit rows={int(m.sum())} {round(time.time()-t0,1)}s', flush=True)
    maes=[]
    for g in (1,2,3):
        X=frames[g-1]; va=np.asarray((X.index>=SPLIT)&(X.index<=END))
        p=np.clip(mdl.predict(X.loc[va,cols]),0,1); yt=X.loc[va,'y'].to_numpy(); ok=np.isfinite(yt)
        maes.append(float(np.abs(p[ok]-yt[ok]).mean()))
        print(f'  {tag} g{g} mae={maes[-1]:.5f}', flush=True)
    print(f'{tag:18s} MEAN pc-MAE={np.mean(maes):.5f}', flush=True)
    return float(np.mean(maes))

if __name__=='__main__':
    out={}
    Xb=cache_X(False,False)
    out['pooled_base']=run_pooled(Xb,'pooled_base')
    del Xb
    Xg=cache_X(True,True)
    out['both']=run_single(Xg,'both')
    out['pooled_both']=run_pooled(Xg,'pooled_both')
    json.dump(out, open(S+'sweep2.json','w'), indent=1); print(out, flush=True)
