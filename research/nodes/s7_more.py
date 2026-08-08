
"""S7-N10 · more diverse members for the pool: DART boosting, ExtraTrees, a second MLP."""
import sys, json, time
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
sys.path.insert(0,'/Users/um-yunsang/BARAM2026/research/scratch')
sys.path.insert(0,'/Users/um-yunsang/BARAM2026/research/nodes')
from harness import surface, MU
from lib import FOLDS, CAPS
N='/Users/um-yunsang/BARAM2026/research/nodes/'
W=0.04; NC=26
A,FR,COLS=surface(('G2','DROP:grid__'))
cf=A['cf'].to_numpy(); grp=A['grp'].to_numpy(); idx=A.index
valid=np.isfinite(cf)&(cf>=0.1); w_prod=np.where(valid,np.clip(cf,0,1.2),0.05)
w_valid=np.where(valid,1.0,0.15); gapv=A['pc_true'].to_numpy()-cf
MEMS=['D','X','M2']
store={k:{'rows':[],'p':[]} for k in MEMS}
for f,(a,b) in FOLDS.items():
    a=pd.Timestamp(a); b=pd.Timestamp(b)
    tr=np.asarray(idx<a); va=np.asarray((idx>=a)&(idx<=b)); keep=np.isfinite(cf[va])
    t0=time.time(); m=tr&np.isfinite(A['pc_true'].to_numpy())
    mu=lgb.LGBMRegressor(**MU); mu.fit(A.loc[m,COLS],A.loc[m,'pc_true'],sample_weight=w_prod[m])
    pc=np.clip(mu.predict(A[COLS]),0,1)
    sel=list(pd.Series(mu.feature_importances_,index=COLS).sort_values(ascending=False).head(150).index)
    B=A[sel].copy(); B['pc_hat']=pc
    for k in (1,2,3): B[f'ig{k}']=(grp==k).astype('float32')
    cls=np.clip(np.nan_to_num(cf,nan=0.0)/W,0,NC-1).astype(int)
    cm=tr&np.isfinite(cf)&(~(gapv>=0.05))
    Xn=np.nan_to_num(B.to_numpy('float64'), nan=0.0, posinf=0.0, neginf=0.0)
    sc=StandardScaler().fit(Xn[tr]); Xs=sc.transform(Xn)
    def emit(name, P):
        store[name]['rows'].append(pd.DataFrame({'fold_id':f,'group_id':grp[va][keep],
            'forecast_kst_dtm':idx[va][keep],'cf':cf[va][keep],
            'mean_gen_g':[float(np.nanmean(cf[tr&(grp==g)])) for g in grp[va][keep]]}))
        store[name]['p'].append(P[keep])
    d=lgb.LGBMClassifier(objective='multiclass', boosting_type='dart', n_estimators=400,
        learning_rate=0.08, num_leaves=31, min_child_samples=60, subsample=0.85, subsample_freq=1,
        colsample_bytree=0.4, reg_lambda=3.0, random_state=20260803, n_jobs=6, verbose=-1)
    d.fit(B[cm],cls[cm],sample_weight=w_valid[cm])
    raw=d.predict_proba(B[va]); P=np.zeros((raw.shape[0],NC)); P[:,np.asarray(d.classes_,int)]=raw; emit('D',P)
    x=ExtraTreesClassifier(n_estimators=500, min_samples_leaf=12, max_features=0.3,
                           n_jobs=6, random_state=20260804)
    x.fit(Xn[cm], cls[cm], sample_weight=w_valid[cm])
    raw=x.predict_proba(Xn[va]); P=np.zeros((raw.shape[0],NC)); P[:,np.asarray(x.classes_,int)]=raw; emit('X',P)
    n2=MLPClassifier(hidden_layer_sizes=(512,), alpha=3e-3, batch_size=512, learning_rate_init=2e-3,
                     max_iter=150, early_stopping=True, n_iter_no_change=10, random_state=20260805)
    n2.fit(Xs[cm], cls[cm])
    raw=n2.predict_proba(Xs[va]); P=np.zeros((raw.shape[0],NC)); P[:,np.asarray(n2.classes_,int)]=raw; emit('M2',P)
    print(f'  {f} D/X/M2 {round(time.time()-t0,1)}s', flush=True)
for name in MEMS:
    pd.concat(store[name]['rows'],ignore_index=True).to_parquet(N+f'S7-N8_{name}_keys.parquet', index=False)
    np.save(N+f'S7-N8_{name}_prob.npy', np.vstack(store[name]['p']))
print('DONE', flush=True)
