
"""S7-N13 · enlarge the member library along axes that produce ERROR diversity rather than
individual strength: temporal weighting, temporal window, seasonal specialisation, resolution,
and a different GBDT implementation.  All share the frozen S5 treatment and G2 surface."""
import sys, json, time
import numpy as np, pandas as pd, lightgbm as lgb, xgboost as xgb
sys.path.insert(0,'/Users/um-yunsang/BARAM2026/research/scratch')
sys.path.insert(0,'/Users/um-yunsang/BARAM2026/research/nodes')
from harness import surface, MU
from lib import FOLDS, CAPS
N='/Users/um-yunsang/BARAM2026/research/nodes/'
W=0.04; NC=26
BASECLF=dict(objective='multiclass', n_estimators=350, learning_rate=0.06, num_leaves=31,
             min_child_samples=60, subsample=0.85, subsample_freq=1, colsample_bytree=0.4,
             reg_lambda=3.0, random_state=20260801, n_jobs=6, verbose=-1)
A,FR,COLS=surface(('G2','DROP:grid__'))
cf=A['cf'].to_numpy(); grp=A['grp'].to_numpy(); idx=A.index
valid=np.isfinite(cf)&(cf>=0.1); w_prod=np.where(valid,np.clip(cf,0,1.2),0.05)
w_valid=np.where(valid,1.0,0.15); gapv=A['pc_true'].to_numpy()-cf
NAMES=['R1','R2','W2','S1','XG','LV']
store={k:{'rows':[],'p':[]} for k in NAMES}
for f,(a,b) in FOLDS.items():
    a=pd.Timestamp(a); b=pd.Timestamp(b)
    tr=np.asarray(idx<a); va=np.asarray((idx>=a)&(idx<=b)); keep=np.isfinite(cf[va])
    age=(a-idx).days.to_numpy().astype(float)
    t0=time.time(); m=tr&np.isfinite(A['pc_true'].to_numpy())
    mu=lgb.LGBMRegressor(**MU); mu.fit(A.loc[m,COLS],A.loc[m,'pc_true'],sample_weight=w_prod[m])
    pc=np.clip(mu.predict(A[COLS]),0,1)
    sel=list(pd.Series(mu.feature_importances_,index=COLS).sort_values(ascending=False).head(150).index)
    B=A[sel].copy(); B['pc_hat']=pc
    for k in (1,2,3): B[f'ig{k}']=(grp==k).astype('float32')
    cls=np.clip(np.nan_to_num(cf,nan=0.0)/W,0,NC-1).astype(int)
    cm=tr&np.isfinite(cf)&(~(gapv>=0.05))
    Xn=np.nan_to_num(B.to_numpy('float32'), nan=0.0, posinf=0.0, neginf=0.0)
    def emit(name,P):
        store[name]['rows'].append(pd.DataFrame({'fold_id':f,'group_id':grp[va][keep],
            'forecast_kst_dtm':idx[va][keep],'cf':cf[va][keep],
            'mean_gen_g':[float(np.nanmean(cf[tr&(grp==g)])) for g in grp[va][keep]]}))
        store[name]['p'].append(P[keep])
    def fit_lgb(params, rows, w):
        c=lgb.LGBMClassifier(**params); c.fit(B[rows],cls[rows],sample_weight=w[rows])
        raw=c.predict_proba(B[va]); P=np.zeros((raw.shape[0],NC)); P[:,np.asarray(c.classes_,int)]=raw
        return P
    emit('R1', fit_lgb(BASECLF, cm, w_valid*np.exp(-np.log(2)*age/365.0)))
    emit('R2', fit_lgb(BASECLF, cm, w_valid*np.exp(-np.log(2)*age/180.0)))
    emit('W2', fit_lgb(BASECLF, cm & (age<=400), w_valid))
    q_va=idx[va].quarter.to_numpy()
    P=np.zeros((int(va.sum()),NC))
    for qq in (1,2,3,4):
        rows=cm & (idx.quarter.to_numpy()==qq)
        if rows.sum()<400: rows=cm
        c=lgb.LGBMClassifier(**BASECLF); c.fit(B[rows],cls[rows],sample_weight=w_valid[rows])
        vsel=(q_va==qq)
        if vsel.sum()==0: continue
        raw=c.predict_proba(B[va][vsel]); Pq=np.zeros((raw.shape[0],NC))
        Pq[:,np.asarray(c.classes_,int)]=raw; P[vsel]=Pq
    emit('S1', P)
    xc=xgb.XGBClassifier(objective='multi:softprob', num_class=NC, n_estimators=400,
                         learning_rate=0.07, max_depth=6, subsample=0.85, colsample_bytree=0.4,
                         reg_lambda=3.0, tree_method='hist', n_jobs=6, random_state=20260806,
                         verbosity=0)
    xc.fit(Xn[cm], cls[cm], sample_weight=w_valid[cm])
    raw=xc.predict_proba(Xn[va]); P=np.zeros((raw.shape[0],NC)); P[:,np.asarray(xc.classes_,int)]=raw
    emit('XG', P)
    emit('LV', fit_lgb(dict(BASECLF, num_leaves=127, min_child_samples=25, n_estimators=250,
                            learning_rate=0.05, colsample_bytree=0.25, random_state=20260807), cm, w_valid))
    print(f'  {f} 6 members {round(time.time()-t0,1)}s', flush=True)
for name in NAMES:
    pd.concat(store[name]['rows'],ignore_index=True).to_parquet(N+f'S7-N8_{name}_keys.parquet', index=False)
    np.save(N+f'S7-N8_{name}_prob.npy', np.vstack(store[name]['p']))
print('DONE', flush=True)
