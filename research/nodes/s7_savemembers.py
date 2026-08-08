
"""S7-N8 · rebuild the member set saving the FULL conditional distribution, so that every
subsequent policy question is answerable without refitting.  Members: pooled / LDAPS-only /
GFS-only / per-group.  All share the frozen S5 treatment and the G2 feature surface."""
import sys, json, time
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0,'/Users/um-yunsang/BARAM2026/research/scratch')
sys.path.insert(0,'/Users/um-yunsang/BARAM2026/research/nodes')
from harness import surface, MU
from lib import FOLDS, CAPS
N='/Users/um-yunsang/BARAM2026/research/nodes/'
W=0.04; NC=26
CLF=dict(objective='multiclass', n_estimators=350, learning_rate=0.06, num_leaves=31,
         min_child_samples=60, subsample=0.85, subsample_freq=1, colsample_bytree=0.4,
         reg_lambda=3.0, random_state=20260801, n_jobs=6, verbose=-1)
A,FR,COLS=surface(('G2','DROP:grid__'))
cf=A['cf'].to_numpy(); grp=A['grp'].to_numpy(); idx=A.index
valid=np.isfinite(cf)&(cf>=0.1); w_prod=np.where(valid,np.clip(cf,0,1.2),0.05)
w_valid=np.where(valid,1.0,0.15); gapv=A['pc_true'].to_numpy()-cf
LD=[c for c in COLS if 'gfs' not in c.lower()]; GF=[c for c in COLS if 'ldaps' not in c.lower()]
MEM={'P':dict(cols=COLS, per=False), 'L':dict(cols=LD, per=False),
     'G':dict(cols=GF, per=False),   'Q':dict(cols=COLS, per=True)}
store={k:{'rows':[],'p':[]} for k in MEM}
for f,(a,b) in FOLDS.items():
    a=pd.Timestamp(a); b=pd.Timestamp(b)
    tr=np.asarray(idx<a); va=np.asarray((idx>=a)&(idx<=b)); keep=np.isfinite(cf[va])
    cls=np.clip(np.nan_to_num(cf,nan=0.0)/W,0,NC-1).astype(int)
    cm=tr&np.isfinite(cf)&(~(gapv>=0.05))
    for name,spec in MEM.items():
        t0=time.time(); cols=spec['cols']
        m=tr&np.isfinite(A['pc_true'].to_numpy())
        mu=lgb.LGBMRegressor(**MU); mu.fit(A.loc[m,cols],A.loc[m,'pc_true'],sample_weight=w_prod[m])
        pc=np.clip(mu.predict(A[cols]),0,1)
        sel=list(pd.Series(mu.feature_importances_,index=cols).sort_values(ascending=False).head(150).index)
        B=A[sel].copy(); B['pc_hat']=pc
        for k in (1,2,3): B[f'ig{k}']=(grp==k).astype('float32')
        P=np.zeros((int(va.sum()),NC))
        if spec['per']:
            for gg in (1,2,3):
                trg=cm&(grp==gg); vag=(grp[va]==gg)
                clf=lgb.LGBMClassifier(**CLF); clf.fit(B[trg],cls[trg],sample_weight=w_valid[trg])
                raw=clf.predict_proba(B[va][vag]); Pg=np.zeros((raw.shape[0],NC))
                Pg[:,np.asarray(clf.classes_,int)]=raw; P[vag]=Pg
        else:
            clf=lgb.LGBMClassifier(**CLF); clf.fit(B[cm],cls[cm],sample_weight=w_valid[cm])
            raw=clf.predict_proba(B[va]); P[:,np.asarray(clf.classes_,int)]=raw
        store[name]['rows'].append(pd.DataFrame({'fold_id':f,'group_id':grp[va][keep],
            'forecast_kst_dtm':idx[va][keep],'cf':cf[va][keep],'pc_hat':pc[va][keep],
            'mean_gen_g':[float(np.nanmean(cf[tr&(grp==gg)])) for gg in grp[va][keep]]}))
        store[name]['p'].append(P[keep])
        print(f'  {f} {name} {round(time.time()-t0,1)}s', flush=True)
for name in MEM:
    pd.concat(store[name]['rows'],ignore_index=True).to_parquet(N+f'S7-N8_{name}_keys.parquet', index=False)
    np.save(N+f'S7-N8_{name}_prob.npy', np.vstack(store[name]['p']))
print('DONE', flush=True)
