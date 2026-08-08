
"""S7-N5 · save the conditional distribution itself, then search the action policy offline."""
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
A, FR, COLS = surface(('G2','DROP:grid__'))
cf=A['cf'].to_numpy(); grp=A['grp'].to_numpy(); idx=A.index
valid=np.isfinite(cf)&(cf>=0.1); w_all=np.where(valid,np.clip(cf,0,1.2),0.05)
gapv=A['pc_true'].to_numpy()-cf
rows=[]; probs=[]
for f,(a,b) in FOLDS.items():
    a=pd.Timestamp(a); b=pd.Timestamp(b)
    tr=np.asarray(idx<a); va=np.asarray((idx>=a)&(idx<=b))
    t0=time.time(); m=tr&np.isfinite(A['pc_true'].to_numpy())
    mu=lgb.LGBMRegressor(**MU); mu.fit(A.loc[m,COLS],A.loc[m,'pc_true'],sample_weight=w_all[m])
    pc=np.clip(mu.predict(A[COLS]),0,1)
    imp=pd.Series(mu.feature_importances_,index=COLS).sort_values(ascending=False)
    sel=list(imp.head(150).index)
    B=A[sel].copy(); B['pc_hat']=pc
    for k in (1,2,3): B[f'ig{k}']=(grp==k).astype('float32')
    cls=np.clip(np.nan_to_num(cf,nan=0.0)/W,0,NC-1).astype(int)
    cm=tr&np.isfinite(cf)&(~(gapv>=0.05))
    clf=lgb.LGBMClassifier(**CLF); clf.fit(B[cm],cls[cm],sample_weight=w_all[cm])
    raw=clf.predict_proba(B[va]); P=np.zeros((raw.shape[0],NC)); P[:,np.asarray(clf.classes_,int)]=raw
    keep=np.isfinite(cf[va])
    rows.append(pd.DataFrame({'fold_id':f,'group_id':grp[va][keep],
                              'forecast_kst_dtm':idx[va][keep],
                              'cf':cf[va][keep],'pc_hat':pc[va][keep],
                              'mean_gen_g':[float(np.nanmean(cf[tr&(grp==g)])) for g in grp[va][keep]]}))
    probs.append(P[keep])
    print(f'  {f} {round(time.time()-t0,1)}s', flush=True)
R=pd.concat(rows,ignore_index=True); PR=np.vstack(probs)
R.to_parquet(N+'S7-N5_keys.parquet', index=False)
np.save(N+'S7-N5_prob.npy', PR)
print('saved', R.shape, PR.shape, flush=True)
