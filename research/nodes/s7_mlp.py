
"""S7-N9 · shallow MLP member (the only model class this project has never fitted) and
S8-N1 · month-level stability of the blend gain."""
import sys, json, time
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
sys.path.insert(0,'/Users/um-yunsang/BARAM2026/research/scratch')
sys.path.insert(0,'/Users/um-yunsang/BARAM2026/research/nodes')
from harness import surface, MU
from lib import official_total, FOLDS, CAPS
N='/Users/um-yunsang/BARAM2026/research/nodes/'
W=0.04; NC=26
A,FR,COLS=surface(('G2','DROP:grid__'))
cf=A['cf'].to_numpy(); grp=A['grp'].to_numpy(); idx=A.index
valid=np.isfinite(cf)&(cf>=0.1); w_prod=np.where(valid,np.clip(cf,0,1.2),0.05)
gapv=A['pc_true'].to_numpy()-cf
rows=[]; probs=[]
for f,(a,b) in FOLDS.items():
    a=pd.Timestamp(a); b=pd.Timestamp(b)
    tr=np.asarray(idx<a); va=np.asarray((idx>=a)&(idx<=b)); keep=np.isfinite(cf[va])
    t0=time.time(); m=tr&np.isfinite(A['pc_true'].to_numpy())
    mu=lgb.LGBMRegressor(**MU); mu.fit(A.loc[m,COLS],A.loc[m,'pc_true'],sample_weight=w_prod[m])
    pc=np.clip(mu.predict(A[COLS]),0,1)
    sel=list(pd.Series(mu.feature_importances_,index=COLS).sort_values(ascending=False).head(150).index)
    B=A[sel].copy(); B['pc_hat']=pc
    for k in (1,2,3): B[f'ig{k}']=(grp==k).astype('float32')
    X=B.to_numpy('float64'); X=np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    sc=StandardScaler().fit(X[tr]); Xs=sc.transform(X)
    cls=np.clip(np.nan_to_num(cf,nan=0.0)/W,0,NC-1).astype(int)
    cm=tr&np.isfinite(cf)&(~(gapv>=0.05))
    net=MLPClassifier(hidden_layer_sizes=(256,128), alpha=1e-3, batch_size=256,
                      learning_rate_init=1e-3, max_iter=120, early_stopping=True,
                      n_iter_no_change=8, random_state=20260801)
    net.fit(Xs[cm], cls[cm])
    raw=net.predict_proba(Xs[va]); P=np.zeros((raw.shape[0],NC)); P[:,np.asarray(net.classes_,int)]=raw
    rows.append(pd.DataFrame({'fold_id':f,'group_id':grp[va][keep],'forecast_kst_dtm':idx[va][keep],
        'cf':cf[va][keep],'mean_gen_g':[float(np.nanmean(cf[tr&(grp==g)])) for g in grp[va][keep]]}))
    probs.append(P[keep])
    print(f'  {f} MLP iters={net.n_iter_} {round(time.time()-t0,1)}s', flush=True)
R=pd.concat(rows,ignore_index=True); PR=np.vstack(probs)
R.to_parquet(N+'S7-N8_M_keys.parquet', index=False); np.save(N+'S7-N8_M_prob.npy', PR)
print('MLP member saved', PR.shape, flush=True)
