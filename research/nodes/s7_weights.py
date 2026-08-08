
"""S7-N6 · classifier sample-weight ablation.
The decision layer needs a calibrated P(cf | x).  Production weighting improves point accuracy
but biases the probability estimate; that is a candidate explanation for my FICR deficit
(0.3817 versus the deployed 0.4062 at a BETTER 1-NMAE)."""
import sys, json, time
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0,'/Users/um-yunsang/BARAM2026/research/scratch')
sys.path.insert(0,'/Users/um-yunsang/BARAM2026/research/nodes')
from harness import surface, MU
from lib import official_total, FOLDS, CAPS
N='/Users/um-yunsang/BARAM2026/research/nodes/'
W=0.04; NC=26; C=(np.arange(NC)+0.5)*W
CLF=dict(objective='multiclass', n_estimators=350, learning_rate=0.06, num_leaves=31,
         min_child_samples=60, subsample=0.85, subsample_freq=1, colsample_bytree=0.4,
         reg_lambda=3.0, random_state=20260801, n_jobs=6, verbose=-1)
ACT=np.arange(0.02,1.0801,0.0025); SC={1:0.985,2:0.989,3:1.005}
err=np.abs(ACT[:,None]-C[None,:]); units=np.where(err<=0.06,4.,np.where(err<=0.08,3.,0.))
TEMPS=[0.6,1.0,1.5,2.0,2.5,3.0,4.0]; GAMMAS=[0.0,0.5,1.0,2.0,3.0,5.0,8.0]

A, FR, COLS = surface(('G2','DROP:grid__'))
cf=A['cf'].to_numpy(); grp=A['grp'].to_numpy(); idx=A.index
valid=np.isfinite(cf)&(cf>=0.1); w_prod=np.where(valid,np.clip(cf,0,1.2),0.05)
w_valid=np.where(valid,1.0,0.15)
gapv=A['pc_true'].to_numpy()-cf
VAR={'prod':w_prod,'valid':w_valid,'none':None}
store={k:{'rows':[],'p':[]} for k in VAR}
for f,(a,b) in FOLDS.items():
    a=pd.Timestamp(a); b=pd.Timestamp(b)
    tr=np.asarray(idx<a); va=np.asarray((idx>=a)&(idx<=b))
    t0=time.time(); m=tr&np.isfinite(A['pc_true'].to_numpy())
    mu=lgb.LGBMRegressor(**MU); mu.fit(A.loc[m,COLS],A.loc[m,'pc_true'],sample_weight=w_prod[m])
    pc=np.clip(mu.predict(A[COLS]),0,1)
    sel=list(pd.Series(mu.feature_importances_,index=COLS).sort_values(ascending=False).head(150).index)
    B=A[sel].copy(); B['pc_hat']=pc
    for k in (1,2,3): B[f'ig{k}']=(grp==k).astype('float32')
    cls=np.clip(np.nan_to_num(cf,nan=0.0)/W,0,NC-1).astype(int)
    cm=tr&np.isfinite(cf)&(~(gapv>=0.05))
    keep=np.isfinite(cf[va])
    for name,w in VAR.items():
        clf=lgb.LGBMClassifier(**CLF)
        clf.fit(B[cm],cls[cm],sample_weight=None if w is None else w[cm])
        raw=clf.predict_proba(B[va]); P=np.zeros((raw.shape[0],NC)); P[:,np.asarray(clf.classes_,int)]=raw
        store[name]['rows'].append(pd.DataFrame({'fold_id':f,'group_id':grp[va][keep],
            'forecast_kst_dtm':idx[va][keep],'cf':cf[va][keep],
            'mean_gen_g':[float(np.nanmean(cf[tr&(grp==g)])) for g in grp[va][keep]]}))
        store[name]['p'].append(P[keep])
    print(f'  {f} {round(time.time()-t0,1)}s', flush=True)

def policy_search(R,P,tag):
    g=R.group_id.to_numpy(); mg=R.mean_gen_g.to_numpy()
    capv=np.array([CAPS[x] for x in g]); hi=np.array([SC[x] for x in g])
    act=R.cf.to_numpy()*capv
    best=None; frames={}
    for tp in TEMPS:
        q=P**(1.0/tp); q=q/q.sum(axis=1,keepdims=True)
        nm=-(q@err.T); fic=(q@((C[None,:]*units).T))
        for gm in GAMMAS:
            u=nm+gm*fic/(4.0*mg[:,None])
            pred=np.minimum(ACT[np.argmax(u,axis=1)],hi)*capv
            d=pd.DataFrame({'group_id':g,'actual_kwh':act,'prediction_kwh':pred,'fold_id':R.fold_id.to_numpy()})
            frames[(tp,gm)]=d
    sc={k:official_total(v[['group_id','actual_kwh','prediction_kwh']])['total'] for k,v in frames.items()}
    bk=max(sc,key=sc.get)
    fo=[]
    for f in FOLDS:
        s2={k:official_total(v[v.fold_id!=f][['group_id','actual_kwh','prediction_kwh']])['total'] for k,v in frames.items()}
        pk=max(s2,key=s2.get); fo.append(frames[pk][frames[pk].fold_id==f])
    fos=official_total(pd.concat(fo,ignore_index=True)[['group_id','actual_kwh','prediction_kwh']])
    r=official_total(frames[bk][['group_id','actual_kwh','prediction_kwh']])
    print(f'{tag:8s} best{bk} in-sample={sc[bk]:.6f} 1-NMAE={r["one_minus_nmae"]:.6f} FICR={r["ficr"]:.6f} '
          f'| fold-outside={fos["total"]:.6f}', flush=True)
    d=frames[bk].copy(); d['forecast_kst_dtm']=R.forecast_kst_dtm.to_numpy()
    return d, fos['total'], sc[bk]

out={}
for name in VAR:
    R=pd.concat(store[name]['rows'],ignore_index=True); P=np.vstack(store[name]['p'])
    d,fo,ins=policy_search(R,P,name)
    d.to_parquet(N+f'S7-N6_{name}.parquet', index=False)
    out[name]=dict(foldout=fo, insample=ins)
json.dump(out, open(N+'S7-N6.json','w'), indent=1)
