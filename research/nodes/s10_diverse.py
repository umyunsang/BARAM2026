
"""S10-N2 · deliberately diverse members inside my family, then a fold-outside blend.
Sources of diversity (each is a declared, single change):
  L   LDAPS-only feature view      (source separation, HEFTCom GEB / B-5)
  G   GFS-only feature view
  C   direct cf target for the point model instead of the physics target pc
  E   epsilon-insensitive (band) objective on the point model  (0 dof: eps fixed at the metric's 0.06)
"""
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
TEMPS=[0.6,1.0,1.5,2.0,2.5,3.0,4.0]; GAMMAS=[0.0,0.5,1.0,2.0,3.0,5.0]
EPS=0.06

def eps_obj(y, p):
    d = p - y
    inside = np.abs(d) <= EPS
    grad = np.where(inside, 0.0, d - np.sign(d)*EPS)
    hess = np.where(inside, 1e-3, 1.0)
    return grad, hess

A,FR,COLS=surface(('G2','DROP:grid__'))
cf=A['cf'].to_numpy(); grp=A['grp'].to_numpy(); idx=A.index
valid=np.isfinite(cf)&(cf>=0.1); w_prod=np.where(valid,np.clip(cf,0,1.2),0.05)
w_valid=np.where(valid,1.0,0.15); gapv=A['pc_true'].to_numpy()-cf
LD=[c for c in COLS if 'gfs' not in c.lower()]
GF=[c for c in COLS if 'ldaps' not in c.lower()]
MEM={'L':dict(cols=LD, target='pc_true', obj=None),
     'G':dict(cols=GF, target='pc_true', obj=None),
     'C':dict(cols=COLS, target='cf',     obj=None),
     'E':dict(cols=COLS, target='pc_true',obj=eps_obj)}
store={k:{'rows':[],'p':[]} for k in MEM}
for f,(a,b) in FOLDS.items():
    a=pd.Timestamp(a); b=pd.Timestamp(b)
    tr=np.asarray(idx<a); va=np.asarray((idx>=a)&(idx<=b)); keep=np.isfinite(cf[va])
    cls=np.clip(np.nan_to_num(cf,nan=0.0)/W,0,NC-1).astype(int)
    cm=tr&np.isfinite(cf)&(~(gapv>=0.05))
    for name,spec in MEM.items():
        t0=time.time()
        m=tr&np.isfinite(A[spec['target']].to_numpy())
        pr=dict(MU)
        if spec['obj'] is not None: pr=dict(MU); pr['objective']=spec['obj']
        mu=lgb.LGBMRegressor(**pr)
        mu.fit(A.loc[m,spec['cols']], A.loc[m,spec['target']], sample_weight=w_prod[m])
        pc=np.clip(mu.predict(A[spec['cols']]),0,1)
        sel=list(pd.Series(mu.feature_importances_,index=spec['cols']).sort_values(ascending=False).head(150).index)
        B=A[sel].copy(); B['pc_hat']=pc
        for k in (1,2,3): B[f'ig{k}']=(grp==k).astype('float32')
        clf=lgb.LGBMClassifier(**CLF); clf.fit(B[cm],cls[cm],sample_weight=w_valid[cm])
        raw=clf.predict_proba(B[va]); P=np.zeros((raw.shape[0],NC)); P[:,np.asarray(clf.classes_,int)]=raw
        store[name]['rows'].append(pd.DataFrame({'fold_id':f,'group_id':grp[va][keep],
            'forecast_kst_dtm':idx[va][keep],'cf':cf[va][keep],
            'mean_gen_g':[float(np.nanmean(cf[tr&(grp==g)])) for g in grp[va][keep]]}))
        store[name]['p'].append(P[keep])
        print(f'  {f} member {name} {round(time.time()-t0,1)}s', flush=True)

def policy_search(R,P,tag):
    g=R.group_id.to_numpy(); mg=R.mean_gen_g.to_numpy()
    capv=np.array([CAPS[x] for x in g]); hi=np.array([SC[x] for x in g]); act=R.cf.to_numpy()*capv
    frames={}
    for tp in TEMPS:
        q=P**(1.0/tp); q=q/np.maximum(q.sum(axis=1,keepdims=True),1e-12)
        nm=-(q@err.T); fic=(q@((C[None,:]*units).T))
        for gm in GAMMAS:
            pred=np.minimum(ACT[np.argmax(nm+gm*fic/(4.0*mg[:,None]),axis=1)],hi)*capv
            frames[(tp,gm)]=pd.DataFrame({'group_id':g,'actual_kwh':act,'prediction_kwh':pred,
                                          'fold_id':R.fold_id.to_numpy()})
    fo=[]
    for f in FOLDS:
        s2={k:official_total(v[v.fold_id!=f][['group_id','actual_kwh','prediction_kwh']])['total'] for k,v in frames.items()}
        pk=max(s2,key=s2.get); fo.append(frames[pk][frames[pk].fold_id==f])
    d=pd.concat(fo,ignore_index=True)
    sc={k:official_total(v[['group_id','actual_kwh','prediction_kwh']])['total'] for k,v in frames.items()}
    bk=max(sc,key=sc.get)
    print(f'{tag}: best{bk} in={sc[bk]:.6f} fold-outside={official_total(d[["group_id","actual_kwh","prediction_kwh"]])["total"]:.6f}', flush=True)
    o=frames[bk].copy(); o['forecast_kst_dtm']=R.forecast_kst_dtm.to_numpy()
    return o
for name in MEM:
    R=pd.concat(store[name]['rows'],ignore_index=True); P=np.vstack(store[name]['p'])
    policy_search(R,P,name).to_parquet(N+f'S10-DIV_{name}.parquet', index=False)
print('DONE', flush=True)
