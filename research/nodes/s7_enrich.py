
"""S7-N12 · teacher enrichment.  The classifier currently conditions on a single point teacher
`pc_hat`.  A band decision needs the conditional SPREAD, not just the location.  Add quantile
teachers on pc and the measured turbine-spread / intra-hour teachers as explicit inputs."""
import sys, json, time
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0,'/Users/um-yunsang/BARAM2026/research/scratch')
sys.path.insert(0,'/Users/um-yunsang/BARAM2026/research/nodes')
from harness import surface, MU
from lib import official_total, FOLDS, CAPS
N='/Users/um-yunsang/BARAM2026/research/nodes/'
S='/Users/um-yunsang/BARAM2026/research/scratch/'
W=0.04; NC=26
CLF=dict(objective='multiclass', n_estimators=350, learning_rate=0.06, num_leaves=31,
         min_child_samples=60, subsample=0.85, subsample_freq=1, colsample_bytree=0.4,
         reg_lambda=3.0, random_state=20260801, n_jobs=6, verbose=-1)
QT=dict(MU, n_estimators=500, objective='quantile')
AUX=dict(MU, n_estimators=500)
A,FR,COLS=surface(('G2','DROP:grid__'))
T=pd.read_parquet(S+'teacher_targets.parquet')
# attach the extra measured teacher targets, aligned per group
for nm in ('v_spread','v_intra','v_mean'):
    col=np.full(len(A), np.nan)
    for g in (1,2,3):
        sel=(A['grp'].to_numpy()==g)
        col[sel]=T[f'g{g}_{nm}'].reindex(A.index[sel]).to_numpy()
    A[f'tgt_{nm}']=col
cf=A['cf'].to_numpy(); grp=A['grp'].to_numpy(); idx=A.index
valid=np.isfinite(cf)&(cf>=0.1); w_prod=np.where(valid,np.clip(cf,0,1.2),0.05)
w_valid=np.where(valid,1.0,0.15); gapv=A['pc_true'].to_numpy()-cf
rows=[]; probs=[]
for f,(a,b) in FOLDS.items():
    a=pd.Timestamp(a); b=pd.Timestamp(b)
    tr=np.asarray(idx<a); va=np.asarray((idx>=a)&(idx<=b)); keep=np.isfinite(cf[va])
    t0=time.time(); m=tr&np.isfinite(A['pc_true'].to_numpy())
    mu=lgb.LGBMRegressor(**MU); mu.fit(A.loc[m,COLS],A.loc[m,'pc_true'],sample_weight=w_prod[m])
    pc=np.clip(mu.predict(A[COLS]),0,1)
    sel=list(pd.Series(mu.feature_importances_,index=COLS).sort_values(ascending=False).head(150).index)
    B=A[sel].copy(); B['pc_hat']=pc
    for q in (0.10,0.90):
        mq=lgb.LGBMRegressor(**dict(QT, alpha=q)); mq.fit(A.loc[m,COLS],A.loc[m,'pc_true'],sample_weight=w_prod[m])
        B[f'pc_q{int(q*100)}']=np.clip(mq.predict(A[COLS]),0,1)
    B['pc_band']=B['pc_q90']-B['pc_q10']
    for nm in ('v_spread','v_intra','v_mean'):
        mm=tr&np.isfinite(A[f'tgt_{nm}'].to_numpy())
        ma=lgb.LGBMRegressor(**AUX); ma.fit(A.loc[mm,COLS],A.loc[mm,f'tgt_{nm}'])
        B[f'hat_{nm}']=ma.predict(A[COLS])
    B['sens']=B['pc_band']*B['hat_v_spread']
    for k in (1,2,3): B[f'ig{k}']=(grp==k).astype('float32')
    cls=np.clip(np.nan_to_num(cf,nan=0.0)/W,0,NC-1).astype(int)
    cm=tr&np.isfinite(cf)&(~(gapv>=0.05))
    clf=lgb.LGBMClassifier(**CLF); clf.fit(B[cm],cls[cm],sample_weight=w_valid[cm])
    raw=clf.predict_proba(B[va]); P=np.zeros((raw.shape[0],NC)); P[:,np.asarray(clf.classes_,int)]=raw
    rows.append(pd.DataFrame({'fold_id':f,'group_id':grp[va][keep],'forecast_kst_dtm':idx[va][keep],
        'cf':cf[va][keep],'mean_gen_g':[float(np.nanmean(cf[tr&(grp==g)])) for g in grp[va][keep]]}))
    probs.append(P[keep])
    print(f'  {f} enriched nfeat={B.shape[1]} {round(time.time()-t0,1)}s', flush=True)
R=pd.concat(rows,ignore_index=True); PR=np.vstack(probs)
R.to_parquet(N+'S7-N8_R_keys.parquet', index=False); np.save(N+'S7-N8_R_prob.npy', PR)
# score it
ACT=np.arange(0.02,1.0801,0.0025); SC={1:0.985,2:0.989,3:1.005}
C=(np.arange(NC)+0.5)*W
err=np.abs(ACT[:,None]-C[None,:]); units=np.where(err<=0.06,4.,np.where(err<=0.08,3.,0.))
g=R.group_id.to_numpy(); mg=R.mean_gen_g.to_numpy()
capv=np.array([CAPS[x] for x in g]); hi=np.array([SC[x] for x in g]); act=R.cf.to_numpy()*capv
mask=(C>=0.10).astype(float)
D0=pd.DataFrame({'fold_id':R.fold_id,'group_id':g,'forecast_kst_dtm':R.forecast_kst_dtm,'actual_kwh':act})
frames={}
for tp in [0.6,1.0,1.5,2.0,2.5,3.0,4.0]:
    q=PR**(1.0/tp); q=q/np.maximum(q.sum(axis=1,keepdims=True),1e-12)
    q=q*mask[None,:]; q=q/np.maximum(q.sum(axis=1,keepdims=True),1e-12)
    nm=-(q@err.T); fic=(q@((C[None,:]*units).T))
    for gm in [0.0,0.5,1.0,2.0,3.0,5.0,8.0,12.0]:
        frames[(tp,gm)]=np.minimum(ACT[np.argmax(nm+gm*fic/(4.0*mg[:,None]),axis=1)],hi)*capv
out=np.empty(len(D0))
for f in FOLDS:
    s=(D0.fold_id==f).to_numpy()
    s2={k:official_total(D0[~s].assign(prediction_kwh=v[~s])[['group_id','actual_kwh','prediction_kwh']])['total'] for k,v in frames.items()}
    out[s]=frames[max(s2,key=s2.get)][s]
D0['prediction_kwh']=out
r=official_total(D0[['group_id','actual_kwh','prediction_kwh']])
print(f'S7-N12 enriched member: fold-outside={r["total"]:.6f} 1-NMAE={r["one_minus_nmae"]:.6f} FICR={r["ficr"]:.6f}', flush=True)
D0.to_parquet(N+'S7-N12_member.parquet', index=False)
