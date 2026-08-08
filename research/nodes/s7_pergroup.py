
"""S7-N7 · per-group classifier (the deployed lineage fits per group; mine pools) and gating ablation,
then S10 cross-family blend with the improved member."""
import sys, json, time
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0,'/Users/um-yunsang/BARAM2026/research/scratch')
sys.path.insert(0,'/Users/um-yunsang/BARAM2026/research/nodes')
from harness import surface, MU
from lib import official_total, FOLDS, CAPS
N='/Users/um-yunsang/BARAM2026/research/nodes/'; AB='/Users/um-yunsang/BARAM2026/artifacts/backtests/metric-aligned-probe/'
W=0.04; NC=26; C=(np.arange(NC)+0.5)*W
CLF=dict(objective='multiclass', n_estimators=350, learning_rate=0.06, num_leaves=31,
         min_child_samples=60, subsample=0.85, subsample_freq=1, colsample_bytree=0.4,
         reg_lambda=3.0, random_state=20260801, n_jobs=6, verbose=-1)
ACT=np.arange(0.02,1.0801,0.0025); SC={1:0.985,2:0.989,3:1.005}
err=np.abs(ACT[:,None]-C[None,:]); units=np.where(err<=0.06,4.,np.where(err<=0.08,3.,0.))
TEMPS=[0.6,1.0,1.5,2.0,2.5,3.0,4.0]; GAMMAS=[0.0,0.5,1.0,2.0,3.0,5.0]

A,FR,COLS=surface(('G2','DROP:grid__'))
cf=A['cf'].to_numpy(); grp=A['grp'].to_numpy(); idx=A.index
valid=np.isfinite(cf)&(cf>=0.1); w_prod=np.where(valid,np.clip(cf,0,1.2),0.05)
w_valid=np.where(valid,1.0,0.15); gapv=A['pc_true'].to_numpy()-cf
VAR={'pergroup_gate':(True,0.05),'pergroup_nogate':(True,1e9),'pooled_nogate':(False,1e9)}
store={k:{'rows':[],'p':[]} for k in VAR}
for f,(a,b) in FOLDS.items():
    a=pd.Timestamp(a); b=pd.Timestamp(b)
    tr=np.asarray(idx<a); va=np.asarray((idx>=a)&(idx<=b)); t0=time.time()
    m=tr&np.isfinite(A['pc_true'].to_numpy())
    mu=lgb.LGBMRegressor(**MU); mu.fit(A.loc[m,COLS],A.loc[m,'pc_true'],sample_weight=w_prod[m])
    pc=np.clip(mu.predict(A[COLS]),0,1)
    sel=list(pd.Series(mu.feature_importances_,index=COLS).sort_values(ascending=False).head(150).index)
    B=A[sel].copy(); B['pc_hat']=pc
    for k in (1,2,3): B[f'ig{k}']=(grp==k).astype('float32')
    cls=np.clip(np.nan_to_num(cf,nan=0.0)/W,0,NC-1).astype(int)
    keep=np.isfinite(cf[va])
    for name,(per,gate) in VAR.items():
        cm=tr&np.isfinite(cf)&(~(gapv>=gate))
        P=np.zeros((int(va.sum()),NC))
        if per:
            for g in (1,2,3):
                trg=cm&(grp==g); vag=(grp[va]==g)
                clf=lgb.LGBMClassifier(**CLF); clf.fit(B[trg],cls[trg],sample_weight=w_valid[trg])
                raw=clf.predict_proba(B[va][vag]); Pg=np.zeros((raw.shape[0],NC))
                Pg[:,np.asarray(clf.classes_,int)]=raw; P[vag]=Pg
        else:
            clf=lgb.LGBMClassifier(**CLF); clf.fit(B[cm],cls[cm],sample_weight=w_valid[cm])
            raw=clf.predict_proba(B[va]); P[:,np.asarray(clf.classes_,int)]=raw
        store[name]['rows'].append(pd.DataFrame({'fold_id':f,'group_id':grp[va][keep],
            'forecast_kst_dtm':idx[va][keep],'cf':cf[va][keep],
            'mean_gen_g':[float(np.nanmean(cf[tr&(grp==g)])) for g in grp[va][keep]]}))
        store[name]['p'].append(P[keep])
    print(f'  {f} {round(time.time()-t0,1)}s', flush=True)

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
    sc={k:official_total(v[['group_id','actual_kwh','prediction_kwh']])['total'] for k,v in frames.items()}
    bk=max(sc,key=sc.get)
    fo=[]
    for f in FOLDS:
        s2={k:official_total(v[v.fold_id!=f][['group_id','actual_kwh','prediction_kwh']])['total'] for k,v in frames.items()}
        pk=max(s2,key=s2.get); fo.append(frames[pk][frames[pk].fold_id==f])
    fos=official_total(pd.concat(fo,ignore_index=True)[['group_id','actual_kwh','prediction_kwh']])
    r=official_total(frames[bk][['group_id','actual_kwh','prediction_kwh']])
    print(f'{tag:16s} best{bk} in={sc[bk]:.6f} 1-NMAE={r["one_minus_nmae"]:.6f} FICR={r["ficr"]:.6f} | fo={fos["total"]:.6f}', flush=True)
    d=frames[bk].copy(); d['forecast_kst_dtm']=R.forecast_kst_dtm.to_numpy()
    return d, fos['total']

members={}
for name in VAR:
    R=pd.concat(store[name]['rows'],ignore_index=True); P=np.vstack(store[name]['p'])
    d,fo=policy_search(R,P,name); d.to_parquet(N+f'S7-N7_{name}.parquet', index=False); members[name]=d

# --- cross-family blend, fold-outside weight ---
key=['fold_id','group_id','forecast_kst_dtm']
fr=[]
for f in FOLDS:
    dd=pd.read_parquet(AB+f'M102_TOP100-{f}-policies.parquet'); dd['fold_id']=f
    fr.append(dd[key+['actual_kwh','T0.5_G1.5']].rename(columns={'T0.5_G1.5':'m102'}))
M=pd.concat(fr,ignore_index=True)
res={}
for name,d in list(members.items())+[('N6_valid',pd.read_parquet(N+'S7-N6_valid.parquet'))]:
    J=M.merge(d[key+['prediction_kwh']].rename(columns={'prediction_kwh':'mine'}), on=key)
    cap=J.group_id.map(CAPS)
    r=float(np.corrcoef((J.mine-J.actual_kwh)/cap,(J.m102-J.actual_kwh)/cap)[0,1])
    fo=[]
    for f in FOLDS:
        oth=J[J.fold_id!=f]; held=J[J.fold_id==f]
        c={w:official_total(oth.assign(prediction_kwh=w*oth.mine+(1-w)*oth.m102)[['group_id','actual_kwh','prediction_kwh']])['total'] for w in np.arange(0,1.001,0.05)}
        pw=max(c,key=c.get); fo.append(held.assign(prediction_kwh=pw*held.mine+(1-pw)*held.m102))
    g=official_total(pd.concat(fo,ignore_index=True)[['group_id','actual_kwh','prediction_kwh']])
    res[name]=dict(corr=r, blend_foldout=g['total'], one_minus_nmae=g['one_minus_nmae'], ficr=g['ficr'])
    print(f'BLEND {name:16s} corr={r:.4f} fold-outside={g["total"]:.6f} (M102 alone 0.629896)', flush=True)
json.dump(res, open(N+'S7-N7_blend.json','w'), indent=1)
