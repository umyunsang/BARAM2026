
"""S7-N2 · conditional predictive distribution by direct multiclass modelling.
This is the one architectural element that explains the deployed lineage's FICR advantage
(FICR 0.4062 at 1-NMAE 0.8550 versus my 0.3715 at a BETTER 1-NMAE 0.8617).
Everything else (S5 treatment, G2 feature surface, decision layer) is held identical."""
import sys, json, time
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0,'/Users/um-yunsang/BARAM2026/research/scratch')
sys.path.insert(0,'/Users/um-yunsang/BARAM2026/research/nodes')
from harness import surface, MU, ACTIONS, TEMPS, GAMMAS
from lib import official_total, FOLDS, CAPS
N='/Users/um-yunsang/BARAM2026/research/nodes/'
W=float(sys.argv[2]) if len(sys.argv)>2 else 0.04
NC=int(round(1.04/W))
CENTRES=(np.arange(NC)+0.5)*W
SC={1:0.985,2:0.989,3:1.005}
CLF=dict(objective='multiclass', n_estimators=350, learning_rate=0.06,
         num_leaves=31, min_child_samples=60, subsample=0.85, subsample_freq=1,
         colsample_bytree=0.4, reg_lambda=3.0, random_state=20260801, n_jobs=6, verbose=-1)
TOPK=int(sys.argv[1]) if len(sys.argv)>1 else 150

def main():
    A, FR, COLS = surface(('G2','DROP:grid__'))
    cf=A['cf'].to_numpy(); grp=A['grp'].to_numpy(); idx=A.index
    valid=np.isfinite(cf)&(cf>=0.1)
    w_all=np.where(valid, np.clip(cf,0,1.2), 0.05)
    gapv=A['pc_true'].to_numpy()-cf
    parts=[]
    for f,(a,b) in FOLDS.items():
        a=pd.Timestamp(a); b=pd.Timestamp(b)
        tr=np.asarray(idx<a); va=np.asarray((idx>=a)&(idx<=b))
        t0=time.time()
        m=tr&np.isfinite(A['pc_true'].to_numpy())
        mu=lgb.LGBMRegressor(**MU); mu.fit(A.loc[m,COLS], A.loc[m,'pc_true'], sample_weight=w_all[m])
        pc=np.clip(mu.predict(A[COLS]),0,1)
        imp=pd.Series(mu.feature_importances_, index=COLS).sort_values(ascending=False)
        sel=list(imp.head(TOPK).index)
        B=A[sel].copy(); B['pc_hat']=pc
        for k in (1,2,3): B[f'ig{k}']=(grp==k).astype('float32')
        cls=np.clip((cf/W).astype(int),0,NC-1)
        cm=tr&np.isfinite(cf)&(~(gapv>=0.05))
        clf=lgb.LGBMClassifier(**CLF)
        clf.fit(B[cm], cls[cm], sample_weight=w_all[cm])
        print(f'  {f} mu+clf fit rows={int(cm.sum())} nfeat={B.shape[1]} {round(time.time()-t0,1)}s', flush=True)
        raw=clf.predict_proba(B[va])
        prob=np.zeros((raw.shape[0], NC), dtype=float)
        prob[:, np.asarray(clf.classes_, dtype=int)] = raw
        gv=grp[va]; cfv=cf[va]; iv=idx[va]
        for g in (1,2,3):
            s=gv==g; keep=np.isfinite(cfv[s])
            P=prob[s]; mean_gen=float(np.nanmean(cf[tr&(grp==g)]))
            err=np.abs(ACTIONS[:,None]-CENTRES[None,:])
            units=np.where(err<=0.06,4.0,np.where(err<=0.08,3.0,0.0))
            rec={'fold_id':f,'group_id':g,'forecast_kst_dtm':iv[s][keep],
                 'actual_kwh':cfv[s][keep]*CAPS[g],'pc_hat':pc[va][s][keep]}
            for tp in TEMPS:
                q=P**(1.0/tp); q=q/q.sum(axis=1,keepdims=True)
                nm=-(q@err.T); fi=(q@(CENTRES[None,:]*units).T)/(4.0*mean_gen)
                for gm in GAMMAS:
                    rec[f'T{tp}_G{gm}']=np.clip(ACTIONS[np.argmax(nm+gm*fi,axis=1)],0,SC[g])[keep]*CAPS[g]
            parts.append(pd.DataFrame(rec))
    Q=pd.concat(parts,ignore_index=True)
    Q.to_parquet(N+f'S7-N2_w{W}_k{TOPK}_full.parquet', index=False)
    pol=[c for c in Q.columns if c.startswith('T')]
    sc={c:official_total(Q.rename(columns={c:'prediction_kwh'})[['group_id','actual_kwh','prediction_kwh']]) for c in pol}
    ser=pd.Series({k:v['total'] for k,v in sc.items()}).sort_values(ascending=False)
    fo=[]
    for f in FOLDS:
        oth=Q[Q.fold_id!=f]; held=Q[Q.fold_id==f]
        s2={c:official_total(oth.rename(columns={c:'prediction_kwh'})[['group_id','actual_kwh','prediction_kwh']])['total'] for c in pol}
        fo.append(held.rename(columns={max(s2,key=s2.get):'prediction_kwh'}))
    fos=official_total(pd.concat(fo,ignore_index=True)[['group_id','actual_kwh','prediction_kwh']])
    best=ser.index[0]
    print(f'S7-N2 w={W} k={TOPK} nclass={NC}: in-sample {ser.iloc[0]:.6f} ({best}) 1-NMAE={sc[best]["one_minus_nmae"]:.6f} '
          f'FICR={sc[best]["ficr"]:.6f} | fold-outside {fos["total"]:.6f}', flush=True)
    json.dump(dict(node=f'S7-N2_w{W}_k{TOPK}', topk=TOPK, width=W, nclass=NC, insample=float(ser.iloc[0]), policy=best,
                   one_minus_nmae=sc[best]['one_minus_nmae'], ficr=sc[best]['ficr'],
                   foldout=fos['total'], grid={k:float(v) for k,v in ser.items()}),
              open(N+f'S7-N2_w{W}_k{TOPK}.json','w'), indent=1)

main()
