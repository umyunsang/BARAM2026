
"""Full-protocol confirmation: expanding-window per fold, pooled dev-2023 scoring,
directly comparable with the deployed lineage (M102_TOP100 T0.5_G1.5 = 0.629896)."""
from __future__ import annotations
import sys, json, time
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0,'/Users/um-yunsang/BARAM2026/research/scratch')
sys.path.insert(0,'/Users/um-yunsang/BARAM2026/research/nodes')
from harness import surface, MU, ACTIONS, TEMPS, GAMMAS
from lib import official_total, FOLDS, CAPS
N='/Users/um-yunsang/BARAM2026/research/nodes/'

def full_run(node_id, tag, gate=0.05, soft_cap={1:0.985,2:0.989,3:1.005},
             prod_weight=True, n_q=81, mu_params=MU, extra_cols=None):
    A, FR, COLS = surface()
    if extra_cols is not None: COLS = extra_cols
    cf_all=A['cf'].to_numpy(); grp=A['grp'].to_numpy()
    valid=np.isfinite(cf_all)&(cf_all>=0.1)
    w_all=np.where(valid, np.clip(cf_all,0,1.2), 0.05) if prod_weight else None
    parts=[]
    for f,(a,b) in FOLDS.items():
        a=pd.Timestamp(a); b=pd.Timestamp(b)
        tr=np.asarray(A.index<a); va=np.asarray((A.index>=a)&(A.index<=b))
        t0=time.time()
        m=tr&np.isfinite(A['pc_true'].to_numpy())
        mdl=lgb.LGBMRegressor(**mu_params)
        mdl.fit(A.loc[m,COLS], A.loc[m,'pc_true'], sample_weight=None if w_all is None else w_all[m])
        pc=np.clip(mdl.predict(A[COLS]),0,1)
        gap=A['pc_true'].to_numpy()-cf_all
        cal=tr&np.isfinite(cf_all)&(~(gap>=gate))
        resid=cf_all-pc
        qs=np.linspace(0.01,0.99,n_q)
        for g in (1,2,3):
            sel=va&(grp==g); mm=cal&(grp==g)
            z=np.quantile(resid[mm][np.isfinite(resid[mm])], qs)
            cap_hi=soft_cap[g] if soft_cap else 1.05
            samples=np.clip(pc[sel][:,None]+z[None,:],0.0,cap_hi)
            mean_gen=float(np.nanmean(cf_all[tr&(grp==g)]))
            cfv=cf_all[sel]; keep=np.isfinite(cfv)
            err=np.abs(ACTIONS[None,:,None]-samples[:,None,:])
            units=np.where(err<=0.06,4.0,np.where(err<=0.08,3.0,0.0))
            rec={'fold_id':f,'group_id':g,'actual_kwh':cfv[keep]*CAPS[g],'pc_hat':pc[sel][keep]}
            for tp in TEMPS:
                wq=np.full(samples.shape[1],1.0/samples.shape[1])**(1.0/tp); wq/=wq.sum()
                nm=-(err*wq).sum(axis=2); fi=((samples[:,None,:]*units)*wq).sum(axis=2)/(4.0*mean_gen)
                for gm in GAMMAS:
                    rec[f'T{tp}_G{gm}']=np.clip(ACTIONS[np.argmax(nm+gm*fi,axis=1)],0,cap_hi)[keep]*CAPS[g]
            parts.append(pd.DataFrame(rec))
        print(f'  {f} fit_rows={int(m.sum())} {round(time.time()-t0,1)}s', flush=True)
    P=pd.concat(parts,ignore_index=True)
    pol=[c for c in P.columns if c.startswith('T')]
    sc={c:official_total(P.rename(columns={c:'prediction_kwh'})[['group_id','actual_kwh','prediction_kwh']]) for c in pol}
    ser=pd.Series({k:v['total'] for k,v in sc.items()}).sort_values(ascending=False)
    best=ser.index[0]
    # fold-outside policy gate: pick the policy on the other two folds, apply to the held-out one
    fo=[]
    for f in FOLDS:
        oth=P[P.fold_id!=f]; held=P[P.fold_id==f]
        s2={c:official_total(oth.rename(columns={c:'prediction_kwh'})[['group_id','actual_kwh','prediction_kwh']])['total'] for c in pol}
        pick=max(s2,key=s2.get)
        fo.append((f,pick,held.rename(columns={pick:'prediction_kwh'})))
    foP=pd.concat([h[['group_id','actual_kwh','prediction_kwh']] for _,_,h in fo],ignore_index=True)
    fos=official_total(foP)
    out=dict(node_id=node_id, tag=tag, protocol='full expanding-window, pooled dev-2023',
             insample_best=float(ser.iloc[0]), insample_policy=best,
             one_minus_nmae=sc[best]['one_minus_nmae'], ficr=sc[best]['ficr'],
             foldout=fos['total'], foldout_1mnmae=fos['one_minus_nmae'], foldout_ficr=fos['ficr'],
             foldout_policies={f:p for f,p,_ in fo},
             deployed_reference=0.6298962296336882)
    print(f'{node_id}: in-sample {out["insample_best"]:.6f} ({best}) | fold-outside {out["foldout"]:.6f} '
          f'| deployed 0.629896', flush=True)
    json.dump(out, open(N+f'{node_id}_full.json','w'), indent=1)
    P.to_parquet(N+f'{node_id}_full.parquet', index=False)
    return out

if __name__=='__main__':
    full_run('S5-CLOSE','P1(0.05)+P7+P4+uncond sigma+nq81 — S5 declared best')
    full_run('S5-CTRL','no preprocessing treatment (control)', gate=1e9, soft_cap=None,
             prod_weight=False, n_q=41)
