
"""S10 · 모델개선전략 — cross-family combination.
My lineage (physics teacher + pooled + S5 preprocessing) is a different model family from the
deployed 46-bin classifier lineage.  AGENTS.md records that every classifier-family member has
error correlation 0.984-0.994 with M115 and the analog member 0.944.  Measure mine.
Row alignment key: (fold_id, group_id, forecast_kst_dtm) — declared explicitly."""
import sys, json, time
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0,'/Users/um-yunsang/BARAM2026/research/scratch')
sys.path.insert(0,'/Users/um-yunsang/BARAM2026/research/nodes')
from harness import surface, MU, ACTIONS, TEMPS, GAMMAS
from lib import official_total, FOLDS, CAPS
N='/Users/um-yunsang/BARAM2026/research/nodes/'
AB='/Users/um-yunsang/BARAM2026/artifacts/backtests/metric-aligned-probe/'

def mine(tag, blocks=('G2','DROP:grid__'), gate=0.05, n_q=81):
    A, FR, COLS = surface(blocks)
    cf_all=A['cf'].to_numpy(); grp=A['grp'].to_numpy(); idx=A.index
    valid=np.isfinite(cf_all)&(cf_all>=0.1)
    w_all=np.where(valid, np.clip(cf_all,0,1.2), 0.05)
    SC={1:0.985,2:0.989,3:1.005}
    parts=[]
    for f,(a,b) in FOLDS.items():
        a=pd.Timestamp(a); b=pd.Timestamp(b)
        tr=np.asarray(idx<a); va=np.asarray((idx>=a)&(idx<=b))
        m=tr&np.isfinite(A['pc_true'].to_numpy())
        t0=time.time()
        mdl=lgb.LGBMRegressor(**MU); mdl.fit(A.loc[m,COLS], A.loc[m,'pc_true'], sample_weight=w_all[m])
        pc=np.clip(mdl.predict(A[COLS]),0,1)
        gapv=A['pc_true'].to_numpy()-cf_all
        cal=tr&np.isfinite(cf_all)&(~(gapv>=gate))
        resid=cf_all-pc; qs=np.linspace(0.01,0.99,n_q)
        for g in (1,2,3):
            sel=va&(grp==g); mm=cal&(grp==g)
            z=np.quantile(resid[mm][np.isfinite(resid[mm])],qs)
            samples=np.clip(pc[sel][:,None]+z[None,:],0.0,SC[g])
            mean_gen=float(np.nanmean(cf_all[tr&(grp==g)]))
            cfv=cf_all[sel]; keep=np.isfinite(cfv)
            err=np.abs(ACTIONS[None,:,None]-samples[:,None,:])
            units=np.where(err<=0.06,4.0,np.where(err<=0.08,3.0,0.0))
            rec={'fold_id':f,'group_id':g,'forecast_kst_dtm':idx[sel][keep],
                 'actual_kwh':cfv[keep]*CAPS[g],'pc_hat':pc[sel][keep]}
            for tp in TEMPS:
                wq=np.full(samples.shape[1],1.0/samples.shape[1])**(1.0/tp); wq/=wq.sum()
                nm=-(err*wq).sum(axis=2); fi=((samples[:,None,:]*units)*wq).sum(axis=2)/(4.0*mean_gen)
                for gm in GAMMAS:
                    rec[f'T{tp}_G{gm}']=np.clip(ACTIONS[np.argmax(nm+gm*fi,axis=1)],0,SC[g])[keep]*CAPS[g]
            parts.append(pd.DataFrame(rec))
        print(f'  {f} {round(time.time()-t0,1)}s', flush=True)
    P=pd.concat(parts,ignore_index=True)
    P.to_parquet(N+f'{tag}_full.parquet', index=False)
    return P

if __name__=='__main__':
    P=mine('S10-MINE2')
    key=['fold_id','group_id','forecast_kst_dtm']
    pol=[c for c in P.columns if c.startswith('T')]
    sc={c:official_total(P.rename(columns={c:'prediction_kwh'})[['group_id','actual_kwh','prediction_kwh']])['total'] for c in pol}
    ser=pd.Series(sc).sort_values(ascending=False)
    print('MINE2 full-protocol best', ser.index[0], round(ser.iloc[0],6), flush=True)
    # fold-outside policy gate on my own member
    fo=[]
    for f in FOLDS:
        oth=P[P.fold_id!=f]; held=P[P.fold_id==f]
        s2={c:official_total(oth.rename(columns={c:'prediction_kwh'})[['group_id','actual_kwh','prediction_kwh']])['total'] for c in pol}
        pick=max(s2,key=s2.get); fo.append(held.rename(columns={pick:'prediction_kwh'}))
    fos=official_total(pd.concat(fo,ignore_index=True)[['group_id','actual_kwh','prediction_kwh']])
    print('MINE2 fold-outside', round(fos['total'],6), flush=True)
    out={'mine_insample':float(ser.iloc[0]),'mine_policy':ser.index[0],'mine_foldout':fos['total'],'members':{}}
    for stem,pd_ in [('M102_TOP100','T0.5_G1.5'),('M113_LGBM_DART','T0.5_G0.5'),('M115_XGBOOST','T0.6_G0.35')]:
        fr=[]
        for f in FOLDS:
            d=pd.read_parquet(AB+f'{stem}-{f}-policies.parquet'); d['fold_id']=f
            fr.append(d[['fold_id','group_id','forecast_kst_dtm','actual_kwh',pd_]].rename(columns={pd_:'other'}))
        O=pd.concat(fr,ignore_index=True)
        J=P.merge(O.drop(columns=['actual_kwh']), on=key)
        sc2={c:official_total(J.rename(columns={c:'prediction_kwh'})[['group_id','actual_kwh','prediction_kwh']])['total'] for c in pol}
        mb=max(sc2,key=sc2.get); J['mine']=J[mb]
        cap=J.group_id.map(CAPS)
        r=float(np.corrcoef((J.mine-J.actual_kwh)/cap,(J.other-J.actual_kwh)/cap)[0,1])
        base=official_total(J.assign(prediction_kwh=J.other)[['group_id','actual_kwh','prediction_kwh']])['total']
        curve={round(float(w),2):official_total(J.assign(prediction_kwh=w*J.mine+(1-w)*J.other)[['group_id','actual_kwh','prediction_kwh']])['total'] for w in np.arange(0,1.001,0.05)}
        bw=max(curve,key=curve.get)
        # fold-outside weight gate
        fw=[]
        for f in FOLDS:
            oth=J[J.fold_id!=f]; held=J[J.fold_id==f]
            c2={w:official_total(oth.assign(prediction_kwh=w*oth.mine+(1-w)*oth.other)[['group_id','actual_kwh','prediction_kwh']])['total'] for w in np.arange(0,1.001,0.05)}
            pw=max(c2,key=c2.get)
            fw.append(held.assign(prediction_kwh=pw*held.mine+(1-pw)*held.other))
        fgs=official_total(pd.concat(fw,ignore_index=True)[['group_id','actual_kwh','prediction_kwh']])
        print(f'{stem:16s} n={len(J)} mine({mb})={sc2[mb]:.6f} other={base:.6f} corr={r:.4f} '
              f'w*={bw:.2f} blend_insample={curve[bw]:.6f} blend_foldout={fgs["total"]:.6f} '
              f'gain_fo={fgs["total"]-base:+.6f}', flush=True)
        out['members'][stem]=dict(n=len(J),mine=sc2[mb],other=base,corr=r,best_w=bw,
            blend_insample=curve[bw],blend_foldout=fgs['total'],gain_foldout=fgs['total']-base,
            curve={str(k):v for k,v in curve.items()})
    json.dump(out, open(N+'S10_crossfamily2.json','w'), indent=1)
