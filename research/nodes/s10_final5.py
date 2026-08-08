
"""S10-FINAL5 · score every member in the library, then the dof ladder against the deployed lineage."""
import sys, json, glob, os
import numpy as np, pandas as pd
sys.path.insert(0,'/Users/um-yunsang/BARAM2026/research/scratch')
from lib import official_total, FOLDS, CAPS
N='/Users/um-yunsang/BARAM2026/research/nodes/'; AB='/Users/um-yunsang/BARAM2026/artifacts/backtests/metric-aligned-probe/'
W=0.04; ACT=np.arange(0.02,1.0801,0.0025); SC={1:0.985,2:0.989,3:1.005}
TEMPS=[0.6,1.0,1.5,2.0,2.5,3.0,4.0]; GAMMAS=[0.0,0.5,1.0,2.0,3.0,5.0,8.0,12.0]
key=['fold_id','group_id','forecast_kst_dtm']
NAMES=[os.path.basename(p).split('_')[1] for p in sorted(glob.glob(N+'S7-N8_*_prob.npy'))]
print('library:', NAMES, flush=True)
R0=pd.read_parquet(N+'S7-N8_P_keys.parquet')
NC=26; C=(np.arange(NC)+0.5)*W
err=np.abs(ACT[:,None]-C[None,:]); units=np.where(err<=0.06,4.,np.where(err<=0.08,3.,0.))
g=R0.group_id.to_numpy(); mg=R0.mean_gen_g.to_numpy()
capv=np.array([CAPS[x] for x in g]); hi=np.array([SC[x] for x in g]); act=R0.cf.to_numpy()*capv
mask=(C>=0.10).astype(float)
D0=pd.DataFrame({'fold_id':R0.fold_id,'group_id':g,'forecast_kst_dtm':R0.forecast_kst_dtm,'actual_kwh':act})
def decide(P):
    frames={}
    for tp in TEMPS:
        q=P**(1.0/tp); q=q/np.maximum(q.sum(axis=1,keepdims=True),1e-12)
        q=q*mask[None,:]; q=q/np.maximum(q.sum(axis=1,keepdims=True),1e-12)
        nm=-(q@err.T); fic=(q@((C[None,:]*units).T))
        for gm in GAMMAS:
            frames[(tp,gm)]=np.minimum(ACT[np.argmax(nm+gm*fic/(4.0*mg[:,None]),axis=1)],hi)*capv
    out=np.empty(len(D0))
    for f in FOLDS:
        s=(D0.fold_id==f).to_numpy()
        s2={k:official_total(D0[~s].assign(prediction_kwh=v[~s])[['group_id','actual_kwh','prediction_kwh']])['total'] for k,v in frames.items()}
        out[s]=frames[max(s2,key=s2.get)][s]
    return out, official_total(D0.assign(prediction_kwh=out)[['group_id','actual_kwh','prediction_kwh']])['total']
J=D0.copy(); PROB={}
for n in NAMES:
    k=pd.read_parquet(N+f'S7-N8_{n}_keys.parquet')
    if len(k)!=len(R0) or not (k['forecast_kst_dtm'].values==R0['forecast_kst_dtm'].values).all():
        print('  SKIP misaligned', n, flush=True); continue
    P=np.load(N+f'S7-N8_{n}_prob.npy'); PROB[n]=P
    out,t=decide(P); J[n]=out
    print(f'  member {n:4s} fold-outside={t:.6f}', flush=True)
dep={'M102_TOP100':'T0.5_G1.5','M113_LGBM_DART':'T0.5_G0.5','M115_XGBOOST':'T0.6_G0.35'}
for stem,pol in dep.items():
    fr=[]
    for f in FOLDS:
        d=pd.read_parquet(AB+f'{stem}-{f}-policies.parquet'); d['fold_id']=f
        fr.append(d[key+[pol]].rename(columns={pol:stem}))
    J=J.merge(pd.concat(fr,ignore_index=True), on=key)
J['DEPAVG']=J[list(dep)].mean(axis=1)
mem=list(PROB)
J['MYALL']=J[mem].mean(axis=1)
outp,tp_=decide(np.mean([PROB[n] for n in mem],axis=0)); J['LAWALL']=outp
print(f'  LAWALL fold-outside={tp_:.6f}', flush=True)
tot=lambda c: official_total(J.assign(prediction_kwh=J[c])[['group_id','actual_kwh','prediction_kwh']])['total']
for c in ['MYALL','LAWALL','DEPAVG']: print(f'  {c:8s} {tot(c):.6f}', flush=True)
def fo(members, grid):
    rows=[]
    for f in FOLDS:
        oth=J[J.fold_id!=f]; held=J[J.fold_id==f]; best=None
        for wv in grid:
            p=sum(w*oth[m] for w,m in zip(wv,members))
            t=official_total(oth.assign(prediction_kwh=p)[['group_id','actual_kwh','prediction_kwh']])['total']
            if best is None or t>best[0]: best=(t,wv)
        rows.append(held.assign(prediction_kwh=sum(w*held[m] for w,m in zip(best[1],members))))
    return official_total(pd.concat(rows,ignore_index=True)[['group_id','actual_kwh','prediction_kwh']])
g2=[(w,1-w) for w in np.arange(0,1.001,0.025)]
res={}
for a in list(dep)+['DEPAVG']:
    for b in mem+['MYALL','LAWALL']:
        res[f'{a}+{b}']=fo((a,b),g2)['total']
top=sorted(res.items(),key=lambda x:-x[1])[:12]
for k,v in top: print(f'  1dof {k:28s} {v:.6f}', flush=True)
bk=max(res,key=res.get); r=fo(tuple(bk.split('+')),g2)
print(f'BEST 1dof {bk} = {r["total"]:.6f} 1-NMAE={r["one_minus_nmae"]:.6f} FICR={r["ficr"]:.6f}', flush=True)
J.to_parquet(N+'S10_members_all3.parquet', index=False)
json.dump({'members':{c:tot(c) for c in mem+['MYALL','LAWALL','DEPAVG']},'blend_1dof':res,
           'best':[bk,r['total']]}, open(N+'S10_final5.json','w'), indent=1)
