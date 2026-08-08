
"""S10-FINAL2 · pooled member library + fold-outside blend, with an explicit dof ladder."""
import sys, json, itertools
import numpy as np, pandas as pd
sys.path.insert(0,'/Users/um-yunsang/BARAM2026/research/scratch')
from lib import official_total, FOLDS, CAPS
N='/Users/um-yunsang/BARAM2026/research/nodes/'; AB='/Users/um-yunsang/BARAM2026/artifacts/backtests/metric-aligned-probe/'
W=0.04; ACT=np.arange(0.02,1.0801,0.0025); SC={1:0.985,2:0.989,3:1.005}
TEMPS=[0.6,1.0,1.5,2.0,2.5,3.0,4.0]; GAMMAS=[0.0,0.5,1.0,2.0,3.0,5.0,8.0,12.0]
key=['fold_id','group_id','forecast_kst_dtm']
def member(name):
    R=pd.read_parquet(N+f'S7-N8_{name}_keys.parquet'); P=np.load(N+f'S7-N8_{name}_prob.npy')
    NC=P.shape[1]; C=(np.arange(NC)+0.5)*W
    err=np.abs(ACT[:,None]-C[None,:]); units=np.where(err<=0.06,4.,np.where(err<=0.08,3.,0.))
    g=R.group_id.to_numpy(); mg=R.mean_gen_g.to_numpy()
    capv=np.array([CAPS[x] for x in g]); hi=np.array([SC[x] for x in g]); act=R.cf.to_numpy()*capv
    mask=(C>=0.10).astype(float); frames={}
    for tp in TEMPS:
        q=P**(1.0/tp); q=q/np.maximum(q.sum(axis=1,keepdims=True),1e-12)
        q=q*mask[None,:]; q=q/np.maximum(q.sum(axis=1,keepdims=True),1e-12)
        nm=-(q@err.T); fic=(q@((C[None,:]*units).T))
        for gm in GAMMAS:
            frames[(tp,gm)]=np.minimum(ACT[np.argmax(nm+gm*fic/(4.0*mg[:,None]),axis=1)],hi)*capv
    D=pd.DataFrame({'fold_id':R.fold_id,'group_id':g,'forecast_kst_dtm':R.forecast_kst_dtm,'actual_kwh':act})
    out=np.empty(len(D))
    for f in FOLDS:
        sel=(D.fold_id==f).to_numpy()
        s2={k:official_total(D[~sel].assign(prediction_kwh=v[~sel])[['group_id','actual_kwh','prediction_kwh']])['total'] for k,v in frames.items()}
        out[sel]=frames[max(s2,key=s2.get)][sel]
    D['prediction_kwh']=out
    return D, official_total(D[['group_id','actual_kwh','prediction_kwh']])['total']
J=None; names=[]
for nm in ('P','L','G','Q','M','D','X','M2'):
    D,fo=member(nm); print(f'member {nm}: fold-outside={fo:.6f}', flush=True)
    d=D[key+['actual_kwh','prediction_kwh']].rename(columns={'prediction_kwh':nm})
    J = d if J is None else J.merge(d.drop(columns=['actual_kwh']), on=key); names.append(nm)
dep={'M102_TOP100':'T0.5_G1.5','M113_LGBM_DART':'T0.5_G0.5','M115_XGBOOST':'T0.6_G0.35'}
for stem,pol in dep.items():
    fr=[]
    for f in FOLDS:
        d=pd.read_parquet(AB+f'{stem}-{f}-policies.parquet'); d['fold_id']=f
        fr.append(d[key+[pol]].rename(columns={pol:stem}))
    J=J.merge(pd.concat(fr,ignore_index=True), on=key)
J['MYALL']=J[names].mean(axis=1)
J['DEPAVG']=J[list(dep)].mean(axis=1)
tot=lambda c: official_total(J.assign(prediction_kwh=J[c])[['group_id','actual_kwh','prediction_kwh']])['total']
for c in names+['MYALL','DEPAVG']+list(dep): print(f'  {c:16s} {tot(c):.6f}', flush=True)
cap=J.group_id.map(CAPS)
print('corr(MYALL, DEPAVG) =', round(float(np.corrcoef((J.MYALL-J.actual_kwh)/cap,(J.DEPAVG-J.actual_kwh)/cap)[0,1]),4), flush=True)
def fo_blend(members, grid):
    rows=[]; picks=[]
    for f in FOLDS:
        oth=J[J.fold_id!=f]; held=J[J.fold_id==f]; best=None
        for wv in grid:
            p=sum(w*oth[m] for w,m in zip(wv,members))
            t=official_total(oth.assign(prediction_kwh=p)[['group_id','actual_kwh','prediction_kwh']])['total']
            if best is None or t>best[0]: best=(t,wv)
        picks.append(best[1])
        rows.append(held.assign(prediction_kwh=sum(w*held[m] for w,m in zip(best[1],members))))
    D=pd.concat(rows,ignore_index=True)
    return official_total(D[['group_id','actual_kwh','prediction_kwh']]), picks, D
g2=[(w,1-w) for w in np.arange(0,1.001,0.05)]
res={}
print('\n--- 1 dof ---', flush=True)
for a in list(dep)+['DEPAVG']:
    for b in ['MYALL','M','X','D','G','Q']:
        r,pk,_=fo_blend((a,b),g2); res[f'{a}+{b}']=r['total']
for k,v in sorted(res.items(),key=lambda x:-x[1])[:8]: print(f'  {k:26s} {v:.6f}', flush=True)
best1=max(res,key=res.get)
print('\n--- 2 dof ---', flush=True)
g3=[(x,y,1-x-y) for x in np.arange(0,1.01,0.05) for y in np.arange(0,1.01-x+1e-9,0.05)]
res3={}
for combo in [('M102_TOP100','M115_XGBOOST','MYALL'),('DEPAVG','MYALL','M'),
              ('M102_TOP100','M115_XGBOOST','M'),('DEPAVG','M','X'),('DEPAVG','MYALL','X')]:
    r,pk,D=fo_blend(combo,g3); res3[str(combo)]=r['total']
    print(f'  {combo}: {r["total"]:.6f} picks={pk}', flush=True)
bk=max(res3,key=res3.get)
print(f'\nBEST 1dof {best1} = {res[best1]:.6f}', flush=True)
print(f'BEST 2dof {bk} = {res3[bk]:.6f}', flush=True)
json.dump({'members':{c:tot(c) for c in names+['MYALL','DEPAVG']+list(dep)},
           'blend_1dof':res,'blend_2dof':res3,'best_1dof':[best1,res[best1]],'best_2dof':[bk,res3[bk]]},
          open(N+'S10_final3.json','w'), indent=1)
J.to_parquet(N+'S10_members_all.parquet', index=False)
