
"""S7/S10 · offline policy + blend search over saved conditional distributions.
Policy family (all 0-dof except the two scalars T, gamma which are selected fold-outside):
  cond  : renormalise the distribution onto cf >= 0.10 before taking the expected-utility argmax
          (the scorer discards every row with actual < 0.10*cap, so the argmax must condition on it)
"""
import sys, json, itertools
import numpy as np, pandas as pd
sys.path.insert(0,'/Users/um-yunsang/BARAM2026/research/scratch')
from lib import official_total, FOLDS, CAPS
N='/Users/um-yunsang/BARAM2026/research/nodes/'; AB='/Users/um-yunsang/BARAM2026/artifacts/backtests/metric-aligned-probe/'
W=0.04; ACT=np.arange(0.02,1.0801,0.0025); SC={1:0.985,2:0.989,3:1.005}
TEMPS=[0.6,1.0,1.5,2.0,2.5,3.0,4.0]; GAMMAS=[0.0,0.5,1.0,2.0,3.0,5.0,8.0,12.0]
key=['fold_id','group_id','forecast_kst_dtm']

def member(name, cond=True):
    R=pd.read_parquet(N+f'S7-N8_{name}_keys.parquet'); P=np.load(N+f'S7-N8_{name}_prob.npy')
    NC=P.shape[1]; C=(np.arange(NC)+0.5)*W
    err=np.abs(ACT[:,None]-C[None,:]); units=np.where(err<=0.06,4.,np.where(err<=0.08,3.,0.))
    g=R.group_id.to_numpy(); mg=R.mean_gen_g.to_numpy()
    capv=np.array([CAPS[x] for x in g]); hi=np.array([SC[x] for x in g]); act=R.cf.to_numpy()*capv
    mask=(C>=0.10).astype(float)
    frames={}
    for tp in TEMPS:
        q=P**(1.0/tp); q=q/np.maximum(q.sum(axis=1,keepdims=True),1e-12)
        if cond:
            q=q*mask[None,:]; q=q/np.maximum(q.sum(axis=1,keepdims=True),1e-12)
        nm=-(q@err.T); fic=(q@((C[None,:]*units).T))
        for gm in GAMMAS:
            pred=np.minimum(ACT[np.argmax(nm+gm*fic/(4.0*mg[:,None]),axis=1)],hi)*capv
            frames[(tp,gm)]=pred
    # fold-outside policy
    D=pd.DataFrame({'fold_id':R.fold_id,'group_id':g,'forecast_kst_dtm':R.forecast_kst_dtm,'actual_kwh':act})
    sc={k:official_total(D.assign(prediction_kwh=v)[['group_id','actual_kwh','prediction_kwh']])['total'] for k,v in frames.items()}
    out=np.empty(len(D))
    for f in FOLDS:
        sel=(D.fold_id==f).to_numpy()
        s2={k:official_total(D[~sel].assign(prediction_kwh=v[~sel])[['group_id','actual_kwh','prediction_kwh']])['total'] for k,v in frames.items()}
        pk=max(s2,key=s2.get); out[sel]=frames[pk][sel]
    D['prediction_kwh']=out
    fo=official_total(D[['group_id','actual_kwh','prediction_kwh']])
    return D, fo['total'], max(sc.values())

J=None; names=[]
for nm in ('P','L','G','Q','M'):
    for cond in (True,):
        D,fo,ins=member(nm,cond)
        print(f'member {nm} cond={cond}: fold-outside={fo:.6f} in-sample={ins:.6f}', flush=True)
        col=f'{nm}'
        d=D[key+['actual_kwh','prediction_kwh']].rename(columns={'prediction_kwh':col})
        J = d if J is None else J.merge(d.drop(columns=['actual_kwh']), on=key)
        names.append(col)
dep={'M102_TOP100':'T0.5_G1.5','M113_LGBM_DART':'T0.5_G0.5','M115_XGBOOST':'T0.6_G0.35'}
for stem,pol in dep.items():
    fr=[]
    for f in FOLDS:
        d=pd.read_parquet(AB+f'{stem}-{f}-policies.parquet'); d['fold_id']=f
        fr.append(d[key+[pol]].rename(columns={pol:stem}))
    J=J.merge(pd.concat(fr,ignore_index=True), on=key)
J['MY']=J[names].mean(axis=1)
tot=lambda c: official_total(J.assign(prediction_kwh=J[c])[['group_id','actual_kwh','prediction_kwh']])['total']
for c in names+['MY']+list(dep): print(f'  {c:16s} {tot(c):.6f}', flush=True)
def fo_blend(members, grid):
    rows=[]
    for f in FOLDS:
        oth=J[J.fold_id!=f]; held=J[J.fold_id==f]; best=None
        for wv in grid:
            p=sum(w*oth[m] for w,m in zip(wv,members))
            t=official_total(oth.assign(prediction_kwh=p)[['group_id','actual_kwh','prediction_kwh']])['total']
            if best is None or t>best[0]: best=(t,wv)
        rows.append(held.assign(prediction_kwh=sum(w*held[m] for w,m in zip(best[1],members))))
    return official_total(pd.concat(rows,ignore_index=True)[['group_id','actual_kwh','prediction_kwh']])
g2=[(w,1-w) for w in np.arange(0,1.001,0.05)]
best=(0,None,None)
for a in dep:
    for b in names+['MY']:
        r=fo_blend((a,b),g2)
        if r['total']>best[0]: best=(r['total'],(a,b),r)
        print(f'  blend {a}+{b}: {r["total"]:.6f}', flush=True)
g3=[(x,y,1-x-y) for x in np.arange(0,1.01,0.05) for y in np.arange(0,1.01-x+1e-9,0.05)]
for combo in [('M102_TOP100','M113_LGBM_DART','MY'),('M113_LGBM_DART','M115_XGBOOST','MY'),('M102_TOP100','M115_XGBOOST','MY')]:
    r=fo_blend(combo,g3)
    print(f'  3way {combo}: {r["total"]:.6f}', flush=True)
    if r['total']>best[0]: best=(r['total'],combo,r)
print('BEST', best[1], round(best[0],6), '1-NMAE', round(best[2]['one_minus_nmae'],6), 'FICR', round(best[2]['ficr'],6), flush=True)
J.to_parquet(N+'S10_members.parquet', index=False)
json.dump({'best_members':list(best[1]),'best_total':best[0]}, open(N+'S10_final2.json','w'), indent=1)
