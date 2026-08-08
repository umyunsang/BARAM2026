
"""S7-N11 · average the DISTRIBUTIONS, not the actions.
Under a step reward the optimal action is argmax_a E[R(|a-y|)] over the predictive law.
Averaging member ACTIONS is not that operation; averaging member LAWS and taking one argmax is."""
import sys, json, itertools
import numpy as np, pandas as pd
sys.path.insert(0,'/Users/um-yunsang/BARAM2026/research/scratch')
from lib import official_total, FOLDS, CAPS
N='/Users/um-yunsang/BARAM2026/research/nodes/'
W=0.04; ACT=np.arange(0.02,1.0801,0.0025); SC={1:0.985,2:0.989,3:1.005}
TEMPS=[0.6,1.0,1.5,2.0,2.5,3.0,4.0,5.0]; GAMMAS=[0.0,0.5,1.0,2.0,3.0,5.0,8.0,12.0]
NAMES=['P','L','G','Q','M','D','X','M2']
R=pd.read_parquet(N+'S7-N8_P_keys.parquet')
PROB={n: np.load(N+f'S7-N8_{n}_prob.npy') for n in NAMES}
for n in NAMES:
    k=pd.read_parquet(N+f'S7-N8_{n}_keys.parquet')
    assert (k['forecast_kst_dtm'].values==R['forecast_kst_dtm'].values).all() and (k.group_id.values==R.group_id.values).all(), n
NC=PROB['P'].shape[1]; C=(np.arange(NC)+0.5)*W
err=np.abs(ACT[:,None]-C[None,:]); units=np.where(err<=0.06,4.,np.where(err<=0.08,3.,0.))
g=R.group_id.to_numpy(); mg=R.mean_gen_g.to_numpy()
capv=np.array([CAPS[x] for x in g]); hi=np.array([SC[x] for x in g]); act=R.cf.to_numpy()*capv
mask=(C>=0.10).astype(float)
D0=pd.DataFrame({'fold_id':R.fold_id,'group_id':g,'forecast_kst_dtm':R.forecast_kst_dtm,'actual_kwh':act})

def decide(P, cond=True):
    frames={}
    for tp in TEMPS:
        q=P**(1.0/tp); q=q/np.maximum(q.sum(axis=1,keepdims=True),1e-12)
        if cond:
            q=q*mask[None,:]; q=q/np.maximum(q.sum(axis=1,keepdims=True),1e-12)
        nm=-(q@err.T); fic=(q@((C[None,:]*units).T))
        for gm in GAMMAS:
            frames[(tp,gm)]=np.minimum(ACT[np.argmax(nm+gm*fic/(4.0*mg[:,None]),axis=1)],hi)*capv
    out=np.empty(len(D0)); picks={}
    for f in FOLDS:
        sel=(D0.fold_id==f).to_numpy()
        s2={k:official_total(D0[~sel].assign(prediction_kwh=v[~sel])[['group_id','actual_kwh','prediction_kwh']])['total'] for k,v in frames.items()}
        pk=max(s2,key=s2.get); picks[f]=pk; out[sel]=frames[pk][sel]
    ins=max(official_total(D0.assign(prediction_kwh=v)[['group_id','actual_kwh','prediction_kwh']])['total'] for v in frames.values())
    return out, official_total(D0.assign(prediction_kwh=out)[['group_id','actual_kwh','prediction_kwh']]), ins, picks

res={}
for n in NAMES:
    _,r,ins,_=decide(PROB[n]); res[n]=r['total']
print('single members:', {k:round(v,6) for k,v in res.items()}, flush=True)
combos={
 'ALL8': NAMES,
 'GBDT4': ['P','L','G','Q'],
 'DIVERSE5': ['P','G','D','X','M'],
 'DIVERSE6': ['P','G','Q','D','X','M'],
 'TOP4': ['X','D','G','P'],
 'NOMLP': ['P','L','G','Q','D','X'],
}
best=(0,None)
for nm,mem in combos.items():
    Pm=np.mean([PROB[k] for k in mem],axis=0)
    out,r,ins,picks=decide(Pm)
    print(f'  law-average {nm:10s} ({len(mem)}): fold-outside={r["total"]:.6f} in-sample={ins:.6f} '
          f'1-NMAE={r["one_minus_nmae"]:.6f} FICR={r["ficr"]:.6f} picks={list(picks.values())}', flush=True)
    if r['total']>best[0]: best=(r['total'],nm,out)
    D0[f'LAW_{nm}']=out
# geometric (log) pooling as an alternative to linear pooling
for nm,mem in [('ALL8',NAMES),('DIVERSE6',['P','G','Q','D','X','M'])]:
    Pg=np.exp(np.mean([np.log(np.clip(PROB[k],1e-9,None)) for k in mem],axis=0))
    Pg=Pg/Pg.sum(axis=1,keepdims=True)
    out,r,ins,_=decide(Pg)
    print(f'  log-pool    {nm:10s}: fold-outside={r["total"]:.6f}', flush=True)
    D0[f'LOG_{nm}']=out
    if r['total']>best[0]: best=(r['total'],'LOG_'+nm,out)
print('BEST law-pooled member:', best[1], round(best[0],6), flush=True)
D0.to_parquet(N+'S7-N11_lawpool.parquet', index=False)
json.dump({'singles':res,'best':[best[1],best[0]]}, open(N+'S7-N11.json','w'), indent=1)
