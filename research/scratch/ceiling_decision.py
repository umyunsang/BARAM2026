
"""How much score is still reachable from the DECISION layer alone, holding the current
point-forecast skill fixed?  Bucket rows by the point forecast and give each bucket the
in-sample-optimal constant action.  That is the information ceiling for `action = h(pc_hat)`."""
import sys, numpy as np, pandas as pd
sys.path.insert(0,'/Users/um-yunsang/BARAM2026/research/scratch')
from lib import official_total, FOLDS, CAPS
S='/Users/um-yunsang/BARAM2026/research/scratch/'
P=pd.concat([pd.read_parquet(S+f'v3_{f}.parquet') for f in FOLDS], ignore_index=True)
P['cf']=P.actual_kwh/P.group_id.map(CAPS)
ACT=np.arange(0.0,1.0801,0.0025)

def score_from(col):
    r=official_total(P.assign(prediction_kwh=P[col]*P.group_id.map(CAPS))[
        ['group_id','actual_kwh','prediction_kwh']])
    return r

def bucket_ceiling(nb, extra=None, label=''):
    P['a_opt']=np.nan
    for g in (1,2,3):
        m=P.group_id==g
        d=P[m]
        key=pd.qcut(d.pc_hat, nb, labels=False, duplicates='drop')
        if extra is not None: key=key.astype(str)+'|'+extra(d).astype(str)
        for k,idx in d.groupby(key).groups.items():
            cf=P.loc[idx,'cf'].to_numpy()
            valid=cf>=0.1
            err=np.abs(ACT[:,None]-cf[None,:])
            units=np.where(err<=0.06,4.0,np.where(err<=0.08,3.0,0.0))
            # exact pooled objective contribution: 0.5*(-mean err) + 0.5*(sum cf*units / (4*sum cf))
            n_all=len(cf)
            nmae_part=-(err.sum(axis=1))/n_all
            w=np.where(valid,cf,0.0)
            ficr_part=(units*w[None,:]).sum(axis=1)/(4.0*max(w.sum(),1e-9))
            # weight the two parts by their pooled shares (approx: equal 0.5/0.5 scaled locally)
            u=0.5*nmae_part + 0.5*ficr_part*(w.sum()/max(P.loc[P.group_id==g,'cf'].clip(lower=0).sum(),1e-9))*len(d)/n_all
            P.loc[idx,'a_opt']=ACT[int(np.argmax(u))]
    r=score_from('a_opt')
    print(f'{label or f"bucket({nb})":34s} total={r["total"]:.6f} 1-NMAE={r["one_minus_nmae"]:.6f} FICR={r["ficr"]:.6f}', flush=True)
    return r

print('reference:')
r=score_from('pc_hat'); print(f'{"raw pc_hat":34s} total={r["total"]:.6f} 1-NMAE={r["one_minus_nmae"]:.6f} FICR={r["ficr"]:.6f}')
r=official_total(P.rename(columns={'T0.5_G1.5':'prediction_kwh'})[['group_id','actual_kwh','prediction_kwh']])
print(f'{"v3 decision layer":34s} total={r["total"]:.6f} 1-NMAE={r["one_minus_nmae"]:.6f} FICR={r["ficr"]:.6f}')
print('\nIN-SAMPLE ceilings (optimistic by construction):')
for nb in (10,20,40,80,160):
    bucket_ceiling(nb)
bucket_ceiling(20, extra=lambda d: (d.kst_dtm.dt.hour//6), label='bucket(20) x hour-quarter')
