
"""V3 end-to-end: NWP -> curve-integrated capacity factor -> conditional predictive
distribution -> settlement-optimal action.  Evaluated on the official local protocol
(pooled dev-2023 Q2/Q3/Q4, single policy)."""
from __future__ import annotations
import sys, json, time, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import KFold
sys.path.insert(0,'/Users/um-yunsang/BARAM2026/research/scratch')
import featbuild
from lib import official_total, FOLDS, CAPS, sharpen_weights

S='/Users/um-yunsang/BARAM2026/research/scratch/'
T=pd.read_parquet(S+'teacher_targets.parquet')
LAB=pd.read_parquet(S+'labels.parquet').set_index('kst_dtm')

TEACHER=dict(objective='l2', n_estimators=900, learning_rate=0.035, num_leaves=63,
             min_child_samples=40, subsample=0.85, subsample_freq=1, colsample_bytree=0.4,
             reg_lambda=3.0, random_state=20260801, n_jobs=6, verbose=-1)

ACTIONS=np.arange(0.05,1.0801,0.0025)
TEMPS=(0.4,0.5,0.6,0.75,0.85,1.0,1.2)
GAMMAS=(0.0,0.2,0.35,0.5,0.75,1.0,1.25,1.5,2.0)


def teacher_fold(g, tr_idx, va_idx, X, target='pc'):
    y=T[f'g{g}_{target}'].reindex(X.index)
    ytr=y[tr_idx]; ok=ytr.notna().to_numpy()
    pos=np.flatnonzero(ok)
    oof=np.full(int(np.sum(tr_idx)), np.nan)
    for fi,hi in KFold(3, shuffle=True, random_state=20260801).split(pos):
        m=lgb.LGBMRegressor(**TEACHER)
        m.fit(X[tr_idx].iloc[pos[fi]], ytr.iloc[pos[fi]])
        oof[pos[hi]]=m.predict(X[tr_idx].iloc[pos[hi]])
    full=lgb.LGBMRegressor(**TEACHER); full.fit(X[tr_idx][ok], ytr[ok])
    return oof, full.predict(X[va_idx])


def residual_distribution(pc_oof, cf_true, n_bucket=8, n_q=41):
    """Empirical conditional residual quantiles, bucketed by predicted level."""
    ok=np.isfinite(pc_oof)&np.isfinite(cf_true)
    p=pc_oof[ok]; a=cf_true[ok]
    edges=np.quantile(p, np.linspace(0,1,n_bucket+1)); edges[0]-=1e-6; edges[-1]+=1e-6
    qs=np.linspace(0.01,0.99,n_q)
    table=np.zeros((n_bucket,n_q))
    for b in range(n_bucket):
        m=(p>=edges[b])&(p<edges[b+1])
        if m.sum()<50: m=np.ones_like(p,bool)
        table[b]=np.quantile(a[m]-p[m], qs)
    return edges, table


def predictive_samples(pc_hat, edges, table):
    b=np.clip(np.searchsorted(edges, pc_hat, side='right')-1, 0, table.shape[0]-1)
    return np.clip(pc_hat[:,None]+table[b], 0.0, 1.05)


def act(samples, temp, gamma, mean_gen):
    """Settlement-optimal action over a per-row density-weighted sample distribution
    (DEF-1 fix: weights depend on each row's own sample values, so temp is no longer inert)."""
    w=sharpen_weights(samples, temp)[:,None,:]
    err=np.abs(ACTIONS[None,:,None]-samples[:,None,:])
    units=np.where(err<=0.06,4.0,np.where(err<=0.08,3.0,0.0))
    nmae_term=-(err*w).sum(axis=2)
    ficr_term=((samples[:,None,:]*units)*w).sum(axis=2)/(4.0*mean_gen)
    u=nmae_term+gamma*ficr_term
    return ACTIONS[np.argmax(u,axis=1)]


def run(tag='v3'):
    per_fold={}
    for f,(a,b) in FOLDS.items():
        a=pd.Timestamp(a); b=pd.Timestamp(b)
        parts=[]
        for g in (1,2,3):
            X=featbuild.build(g)
            tr=np.asarray(X.index<a); va=np.asarray((X.index>=a)&(X.index<=b))
            t0=time.time()
            oof, pv = teacher_fold(g, tr, va, X)
            cf_tr=(LAB[f'kpx_group_{g}'].reindex(X.index[tr])/CAPS[g]).to_numpy()
            edges,table=residual_distribution(oof, cf_tr)
            mean_gen=float(np.nanmean(cf_tr))
            pc_hat=np.clip(pv,0,1.0)
            samples=predictive_samples(pc_hat, edges, table)
            cf_va=(LAB[f'kpx_group_{g}'].reindex(X.index[va])/CAPS[g]).to_numpy()
            keep=np.isfinite(cf_va)
            rec={'group_id':g,'kst_dtm':X.index[va][keep],
                 'actual_kwh':cf_va[keep]*CAPS[g],'pc_hat':pc_hat[keep]}
            for tp in TEMPS:
                for gm in GAMMAS:
                    rec[f'T{tp}_G{gm}']=act(samples[keep],tp,gm,mean_gen)*CAPS[g]
            parts.append(pd.DataFrame(rec))
            print(f'  {f} g{g} teacher_mae={np.nanmean(np.abs(pc_hat[keep]-cf_va[keep])):.4f} '
                  f'{round(time.time()-t0,1)}s', flush=True)
            del X
        per_fold[f]=pd.concat(parts,ignore_index=True)
        per_fold[f].to_parquet(S+f'{tag}_{f}.parquet', index=False)
    pooled=pd.concat(per_fold.values(), ignore_index=True)
    sc={}
    for tp in TEMPS:
        for gm in GAMMAS:
            c=f'T{tp}_G{gm}'
            sc[c]=official_total(pooled.rename(columns={c:'prediction_kwh'})[
                ['group_id','actual_kwh','prediction_kwh']])
    ser=pd.Series({k:v['total'] for k,v in sc.items()}).sort_values(ascending=False)
    print(ser.head(10).to_string(), flush=True)
    best=ser.index[0]
    print(f'BEST {best}: total={sc[best]["total"]:.6f} 1-NMAE={sc[best]["one_minus_nmae"]:.6f} '
          f'FICR={sc[best]["ficr"]:.6f}', flush=True)
    raw=official_total(pooled.assign(prediction_kwh=pooled.pc_hat*pooled.group_id.map(CAPS))[
        ['group_id','actual_kwh','prediction_kwh']])
    print(f'RAW point (no decision layer): total={raw["total"]:.6f} 1-NMAE={raw["one_minus_nmae"]:.6f} '
          f'FICR={raw["ficr"]:.6f}', flush=True)
    json.dump({'best':best,'scores':{k:v['total'] for k,v in sc.items()},'raw':raw['total']},
              open(S+f'{tag}_scores.json','w'), indent=1)
    return pooled, sc

if __name__=='__main__':
    run(sys.argv[1] if len(sys.argv)>1 else 'v3')
