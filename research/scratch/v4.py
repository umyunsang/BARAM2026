
"""V4: S5 node runner.  Fixes DEF-1 (histogram the predictive samples into bins BEFORE
temperature sharpening, matching the repo's decision layer exactly) and runs the
S5-N1 availability-separated training variants against their control.

variants
  A  target=label cf, train on ALL rows,   residual table from ALL rows      [control]
  B  target=label cf, train on CLEAN rows, residual table from CLEAN rows    [S5-N1]
  C  target=label cf, train on CLEAN rows, residual table = clean + outage mixture
  D  target=pc (physics-denoised label), train on ALL rows, residuals from ALL rows
"""
from __future__ import annotations
import sys, json, time, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import KFold
sys.path.insert(0,'/Users/um-yunsang/BARAM2026/research/scratch')
import featbuild
from lib import official_total, FOLDS, CAPS

S='/Users/um-yunsang/BARAM2026/research/scratch/'
T=pd.read_parquet(S+'teacher_targets.parquet')
LAB=pd.read_parquet(S+'labels.parquet').set_index('kst_dtm')

TEACHER=dict(objective='l2', n_estimators=700, learning_rate=0.04, num_leaves=63,
             min_child_samples=40, subsample=0.85, subsample_freq=1, colsample_bytree=0.3,
             reg_lambda=3.0, random_state=20260801, n_jobs=6, verbose=-1)

CENTERS=np.arange(0.01,1.0601,0.02)          # 53 bins of width 0.02
ACTIONS=np.arange(0.05,1.0801,0.0025)
ERR=np.abs(ACTIONS[:,None]-CENTERS[None,:])
UNITS=np.where(ERR<=0.06,4.0,np.where(ERR<=0.08,3.0,0.0))
TEMPS=(0.4,0.5,0.6,0.75,0.85,1.0,1.2)
GAMMAS=(0.0,0.2,0.35,0.5,0.75,1.0,1.25,1.5,2.0)
OUTAGE_TH=-0.05


def to_prob(samples):
    """Histogram equal-weight predictive samples onto the 53-bin grid."""
    edges=np.concatenate([[ -1e9], CENTERS[:-1]+0.01, [1e9]])
    idx=np.searchsorted(edges, samples, side='right')-1
    idx=np.clip(idx,0,len(CENTERS)-1)
    P=np.zeros((samples.shape[0], len(CENTERS)))
    np.add.at(P, (np.repeat(np.arange(samples.shape[0]), samples.shape[1]), idx.ravel()), 1.0)
    return P/P.sum(axis=1, keepdims=True)


def act(P, temp, gamma, mean_gen):
    c=P**(1.0/temp); c=c/c.sum(axis=1, keepdims=True)
    u=-(c@ERR.T) + gamma*(c@(CENTERS[None,:]*UNITS).T)/(4.0*mean_gen)
    return ACTIONS[np.argmax(u,axis=1)]


def crossfit(X, tr, y, fitmask):
    pos=np.flatnonzero(fitmask)
    oof=np.full(int(tr.sum()), np.nan)
    Xtr=X[tr]
    for fi,hi in KFold(3, shuffle=True, random_state=20260801).split(pos):
        m=lgb.LGBMRegressor(**TEACHER); m.fit(Xtr.iloc[pos[fi]], y.iloc[pos[fi]])
        oof[pos[hi]]=m.predict(Xtr.iloc[pos[hi]])
    full=lgb.LGBMRegressor(**TEACHER); full.fit(Xtr[fitmask], y[fitmask])
    return oof, full


def resid_table(pred, actual, n_bucket=8, n_q=199):
    ok=np.isfinite(pred)&np.isfinite(actual)
    p=pred[ok]; a=actual[ok]
    edges=np.quantile(p, np.linspace(0,1,n_bucket+1)); edges[0]-=1e-6; edges[-1]+=1e-6
    qs=np.linspace(0.0025,0.9975,n_q)
    tab=np.zeros((n_bucket,n_q))
    for b in range(n_bucket):
        m=(p>=edges[b])&(p<edges[b+1])
        if m.sum()<60: m=np.ones_like(p,bool)
        tab[b]=np.quantile(a[m]-p[m], qs)
    return edges, tab


def samples_from(pred, edges, tab):
    b=np.clip(np.searchsorted(edges, pred, side='right')-1, 0, tab.shape[0]-1)
    return np.clip(pred[:,None]+tab[b], 0.0, 1.05)


def run(variant):
    per_fold={}
    for f,(a,b) in FOLDS.items():
        a=pd.Timestamp(a); b=pd.Timestamp(b); parts=[]
        for g in (1,2,3):
            t0=time.time()
            X=featbuild.build(g)
            tr=np.asarray(X.index<a); va=np.asarray((X.index>=a)&(X.index<=b))
            cf=(LAB[f'kpx_group_{g}'].reindex(X.index)/CAPS[g])
            pc=T[f'g{g}_pc'].reindex(X.index)
            r=(cf-pc)
            clean=(r>OUTAGE_TH).fillna(False).to_numpy()
            cf_tr=cf[tr]; pc_tr=pc[tr]
            valid=cf_tr.notna().to_numpy()
            if variant=='D':
                y=pc_tr; fitmask=pc_tr.notna().to_numpy()
            elif variant=='A':
                y=cf_tr; fitmask=valid
            else:
                y=cf_tr; fitmask=valid & clean[tr]
            oof, full = crossfit(X, tr, y, fitmask)
            pred_va=np.clip(full.predict(X[va]),0,1.05)
            act_tr=cf_tr.to_numpy()
            if variant in ('A','D'):
                edges,tab=resid_table(oof, act_tr)
            elif variant=='B':
                edges,tab=resid_table(np.where(clean[tr],oof,np.nan), np.where(clean[tr],act_tr,np.nan))
            else:  # C  mixture
                e1,t1=resid_table(np.where(clean[tr],oof,np.nan), np.where(clean[tr],act_tr,np.nan))
                bad=~clean[tr]
                e2,t2=resid_table(np.where(bad,oof,np.nan), np.where(bad,act_tr,np.nan), n_bucket=1)
                q=float(np.nanmean(bad[valid]))
                n2=max(int(round(t1.shape[1]*q/(1-q))),1)
                t2s=np.quantile(t2[0], np.linspace(0.01,0.99,n2))
                tab=np.concatenate([t1, np.tile(t2s,(t1.shape[0],1))],axis=1); edges=e1
            samples=samples_from(pred_va, edges, tab)
            P=to_prob(samples)
            mean_gen=float(np.nanmean(act_tr))
            cf_va=cf[va].to_numpy(); keep=np.isfinite(cf_va)
            rec={'group_id':g,'kst_dtm':X.index[va][keep],'actual_kwh':cf_va[keep]*CAPS[g],
                 'point':pred_va[keep]}
            for tp in TEMPS:
                for gm in GAMMAS:
                    rec[f'T{tp}_G{gm}']=act(P[keep],tp,gm,mean_gen)*CAPS[g]
            parts.append(pd.DataFrame(rec))
            print(f'  [{variant}] {f} g{g} mae={np.nanmean(np.abs(pred_va[keep]-cf_va[keep])):.4f} '
                  f'fit_rows={int(fitmask.sum())} {round(time.time()-t0,1)}s', flush=True)
            del X, full
        per_fold[f]=pd.concat(parts,ignore_index=True)
    pooled=pd.concat(per_fold.values(), ignore_index=True)
    pooled.to_parquet(S+f'v4_{variant}.parquet', index=False)
    sc={}
    for tp in TEMPS:
        for gm in GAMMAS:
            c=f'T{tp}_G{gm}'
            sc[c]=official_total(pooled.rename(columns={c:'prediction_kwh'})[['group_id','actual_kwh','prediction_kwh']])
    ser=pd.Series({k:v['total'] for k,v in sc.items()}).sort_values(ascending=False)
    best=ser.index[0]
    raw=official_total(pooled.assign(prediction_kwh=pooled.point*pooled.group_id.map(CAPS))[
        ['group_id','actual_kwh','prediction_kwh']])
    print(f'[{variant}] BEST {best}: total={sc[best]["total"]:.6f} 1-NMAE={sc[best]["one_minus_nmae"]:.6f} '
          f'FICR={sc[best]["ficr"]:.6f} | raw point total={raw["total"]:.6f}', flush=True)
    print(ser.head(6).round(6).to_string(), flush=True)
    json.dump({'variant':variant,'best':best,'best_score':sc[best]['total'],
               'one_minus_nmae':sc[best]['one_minus_nmae'],'ficr':sc[best]['ficr'],
               'raw_point':raw['total'],'grid':{k:v['total'] for k,v in sc.items()}},
              open(S+f'v4_{variant}.json','w'), indent=1)

if __name__=='__main__':
    for v in sys.argv[1:]: run(v)
