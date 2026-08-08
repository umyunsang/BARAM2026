
"""S7b: optimise the pooled teacher. Evaluate pc-MAE on METRIC-VALID rows (cf>=0.1),
which is the surface the official score actually uses."""
import sys, json, time, numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0,'/Users/um-yunsang/BARAM2026/research/scratch')
import featbuild
from lib import CAPS
S='/Users/um-yunsang/BARAM2026/research/scratch/'
T=pd.read_parquet(S+'teacher_targets.parquet')
LAB=pd.read_parquet(S+'labels.parquet').set_index('kst_dtm')
SPLIT=pd.Timestamp('2023-04-01 01:00:00'); END=pd.Timestamp('2024-01-01 00:00:00')

print('building features...', flush=True)
FR={}
for g in (1,2,3):
    X=featbuild.build2(g, geom=True, grid=True)
    for k in (1,2,3): X[f'is_g{k}']=np.float32(k==g)
    X['y']=T[f'g{g}_pc'].reindex(X.index).to_numpy()
    X['cf']=(LAB[f'kpx_group_{g}']/CAPS[g]).reindex(X.index).to_numpy()
    FR[g]=X
COLS=[c for c in FR[1].columns if c not in ('y','cf')]
A=pd.concat(FR.values())
TR=np.asarray(A.index<SPLIT)
print('pooled rows', A.shape, flush=True)

def evaluate(preds):
    """preds: dict g -> array on validation rows."""
    m=[]
    for g in (1,2,3):
        X=FR[g]; va=np.asarray((X.index>=SPLIT)&(X.index<=END))
        yt=X.loc[va,'y'].to_numpy(); cf=X.loc[va,'cf'].to_numpy()
        ok=np.isfinite(yt)&np.isfinite(cf)&(cf>=0.1)
        m.append(float(np.abs(preds[g][ok]-yt[ok]).mean()))
    return m, float(np.mean(m))

def fit_predict(params, weight=None, rows=None, seeds=(20260801,)):
    m=np.isfinite(A['y'].to_numpy())&TR
    if rows is not None: m = m & rows
    Xtr=A.loc[m,COLS]; ytr=A.loc[m,'y']
    w=None if weight is None else weight[m]
    acc={g:[] for g in (1,2,3)}
    for sd in seeds:
        p=dict(params); p['random_state']=sd
        mdl=lgb.LGBMRegressor(**p); mdl.fit(Xtr,ytr,sample_weight=w)
        for g in (1,2,3):
            X=FR[g]; va=np.asarray((X.index>=SPLIT)&(X.index<=END))
            acc[g].append(np.clip(mdl.predict(X.loc[va,COLS]),0,1))
    return {g:np.mean(acc[g],axis=0) for g in (1,2,3)}, int(m.sum())

BASE=dict(objective='l2', n_estimators=900, learning_rate=0.035, num_leaves=63,
          min_child_samples=40, subsample=0.85, subsample_freq=1, colsample_bytree=0.4,
          reg_lambda=3.0, n_jobs=6, verbose=-1)
CFA=A['cf'].to_numpy()
W_valid=np.where(np.isfinite(CFA)&(CFA>=0.1), 1.0, 0.15)
W_prod =np.where(np.isfinite(CFA)&(CFA>=0.1), np.clip(CFA,0,1.2), 0.05)

TRIALS={
 'ref            ': (BASE, None, None, (20260801,)),
 'big            ': (dict(BASE,n_estimators=3000,learning_rate=0.015,num_leaves=127,colsample_bytree=0.3), None, None, (20260801,)),
 'big_l1         ': (dict(BASE,objective='l1',n_estimators=3000,learning_rate=0.015,num_leaves=127,colsample_bytree=0.3), None, None, (20260801,)),
 'big_wvalid     ': (dict(BASE,n_estimators=3000,learning_rate=0.015,num_leaves=127,colsample_bytree=0.3), W_valid, None, (20260801,)),
 'big_wprod      ': (dict(BASE,n_estimators=3000,learning_rate=0.015,num_leaves=127,colsample_bytree=0.3), W_prod, None, (20260801,)),
 'big_bag4       ': (dict(BASE,n_estimators=3000,learning_rate=0.015,num_leaves=127,colsample_bytree=0.3), None, None, (1,2,3,4)),
 'huge           ': (dict(BASE,n_estimators=6000,learning_rate=0.008,num_leaves=255,colsample_bytree=0.25,min_child_samples=60), None, None, (20260801,)),
}
out={}
store={}
for k,(pr,w,rows,seeds) in TRIALS.items():
    t0=time.time(); preds,n=fit_predict(pr,w,rows,seeds)
    per,mean=evaluate(preds); out[k]=dict(per=per,mean=mean,rows=n,secs=round(time.time()-t0,1))
    store[k]=preds
    print(f'{k} valid pc-MAE per-group={[round(x,5) for x in per]} MEAN={mean:.5f} rows={n} {out[k]["secs"]}s', flush=True)
# blend of the diverse members
for combo in [('big            ','big_l1         '),('big            ','big_l1         ','big_wprod      '),
              ('big_bag4       ','big_l1         '),('huge           ','big_l1         ','big_bag4       ')]:
    preds={g:np.mean([store[c][g] for c in combo],axis=0) for g in (1,2,3)}
    per,mean=evaluate(preds)
    print(f'BLEND {"+".join(c.strip() for c in combo):40s} MEAN={mean:.5f}', flush=True)
    out['BLEND '+'+'.join(c.strip() for c in combo)]=dict(per=per,mean=mean)
json.dump(out, open(S+'sweep3.json','w'), indent=1)
np.save(S+'sweep3_preds.npy', {k:{g:v[g] for g in (1,2,3)} for k,v in store.items()}, allow_pickle=True)
