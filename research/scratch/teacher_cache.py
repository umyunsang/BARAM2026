
"""One expensive pass that makes every downstream S5/S7 experiment instant.
For each fold x group: cross-fitted teacher predictions on the training window,
full-fit predictions on the validation window, plus the measured physics target
(for availability gating) and the true capacity factor."""
import sys, time, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import KFold
sys.path.insert(0,'/Users/um-yunsang/BARAM2026/research/scratch')
import featbuild
from lib import FOLDS, CAPS
S='/Users/um-yunsang/BARAM2026/research/scratch/'
T=pd.read_parquet(S+'teacher_targets.parquet')
LAB=pd.read_parquet(S+'labels.parquet').set_index('kst_dtm')
P=dict(objective='l2', n_estimators=700, learning_rate=0.04, num_leaves=63,
       min_child_samples=40, subsample=0.85, subsample_freq=1, colsample_bytree=0.35,
       reg_lambda=3.0, random_state=20260801, n_jobs=6, verbose=-1)
TARGETS=['pc','v_mean']

for f,(a,b) in FOLDS.items():
    a=pd.Timestamp(a); b=pd.Timestamp(b)
    for g in (1,2,3):
        X=featbuild.build(g)
        tr=np.asarray(X.index<a); va=np.asarray((X.index>=a)&(X.index<=b))
        base=dict(kst_dtm=X.index, split=np.where(tr,'train',np.where(va,'valid','none')))
        rec=pd.DataFrame(base)
        rec['cf']=(LAB[f'kpx_group_{g}']/CAPS[g]).reindex(X.index).to_numpy()
        rec['pc_true']=T[f'g{g}_pc'].reindex(X.index).to_numpy()
        rec['v_true']=T[f'g{g}_v_mean'].reindex(X.index).to_numpy()
        for tgt in TARGETS:
            y=T[f'g{g}_{tgt}'].reindex(X.index)
            ytr=y[tr]; ok=ytr.notna().to_numpy(); pos=np.flatnonzero(ok)
            oof=np.full(int(tr.sum()), np.nan); t0=time.time()
            for fi,hi in KFold(3, shuffle=True, random_state=20260801).split(pos):
                m=lgb.LGBMRegressor(**P); m.fit(X[tr].iloc[pos[fi]], ytr.iloc[pos[fi]])
                oof[pos[hi]]=m.predict(X[tr].iloc[pos[hi]])
            full=lgb.LGBMRegressor(**P); full.fit(X[tr][ok], ytr[ok])
            col=np.full(len(X), np.nan); col[tr]=oof; col[va]=full.predict(X[va])
            rec[f'hat_{tgt}']=col
            print(f'{f} g{g} {tgt} {round(time.time()-t0,1)}s', flush=True)
        rec[rec.split!='none'].to_parquet(S+f'cache_{f}_g{g}.parquet', index=False)
        del X
print('DONE', flush=True)
