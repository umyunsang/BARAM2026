
import sys, numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0,'/Users/um-yunsang/BARAM2026/research/scratch')
from lib import official_total, FOLDS, CAPS

X = pd.read_parquet('/Users/um-yunsang/BARAM2026/research/scratch/scada_hourly_features.parquet')
lab = pd.read_parquet('/Users/um-yunsang/BARAM2026/research/scratch/labels.parquet').set_index('kst_dtm')

GT = {1:[f'vestas_wtg{i:02d}' for i in range(1,7)],
      2:[f'vestas_wtg{i:02d}' for i in range(7,13)],
      3:[f'unison_wtg{i:02d}' for i in range(1,6)]}

# derived per-group aggregates of the measured field
D = pd.DataFrame(index=X.index)
for g,ts in GT.items():
    m  = X[[f'{t}_ws_m' for t in ts]]
    s  = X[[f'{t}_ws_s' for t in ts]]
    lo = X[[f'{t}_ws_lo' for t in ts]]
    hi = X[[f'{t}_ws_hi' for t in ts]]
    D[f'g{g}_ws_mean']    = m.mean(axis=1)
    D[f'g{g}_ws_spread']  = m.std(axis=1)          # across turbines
    D[f'g{g}_ws_intra']   = s.mean(axis=1)         # within-hour temporal std
    D[f'g{g}_ws_min']     = lo.min(axis=1)
    D[f'g{g}_ws_max']     = hi.max(axis=1)
    sn = X[[f'{t}_wd_sin' for t in ts]].mean(axis=1)
    cs = X[[f'{t}_wd_cos' for t in ts]].mean(axis=1)
    D[f'g{g}_wd_sin']=sn; D[f'g{g}_wd_cos']=cs
    D[f'g{g}_wd_consistency']=np.hypot(sn,cs)
D.to_parquet('/Users/um-yunsang/BARAM2026/research/scratch/scada_group_aggs.parquet')

def run(feat_of_group, tag):
    preds=[]
    for f,(a,b) in FOLDS.items():
        a=pd.Timestamp(a); b=pd.Timestamp(b)
        for g in (1,2,3):
            cols=feat_of_group(g)
            F = pd.concat([X,D],axis=1)[cols]
            y = lab[f'kpx_group_{g}']
            tr = F.index[F.index<a]; va = F.index[(F.index>=a)&(F.index<=b)]
            Xtr=F.loc[tr]; ytr=y.reindex(tr); m=ytr.notna(); Xtr,ytr=Xtr[m],ytr[m]
            Xva=F.loc[va]; yva=y.reindex(va); mv=yva.notna(); Xva,yva=Xva[mv],yva[mv]
            mdl=lgb.LGBMRegressor(n_estimators=800, learning_rate=0.05, num_leaves=31,
                min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                random_state=20260801, n_jobs=6, verbose=-1)
            mdl.fit(Xtr,ytr)
            p=np.clip(mdl.predict(Xva),0,CAPS[g])
            preds.append(pd.DataFrame({'fold_id':f,'group_id':g,'kst_dtm':Xva.index,
                'actual_kwh':yva.to_numpy(float),'prediction_kwh':p}))
    P=pd.concat(preds,ignore_index=True); r=official_total(P)
    print(f'{tag:34s} total={r["total"]:.6f}  1-NMAE={r["one_minus_nmae"]:.6f}  FICR={r["ficr"]:.6f}')
    return P,r

run(lambda g:[f'g{g}_ws_mean'], 'O1 group-mean ws only')
run(lambda g:[f'g{g}_ws_mean',f'g{g}_ws_intra',f'g{g}_ws_spread'], 'O2 +intra/spread')
run(lambda g:[f'g{g}_ws_mean',f'g{g}_ws_intra',f'g{g}_ws_spread',f'g{g}_ws_min',f'g{g}_ws_max',
              f'g{g}_wd_sin',f'g{g}_wd_cos',f'g{g}_wd_consistency'], 'O3 +minmax+dir')
run(lambda g:[c for c in X.columns if any(t in c for t in GT[g])], 'O4 per-turbine full')
