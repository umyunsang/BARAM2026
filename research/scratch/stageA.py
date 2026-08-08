
import sys, json, numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0,'/Users/um-yunsang/BARAM2026/research/scratch')
from lib import FOLDS, CAPS

CACHE='/Users/um-yunsang/BARAM2026/artifacts/cache/920be0c458d820e855bf79dd25723146f52ce1736138aedba5e6bc853f1f720b/'
feat = pd.read_parquet(CACHE+'train_features.parquet')
print('features', feat.shape)
D = pd.read_parquet('/Users/um-yunsang/BARAM2026/research/scratch/scada_group_aggs.parquet')

drop = {'forecast_kst_dtm','data_available_kst_dtm','forecast_id','issuance_batch',
        'manufacturer','model','operating_day'}
num = [c for c in feat.columns if c not in drop and feat[c].dtype.kind in 'ifb']
print('numeric feature count', len(num))

rows=[]
for g in (1,2,3):
    sub = feat[feat.group_id==g].set_index('forecast_kst_dtm')
    tgt = D[f'g{g}_ws_mean'].reindex(sub.index)
    rows.append((g, sub, tgt))

out={}
for g,sub,tgt in rows:
    for f,(a,b) in FOLDS.items():
        a=pd.Timestamp(a); b=pd.Timestamp(b)
        tr = sub.index[sub.index<a]; va = sub.index[(sub.index>=a)&(sub.index<=b)]
        ytr = tgt.reindex(tr); m = ytr.notna()
        mdl = lgb.LGBMRegressor(n_estimators=1500, learning_rate=0.03, num_leaves=63,
            min_child_samples=40, subsample=0.85, subsample_freq=1, colsample_bytree=0.7,
            reg_lambda=2.0, random_state=20260801, n_jobs=6, verbose=-1)
        mdl.fit(sub.loc[tr,num][m.to_numpy()], ytr[m])
        p = mdl.predict(sub.loc[va,num])
        yv = tgt.reindex(va)
        ok = yv.notna().to_numpy()
        err = p[ok]-yv[ok].to_numpy()
        out[(g,f)] = dict(rmse=float(np.sqrt((err**2).mean())), mae=float(np.abs(err).mean()),
                          bias=float(err.mean()), sd_true=float(yv[ok].std()),
                          r2=float(1-(err**2).mean()/yv[ok].var()))
        pd.DataFrame({'kst_dtm':va,'pred_ws':p,'true_ws':yv.to_numpy()}).to_parquet(
            f'/Users/um-yunsang/BARAM2026/research/scratch/stageA_g{g}_{f}.parquet', index=False)
        print(g, f, {k:round(v,4) for k,v in out[(g,f)].items()})
json.dump({f'g{k[0]}_{k[1]}':v for k,v in out.items()},
          open('/Users/um-yunsang/BARAM2026/research/scratch/stageA_metrics.json','w'), indent=1)
