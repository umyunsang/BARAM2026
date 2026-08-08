
"""Fast Stage-A iteration harness: train < 2023-04-01, validate 2023-04-01..2023-12-31."""
import sys, json, time, numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0,'/Users/um-yunsang/BARAM2026/research/scratch')

CACHE='/Users/um-yunsang/BARAM2026/artifacts/cache/920be0c458d820e855bf79dd25723146f52ce1736138aedba5e6bc853f1f720b/'
SPLIT=pd.Timestamp('2023-04-01 01:00:00'); END=pd.Timestamp('2024-01-01 00:00:00')

def load_base():
    feat = pd.read_parquet(CACHE+'train_features.parquet')
    drop = {'forecast_id','issuance_batch','manufacturer','model','operating_day'}
    num=[c for c in feat.columns if c not in drop and feat[c].dtype.kind in 'ifb']
    return feat, num

LAG_BASE_PATTERNS = ('_spatial__idw__wind','_spatial__idw__heightAboveGround','_spatial__idw__etc_0_blh',
                     '_spatial__idw__surface_0_sp','_spatial__idw__surface_0_gust','phys__','phys_v2__')
LAG_EXTRA = ['gfs__wind80_speed__mean','gfs__wind100_speed__mean','gfs__pbl_wind_speed__mean',
             'gfs__wind850_speed__mean','ldaps__wind50max_speed__mean','ldaps__wind50min_speed__mean',
             'ldaps__wind10_speed__mean','ldaps__wind5_speed__mean','ldaps__etc_0_blh__mean',
             'gfs__surface_0_gust__mean']
LAGS=(-6,-3,-2,-1,1,2,3,6)

def add_lags(sub):
    """sub indexed by forecast_kst_dtm, one group; lags stay inside the issuance batch."""
    base=[c for c in sub.columns if any(p in c for p in LAG_BASE_PATTERNS)]
    base=[c for c in base if sub[c].dtype.kind in 'if']
    base=sorted(set(base+[c for c in LAG_EXTRA if c in sub.columns]))
    batch = sub['data_available_kst_dtm']
    out={}
    B=sub[base]
    for L in LAGS:
        sh=B.shift(L)
        same=batch.shift(L).eq(batch).to_numpy()
        sh=sh.mask(~pd.DataFrame(np.repeat(same[:,None], sh.shape[1], axis=1),
                                 index=sh.index, columns=sh.columns))
        for c in base:
            out[f'{c}__lag{L}']=sh[c]
            out[f'{c}__d{L}']=sub[c]-sh[c]
    return pd.DataFrame(out, index=sub.index), base

def evaluate(use_lags, tag, n_estimators=1500, colsample=0.5):
    feat, num = load_base()
    D = pd.read_parquet('/Users/um-yunsang/BARAM2026/research/scratch/scada_group_aggs.parquet')
    res={}
    for g in (1,2,3):
        sub = feat[feat.group_id==g].sort_values('forecast_kst_dtm').set_index('forecast_kst_dtm')
        cols=[c for c in num if c!='group_id']
        F = sub[cols]
        if use_lags:
            L,_ = add_lags(sub); F = pd.concat([F,L],axis=1)
        y = D[f'g{g}_ws_mean'].reindex(F.index)
        tr = F.index<SPLIT; va=(F.index>=SPLIT)&(F.index<=END)
        ytr=y[tr]; m=ytr.notna()
        t0=time.time()
        mdl=lgb.LGBMRegressor(n_estimators=n_estimators, learning_rate=0.03, num_leaves=63,
            min_child_samples=40, subsample=0.85, subsample_freq=1, colsample_bytree=colsample,
            reg_lambda=2.0, random_state=20260801, n_jobs=6, verbose=-1)
        mdl.fit(F[tr][m.to_numpy()], ytr[m])
        p=mdl.predict(F[va]); yv=y[va]; ok=yv.notna().to_numpy()
        e=p[ok]-yv[ok].to_numpy()
        res[g]=dict(rmse=float(np.sqrt((e**2).mean())), mae=float(np.abs(e).mean()),
                    n_feat=F.shape[1], secs=round(time.time()-t0,1))
        print(f'  {tag} g{g}: rmse={res[g]["rmse"]:.4f} mae={res[g]["mae"]:.4f} nfeat={F.shape[1]} {res[g]["secs"]}s', flush=True)
        pd.DataFrame({'kst_dtm':F.index[va],'pred':p,'true':yv.to_numpy()}).to_parquet(
            f'/Users/um-yunsang/BARAM2026/research/scratch/fastA_{tag}_g{g}.parquet', index=False)
    mean_rmse=float(np.mean([res[g]['rmse'] for g in res]))
    print(f'{tag}: MEAN RMSE={mean_rmse:.4f}', flush=True)
    return res, mean_rmse

if __name__=='__main__':
    which=sys.argv[1] if len(sys.argv)>1 else 'both'
    out={}
    if which in ('base','both'):
        out['base']=evaluate(False,'base')
    if which in ('lags','both'):
        out['lags']=evaluate(True,'lags')
    json.dump({k:(v[0],v[1]) for k,v in out.items()},
              open(f'/Users/um-yunsang/BARAM2026/research/scratch/fastA_{which}.json','w'), indent=1, default=str)
