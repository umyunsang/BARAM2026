
import sys, numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0,'/Users/um-yunsang/BARAM2026/research/scratch')
from lib import official_total, FOLDS, CAPS

D = pd.read_parquet('/Users/um-yunsang/BARAM2026/research/scratch/scada_group_aggs.parquet')
lab = pd.read_parquet('/Users/um-yunsang/BARAM2026/research/scratch/labels.parquet').set_index('kst_dtm')
rng = np.random.default_rng(20260801)

def sweep(sigma):
    preds=[]
    for f,(a,b) in FOLDS.items():
        a=pd.Timestamp(a); b=pd.Timestamp(b)
        for g in (1,2,3):
            ws = D[f'g{g}_ws_mean']
            noisy = ws + rng.normal(0, sigma, len(ws))
            F = pd.DataFrame({'ws':noisy}, index=D.index)
            y = lab[f'kpx_group_{g}']
            tr=F.index[F.index<a]; va=F.index[(F.index>=a)&(F.index<=b)]
            Xtr=F.loc[tr]; ytr=y.reindex(tr); m=ytr.notna()&Xtr['ws'].notna(); Xtr,ytr=Xtr[m],ytr[m]
            Xva=F.loc[va]; yva=y.reindex(va); mv=yva.notna()&Xva['ws'].notna(); Xva,yva=Xva[mv],yva[mv]
            mdl=lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31,
                min_child_samples=60, random_state=20260801, n_jobs=6, verbose=-1)
            mdl.fit(Xtr,ytr)
            p=np.clip(mdl.predict(Xva),0,CAPS[g])
            preds.append(pd.DataFrame({'group_id':g,'actual_kwh':yva.to_numpy(float),'prediction_kwh':p}))
    P=pd.concat(preds,ignore_index=True); r=official_total(P)
    return r

for s in [0.0,0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.5]:
    r=sweep(s)
    print(f'sigma={s:4.2f}  total={r["total"]:.6f}  1-NMAE={r["one_minus_nmae"]:.6f}  FICR={r["ficr"]:.6f}')
