
import sys, numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0,'/Users/um-yunsang/BARAM2026/research/scratch')
from lib import official_total, FOLDS, CAPS

X = pd.read_parquet('/Users/um-yunsang/BARAM2026/research/scratch/scada_hourly_features.parquet')
lab = pd.read_parquet('/Users/um-yunsang/BARAM2026/research/scratch/labels.parquet').set_index('kst_dtm')

def run(feat_cols, tag, groups=(1,2,3)):
    preds = []
    for f,(a,b) in FOLDS.items():
        a=pd.Timestamp(a); b=pd.Timestamp(b)
        for g in groups:
            y = lab[f'kpx_group_{g}']
            tr_idx = X.index[(X.index < a)]
            va_idx = X.index[(X.index >= a) & (X.index <= b)]
            Xtr = X.loc[tr_idx, feat_cols]; ytr = y.reindex(tr_idx)
            m = ytr.notna() & Xtr.notna().any(axis=1)
            Xtr, ytr = Xtr[m], ytr[m]
            Xva = X.loc[va_idx, feat_cols]; yva = y.reindex(va_idx)
            mv = yva.notna()
            Xva, yva = Xva[mv], yva[mv]
            mdl = lgb.LGBMRegressor(n_estimators=800, learning_rate=0.05, num_leaves=31,
                                    min_child_samples=40, subsample=0.8, subsample_freq=1,
                                    colsample_bytree=0.8, random_state=20260801, n_jobs=6, verbose=-1)
            mdl.fit(Xtr, ytr)
            p = np.clip(mdl.predict(Xva), 0, CAPS[g])
            preds.append(pd.DataFrame({'fold_id':f,'group_id':g,'kst_dtm':Xva.index,
                                       'actual_kwh':yva.to_numpy(float),'prediction_kwh':p}))
    P = pd.concat(preds, ignore_index=True)
    r = official_total(P)
    print(f'{tag}: total={r["total"]:.6f} 1-NMAE={r["one_minus_nmae"]:.6f} FICR={r["ficr"]:.6f}  gnmae={ {k:round(v,4) for k,v in r["group_nmae"].items()} }')
    return P, r

allc = list(X.columns)
ve_g1 = [c for c in allc if any(f'vestas_wtg{i:02d}' in c for i in range(1,7))]
ve_g2 = [c for c in allc if any(f'vestas_wtg{i:02d}' in c for i in range(7,13))]
un_g3 = [c for c in allc if 'unison' in c]

P_all, r_all = run(allc, 'ORACLE_SCADA_ALL17')
P_all.to_parquet('/Users/um-yunsang/BARAM2026/research/scratch/oracle_scada_all.parquet')
