
"""Decisive test: is the measured-wind oracle inflated by availability leaking through the
own-group nacelle anemometers? Predict each group's power from OTHER groups' nacelle wind only."""
import sys, numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0,'/Users/um-yunsang/BARAM2026/research/scratch')
from lib import official_total, FOLDS, CAPS

X = pd.read_parquet('/Users/um-yunsang/BARAM2026/research/scratch/scada_hourly_features.parquet')
D = pd.read_parquet('/Users/um-yunsang/BARAM2026/research/scratch/scada_group_aggs.parquet')
lab = pd.read_parquet('/Users/um-yunsang/BARAM2026/research/scratch/labels.parquet').set_index('kst_dtm')
GT = {1:[f'vestas_wtg{i:02d}' for i in range(1,7)],
      2:[f'vestas_wtg{i:02d}' for i in range(7,13)],
      3:[f'unison_wtg{i:02d}' for i in range(1,6)]}
F = pd.concat([X, D], axis=1)

def run(colsel, tag):
    preds=[]
    for f,(a,b) in FOLDS.items():
        a=pd.Timestamp(a); b=pd.Timestamp(b)
        for g in (1,2,3):
            cols = colsel(g)
            Fg = F[cols]
            y = lab[f'kpx_group_{g}']
            tr=Fg.index[Fg.index<a]; va=Fg.index[(Fg.index>=a)&(Fg.index<=b)]
            Xtr=Fg.loc[tr]; ytr=y.reindex(tr); m=ytr.notna()&Xtr.notna().any(axis=1); Xtr,ytr=Xtr[m],ytr[m]
            Xva=Fg.loc[va]; yva=y.reindex(va); mv=yva.notna(); Xva,yva=Xva[mv],yva[mv]
            mdl=lgb.LGBMRegressor(n_estimators=800, learning_rate=0.05, num_leaves=31,
                min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                random_state=20260801, n_jobs=6, verbose=-1)
            mdl.fit(Xtr,ytr)
            p=np.clip(mdl.predict(Xva),0,CAPS[g])
            preds.append(pd.DataFrame({'group_id':g,'actual_kwh':yva.to_numpy(float),'prediction_kwh':p}))
    P=pd.concat(preds,ignore_index=True); r=official_total(P)
    print(f'{tag:38s} total={r["total"]:.6f}  1-NMAE={r["one_minus_nmae"]:.6f}  FICR={r["ficr"]:.6f}', flush=True)
    return P,r

own      = lambda g: [c for c in X.columns if any(t in c for t in GT[g])]
other    = lambda g: [c for c in X.columns if any(t in c for t in sum((GT[k] for k in (1,2,3) if k!=g), []))]
othermean= lambda g: [f'g{k}_ws_mean' for k in (1,2,3) if k!=g]
ownmean  = lambda g: [f'g{g}_ws_mean']
allmean  = lambda g: [f'g{k}_ws_mean' for k in (1,2,3)]

run(own,       'A own-group per-turbine (leaky)')
run(ownmean,   'B own-group mean scalar (leaky)')
run(other,     'C OTHER-groups per-turbine (clean)')
run(othermean, 'D OTHER-groups mean scalars (clean)')
run(allmean,   'E all-three mean scalars')
