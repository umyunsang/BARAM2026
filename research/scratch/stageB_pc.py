
"""Stage A on the re-parameterised target: regress the curve-integrated capacity factor
`pc` directly from NWP, then emit pc_hat * capacity as the point forecast.
Fast protocol (train < 2023-04-01, validate 2023-04-01..2023-12-31) for iteration."""
import sys, json, time, numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0,'/Users/um-yunsang/BARAM2026/research/scratch')
import featbuild
from lib import official_total, CAPS

SPLIT=pd.Timestamp('2023-04-01 01:00:00'); END=pd.Timestamp('2024-01-01 00:00:00')
S='/Users/um-yunsang/BARAM2026/research/scratch/'
T=pd.read_parquet(S+'teacher_targets.parquet')
lab=pd.read_parquet(S+'labels.parquet').set_index('kst_dtm')

P_L2  = dict(objective='l2',  n_estimators=2000, learning_rate=0.025, num_leaves=63,
             min_child_samples=40, subsample=0.85, subsample_freq=1, colsample_bytree=0.5,
             reg_lambda=3.0, random_state=20260801, n_jobs=6, verbose=-1)
P_L1  = dict(P_L2, objective='l1')

def go(target, params, tag, transform=None):
    rows=[]; diag={}
    for g in (1,2,3):
        X=featbuild.build(g)
        y=T[f'g{g}_{target}'].reindex(X.index)
        tr=X.index<SPLIT; va=(X.index>=SPLIT)&(X.index<=END)
        m=y[tr].notna().to_numpy()
        mdl=lgb.LGBMRegressor(**params); mdl.fit(X[tr][m], y[tr][m])
        p=mdl.predict(X[va])
        yt=y[va].to_numpy(); ok=np.isfinite(yt)
        diag[g]=dict(mae=float(np.abs(p[ok]-yt[ok]).mean()), rmse=float(np.sqrt(((p[ok]-yt[ok])**2).mean())))
        a=lab[f'kpx_group_{g}'].reindex(X.index[va]).to_numpy()
        pred = np.clip(p,0,1)*CAPS[g] if transform is None else transform(np.clip(p,0,1))*CAPS[g]
        keep=np.isfinite(a)
        rows.append(pd.DataFrame({'group_id':g,'actual_kwh':a[keep],'prediction_kwh':pred[keep],
                                  'kst_dtm':X.index[va][keep], 'pc_hat':np.clip(p,0,1)[keep]}))
        del X, mdl
    P=pd.concat(rows,ignore_index=True)
    r=official_total(P[['group_id','actual_kwh','prediction_kwh']])
    print(f'{tag:34s} total={r["total"]:.6f} 1-NMAE={r["one_minus_nmae"]:.6f} FICR={r["ficr"]:.6f} '
          f'| teacher MAE(cf)={ {g:round(v["mae"],4) for g,v in diag.items()} }', flush=True)
    return P, r, diag

if __name__=='__main__':
    P1,_,_ = go('pc', P_L2, 'pc target, L2')
    P1.to_parquet(S+'B_pc_l2.parquet', index=False)
    P2,_,_ = go('pc', P_L1, 'pc target, L1')
    P2.to_parquet(S+'B_pc_l1.parquet', index=False)
