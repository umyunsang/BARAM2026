
"""S11 · 분석방법의 적절성 — information-floor test.
If two hours have (near) identical NWP states but different outcomes, no function of the NWP
can separate them.  Estimate that irreducible dispersion directly by k-nearest-neighbour
conditional MAD inside the training period, excluding temporally adjacent hours (+-72 h) so
that autocorrelation cannot masquerade as state similarity.
Then convert the required score gain into a required pc-MAE and compare."""
import sys, json
import numpy as np, pandas as pd
from sklearn.neighbors import NearestNeighbors
sys.path.insert(0,'/Users/um-yunsang/BARAM2026/research/scratch')
sys.path.insert(0,'/Users/um-yunsang/BARAM2026/research/nodes')
from harness import surface
N='/Users/um-yunsang/BARAM2026/research/nodes/'

A,FR,COLS=surface(('G2','DROP:grid__'))
CORE=['atm__hub_consensus','ldaps_spatial__idw__wind50max_speed','gfs_spatial__idw__wind100_speed',
      'ldaps_spatial__idw__wind50max_dir_sin','ldaps_spatial__idw__wind50max_dir_cos',
      'ldaps_spatial__idw__etc_0_blh','atm__alpha_100_80','atm__theta850_minus_t2',
      'g2__l50x__rng','g2__g100__mean','ldaps_spatial__idw__heightAboveGround_2_t',
      'cal__doy_sin','cal__doy_cos','cal__hour_sin','cal__hour_cos']
CORE=[c for c in CORE if c in A.columns]
res={}
for g in (1,2,3):
    m=(A['grp'].to_numpy()==g)&np.isfinite(A['pc_true'].to_numpy())&(A.index<pd.Timestamp('2024-01-01'))
    X=A.loc[m,CORE].to_numpy('float64'); y=A.loc[m,'pc_true'].to_numpy(); t=A.index[m].values.astype('datetime64[h]').astype(np.int64)
    X=np.nan_to_num(X, nan=np.nanmedian(X))
    Z=(X-X.mean(0))/np.maximum(X.std(0),1e-6)
    nn=NearestNeighbors(n_neighbors=60).fit(Z)
    dist,ind=nn.kneighbors(Z)
    out={}
    for k in (3,5,10,20):
        mads=[]; rads=[]
        for i in range(len(Z)):
            cand=ind[i][np.abs(t[ind[i]]-t[i])>72]
            if len(cand)<k: continue
            sel=cand[:k]
            mads.append(np.abs(y[sel]-y[i]).mean()*k/(k+1))
            rads.append(np.linalg.norm(Z[sel[-1]]-Z[i]))
        out[k]=dict(cond_mad=float(np.mean(mads)), n=len(mads), mean_radius=float(np.mean(rads)))
    res[g]=out
    print(f'g{g}: ' + '  '.join(f'k={k}: MAD={v["cond_mad"]:.4f} r={v["mean_radius"]:.2f}' for k,v in out.items()), flush=True)
mean_by_k={k: float(np.mean([res[g][k]['cond_mad'] for g in (1,2,3)])) for k in (3,5,10,20)}
print('mean conditional MAD by k:', {k:round(v,4) for k,v in mean_by_k.items()}, flush=True)
json.dump(dict(per_group=res, mean_by_k=mean_by_k, core_features=CORE),
          open(N+'S11_information_floor.json','w'), indent=1)
