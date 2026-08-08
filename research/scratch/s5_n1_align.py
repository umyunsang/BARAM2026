
"""S5-N1 / P10 — NWP valid-time alignment audit.  Zero degrees of freedom.
If the archive's forecast_kst_dtm is off by an hour relative to the hour-ending label
convention, every downstream model pays for it.  Scan shifts -3..+3."""
import sys, numpy as np, pandas as pd
sys.path.insert(0,'/Users/um-yunsang/BARAM2026/research/scratch')
import featbuild
S='/Users/um-yunsang/BARAM2026/research/scratch/'
T=pd.read_parquet(S+'teacher_targets.parquet')
LAB=pd.read_parquet(S+'labels.parquet').set_index('kst_dtm')
from lib import CAPS
PROBE=['ldaps_spatial__idw__wind50max_speed','gfs_spatial__idw__wind100_speed',
       'phys__hub117_speed','ldaps_spatial__idw__wind10_speed','atm__hub_consensus']
rows=[]
for g in (1,2,3):
    X=featbuild.build(g)
    v=T[f'g{g}_v_mean'].reindex(X.index)
    cf=(LAB[f'kpx_group_{g}']/CAPS[g]).reindex(X.index)
    for col in PROBE:
        s=X[col]
        for k in range(-3,4):
            sk=s.shift(k)
            r_ws=sk.corr(v); r_cf=sk.corr(cf)
            rows.append(dict(group=g,col=col,shift=k,corr_ws=r_ws,corr_cf=r_cf))
    del X
D=pd.DataFrame(rows)
piv=D.pivot_table(index=['col','shift'],columns='group',values='corr_ws')
piv['mean']=piv.mean(axis=1)
print('=== corr(NWP shifted by k, measured hub wind) — hour-ending label convention ===')
print(piv.round(4).to_string())
best=D.groupby(['group','col']).apply(lambda d: d.loc[d.corr_ws.idxmax(),'shift'], include_groups=False)
print('\nargmax shift per (group,col):'); print(best.to_string())
print('\n=== same against capacity factor ===')
p2=D.pivot_table(index=['col','shift'],columns='group',values='corr_cf'); p2['mean']=p2.mean(axis=1)
print(p2.round(4).to_string())
print('\nVERDICT:', 'shift 0 is optimal everywhere' if (best==0).all() else f'NON-ZERO SHIFT FOUND: {best[best!=0].to_dict()}')
