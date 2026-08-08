
"""S7 · 모델링 child nodes — conditional predictive distribution.
Frozen S5 treatment + S6-B1 block; only the calibrator changes."""
import sys, json, numpy as np, pandas as pd
sys.path.insert(0,'/Users/um-yunsang/BARAM2026/research/nodes')
from harness import run
def gap(d): return d['pc_true'].to_numpy()-d['cf'].to_numpy()
def valid(d): return np.isfinite(d['cf'].to_numpy())&(d['cf'].to_numpy()>=0.1)
S5=dict(calib_rows=lambda d: ~(gap(d)>=0.05), soft_cap={1:0.985,2:0.989,3:1.005},
        teacher_weight=lambda d: np.where(valid(d), np.clip(d['cf'].to_numpy(),0,1.2), 0.05),
        conditional_sigma=False, n_quantile=81)
RES=[]
RES.append(run('S7-N0','control: unconditional residual quantiles', blocks=('B1',), **S5))
for nb in (4,8,12,16,24):
    RES.append(run(f'S7-N1_{nb}', f'residual quantiles conditioned on pc_hat ({nb} buckets)',
                   blocks=('B1',), calib_buckets=nb, **S5))
json.dump(RES, open('/Users/um-yunsang/BARAM2026/research/nodes/S7_batch1.json','w'), indent=1)
base=RES[0]['best']
print('\n=== S7 batch1 ===', flush=True)
for r in RES: print(f'{r["node_id"]:12s} {r["best"]:.6f} delta={r["best"]-base:+.6f} '
                    f'1-NMAE={r["one_minus_nmae"]:.6f} FICR={r["ficr"]:.6f}  {r["tag"]}', flush=True)
