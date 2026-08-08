
"""S6 · 피처구성 child nodes.  Frozen S5 treatment; exactly one feature block changes per node."""
import sys, json, numpy as np, pandas as pd
sys.path.insert(0,'/Users/um-yunsang/BARAM2026/research/nodes')
from harness import run, surface
def gap(d): return d['pc_true'].to_numpy()-d['cf'].to_numpy()
def valid(d): return np.isfinite(d['cf'].to_numpy())&(d['cf'].to_numpy()>=0.1)
S5=dict(calib_rows=lambda d: ~(gap(d)>=0.05), soft_cap={1:0.985,2:0.989,3:1.005},
        teacher_weight=lambda d: np.where(valid(d), np.clip(d['cf'].to_numpy(),0,1.2), 0.05),
        conditional_sigma=False, n_quantile=81)
RES=[]
RES.append(run('S6-N0','S5 treatment, no new block (control)', blocks=(), **S5))
for b in ('B1','B3','B10','B2'):
    RES.append(run(f'S6-N_{b}', f'+block {b}', blocks=(b,), **S5))
json.dump(RES, open('/Users/um-yunsang/BARAM2026/research/nodes/S6_batch1.json','w'), indent=1)
base=RES[0]['best']
print('\n=== S6 batch1 (control = S5 treatment) ===', flush=True)
for r in RES: print(f'{r["node_id"]:12s} {r["best"]:.6f} delta={r["best"]-base:+.6f} raw={r["raw_point"]:.6f}  {r["tag"]}', flush=True)
