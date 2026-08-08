
"""S6 batch2 (C01/C02 defect fix) + S7 batch2 (C14 variance inflation)."""
import sys, json, numpy as np, pandas as pd
sys.path.insert(0,'/Users/um-yunsang/BARAM2026/research/nodes')
from harness import run
def gap(d): return d['pc_true'].to_numpy()-d['cf'].to_numpy()
def valid(d): return np.isfinite(d['cf'].to_numpy())&(d['cf'].to_numpy()>=0.1)
S5=dict(calib_rows=lambda d: ~(gap(d)>=0.05), soft_cap={1:0.985,2:0.989,3:1.005},
        teacher_weight=lambda d: np.where(valid(d), np.clip(d['cf'].to_numpy(),0,1.2), 0.05),
        conditional_sigma=False, n_quantile=81)
RES=[]
RES.append(run('S6-N0b','control = S5 + B1', blocks=('B1',), **S5))
RES.append(run('S6-C01','+G2 true-geometry grid block (old grid kept)', blocks=('B1','G2'), **S5))
RES.append(run('S6-C02','G2 replaces defective grid__ block', blocks=('B1','G2','DROP:grid__'), **S5))
RES.append(run('S6-C02b','G2 only, drop grid__ and b1__ dup', blocks=('G2','DROP:grid__'), **S5))
for b in (1.05,1.10,1.15,1.25):
    RES.append(run(f'S7-C14_{b}', f'variance inflation {b}', blocks=('B1',), var_inflate=b, **S5))
json.dump(RES, open('/Users/um-yunsang/BARAM2026/research/nodes/S6S7_batch2.json','w'), indent=1)
base=RES[0]['best']
print('\n=== batch2 vs control(S5+B1) ===', flush=True)
for r in RES: print(f'{r["node_id"]:14s} {r["best"]:.6f} delta={r["best"]-base:+.6f} '
                    f'raw={r["raw_point"]:.6f} 1-NMAE={r["one_minus_nmae"]:.6f} FICR={r["ficr"]:.6f}  {r["tag"]}', flush=True)
