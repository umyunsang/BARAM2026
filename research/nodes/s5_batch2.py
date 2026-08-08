
"""S5 batch 2 — refine the winning treatment and test the lane's declared interaction rules."""
import sys, json, numpy as np, pandas as pd
sys.path.insert(0,'/Users/um-yunsang/BARAM2026/research/nodes')
from harness import run, surface
A,_,_ = surface()
def gap(d): return d['pc_true'].to_numpy()-d['cf'].to_numpy()
def valid(d): return np.isfinite(d['cf'].to_numpy())&(d['cf'].to_numpy()>=0.1)
CAPS_SOFT={1:0.985,2:0.989,3:1.005}
RES=[]
for th in (0.05,0.075):
    RES.append(run(f'S5-N1_{th}', f'P1 gate th={th}', calib_rows=lambda d,t=th: ~(gap(d)>=t)))
RES.append(run('S5-N8','P1(0.10) + P7 softcap',
               calib_rows=lambda d: ~(gap(d)>=0.10), soft_cap=CAPS_SOFT))
RES.append(run('S5-N9','P1(0.10) + P7 + P4 prodweight',
               calib_rows=lambda d: ~(gap(d)>=0.10), soft_cap=CAPS_SOFT,
               teacher_weight=lambda d: np.where(valid(d), np.clip(d['cf'].to_numpy(),0,1.2), 0.05)))
RES.append(run('S5-N12','control, unconditional sigma', conditional_sigma=False))
RES.append(run('S5-N13','P1(0.10)+P7, unconditional sigma',
               calib_rows=lambda d: ~(gap(d)>=0.10), soft_cap=CAPS_SOFT, conditional_sigma=False))
for off in (0.01,0.02):
    RES.append(run(f'S5-N11_{off}', f'P1(0.10)+P7 + g3 offset {off}',
                   calib_rows=lambda d: ~(gap(d)>=0.10), soft_cap=CAPS_SOFT,
                   group_offset={1:0.0,2:0.0,3:off}))
json.dump(RES, open('/Users/um-yunsang/BARAM2026/research/nodes/S5_batch2.json','w'), indent=1)
print('\n=== S5 batch2 (control S5-N0 = 0.595309) ===', flush=True)
for r in RES: print(f'{r["node_id"]:12s} {r["best"]:.6f} delta={r["best"]-0.595309:+.6f}  {r["tag"]}', flush=True)
