
"""S5 · 데이터전처리 — child nodes.  Each node changes exactly one declared treatment
against the frozen S5-N0 control.  Screening protocol; fold-outside confirmation later."""
import sys, json, numpy as np, pandas as pd
sys.path.insert(0,'/Users/um-yunsang/BARAM2026/research/nodes')
from harness import run, surface, DEFAULT

A,_,_ = surface()

def gap(df): return df['pc_true'].to_numpy() - df['cf'].to_numpy()
def valid(df): return np.isfinite(df['cf'].to_numpy()) & (df['cf'].to_numpy()>=0.1)

RES=[]
RES.append(run('S5-N0','control (no preprocessing treatment)'))

# P1 availability-deficit gating, 3 thresholds
for th in (0.25, 0.15, 0.10):
    RES.append(run(f'S5-N1_{th}', f'P1 gate: drop calib rows gap>={th}',
                   calib_rows=lambda d,t=th: ~(gap(d)>=t)))

# P3 metric-irrelevant down-weighting on the teacher
for a in (0.3, 0.0):
    RES.append(run(f'S5-N3_{a}', f'P3 teacher weight cf<0.1 -> {a}',
                   teacher_weight=lambda d,a=a: np.where(valid(d),1.0,a)))

# P4 production weighting on the teacher
RES.append(run('S5-N4','P4 teacher weight ∝ production',
               teacher_weight=lambda d: np.where(valid(d), np.clip(d['cf'].to_numpy(),0,1.2), 0.05)))

# P7 measured soft cap (g1/g2 hard ceiling 0.985/0.989, g3 1.005)
RES.append(run('S5-N7','P7 soft cap at measured ceiling',
               soft_cap={1:0.985,2:0.989,3:1.005}))

json.dump(RES, open('/Users/um-yunsang/BARAM2026/research/nodes/S5_batch1.json','w'), indent=1)
base=RES[0]['best']
print('\n=== S5 batch1 vs control ===', flush=True)
for r in RES:
    print(f'{r["node_id"]:12s} {r["best"]:.6f}  delta={r["best"]-base:+.6f}  {r["tag"]}', flush=True)
