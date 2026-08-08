
"""S5 batch 3 — combine the winners, then close the stage with a declared best treatment."""
import sys, json, numpy as np, pandas as pd
sys.path.insert(0,'/Users/um-yunsang/BARAM2026/research/nodes')
from harness import run, surface
A,_,_=surface()
def gap(d): return d['pc_true'].to_numpy()-d['cf'].to_numpy()
def valid(d): return np.isfinite(d['cf'].to_numpy())&(d['cf'].to_numpy()>=0.1)
PW=lambda d: np.where(valid(d), np.clip(d['cf'].to_numpy(),0,1.2), 0.05)
SC={1:0.985,2:0.989,3:1.005}
RES=[]
for th in (0.03,0.05,0.075,0.10,0.15):
    RES.append(run(f'S5-N14_{th}', f'P1({th})+P7+P4+uncond-sigma',
                   calib_rows=lambda d,t=th: ~(gap(d)>=t), soft_cap=SC,
                   teacher_weight=PW, conditional_sigma=False))
RES.append(run('S5-N15','P1(0.05)+P7+P4+cond-sigma',
               calib_rows=lambda d: ~(gap(d)>=0.05), soft_cap=SC,
               teacher_weight=PW, conditional_sigma=True))
RES.append(run('S5-N18','P1(0.05)+P7+P4+uncond+n_q=81',
               calib_rows=lambda d: ~(gap(d)>=0.05), soft_cap=SC,
               teacher_weight=PW, conditional_sigma=False, n_quantile=81))
json.dump(RES, open('/Users/um-yunsang/BARAM2026/research/nodes/S5_batch3.json','w'), indent=1)
print('\n=== S5 batch3 (control 0.595309, best-so-far 0.608618) ===', flush=True)
for r in RES: print(f'{r["node_id"]:14s} {r["best"]:.6f} delta={r["best"]-0.595309:+.6f}  {r["tag"]}', flush=True)
