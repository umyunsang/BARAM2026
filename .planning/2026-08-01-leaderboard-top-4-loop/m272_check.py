
import sys, json, glob
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT/"src"))
exec(open(ROOT/".planning/2026-08-01-leaderboard-top-4-loop/m272_online_equivalent_policy.py").read().split("def main()")[0].replace('if __name__','#'))
store = load(str(ROOT/"artifacts/cache/m271_decision_surface/994ae6dff5796332daf21a6f"))

def per_fold(T,G):
    out={}
    for f in FOLDS:
        c=store[f]; rate=c["meta"]["actual_kwh"].to_numpy(float)/c["cap"]; ok=np.isfinite(rate)
        norms={int(g):float(np.nanmean(rate[(c["group"]==g)&ok])) for g in np.unique(c["group"])}
        pred=decide(c["prob"],c["group"],T,G,norms)*c["cap"]
        d=c["meta"][["forecast_id","forecast_kst_dtm","group_id","actual_kwh"]].copy(); d["prediction_kwh"]=pred
        s=evaluate_official(d[ok],CAPACITIES_KWH); out[f]=float(s.total)
    return out

cands={"배포 T0.5_G1.5":(0.5,1.5),"기존격자최적 T1.2_G2":(1.2,2.0),"확장최적 T2_G4":(2.0,4.0),
       "T1.6_G3":(1.6,3.0),"T2.5_G4":(2.5,4.0),"T3_G6":(3.0,6.0)}
print(f"{'정책':22s} {'Q2':>9} {'Q3':>9} {'Q4':>9}   {'pooled':>9}")
res={}
for n,(T,G) in cands.items():
    pf=per_fold(T,G); res[n]=pf
    print(f"{n:22s} {pf['dev-2023-Q2']:9.6f} {pf['dev-2023-Q3']:9.6f} {pf['dev-2023-Q4']:9.6f}")
base=res["배포 T0.5_G1.5"]
print("\n배포 대비 폴드별 델타 (3/3 양수여야 견고):")
for n,pf in res.items():
    if n.startswith("배포"): continue
    d=[pf[f]-base[f] for f in FOLDS]
    print(f"  {n:22s} {d[0]:+.6f} {d[1]:+.6f} {d[2]:+.6f}   양수 {sum(x>0 for x in d)}/3")
print("\n온도축 단조성 (G=4 고정):")
for T in (0.5,0.75,1.0,1.2,1.4,1.6,2.0,2.5,3.0,4.0):
    pf=per_fold(T,4.0); print(f"  T={T:<4} pooled_folds mean={np.mean(list(pf.values())):.6f}")
