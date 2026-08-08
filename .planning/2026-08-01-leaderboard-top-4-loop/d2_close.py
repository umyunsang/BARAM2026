from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, "src")
from baram.loop.registry import SpecRegistry
reg = SpecRegistry(Path("."))
V = {
 "D2_G1_1": ("IMPLEMENTATION_DIFFERS_BUT_DATA_FORBIDS_PAPER_CONSTRUCTION",
   "C1N77 split features disjointly (gfs 50/101, ldaps 40/101) so each arm saw 40-50% of pooled "
   "information; the paper applies the same full pipeline per source. GFS and LDAPS expose different "
   "variable sets here (GFS has 80m/100m/PBL/upper-air, LDAPS has 50m max/min, radiation, cloud), so "
   "the paper construction cannot be reproduced without a common subset that reintroduces the loss."),
 "D2_G3_1": ("DIURNAL_BIAS_TOO_SMALL_OR_UNSTABLE",
   "hit rate 0.299-0.423 by hour, signed bias +0.0266..+0.0700 (max |bias| = 117% of the 0.06 band "
   "half-width) but fold-to-fold correlation of the hourly bias is only 0.220. Side finding: the "
   "signed bias is POSITIVE at all 24 hours, mean about +0.047 - a global over-prediction on eligible rows."),
 "D2_G2_1": ("WAKE_GEOMETRY_EXPLAINS_RESIDUAL_WEAKLY",
   "unwaked fraction varies strongly with direction (g3 spread 0.800; 80% waked at 150 deg). Residual "
   "correlation is significant for g2 (rho +0.1033) and g3 (rho +0.1347) against |error|, and negative "
   "against band hit (g3 -0.1415), i.e. the feature acts as a wind-regime proxy. Explained variance is "
   "only 1-2% and C1N3 already closed the sector axis with WIND_SECTOR_LOSS_IS_PROPORTIONAL."),
}
for k, (note, detail) in V.items():
    s = reg.get(k)
    if s:
        s.status = "retired"
        s.outcome = {"accepted": False, "note": note, "detail": detail}
reg.save()
print("D2 판정 기록:")
for k in V:
    s = reg.get(k)
    print(f"  {k:10s} {s.outcome['note']}")
