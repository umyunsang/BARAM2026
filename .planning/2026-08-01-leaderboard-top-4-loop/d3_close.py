from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, "src")
from baram.loop.registry import SpecRegistry
reg = SpecRegistry(Path("."))
V = {
 "D3_H2_1": ("SMOOTH_MEMBER_EXISTS_PREMISE_HOLDS",
   "unique-value ratio audit over 14 stored members: 11 discrete (0.009-0.032), 2 near-continuous "
   "(M113 0.174, M115 0.210), 1 smooth (M244 0.966). Browell's smooth counterpart already exists as "
   "the analog lineage, which also explains why M244 is the only decorrelated member (0.8436)."),
 "D3_H2_2": ("UNCONDITIONAL_RESIDUAL_DECISION_LAYER_HURTS",
   "wrapping M244 point predictions in a fold-outside empirical residual distribution and running the "
   "expected-utility decision drops Total 0.605760 -> 0.581140 (FICR 0.3506 -> 0.3026). The residual "
   "quantiles are identical for every row, so the distribution carries no conditional information."),
 "D3_H2_3": ("REACHABILITY_SWITCH_BELOW_UNIFORM_BLEND",
   "per-row member switching on predicted band-hit probability gains +0.003131 over all-discrete "
   "(best tau 0.40) but loses -0.001148 to the uniform w=0.7 blend, because switching uses one member "
   "per row and therefore gets no variance reduction."),
 "D3_H2_4": ("CONDITIONAL_WEIGHT_BELOW_THRESHOLD_AND_NOT_DEPLOYABLE",
   "reachability-conditional weighting keeps both members and beats the uniform blend by +0.000835 "
   "fold-outside (82% of the 0.001013 threshold), with in-sample->fold-outside decay of only -0.0018. "
   "NOT DEPLOYABLE: p_band requires the 46-bin distribution on the 2025 test period, which is persisted "
   "nowhere and which the M261 builder does not save. A full-history refit would be required."),
 "D3_H3_1": ("GLOBAL_SHIFT_NO_GAIN",
   "shifting predictions by the -0.047 bias found in D2_G3_1 drops Total 0.628337 -> 0.603634 "
   "(FICR 0.4023 -> 0.3506). The positive bias is the decision layer's intended action under "
   "generation-weighted settlement, not an error. Independently reconfirms C1N22 on eligible rows."),
}
for k, (note, detail) in V.items():
    s = reg.get(k)
    if s is None:
        continue
    s.status = "retired"
    s.outcome = {"accepted": False, "note": note, "detail": detail}
reg.save()
print("D3 판정 기록:")
for k in V:
    s = reg.get(k)
    if s: print(f"  {k:10s} {s.outcome['note']}")
