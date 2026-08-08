from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, "src")
from baram.loop.registry import SpecRegistry
reg = SpecRegistry(Path("."))
s = reg.get("D4_J1_1")
s.status = "retired"
s.outcome = {"accepted": False, "note": "GATE_POWER_CONSISTENT_WITH_NOMINAL",
 "detail": "Injecting known-size genuine improvements (pred' = actual + k*(pred-actual)) into the deployed "
           "prediction and applying the frozen gate unmodified: k=0.995 (0.5% error reduction, delta "
           "+0.000972) already passes all four conditions. Measured detection limit +0.000972 matches the "
           "nominal 0.001013 threshold. The gate is NOT discarding real gains; it screens on month-to-month "
           "consistency rather than magnitude, which is why M284 (+0.001207, positive in only 4 of 9 months, "
           "median -0.001235) was rejected while a same-size consistent gain passes."}
reg.save()
print("D4_J1_1 ->", s.outcome["note"])
print("\n채택(accepted=True) 노드:")
for x in reg.all():
    if (x.outcome or {}).get("accepted"):
        print(f"  {x.id:24s} {x.status:9s} {(x.outcome or {}).get('note')}")
