"""Admission gate: a candidate must survive a check against recorded evidence
before it can be dispatched. Introduced after N301 was routed into an axis that
C1N87/C1N89 had already closed."""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, "src")
from baram.loop.registry import SpecRegistry

reg = SpecRegistry(Path("."))

# N404 admission: its revival premise is "public forecast archives are now permitted",
# i.e. an alternative NWP source is obtainable. M281 measured that premise directly.
verdict = {
    "spec": "N404",
    "premise": "alternative NWP source usable => ramp/weather-pattern lane revived",
    "test": "M281 — Open-Meteo leak-safe variable (previous_day2) coverage by year",
    "finding": "2023 (validation surface) EMPTY for both icon_global and ecmwf_ifs025; "
               "2025 full. A member built on external NWP cannot be validated on any "
               "leak-safe surface (2024 is the twice-consumed lockbox).",
    "result": "ADMISSION_FAILED_PREMISE_UNDERMINED",
}
s = reg.get("N404")
if s:
    s.status = "retired"
    s.outcome = {"accepted": False, "note": verdict["result"], "detail": verdict}
reg.save()
print(json.dumps(verdict, ensure_ascii=False, indent=1))
print("\n남은 후보:")
for c in reg.candidates():
    print(f"   {c.id} [{c.subcapability}] score_bearing={c.score_bearing}  {c.summary[:52]}")
