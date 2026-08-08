"""N403 admission: pretrained time-series foundation models.

Value proposition is temporal/sequence structure. The project already measured the
ORACLE bound on that axis, which caps any method that exploits it.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, "src")
from baram.loop.registry import SpecRegistry

reg = SpecRegistry(Path("."))
verdict = {
    "spec": "N403",
    "premise": "pretrained TS foundation models exploit temporal structure the row-wise model misses",
    "prior_evidence": [
        "progress.md 2026-08-04: within-issuance residual autocorrelation 0.818 (lag1), "
        "between-issuance residual variance 30.0/32.7/30.4% — the structure is large and real",
        "progress.md 2026-08-04: an ORACLE per-issuance offset correction moved 1-NMAE +0.020365 "
        "and FICR -0.014849, for a Total change of only +0.002758",
        "mechanism: the deployed prediction is an ACTION under a step reward, not a conditional mean; "
        "moving it toward the conditional mean improves point accuracy and damages settlement by construction",
    ],
    "oracle_bound_total": 0.002758,
    "requirement_total": 0.023473,
    "test": "does the oracle bound on the temporal axis reach the requirement?",
    "finding": "0.002758 < 0.023473 — a PERFECT exploitation of temporal structure covers 11.7% of the gap",
    "result": "ADMISSION_FAILED_ORACLE_BOUND_BELOW_REQUIREMENT",
    "note": "compute/licence gates were not even reached; the axis is capped on evidence regardless",
}
s = reg.get("N403")
if s:
    s.status = "retired"
    s.outcome = {"accepted": False, "note": verdict["result"], "detail": verdict}
reg.save()
print(json.dumps(verdict, ensure_ascii=False, indent=1)[:1200])
print("\n남은 score_bearing 후보:", [c.id for c in reg.candidates() if c.score_bearing])
