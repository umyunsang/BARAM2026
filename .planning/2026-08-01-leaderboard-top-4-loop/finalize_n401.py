import sys, json
from pathlib import Path
sys.path.insert(0, "src")
from baram.loop.registry import SpecRegistry
from baram.loop import ontology as onto, router as routing
reg = SpecRegistry(Path("."))
s = reg.get("N401")
if s:
    s.status = "retired"
    s.outcome = {"accepted": False, "note": "CQR_DOES_NOT_BEAT_ACCEPTED",
                 "gain": -0.005965, "gate": "all four conditions FAIL"}
reg.save()
ontology = onto.load_ontology(Path("."))
results = [{"subcapability": x.subcapability} for x in reg.all() if x.status == "retired"]
state = {"cycle": 5, "best_score": 0.0, "stagnation": 4, "results": results,
         "research_runs": 1, "budget": {"target_score": 0.66, "max_cycles": 40}}
t, r = routing.route(state, ontology, reg.candidates())
print(f"후보 소진 후 라우터 판정: {t}\n사유: {r}")
print(f"남은 candidate: {[c.id for c in reg.candidates()]}")
print(f"score_bearing candidate: {[c.id for c in reg.candidates() if c.score_bearing]}")
