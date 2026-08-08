import sys, json
from pathlib import Path
sys.path.insert(0, "src")
from baram.loop.registry import SpecRegistry, NodeSpec
reg = SpecRegistry(Path("."))
SB = {"N301": True, "N302": True, "N303": True, "N307": True,
      "N304": False, "N305": False, "N306": False, "N308": False}
for sid, flag in SB.items():
    s = reg.get(sid)
    if s: s.score_bearing = flag
for s in reg.all():
    if s.status == "retired": s.score_bearing = True
reg.save()
print("score_bearing 태깅:", {k: v for k, v in SB.items()})
