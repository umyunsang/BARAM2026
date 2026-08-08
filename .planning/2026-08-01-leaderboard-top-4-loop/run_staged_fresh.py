from __future__ import annotations
import sys, json, uuid
from pathlib import Path
sys.path.insert(0, "src")
from baram.loop.graph_staged import build_staged_graph, RECURSION_LIMIT
from baram.loop.registry import SpecRegistry
ROOT = Path(".")
reg = SpecRegistry(ROOT)
app = build_staged_graph(ROOT, reg)
tid = "baram-staged-" + uuid.uuid4().hex[:8]
cfg = {"configurable": {"thread_id": tid}, "recursion_limit": RECURSION_LIMIT}
out = app.invoke({"cycle": 0, "best_score": 0.0, "stagnation": 0, "stage_history": {},
                  "budget": {"target_score": 0.66, "max_cycles": 60}}, cfg)
print("thread:", tid, "| stage:", out.get("stage"), "| stop:", out.get("stop_reason"))
h = out.get("stage_history", {}).get("D1", {})
print("D1 done:", h.get("done"), "| verdicts:", len((h.get("summary") or {}).get("verdicts", [])))
