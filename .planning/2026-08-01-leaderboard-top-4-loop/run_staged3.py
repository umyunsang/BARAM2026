from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, "src")
from baram.loop.graph_staged import build_staged_graph, RECURSION_LIMIT
from baram.loop.registry import SpecRegistry
ROOT = Path(".")
reg = SpecRegistry(ROOT)
app = build_staged_graph(ROOT, reg)
cfg = {"configurable": {"thread_id": "baram-staged-v3"}, "recursion_limit": RECURSION_LIMIT}
out = app.invoke({"cycle": 0, "best_score": 0.0, "stagnation": 0, "stage_history": {},
                  "budget": {"target_score": 0.66, "max_cycles": 60}}, cfg)
print("stage:", out.get("stage"), "| stop:", out.get("stop_reason"))
