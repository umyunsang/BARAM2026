from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, "src")
from baram.loop.graph_staged import build_staged_graph, RECURSION_LIMIT
from baram.loop.registry import SpecRegistry
from baram.loop import stages as stg

ROOT = Path(".")
reg = SpecRegistry(ROOT)
app = build_staged_graph(ROOT, reg)
cfg = {"configurable": {"thread_id": "baram-staged-v2"}, "recursion_limit": RECURSION_LIMIT}
out = app.invoke({"cycle": 0, "best_score": 0.0, "stagnation": 0, "stage_history": {},
                  "budget": {"target_score": 0.66, "max_cycles": 60}}, cfg)
print("stage:", out.get("stage"), "| depth:", out.get("depth"), "| phase:", out.get("phase"))
print("stop_reason:", out.get("stop_reason"))
print("\n원장:")
for e in (out.get("ledger") or [])[:12]:
    print("  ", json.dumps(e, ensure_ascii=False)[:150])
print("\n=== D1 단계에서 생성된 노드 ===")
for s in reg.all():
    if s.arguments.get("stage") == "D1":
        print(f"  {s.id:12s} [{s.subcapability}] {s.status:10s} {s.summary[:52]}")
        print(f"  {'':12s}  origin: {s.origin[:88]}")
print("\n후보 총 " + str(len(reg.candidates())) + "건")
