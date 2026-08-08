"""단계 발굴 루프 실행기. 리서치 요청에서 멈추고, 충족되면 다음 단계로 내려간다."""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, "src")
from baram.loop.graph_staged import build_staged_graph, RECURSION_LIMIT
from baram.loop.registry import SpecRegistry
from baram.loop import stages as stg, research as rsch

ROOT = Path(".")
reg = SpecRegistry(ROOT)
app = build_staged_graph(ROOT, reg)
cfg = {"configurable": {"thread_id": "baram-staged-v1"}, "recursion_limit": RECURSION_LIMIT}

out = app.invoke({"cycle": 0, "best_score": 0.0, "stagnation": 0, "stage_history": {},
                  "budget": {"target_score": 0.66, "max_cycles": 60}}, cfg)
print("=== 실행 결과 ===")
print("stage:", out.get("stage"), "| depth:", out.get("depth"), "| phase:", out.get("phase"))
print("stop_reason:", out.get("stop_reason"))
print("\n원장:")
for e in (out.get("ledger") or [])[:8]:
    print("  ", json.dumps(e, ensure_ascii=False)[:130])
print("\n=== 단계 계획 ===")
for s in stg.STAGES:
    dep = "<-" + ",".join(s.depends_on) if s.depends_on else "(진입)"
    print(f"  {s.id} {s.category:8s} {dep:12s} 쿼리 {len(s.research_queries)}개")
req = rsch.request_path(ROOT, "D1")
if req.exists():
    d = json.loads(req.read_text(encoding="utf-8"))
    print(f"\n=== D1 리서치 요청서 ({req}) ===")
    for q in d["queries"]: print("  Q:", q)
    print("  계약:", json.dumps(d["contract"]["each_finding_requires"], ensure_ascii=False))
    print("  실행가능 태그:", d["contract"]["actionable_tags"])
