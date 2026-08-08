"""Run the routed loop. The router picks the node; I do not."""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, "src")
from baram.loop import build_graph, SpecRegistry
from baram.loop.graph import RECURSION_LIMIT
from baram.loop import ontology as onto, router as routing

ROOT = Path(".")
reg = SpecRegistry(ROOT)
ontology = onto.load_ontology(ROOT)

print("=== 라우터 입력 상태 ===")
results = [{"subcapability": s.subcapability} for s in reg.all() if s.status == "retired"]
load = onto.category_load(ontology, results)
print("카테고리 부하(온톨로지 증거 + 이번 세션):")
for k, v in sorted(load.items(), key=lambda x: -x[1]):
    print(f"   {k:8s} {v:3d}  {'█'*min(v,45)}")

state = {"cycle": 0, "best_score": 0.0, "stagnation": 11,
         "results": results, "research_runs": 1,
         "budget": {"target_score": 0.66, "max_cycles": 40}}

target, reason = routing.route(state, ontology, reg.candidates())
print(f"\n>>> 라우터 판정: {target}   사유: {reason}")
if target == "dispatch":
    spec = routing.select_spec(state, ontology, reg.candidates())
    print(f">>> 선택 스펙: {spec.id} [{spec.subcapability}] {spec.summary}")

print("\n=== 정체 트리거 무효화 시 (stagnation=0) 라우터가 고르는 것 ===")
s2 = dict(state); s2["stagnation"] = 0
t2, r2 = routing.route(s2, ontology, reg.candidates())
print(f">>> {t2}  사유: {r2}")
if t2 == "dispatch":
    sp = routing.select_spec(s2, ontology, reg.candidates())
    print(f">>> 선택 스펙: {sp.id} [{sp.subcapability}] {sp.summary}")

print("\n=== 그래프 컴파일 및 스모크 실행 ===")
app = build_graph(ROOT, reg)
cfg = {"configurable": {"thread_id": "baram-loop-v1"}, "recursion_limit": RECURSION_LIMIT}
out = app.invoke({"cycle": 0, "best_score": 0.0, "stagnation": 11, "results": results,
                  "research_runs": 1, "budget": {"target_score": 0.66, "max_cycles": 2}}, cfg)
print("종료 phase:", out.get("phase"), "| cycle:", out.get("cycle"), "| stop:", out.get("stop_reason"))
print("원장:")
for e in out.get("ledger", [])[:10]:
    print("  ", json.dumps(e, ensure_ascii=False)[:150])
