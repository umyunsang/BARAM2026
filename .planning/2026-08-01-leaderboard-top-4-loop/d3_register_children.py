from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, "src")
from baram.loop.registry import NodeSpec, SpecRegistry
reg = SpecRegistry(Path("."))
CH = [
 ("D3_H2_2","C5.5","무조건부 잔차 결정층을 평활 멤버(M244)에 적용",
  "derived://D3_H2_1", "UNCONDITIONAL_RESIDUAL_DECISION_LAYER_HURTS",
  "Total 0.605760 -> 0.581140, FICR 0.3506 -> 0.3026. 잔차 분위수가 전 행 동일이라 조건부 정보 없음."),
 ("D3_H2_3","C5.5","밴드 도달가능성 조건부 멤버 전환",
  "derived://D3_H2_2", "REACHABILITY_SWITCH_BELOW_UNIFORM_BLEND",
  "전부이산 대비 +0.003131 (tau 0.40) 이나 균일 w=0.7 대비 -0.001148. 전환은 분산감소가 없음."),
 ("D3_H2_4","C5.5","밴드 도달가능성 조건부 가중 결합 (두 멤버 항상 기여)",
  "derived://D3_H2_3", "CONDITIONAL_WEIGHT_BELOW_THRESHOLD_AND_NOT_DEPLOYABLE",
  "fold-외 균일 대비 +0.000835 (문턱의 82%), 인샘플->fold-외 붕괴 -0.0018 로 안정. "
  "그러나 p_band 가 2025 46-bin 분포를 요구하는데 어디에도 저장돼 있지 않아 재적합 없이는 배포 불가."),
]
for sid, sub, summary, origin, note, detail in CH:
    reg.register(NodeSpec(id=sid, kind="analysis", subcapability=sub, summary=summary,
                          origin=origin, status="retired", score_bearing=True,
                          arguments={"stage": "D3"},
                          outcome={"accepted": False, "note": note, "detail": detail}))
reg.save()
print("D3 전체 노드:")
for s in reg.all():
    if s.arguments.get("stage") == "D3" or s.id.startswith("D3_"):
        print(f"  {s.id:10s} {s.status:9s} {(s.outcome or {}).get('note')}")
print()
tot = len(reg.all()); ret = sum(1 for s in reg.all() if s.status == "retired")
print(f"레지스트리 총 {tot}건 / retired {ret}건 / candidate {tot-ret}건")
