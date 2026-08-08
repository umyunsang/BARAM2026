"""D1 — 전처리 단계: 딥리서치 결과를 노드로 변환하고 레지스트리에 등록.

이 세션은 31 건의 실험 중 전처리 0 건, 피처구성 1 건이었고 58% 가 모델링·개선전략에 몰렸다.
사용자 지적에 따라 **단계별 깊이 발굴**로 전환한다: 전처리 딥리서치 -> 노드 생성 -> 실행 ->
그 결과로 피처구성 딥리서치 -> 노드 생성 -> 실행 -> ...

## 딥리서치 산출 (2026-08-06, Serper 경유)
"""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, "src")
from baram.loop.registry import NodeSpec, SpecRegistry

reg = SpecRegistry(Path("."))

FINDINGS = [
 dict(id="F1", claim="풍속 편향보정이 풍력 예측 정확도에 미치는 영향을 75 개 단지로 측정",
      source="Spiliotis et al. 2025, ScienceDirect S2213138825004308 (Cited 9)",
      relevance="우리 파이프라인에 NWP 풍속 편향보정 단계가 없다. C1N21/22/23 의 MOS 는 "
                "**출력(발전량) MOS** 였지 풍속 MOS 가 아니다.", tag="near_match_only"),
 dict(id="F2", claim="SCADA power-curve cleaning(감발·정지·이상 제거)은 원자료 사용의 필수 단계",
      source="Morrison et al. 2022, Univ. Glasgow (Cited 143); Wang 2025 MDPI JMSE 13(3) 410 (Cited 11)",
      relevance="C1N87/C1N89 가 감발·이상 제거를 시도했으나 **teacher 표적 정제**로만 했고, "
                "제어전략 기반 필터(가변속·가변피치 MPPT 영역 구분)는 미시도.", tag="near_match_only"),
 dict(id="F3", claim="power-law 허브고도 외삽의 한계를 정량화하고 대안을 제시",
      source="IOP 2026 10.1088/3049-4753/ae4c8d; ACP 23:3181 (2023) ML 기반 허브고도 풍속",
      relevance="**GFS 는 80m·100m 풍속을 직접 제공하는데 선택 피처 100 개에 0 개**. "
                "teacher sitewind 13 개가 10m/50m 에서 추정으로 대체 중.", tag="directly_supported"),
 dict(id="F4", claim="주파수영역 보간으로 결측 관측을 재구성",
      source="Xu 2025, MDPI Applied Sciences 16(1) 305",
      relevance="테스트 LDAPS 결측 752 셀(0.016%)은 이미 선형보간. 규모가 무의미하므로 부적용.",
      tag="insufficient"),
]

NODES = [
 dict(id="D1_A", sub="C3.6", kind="analysis", score_bearing=True,
      summary="GFS 80m/100m 풍속의 피처풀 존재 여부 감사 — 선택 탈락인가 애초 부재인가",
      origin="research://IOP-3049-4753-ae4c8d + ACP-23-3181 (F3, directly_supported)",
      note="가장 먼저 실행. 결과가 D1_B/D1_C 의 설계를 결정한다."),
 dict(id="D1_B", sub="C3.6", kind="feature", score_bearing=True,
      summary="허브고도 풍속 직접 사용 대 power-law 외삽 대 teacher 추정 3자 비교",
      origin="research://IOP-3049-4753-ae4c8d (F3)",
      note="D1_A 가 '애초 부재' 를 반환할 때만 실행."),
 dict(id="D1_C", sub="C3.2", kind="analysis", score_bearing=True,
      summary="NWP 풍속 자체의 편향보정(quantile mapping / MOS on wind, not on power)",
      origin="research://Spiliotis-2025-S2213138825004308 (F1)",
      note="기존 C1N21/22/23 은 출력 MOS 였음. 풍속 단계 보정은 미시도."),
 dict(id="D1_D", sub="C3.4", kind="analysis", score_bearing=True,
      summary="제어전략 기반 SCADA 필터(MPPT/정격/피치 영역 구분) 후 teacher 재학습",
      origin="research://Morrison-2022 + Wang-2025-JMSE-13-410 (F2)",
      note="C1N87/89 는 잔차·비지도 규칙이었고 제어영역 구분은 미시도."),
]

for n in NODES:
    reg.register(NodeSpec(id=n["id"], kind=n["kind"], subcapability=n["sub"],
                          summary=n["summary"], origin=n["origin"],
                          status="candidate", score_bearing=n["score_bearing"],
                          arguments={"note": n["note"]}))
reg.save()

Path("reports/D1_preprocessing_research.json").write_text(json.dumps(dict(
    stage="D1_PREPROCESSING", queries=4, findings=FINDINGS, nodes_created=NODES,
    method="staged depth-first excavation: research -> nodes -> execute -> next stage",
    agent_upload=False, model_fits=0), indent=1, ensure_ascii=False))

print("=== D1 전처리 단계: 딥리서치 -> 노드 생성 ===")
for f in FINDINGS:
    print(f"  [{f['tag']:18s}] {f['id']}  {f['claim'][:52]}")
print()
for n in NODES:
    print(f"  신규노드 {n['id']:6s} [{n['sub']}] {n['summary'][:56]}")
print(f"\n레지스트리 후보 {len([c for c in reg.candidates()])}건")
