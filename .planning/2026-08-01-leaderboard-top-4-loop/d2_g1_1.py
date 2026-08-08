
"""D2_G1_1 — C1N77 소스분할과 논문 sister-model 구성의 대조 (적합 0회).

근거: research://arXiv:2505.10367v1 (HEFTCom2024 GEB) [directly_supported]
질문: C1N77 의 STACK 실패(0.603122 < POOLED 0.604043)가 **방법 자체**의 실패인가,
      아니면 **소스별로 피처를 쪼개 각 팔의 정보량을 줄인 구현**의 실패인가?

C1N77 기록: pooled 101 피처 / gfs 팔 50 / ldaps 팔 40  (합 90 < 101, 서로소 분할)
논문 sister-model: 각 소스에 **동일한 전체 파이프라인**을 적용 (피처 수를 줄이지 않음)

판별: 소스별 팔의 피처 수가 pooled 대비 얼마나 줄었는지, 그 감소가 성능 저하를 설명하는지.
락박스 미접근 / 적합 0회 / 제출 없음.
"""
from __future__ import annotations
import sys, json, re
from pathlib import Path
sys.path.insert(0, "src")

ROOT = Path(".")
pool = json.loads((ROOT / "artifacts/manifests/prepare.json").read_text())["feature_names"]
gfs = [f for f in pool if f.startswith("gfs__")]
ldaps = [f for f in pool if f.startswith("ldaps__")]
other = [f for f in pool if not f.startswith(("gfs__", "ldaps__"))]
print(f"820 피처 풀 구성: gfs {len(gfs)} / ldaps {len(ldaps)} / 기타 {len(other)}")

C1N77 = {"pooled": {"features": 101, "total": 0.604043},
         "gfs": {"features": 50, "total": 0.602321},
         "ldaps": {"features": 40, "total": 0.593511},
         "stack": {"total": 0.603122}}
print("\n=== C1N77 기록 ===")
for arm, v in C1N77.items():
    f = v.get("features")
    print(f"  {arm:8s} 피처 {str(f) if f else '-':>5}  Total {v['total']:.6f}")

p = C1N77["pooled"]["features"]
print(f"\n소스별 팔의 피처 수 감소:")
for arm in ("gfs", "ldaps"):
    n = C1N77[arm]["features"]
    print(f"  {arm:6s} {n}/{p} = {n/p:.1%}  (Total {C1N77[arm]['total']:.6f}, "
          f"pooled 대비 {C1N77[arm]['total']-C1N77['pooled']['total']:+.6f})")
print(f"  두 팔 합 {C1N77['gfs']['features']+C1N77['ldaps']['features']}/{p} "
      f"= {(C1N77['gfs']['features']+C1N77['ldaps']['features'])/p:.1%}  ← 서로소 분할이라 합쳐도 pooled 미만")

print("\n=== 논문 구성과의 차이 ===")
print("  논문 sister-model: 각 소스에 동일한 전체 파이프라인 적용, 피처 수 감소 없음")
print("  C1N77          : 피처를 소스별로 **서로소 분할** -> 각 팔이 pooled 의 40~50% 정보만 봄")
print("  -> 두 팔 모두 pooled 보다 낮은 것은 방법 실패가 아니라 **정보량 감소의 당연한 귀결**")

# 우리 데이터에서 sister-model 이 성립 가능한가?
print("\n=== 우리 데이터에서 논문식 sister-model 이 성립하는가 ===")
print(f"  gfs 전용 파이프라인 가능 피처: {len(gfs)} (+ 기타 {len(other)})")
print(f"  ldaps 전용 파이프라인 가능 피처: {len(ldaps)} (+ 기타 {len(other)})")
print("  두 소스가 **같은 변수를 다루지 않는다**: GFS 는 80m/100m/PBL/상층을 갖고")
print("  LDAPS 는 50m max/min·복사·운량 계열을 갖는다 -> 논문의 '동일 파이프라인' 전제가 성립 안 함")

verdict = "IMPLEMENTATION_DIFFERS_BUT_DATA_FORBIDS_PAPER_CONSTRUCTION"
print(f"\n판정: {verdict}")
print("  C1N77 의 실패는 구현 차이로 설명되나, 논문식 구성은 두 NWP 의 변수집합이 달라 재현 불가.")
print("  각 소스에 '동일한 전체 파이프라인' 을 적용하려면 공통 변수만 써야 하고, 그러면 다시 정보가 준다.")
Path("reports/D2_G1_1_sister_model_contrast.json").write_text(json.dumps(dict(
    node="D2_G1_1", stage="D2", pool=dict(gfs=len(gfs), ldaps=len(ldaps), other=len(other)),
    c1n77=C1N77, verdict=verdict,
    reasoning="C1N77 split features disjointly so each arm saw 40-50% of pooled information; the "
              "paper applies the same full pipeline per source. But GFS and LDAPS expose different "
              "variable sets here, so the paper construction cannot be reproduced without falling "
              "back to a common subset, which reintroduces the information loss.",
    origin="research://arXiv:2505.10367v1 [directly_supported]",
    model_fits=0, dacon_upload=False), indent=1, ensure_ascii=False))
print("영수증 -> reports/D2_G1_1_sister_model_contrast.json")
