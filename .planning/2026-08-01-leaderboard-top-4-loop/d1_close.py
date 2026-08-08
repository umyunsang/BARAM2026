"""D1_F2_1 admission — 제어전략 기반 SCADA 필터.

근거: research://Morrison 2022 (Cited 143) + Wang 2025 JMSE 13(3):410 [near_match_only]

## 선행 증거
- C1N87 (감발 정제): `잔차 < -0.15 AND 나셀풍속 >= 10` 손으로 쓴 규칙 -> **-0.001194**
  판정 CURTAILMENT_CLEANING_DOES_NOT_HELP
- C1N89 (이상 정제): 규칙을 손으로 쓰지 않고 **비지도 탐지**에 맡김 -> **-0.008177**
  판정 ANOMALY_CLEANING_DOES_NOT_HELP_**LANE_CLOSES**
  리포트 원문: "N1 은 좁고 물리 가정이 박힌 규칙이었고 실패했다. 여기서는 규칙을 손으로 쓰지 않고
  비지도 탐지에 맡긴다" — 그것도 실패, 그리고 더 나빴다

## 판정 논리
제어전략 영역 구분(MPPT/정격/피치)은 **손으로 쓴 물리 규칙**이며, C1N89 의 비지도 탐지보다
**엄격히 더 제약적인 가설공간**이다. 더 일반적인 방법이 실패하고 레인이 닫혔으므로,
그 부분집합이 성공할 것으로 기대할 근거가 없다.

또한 두 선행 실험 모두 **더 나쁜 쪽으로** 갔다 (-0.0012, -0.0082). 방향성 일관.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, "src")
from baram.loop.registry import SpecRegistry

reg = SpecRegistry(Path("."))
verdict = {
    "node": "D1_F2_1", "stage": "D1",
    "origin": "research://Morrison-2022 + Wang-2025-JMSE-13-410 [near_match_only]",
    "prior_failures": [
        {"node": "C1N87", "method": "hand rule (residual<-0.15 & ws>=10)", "delta": -0.001194,
         "verdict": "CURTAILMENT_CLEANING_DOES_NOT_HELP"},
        {"node": "C1N89", "method": "unsupervised anomaly detection", "delta": -0.008177,
         "verdict": "ANOMALY_CLEANING_DOES_NOT_HELP_LANE_CLOSES"},
    ],
    "reasoning": "control-region filtering is a hand-written physical rule, i.e. a strictly narrower "
                 "hypothesis space than the unsupervised detector that already failed and closed the "
                 "lane. Both prior attempts moved in the negative direction. No basis to expect the "
                 "subset to succeed where the superset failed.",
    "result": "ADMISSION_FAILED_SUPERSET_ALREADY_CLOSED",
}
s = reg.get("D1_F2_1")
if s:
    s.status = "retired"
    s.outcome = {"accepted": False, "note": verdict["result"], "detail": verdict}
s2 = reg.get("D1_F3_2")
if s2:
    s2.status = "retired"
    s2.outcome = {"accepted": False, "note": "ADMISSION_FAILED_PREMISE_UNDERMINED",
                  "detail": "D1_F3_1 showed hub-height winds are in the pool and were deselected; "
                            "the premise 'absent from pool' is false"}
s3 = reg.get("D1_F1_1")
if s3:
    s3.status = "retired"
    s3.outcome = {"accepted": False, "note": "NO_BIAS_TO_CORRECT_QMAP_WORSENS",
                  "detail": "teacher wind bias -0.0056 m/s (already unbiased); quantile mapping "
                            "raises sigma 1.5866 -> 1.6273 (+2.56%)"}
s4 = reg.get("D1_F3_1")
if s4:
    s4.status = "retired"
    s4.outcome = {"accepted": False, "note": "PRESENT_BUT_DESELECTED",
                  "detail": "GFS 80m/100m 18+18 features in the 820-feature pool, deselected by "
                            "feature selection; even 10m winds were deselected in favour of teacher sitewind"}
reg.save()
print(json.dumps(verdict, ensure_ascii=False, indent=1)[:1100])
print("\nD1 노드 최종 상태:")
for n in ("D1_F3_1", "D1_F3_2", "D1_F1_1", "D1_F2_1"):
    sp = reg.get(n)
    if sp: print(f"  {n:10s} {sp.status:9s} {sp.outcome.get('note')}")
