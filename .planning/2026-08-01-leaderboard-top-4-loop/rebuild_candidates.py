"""Rebuild the candidate set from verified evidence.

Lesson recorded: N301 was registered as unexplored while `C1N87`/`C1N89` had
already closed it. Those cycles sat in the unrecorded-cycle backlog, so the
coverage view was wrong and the router was fed a polluted input. Every candidate
below now carries an explicit `origin` citing the evidence that establishes it is
genuinely unopened.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, "src")
from baram.loop.registry import NodeSpec, SpecRegistry

reg = SpecRegistry(Path("."))

# demote the wrongly-registered ones with their real verdicts
CLOSED = {
    "N301": "C1N87 CURTAILMENT_CLEANING_DOES_NOT_HELP (-0.001194) / C1N89 ANOMALY_CLEANING_DOES_NOT_HELP (-0.008177)",
    "N307": "teacher/sitewind 계열은 C1N42/C1N58/C1N82~84 에서 다수 실행됨 — 미탐색 아님",
    "N302": "발행배치 경계 정렬은 m269_stage_b / m261_diagnostic 에서 다뤄짐",
    "N303": "적격모집단은 C1N23_ELIGIBLE_MOS(NO_STABLE_METRIC_ALIGNED_BIAS) 및 M282 에서 다뤄짐",
    "N306": "월별 게이트 구조는 m270_monthly_validation 에서 확립됨",
    "N308": "레짐 진단은 m270_regime_diagnosis 에서 실행됨",
}
for sid, verdict in CLOSED.items():
    s = reg.get(sid)
    if s:
        s.status = "retired"
        s.outcome = {"accepted": False, "note": "ADMISSION_FAILED_PRIOR_EVIDENCE", "prior": verdict}

# verified-unopened lanes, per reports/m270_revived_lanes.md section 4
UNOPENED = [
    ("N401", "model", "C5.2",
     "심층 확률예측(CQR 우선) — LightGBM 분위수 위 conformal 교정으로 조건부 커버리지 확보",
     "research://arXiv:2602.13010v2 (CQR nCRPS 5.7% 대 NGBoost 6.0%) + reports/m270_revived_lanes.md#4 UNOPENED"),
    ("N402", "model", "C5.2",
     "Treeffuser/분포회귀 — 튜닝 전제 하 최상위 nCRPS 5.6%. 라이선스·공개일 게이트 선행 필요",
     "research://arXiv:2602.13010v2 + reports/m270_revived_lanes.md#4 UNOPENED"),
    ("N403", "model", "C5.1",
     "사전학습 시계열 파운데이션 모델 — 2026-07-05 이전 공개 + 상업이용 OSS 한정, 로컬 로드",
     "reports/m270_revived_lanes.md#4 UNOPENED (P2 역량 게이트 대상)"),
    ("N404", "analysis", "C1.6",
     "ramp/weather-pattern 분류 레인 — 차단 전제(대체 NWP 부재)가 해소되어 REVIVED",
     "reports/m270_revived_lanes.md#3 M251 NO_SELECTABLE_NWP"),
]
for sid, kind, sub, summary, origin in UNOPENED:
    reg.register(NodeSpec(id=sid, kind=kind, subcapability=sub, summary=summary,
                          origin=origin, status="candidate", score_bearing=True))

reg.save()
cand = reg.candidates()
print(f"강등: {len(CLOSED)}건 (ADMISSION_FAILED_PRIOR_EVIDENCE)")
print(f"검증된 미탐색 후보: {len(cand)}건")
for s in cand:
    print(f"   {s.id} [{s.subcapability}] {s.summary[:56]}")
