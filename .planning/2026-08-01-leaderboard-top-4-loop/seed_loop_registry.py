"""Seed the loop registry: retro-register this session's experiments, then
enumerate genuinely unexplored candidate specs so the router can balance load."""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, "src")
from baram.loop.registry import NodeSpec, SpecRegistry

ROOT = Path(".")
reg = SpecRegistry(ROOT)

RETIRED = [
    ("M272", "decision", "C9.1", "정책격자 확장(재구성 표면)", {"score": 0.606051, "accepted": False, "note": "배포 미전이"}),
    ("M274", "analysis", "C5.2", "분포축 개방·게이트 판독", {"accepted": False, "note": "게이트 정당 확인"}),
    ("M275", "analysis", "C5.2", "밴드 캘리브레이션 재평가", {"accepted": False, "note": "D1>D2, 내 주장 철회"}),
    ("M276", "analysis", "C9.1", "결정온도=캘리브레이션 반증실험", {"accepted": False, "note": "기전 규명"}),
    ("M277", "decision", "C9.2", "조건부 캘리브레이션", {"score": 0.606043, "accepted": False, "note": "세분화 단조악화"}),
    ("M278", "model", "C5.3", "C1N41 재판정(적합6)", {"accepted": False, "note": "96% 아티팩트"}),
    ("M279", "decision", "C9.1", "격자확장 배포 전이", {"score": 0.628080, "accepted": False, "note": "개선 0"}),
    ("M281", "analysis", "C4.4", "외부NWP 전제 재검정", {"accepted": False, "note": "검증면 부재"}),
    ("M282", "decision", "C9.1", "적격성 마스크(배포)", {"accepted": False, "note": "무효 no-op"}),
    ("M283", "analysis", "C5.4", "구간 지지 정의", {"score": 0.609429, "accepted": False, "note": "격차 17.6% 규명"}),
    ("M284", "decision", "C5.4", "구간 내부적분(배포)", {"score": 0.629545, "accepted": False, "note": "게이트 기각"}),
    ("M285", "analysis", "C6.3", "월별 게이트 판정", {"accepted": False, "note": "M284 기각"}),
    ("M286", "decision", "C9.1", "오프셋 모형 정책 재선택", {"accepted": False, "note": "사전확약만 동결, 미실행"}),
]
for sid, kind, sub, summary, outcome in RETIRED:
    reg.register(NodeSpec(id=sid, kind=kind, subcapability=sub, summary=summary,
                          status="retired", outcome=outcome))

# Unexplored candidates. Categories deliberately span the under-loaded areas that
# the M273 coverage audit exposed (전처리 / 예측성능 / 평가지표 / 검증전략).
CANDIDATES = [
    ("N301", "analysis", "C3.4", "curtailment·정지 구간 타깃 정제 — 라벨의 비기상 구간을 식별하고 학습에서 제외했을 때의 효과"),
    ("N302", "analysis", "C3.3", "발행배치 경계 정렬 감사 — 반개구간 라벨과 예보시각 정렬의 체계적 편향 재검"),
    ("N303", "analysis", "C2.2", "적격모집단 선택편향 — actual>=0.10C 컷오프가 만드는 조건부 분포 왜곡의 정량화"),
    ("N304", "analysis", "C7.2", "배포 후보 재현성 byte 검증 — 2차 평가 산출물 요건 선제 충족"),
    ("N305", "analysis", "C7.4", "성능주장 범위관리 — 로컬↔온라인 오프셋 계급별 신뢰구간 문서화"),
    ("N306", "analysis", "C6.1", "시계열 분할 정합 재감사 — 월별 게이트가 pooled 이득을 기각하는 구조적 원인"),
    ("N307", "feature", "C4.6", "SCADA teacher 특권정보(LUPI) 활용 — train 전용 신호의 정식 활용 경로"),
    ("N308", "analysis", "C1.6", "레짐·이상치 진단 — 월별 게이트 실패가 특정 레짐에 몰리는지"),
]
for sid, kind, sub, summary in CANDIDATES:
    reg.register(NodeSpec(id=sid, kind=kind, subcapability=sub, summary=summary, status="candidate"))

reg.save()
print(f"등록: retired {len(RETIRED)} / candidate {len(CANDIDATES)} → {reg.path}")
