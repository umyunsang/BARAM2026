"""M271 조건부 엣지 라우터 — 동결 표 (그래프 엔지니어링 원칙 2).

노드 실행 -> `EVIDENCE` -> 라우터가 다음 노드를 **동적 선택**한다. 표는 선언적이고 버전이
고정되며 실행 전에 동결된다.

이 파일이 계획의 중심이다. 요점은 C3 와 C4 다.

    같은 "게이트 실패" 라도 서명이 다르면 다른 리서치로 간다.
      G1 실패 & G3 통과  = 효과는 있으나 비일관 -> 체제조건부 기법을 찾는다
      G3 실패 & G1 통과  = 일관하나 미미      -> 효과 증폭 기법을 찾는다

일괄 리서치는 이 구분을 원리적으로 할 수 없다. 실험 이전의 무지 상태에서 질의가 고정되기
때문이다.

C10 은 엔진 자체를, C12 는 결손 분해 자체를 루프 안에 넣는다. 리서치가 기존 결손을
소비하기만 하는 게 아니라 새 방향을 열 수 있어야 하기 때문이다.

임계값의 지위: `TAU_DEFICIT_MASS` 와 `EPSILON_INFORMATION` 은 **선언 관례**이지 보정된
값이 아니다. 이 문제에서 held-out 으로 적합한 바 없다. 계획 R3 가 이를 잔여 리스크로
기록하고 있으며, 정체 시 C10 이 표 자체를 개정한다.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

ROUTER_VERSION = "M271_ROUTER_v2_frozen_2026-08-04"
ROUTER_V1_VERSION = "M271_ROUTER_v1_frozen_2026-08-04"

# v1 -> v2 개정 근거 (`reports/m271_p4_bootstrap.md` 가 측정한 결함).
#
# 임계값을 후보가 통과하도록 조정한 것이 **아니다**. 동작 사상의 범주 오류를 고쳤다.
# v1 을 지우지 않고 여기 기록으로 남기며 롤백 경로를 유지한다(계획 R3).
#
#   결함 1: C12 가 `novel_mechanism is not None` 하나로 발화해 성질이 다른 둘을 같은
#           REFINE_AXIS 로 보냈다.
#             A4 `wind_sector`              = 손실의 분할 축     -> REFINE_AXIS 가 옳다
#             A1 `effective_sample_size`    = 검증 표면의 성질   -> 분할 축이 아니다
#             A3 `grid_resolution_mismatch` = 전 행에 균일 적용  -> 분할 축이 아니다
#           손실을 "유효표본수별" 로 쪼갤 수는 없다. 축 종류를 증거가 선언하게 하고
#           (`axis_kind`), 분할 축이 아닌 구조적 사실은 C13 으로 보낸다.
#
#   결함 2: 확인된 부정에 대한 조건이 없었다. A2 는 "진짜 미사용 컬럼 MI 가 선언 컬럼의
#           0.414 배" 라는 명확한 부정을 냈으나 information 이 epsilon 위라 C8 이 걸리지
#           않고 deficit_cell 이 없어 C1 도 걸리지 않아 NONE 이 됐다. C8 의 의미는 "정보를
#           못 냈다" 인데 A2 는 정보를 냈고 그것이 부정이다. C14 로 분리한다.
#
#   결함 3: A7 이 원장을 세웠는데 deficit_cell 을 들고 있지 않아 아무 조건도 걸리지 않았다.
#           원장이 서면 상위 미설명 셀에 C1 이 발화해야 한다. C15 로 처리한다.

# 선언 관례. 보정된 값이 아니다.
TAU_DEFICIT_MASS = 0.010  # 이 이상 남은 결손 질량이면 레인을 굶은 것으로 본다
EPSILON_INFORMATION = 0.02  # 이 미만 정보량이면 굴착지점을 접는다
STALL_LIMIT = 3  # 결손 질량 무감소가 이만큼 이어지면 정체

RESEARCH_KINDS = (
    "explain",
    "replace_mechanism",
    "regime_conditional",
    "amplify",
    "anomaly",
    "lane_expand",
    "scale_up",
    "loop_engine",
)


@dataclass(frozen=True)
class Evidence:
    """노드 실행이 낸 증거 서명. 라우터가 읽는 유일한 입력이다."""

    evidence_id: str
    node_id: str
    lane: str
    deficit_cell: str | None = None
    # 게이트 서명. `m270_gate.py` 의 4조건을 그대로 받는다.
    gate: dict[str, bool] | None = None
    # 사전확약 대비 부호
    sign: int = 0
    predeclared_sign: int = 0
    # 이 노드가 만들어낸 정보량 (0~1 규모의 상대값)
    information: float = 1.0
    refutes_mechanism: str | None = None
    confirms: bool = False
    # 기존 결손 분해로 설명되지 않는 기전을 발견했는가
    novel_mechanism: str | None = None
    # v2: 그 기전이 **손실의 분할 축**인지 구조적 사실인지 증거가 선언한다.
    #   "partition"  = 행을 이 축으로 쪼갤 수 있다 -> C12 REFINE_AXIS
    #   "structural" = 전 행에 균일하거나 표면의 성질이다 -> C13 구조적 결과 리서치
    axis_kind: str | None = None
    # v2: 확인된 부정. 정보를 냈고 그 정보가 '이 축에는 없다' 일 때 참이다.
    negative_finding: bool = False
    # v2: 원장을 세운 노드가 상위 미설명 셀을 넘겨줄 때 쓴다.
    seeds_cells: tuple[str, ...] = ()
    # 회수 실험의 기대 이득과 소요 시간
    expected_gain: float = 0.0
    expected_hours: float = 1.0

    def gate_flag(self, name: str) -> bool | None:
        return None if self.gate is None else bool(self.gate.get(name))


@dataclass(frozen=True)
class RouterDecision:
    condition: str
    action: str
    targets: tuple[dict[str, Any], ...] = ()
    reason: str = ""
    considered: tuple[str, ...] = ()
    router_version: str = ROUTER_VERSION


@dataclass(frozen=True)
class Condition:
    code: str
    priority: int  # 동결 tie-break 순서
    action: str
    research_kind: str | None
    describe: str
    fires: Callable[[Evidence, dict], bool]
    build: Callable[[Evidence, dict], tuple[dict[str, Any], ...]]
    # C11 선택에 쓰는 기본 가치. 선언 관례다.
    base_value: float = 0.0
    voi: float = 0.0
    exploration: float = 0.0


# ---------------------------------------------------------------- 발화 조건


def _lane_starved(evidence: Evidence, state: dict) -> list[str]:
    census = state.get("lane_live_counts", {})
    mass = state.get("lane_deficit_mass", {})
    return sorted(
        lane
        for lane, live in census.items()
        if live == 0 and float(mass.get(lane, 0.0)) > TAU_DEFICIT_MASS
    )


def _research_target(evidence: Evidence, kind: str, scope: str) -> dict[str, Any]:
    """리서치 범위는 **발화 노드의 레인**으로 제한된다. 영역 독립성을 여기서 강제한다."""
    return {
        "operator": "RESEARCH_DIRECTION",
        "kind": kind,
        "lane": evidence.lane,
        "deficit_cell": evidence.deficit_cell,
        "origin_node": evidence.node_id,
        "scope": scope,
    }


CONDITIONS: tuple[Condition, ...] = (
    Condition(
        code="C1",
        priority=1,
        action="RESEARCH_DIRECTION",
        research_kind="explain",
        describe="기전이 없는 결손 셀",
        fires=lambda e, s: e.deficit_cell is not None
        and s.get("cell_status", {}).get(e.deficit_cell) == "UNEXPLAINED",
        build=lambda e, s: (_research_target(e, "explain", "그 결손 셀의 물리·통계 기전"),),
        base_value=0.30,
        voi=0.20,
    ),
    Condition(
        code="C2",
        priority=2,
        action="RESEARCH_DIRECTION",
        research_kind="replace_mechanism",
        describe="증거가 기전을 반증",
        fires=lambda e, s: e.refutes_mechanism is not None,
        build=lambda e, s: (
            _research_target(
                e, "replace_mechanism", f"반증된 기전 {e.refutes_mechanism} 의 대체 후보만"
            ),
        ),
        base_value=0.35,
        voi=0.25,
    ),
    Condition(
        code="C3",
        priority=3,
        action="RESEARCH_DIRECTION",
        research_kind="regime_conditional",
        describe="G1 실패 & G3 통과 — 효과 있으나 비일관",
        fires=lambda e, s: e.gate is not None
        and e.gate_flag("G1") is False
        and e.gate_flag("G3") is True,
        build=lambda e, s: (
            _research_target(e, "regime_conditional", "체제조건부·국소전문가·혼합전문가"),
        ),
        base_value=0.40,
        voi=0.15,
    ),
    Condition(
        code="C4",
        priority=4,
        action="RESEARCH_DIRECTION",
        research_kind="amplify",
        describe="G3 실패 & G1 통과 — 일관하나 미미",
        fires=lambda e, s: e.gate is not None
        and e.gate_flag("G3") is False
        and e.gate_flag("G1") is True,
        build=lambda e, s: (_research_target(e, "amplify", "효과 증폭·스태킹·앙상블"),),
        base_value=0.35,
        voi=0.15,
    ),
    Condition(
        code="C5",
        priority=5,
        action="RESEARCH_DIRECTION",
        research_kind="anomaly",
        describe="사전확약과 부호가 반대",
        fires=lambda e, s: e.predeclared_sign != 0 and e.sign != 0
        and e.sign != e.predeclared_sign,
        build=lambda e, s: (_research_target(e, "anomaly", "부호 역전의 기전"),),
        base_value=0.25,
        voi=0.40,  # 부호 역전은 정보가치가 가장 높다
    ),
    Condition(
        code="C6",
        priority=6,
        action="SEND_FANOUT",
        research_kind="lane_expand",
        describe="레인 기아 — 생존가설 0 이고 결손 질량이 임계 초과",
        fires=lambda e, s: bool(_lane_starved(e, s)),
        build=lambda e, s: tuple(
            {
                "operator": "RESEARCH_DIRECTION",
                "kind": "lane_expand",
                "lane": lane,
                "deficit_cell": None,
                "origin_node": e.node_id,
                "scope": f"레인 {lane} 의 SOTA·벤치마크 확장",
            }
            for lane in _lane_starved(e, s)
        ),
        base_value=0.20,
        voi=0.30,
        exploration=0.25,
    ),
    Condition(
        code="C7",
        priority=7,
        action="RESEARCH_DIRECTION",
        research_kind="scale_up",
        describe="확인됨 & 잔여 질량 있음 — 같은 방향 고도화",
        fires=lambda e, s: e.confirms
        and float(s.get("residual_mass", 0.0)) > TAU_DEFICIT_MASS,
        build=lambda e, s: (
            {
                **_research_target(e, "scale_up", "확인된 방향의 고도화"),
                # 새 방향으로 튀지 않도록 발원 방향을 명시적으로 물려준다.
                "inherit_direction": s.get("node_direction", {}).get(e.node_id),
            },
        ),
        base_value=0.45,
        voi=0.05,
    ),
    Condition(
        code="C8",
        priority=8,
        action="PRUNE",
        research_kind=None,
        describe="정보량 미달 — 굴착지점 종결. 리서치 없음",
        fires=lambda e, s: e.information < EPSILON_INFORMATION,
        build=lambda e, s: (
            {
                "operator": "PRUNE",
                "node_id": e.node_id,
                "premise": "INFORMATION_BELOW_EPSILON",
                "lane": e.lane,
            },
        ),
        base_value=0.10,
    ),
    Condition(
        code="C9",
        priority=9,
        action="REVIVE",
        research_kind=None,
        describe="전제가 뒤집힘 — 체크포인트 포크로 하위그래프 부활",
        fires=lambda e, s: bool(s.get("flipped_premises")),
        build=lambda e, s: tuple(
            {"operator": "REVIVE", "premise": code} for code in s.get("flipped_premises", [])
        ),
        base_value=0.50,
        voi=0.20,
    ),
    Condition(
        code="C10",
        priority=10,
        action="RESEARCH_DIRECTION",
        research_kind="loop_engine",
        describe="루프 정체 — 라우터 표 자체의 SOTA 를 조사해 엔진을 제자리 개선",
        fires=lambda e, s: int(s.get("guards", {}).get("stall_counter", 0)) >= STALL_LIMIT,
        build=lambda e, s: (
            {
                "operator": "RESEARCH_DIRECTION",
                "kind": "loop_engine",
                "lane": "L8",
                "deficit_cell": None,
                "origin_node": e.node_id,
                "scope": "루프·그래프 엔지니어링 라우터 표의 SOTA",
            },
        ),
        base_value=0.15,
        voi=0.35,
        exploration=0.30,
    ),
    Condition(
        code="C12",
        priority=12,
        action="REFINE_AXIS",
        research_kind=None,
        describe="새 기전이 **손실의 분할 축** — 결손 축 재분해",
        # v2: `axis_kind == "partition"` 을 요구한다. v1 은 이 조건이 없어 분할 축이 아닌
        # 구조적 사실까지 REFINE_AXIS 로 보냈다(범주 오류).
        fires=lambda e, s: e.novel_mechanism is not None and e.axis_kind == "partition",
        build=lambda e, s: (
            {
                "operator": "REFINE_AXIS",
                "cell": e.deficit_cell,
                "axis": e.novel_mechanism,
                "origin_node": e.node_id,
            },
        ),
        base_value=0.30,
        voi=0.35,
    ),
    Condition(
        code="C13",
        priority=13,
        action="RESEARCH_DIRECTION",
        research_kind="structural_consequence",
        describe="새 기전이 구조적 사실 — 분할이 아니라 그 결과를 조사",
        fires=lambda e, s: e.novel_mechanism is not None and e.axis_kind == "structural",
        build=lambda e, s: (
            _research_target(
                e,
                "structural_consequence",
                f"구조적 사실 '{e.novel_mechanism}' 이 이 레인의 접근에 미치는 결과",
            ),
        ),
        base_value=0.35,
        voi=0.30,
    ),
    Condition(
        code="C14",
        priority=14,
        action="CLOSE_AXIS",
        research_kind=None,
        describe="확인된 부정 — 그 축을 닫는다. C8(정보 못 냄)과 다르다",
        fires=lambda e, s: e.negative_finding and e.information >= EPSILON_INFORMATION,
        build=lambda e, s: (
            {
                "operator": "CLOSE_AXIS",
                "lane": e.lane,
                "origin_node": e.node_id,
                "premise": "MEASURED_NEGATIVE",
                "note": "정보를 냈고 그것이 '이 축에는 없다' 이다. 전제가 뒤집히면 C9 이 부활.",
            },
        ),
        base_value=0.25,
        voi=0.10,
    ),
    Condition(
        code="C15",
        priority=15,
        action="SEND_FANOUT",
        research_kind="explain",
        describe="원장이 세워짐 — 상위 미설명 셀에 기전 리서치를 뿌린다",
        fires=lambda e, s: bool(e.seeds_cells),
        build=lambda e, s: tuple(
            {
                "operator": "RESEARCH_DIRECTION",
                "kind": "explain",
                "lane": s.get("cell_lane", {}).get(cell, e.lane),
                "deficit_cell": cell,
                "origin_node": e.node_id,
                "scope": f"결손 셀 {cell} 의 물리·통계 기전",
            }
            for cell in e.seeds_cells
        ),
        base_value=0.40,
        voi=0.25,
        exploration=0.10,
    ),
)

BY_CODE = {c.code: c for c in CONDITIONS}


# ---------------------------------------------------------------- C11 선택


def _value(condition: Condition, evidence: Evidence) -> float:
    """시간당 기대 Total 이득 + 정보가치 + 탐험항. 선언 관례다."""
    gain_per_hour = evidence.expected_gain / max(evidence.expected_hours, 1e-6)
    return condition.base_value + gain_per_hour + condition.voi + condition.exploration


def decide(evidence: Evidence, state: dict) -> RouterDecision:
    """조건부 엣지 판정. 동일 입력에 동일 출력이어야 한다."""
    firing = [c for c in CONDITIONS if c.fires(evidence, state)]
    considered = tuple(c.code for c in firing)

    if not firing:
        return RouterDecision(
            condition="NONE",
            action="HALT",
            reason="발화 조건 없음. 프론티어에서 다음 노드를 고른다.",
            considered=considered,
        )

    if len(firing) == 1:
        chosen = firing[0]
        return RouterDecision(
            condition=chosen.code,
            action=chosen.action,
            targets=chosen.build(evidence, state),
            reason=chosen.describe,
            considered=considered,
        )

    # C11: 복수 발화. 가치 최대, 동률은 동결 priority, 그다음 code 사전순.
    ranked = sorted(firing, key=lambda c: (-_value(c, evidence), c.priority, c.code))
    chosen = ranked[0]
    return RouterDecision(
        condition=chosen.code,
        action=chosen.action,
        targets=chosen.build(evidence, state),
        reason=f"C11 선택: {len(firing)} 개 발화 중 가치 최대 ({chosen.describe})",
        considered=considered,
    )


def table() -> list[dict[str, Any]]:
    """동결 표를 사람이 읽을 형태로."""
    return [
        {
            "code": c.code,
            "priority": c.priority,
            "action": c.action,
            "research_kind": c.research_kind,
            "describe": c.describe,
        }
        for c in sorted(CONDITIONS, key=lambda x: x.priority)
    ]
