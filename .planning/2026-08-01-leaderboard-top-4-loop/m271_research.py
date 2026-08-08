"""M271 리서치 실행기 — 방법(내부 ①)·방향(외부 ①)·실체화(외부 ②).

**계획의 파일 분할과 다르다.** 계획은 `m271_method_research.py` /
`m271_research_direction.py` / `m271_research_sota.py` 세 파일을 지정했으나, 세 연산자가
사전확약·레인 범위 강제·적재 기계를 전부 공유하므로 한 모듈에 연산자 세 개로 둔다. 세 파일로
쪼개면 200 줄이 세 번 복제된다. 연산자 자체는 계획대로 셋이고 분리도 유지된다.

**리서치는 이 스크립트가 수행하지 않는다.** 공식 규칙 6조가 원격 모델 API 를 금지하고
프로젝트 런타임에 검색 API 도 없다. 실제 조사는 세션 내 `WebSearch`/`WebFetch` 로 이뤄지며,
이 모듈의 역할은 셋이다.

    1. 라우터가 넘긴 표적에서 **사전확약**을 구성한다 (경계된 질의·소스등급·결정영향·중단조건)
    2. 적재 시 **레인 범위**를 강제한다 — 발화 노드의 레인 밖 소스는 거부된다 (영역 독립성)
    3. 2단계 분리를 강제한다 — `research_sota` 는 `DIRECTION` 없이 발화할 수 없다

리서치 두 종류의 발화 시점이 반대라는 점이 이 모듈의 존재 이유다.

    방법 리서치 (내부 ①): "이 과업을 제대로 수행하는 검증된 프레임워크는?"  -> 실행 **전**
    방향 리서치 (외부 C1~C12): "이 결손을 무엇이 설명하는가?"                -> 증거 **후**
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

APPLICABILITY_TAGS = (
    "directly_supported",
    "near_match_only",
    "contradicts_premise",
    "insufficient",
)

SOURCE_CLASSES = ("official_standard", "peer_reviewed", "official_docs", "other")


class LaneScopeViolation(RuntimeError):
    """발화 노드의 레인 밖 소스를 적재하려 했다. 영역 독립성이 깨진다."""


class StageOrderViolation(RuntimeError):
    """방향 없이 SOTA 를 찾으려 했다. 일괄 리서치로의 퇴행이다."""


@dataclass(frozen=True)
class Precommitment:
    """발화 시점에 동결한다. 결과를 본 뒤 질의를 넓히지 않는다."""

    operator: str  # METHOD_RESEARCH | RESEARCH_DIRECTION | RESEARCH_SOTA
    kind: str
    lane: str
    origin_node: str
    query: str
    source_class: tuple[str, ...]
    decision_impact: str
    stop_condition: str
    deficit_cell: str | None = None

    def digest(self) -> str:
        return hashlib.sha256(
            json.dumps(asdict(self), ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest()


@dataclass
class Source:
    source_id: str
    lane: str
    title: str
    origin: str
    source_class: str
    applicability: str
    claim: str
    conditions_differ: str = ""
    license: str | None = None

    def __post_init__(self) -> None:
        if self.applicability not in APPLICABILITY_TAGS:
            raise ValueError(f"unknown applicability tag: {self.applicability}")
        if self.source_class not in SOURCE_CLASSES:
            raise ValueError(f"unknown source class: {self.source_class}")


@dataclass
class Direction:
    """논문·연구자료가 세운 방향. `RESEARCH_SOTA` 는 이것 없이 발화할 수 없다."""

    direction_id: str
    lane: str
    mechanism: str
    why_this_deficit: str
    deficit_cell: str | None
    applicability: str
    source_ids: list[str] = field(default_factory=list)
    # 기존 결손 분해로 설명되지 않는 기전이면 C12 가 축을 재분해한다.
    novel_axis: str | None = None


@dataclass
class Specification:
    """SOTA·벤치마크가 실체화한 구현 사양. `MATERIALIZE` 의 입력이다."""

    spec_id: str
    direction_id: str
    lane: str
    method_name: str
    settings: dict[str, Any]
    required_inputs: list[str]
    within_official_rules: bool
    reported_performance: str
    our_conditions_differ: str
    expected_effect_range: str
    source_ids: list[str] = field(default_factory=list)


# ---------------------------------------------------------------- 사전확약


def precommit(target: dict[str, Any], operator: str) -> Precommitment:
    """라우터 표적에서 사전확약을 구성한다."""
    lane = target.get("lane")
    kind = target.get("kind")
    if not lane or not kind:
        raise ValueError("router target must carry lane and kind")
    scope = target.get("scope", "")
    return Precommitment(
        operator=operator,
        kind=kind,
        lane=lane,
        origin_node=target.get("origin_node", "n"),
        deficit_cell=target.get("deficit_cell"),
        query=f"[{lane}/{kind}] {scope}",
        source_class=("official_standard", "peer_reviewed", "official_docs"),
        decision_impact=(
            f"레인 {lane} 의 결손 {target.get('deficit_cell')} 에 대한 다음 실험 사양을 결정한다"
        ),
        stop_condition=(
            "적용성 태그가 붙은 소스를 1건 이상 확보했거나, 2회 조사 후에도 이 과제 조건에 "
            "정합하는 것이 없다고 판정되면 중단"
        ),
    )


# ---------------------------------------------------------------- 적재


def ingest_sources(pre: Precommitment, sources: list[Source]) -> dict[str, Source]:
    """레인 범위를 강제하며 적재한다. 발화 노드의 레인 밖 소스는 거부된다."""
    out: dict[str, Source] = {}
    for source in sources:
        if source.lane != pre.lane:
            raise LaneScopeViolation(
                f"source {source.source_id} is lane {source.lane}, "
                f"but the firing node is lane {pre.lane}. Domain independence is enforced."
            )
        out[source.source_id] = source
    return out


# ---------------------------------------------------------------- 연산자 1: 방법 리서치


def method_research(node_id: str, lane: str, task: str) -> Precommitment:
    """내부 루프 ① — 이 노드의 과업을 수행할 검증된 프레임워크·방법을 찾는다.

    도메인 결론이 아니라 **수행 방법**을 찾는 단계다. 데이터 특성을 모르는 상태에서도
    수행할 수 있고, 오히려 모든 노드가 실행 **전에** 해야 한다.
    """
    return Precommitment(
        operator="METHOD_RESEARCH",
        kind="method",
        lane=lane,
        origin_node=node_id,
        query=f"[{lane}/method] {task} 를 수행할 검증된 프레임워크·방법",
        source_class=("official_standard", "peer_reviewed", "official_docs"),
        decision_impact=f"{node_id} 의 ② 사양(구현 방식)을 결정한다",
        stop_condition=(
            "방법을 1건 이상 식별하고 라이선스·적용성 태그를 붙였거나, 2회 조사 후에도 "
            "이 과제 조건에 정합하는 것이 없다고 판정되면 중단"
        ),
    )


# ---------------------------------------------------------------- 연산자 2: 방향 리서치


def research_direction(
    target: dict[str, Any], sources: list[Source], mechanism: str, why: str,
    applicability: str, novel_axis: str | None = None,
) -> tuple[Precommitment, dict[str, Source], Direction]:
    """외부 ① — 논문·연구자료로 방향을 세운다."""
    pre = precommit(target, "RESEARCH_DIRECTION")
    stored = ingest_sources(pre, sources)
    direction = Direction(
        direction_id=f"dir::{pre.origin_node}::{pre.kind}::{pre.lane}",
        lane=pre.lane,
        mechanism=mechanism,
        why_this_deficit=why,
        deficit_cell=pre.deficit_cell,
        applicability=applicability,
        source_ids=sorted(stored),
        novel_axis=novel_axis,
    )
    return pre, stored, direction


# ---------------------------------------------------------------- 연산자 3: 실체화 리서치


def research_sota(
    direction: Direction | None,
    target: dict[str, Any],
    sources: list[Source],
    *,
    method_name: str,
    settings: dict[str, Any],
    required_inputs: list[str],
    within_official_rules: bool,
    reported_performance: str,
    our_conditions_differ: str,
    expected_effect_range: str,
) -> tuple[Precommitment, dict[str, Source], Specification]:
    """외부 ② — SOTA·벤치마크로 구현 사양을 실체화한다.

    **방향이 정해진 뒤에만 발화한다.** 방향 없이 SOTA 를 찾으면 실험 이전의 무지 상태에서
    질의가 고정되는 일괄 리서치로 퇴행한다.

    벤치마크 수치는 **참조값이지 요구사항이 아니다.** 사용자가 계약 변경을 수락하기 전까지
    목표가 되지 않으므로 `reported_performance` 와 `our_conditions_differ` 를 항상 함께
    기록한다.
    """
    if direction is None:
        raise StageOrderViolation(
            "research_sota fired without a DIRECTION. That is a regression to batch research."
        )
    pre = precommit(target, "RESEARCH_SOTA")
    if direction.lane != pre.lane:
        raise LaneScopeViolation(
            f"direction lane {direction.lane} != firing lane {pre.lane}"
        )
    stored = ingest_sources(pre, sources)
    spec = Specification(
        spec_id=f"spec::{direction.direction_id}",
        direction_id=direction.direction_id,
        lane=pre.lane,
        method_name=method_name,
        settings=settings,
        required_inputs=required_inputs,
        within_official_rules=within_official_rules,
        reported_performance=reported_performance,
        our_conditions_differ=our_conditions_differ,
        expected_effect_range=expected_effect_range,
        source_ids=sorted(stored),
    )
    return pre, stored, spec
