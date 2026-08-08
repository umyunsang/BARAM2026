"""M271 엔진 조립 — LangGraph `StateGraph` 위의 연산자 루프.

고정되는 것은 연산자 집합이고 성장하는 것은 NetworkX 발굴 그래프다. 여기서는 연산자 7 개를
노드로 등록하고 조건부 엣지로 잇는다. 새 실험 노드는 `Send("EXPERIMENT", {spec})` 로
파라미터화된 인스턴스로 실행되므로 컴파일 시점 토폴로지는 고정이어도 논리 노드는 무한히
자랄 수 있다.

  * 원칙 1 (State): `m271_state.M271State` 채널 + 선언 리듀서 + 체크포인터
  * 원칙 2 (조건부 엣지): `m271_router.decide` 가 다음 노드를 동적 선택
  * 원칙 3 (병렬): 정적 fan-out 은 superstep, 동적 fan-out 은 `Send`, 수렴 노드가 fan-in

P0 게이트(`reports/m271_framework_gate.md`)가 이 네 가지를 LangGraph 1.2.10 에서 실증했다.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

sys.path.insert(0, str(Path(__file__).resolve().parent))

from m271_router import Evidence, decide
from m271_state import M271State, initial_state

# 연산자 집합. 이 셋은 컴파일 시점에 고정된다.
OPERATORS = (
    "MEASURE",
    "RESEARCH_DIRECTION",
    "RESEARCH_SOTA",
    "MATERIALIZE",
    "EXPERIMENT",
    "GATE",
    "PRUNE",
    "JOIN",
)


def _event(node_id: str, kind: str, seq: int, **payload: Any) -> dict[str, Any]:
    return {
        "event_id": f"{node_id}:{kind}:{seq}",
        "node_id": node_id,
        "kind": kind,
        "seq": seq,
        "ts": payload.pop("ts", "1970-01-01T00:00:00Z"),
        **payload,
    }


# ---------------------------------------------------------------- 연산자


def op_measure(state: M271State) -> dict[str, Any]:
    node_id = state.get("pending_evidence", {}).get("node_id", "MEASURE")
    return {"events": [_event(node_id, "MEASURE_DONE", 1)]}


def _spec(state: Any) -> dict[str, Any]:
    """`Send` 로 도착하면 페이로드 자체가 state 다. 일반 엣지로 오면 `pending_evidence` 다.

    이 구분을 놓치면 노드가 기본값으로 떨어져 라우터의 판정이 조용히 무시된다. 실제로
    한 번 그렇게 됐고, 검증 단언이 '무언가 일어났다' 만 봐서 걸러내지 못했다.
    """
    if isinstance(state, dict) and "operator" in state:
        return dict(state)
    return dict((state or {}).get("pending_evidence", {}))


def op_research_direction(state: M271State) -> dict[str, Any]:
    """리서치 범위는 발화 노드의 레인으로 제한된다(영역 독립성)."""
    spec = _spec(state)
    if "operator" in spec and spec.get("operator") != "RESEARCH_DIRECTION":
        raise RuntimeError(f"wrong operator routed here: {spec.get('operator')}")
    lane = spec.get("lane")
    kind = spec.get("kind")
    if not lane or not kind:
        raise RuntimeError(
            "RESEARCH_DIRECTION received no lane/kind. The router decision was dropped."
        )
    origin = spec.get("origin_node", "n")
    key = f"dir::{origin}::{kind}::{lane}"
    return {
        "sources": {key: {"lane": lane, "kind": kind, "scope": spec.get("scope")}},
        "events": [_event(origin, "RESEARCH_DIRECTION", 2, lane=lane, research_kind=kind)],
    }


def op_research_sota(state: M271State) -> dict[str, Any]:
    """방향이 정해진 뒤에만 발화한다. `DIRECTION` 없이 단독 발화는 거부한다."""
    spec = state.get("pending_evidence", {})
    if not spec.get("direction_id"):
        raise RuntimeError(
            "RESEARCH_SOTA fired without a DIRECTION node. "
            "That is a regression to batch research and is refused."
        )
    key = f"sota::{spec['direction_id']}"
    return {
        "methods": {key: {"direction_id": spec["direction_id"], "lane": spec.get("lane")}},
        "events": [_event(spec.get("origin_node", "n"), "RESEARCH_SOTA", 3)],
    }


def op_materialize(state: M271State) -> dict[str, Any]:
    spec = state.get("pending_evidence", {})
    node_id = spec.get("new_node_id", "EXP")
    return {
        "hypotheses": {node_id: dict(spec)},
        "frontier": [node_id],
        "events": [_event(node_id, "MATERIALIZE", 4)],
    }


def op_experiment(state: M271State) -> dict[str, Any]:
    spec = state.get("pending_evidence", {})
    node_id = spec.get("node_id", "EXP")
    return {"events": [_event(node_id, "EXPERIMENT", 5)]}


def op_gate(state: M271State) -> dict[str, Any]:
    spec = state.get("pending_evidence", {})
    node_id = spec.get("node_id", "EXP")
    return {"events": [_event(node_id, "GATE", 6)]}


def op_prune(state: M271State) -> dict[str, Any]:
    spec = state.get("pending_evidence", {})
    return {"events": [_event(spec.get("node_id", "n"), "PRUNE", 7)]}


def op_join(state: M271State) -> dict[str, Any]:
    guards = dict(state.get("guards", {}))
    guards["macrostep"] = int(guards.get("macrostep", 0)) + 1
    return {"guards": guards, "events": [_event("JOIN", "JOIN", 8)]}


# ---------------------------------------------------------------- 조건부 엣지


def route_after_gate(state: M271State) -> str | list[Send]:
    """`m271_router.decide` 의 판정을 LangGraph 엣지로 옮긴다."""
    raw = state.get("pending_evidence", {})
    evidence = Evidence(
        evidence_id=raw.get("evidence_id", "e"),
        node_id=raw.get("node_id", "n"),
        lane=raw.get("lane", "L8"),
        deficit_cell=raw.get("deficit_cell"),
        gate=raw.get("gate"),
        sign=int(raw.get("sign", 0)),
        predeclared_sign=int(raw.get("predeclared_sign", 0)),
        information=float(raw.get("information", 1.0)),
        refutes_mechanism=raw.get("refutes_mechanism"),
        confirms=bool(raw.get("confirms", False)),
        novel_mechanism=raw.get("novel_mechanism"),
        expected_gain=float(raw.get("expected_gain", 0.0)),
        expected_hours=float(raw.get("expected_hours", 1.0)),
    )
    decision = decide(evidence, dict(state.get("guards", {}) | raw.get("router_context", {})))

    if decision.action == "HALT":
        return "JOIN"
    if decision.action == "PRUNE":
        return "PRUNE"
    if decision.action in {"REVIVE", "REFINE_AXIS"}:
        return "JOIN"
    # 동적 fan-out: 분기 수를 실행 전에 모른다.
    return [Send("RESEARCH_DIRECTION", dict(t)) for t in decision.targets]


def build_app(checkpointer: Any | None = None):
    builder = StateGraph(M271State)
    builder.add_node("MEASURE", op_measure)
    builder.add_node("RESEARCH_DIRECTION", op_research_direction)
    builder.add_node("RESEARCH_SOTA", op_research_sota)
    builder.add_node("MATERIALIZE", op_materialize)
    builder.add_node("EXPERIMENT", op_experiment)
    builder.add_node("GATE", op_gate)
    builder.add_node("PRUNE", op_prune)
    builder.add_node("JOIN", op_join)

    builder.add_edge(START, "MEASURE")
    builder.add_edge("MEASURE", "GATE")
    builder.add_conditional_edges(
        "GATE", route_after_gate, ["RESEARCH_DIRECTION", "PRUNE", "JOIN"]
    )
    builder.add_edge("RESEARCH_DIRECTION", "JOIN")
    builder.add_edge("PRUNE", "JOIN")
    builder.add_edge("JOIN", END)
    return builder.compile(checkpointer=checkpointer or InMemorySaver())


def fresh_state(**overrides: Any) -> M271State:
    state = initial_state(
        rules={"target_total": 0.66, "deadline": "2026-08-14T09:59:00+09:00"},
        data_contract={"open_zip_sha256": "pending"},
        budget={"wallclock_hours": 24.0, "model_workers": 6},
    )
    state.update(overrides)
    return state
