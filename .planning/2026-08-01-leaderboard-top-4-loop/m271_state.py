"""M271 상태 저장소 — 채널 스키마와 리듀서 (그래프 엔지니어링 원칙 1).

모든 노드 시그니처는 `node(state) -> delta` 다. 노드는 상태를 직접 변형하지 않고, 델타는
**채널별로 선언된 리듀서**를 통해서만 병합된다. 상태는 모든 노드에 일관된 컨텍스트로
전달된다.

재현성이 여기서 결정된다. P0 게이트가 LangGraph 1.2.10 에서 병렬 분기의 쓰기가 노드 등록
순서로 리듀스됨을 측정했지만 그것은 **문서화된 보증이 아니다**. 따라서 모든 리듀서를
순서 무관하게 쓰거나 병합 후 정규 정렬한다. 그러면 스케줄과 무관하게 같은 상태가 나온다.

세 가지 규율:
  1. `events` 는 append 후 `(ts, node_id, seq)` 정규 정렬 → 순서 무관
  2. `sources/methods/hypotheses/evidence` 는 **1회 기록 후 불변**, 충돌은 오류
  3. `deficits` 는 **셀 단위 쓰기 소유권**, 다른 소유자의 셀을 쓰면 오류
"""

from __future__ import annotations

import json
from typing import Annotated, Any, TypedDict

# ---------------------------------------------------------------- 리듀서


class ReducerConflict(RuntimeError):
    """두 분기가 같은 키를 서로 다른 값으로 쓰려 했다. 조용히 덮지 않는다."""


def append_sorted(left: list[dict] | None, right: list[dict] | None) -> list[dict]:
    """이벤트 append 후 정규 정렬. 완료 순서와 무관하게 같은 결과를 낸다."""
    merged = list(left or []) + list(right or [])
    seen: set[str] = set()
    unique: list[dict] = []
    for event in merged:
        key = event.get("event_id") or json.dumps(event, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        unique.append(event)
    return sorted(
        unique,
        key=lambda e: (str(e.get("ts", "")), str(e.get("node_id", "")), int(e.get("seq", 0))),
    )


def write_once(left: dict | None, right: dict | None) -> dict:
    """키 병합. 같은 키에 다른 값을 쓰면 오류. 기록은 불변이다."""
    merged = dict(left or {})
    for key, value in (right or {}).items():
        if key in merged and merged[key] != value:
            raise ReducerConflict(f"write-once channel already holds a different value: {key}")
        merged[key] = value
    return merged


def owned_cells(left: dict | None, right: dict | None) -> dict:
    """결손 셀 병합. 셀마다 소유자가 있고 다른 소유자는 쓸 수 없다.

    두 분기가 같은 셀을 동시에 갱신하는 상황 자체를 배제하므로, 병합 순서가 결과를 바꿀 수
    없다. 이것이 병렬 실행과 순차 실행을 같게 만드는 두 번째 장치다.
    """
    merged = dict(left or {})
    for key, value in (right or {}).items():
        existing = merged.get(key)
        if existing is None:
            merged[key] = value
            continue
        old_owner, new_owner = existing.get("owner"), value.get("owner")
        if old_owner is not None and new_owner is not None and old_owner != new_owner:
            raise ReducerConflict(
                f"deficit cell {key} is owned by {old_owner}, {new_owner} may not write it"
            )
        merged[key] = value
    return merged


def best_score(left: dict | None, right: dict | None) -> dict:
    """공식 Total 최대. 동률은 node_id 사전순으로 고정한다."""
    if not left:
        return dict(right or {})
    if not right:
        return dict(left)
    lt, rt = float(left.get("total", -1.0)), float(right.get("total", -1.0))
    if rt > lt:
        return dict(right)
    if lt > rt:
        return dict(left)
    return dict(min([left, right], key=lambda d: str(d.get("node_id", ""))))


def union_sorted(left: list[str] | None, right: list[str] | None) -> list[str]:
    return sorted(set(left or []) | set(right or []))


def merge_scalar(left: Any, right: Any) -> Any:
    """마지막 기록 우선. 불변 채널과 계산 채널에 쓴다."""
    return left if right is None else right


def subtract_budget(left: dict | None, right: dict | None) -> dict:
    """예산 감산. 절대값이 아니라 사용량을 누적해 순서 무관하게 만든다."""
    merged = dict(left or {})
    for key, value in (right or {}).items():
        if key.startswith("spent_"):
            merged[key] = float(merged.get(key, 0.0)) + float(value)
        else:
            merged[key] = value
    return merged


# ---------------------------------------------------------------- 상태 스키마


class M271State(TypedDict, total=False):
    """모든 노드에 일관되게 전달되는 상태 저장소."""

    # 불변 컨텍스트 (읽기 전용)
    rules: Annotated[dict, merge_scalar]
    data_contract: Annotated[dict, merge_scalar]

    # 누적 채널
    events: Annotated[list[dict], append_sorted]
    sources: Annotated[dict, write_once]
    methods: Annotated[dict, write_once]
    hypotheses: Annotated[dict, write_once]
    evidence: Annotated[dict, write_once]

    # 상태 채널
    deficits: Annotated[dict, owned_cells]
    best_local: Annotated[dict, best_score]
    offsets: Annotated[dict, write_once]

    # 루프 제어
    frontier: Annotated[list[str], union_sorted]
    budget: Annotated[dict, subtract_budget]
    guards: Annotated[dict, merge_scalar]

    # 라우팅 입력 — 직전 노드가 낸 증거 서명
    pending_evidence: Annotated[dict, merge_scalar]


CHANNELS: tuple[str, ...] = (
    "rules",
    "data_contract",
    "events",
    "sources",
    "methods",
    "hypotheses",
    "evidence",
    "deficits",
    "best_local",
    "offsets",
    "frontier",
    "budget",
    "guards",
    "pending_evidence",
)

REDUCERS = {
    "rules": merge_scalar,
    "data_contract": merge_scalar,
    "events": append_sorted,
    "sources": write_once,
    "methods": write_once,
    "hypotheses": write_once,
    "evidence": write_once,
    "deficits": owned_cells,
    "best_local": best_score,
    "offsets": write_once,
    "frontier": union_sorted,
    "budget": subtract_budget,
    "guards": merge_scalar,
    "pending_evidence": merge_scalar,
}

# 순서 무관 리듀서. 병렬 실행이 순차 실행과 같은 결과를 내는 근거다.
ORDER_INSENSITIVE = frozenset(
    {"events", "sources", "methods", "hypotheses", "evidence", "deficits", "best_local",
     "frontier", "budget"}
)


def initial_state(rules: dict, data_contract: dict, budget: dict) -> M271State:
    return {
        "rules": rules,
        "data_contract": data_contract,
        "events": [],
        "sources": {},
        "methods": {},
        "hypotheses": {},
        "evidence": {},
        "deficits": {},
        "best_local": {},
        "offsets": {},
        "frontier": [],
        "budget": budget,
        "guards": {"macrostep": 0, "stall_counter": 0, "loop_engine_visits": 0},
        "pending_evidence": {},
    }


def state_digest(state: M271State) -> str:
    """상태의 내용 해시. 병렬 실행과 순차 실행을 비교하는 데 쓴다."""
    import hashlib

    payload = {k: state.get(k) for k in CHANNELS}
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
