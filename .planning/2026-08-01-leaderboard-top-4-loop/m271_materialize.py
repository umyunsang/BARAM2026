"""M271 실체화 — 사양을 발굴 그래프의 새 노드로 만든다.

리서치 사이클 ③ 이다. 여기서 그래프가 실제로 자란다.

  * NetworkX 발굴 그래프에 새 `EXPERIMENT` 노드를 만든다
  * 계보 엣지 3 종을 건다 — `derived_from`(SOURCE) · `addresses`(DEFICIT) ·
    `instantiates`(DIRECTION)
  * **사전확약을 동결한다** — 예상 부호, 기대 효과크기, 적용 게이트, 중단 조건.
    이 동결이 C5(부호 역전)의 판정 기준이 된다.

실행 가능한 사양이 안 나오면 여기서 종결하고 그 사실을 증거로 남긴다. 억지로 노드를 만들어
프론티어를 채우지 않는다.

`StateGraph` 노드 집합은 컴파일 시점에 고정되므로 여기서 만든 논리 노드는 런타임에
`Send("EXPERIMENT", {spec})` 파라미터 인스턴스로 실행된다.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import m271_excavation_graph as xg
from m271_research import Specification

LINEAGE_EDGES = ("derived_from", "addresses", "instantiates")


class UnmaterialisableSpec(RuntimeError):
    """실행 가능한 사양이 아니다. 노드를 만들지 않고 그 사실을 증거로 남긴다."""


@dataclass(frozen=True)
class Predeclaration:
    """실행 **전에** 동결한다. 결과를 본 뒤 재해석하지 않는다."""

    node_id: str
    expected_sign: int  # +1 개선 기대 / -1 악화 기대 / 0 기대 없음
    expected_effect: float  # Total 단위의 기대 효과크기
    gate_version: str
    stop_condition: str
    parent_candidate: str

    def digest(self) -> str:
        return hashlib.sha256(
            json.dumps(asdict(self), ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest()


def _validate(spec: Specification) -> None:
    if not spec.method_name:
        raise UnmaterialisableSpec("spec has no method name")
    if not spec.required_inputs:
        raise UnmaterialisableSpec("spec declares no required inputs")
    if not spec.within_official_rules:
        raise UnmaterialisableSpec(
            f"spec {spec.spec_id} needs inputs outside the official rule boundary"
        )


def materialize(
    graph: xg.ExcavationGraph,
    spec: Specification,
    *,
    node_id: str,
    parent_node: str,
    deficit_cell: str | None,
    direction_id: str,
    source_ids: list[str],
    expected_sign: int,
    expected_effect: float,
    gate_version: str,
    stop_condition: str,
    parent_candidate: str,
) -> tuple[str, Predeclaration]:
    """새 실험 노드를 만들고 계보와 사전확약을 건다."""
    _validate(spec)

    # 프론티어에서만 확장한다. 폐기 노드는 여기서 예외로 거부된다.
    graph.expand(
        parent_node,
        node_id,
        node_type="EXPERIMENT",
        lane=spec.lane,
        status=xg.PROPOSED,
        spec_id=spec.spec_id,
        method_name=spec.method_name,
    )

    for source_id in sorted(source_ids):
        if source_id not in graph.graph:
            graph.add_node(source_id, node_type="SOURCE", lane=spec.lane, status=xg.LIVE)
        graph.link(source_id, node_id, "derived_from")

    if deficit_cell is not None:
        if deficit_cell not in graph.graph:
            graph.add_node(deficit_cell, node_type="DEFICIT", lane=spec.lane, status=xg.LIVE)
        graph.link(node_id, deficit_cell, "addresses")

    if direction_id not in graph.graph:
        graph.add_node(direction_id, node_type="DIRECTION", lane=spec.lane, status=xg.LIVE)
    graph.link(direction_id, node_id, "instantiates")

    pre = Predeclaration(
        node_id=node_id,
        expected_sign=expected_sign,
        expected_effect=expected_effect,
        gate_version=gate_version,
        stop_condition=stop_condition,
        parent_candidate=parent_candidate,
    )
    graph.graph.nodes[node_id]["predeclaration"] = asdict(pre)
    graph.graph.nodes[node_id]["spec_hash"] = pre.digest()
    graph.transition(node_id, xg.PREDECLARED)
    return node_id, pre


def lineage_of(graph: xg.ExcavationGraph, node_id: str) -> dict[str, list[str]]:
    """계보 엣지 3 종을 조회한다. 검증 8 번이 이걸 본다."""
    out: dict[str, list[str]] = {kind: [] for kind in LINEAGE_EDGES}
    for source, _, data in graph.graph.in_edges(node_id, data=True):
        kind = data.get("kind")
        if kind in out:
            out[kind].append(source)
    for _, target, data in graph.graph.out_edges(node_id, data=True):
        kind = data.get("kind")
        if kind in out:
            out[kind].append(target)
    return {k: sorted(v) for k, v in out.items()}


def unmaterialisable_evidence(spec: Specification, reason: str) -> dict[str, Any]:
    """노드를 만들지 못했을 때 남기는 증거. 실패도 기록이다."""
    return {
        "spec_id": spec.spec_id,
        "direction_id": spec.direction_id,
        "lane": spec.lane,
        "outcome": "UNMATERIALISABLE",
        "reason": reason,
        "note": "억지로 노드를 만들어 프론티어를 채우지 않는다.",
    }
