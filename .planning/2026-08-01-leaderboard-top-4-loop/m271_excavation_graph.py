"""M271 발굴 그래프 — 성장·생명주기·프론티어·가지치기·부활.

챔피언 노드는 미리 존재하지 않는다. 파 내려가면서 층층이 만들어지므로 그래프가 증거에
따라 유동적으로 확장되어야 한다. 고정되는 것은 연산자 집합이고 성장하는 것이 이 그래프다.

`StateGraph` 는 노드 집합이 컴파일 시점에 고정되므로 새 노드는 여기서 생성되고 런타임에는
`Send` 파라미터 인스턴스로 실행된다. 계보와 부활 판정은 전부 이 층이 보유한다.

가지치기의 성질:
  * `PRUNED` 는 흡수 상태다. `expand()` 는 우선순위를 낮추는 게 아니라 **예외로 거부**한다.
  * 폐기는 하위로 전파하되, 다른 `LIVE` 조상에서 **별개 전제**로 도달 가능한 후손은 살린다.
  * 폐기 사유는 **기계검증 술어**로 남긴다. 영구 삭제가 아니라 그 전제가 유지되는 동안의
    흡수이며, 전제가 뒤집히면 되살린다. 이전 세션의 `REVIVABLE_PREMISES` 키워드 매칭이
    하려다 실패한 일이다.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import networkx as nx

# 생명주기. 마지막 셋은 흡수 상태다.
PROPOSED = "PROPOSED"
PREDECLARED = "PREDECLARED"
RUNNING = "RUNNING"
EVIDENCED = "EVIDENCED"
LIVE = "LIVE"
PRUNED = "PRUNED"
CHAMPION = "CHAMPION"

LIFECYCLE: tuple[str, ...] = (PROPOSED, PREDECLARED, RUNNING, EVIDENCED, LIVE, PRUNED, CHAMPION)
ABSORBING = frozenset({PRUNED, CHAMPION})

LEGAL_TRANSITIONS: dict[str, frozenset[str]] = {
    PROPOSED: frozenset({PREDECLARED, PRUNED}),
    PREDECLARED: frozenset({RUNNING, PRUNED}),
    RUNNING: frozenset({EVIDENCED, PRUNED}),
    EVIDENCED: frozenset({LIVE, PRUNED, CHAMPION}),
    LIVE: frozenset({PRUNED, CHAMPION}),
    PRUNED: frozenset(),  # 흡수
    CHAMPION: frozenset(),  # 흡수
}

LANES: tuple[str, ...] = ("L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8")

# 계보 엣지. MATERIALIZE 가 새 노드를 만들 때 이 셋을 건다.
EDGE_KINDS: tuple[str, ...] = (
    "derived_from",  # SOURCE -> node
    "addresses",  # node -> DEFICIT
    "instantiates",  # DIRECTION -> node
    "depends_on",
    "supersedes",
    "follows",  # 시간순 인접. 의존이 아니다
)

# 폐기가 전파되는 엣지. **의존만 전파한다.**
#
# 폐기 전제는 "이 탐구선을 닫는다"는 뜻이므로, 같은 탐구선을 잇는 후손에게만 번져야
# 한다. `follows` 는 사이클 번호가 이어진다는 것 이상을 뜻하지 않고, `addresses` 는
# 오히려 부모의 결함을 **고치러 온** 노드이므로 그 결함을 물려받으면 안 된다.
# `supersedes` 는 신->구 방향이라 전파하면 새 노드가 옛 노드를 죽인다.
#
# 이 구분이 없을 때 실제로 벌어진 일: `C1N32` 하나의 폐기가 `follows` 사슬을 타고
# 30 개 노드를 삼켰고, 그 안에 **현재 챔피언을 만든 C1N35** 가 있었다.
DEPENDENCY_KINDS: frozenset[str] = frozenset(
    {"depends_on", "derived_from", "instantiates"}
)


class PruneViolation(RuntimeError):
    """폐기 노드에서 확장을 시도했다. 우선순위 문제가 아니라 금지다."""


class LifecycleViolation(RuntimeError):
    """생명주기가 허용하지 않는 전이를 시도했다."""


@dataclass
class Premise:
    """기계검증 폐쇄 전제.

    `code` 는 안정 식별자, `holds` 는 상태를 받아 전제가 아직 참인지 판정하는 술어다.
    술어가 거짓으로 뒤집히면 그 전제로 폐기된 하위그래프가 부활 대상이 된다.
    """

    code: str
    description: str
    holds: Callable[[dict], bool]

    def evaluate(self, state: dict) -> bool:
        return bool(self.holds(state))


@dataclass
class ExcavationGraph:
    graph: nx.DiGraph = field(default_factory=nx.DiGraph)
    premises: dict[str, Premise] = field(default_factory=dict)

    # ---------------------------------------------------------------- 성장
    def add_node(
        self,
        node_id: str,
        *,
        node_type: str,
        lane: str,
        parents: list[str] | None = None,
        spec_hash: str | None = None,
        status: str = PROPOSED,
        **attrs: Any,
    ) -> str:
        if lane not in LANES:
            raise ValueError(f"unknown lane: {lane}")
        if status not in LIFECYCLE:
            raise ValueError(f"unknown status: {status}")
        for parent in parents or []:
            if parent not in self.graph:
                raise KeyError(f"parent not in graph: {parent}")
            if self.graph.nodes[parent]["status"] == PRUNED:
                raise PruneViolation(
                    f"cannot grow from a pruned node: {parent} -> {node_id}. "
                    "PRUNED is absorbing; revive the premise first."
                )
        self.graph.add_node(
            node_id,
            node_type=node_type,
            lane=lane,
            status=status,
            spec_hash=spec_hash,
            premise=None,
            **attrs,
        )
        for parent in parents or []:
            self.graph.add_edge(parent, node_id, kind="depends_on")
        return node_id

    def link(self, source: str, target: str, kind: str) -> None:
        if kind not in EDGE_KINDS:
            raise ValueError(f"unknown edge kind: {kind}")
        for node in (source, target):
            if node not in self.graph:
                raise KeyError(f"node not in graph: {node}")
        self.graph.add_edge(source, target, kind=kind)

    def expand(self, parent: str, child_id: str, **kwargs: Any) -> str:
        """프론티어에서만 확장한다. 폐기 노드는 정적으로 거부된다."""
        if parent not in self.graph:
            raise KeyError(f"node not in graph: {parent}")
        status = self.graph.nodes[parent]["status"]
        if status == PRUNED:
            raise PruneViolation(
                f"expand() refused: {parent} is PRUNED (absorbing). "
                f"premise={self.graph.nodes[parent].get('premise')}"
            )
        if status == CHAMPION:
            raise PruneViolation(f"expand() refused: {parent} is CHAMPION (absorbing).")
        return self.add_node(child_id, parents=[parent], **kwargs)

    # ---------------------------------------------------------------- 생명주기
    def transition(self, node_id: str, new_status: str) -> None:
        current = self.graph.nodes[node_id]["status"]
        if new_status not in LEGAL_TRANSITIONS[current]:
            raise LifecycleViolation(f"{node_id}: {current} -> {new_status} is not legal")
        self.graph.nodes[node_id]["status"] = new_status

    def frontier(self) -> list[str]:
        """확장 가능한 노드 = LIVE 집합. 정렬해 결정성을 보장한다."""
        return sorted(n for n, d in self.graph.nodes(data=True) if d["status"] == LIVE)

    # ---------------------------------------------------------------- 가지치기
    def prune(self, node_id: str, premise_code: str) -> list[str]:
        """노드와 그 하위를 폐기한다. 다른 LIVE 조상에서 별개 전제로 닿는 후손은 살린다."""
        if premise_code not in self.premises:
            raise KeyError(f"prune premise is not registered: {premise_code}")
        pruned: list[str] = []
        self._mark_pruned(node_id, premise_code, pruned)

        # 의존 엣지가 하나도 없는 노드는 뷰에 없다 — 전파할 하류가 없다는 뜻이다.
        view = self._dependency_view()
        if node_id not in view:
            return sorted(pruned)

        for descendant in sorted(nx.descendants(view, node_id)):
            if self.graph.nodes[descendant]["status"] == PRUNED:
                continue
            if self._reachable_from_live_elsewhere(descendant, blocked=node_id):
                continue
            self._mark_pruned(descendant, premise_code, pruned)
        return sorted(pruned)

    def _mark_pruned(self, node_id: str, premise_code: str, sink: list[str]) -> None:
        data = self.graph.nodes[node_id]
        if data["status"] == PRUNED:
            return
        if PRUNED not in LEGAL_TRANSITIONS[data["status"]]:
            raise LifecycleViolation(f"{node_id}: {data['status']} -> PRUNED is not legal")
        data["status"] = PRUNED
        data["premise"] = premise_code
        sink.append(node_id)

    def _dependency_view(self) -> nx.DiGraph:
        """의존 엣지만 남긴 뷰. 폐기 전파와 도달성은 이 위에서만 본다."""
        return self.graph.edge_subgraph(
            [
                (u, v)
                for u, v, kind in self.graph.edges(data="kind")
                if kind in DEPENDENCY_KINDS
            ]
        )

    def _reachable_from_live_elsewhere(self, node_id: str, blocked: str) -> bool:
        """`blocked` 를 거치지 않고 LIVE 조상에서 **의존을 통해** 도달 가능한지."""
        view = self._dependency_view()
        if node_id not in view:
            return False
        for ancestor in nx.ancestors(view, node_id):
            if ancestor == blocked:
                continue
            if self.graph.nodes[ancestor]["status"] != LIVE:
                continue
            trimmed = view.subgraph([n for n in view.nodes if n != blocked])
            if ancestor in trimmed and node_id in trimmed and nx.has_path(
                trimmed, ancestor, node_id
            ):
                return True
        return False

    # ---------------------------------------------------------------- 부활
    def register_premise(self, premise: Premise) -> None:
        self.premises[premise.code] = premise

    def flipped_premises(self, state: dict) -> list[str]:
        """더 이상 참이 아닌 전제. 이들로 폐기된 하위그래프가 부활 대상이다."""
        return sorted(code for code, p in self.premises.items() if not p.evaluate(state))

    def revive(self, premise_code: str) -> list[str]:
        """전제가 뒤집혔을 때 그 전제로 폐기된 노드를 되살린다."""
        if premise_code not in self.premises:
            raise KeyError(f"unknown premise: {premise_code}")
        revived = []
        for node, data in self.graph.nodes(data=True):
            if data["status"] == PRUNED and data.get("premise") == premise_code:
                data["status"] = LIVE
                data["premise"] = None
                revived.append(node)
        return sorted(revived)

    # ---------------------------------------------------------------- 조회
    def lane_census(self) -> dict[str, dict[str, int]]:
        census = {lane: dict.fromkeys(LIFECYCLE, 0) for lane in LANES}
        for _, data in self.graph.nodes(data=True):
            census[data["lane"]][data["status"]] += 1
        return census

    def live_count(self, lane: str) -> int:
        return sum(
            1
            for _, d in self.graph.nodes(data=True)
            if d["lane"] == lane and d["status"] == LIVE
        )

    def generations(self) -> list[list[str]]:
        """의존 그래프의 위상 세대. 같은 세대는 서로 독립이므로 병렬 실행 대상이다."""
        dependency = nx.DiGraph()
        dependency.add_nodes_from(self.graph.nodes)
        for u, v, data in self.graph.edges(data=True):
            if data.get("kind") == "depends_on":
                dependency.add_edge(u, v)
        return [sorted(layer) for layer in nx.topological_generations(dependency)]

    def digest(self) -> str:
        """그래프 내용 해시. 두 번 재구성해 같은지 확인하는 데 쓴다."""
        import hashlib
        import json

        payload = {
            "nodes": sorted(
                (n, {k: v for k, v in d.items() if k != "_runtime"})
                for n, d in self.graph.nodes(data=True)
            ),
            "edges": sorted((u, v, d.get("kind")) for u, v, d in self.graph.edges(data=True)),
        }
        text = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
