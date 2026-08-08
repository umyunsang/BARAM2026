
"""Graph engineering: the excavation is a DAG of nodes, persisted, with lineage and a champion.

Why a graph and not the previous flat ladder.  The old workflow was a list of eleven stages that
each got closed once.  Every stage's closure was recorded before the binding constraint was
known, and S12-N21 found that all eighteen S12 nodes were siblings at one rung -- the ladder had
no way to express "this rung must be re-entered because an upstream fact changed".  A DAG does:
a measurement is a node, it can invalidate ancestors, and re-entry is an edge, not a rewrite.
"""
from __future__ import annotations
import json, os, datetime
from pathlib import Path
import networkx as nx

from contract import Node, CHAMPION_SEED, admit

ROOT = Path('/Users/um-yunsang/BARAM2026/research/engine')
STORE = ROOT / 'graph.json'


class ExcavationGraph:
    def __init__(self):
        self.G = nx.DiGraph()
        self.nodes: dict[str, Node] = {}
        self.champion: str | None = None
        self.champion_record: dict = dict(CHAMPION_SEED)
        self.comparisons: int = 0
        self.log: list[dict] = []

    # ---------------------------------------------------------------- persistence
    def save(self, path: Path = STORE):
        path.parent.mkdir(parents=True, exist_ok=True)
        json.dump({'nodes': {k: v.to_dict() for k, v in self.nodes.items()},
                   'edges': list(self.G.edges()),
                   'champion': self.champion,
                   'champion_record': self.champion_record,
                   'comparisons': self.comparisons,
                   'log': self.log,
                   'saved': datetime.datetime.now().isoformat(timespec='seconds')},
                  open(path, 'w'), indent=1, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path = STORE) -> 'ExcavationGraph':
        g = cls()
        if not Path(path).exists():
            return g
        d = json.load(open(path))
        for k, v in d['nodes'].items():
            g.nodes[k] = Node(**v)
            g.G.add_node(k)
        g.G.add_edges_from([tuple(e) for e in d['edges']])
        g.champion = d.get('champion')
        g.champion_record = d.get('champion_record', dict(CHAMPION_SEED))
        g.comparisons = d.get('comparisons', 0)
        g.log = d.get('log', [])
        return g

    # ---------------------------------------------------------------- mutation
    def propose(self, node: Node) -> tuple[bool, str]:
        ok, why = admit(node)
        node.status = 'proposed' if ok else 'refused'
        node.notes = (node.notes + ' | ' if node.notes else '') + why
        self.nodes[node.id] = node
        self.G.add_node(node.id)
        if node.parent:
            self.G.add_edge(node.parent, node.id)
        self._log('propose', node.id, why)
        return ok, why

    def set_status(self, nid: str, status: str, **kw):
        n = self.nodes[nid]
        n.status = status
        for k, v in kw.items():
            setattr(n, k, v)
        self._log('status', nid, status)

    def record_result(self, nid: str, result: dict):
        self.nodes[nid].result = result
        self.nodes[nid].status = 'run'
        self._log('result', nid, json.dumps(result, default=str)[:300])

    def record_arbitration(self, nid: str, arb: dict, took_champion: bool):
        self.nodes[nid].arbitration = arb
        self.comparisons += 1
        if took_champion:
            if self.champion:
                self.nodes[self.champion].status = 'arbitrated'
            self.champion = nid
            self.nodes[nid].status = 'champion'
            self.champion_record = {'id': nid, 'title': self.nodes[nid].title,
                                    'score': self.nodes[nid].result, 'provenance':
                                    self.nodes[nid].prereg.get('provenance')}
        else:
            self.nodes[nid].status = 'rejected'
        self._log('arbitrate', nid, f'took_champion={took_champion} n_comp={self.comparisons}')

    def _log(self, kind: str, nid: str, msg: str):
        self.log.append({'t': datetime.datetime.now().isoformat(timespec='seconds'),
                         'kind': kind, 'node': nid, 'msg': msg})

    # ---------------------------------------------------------------- views
    def frontier(self) -> list[Node]:
        """Admitted, not yet run."""
        return [n for n in self.nodes.values() if n.status in ('proposed', 'selected')]

    def summary(self) -> str:
        from collections import Counter
        c = Counter(n.status for n in self.nodes.values())
        ch = self.champion_record
        s = [f'nodes={len(self.nodes)} {dict(c)}  comparisons={self.comparisons}',
             f'champion: {ch.get("id")} {ch.get("title")}  '
             f'total={ch.get("score",{}).get("total")}']
        return '\n'.join(s)
