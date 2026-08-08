"""M270 experiment control: NetworkX hypothesis graph + Optuna ask-and-tell loop.

Replaces the bespoke graph builder with verified frameworks.

  * NetworkX (BSD-3-Clause) holds the hypothesis graph. Prune propagation and premise
    dependency become standard graph queries (`descendants` / `ancestors`) instead of
    hand-rolled traversal.
  * Optuna (MIT) drives the loop through its ASK-AND-TELL interface with a persistent
    SQLite study, so a session interruption does not lose loop state.

The loop discipline is unchanged and deliberately NOT delegated: every experiment is
predeclared with a frozen gate, run exactly once, and reproduced byte-for-byte. Optuna
only carries bookkeeping and ordering. Automatic hyperparameter search is FORBIDDEN here
because it reintroduces the same-fold overfitting this project has repeatedly rejected.

Read-only over the planning record. No model is fitted, no 2024 row is read.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import networkx as nx
import optuna

ROOT = Path(__file__).resolve().parents[2]
PLAN_DIR = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"

TASK_PLAN = PLAN_DIR / "task_plan.md"
PROGRESS = PLAN_DIR / "progress.md"
GRAPH_OUT = PLAN_DIR / "m270_graph.json"
STUDY_DB = PLAN_DIR / "m270_study.db"
STUDY_NAME = "m270_experiment_loop"

NODE_RE = re.compile(r"\bM(\d{1,3})\b")
PARENT_RE = re.compile(r"\bexact M(\d{1,3})\b|\bM(\d{1,3}) parent\b|\bover M(\d{1,3})\b")

VERDICTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("rejected", ("reject", "close ", "closed", "do not ", "stop ", "forbid")),
    ("promoted", ("promote", "pass the", "build and freeze", "make ", "keep ", "retain")),
    ("built", ("build ", "hand off")),
    ("predeclared", ("predeclare",)),
)

REVIVABLE_PREMISES: dict[str, tuple[str, ...]] = {
    "NO_SELECTABLE_NWP": (
        "selectable nwp", "multiple nwp configurations", "unavailable selectable",
    ),
    "NO_LIVE_OBSERVATION": (
        "live observations", "unavailable at inference", "contemporaneous scada",
    ),
    "NO_EXTERNAL_DATA": ("external data", "requires unavailable", "unavailable inputs"),
    "NO_NEW_DEPENDENCY": (
        "tft", "ngboost", "deep tier", "dependency change", "optional dependency",
    ),
    "SHORT_GROUP3_HISTORY": (
        "group 3 begins in 2023", "missing 2022 group-3", "zero overlap",
    ),
}
EVIDENCE_PHRASES = (
    "regressed", "bootstrap positivity", "fell below", "negative",
    "worsened", "collapsed", "did not improve", "failed",
)
DUPLICATE_PHRASES = ("already", "duplicate", "duplicates", "covered")
ORACLE_PHRASES = ("same-fold", "oracle")

# Frozen candidate lanes carried forward from the P0 manual review and the P2 gate.
# These seed the Optuna queue; each still needs its own predeclaration before running.
CANDIDATE_LANES: tuple[dict[str, str], ...] = (
    {
        "lane": "gefs_ensemble_spread_features",
        "disposition": "UNOPENED",
        "axis": "EXTERNAL_DATA",
        "rationale": (
            "supplied GFS/LDAPS are deterministic single runs, so ensemble spread - a direct"
            " physical uncertainty covariate - is structurally absent. M269 diagnosed"
            " insufficient sharpness of the conditional distribution, which is what this signal"
            " addresses."
        ),
        "gate_hint": "previous-day 18Z cycle only; availability proven from the S3 key path",
    },
    {
        "lane": "distributional_learning",
        "disposition": "UNOPENED",
        "axis": "NEW_DEPENDENCY",
        "rationale": (
            "NGBoost / CRPS-style objectives model the conditional distribution directly rather"
            " than post-processing a 46-bin classifier whose oracle ceiling M269 measured at"
            " pooled FICR 0.4227 against a 0.44675 requirement."
        ),
        "gate_hint": "pin versions; fix seeds and threads; reject the family if not reproducible",
    },
    {
        "lane": "ramp_weather_pattern_classification",
        "disposition": "REVIVED",
        "axis": "EXTERNAL_DATA",
        "rationale": (
            "closed only because the competition exposed a single NWP source; public forecast"
            " archives remove that blocker. The closure carried no negative evidence."
        ),
        "gate_hint": "reuse the existing analog/PCA machinery; do not reopen rejected variants",
    },
    {
        "lane": "pretrained_timeseries_foundation_models",
        "disposition": "UNOPENED",
        "axis": "PRETRAINED_WEIGHTS",
        "rationale": (
            "eligible under Apache-2.0 with a large margin to the 2026-07-05 cutoff, but these"
            " forecast from series history while this task is covariate-driven at 12-36h lead."
            " Low priority until the higher-value axes are exhausted."
        ),
        "gate_hint": "Apache-2.0 models only; Moirai family is forbidden (CC-BY-NC-4.0)",
    },
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _decision_rows() -> list[tuple[str, str]]:
    text = TASK_PLAN.read_text(encoding="utf-8")
    start = text.index("## Decisions Made")
    end = text.index("## Active success criterion")
    rows: list[tuple[str, str]] = []
    for line in text[start:end].split("\n"):
        if not line.startswith("|") or line.count("|") < 3:
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 2 or cells[0] in {"Decision", ""} or set(cells[0]) <= set("-: "):
            continue
        rows.append((cells[0], cells[1]))
    return rows


def _classify_verdict(decision: str) -> str | None:
    lowered = decision.lower()
    for status, keywords in VERDICTS:
        if any(keyword in lowered for keyword in keywords):
            return status
    return None


def _classify_premise(text: str) -> tuple[str, str | None, bool]:
    lowered = text.lower()
    for code, phrases in REVIVABLE_PREMISES.items():
        for phrase in phrases:
            if phrase in lowered:
                return code, phrase, True
    for phrase in ORACLE_PHRASES:
        if phrase in lowered:
            return "SAME_FOLD_ORACLE_ONLY", phrase, False
    for phrase in DUPLICATE_PHRASES:
        if phrase in lowered:
            return "DUPLICATE_COVERAGE", phrase, False
    for phrase in EVIDENCE_PHRASES:
        if phrase in lowered:
            return "EVIDENCE_NEGATIVE", phrase, False
    return "UNCLASSIFIED", None, False


def build_graph() -> nx.DiGraph:
    """Construct the hypothesis graph.

    Closure premises are attributed ONLY from a node's own closing row. Two earlier bugs
    are fenced out here: reading phrases from rows that merely mention a node, and reading
    prohibition clauses in predeclaration rows as capability blockers.
    """
    graph = nx.DiGraph()
    closing_text: dict[str, list[str]] = {}

    for decision, rationale in _decision_rows():
        subject_match = NODE_RE.search(decision)
        if not subject_match:
            continue
        subject = f"M{subject_match.group(1)}"
        verdict = _classify_verdict(decision)
        if subject not in graph:
            graph.add_node(subject, status="documented", decision_rows=0, progress_rows=0)
        graph.nodes[subject]["decision_rows"] += 1
        if verdict:
            graph.nodes[subject]["status"] = verdict
        if verdict == "rejected":
            closing_text.setdefault(subject, []).append(decision + " " + rationale)

        blob = decision + " " + rationale
        parents = {f"M{g}" for match in PARENT_RE.finditer(blob) for g in match.groups() if g}
        for other in {f"M{v}" for v in NODE_RE.findall(blob)} - {subject}:
            if other not in graph:
                graph.add_node(other, status="documented", decision_rows=0, progress_rows=0)
            kind = "parent" if other in parents else "references"
            if not graph.has_edge(other, subject):
                graph.add_edge(other, subject, kind=kind)
            elif kind == "parent":
                graph.edges[other, subject]["kind"] = "parent"

    for line in PROGRESS.read_text(encoding="utf-8").split("\n"):
        stripped = line.strip()
        if not stripped.startswith("-"):
            continue
        for value in set(NODE_RE.findall(stripped)):
            node = f"M{value}"
            if node not in graph:
                graph.add_node(node, status="documented", decision_rows=0, progress_rows=0)
            graph.nodes[node]["progress_rows"] += 1

    for node, data in graph.nodes(data=True):
        blob = " ".join(closing_text.get(node, []))
        if blob:
            code, phrase, revivable = _classify_premise(blob)
        else:
            code, phrase, revivable = "NO_SUBJECT_ROW", None, False
        data["closure_premise"] = code
        data["premise_match"] = phrase
        data["revivable"] = bool(revivable and data["status"] == "rejected")
    return graph


def persist_graph(graph: nx.DiGraph) -> str:
    """Write the graph in NetworkX node-link format with a deterministic ordering."""
    payload = nx.node_link_data(graph, edges="links")
    payload["nodes"] = sorted(payload["nodes"], key=lambda n: int(str(n["id"])[1:]))
    payload["links"] = sorted(payload["links"], key=lambda e: (str(e["source"]), str(e["target"])))
    payload["sources_sha256"] = {
        "task_plan.md": sha256_file(TASK_PLAN),
        "progress.md": sha256_file(PROGRESS),
    }
    payload["candidate_lanes"] = list(CANDIDATE_LANES)
    GRAPH_OUT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    return sha256_file(GRAPH_OUT)


def open_study() -> optuna.Study:
    """Persistent ask-and-tell study. Sampler is seeded so ordering is reproducible."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    return optuna.create_study(
        study_name=STUDY_NAME,
        storage=f"sqlite:///{STUDY_DB}",
        direction="maximize",  # official Total on strict chronology-safe OOF
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=20260804),
    )


def seed_queue(study: optuna.Study) -> list[str]:
    """Enqueue the frozen candidate lanes once. Each still needs its own predeclaration."""
    # One fixed categorical space over every known lane. Optuna forbids a per-trial
    # value space, and a fixed space is also the honest encoding: the loop chooses
    # among a declared set rather than inventing options as it goes.
    space = {
        "lane": optuna.distributions.CategoricalDistribution(
            [lane["lane"] for lane in CANDIDATE_LANES]
        )
    }
    already = {t.user_attrs.get("lane") for t in study.get_trials(deepcopy=False)}
    queued: list[str] = []
    for lane in CANDIDATE_LANES:
        if lane["lane"] in already:
            continue
        study.enqueue_trial({"lane": lane["lane"]})
        trial = study.ask(space)
        for key, value in lane.items():
            trial.set_user_attr(key, value)
        trial.set_user_attr("state", "PREDECLARATION_REQUIRED")
        queued.append(lane["lane"])
    return queued


def main() -> None:
    graph = build_graph()
    digest = persist_graph(graph)

    study = open_study()
    queued = seed_queue(study)

    revived = sorted(
        (n for n, d in graph.nodes(data=True) if d["revivable"]),
        key=lambda n: int(n[1:]),
    )
    statuses: dict[str, int] = {}
    for _, data in graph.nodes(data=True):
        statuses[str(data["status"])] = statuses.get(str(data["status"]), 0) + 1

    print(f"graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    print(f"  is_dag={nx.is_directed_acyclic_graph(graph)}  sha256={digest[:16]}...")
    print(f"  status={dict(sorted(statuses.items(), key=lambda kv: -kv[1]))}")
    print(f"  classifier-revived={revived}")
    parent_edges = [(u, v) for u, v, d in graph.edges(data=True) if d["kind"] == "parent"]
    print(f"  parent edges={len(parent_edges)} (e.g. {parent_edges[:4]})")
    total = len(study.get_trials(deepcopy=False))
    print(f"study: {STUDY_NAME} trials={total} newly_queued={queued}")


if __name__ == "__main__":
    main()
