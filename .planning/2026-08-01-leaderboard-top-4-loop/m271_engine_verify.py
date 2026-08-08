"""M271 P2 엔진 검증 — 승인된 계획 §7 항목을 실행 가능한 형태로.

말이 아니라 실행으로 판정한다. 특히 다음 넷은 실패하면 엔진이 계획대로 동작하지 않는
것이므로 게이트다.

  3  조건 분기 실증 — 같은 "게이트 실패" 라도 서명이 다르면 다른 리서치로 가는가
  11 폐기 노드 확장 금지 — `PRUNED` 에서 `expand()` 가 **예외로 거부**되는가
  13 프론티어 비지 않음 — 전 노드를 폐기해도 결손이 남으면 프론티어가 복구되는가
  9  2단계 리서치 분리 — `RESEARCH_SOTA` 가 `DIRECTION` 없이 단독 발화하면 거부되는가
  17 후보 공간 개방 — 런타임에 4 개를 넘겨 자라는가 (직전 세션 실패 모드의 역)

읽기 전용. 모델을 적합하지 않고 2024 행을 읽지 않는다.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import m271_deficit as deficit_mod
import m271_excavation_graph as xg
from langgraph.checkpoint.memory import InMemorySaver
from m271_app import build_app, fresh_state
from m271_router import ROUTER_VERSION, Evidence, decide, table
from m271_state import (
    ReducerConflict,
    append_sorted,
    owned_cells,
    write_once,
)

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
REPORT_MD = REPORTS / "m271_engine_verify.md"
RECEIPT = REPORTS / "m271_engine_verify_receipt.json"

RESULTS: dict[str, dict[str, Any]] = {}


def record(name: str, passed: bool, detail: Any) -> None:
    RESULTS[name] = {"passed": bool(passed), "detail": detail}
    print(f"[{'PASS' if passed else 'FAIL'}] {name}")


def _evidence(**kwargs: Any) -> Evidence:
    base = {"evidence_id": "e", "node_id": "n", "lane": "L2"}
    return Evidence(**{**base, **kwargs})


# ---------------------------------------------------------------- 1 라우터 결정성
def check_router_determinism() -> None:
    ev = _evidence(gate={"G1": False, "G2": True, "G3": True, "G4": True}, confirms=True)
    ctx = {"residual_mass": 0.2, "cell_status": {}, "lane_live_counts": {}, "guards": {}}
    first = decide(ev, ctx)
    second = decide(ev, ctx)
    record(
        "1_router_deterministic",
        (first.condition, first.action, first.targets)
        == (second.condition, second.action, second.targets),
        {
            "condition": first.condition,
            "considered": list(first.considered),
            "reason": first.reason,
        },
    )


# ---------------------------------------------------------------- 3 조건 분기 실증
def check_gate_signature_branching() -> None:
    ctx = {"residual_mass": 0.0, "cell_status": {}, "lane_live_counts": {}, "guards": {}}
    inconsistent = decide(
        _evidence(gate={"G1": False, "G2": True, "G3": True, "G4": True}), ctx
    )
    weak = decide(_evidence(gate={"G1": True, "G2": True, "G3": False, "G4": True}), ctx)
    kinds = (
        inconsistent.targets[0].get("kind") if inconsistent.targets else None,
        weak.targets[0].get("kind") if weak.targets else None,
    )
    ok = (
        inconsistent.condition == "C3"
        and weak.condition == "C4"
        and kinds[0] != kinds[1]
    )
    record(
        "3_gate_signature_branches",
        ok,
        {
            "G1fail_G3pass": {"condition": inconsistent.condition, "kind": kinds[0]},
            "G3fail_G1pass": {"condition": weak.condition, "kind": kinds[1]},
            "why": "같은 게이트 실패라도 서명이 다르면 다른 리서치로 가야 한다",
        },
    )


# ---------------------------------------------------------------- 4 동적 fan-out
def check_dynamic_fanout() -> None:
    counts = {}
    for lanes in ([("L2", 0.05)], [("L2", 0.05), ("L3", 0.05), ("L7", 0.05)]):
        ctx = {
            "lane_live_counts": {k: 0 for k, _ in lanes},
            "lane_deficit_mass": dict(lanes),
            "residual_mass": 0.0,
            "cell_status": {},
            "guards": {},
        }
        decision = decide(_evidence(), ctx)
        counts[len(lanes)] = (decision.condition, len(decision.targets))
    ok = counts[1] == ("C6", 1) and counts[3] == ("C6", 3)
    record("4_dynamic_fanout_follows_state", ok, counts)


# ---------------------------------------------------------------- 9 2단계 리서치 분리
def check_two_stage_research() -> None:
    from m271_app import op_research_sota

    try:
        op_research_sota({"pending_evidence": {"lane": "L2"}})
        refused = False
    except RuntimeError:
        refused = True
    allowed = op_research_sota(
        {"pending_evidence": {"direction_id": "dir::x", "lane": "L2", "origin_node": "n"}}
    )
    record(
        "9_two_stage_research_separation",
        refused and bool(allowed.get("methods")),
        {"sota_without_direction_refused": refused, "with_direction_ok": bool(allowed)},
    )


# ---------------------------------------------------------------- 11 폐기 노드 확장 금지
def check_prune_absorbing() -> None:
    graph = xg.ExcavationGraph()
    graph.register_premise(
        xg.Premise("TEST_PREMISE", "테스트", lambda s: not s.get("flip", False))
    )
    graph.add_node("root", node_type="MEASURE", lane="L2", status=xg.LIVE)
    graph.add_node("child", node_type="EXPERIMENT", lane="L2", parents=["root"], status=xg.LIVE)
    graph.add_node("grand", node_type="EXPERIMENT", lane="L2", parents=["child"], status=xg.LIVE)
    # 다른 LIVE 조상에서 별개 경로로 닿는 후손
    graph.add_node("alt", node_type="MEASURE", lane="L3", status=xg.LIVE)
    graph.graph.add_edge("alt", "grand", kind="depends_on")

    pruned = graph.prune("child", "TEST_PREMISE")

    refused = False
    try:
        graph.expand("child", "newborn", node_type="EXPERIMENT", lane="L2")
    except xg.PruneViolation:
        refused = True

    grand_alive = graph.graph.nodes["grand"]["status"] == xg.LIVE
    record(
        "11_prune_absorbing_and_multiparent",
        refused and "child" in pruned and grand_alive,
        {
            "pruned": pruned,
            "expand_refused": refused,
            "grandchild_survived_via_other_live_ancestor": grand_alive,
        },
    )


# ---------------------------------------------------------------- 13 프론티어 비지 않음
def check_frontier_refills() -> None:
    ledger = deficit_mod.DeficitLedger(
        total=0.6286,
        target=0.66,
        cells={
            "c1": {
                "key": "c1", "axes": {}, "rows": 10, "loss_share": 0.3714,
                "ficr_loss": 0.3, "nmae_loss": 0.0714, "gen_weight": 1.0,
                "mean_unit": 2.0, "mechanism": None, "recoverable_estimate": None,
                "status": deficit_mod.UNEXPLAINED, "owner": None,
            }
        },
        axes=["group_id"],
    )
    ledger.assert_identity()

    graph = xg.ExcavationGraph()
    graph.register_premise(xg.Premise("P", "테스트", lambda s: True))
    graph.add_node("only", node_type="EXPERIMENT", lane="L2", status=xg.LIVE)
    graph.prune("only", "P")
    empty = graph.frontier()

    # 결손이 남아 있으므로 C1 이 발화해 새 노드가 태어나야 한다.
    ctx = {
        "cell_status": {"c1": ledger.cells["c1"]["status"]},
        "lane_live_counts": {"L2": 0},
        "lane_deficit_mass": {"L2": ledger.unexplained_mass()},
        "residual_mass": ledger.residual_mass(),
        "guards": {},
    }
    decision = decide(_evidence(deficit_cell="c1"), ctx)
    refilled = decision.action in {"RESEARCH_DIRECTION", "SEND_FANOUT"}
    record(
        "13_frontier_refills_from_deficit",
        empty == [] and refilled and ledger.unexplained_mass() > 0,
        {
            "frontier_after_pruning_all": empty,
            "unexplained_mass": ledger.unexplained_mass(),
            "condition": decision.condition,
            "action": decision.action,
        },
    )


# ---------------------------------------------------------------- 리듀서 순서 무관
def check_reducer_order_insensitivity() -> None:
    a = [{"event_id": "a", "ts": "2", "node_id": "x", "seq": 1}]
    b = [{"event_id": "b", "ts": "1", "node_id": "y", "seq": 1}]
    forward = append_sorted(append_sorted([], a), b)
    backward = append_sorted(append_sorted([], b), a)

    conflict = False
    try:
        write_once({"k": 1}, {"k": 2})
    except ReducerConflict:
        conflict = True

    ownership = False
    try:
        owned_cells({"c": {"owner": "n1"}}, {"c": {"owner": "n2"}})
    except ReducerConflict:
        ownership = True

    record(
        "2_reducers_order_insensitive_and_guarded",
        forward == backward and conflict and ownership,
        {
            "append_order_insensitive": forward == backward,
            "write_once_conflict_raised": conflict,
            "cell_ownership_enforced": ownership,
        },
    )


# ---------------------------------------------------------------- 그래프 재구성 결정성
def check_graph_digest_stable() -> None:
    def build() -> xg.ExcavationGraph:
        g = xg.ExcavationGraph()
        g.add_node("a", node_type="MEASURE", lane="L1", status=xg.LIVE)
        g.add_node("b", node_type="EXPERIMENT", lane="L2", parents=["a"], status=xg.LIVE)
        g.link("a", "b", "instantiates")
        return g

    record(
        "5_graph_digest_reproducible",
        build().digest() == build().digest(),
        {"digest": build().digest()[:16]},
    )


# ---------------------------------------------------------------- 부활
def check_revival() -> None:
    graph = xg.ExcavationGraph()
    graph.register_premise(
        xg.Premise(
            "NO_EXTERNAL_DATA",
            "외부 공개데이터를 쓸 수 없다",
            lambda s: not s.get("external_data_allowed", False),
        )
    )
    graph.add_node("ext", node_type="EXPERIMENT", lane="L2", status=xg.LIVE)
    graph.prune("ext", "NO_EXTERNAL_DATA")
    before = graph.graph.nodes["ext"]["status"]

    flipped = graph.flipped_premises({"external_data_allowed": True})
    revived = graph.revive("NO_EXTERNAL_DATA") if flipped else []
    record(
        "7_premise_flip_revives_subgraph",
        before == xg.PRUNED and flipped == ["NO_EXTERNAL_DATA"] and revived == ["ext"],
        {"before": before, "flipped": flipped, "revived": revived,
         "after": graph.graph.nodes["ext"]["status"]},
    )


# ---------------------------------------------------------------- LangGraph 실행
def check_app_runs_and_checkpoints() -> None:
    app = build_app(InMemorySaver())
    cfg = {"configurable": {"thread_id": "m271-verify"}}
    state = fresh_state(
        pending_evidence={
            "node_id": "n1",
            "lane": "L2",
            "gate": {"G1": False, "G2": True, "G3": True, "G4": True},
            "router_context": {
                "cell_status": {},
                "lane_live_counts": {},
                "lane_deficit_mass": {},
                "residual_mass": 0.0,
            },
        }
    )
    out = app.invoke(state, cfg)
    history = list(app.get_state_history(cfg))

    # 단언은 "무언가 일어났다" 가 아니라 "**라우터가 고른 그것**이 일어났다" 를 봐야 한다.
    # 느슨한 단언이 실제 배선 버그(Send 페이로드 미수신)를 한 번 통과시켰다.
    expected = decide(
        _evidence(
            node_id="n1", lane="L2", gate={"G1": False, "G2": True, "G3": True, "G4": True}
        ),
        {"cell_status": {}, "lane_live_counts": {}, "lane_deficit_mass": {},
         "residual_mass": 0.0, "guards": {}},
    )
    want_kind = expected.targets[0]["kind"]
    want_lane = expected.targets[0]["lane"]
    landed = out.get("sources", {})
    matched = any(
        v.get("kind") == want_kind and v.get("lane") == want_lane for v in landed.values()
    )
    record(
        "6_langgraph_app_routes_the_decided_target",
        bool(out.get("events")) and len(history) > 1 and matched,
        {
            "router_condition": expected.condition,
            "expected_kind": want_kind,
            "expected_lane": want_lane,
            "landed_sources": {
                k: {"kind": v.get("kind"), "lane": v.get("lane")} for k, v in landed.items()
            },
            "matched": matched,
            "checkpoints": len(history),
            "macrostep": out.get("guards", {}).get("macrostep"),
        },
    )


# ---------------------------------------------------------------- 원장 생성성
def check_ledger_generativity() -> None:
    ledger = deficit_mod.DeficitLedger.from_a7()
    before = len(ledger.cells)
    mass_before = ledger.loss_sum()
    target = ledger.top(1)[0]["key"]
    created = ledger.refine_axis(target, "wind_sector", {"240": 0.6, "other": 0.4})
    after = len(ledger.cells)
    ledger.assert_identity()
    record(
        "8_ledger_is_generative",
        after > before and abs(ledger.loss_sum() - mass_before) < 1e-12,
        {
            "cells_before": before,
            "cells_after": after,
            "created": created,
            "mass_preserved": abs(ledger.loss_sum() - mass_before) < 1e-12,
            "unexplained_after": len(ledger.unexplained()),
        },
    )


# ---------------------------------------------------------------- P3 실행기
def check_gate_signature_normalisation() -> None:
    """`m270_gate.py` 의 긴 조건 문자열을 C3/C4 가 읽을 평면 서명으로 바꾸는가."""
    import m271_evidence as ev_mod
    from m270_gate import GateResult

    result = GateResult(
        passed=False,
        conditions={
            "G1 sign-test p <= 0.10 (p=0.5000, 5/9)": False,
            "G2 median delta > 0 (median=+0.000057)": True,
            "G3 bootstrap q05 > 0 (q05=-0.001224)": False,
            "G4 worst month >= -0.010 (worst=-0.006998)": True,
        },
        evidence={
            "months_scored": 9,
            "positive_fraction": 5 / 9,
            "median_total_delta": 0.000057,
            "block_bootstrap_positive_fraction": 0.42,
        },
    )
    flat = ev_mod.normalize_gate(result)

    broken = False
    try:
        ev_mod.normalize_gate(GateResult(passed=False, conditions={"no token here": True}))
    except ev_mod.GateSignatureError:
        broken = True

    evidence = ev_mod.build_evidence(
        evidence_id="e1", node_id="M251", lane="L7", gate_result=result, predeclared_sign=1
    )
    record(
        "14_gate_signature_normalised",
        flat == {"G1": False, "G2": True, "G3": False, "G4": True}
        and broken
        and evidence.gate is not None
        and evidence.confirms is False,
        {
            "flat": flat,
            "malformed_refused": broken,
            "sign": evidence.sign,
            "information": round(evidence.information, 4),
            "confirms": evidence.confirms,
            "case": "M251 의 실제 게이트 실패 패턴(G1·G3 실패)을 그대로 넣었다",
        },
    )


def check_research_guards() -> None:
    """레인 범위와 2단계 순서를 실행기가 강제하는가."""
    import m271_research as rs

    target = {"lane": "L2", "kind": "explain", "origin_node": "n1", "scope": "s"}
    pre = rs.precommit(target, "RESEARCH_DIRECTION")

    lane_violation = False
    try:
        rs.ingest_sources(
            pre,
            [
                rs.Source(
                    source_id="s1", lane="L7", title="t", origin="o",
                    source_class="peer_reviewed", applicability="near_match_only", claim="c",
                )
            ],
        )
    except rs.LaneScopeViolation:
        lane_violation = True

    stage_violation = False
    try:
        rs.research_sota(
            None, target, [], method_name="m", settings={}, required_inputs=["x"],
            within_official_rules=True, reported_performance="", our_conditions_differ="",
            expected_effect_range="",
        )
    except rs.StageOrderViolation:
        stage_violation = True

    record(
        "15_research_guards_enforced",
        lane_violation and stage_violation and bool(pre.digest()),
        {
            "out_of_lane_source_refused": lane_violation,
            "sota_without_direction_refused": stage_violation,
            "precommitment_digest": pre.digest()[:16],
        },
    )


def check_materialize_lineage() -> None:
    """새 노드가 계보 엣지 3 종과 함께 태어나는가. 실행 불가 사양은 거부되는가."""
    import m271_materialize as mat
    import m271_research as rs

    graph = xg.ExcavationGraph()
    graph.register_premise(xg.Premise("P", "테스트", lambda s: True))
    graph.add_node("parent", node_type="MEASURE", lane="L2", status=xg.LIVE)

    spec = rs.Specification(
        spec_id="spec::d1", direction_id="d1", lane="L2", method_name="diurnal_bias",
        settings={"axis": "hour"}, required_inputs=["ldaps_10u", "ldaps_10v"],
        within_official_rules=True, reported_performance="n/a",
        our_conditions_differ="산악 능선, 허브 117m", expected_effect_range="0.000~0.005",
        source_ids=["src1"],
    )
    node_id, pre = mat.materialize(
        graph, spec, node_id="EXP1", parent_node="parent", deficit_cell="cell::x",
        direction_id="d1", source_ids=["src1"], expected_sign=1, expected_effect=0.003,
        gate_version="M270_MONTHLY_GATE_v1_frozen_2026-08-04",
        stop_condition="1회 실행", parent_candidate="M107",
    )
    lineage = mat.lineage_of(graph, node_id)

    refused = False
    bad = rs.Specification(
        spec_id="spec::bad", direction_id="d1", lane="L2", method_name="x",
        settings={}, required_inputs=["era5_reanalysis"], within_official_rules=False,
        reported_performance="", our_conditions_differ="", expected_effect_range="",
    )
    try:
        mat.materialize(
            graph, bad, node_id="EXP2", parent_node="parent", deficit_cell=None,
            direction_id="d1", source_ids=[], expected_sign=1, expected_effect=0.0,
            gate_version="v", stop_condition="", parent_candidate="M107",
        )
    except mat.UnmaterialisableSpec:
        refused = True

    all_three = all(lineage[k] for k in mat.LINEAGE_EDGES)
    record(
        "16_materialize_creates_lineage",
        all_three and refused and graph.graph.nodes[node_id]["status"] == xg.PREDECLARED,
        {
            "lineage": lineage,
            "all_three_edge_kinds": all_three,
            "out_of_rule_spec_refused": refused,
            "predeclaration_digest": pre.digest()[:16],
            "status": graph.graph.nodes[node_id]["status"],
        },
    )


def check_candidate_space_grows() -> None:
    """후보 공간이 런타임에 자라는가. 직전 세션의 고정 4-원소와의 결정적 차이다."""
    import tempfile

    import m271_experiment as exp

    with tempfile.TemporaryDirectory() as tmp:
        study = exp.open_study(Path(tmp) / "verify.db")
        sizes = []
        for i in range(7):  # 4 개를 넘겨 등록한다
            exp.enqueue_hypothesis(
                study,
                exp.Candidate(
                    node_id=f"H{i}", lane="L2", deficit_cell=None,
                    expected_gain=0.001 * i, expected_hours=1.0, exploration=0.1,
                ),
            )
            sizes.append(len(exp.queue_summary(study)["nodes"]))

        chosen = exp.select_next(
            [
                exp.Candidate("A", "L2", None, expected_gain=0.002, expected_hours=1.0),
                exp.Candidate("B", "L3", None, expected_gain=0.001, expected_hours=1.0,
                              voi=0.5),
            ]
        )
        budget_ok = False
        try:
            exp.check_worker_budget(concurrent_nodes=5, threads_per_node=6)
        except exp.WorkerBudgetExceeded:
            budget_ok = True

    record(
        "17_candidate_space_is_open",
        sizes[-1] == 7 and chosen is not None and chosen.node_id == "B" and budget_ok,
        {
            "queue_growth": sizes,
            "exceeded_m270_fixed_four": sizes[-1] > 4,
            "c11_picked": chosen.node_id if chosen else None,
            "c11_reason": "정보가치가 높은 B 가 기대이득만 큰 A 를 이긴다",
            "worker_budget_enforced": budget_ok,
        },
    )


def write_artifacts(all_passed: bool) -> None:
    lines = [
        "# M271 P2 — 엔진 검증",
        "",
        f"- 판정일: {datetime.now(UTC).strftime('%Y-%m-%d')} (UTC)",
        f"- 라우터 버전: `{ROUTER_VERSION}`",
        f"- 판정: **{'PASS' if all_passed else 'FAIL'}**",
        "",
        "승인된 계획 §7 항목을 실행으로 판정한다. 말로 통과시키지 않는다.",
        "",
        "| 항목 | 판정 |",
        "|---|---|",
    ]
    for name, res in RESULTS.items():
        lines.append(f"| `{name}` | {'PASS' if res['passed'] else '**FAIL**'} |")

    branch = RESULTS["3_gate_signature_branches"]["detail"]
    prune = RESULTS["11_prune_absorbing_and_multiparent"]["detail"]
    frontier = RESULTS["13_frontier_refills_from_deficit"]["detail"]
    ledger = RESULTS["8_ledger_is_generative"]["detail"]

    lines += [
        "",
        "## 핵심 판정",
        "",
        "### 조건 분기 (계획의 요점)",
        "",
        "같은 게이트 실패라도 서명이 다르면 다른 리서치로 가야 한다. 일괄 리서치는 이 구분을",
        "원리적으로 할 수 없다.",
        "",
        "| 증거 서명 | 발화 조건 | 리서치 종류 |",
        "|---|---|---|",
        f"| G1 실패 & G3 통과 | `{branch['G1fail_G3pass']['condition']}` | "
        f"`{branch['G1fail_G3pass']['kind']}` |",
        f"| G3 실패 & G1 통과 | `{branch['G3fail_G1pass']['condition']}` | "
        f"`{branch['G3fail_G1pass']['kind']}` |",
        "",
        "### 폐기의 흡수성",
        "",
        f"- 폐기된 노드: `{prune['pruned']}`",
        f"- `expand()` 예외 거부: **{prune['expand_refused']}**",
        f"- 다른 LIVE 조상으로 살아남은 후손: **"
        f"{prune['grandchild_survived_via_other_live_ancestor']}**",
        "",
        "우선순위를 낮추는 게 아니라 금지다. 다만 폐기 사유가 기계검증 술어로 남으므로",
        "전제가 뒤집히면 C9 이 되살린다.",
        "",
        "### 프론티어가 비지 않음 (직전 세션 실패 모드의 역)",
        "",
        f"- 전 노드 폐기 후 프론티어: `{frontier['frontier_after_pruning_all']}`",
        f"- 남은 미설명 결손 질량: `{frontier['unexplained_mass']:.6f}`",
        f"- 발화한 조건: `{frontier['condition']}` -> `{frontier['action']}`",
        "",
        "직전 세션은 하드코딩 4-튜플이 소진되면 멈췄다. 여기서는 결손 질량이 남는 한 조건이",
        "발화해 새 노드를 낳는다.",
        "",
        "### 원장의 생성성",
        "",
        f"- 셀 {ledger['cells_before']} -> {ledger['cells_after']} (축 추가)",
        f"- 손실 질량 보존: **{ledger['mass_preserved']}**",
        f"- 미설명 셀: {ledger['unexplained_after']}",
        "",
        "축을 추가하면 새 셀이 생기고 전부 `UNEXPLAINED` 로 태어나므로 C1 발화 대상이 늘어난다.",
        "손실 질량은 보존되므로 A7 이 확인한 가법 항등식이 유지된다.",
        "",
        "## 동결 라우터 표",
        "",
        "| 코드 | 우선순위 | 동작 | 리서치 종류 | 조건 |",
        "|---|---:|---|---|---|",
    ]
    for row in table():
        kind = f"`{row['research_kind']}`" if row["research_kind"] else "-"
        lines.append(
            f"| `{row['code']}` | {row['priority']} | `{row['action']}` | {kind} | "
            f"{row['describe']} |"
        )

    lines += [
        "",
        "## 남는 한계",
        "",
        "1. 임계값 `TAU_DEFICIT_MASS`·`EPSILON_INFORMATION`·`STALL_LIMIT` 은 **선언 관례**이지",
        "   보정된 값이 아니다. 이 문제에서 held-out 으로 적합한 바 없다(계획 R3).",
        "2. C11 의 가치 함수도 선언 관례다. 기대이득·정보가치·탐험항의 상대 비중에 근거가 없다.",
        "3. 병렬 리듀서 적용 순서의 결정성은 P0 이 측정한 행동이지 API 보증이 아니다. 그래서",
        "   리듀서를 순서 무관하게 쓰고 셀 쓰기 소유권을 강제한다.",
        "4. 이 검증은 엔진이 **계획대로 배선되었는지**만 판정한다. 라우팅이 좋은 리서치를",
        "   고르는지는 실제 사이클을 돌려야 알 수 있다.",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    receipt = {
        "schema_version": 1,
        "stage": "M271_P2_ENGINE_VERIFY",
        "verdict": "PASS" if all_passed else "FAIL",
        "decided_utc": datetime.now(UTC).isoformat(),
        "router_version": ROUTER_VERSION,
        "router_table": table(),
        "checks": RESULTS,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "report_sha256": hashlib.sha256(REPORT_MD.read_bytes()).hexdigest(),
        "dacon_upload": False,
        "external_actions": [],
        "model_fits": 0,
        "lockbox_reopened": False,
        "new_2024_evaluation": False,
    }
    RECEIPT.write_text(
        json.dumps(receipt, indent=1, ensure_ascii=False, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    check_router_determinism()
    check_reducer_order_insensitivity()
    check_gate_signature_branching()
    check_dynamic_fanout()
    check_graph_digest_stable()
    check_app_runs_and_checkpoints()
    check_revival()
    check_ledger_generativity()
    check_two_stage_research()
    check_prune_absorbing()
    check_frontier_refills()
    check_gate_signature_normalisation()
    check_research_guards()
    check_materialize_lineage()
    check_candidate_space_grows()

    all_passed = all(r["passed"] for r in RESULTS.values())
    write_artifacts(all_passed)
    print(f"\nverdict={'PASS' if all_passed else 'FAIL'}  ({len(RESULTS)} checks)")
    print(f"report  -> {REPORT_MD}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
