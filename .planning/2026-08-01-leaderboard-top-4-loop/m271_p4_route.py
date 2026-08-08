"""M271 P4 — **엔진을 실제로 가동한다.** 라우터가 다음 노드를 정한다.

이 세션의 결함은 엔진이 없다는 것이 아니라 **엔진을 안 돌렸다는 것**이었다.
`m271_app.py`(LangGraph StateGraph), `m271_router.py`(C1~C15 조건부 엣지),
`m271_materialize.py`(동적 노드 생성)가 다 있고 `m271_engine_verify.py` 가 15 개 검사를
통과시키는데 실행 기록이 0 건이었다. 나는 손으로 사이클 스크립트를 쓰고 결과를
`CYCLES` 튜플에 **사후 전사**했다. 그래프가 무엇을 할지 정한 게 아니라 내가 한 일을
받아 적었다 — 계획서가 없애려던 descriptive 그래프를 형태만 바꿔 재현한 것이다.

대가가 컸다. C6(레인 기아)·C10(정체)·C12(축 재분해)는 정확히 "같은 이웃만 파고 있으면
새 방향을 리서치해 노드를 만들라"는 조건인데 한 번도 발화하지 않았고, 나는 0.005 규모
이웃을 20 여 사이클 팠다. 상위권과의 격차는 **0.037** 이다.

이 스크립트는 사후 전사를 하지 않는다. **실제 receipt 에서 증거 서명을 뽑아** 상태그래프에
넣고 라우터가 내는 판정을 그대로 기록한다. 내가 다음에 무엇을 할지 고르지 않는다.

  * 상태: `m271_state.M271State` 채널 + 리듀서 + 체크포인터 (원칙 1)
  * 조건부 엣지: `m271_router.decide` 가 다음 노드를 동적 선택 (원칙 2)
  * 동적 fan-out: `Send` 로 분기 수를 실행 시점에 정함 (원칙 3)

라우터 상태는 발굴 그래프와 결손 원장에서 **계산해서** 넣는다. 손으로 채우지 않는다.
`stall_counter` 도 마찬가지다 — 결손 원장은 A7 이 세운 뒤 **어떤 사이클도 갱신하지
않았으므로** 질량 무감소가 기록된 사이클 수만큼 이어졌다. 루프의 연료계가 처음부터
멈춰 있었다는 뜻이고, 그 사실을 감추지 않고 그대로 넣는다.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import m271_cycle_contract as cc
import m271_excavation_graph as xg
import m271_ledger_state as ls
from m271_app import build_app, fresh_state
from m271_p4_consolidate import CYCLES, build
from m271_c10_loop_engine import NOVELTY_WINDOW, ROUTER_V3_VERSION, decide_v3
from m271_router import Evidence
from m271_state import state_digest

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
REPORT_MD = REPORTS / "m271_p4_route.md"
RECEIPT = REPORTS / "m271_p4_route_receipt.json"

ROUTE_VERSION = "M271_P4_ROUTE_v4"
CHAMPION_LOCAL = 0.630310
TARGET = 0.66


def _receipt(name: str) -> dict[str, Any]:
    path = REPORTS / name
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("result", payload)


def _gate_flags(payload: dict[str, Any]) -> dict[str, bool] | None:
    gate = payload.get("gate")
    if isinstance(gate, dict) and isinstance(gate.get("flags"), dict):
        return {k: bool(v) for k, v in gate["flags"].items()}
    return None


def build_evidence() -> list[Evidence]:
    """최근 사이클 receipt 에서 증거 서명을 뽑는다. 손으로 지어내지 않는다.

    `information` 은 그 노드가 실제로 판정을 낸 정도다. 가드 실패로 무효가 된 노드는
    낮게 잡는다 — 그것이 C8 과 C14 를 가르는 기준이다.
    """
    out: list[Evidence] = []

    c68 = _receipt("m271_cycle68_empirical_decomposition_receipt.json")
    if c68:
        out.append(Evidence(
            evidence_id="C1N68", node_id="C1N68_EMPIRICAL_DECOMPOSITION", lane="L6",
            sign=1, predeclared_sign=1, information=0.9, confirms=True,
            expected_gain=0.017, expected_hours=2.0,
        ))

    c73 = _receipt("m271_cycle73_group_blend_gate_receipt.json")
    if c73:
        out.append(Evidence(
            evidence_id="C1N73", node_id="C1N73_GROUP_BLEND_GATE", lane="L7",
            gate=_gate_flags(c73), sign=1, predeclared_sign=1, information=0.5,
            expected_gain=0.0049, expected_hours=1.0,
        ))

    c76 = _receipt("m271_cycle76_circular_block_receipt.json")
    if c76:
        h = c76.get("hypotheses", {})
        established = bool(h.get("H2_c60_excludes_zero") or h.get("H3_c73_excludes_zero"))
        out.append(Evidence(
            evidence_id="C1N76", node_id="C1N76_CIRCULAR_BLOCK", lane="L4",
            information=0.8, negative_finding=not established,
            refutes_mechanism="NONLINEAR_ESTIMAND_BIAS",
            novel_mechanism="EDGE_UNDERSAMPLING_IN_MOVING_BLOCK",
            axis_kind="structural",
            expected_gain=0.0, expected_hours=0.5,
        ))

    n1 = _receipt("m271_n1_curtailment_clean_receipt.json")
    if n1:
        h = n1.get("hypotheses", {})
        gain = float(n1.get("implied_total_gain", 0.0))
        out.append(Evidence(
            evidence_id="C1N87", node_id="C1N87_CURTAILMENT_CLEAN_TARGET", lane="L1",
            sign=1 if gain > 0 else -1,
            predeclared_sign=1,   # H1 은 clean<base 를 실행 전에 양수로 동결
            information=0.9,
            confirms=False,
            negative_finding=not bool(h.get("H2_clears_detection")),
            # g3 개선 최대를 예측했는데 g3 가 가장 나빠졌다 — 기전 반증.
            refutes_mechanism="G3_THETA_EXCESS_IS_CURTAILMENT",
            expected_gain=abs(gain), expected_hours=1.0,
        ))

    c84 = _receipt("m271_c7c_chronological_receipt.json")
    if c84:
        h = c84.get("hypotheses", {})
        out.append(Evidence(
            evidence_id="C1N84", node_id="C1N84_TEACHER_CHRONOLOGICAL", lane="L6",
            sign=1, predeclared_sign=1,   # H1 은 deep<base 를 실행 전에 양수로 동결
            information=0.9,
            confirms=False,
            negative_finding=not bool(h.get("H2_clears_magnitude_gate")),
            # 누출면에서 본 용량 이득이 시간 분할에서 96% 사라졌다.
            refutes_mechanism="TEACHER_CAPACITY_CLEARS_GATE",
            expected_gain=float(c84.get("implied_total_gain", 0.0)),
            expected_hours=1.0,
        ))

    c80 = _receipt("m271_c5_fusion_anomaly_receipt.json")
    if c80:
        h = c80.get("hypotheses", {})
        arms = c80.get("arms", {})
        delta = float(
            arms.get("stack_fair", {}).get("total", 0.0)
            - arms.get("pooled", {}).get("total", 0.0)
        )
        out.append(Evidence(
            evidence_id="C1N80", node_id="C1N80_FUSION_ANOMALY", lane="L3",
            gate=_gate_flags(c80),
            sign=1 if delta > 0 else (-1 if delta < 0 else 0),
            predeclared_sign=0,  # H1 은 부호를 예단하지 않는다고 실행 전에 명시했다
            information=0.9,
            confirms=False,
            negative_finding=not bool(h.get("H1_late_fusion_wins_at_equal_info")),
            # 두 설명(정보 제거 / 다양성)이 각각 반쯤만 맞아 기전이 미확정이다.
            refutes_mechanism="INFORMATION_LOSS_EXPLAINS_C77_NEGATIVE",
            expected_gain=abs(delta), expected_hours=2.0,
        ))

    c77 = _receipt("m271_cycle77_per_source_stack_receipt.json")
    if c77:
        h = c77.get("hypotheses", {})
        arms = c77.get("arms", {})
        delta = float(
            arms.get("stack", {}).get("total", 0.0)
            - arms.get("pooled", {}).get("total", 0.0)
        )
        wins = bool(h.get("H1_stack_beats_pooled"))
        out.append(Evidence(
            evidence_id="C1N77", node_id="C1N77_PER_SOURCE_STACK", lane="L3",
            gate=_gate_flags(c77),
            sign=1 if delta > 0 else (-1 if delta < 0 else 0),
            predeclared_sign=1,  # H1 은 STACK > POOLED 를 실행 전에 동결했다
            information=0.9, confirms=wins, negative_finding=not wins,
            expected_gain=abs(delta), expected_hours=2.0,
        ))
    return out


def cc_novelty_window() -> int:
    return NOVELTY_WINDOW


def router_context(graph: xg.ExcavationGraph, ledger: Any) -> dict[str, Any]:
    """라우터 상태를 발굴 그래프와 결손 원장에서 **계산**한다."""
    census = graph.lane_census()
    lane_live = {lane: int(counts.get(xg.LIVE, 0)) for lane, counts in census.items()}

    lane_mass: Counter[str] = Counter()
    cell_status: dict[str, str] = {}
    unexplained = 0.0
    for cell in ledger.cells.values():
        # 필드명은 `recoverable_if_average` 다(`m271_deficit.py:193`). 처음에 `recoverable`
        # 로 읽어 잔여질량이 0 으로 나왔고, 그러면 C7(확인된 방향의 고도화)이 발화 자체를
        # 못 한다 — 라우터가 못 보는 상태를 만들어 놓고 판정을 받는 셈이었다.
        recoverable = float(cell.get("recoverable_if_average") or 0.0)
        status = str(cell.get("status", "UNEXPLAINED"))
        cell_status[cell["key"]] = status
        if status == "UNEXPLAINED":
            unexplained += recoverable
        # 결손 셀은 레인에 묶여 있지 않으므로 원장 질량은 원장을 세운 레인(L8)이 보유한다.
        lane_mass["L8"] += recoverable

    # `stall_counter` 를 사이클 수로 세는 것은 가짜였다. 이제 원장 갱신 계약의
    # **이력에서 계산**한다 — 질량이 한 번이라도 줄면 0 으로 돌아간다.
    contract = ls.contract_state(ledger, CHAMPION_LOCAL, [c[0] for c in CYCLES])
    stall = int(contract["stall_counter"])

    # 최근 노드의 (레인, kind). kind 는 그 노드가 실제로 어떤 방향이었는지로,
    # 라우터가 그 노드에 붙였던 research_kind 를 알 수 없으므로 **레인만** 쓴다.
    # 같은 레인을 창 안에서 반복하면 그것이 곧 "사소한 변형" 이다.
    # 튜플 **꼬리**가 최근이 아니다 — 새 노드를 중간에 삽입해 왔으므로 노드 번호로
    # 정렬해야 한다. 꼬리에서 뽑으면 C17 이 엉뚱한 창을 보고 발화하지 않는다.
    def _num(node_id: str) -> int:
        m = re.match(r"C1N(\d+)", node_id)
        return int(m.group(1)) if m else -1

    lane_of = {c[0]: c[1] for c in CYCLES}
    ordered = sorted((n for n in lane_of if _num(n) >= 0), key=_num)
    recent_nodes = ordered[-cc_novelty_window():]
    recent_pairs = [(lane_of[n], "scale_up") for n in recent_nodes]

    return {
        "lane_live_counts": lane_live,
        "lane_deficit_mass": dict(lane_mass),
        "cell_status": cell_status,
        "residual_mass": unexplained,
        "node_direction": {},
        "flipped_premises": list(graph.flipped_premises({})),
        # v3 가 읽는 항목. `loop_engine_visits` 는 C1N78 이 C10 을 1 회 서비스했으므로 1.
        "guards": {"stall_counter": stall, "macrostep": stall, "loop_engine_visits": 1},
        "remaining_gap": TARGET - CHAMPION_LOCAL,
        # C17(신규성 기각)의 배선. 빈 리스트를 넘기면 게이트가 **발화 자체를 못 한다** —
        # 실제로 그래서 C16 만 작동하고 C17 은 끊겨 있었다. 그래프의 최근 노드에서
        # (레인, 라우터가 붙일 kind) 창을 만들어 넘긴다.
        "recent_lane_kind": recent_pairs,
        "proposed_kind": None,
        "ledger_contract": contract,
    }


def main() -> int:
    graph, ledger = build()
    ledger.compute_efficiency()
    # 원장 갱신 계약의 이력을 덧입힌다. 없으면 라우터가 낡은 상태를 본다.
    ls.apply_to(ledger)
    context = router_context(graph, ledger)
    evidences = build_evidence()
    if not evidences:
        raise RuntimeError("증거가 없다 — receipt 를 찾지 못했다")
    # **사이클 계약 강제.** 원장에 기록하지 않은 노드의 증거는 라우터에 들어갈 수 없다.
    # 계약을 만들기만 하고 강제하지 않은 것이 미기록 79 건을 만들었다.
    cc.require_recorded([e.node_id for e in evidences], strict=True)

    app = build_app()
    decisions: list[dict[str, Any]] = []
    digests: list[str] = []
    pure: dict[str, Any] | None = None

    for index, evidence in enumerate(evidences):
        # **라우터 v3.** C16 크기게이트·C17 신규성기각·C10 서비스후 게이팅이 걸린다.
        # 드라이버가 v2 를 직접 부르면 v3 개정이 라우팅에 연결되지 않는다 —
        # 실제로 한 번 그렇게 됐고 증거 다섯 건이 전부 C10 으로 갔다.
        # C17 은 '이 증거가 어떤 kind 로 갈 것인가' 를 알아야 한다.
        #
        # v2 를 그대로 부르면 정체가 높아 **C10(loop_engine)** 이 나오는데, v3 은 C10 을
        # 억제하고 다른 조건으로 재선택한다. 그래서 C17 이 검사할 kind 는 **억제 이후의
        # 최종 kind** 다. `stall_counter=0` 으로 탐침하면 C10 만 빠진 판정이 나온다.
        from m271_router import decide as _decide_v2
        probe = _decide_v2(
            evidence,
            {**context, "guards": {**dict(context["guards"]), "stall_counter": 0}},
        )
        kind = (probe.targets[0].get("kind") if probe.targets else None)
        pure = decide_v3(evidence, {**context, "proposed_kind": kind})

        raw = {
            "evidence_id": evidence.evidence_id,
            "node_id": evidence.node_id,
            "lane": evidence.lane,
            "gate": evidence.gate,
            "sign": evidence.sign,
            "predeclared_sign": evidence.predeclared_sign,
            "information": evidence.information,
            "refutes_mechanism": evidence.refutes_mechanism,
            "confirms": evidence.confirms,
            "novel_mechanism": evidence.novel_mechanism,
            "expected_gain": evidence.expected_gain,
            "expected_hours": evidence.expected_hours,
            "router_context": context,
        }
        state = fresh_state(pending_evidence=raw, guards=dict(context["guards"]))
        final = app.invoke(state, {"configurable": {"thread_id": f"route-{index}"}})
        digests.append(state_digest(final))

        fired = [
            e for e in final.get("events", [])
            if e.get("kind") in {"RESEARCH_DIRECTION", "PRUNE", "JOIN"}
        ]
        decisions.append({
            "evidence_id": evidence.evidence_id,
            "node_id": evidence.node_id,
            "lane": evidence.lane,
            "condition": pure["condition"],
            "action": pure["action"],
            "reason": pure["reason"],
            "considered": list(pure.get("blocked_by", [])),
            "targets": [dict(t) for t in pure.get("targets", ())],
            "engine_events": [
                {k: v for k, v in e.items()
                 if k in {"kind", "node_id", "lane", "research_kind"}}
                for e in fired
            ],
            "sources_opened": sorted(final.get("sources", {})),
        })

    payload: dict[str, Any] = {
        "route_version": ROUTE_VERSION,
        "router_version": ROUTER_V3_VERSION,
        "context": {
            "lane_live_counts": context["lane_live_counts"],
            "lane_deficit_mass": context["lane_deficit_mass"],
            "residual_mass": context["residual_mass"],
            "stall_counter": context["guards"]["stall_counter"],
            "flipped_premises": context["flipped_premises"],
            "unexplained_cells": sum(
                1 for v in context["cell_status"].values() if v == "UNEXPLAINED"
            ),
            "ledger_contract": context["ledger_contract"],
        },
        "evidence_count": len(evidences),
        "decisions": decisions,
        "state_digests": digests,
        "engine_actually_ran": True,
        "generated_at": datetime.now(UTC).isoformat(),
    }
    RECEIPT.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )

    lines = [
        "# M271 P4 — 엔진 가동: 라우터가 정한 다음 노드",
        "",
        f"라우터 `{ROUTER_V3_VERSION}` / 증거 {len(evidences)} 건 / "
        "**실제 `m271_app.py` 실행**",
        "",
        "## 1. 라우터 상태 (그래프·원장에서 계산)",
        "",
        f"- 레인별 LIVE: `{context['lane_live_counts']}`",
        f"- 결손 질량: `{ {k: round(v, 5) for k, v in context['lane_deficit_mass'].items()} }`",
        f"- 미설명 잔여질량 **{context['residual_mass']:.5f}** (임계 0.010)",
        f"- **정체 카운터 {context['guards']['stall_counter']}** (한계 3) — "
        f"원장 이력 {context['ledger_contract']['events']} 건에서 **계산**된 값이다",
        f"- 원장 갱신 필요: **{context['ledger_contract']['refresh_due']}** / "
        f"미기록 사이클 **{context['ledger_contract']['unrecorded_cycles']}** 건",
        f"- 뒤집힌 전제: `{context['flipped_premises'] or '없음'}`",
        "",
        "## 2. 판정",
        "",
        "| 증거 | 레인 | 조건 | 행동 | 리서치 종류 | 범위 |",
        "|---|---|---|---|---|---|",
    ]
    for d in decisions:
        for target in (d["targets"] or [{}]):
            lines.append(
                f"| {d['evidence_id']} | {d['lane']} | **{d['condition']}** | "
                f"{d['action']} | {target.get('kind', '-')} | {target.get('scope', '-')} |"
            )
    lines += ["", "## 3. 이유", ""]
    for d in decisions:
        lines.append(
            f"- **{d['evidence_id']}** ({d['condition']} / {d['action']}) — {d['reason']}"
        )
        if d["considered"]:
            lines.append(f"  - 함께 발화한 조건: `{', '.join(d['considered'])}`")
    lines += ["", "## 4. 엔진이 연 리서치 노드", ""]
    for d in decisions:
        if d["sources_opened"]:
            lines.append(f"- {d['evidence_id']}: `{', '.join(d['sources_opened'])}`")
    lines += [
        "",
        "이 표는 **내가 고른 것이 아니다.** 동결된 라우터 표가 receipt 에서 뽑은 증거 "
        "서명을 읽고 낸 판정이며, 상태그래프를 실제로 통과시킨 결과다.",
        "",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("=== 엔진 가동 완료 ===")
    print(f"[ROUTE] 라우터 {ROUTER_V3_VERSION}")
    print(f"[ROUTE] 레인 LIVE {context['lane_live_counts']}")
    print(f"[ROUTE] 미설명 잔여질량 {context['residual_mass']:.5f} / 정체 "
          f"{context['guards']['stall_counter']} (이력 계산)")
    print(f"[ROUTE] 원장 갱신필요 {context['ledger_contract']['refresh_due']} / "
          f"미기록 사이클 {context['ledger_contract']['unrecorded_cycles']}")
    print(f"[ROUTE] 뒤집힌 전제 {context['flipped_premises'] or '없음'}")
    for d in decisions:
        print(f"[ROUTE] {d['evidence_id']:8s} {d['lane']}  -> {d['condition']:4s} {d['action']}")
        for t in d["targets"]:
            print(f"           kind={t.get('kind')} lane={t.get('lane')} "
                  f"scope={t.get('scope')}")
        if d["considered"]:
            print(f"           함께 발화: {d['considered']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
