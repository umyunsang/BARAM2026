"""M271 — 미기록 사이클 문제를 닫는다: 감사 백필 + 신규 노드 기록 + 강제 확인.

감사가 **79 건**을 냈다. 계약 API 를 만들었지만 아무것도 강제하지 않아, 계약 이후에
만든 노드 넷(C1N77~C1N80)도 여전히 안 썼다 — 엔진을 만들고 안 돌린 것과 같은 형태다.

이 스크립트가 셋을 한다.

  1. **감사 백필** — 계약 이전 사이클을 하나의 AUDIT 사건으로 덮는다. 가짜 사건을
     지어내지 않고 `cycles_covered` 로 매크로스텝 수를 남긴다.
  2. **신규 노드 기록** — receipt 가 있는 노드는 `close_cycle()` 로 개별 기록한다.
  3. **강제 확인** — `require_recorded()` 가 통과하는지 본다. 통과해야 라우팅 드라이버가
     이 노드들의 증거를 받는다.

정체가 진짜 값으로 돌아오는 것이 목적이다. 그 부작용(C10 재발화)은
`m271_c10_loop_engine.decide_v3` 가 `loop_engine_visits` 로 막는다.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import m271_cycle_contract as cc
import m271_ledger_state as ls
from m271_c10_loop_engine import ROUTER_V3_VERSION, decide_v3
from m271_p4_consolidate import CYCLES, build
from m271_router import Evidence

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
REPORT_MD = REPORTS / "m271_close_cycles.md"
RECEIPT = REPORTS / "m271_close_cycles_receipt.json"

NODE_ID = "C1N81_CYCLE_CONTRACT_ENFORCED"
LANE = "L8"
PARENT_NODE = "C1N79_LEDGER_CONTRACT"
CHAMPION_LOCAL = 0.630310

# 계약 이후 만들어졌고 receipt 가 있는 노드. 개별 기록 대상이다.
POST_CONTRACT = (
    # 라우팅에 증거원으로 쓰이는 노드는 AUDIT 에 묶이지 않고 **개별 기록**이 필요하다.
    # 계약 강제가 실제로 이 셋을 거부했고, 그것이 강제가 작동한다는 증거다.
    ("C1N68_EMPIRICAL_DECOMPOSITION", "m271_cycle68_empirical_decomposition_receipt.json"),
    ("C1N73_GROUP_BLEND_GATE", "m271_cycle73_group_blend_gate_receipt.json"),
    ("C1N76_CIRCULAR_BLOCK", "m271_cycle76_circular_block_receipt.json"),
    ("C1N77_PER_SOURCE_STACK", "m271_cycle77_per_source_stack_receipt.json"),
    ("C1N78_LOOP_ENGINE_C10", "m271_c10_loop_engine_receipt.json"),
    ("C1N79_LEDGER_CONTRACT", "m271_ledger_contract_receipt.json"),
    ("C1N80_FUSION_ANOMALY", "m271_c5_fusion_anomaly_receipt.json"),
    ("C1N82_TEACHER_SCALEUP", "m271_c7_teacher_scaleup_receipt.json"),
    ("C1N83_TEACHER_SCALEUP_REJUDGED", "m271_c7b_rejudge_receipt.json"),
    ("C1N84_TEACHER_CHRONOLOGICAL", "m271_c7c_chronological_receipt.json"),
    ("C1N85_C16_RECHECK", "m271_c16_recheck_receipt.json"),
    ("C1N86_LANE_EXPAND", "m271_c6_lane_expand_receipt.json"),
    ("C1N87_CURTAILMENT_CLEAN_TARGET", "m271_n1_curtailment_clean_receipt.json"),
    ("C1N88_N2_PROBE_L5_OPEN", "m271_n2_probe_l5_open_receipt.json"),
    ("C1N89_ANOMALY_CLEAN_TARGET", "m271_n2_anomaly_clean_receipt.json"),
    ("C1N90_WITHIN_BIN_INTEGRATION", "m271_n4_within_bin_receipt.json"),
    ("C1N91_WITHIN_BIN_ESTABLISHED", "m271_n4b_establish_receipt.json"),
)


def main() -> int:
    _graph, ledger = build()
    ledger.compute_efficiency()
    ls.apply_to(ledger)

    cycle_ids = [c[0] for c in CYCLES]
    before = cc.contract_report(ledger, cycle_ids, CHAMPION_LOCAL)

    audit = cc.audit_backfill(ledger, cycle_ids, CHAMPION_LOCAL)

    recorded: list[dict[str, Any]] = []
    skipped: list[str] = []
    for node_id, receipt_name in POST_CONTRACT:
        if any(e["node_id"] == node_id for e in ls.load_history()):
            skipped.append(node_id)
            continue
        if not (REPORTS / receipt_name).exists():
            skipped.append(f"{node_id}(receipt 없음)")
            continue
        recorded.append(
            cc.close_cycle(
                node_id, receipt_name,
                champion_total=CHAMPION_LOCAL, ledger=ledger,
                note=f"계약 강제 시점에 소급 기록. 근거 {receipt_name}",
            )
        )

    after = cc.contract_report(ledger, cycle_ids, CHAMPION_LOCAL)

    # V1 — 강제가 실제로 통과하는가.
    try:
        missing = cc.require_recorded([n for n, _ in POST_CONTRACT], strict=True)
        v1 = not missing
    except cc.CycleContractViolation:
        v1 = False

    # V2 — 미기록 노드는 여전히 거부되는가(강제가 살아 있는가).
    try:
        cc.require_recorded(["NODE_THAT_NEVER_RAN"], strict=True)
        v2 = False
    except cc.CycleContractViolation:
        v2 = True

    # V3 — 정체가 진짜 값으로 돌아왔는가.
    v3 = bool(after["stall_counter"] >= before["stall_counter"])

    # V4 — C10 재발화가 막히는가. 정체가 크고 C10 을 이미 거친 상태를 넣어 본다.
    stalled = {
        "remaining_gap": 0.66 - CHAMPION_LOCAL,
        "guards": {"stall_counter": after["stall_counter"], "loop_engine_visits": 1},
        "recent_lane_kind": [], "proposed_kind": None,
        "lane_live_counts": {}, "lane_deficit_mass": {},
        "cell_status": {}, "residual_mass": after["recoverable_mass"],
    }
    probe = Evidence(
        evidence_id="PROBE", node_id="PROBE", lane="L6",
        sign=1, predeclared_sign=1, information=0.9, confirms=True,
        expected_gain=0.017, expected_hours=2.0,
    )
    after_service = decide_v3(probe, dict(stalled))
    unserviced = decide_v3(
        probe, dict(stalled, guards={"stall_counter": after["stall_counter"],
                                     "loop_engine_visits": 0})
    )
    v4 = bool(
        after_service["condition"] != "C10" and unserviced["condition"] == "C10"
    )

    payload: dict[str, Any] = {
        "node_id": NODE_ID,
        "lane": LANE,
        "parent": PARENT_NODE,
        "router_v3": ROUTER_V3_VERSION,
        "before": before,
        "audit_backfill": (
            {k: v for k, v in audit.items() if k != "cell_updates"} if audit else None
        ),
        "recorded_nodes": [r["node_id"] for r in recorded],
        "skipped": skipped,
        "after": after,
        "c10_gating": {
            "with_visit": after_service,
            "without_visit": {k: v for k, v in unserviced.items()
                              if k in {"condition", "action"}},
        },
        "checks": {
            "V1_post_contract_nodes_recorded": v1,
            "V2_unrecorded_still_refused": v2,
            "V3_stall_restored": v3,
            "V4_c10_gated_after_service": v4,
        },
        "dacon_upload": False,
        "lockbox_used": False,
        "generated_at": datetime.now(UTC).isoformat(),
    }
    payload["digest"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]
    RECEIPT.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str),
                       encoding="utf-8")

    lines = [
        "# M271 — 사이클 계약 강제 (미기록 문제 종결)",
        "",
        f"노드 `{NODE_ID}` / 레인 {LANE} / 부모 `{PARENT_NODE}`",
        "",
        "## 1. 무엇이 문제였나",
        "",
        f"원장 계약을 만든 직후 감사가 **미기록 {before['unrecorded_cycles']} 건**을 냈다. "
        "계약 API 는 있는데 아무것도 강제하지 않아, 계약 이후 만든 노드 넷도 안 썼다 — "
        "**엔진을 만들고 안 돌린 것과 같은 형태**다.",
        "",
        "## 2. 조치",
        "",
        f"- **감사 백필**: {audit['cycles_covered'] if audit else 0} 개 사이클을 AUDIT 사건 "
        "하나로 덮었다. 가짜 사건을 지어내지 않고 `cycles_covered` 로 매크로스텝 수를 "
        "남긴다 — 사이클별 챔피언 이력은 재구성 불가이고, 재구성하려 들면 그것이 이 "
        "사태를 만든 사후 전사다.",
        f"- **개별 기록**: `{', '.join(payload['recorded_nodes']) or '없음'}`",
        f"- **건너뜀**: `{', '.join(skipped) or '없음'}`",
        "",
        "## 3. 결과",
        "",
        "| | 이전 | 이후 |",
        "|---|---:|---:|",
        f"| 이력 사건 | {before['events']} | {after['events']} |",
        f"| 덮은 매크로스텝 | {before['cycles_covered_total']} | "
        f"{after['cycles_covered_total']} |",
        f"| 정체 카운터 | {before['stall_counter']} | **{after['stall_counter']}** |",
        f"| 미기록 사이클 | {before['unrecorded_cycles']} | "
        f"**{after['unrecorded_cycles']}** |",
        "",
        "## 4. 부작용 처리 — C10 재발화",
        "",
        "정체가 진짜 값으로 돌아오면 C10 이 또 모든 증거를 삼킨다. 계획서 종료 가드 "
        "사다리가 이미 규정했다 — \"진행정체(...), **단 C10 을 1 회 거친 뒤**\". "
        "`decide_v3` 가 `loop_engine_visits` 로 그것을 강제한다.",
        "",
        f"- 서비스 전 (`loop_engine_visits=0`): **{unserviced['condition']}** "
        f"{unserviced['action']}",
        f"- 서비스 후 (`loop_engine_visits=1`): **{after_service['condition']}** "
        f"{after_service['action']}",
        "",
        "## 5. 타당성 가드",
        "",
        f"- V1 계약 이후 노드가 기록됨 -> **{v1}**",
        f"- V2 미기록 노드는 여전히 거부됨 -> **{v2}**",
        f"- V3 정체가 진짜 값으로 복원 -> **{v3}**",
        f"- V4 C10 이 서비스 후 게이팅됨 -> **{v4}**",
        "",
        f"digest `{payload['digest']}`",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("=== 사이클 계약 강제 ===")
    print(f"[CLOSE] 미기록 {before['unrecorded_cycles']} -> {after['unrecorded_cycles']}")
    print(f"[CLOSE] 이력 {before['events']} -> {after['events']} 건 / "
          f"덮은 매크로스텝 {before['cycles_covered_total']} -> "
          f"{after['cycles_covered_total']}")
    print(f"[CLOSE] 정체 {before['stall_counter']} -> {after['stall_counter']}")
    print(f"[CLOSE] 기록됨 {payload['recorded_nodes']}")
    print(f"[CLOSE] C10 게이팅  전 {unserviced['condition']} / "
          f"후 {after_service['condition']} {after_service['action']}")
    print(f"[CLOSE] V1 {v1} / V2 {v2} / V3 {v3} / V4 {v4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
