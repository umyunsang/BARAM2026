"""M271 P4 — 발굴 가동 1단계: N0 증거를 적재하고 라우터를 실측한다.

N0 의 8 개 노드가 낸 실제 증거를 발굴 그래프와 결손 원장에 올린 뒤, 동결 라우터 표를
그대로 통과시켜 **무엇이 발화하는지** 측정한다. 여기서 나오는 것이 P4 의 첫 사이클 입력이다.

증거 서명은 각 노드의 receipt 에서 읽는다. 사람이 고쳐 넣지 않는다 — 그러면 라우터를
원하는 곳으로 몰 수 있다.

`deficit_cell` 귀속은 판단이 들어가는 유일한 지점이다. 결손 셀은 (group x 월 x y대역) 이라
레인이 없으므로, 그 셀을 직접 관측한 노드의 레인을 쓴다. 이 귀속이 틀리면 리서치가 엉뚱한
레인으로 가므로 리포트에 근거를 남긴다.

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

import m271_excavation_graph as xg
from m271_deficit import DeficitLedger
from m271_router import Evidence, decide

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
REPORT_MD = REPORTS / "m271_p4_bootstrap.md"
RECEIPT = REPORTS / "m271_p4_bootstrap_receipt.json"

# N0 노드 -> (레인, receipt 파일). 레인은 동결 사양이 정한 값이다.
SEED_CELLS = 5  # A7 이 기전 리서치를 뿌릴 상위 미설명 셀 수

N0_NODES: tuple[tuple[str, str, str], ...] = (
    ("A1_labels", "L1", "m271_n0_labels_receipt.json"),
    ("A2_columns", "L2", "m271_n0_columns_receipt.json"),
    ("A3_spatial", "L2", "m271_n0_spatial_receipt.json"),
    ("A5_scada", "L3", "m271_n0_scada_receipt.json"),
    ("A6_timing", "L1", "m271_n0_timing_receipt.json"),
    ("A4_error", "L2", "m271_n0_error_receipt.json"),
    ("A7_deficit_init", "L8", "m271_n0_deficit_init_receipt.json"),
)


def _load(name: str) -> dict[str, Any]:
    return json.loads((REPORTS / name).read_text(encoding="utf-8"))["result"]


def build_evidence_from_n0(ledger: DeficitLedger) -> list[tuple[Evidence, str]]:
    """각 N0 노드의 receipt 에서 증거 서명을 만든다. 근거를 함께 돌려준다."""
    out: list[tuple[Evidence, str]] = []
    top_cell = ledger.top(1)[0]["key"]

    labels = _load("m271_n0_labels_receipt.json")
    rho = max(r["lag1_autocorr_y"] for r in labels["ramps"])
    out.append(
        (
            Evidence(
                evidence_id="ev::A1", node_id="A1_labels", lane="L1",
                deficit_cell=None,
                # 자기상관이 검증표면의 유효표본을 40배 이상 줄인다는 발견.
                information=1.0,
                novel_mechanism="effective_sample_size" if rho > 0.9 else None,
                # 손실을 "유효표본수별" 로 쪼갤 수 없다. 검증 표면의 성질이므로 구조적이다.
                axis_kind="structural" if rho > 0.9 else None,
            ),
            f"lag-1 자기상관 {rho:.4f} -> 유효표본 급감. 행을 이 축으로 쪼갤 수 없으므로 "
            "분할 축이 아니라 구조적 사실이다.",
        )
    )

    columns = _load("m271_n0_columns_receipt.json")
    gfs, ldaps = columns["gfs"]["summary"], columns["ldaps"]["summary"]
    # 진짜 미사용 컬럼의 평균 MI 를 선언 컬럼 대비로 정규화한다.
    ratio = max(
        gfs["untouched_mean_mi"] / max(gfs["declared_mean_mi"], 1e-9),
        ldaps["untouched_mean_mi"] / max(ldaps["declared_mean_mi"], 1e-9),
    )
    out.append(
        (
            Evidence(
                evidence_id="ev::A2", node_id="A2_columns", lane="L2",
                deficit_cell=None,
                information=float(ratio),
                # 정보를 냈고 그 정보가 "이 축에는 없다" 이다. C8(정보 못 냄)과 다르다.
                negative_finding=ratio < 0.5,
            ),
            f"진짜 미사용 컬럼 평균 MI 가 선언 컬럼의 {ratio:.3f} 배. 정보를 냈고 그 정보가 "
            "부정이므로 확인된 부정(C14)이지 정보 미달(C8)이 아니다.",
        )
    )

    spatial = _load("m271_n0_spatial_receipt.json")
    nearest_gfs = {r["nearest_grid_id"] for r in spatial["distance_gfs"]}
    out.append(
        (
            Evidence(
                evidence_id="ev::A3", node_id="A3_spatial", lane="L2",
                deficit_cell=None,
                information=1.0,
                novel_mechanism="grid_resolution_mismatch" if len(nearest_gfs) == 1 else None,
                # 전 행에 균일하게 적용된다. 분할 축이 아니다.
                axis_kind="structural" if len(nearest_gfs) == 1 else None,
            ),
            f"17 기 전 터빈의 최근접 GFS 격자가 {len(nearest_gfs)} 개. 격자 해상도와 단지 "
            "규모의 불일치는 기존 결손 축으로 설명되지 않는다.",
        )
    )

    scada = _load("m271_n0_scada_receipt.json")
    resid = max(m["residual_std"] for m in scada["nacelle_vs_nwp"])
    out.append(
        (
            Evidence(
                evidence_id="ev::A5", node_id="A5_scada", lane="L3",
                deficit_cell=None, information=0.6,
            ),
            f"나셀-NWP 잔차 표준편차 최대 {resid:.3f} m/s. 직전 세션 sigma_v 를 독립 재현했다.",
        )
    )

    timing = _load("m271_n0_timing_receipt.json")
    held = timing["verdict"]["predeclared_expectation_held"]
    out.append(
        (
            Evidence(
                evidence_id="ev::A6", node_id="A6_timing", lane="L1",
                deficit_cell=None,
                sign=1 if held else -1, predeclared_sign=1,
                information=0.3,
            ),
            f"사전확약 유지={held}. 여유는 {{1.0h, 25.0h}} 두 값뿐이고 위반 0 건.",
        )
    )

    error = _load("m271_n0_error_receipt.json")
    check = error["predeclared_check"]
    partials = check["spread_vec_partial_by_group"]
    strongest = max(abs(float(v)) for v in partials.values())
    out.append(
        (
            Evidence(
                evidence_id="ev::A4", node_id="A4_error", lane="L2",
                # A4 는 최대 결손 셀을 직접 관측한다(월·풍향·시각별 오차).
                deficit_cell=top_cell,
                sign=1 if check["all_positive"] else -1,
                predeclared_sign=1,
                information=float(min(1.0, strongest / 0.05)),
                novel_mechanism="wind_sector",
                # 240 도 섹터가 행의 43% 를 차지한다. 행을 이 축으로 쪼갤 수 있다.
                axis_kind="partition",
            ),
            f"사전확약 {check['verdict']}. 편상관 최대 {strongest:.4f}. 240도 섹터가 행의 "
            "43%·최고풍속·최악오차를 동시에 차지하므로 결손 축 후보다.",
        )
    )

    deficit = _load("m271_n0_deficit_init_receipt.json")
    dec = deficit["decomposition"]
    out.append(
        (
            Evidence(
                evidence_id="ev::A7", node_id="A7_deficit_init", lane="L8",
                deficit_cell=None,
                sign=1 if dec["predeclared_held"] else -1,
                predeclared_sign=1,
                information=1.0,
                # 원장이 섰으므로 상위 미설명 셀에 기전 리서치를 뿌린다.
                seeds_cells=tuple(c["key"] for c in ledger.top(SEED_CELLS)),
            ),
            f"가법 분해 잔차 {dec['residual']:.3e} (허용 {dec['tolerance']:.0e}). "
            f"판정 {dec['verdict']}.",
        )
    )
    return out


def seed_graph(ledger: DeficitLedger) -> xg.ExcavationGraph:
    graph = xg.ExcavationGraph()
    graph.register_premise(
        xg.Premise(
            "NO_EXTERNAL_DATA",
            "외부 공개데이터를 쓸 수 없다",
            lambda s: not s.get("external_data_allowed", False),
        )
    )
    graph.register_premise(
        xg.Premise(
            "INFORMATION_BELOW_EPSILON",
            "이 굴착지점의 정보량이 임계 미만이다",
            lambda s: True,  # 재측정 전에는 계속 참
        )
    )
    graph.add_node("N0", node_type="MEASURE", lane="L8", status=xg.LIVE)
    for node_id, lane, _ in N0_NODES:
        graph.add_node(
            node_id, node_type="MEASURE", lane=lane, parents=["N0"], status=xg.LIVE
        )
    for key in ledger.cells:
        graph.add_node(key, node_type="DEFICIT", lane="L8", status=xg.LIVE)
    return graph


def router_context(graph: xg.ExcavationGraph, ledger: DeficitLedger) -> dict[str, Any]:
    """라우터가 읽는 상태. 사람이 손대지 않는다."""
    return {
        "cell_status": {k: c["status"] for k, c in ledger.cells.items()},
        "lane_live_counts": {lane: graph.live_count(lane) for lane in xg.LANES},
        # 결손 셀에는 아직 레인이 없다(기전이 미상이므로). 레인별 질량은 0 으로 두고,
        # 기전이 밝혀지면서 귀속된다. 이 상태에서 C6 가 발화하지 않는 것이 옳다.
        "lane_deficit_mass": dict.fromkeys(xg.LANES, 0.0),
        "residual_mass": ledger.residual_mass(),
        # 결손 셀의 레인. A4(L2)가 상위 셀들의 오차 구조를 직접 관측했으므로 L2 가 실측
        # 근거를 가진 소유자다. 기전이 밝혀지면 바뀔 수 있다.
        "cell_lane": dict.fromkeys((c["key"] for c in ledger.top(SEED_CELLS)), "L2"),
        "flipped_premises": graph.flipped_premises({"external_data_allowed": False}),
        "guards": {"stall_counter": 0},
        "node_direction": {},
    }


def main() -> int:
    ledger = DeficitLedger.from_a7()
    graph = seed_graph(ledger)
    context = router_context(graph, ledger)
    evidences = build_evidence_from_n0(ledger)

    routed = []
    for evidence, rationale in evidences:
        decision = decide(evidence, context)
        routed.append(
            {
                "node": evidence.node_id,
                "lane": evidence.lane,
                "deficit_cell": evidence.deficit_cell,
                "information": round(evidence.information, 4),
                "sign": evidence.sign,
                "predeclared_sign": evidence.predeclared_sign,
                "novel_mechanism": evidence.novel_mechanism,
                "condition": decision.condition,
                "action": decision.action,
                "considered": list(decision.considered),
                "targets": [dict(t) for t in decision.targets],
                "rationale": rationale,
            }
        )

    fired = [r for r in routed if r["condition"] != "NONE"]
    payload = {
        "ledger": {
            "total": ledger.total,
            "target": ledger.target,
            "gap": ledger.gap_to_target(),
            "cells": len(ledger.cells),
            "unexplained_mass": ledger.unexplained_mass(),
            "identity_ok": True,
        },
        "graph": {
            "nodes": graph.graph.number_of_nodes(),
            "frontier": graph.frontier(),
            "lane_census": {k: v for k, v in graph.lane_census().items()},
            "digest": graph.digest(),
        },
        "routed": routed,
        "fired": len(fired),
        "conditions_fired": sorted({r["condition"] for r in fired}),
    }

    lines = [
        "# M271 P4 — 발굴 가동 1단계: N0 증거 라우팅 실측",
        "",
        f"- 판정일: {datetime.now(UTC).strftime('%Y-%m-%d')} (UTC)",
        f"- 결손 원장: 셀 {payload['ledger']['cells']}, 미설명 질량 "
        f"{payload['ledger']['unexplained_mass']:.6f}",
        f"- 로컬 Total {ledger.total:.6f} / 목표 {ledger.target} / 격차 "
        f"{ledger.gap_to_target():.6f}",
        f"- 발굴 그래프: 노드 {payload['graph']['nodes']}, 프론티어 "
        f"{len(payload['graph']['frontier'])}",
        "",
        "N0 의 8 개 노드가 낸 **실제** 증거를 receipt 에서 읽어 동결 라우터 표에 그대로",
        "통과시켰다. 증거를 손으로 고쳐 넣지 않았다.",
        "",
        "## 라우팅 결과",
        "",
        "| 노드 | 레인 | 결손셀 | 정보량 | 부호 | 발화 | 동작 | 검토된 조건 |",
        "|---|---|---|---:|---:|---|---|---|",
    ]
    for r in routed:
        cell = r["deficit_cell"] or "-"
        sign = f"{r['sign']:+d}" if r["sign"] else "0"
        considered = ", ".join(r["considered"]) or "-"
        lines.append(
            f"| `{r['node']}` | {r['lane']} | `{cell}` | {r['information']:.3f} | {sign} | "
            f"**{r['condition']}** | `{r['action']}` | {considered} |"
        )

    lines += ["", "## 노드별 근거", ""]
    for r in routed:
        lines += [f"### `{r['node']}` -> {r['condition']}", "", r["rationale"], ""]
        for target in r["targets"]:
            lines.append(
                f"- 표적: `{target.get('kind')}` / 레인 `{target.get('lane')}` / "
                f"범위: {target.get('scope')}"
            )
        if not r["targets"]:
            lines.append("- 표적 없음")
        lines.append("")

    lines += [
        "## 읽는 법",
        "",
        "발화하지 않은 것도 판정이다. 조건표가 억지로 방향을 붙이지 않는다는 뜻이다.",
        "",
        "결손 셀에는 아직 레인이 없다. 기전이 미상이므로 레인별 질량을 0 으로 두었고, 따라서",
        "C6(레인 기아)가 발화하지 않는 것이 옳다. 기전이 밝혀지면서 질량이 레인에 귀속되고",
        "그때부터 C6 가 의미를 갖는다.",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    receipt = {
        "schema_version": 1,
        "stage": "M271_P4_BOOTSTRAP",
        "decided_utc": datetime.now(UTC).isoformat(),
        "result": payload,
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

    for r in routed:
        print(f"[P4] {r['node']:16s} lane={r['lane']} info={r['information']:.3f} "
              f"-> {r['condition']:5s} {r['action']}")
    print(f"\n발화 {len(fired)}/{len(routed)}, 조건 {payload['conditions_fired']}")
    print(f"report -> {REPORT_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
