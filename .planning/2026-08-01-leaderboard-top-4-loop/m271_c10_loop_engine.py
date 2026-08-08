"""M271 C10 — 라우터가 지시한 루프엔진 제자리 개선. 실제 딥리서치 기반.

**이 노드는 내가 고른 것이 아니다.** `m271_p4_route.py` 가 상태그래프를 실제로 통과시켰고
증거 세 건이 전부 C10 으로 갔다 — `kind=loop_engine`, `lane=L8`,
scope "루프·그래프 엔지니어링 라우터 표의 SOTA".

    C1N68  L6  -> C10   함께 발화 [C7, C10]
    C1N73  L7  -> C10   함께 발화 [C10]
    C1N76  L4  -> C10   함께 발화 [C2, C10, C13, C14]

발화 조건은 `stall_counter >= 3` 인데 실측 **77** 이다. 결손 원장이 A7 이후 한 번도
갱신되지 않아 질량 무감소가 기록된 사이클 수만큼 이어졌기 때문이다. 엔진을 돌렸다면
아주 오래전에 멈춰 세웠을 것이고, 나는 0.005 규모 이웃을 20 여 사이클 팠다.
상위권과의 격차는 0.037 이다.

**① 방향 리서치 (실제 수행, 2026-08-06)**

  ShinkaEvolve (Sakana AI, ICLR 2026) 가 세 축을 제시하고 각각이 내 결함에 대응한다.
  https://arxiv.org/abs/2509.19349 / https://sakana.ai/shinka-evolve/

    (가) **novelty rejection-sampling** — "기존 프로그램의 사소한 변형을 평가하느라
         시간을 낭비하지 않도록" 아카이브와의 임베딩 유사도가 임계를 넘으면 제안을
         기각한다.
         -> 내 결함: 결정층 미세변형(C60 온도, C61 격자, C71 가중, C72·C73 결합)을
            연달아 팠다. 유사도 기각이 있었으면 두 번째에서 걸렸다.

    (나) **exploration-exploitation 균형 부모 선택**
         -> 내 결함: C11 의 `base_value / voi / exploration` 은 **선언 상수**이고
            계획서 R3 가 "보정된 값이 아니다" 라고 이미 기록해 뒀다.

    (다) **bandit 기반 선택** — 성과에 따라 가중이 갱신된다.
         -> 내 결함: 표가 고정이라 20 번 실패해도 같은 우선순위를 유지한다.

  Bayesian optimization 의 획득함수는 기대개선을 **불확실성 대비**로 잰다.
  https://arxiv.org/pdf/2507.01903
    -> 내 결함: `expected_gain` 이 정규화 없이 들어가 **0.005 든 0.037 든 같은 방식으로
       취급**된다. 남은 격차 대비 크기를 보는 항이 아예 없다.

  **적용성 태그**: (가)(다) `directly_supported`, (나) `near_match_only`
  (ShinkaEvolve 는 프로그램 진화, 우리는 실험 라우팅이라 부모 선택 개념이 1:1 은 아니다).

**② 개선 사양 — 라우터 v3**

  **표를 후보가 통과하도록 조정하지 않는다.** v2 가 v1 을 고칠 때와 같은 규율이다 —
  범주 오류와 누락 조건만 고치고 임계값은 근거 없이 건드리지 않는다. v2 를 지우지 않고
  롤백 경로를 유지한다.

  R1  **크기 게이트 (C16)**  획득함수의 정규화를 도입한다.
      `expected_gain < MAGNITUDE_FLOOR * remaining_gap` 이면 그 방향을 **추격하지
      않는다**. `remaining_gap` 은 목표 - 현재 로컬이고 상태에서 온다.
      `MAGNITUDE_FLOOR = 0.15` 로 동결 — 20 사이클치 후보(0.005/0.030 = 0.167)가
      **경계 바로 위**에 오도록 잡은 것이 아니라, "격차의 1/7 미만은 단독으로 의미 없다"
      는 선언 관례다. 이 값은 보정된 바 없으며 R3 와 같은 지위다.

  R2  **신규성 기각 (C17)**  최근 `NOVELTY_WINDOW` 개 노드와 **같은 레인 + 같은
      research_kind** 조합이 `NOVELTY_LIMIT` 회 이상 반복되면 그 방향을 기각하고
      레인 확장으로 보낸다. ShinkaEvolve 의 임베딩 유사도를 우리 자료구조에 맞게
      **레인x종류 반복**으로 대체한다 — 우리에겐 프로그램 텍스트가 아니라 노드
      메타데이터가 있다.

  R3  **원장 갱신 강제**  `stall_counter` 가 77 이 된 근본 원인은 결손 원장이 A7 이후
      한 번도 갱신되지 않은 것이다. 연료계가 죽어 있으면 정체 탐지가 무의미하다.
      이 노드는 그 사실을 **구조적 결함으로 기록**하고, 원장 갱신을 사이클 계약에
      넣을 것을 명시한다.

  **타당성 가드**
    V1  v3 가 v2 의 판정을 **바꾸지 않는 경우가 있어야 한다** — C16·C17 이 발화하지
        않는 증거에서는 v2 와 동일한 조건·행동이 나온다. 전부 바뀌면 표를 갈아엎은 것이다.
    V2  v3 가 **이번 실패를 실제로 잡는다** — C60/C61/C71/C72/C73 계열 증거를 넣으면
        C16(크기) 또는 C17(신규성)이 발화한다. 못 잡으면 개선이 아니다.
    V3  결정성 — 같은 입력에 같은 출력. `m271_engine_verify.py` 1 번 검사와 같은 규율.

  사전확약:
    H1  V1·V2·V3 이 모두 참이다.
    H2  v3 로 다시 라우팅하면 C1N68(확인된 방향, 이득 0.017 = 격차의 0.57)은
        **C16 에 걸리지 않는다**. 자릿수가 맞는 방향은 통과해야 한다.
    H3  C1N73(이득 0.0049 = 격차의 0.16)은 C16 경계 근처다. 걸리든 통과하든
        **경계값을 결과를 보고 조정하지 않는다.**

**진단·엔진 개선 전용.** 모델을 바꾸지 않는다. 게이트·lockbox·외부데이터 미사용.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from m271_router import (
    CONDITIONS,
    EPSILON_INFORMATION,
    ROUTER_VERSION,
    Evidence,
    decide,
)

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
REPORT_MD = REPORTS / "m271_c10_loop_engine.md"
RECEIPT = REPORTS / "m271_c10_loop_engine_receipt.json"

NODE_ID = "C1N78_LOOP_ENGINE_C10"
LANE = "L8"
PARENT_NODE = "C1N76_CIRCULAR_BLOCK"

ROUTER_V3_VERSION = "M271_ROUTER_v3_frozen_2026-08-06"
MAGNITUDE_FLOOR = 0.15  # 남은 격차 대비. 선언 관례이며 보정된 값이 아니다.
NOVELTY_WINDOW = 6
NOVELTY_LIMIT = 3

RESEARCH = {
    "performed_at": "2026-08-06",
    "trigger": "라우터 C10 (stall_counter 77 >= 3)",
    "kind": "loop_engine",
    "lane": "L8",
    "scope": "루프·그래프 엔지니어링 라우터 표의 SOTA",
    "sources": [
        {
            "url": "https://arxiv.org/abs/2509.19349",
            "class": "peer_reviewed",
            "finding": "ShinkaEvolve — novelty rejection-sampling, exploration-exploitation "
                       "균형 부모 선택, bandit 기반 선택. 150 표본으로 circle packing SOTA",
            "applicability": "directly_supported",
        },
        {
            "url": "https://sakana.ai/shinka-evolve/",
            "class": "official_docs",
            "finding": "임베딩 코사인 유사도가 임계 초과면 제안 기각 — '사소한 변형' 낭비 차단",
            "applicability": "directly_supported",
        },
        {
            "url": "https://arxiv.org/pdf/2507.01903",
            "class": "peer_reviewed",
            "finding": "AI4Research 서베이 — 획득함수가 기대개선을 불확실성 대비로 정규화",
            "applicability": "near_match_only",
        },
    ],
    "decision_impact": "라우터 v3 에 크기 게이트(C16)와 신규성 기각(C17)을 추가",
    "stop_condition": "V2 가 거짓이면(이번 실패를 못 잡으면) 개선이 아니므로 채택하지 않는다",
}


def remaining_gap(state: dict) -> float:
    return float(state.get("remaining_gap", 0.0))


def fires_c16(evidence: Evidence, state: dict) -> bool:
    """크기 게이트 — 남은 격차 대비 너무 작은 방향은 추격하지 않는다."""
    gap = remaining_gap(state)
    if gap <= 0.0:
        return False
    return evidence.expected_gain < MAGNITUDE_FLOOR * gap


def fires_c17(evidence: Evidence, state: dict) -> bool:
    """신규성 기각 — 같은 레인 x 같은 종류가 창 안에서 반복되면 기각한다."""
    recent = list(state.get("recent_lane_kind", []))[-NOVELTY_WINDOW:]
    key = (evidence.lane, state.get("proposed_kind"))
    if key[1] is None:
        return False
    return sum(1 for r in recent if tuple(r) == key) >= NOVELTY_LIMIT


def c10_already_serviced(state: dict) -> bool:
    """C10 을 이미 한 번 거쳤는가.

    계획서의 종료 가드 사다리가 규정한다 — "진행정체(N 매크로스텝간 결손질량 무감소,
    **단 C10 을 1 회 거친 뒤**)". 즉 C10 은 한 번 발화하고, 그 뒤에도 정체가 이어지면
    그것은 **종료 조건**이지 또 다른 C10 이 아니다.

    v2 의 C10 조건(`m271_router.py:299`)은 `stall_counter` 만 읽고 `loop_engine_visits`
    를 보지 않는다. 원장 계약으로 정체가 진짜 값으로 돌아오면 C10 이 모든 증거를 다시
    삼켜 다른 조건을 전부 가린다 — 실제로 한 번 그렇게 됐다.
    """
    return int(state.get("guards", {}).get("loop_engine_visits", 0)) >= 1


def decide_v3(evidence: Evidence, state: dict) -> dict[str, Any]:
    """v2 판정 위에 C16·C17 을 얹는다. v2 를 대체하지 않고 **선행 검사**로 둔다."""
    base = decide(evidence, state)
    blocked: list[str] = []

    # C10 을 이미 서비스했으면 **후보 집합에서 빼고 다시 고른다.**
    #
    # 처음엔 곧바로 HALT 로 보냈는데 그것이 틀렸다. 계획서의 종료 가드 사다리는
    # "다른 것이 아무것도 발화하지 않을 때" 의 사다리다. C7·C2·C5·C13·C14 가 살아
    # 있는데 HALT 로 보내면 종료가 아니라 **눈을 감는 것**이다 — 실제로 증거 다섯 건이
    # 전부 HALT 로 갔다.
    #
    # `stall_counter` 를 0 으로 낮춘 상태로 재판정하면 C10 만 발화에서 빠지고 나머지
    # 조건들이 C11 가치순으로 자연스럽게 재선택된다. 조건표를 손대지 않는다.
    if base.condition == "C10" and c10_already_serviced(state):
        without_c10 = decide(
            evidence,
            {**state, "guards": {**dict(state.get("guards", {})), "stall_counter": 0}},
        )
        if without_c10.condition == "NONE":
            return {
                "condition": "TERMINATION_GUARD",
                "action": "HALT",
                "reason": (
                    "정체가 지속되고 C10 을 이미 1 회 거쳤는데 다른 조건이 하나도 "
                    "발화하지 않는다. 계획서 종료 가드 사다리의 종료 조건이다."
                ),
                "blocked_by": ["C10_ALREADY_SERVICED", "NO_OTHER_CONDITION"],
                "targets": (),
                "v2_would_have": {"condition": base.condition, "action": base.action},
                "router_version": ROUTER_V3_VERSION,
            }
        base = without_c10
        blocked.append("C10_SUPPRESSED_ALREADY_SERVICED")

    if fires_c16(evidence, state):
        blocked.append("C16")
    if fires_c17(evidence, state):
        blocked.append("C17")

    hard = [b for b in blocked if b in {"C16", "C17"}]
    if hard:
        return {
            "condition": hard[0],
            "action": "REJECT_DIRECTION",
            "reason": (
                f"C16 크기 게이트 (이득 {evidence.expected_gain:.5f} < "
                f"{MAGNITUDE_FLOOR} x 격차 {remaining_gap(state):.5f})"
                if hard[0] == "C16"
                else f"C17 신규성 기각 (레인 {evidence.lane} 반복 >= {NOVELTY_LIMIT})"
            ),
            "blocked_by": blocked,
            "targets": (),
            "v2_would_have": {"condition": base.condition, "action": base.action},
            "router_version": ROUTER_V3_VERSION,
        }
    return {
        "condition": base.condition,
        "action": base.action,
        "reason": base.reason,
        "targets": base.targets,
        "blocked_by": blocked,
        "v2_would_have": {"condition": base.condition, "action": base.action},
        "router_version": ROUTER_V3_VERSION,
    }


def main() -> int:
    # 이번 실패를 그대로 재구성한 증거들. 전부 결정층 미세변형이고 이득이 0.005 규모다.
    gap = 0.66 - 0.630310
    failure_state = {
        "remaining_gap": gap,
        "lane_live_counts": {},
        "lane_deficit_mass": {},
        "cell_status": {},
        "residual_mass": 0.04875,
        "guards": {"stall_counter": 77},
        "recent_lane_kind": [("L7", "amplify")] * 4,
        "proposed_kind": "amplify",
    }
    failures = [
        Evidence(evidence_id="C1N60", node_id="C1N60_LEVEL_TEMPERATURE", lane="L7",
                 gate={"G1": False, "G2": True, "G3": False, "G4": True},
                 sign=1, predeclared_sign=1, information=0.7,
                 expected_gain=0.008990, expected_hours=1.5),
        Evidence(evidence_id="C1N61", node_id="C1N61_GRID_BOUNDARY", lane="L7",
                 sign=-1, predeclared_sign=1, information=0.6,
                 expected_gain=0.001169, expected_hours=1.0),
        Evidence(evidence_id="C1N71", node_id="C1N71_TEACHER_WEIGHT", lane="L2",
                 sign=1, predeclared_sign=1, information=0.7,
                 expected_gain=0.001238, expected_hours=1.0),
        Evidence(evidence_id="C1N73", node_id="C1N73_GROUP_BLEND_GATE", lane="L7",
                 gate={"G1": False, "G2": True, "G3": False, "G4": False},
                 sign=1, predeclared_sign=1, information=0.5,
                 expected_gain=0.004931, expected_hours=1.0),
    ]

    caught = []
    for evidence in failures:
        verdict = decide_v3(evidence, dict(failure_state))
        caught.append({
            "evidence_id": evidence.evidence_id,
            "lane": evidence.lane,
            "expected_gain": evidence.expected_gain,
            "gain_over_gap": evidence.expected_gain / gap,
            "v3": verdict,
        })

    # V1 — C16·C17 이 발화하지 않는 증거에서는 v2 와 같은 판정이 나와야 한다.
    passthrough = Evidence(
        evidence_id="C1N68", node_id="C1N68_EMPIRICAL_DECOMPOSITION", lane="L6",
        sign=1, predeclared_sign=1, information=0.9, confirms=True,
        expected_gain=0.017, expected_hours=2.0,
    )
    clean_state = {**failure_state, "recent_lane_kind": [], "proposed_kind": None}
    pass_verdict = decide_v3(passthrough, dict(clean_state))
    v1 = bool(
        not pass_verdict["blocked_by"]
        and pass_verdict["condition"] == pass_verdict["v2_would_have"]["condition"]
    )

    # V2 — 이번 실패 계열을 실제로 잡는가.
    blocked_count = sum(1 for c in caught if c["v3"]["blocked_by"])
    v2 = bool(blocked_count >= 3)

    # V3 — 결정성.
    repeat = [decide_v3(e, dict(failure_state)) for e in failures]
    v3 = bool(all(
        repeat[i]["condition"] == caught[i]["v3"]["condition"] for i in range(len(failures))
    ))

    h1 = bool(v1 and v2 and v3)
    h2 = bool(not pass_verdict["blocked_by"])
    c73 = next(c for c in caught if c["evidence_id"] == "C1N73")
    h3_note = (
        f"C1N73 이득비 {c73['gain_over_gap']:.3f} vs 문턱 {MAGNITUDE_FLOOR} — "
        f"{'차단' if c73['v3']['blocked_by'] else '통과'}. 경계값을 결과를 보고 조정하지 않았다."
    )

    verdict_code = (
        "ROUTER_V3_ADOPTED" if h1
        else "ROUTER_V3_REJECTED_DOES_NOT_CATCH_THE_FAILURE"
    )

    payload: dict[str, Any] = {
        "node_id": NODE_ID,
        "lane": LANE,
        "parent": PARENT_NODE,
        "triggered_by": "C10 (라우터가 지시)",
        "research": RESEARCH,
        "router_v2": ROUTER_VERSION,
        "router_v3": ROUTER_V3_VERSION,
        "revisions": {
            "C16_magnitude_floor": {
                "value": MAGNITUDE_FLOOR,
                "rule": "expected_gain < MAGNITUDE_FLOOR * remaining_gap 이면 REJECT_DIRECTION",
                "status": "선언 관례. 보정된 값이 아니다(계획 R3 와 같은 지위).",
            },
            "C17_novelty_rejection": {
                "window": NOVELTY_WINDOW,
                "limit": NOVELTY_LIMIT,
                "rule": "같은 레인 x 같은 kind 가 창 안에서 limit 회 이상이면 REJECT_DIRECTION",
                "adapted_from": "ShinkaEvolve 임베딩 유사도 기각 -> 노드 메타데이터 반복",
            },
            "R3_ledger_update_contract": {
                "defect": "결손 원장이 A7 이후 한 번도 갱신되지 않아 stall_counter 가 77",
                "consequence": "연료계가 죽어 정체 탐지가 무의미했다",
                "required": "사이클 계약에 원장 갱신을 넣는다",
            },
        },
        "epsilon_information": EPSILON_INFORMATION,
        "condition_count_v2": len(CONDITIONS),
        "remaining_gap": gap,
        "failure_replay": caught,
        "passthrough_check": pass_verdict,
        "checks": {
            "V1_v2_preserved_when_not_firing": v1,
            "V2_catches_this_failure": v2,
            "V2_blocked_count": blocked_count,
            "V3_deterministic": v3,
        },
        "hypotheses": {"H1_all_guards": h1, "H2_large_direction_passes": h2},
        "H3_note": h3_note,
        "verdict": verdict_code,
        "changes_model": False,
        "dacon_upload": False,
        "lockbox_used": False,
        "external_actions": ["WebSearch"],
        "generated_at": datetime.now(UTC).isoformat(),
    }
    payload["digest"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]
    RECEIPT.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str),
                       encoding="utf-8")

    lines = [
        "# M271 C10 — 루프엔진 제자리 개선 (라우터가 지시한 노드)",
        "",
        f"노드 `{NODE_ID}` / 레인 {LANE} / 부모 `{PARENT_NODE}`",
        "",
        "**내가 고른 노드가 아니다.** `m271_p4_route.py` 가 상태그래프를 통과시켰고 증거 "
        "세 건이 전부 C10 으로 갔다 (`stall_counter` 77 >= 3).",
        "",
        "## 1. 방향 리서치 (실제 수행)",
        "",
    ]
    for s in RESEARCH["sources"]:
        lines.append(f"- {s['finding']} — <{s['url']}> (`{s['applicability']}`)")
    lines += [
        "",
        "## 2. 라우터 v3 개정",
        "",
        f"- **C16 크기 게이트** — `expected_gain < {MAGNITUDE_FLOOR} x 남은격차` 면 "
        "방향을 추격하지 않는다. 획득함수 정규화의 최소형이다.",
        f"- **C17 신규성 기각** — 같은 레인 x 같은 kind 가 최근 {NOVELTY_WINDOW} 개 중 "
        f"{NOVELTY_LIMIT} 회 이상이면 기각. ShinkaEvolve 의 유사도 기각을 노드 "
        "메타데이터로 옮긴 것.",
        "- **원장 갱신 계약** — `stall_counter` 77 의 근본 원인. 연료계가 죽어 있었다.",
        "",
        f"남은 격차 **{gap:.6f}** (목표 0.66 - 챔피언 0.630310)",
        "",
        "## 3. 이번 실패 재생 — v3 가 잡는가",
        "",
        "| 증거 | 레인 | 기대이득 | 격차대비 | v3 판정 | v2 였다면 |",
        "|---|---|---:|---:|---|---|",
    ]
    for c in caught:
        v3v = c["v3"]
        lines.append(
            f"| {c['evidence_id']} | {c['lane']} | {c['expected_gain']:.6f} | "
            f"{c['gain_over_gap']:.3f} | **{v3v['condition']}** {v3v['action']} | "
            f"{v3v['v2_would_have']['condition']} {v3v['v2_would_have']['action']} |"
        )
    lines += [
        "",
        "## 4. 타당성 가드",
        "",
        f"- V1 발화하지 않는 증거에서 v2 판정 보존 -> **{v1}** "
        f"(C1N68: {pass_verdict['condition']} / {pass_verdict['action']})",
        f"- V2 이번 실패를 잡는다 ({blocked_count}/4 차단) -> **{v2}**",
        f"- V3 결정성 -> **{v3}**",
        "",
        "## 5. 사전확약",
        "",
        f"- H1 세 가드 전부 -> **{h1}**",
        f"- H2 자릿수 맞는 방향(C1N68, 격차대비 {0.017 / gap:.2f})은 통과 -> **{h2}**",
        f"- H3 {h3_note}",
        "",
        "## 6. 판정",
        "",
        f"**{verdict_code}**",
        "",
        "임계값은 **선언 관례**이며 보정된 바 없다(계획 R3 와 같은 지위). 후보가 통과하도록 "
        "조정하지 않았고, v2 는 지우지 않아 롤백 경로가 남는다.",
        "",
        f"digest `{payload['digest']}`",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("=== C10 완료 ===")
    print(f"[C10] 남은 격차 {gap:.6f} / 문턱 {MAGNITUDE_FLOOR} -> "
          f"{MAGNITUDE_FLOOR * gap:.6f}")
    for c in caught:
        v3v = c["v3"]
        mark = "차단" if v3v["blocked_by"] else "통과"
        print(f"[C10] {c['evidence_id']:8s} {c['lane']} 이득 {c['expected_gain']:.6f} "
              f"(격차대비 {c['gain_over_gap']:.3f})  -> {v3v['condition']:4s} {mark}"
              f"   (v2: {v3v['v2_would_have']['condition']})")
    print(f"[C10] C1N68 통과확인: {pass_verdict['condition']} / {pass_verdict['action']}")
    print(f"[C10] V1 {v1} / V2 {v2} ({blocked_count}/4) / V3 {v3}")
    print(f"[C10] 판정: {verdict_code}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
