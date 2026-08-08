"""M271 P4 사이클 24 — 멤버 수 축의 전제 재검 (C9).

사이클 20 이 이 프로젝트의 사각지대를 하나 드러냈다. 결합자를 `mean` 에서 `median` 으로
바꿨는데 **alpha 사다리를 다시 돌리지 않았다.** 다시 돌리자 최적점이 0.50 에서 1.00 으로
옮겨갔다. 결합자와 alpha 는 독립이 아니었다.

같은 의심이 하나 더 있다. 사이클 13 은 "멤버 12 개(E1)가 4 개(E3)보다 나쁘다" 며 멤버 수
축을 닫았다. 그 판정은 **전부 `mean` 하에서** 났다. mean 은 이상치 하나에 끌려다니므로
멤버를 늘리면 나쁜 멤버가 섞일 위험이 그대로 반영된다. **median 은 그렇지 않다** — 12 개의
median 은 4 개의 median 보다 구조적으로 더 안정적이다.

C9(전제 뒤집힘) 조건이다. 폐기 전제가 특정 결합자에 매여 있었고 그 결합자가 교체됐다.

① 방법 리서치 (실행 전)
  - 새 방법 없음. 사이클 13 이 **실행 전에 동결한** 앙상블 집합 3 개를 그대로 재사용한다.
    새 집합을 지금 만들면 이 프로젝트가 반복 기각해 온 same-fold 선택이 된다.
  - 강건 통계의 기본 사실이 근거다: 표본 중앙값의 붕괴점은 50% 로 표본 크기와 무관하지만,
    **분산**은 표본이 커질수록 줄어든다. 평균은 붕괴점이 0 이라 나쁜 멤버 하나로 무너진다.
    따라서 "멤버 증가" 의 효과 부호가 두 결합자에서 다를 선험적 이유가 있다.

② 사양 동결

  사이클 13 의 동결 집합만 쓴다.
    E2  M102 + M244                          (2 개)
    E3  M102 + M244 + DART + XGBoost         (4 개, 현 승격후보)
    E1  가용 12 개 전부                       (12 개)
  결합자는 `median` 고정, blend 없음(alpha=1, 사이클 20 판정).

  **승격 규칙 (실행 전 동결)**
    R1  `M271_MEDIAN4` 대비 pooled Total 개선.
    R2  `M271_MEDIAN4` 를 부모로 한 **동결 게이트 통과**.
    R3  `mean` 하에서도 같은 방향인지는 **묻지 않는다**. 이 노드의 주장은 결합자에 따라
        부호가 다르다는 것이므로, mean 결과는 대조군으로 **함께 보고**하되 승격 조건에
        넣지 않는다.
  둘 다 만족하는 집합이 복수면 **멤버 수가 많은 쪽**을 택한다(분산감소가 기전이므로).

  사전확약(실행 전 동결):
    H1  `median` 하에서 E1(12 개)이 E3(4 개)보다 낫다. — 사이클 13 판정의 부호 역전
    H2  `mean` 하에서는 여전히 E1 이 E3 보다 나쁘다. — 사이클 13 재현
    H1·H2 가 **둘 다** 성립해야 "결합자 의존" 주장이 선다. H1 만 성립하면 단순 개선이고,
    H2 만 성립하면 사이클 13 이 옳고 이 노드는 헛짚은 것이다.

**게이트를 수정하지 않는다.** 읽기만 한다.

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

from m270_gate import GATE_VERSION, evaluate_gate
from m271_cycle13_ensemble import ENSEMBLES
from m271_cycle17_combiner import combine, stack_members
from m271_evaluate_candidate import official

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
REPORT_MD = REPORTS / "m271_cycle24_member_count.md"
RECEIPT = REPORTS / "m271_cycle24_member_count_receipt.json"

NODE_ID = "C1N24_MEMBER_COUNT_REOPEN"
LANE = "L7"
PARENT_NODE = "C1N23_ELIGIBLE_MOS"
REOPENS = "C1N13_ENSEMBLE_VARIANCE"
INCUMBENT = "M271_MEDIAN4"
INCUMBENT_SET = "E3_FOUR_FAMILY"
COMBINER = "median"
CONTROL_COMBINER = "mean"
LADDER = ("E2_CLASSIFIER_PLUS_ANALOG", "E3_FOUR_FAMILY", "E1_ALL_EQUAL")


def build(name: str, operator: str):
    members = ENSEMBLES[name]
    stacked = stack_members(members)
    return combine(stacked, len(members), operator), len(members)


def main() -> int:
    incumbent, _ = build(INCUMBENT_SET, COMBINER)
    incumbent_score = official(incumbent)

    rows = []
    for name in LADDER:
        entry: dict[str, Any] = {"ensemble": name}
        for operator in (COMBINER, CONTROL_COMBINER):
            candidate, k = build(name, operator)
            score = official(candidate)
            gate = evaluate_gate(candidate, incumbent)
            stats = gate.evidence
            entry["n_members"] = k
            entry[operator] = {
                **score,
                "delta_vs_incumbent": score["total"] - incumbent_score["total"],
                "gate_passed": bool(gate.passed),
                "gate": {la.split()[0]: bool(ok) for la, ok in gate.conditions.items()},
                "positive_months": int(stats["positive_months"]),
                "months_scored": int(stats["months_scored"]),
                "sign_test_p": float(stats["sign_test_p_greater"]),
                "bootstrap_q05": float(stats["block_bootstrap_q05"]),
            }
        rows.append(entry)

    by_name = {r["ensemble"]: r for r in rows}
    e1, e3 = by_name["E1_ALL_EQUAL"], by_name["E3_FOUR_FAMILY"]
    h1 = bool(e1[COMBINER]["total"] > e3[COMBINER]["total"])
    h2 = bool(e1[CONTROL_COMBINER]["total"] < e3[CONTROL_COMBINER]["total"])

    eligible = [
        r for r in rows
        if r[COMBINER]["delta_vs_incumbent"] > 0 and r[COMBINER]["gate_passed"]
    ]
    chosen = max(eligible, key=lambda r: r["n_members"]) if eligible else None
    promoted_total = chosen[COMBINER]["total"] if chosen else incumbent_score["total"]

    verdict = (
        f"MEMBER_COUNT_REOPENED_PROMOTE_{chosen['ensemble']}" if chosen
        else "MEMBER_COUNT_STAYS_CLOSED"
    )
    check = {
        "H1_expectation": "median 하에서 E1(12) > E3(4) — 사이클 13 부호 역전",
        "H1_held": h1,
        "H2_expectation": "mean 하에서는 여전히 E1(12) < E3(4) — 사이클 13 재현",
        "H2_held": h2,
        "combiner_dependence_established": bool(h1 and h2),
        "promotion_rule_frozen_before_run": [
            "R1 Total 개선", "R2 동결 게이트 통과(부모=M271_MEDIAN4)",
            "복수 자격시 멤버 수 많은 쪽",
        ],
        "chosen": chosen["ensemble"] if chosen else None,
        "verdict": verdict,
    }

    payload = {
        "node": NODE_ID,
        "parent_node": PARENT_NODE,
        "reopens": REOPENS,
        "gate_version": GATE_VERSION,
        "gate_modified": False,
        "incumbent": INCUMBENT,
        "incumbent_score": incumbent_score,
        "ladder": rows,
        "predeclared_check": check,
        "promoted_total": promoted_total,
        "gap_to_target": 0.66 - promoted_total,
    }

    lines = [
        "# M271 P4 사이클 24 — 멤버 수 축의 전제 재검",
        "",
        f"- 판정일: {datetime.now(UTC).strftime('%Y-%m-%d')} (UTC)",
        f"- 노드: `{NODE_ID}` / 레인 {LANE} / 부모 `{PARENT_NODE}`",
        f"- 재검 대상: `{REOPENS}` (사이클 13 — 멤버 수 축을 `mean` 하에서 닫았다)",
        f"- 기존 승격후보 `{INCUMBENT}` Total **{incumbent_score['total']:.6f}**",
        f"- 게이트: `{GATE_VERSION}` (읽기만 함)",
        "",
        "## 1. 왜 다시 여는가",
        "",
        "사이클 20 에서 결합자를 바꾸자 alpha 최적점이 0.50 -> 1.00 으로 옮겨갔다. 결합자와",
        "alpha 는 독립이 아니었다. 멤버 수도 마찬가지일 수 있다 — 표본 중앙값의 붕괴점은",
        "50% 로 표본 크기와 무관하지만 **분산은 표본이 클수록 준다**. 평균은 붕괴점이 0 이라",
        "나쁜 멤버 하나로 무너진다. 두 결합자에서 부호가 다를 선험적 이유가 있다.",
        "",
        "집합은 사이클 13 이 **실행 전에 동결한** 3 개를 그대로 쓴다. 지금 새 집합을 만들면",
        "same-fold 선택이 된다.",
        "",
        "## 2. 사다리 — 두 결합자 나란히",
        "",
        "| 집합 | 멤버 | 결합자 | Total | 1-NMAE | FICR | 기존대비 | G1G2G3G4 | 양수월 | q05 "
        "| 게이트 |",
        "|---|---:|---|---:|---:|---:|---:|:---:|---:|---:|:---:|",
    ]
    for r in rows:
        for operator in (COMBINER, CONTROL_COMBINER):
            d = r[operator]
            flags = "".join("O" if d["gate"].get(x) else "-" for x in ("G1", "G2", "G3", "G4"))
            mark = "**" if operator == COMBINER else ""
            lines.append(
                f"| {r['ensemble']} | {r['n_members']} | {mark}`{operator}`{mark} | "
                f"{mark}{d['total']:.6f}{mark} | {d['one_minus_nmae']:.6f} | {d['ficr']:.6f} | "
                f"{d['delta_vs_incumbent']:+.6f} | `{flags}` | "
                f"{d['positive_months']}/{d['months_scored']} | {d['bootstrap_q05']:+.6f} | "
                f"{'**통과**' if d['gate_passed'] else '기각'} |"
            )

    lines += [
        "",
        "## 3. 사전확약 대조",
        "",
        f"- H1 `{check['H1_expectation']}` -> **{h1}** "
        f"(E1 {e1[COMBINER]['total']:.6f} vs E3 {e3[COMBINER]['total']:.6f})",
        f"- H2 `{check['H2_expectation']}` -> **{h2}** "
        f"(E1 {e1[CONTROL_COMBINER]['total']:.6f} vs E3 {e3[CONTROL_COMBINER]['total']:.6f})",
        "",
        f"결합자 의존 성립: **{check['combiner_dependence_established']}**",
        "",
        f"판정: **{verdict}**",
        "",
        f"승격 Total **{promoted_total:.6f}**, 목표 0.66 까지 **{0.66 - promoted_total:+.6f}**.",
        "",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    receipt: dict[str, Any] = {
        "schema_version": 1,
        "stage": "M271_P4_CYCLE24_MEMBER_COUNT",
        "node": NODE_ID,
        "lane": LANE,
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

    for r in rows:
        for operator in (COMBINER, CONTROL_COMBINER):
            d = r[operator]
            print(f"[C24] {r['ensemble']:>26} k={r['n_members']:>2} {operator:>6}  "
                  f"Total {d['total']:.6f}  기존대비 {d['delta_vs_incumbent']:+.6f}  "
                  f"게이트 {'통과' if d['gate_passed'] else '기각'}")
    print(f"[C24] H1 median 부호역전 {h1} | H2 mean 재현 {h2} | "
          f"결합자 의존 {check['combiner_dependence_established']}")
    print(f"[C24] 판정: {verdict}  ->  Total {promoted_total:.6f} "
          f"(목표까지 {0.66 - promoted_total:+.6f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
