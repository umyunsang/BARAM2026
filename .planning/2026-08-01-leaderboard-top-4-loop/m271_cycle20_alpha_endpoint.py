"""M271 P4 사이클 20 — alpha 사다리의 끝점. 배포 정책은 섞을 가치가 있는가.

사이클 19 가 참조값으로 찍은 숫자 하나가 앞뒤가 맞지 않았다. 순수 `median` 앙상블 단독이
`0.636597` 인데 사이클 17 의 블렌드(`0.5 x 배포 + 0.5 x median 앙상블`)는 `0.634310` 이다.
섞는 쪽이 더 나쁘다.

사이클 14 는 `mean` 결합자로 alpha 사다리 {0.25, 0.5, 0.75} 를 돌렸고 **alpha=1.0 을
넣지 않았다**. 배포 정책을 섞는다는 것이 당시엔 자명한 전제였기 때문이다. 강건 결합자로
바뀐 지금 그 전제를 다시 재야 한다.

① 방법 리서치 (실행 전)
  - 이 노드의 과업은 새 방법을 찾는 것이 아니라 **동결된 사다리 하나를 끝까지 미는 것**
    이다. 새 방법 리서치 없음. 사이클 17 의 결합자 근거(Elliott & Timmermann 2004)와
    사이클 13~14 의 사다리 설계를 그대로 재사용한다.
  - 대신 **선택 편향 방어**가 이 노드의 방법론적 쟁점이다. 사다리 다섯 점을 다 보고
    최고점을 고르면 같은 fold 선택이다. 그래서 승격 규칙을 **실행 전에** 동결한다.

② 사양 동결

  사다리: alpha in {0.00, 0.25, 0.50, 0.75, 1.00}. alpha=0 은 배포 정책 그 자체다.
  결합자는 사이클 17 이 고른 `median` 로 고정, 멤버 4 개 고정.

  **승격 규칙 (실행 전 동결)** — 최고점을 고르지 않는다. 다음을 모두 만족할 때만
  `alpha=1.00` 을 승격한다.
    R1  Total 이 alpha 에 대해 **단조 증가**한다. 끝점이 추세의 끝이지 튀는 점이 아님.
    R2  alpha=1.00 이 **동결 게이트를 통과**한다 (부모 = 배포 정책).
    R3  alpha=1.00 이 alpha=0.50(현 승격 후보) 대비로도 **게이트를 통과**한다.
  하나라도 어긋나면 승격하지 않고 사이클 16 이 감사한 alpha=0.50 을 유지한다.

  사전확약(실행 전 동결):
    H1  R1 단조성이 성립한다.
    H2  R2 가 성립한다.
    H3  R3 가 성립한다.
  예상: 사이클 19 의 참조값이 우연이 아니라면 셋 다 성립한다. H1 이 기각되면 사이클 19 의
  숫자는 사다리의 국소 요철이고 승격하지 않는다.

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

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from m270_gate import GATE_VERSION, evaluate_gate
from m270_monthly_validation import load_predictions
from m271_cycle13_ensemble import ENSEMBLES
from m271_cycle14_shrinkblend import blend
from m271_cycle17_combiner import combine, stack_members
from m271_evaluate_candidate import official

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
REPORT_MD = REPORTS / "m271_cycle20_alpha_endpoint.md"
RECEIPT = REPORTS / "m271_cycle20_alpha_endpoint_receipt.json"

NODE_ID = "C1N20_ALPHA_ENDPOINT"
LANE = "L7"
PARENT_NODE = "C1N19_ENVELOPE_ORACLE"
SUPERSEDES_CANDIDATE = "M271_SHRINKBLEND_A05"
BASE_ENSEMBLE = "E3_FOUR_FAMILY"
COMBINER = "median"
DEPLOYED = "T0.5_G1.5"
LADDER = (0.00, 0.25, 0.50, 0.75, 1.00)
INCUMBENT_ALPHA = 0.50
ENDPOINT_ALPHA = 1.00


def gate_row(candidate, reference) -> dict[str, Any]:
    gate = evaluate_gate(candidate, reference)
    stats = gate.evidence
    return {
        "passed": bool(gate.passed),
        "flags": {label.split()[0]: bool(ok) for label, ok in gate.conditions.items()},
        "positive_months": int(stats["positive_months"]),
        "months_scored": int(stats["months_scored"]),
        "sign_test_p": float(stats["sign_test_p_greater"]),
        "median_delta": float(stats["median_total_delta"]),
        "bootstrap_q05": float(stats["block_bootstrap_q05"]),
        "min_delta": float(stats["min_total_delta"]),
    }


def main() -> int:
    members = ENSEMBLES[BASE_ENSEMBLE]
    k = len(members)
    stacked = stack_members(members)
    ensemble = combine(stacked, k, COMBINER)
    parent = load_predictions(DEPLOYED)
    parent_score = official(parent)

    built = {a: blend(parent, ensemble, a) for a in LADDER}
    incumbent = built[INCUMBENT_ALPHA]

    rows = []
    for a in LADDER:
        candidate = built[a]
        score = official(candidate)
        rows.append(
            {
                "alpha": a,
                **score,
                "delta_vs_deployed": score["total"] - parent_score["total"],
                "gate_vs_deployed": gate_row(candidate, parent),
                "gate_vs_incumbent": (
                    gate_row(candidate, incumbent) if a != INCUMBENT_ALPHA else None
                ),
            }
        )

    totals = [r["total"] for r in rows]
    monotone = bool(np.all(np.diff(totals) > 0))
    endpoint = next(r for r in rows if r["alpha"] == ENDPOINT_ALPHA)
    r1 = monotone
    r2 = bool(endpoint["gate_vs_deployed"]["passed"])
    r3 = bool(endpoint["gate_vs_incumbent"]["passed"])
    promote = bool(r1 and r2 and r3)

    check = {
        "R1_expectation": "Total 이 alpha 에 대해 단조 증가",
        "R1_held": r1,
        "R2_expectation": "alpha=1.00 이 배포 대비 동결 게이트 통과",
        "R2_held": r2,
        "R3_expectation": "alpha=1.00 이 현 승격후보(alpha=0.50) 대비로도 게이트 통과",
        "R3_held": r3,
        "promote_endpoint": promote,
        "verdict": (
            "ENDPOINT_PROMOTED_BLEND_UNNECESSARY" if promote
            else "ENDPOINT_REJECTED_KEEP_ALPHA_050"
        ),
    }

    best_total = endpoint["total"] if promote else rows[LADDER.index(INCUMBENT_ALPHA)]["total"]
    payload = {
        "node": NODE_ID,
        "parent_node": PARENT_NODE,
        "supersedes_if_promoted": SUPERSEDES_CANDIDATE,
        "gate_version": GATE_VERSION,
        "gate_modified": False,
        "combiner": COMBINER,
        "members": list(members),
        "ladder": list(LADDER),
        "parent": {"policy": DEPLOYED, **parent_score},
        "rungs": rows,
        "promotion_rule_frozen_before_run": [
            check["R1_expectation"], check["R2_expectation"], check["R3_expectation"]
        ],
        "predeclared_check": check,
        "promoted_total": best_total,
        "gap_to_target": 0.66 - best_total,
    }

    lines = [
        "# M271 P4 사이클 20 — alpha 사다리의 끝점",
        "",
        f"- 판정일: {datetime.now(UTC).strftime('%Y-%m-%d')} (UTC)",
        f"- 노드: `{NODE_ID}` / 레인 {LANE} / 부모 `{PARENT_NODE}`",
        f"- 결합자 `{COMBINER}` 고정, 멤버 {k} 개 고정. **alpha 만** 움직인다.",
        f"- 게이트: `{GATE_VERSION}` (읽기만 함)",
        "",
        "## 0. 승격 규칙 — 실행 **전에** 동결",
        "",
        "사다리 다섯 점을 다 보고 최고점을 고르면 같은 fold 선택이다. 그래서 끝점 승격에",
        "세 조건을 미리 걸었다.",
        "",
        f"- R1 {check['R1_expectation']}",
        f"- R2 {check['R2_expectation']}",
        f"- R3 {check['R3_expectation']}",
        "",
        "## 1. 사다리",
        "",
        "`alpha=0` 은 배포 정책 그 자체, `alpha=1` 은 앙상블 단독이다.",
        "",
        "| alpha | Total | 1-NMAE | FICR | 배포대비 | G1G2G3G4 | 양수월 | p | q05 | 게이트 |",
        "|---:|---:|---:|---:|---:|:---:|---:|---:|---:|:---:|",
    ]
    for r in rows:
        g = r["gate_vs_deployed"]
        flags = "".join("O" if g["flags"].get(x) else "-" for x in ("G1", "G2", "G3", "G4"))
        mark = "**" if r["alpha"] == ENDPOINT_ALPHA else ""
        lines.append(
            f"| {mark}{r['alpha']:.2f}{mark} | {mark}{r['total']:.6f}{mark} | "
            f"{r['one_minus_nmae']:.6f} | {r['ficr']:.6f} | {r['delta_vs_deployed']:+.6f} | "
            f"`{flags}` | {g['positive_months']}/{g['months_scored']} | "
            f"{g['sign_test_p']:.4f} | {g['bootstrap_q05']:+.6f} | "
            f"{'**통과**' if g['passed'] else '기각'} |"
        )

    lines += [
        "",
        f"단조성: **{monotone}** (차분 "
        + ", ".join(f"{d:+.6f}" for d in np.diff(totals))
        + ")",
        "",
        "## 2. 현 승격후보(alpha=0.50) 대비 게이트",
        "",
        "새 후보가 기존 후보를 실제로 대체하는지는 **기존 후보를 부모로 놓고** 재야 한다.",
        "",
        "| alpha | G1G2G3G4 | 양수월 | p | q05 | 최소월 델타 | 게이트 |",
        "|---:|:---:|---:|---:|---:|---:|:---:|",
    ]
    for r in rows:
        g = r["gate_vs_incumbent"]
        if g is None:
            continue
        flags = "".join("O" if g["flags"].get(x) else "-" for x in ("G1", "G2", "G3", "G4"))
        lines.append(
            f"| {r['alpha']:.2f} | `{flags}` | {g['positive_months']}/{g['months_scored']} | "
            f"{g['sign_test_p']:.4f} | {g['bootstrap_q05']:+.6f} | {g['min_delta']:+.6f} | "
            f"{'**통과**' if g['passed'] else '기각'} |"
        )

    lines += [
        "",
        "## 3. 사전확약 대조",
        "",
        f"- R1 `{check['R1_expectation']}` -> **{r1}**",
        f"- R2 `{check['R2_expectation']}` -> **{r2}**",
        f"- R3 `{check['R3_expectation']}` -> **{r3}**",
        "",
        f"판정: **{check['verdict']}**",
        "",
        f"승격 Total **{best_total:.6f}**, 목표 0.66 까지 **{0.66 - best_total:+.6f}**.",
        "",
    ]
    if promote:
        lines += [
            "## 4. 이것이 뜻하는 것",
            "",
            "배포 정책 `T0.5_G1.5` 를 섞는 것은 이득이 아니라 **희석**이었다. 사이클 14 가",
            "alpha=0.5 를 고른 것은 `mean` 결합자 하에서였고, 그때는 사다리 안쪽이 최선이라고",
            "볼 근거가 있었다. 강건 결합자로 바꾸자 끝점이 최선이 됐다. 결합자와 alpha 는",
            "독립이 아니었다.",
        ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    receipt: dict[str, Any] = {
        "schema_version": 1,
        "stage": "M271_P4_CYCLE20_ALPHA_ENDPOINT",
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
        g = r["gate_vs_deployed"]
        print(f"[C20] alpha={r['alpha']:.2f}  Total {r['total']:.6f}  "
              f"배포대비 {r['delta_vs_deployed']:+.6f}  "
              f"게이트 {'통과' if g['passed'] else '기각'}")
    print(f"[C20] R1 단조 {r1} | R2 배포대비게이트 {r2} | R3 기존후보대비게이트 {r3}")
    print(f"[C20] 판정: {check['verdict']}  ->  Total {best_total:.6f} "
          f"(목표까지 {0.66 - best_total:+.6f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
