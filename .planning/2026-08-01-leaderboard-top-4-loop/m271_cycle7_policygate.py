"""M271 P4 사이클 7 — 배포 정책 대안을 동결 월별 게이트로 판정한다.

사이클 4 가 전역최적 `T0.6_G0.2` 가 배포 `T0.5_G1.5` 를 pooled 로 `+0.001041` 앞선다고
측정했다. 그러나 그것은 **same-fold 선택**이었다. `m270_gate.py` 는 정확히 이런 후보를
판정하라고 2026-08-04 에 동결된 도구다.

이 노드는 63 개 정책 전부를 배포 정책 대비 **동결 게이트**에 통과시킨다.

  G1  월별 부호검정 one-sided p <= 0.10
  G2  월별 Total 델타 중앙값 > 0
  G3  블록 부트스트랩 5% 분위 > 0
  G4  최악월 델타 >= -0.010

**게이트를 수정하지 않는다.** `m270_gate.py` 는 재동결이 금지되어 있고 여기서는 읽기만 한다.
pooled 이득이 커 보여도 게이트가 기각하면 기각이다 — M251 이 pooled `+0.001223` 이었는데도
게이트에서 떨어진 전례가 있다.

사전확약(실행 전 동결):
  H1  게이트를 통과하는 정책이 **하나 이상** 있다.
  기각되면 결정정책 축은 닫힌다. 통과하면 그 후보가 첫 검증된 이득이다.

다중비교 주의: 63 개를 한 게이트에 넣으면 우연 통과가 생긴다. 통과 개수를 함께 보고하고,
G1 의 alpha=0.10 에서 63 개 중 우연 기대치가 몇인지 명시한다.

읽기 전용. 모델을 적합하지 않고 2024 행을 읽지 않는다.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from m270_gate import GATE_VERSION, evaluate_gate
from m270_monthly_validation import PROBE, load_predictions, score_frame

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
REPORT_MD = REPORTS / "m271_cycle7_policygate.md"
RECEIPT = REPORTS / "m271_cycle7_policygate_receipt.json"

NODE_ID = "C1N7_POLICY_FROZEN_GATE"
LANE = "L4"  # 검증 전략 — 게이트로 후보를 판정한다
DEPLOYED = "T0.5_G1.5"


def policy_names() -> list[str]:
    frame = pd.read_parquet(PROBE / "M269_PROBE_TOP100-dev-2023-Q2-policies.parquet")
    return sorted(c for c in frame.columns if c.startswith("T"))


def main() -> int:
    policies = policy_names()
    parent = load_predictions(DEPLOYED)
    parent_pooled = score_frame(parent)

    rows: list[dict[str, Any]] = []
    for policy in policies:
        if policy == DEPLOYED:
            continue
        candidate = load_predictions(policy)
        pooled = score_frame(candidate)
        result = evaluate_gate(candidate, parent)
        stats = result.evidence
        rows.append(
            {
                "policy": policy,
                "pooled_total": pooled["total"],
                "pooled_delta": pooled["total"] - parent_pooled["total"],
                "passed": bool(result.passed),
                # 게이트 서명을 라우터가 읽을 평면 형태로 정규화한다.
                "gate": {
                    label.split()[0]: bool(ok) for label, ok in result.conditions.items()
                },
                "months_scored": int(stats["months_scored"]),
                "positive_months": int(stats["positive_months"]),
                "sign_test_p": float(stats["sign_test_p_greater"]),
                "median_delta": float(stats["median_total_delta"]),
                "bootstrap_q05": float(stats["block_bootstrap_q05"]),
                "min_delta": float(stats["min_total_delta"]),
            }
        )

    rows.sort(key=lambda r: -r["pooled_delta"])
    passed = [r for r in rows if r["passed"]]
    positive_pooled = [r for r in rows if r["pooled_delta"] > 0]

    # 다중비교: G1 의 alpha=0.10 에서 순수 우연으로 기대되는 통과 수의 상한.
    chance_g1 = 0.10 * len(rows)

    check = {
        "H1_expectation": "동결 게이트를 통과하는 정책이 하나 이상 있다",
        "H1_held": bool(passed),
        "candidates_tested": len(rows),
        "pooled_positive": len(positive_pooled),
        "gate_passed": len(passed),
        "chance_expectation_g1_alone": chance_g1,
        "verdict": "VALIDATED_CANDIDATE_FOUND" if passed else "POLICY_AXIS_CLOSED",
    }
    payload = {
        "gate_version": GATE_VERSION,
        "deployed": {"policy": DEPLOYED, "pooled": parent_pooled},
        "results": rows,
        "passed": passed,
        "predeclared_check": check,
    }

    lines = [
        "# M271 P4 사이클 7 — 정책 대안의 동결 게이트 판정",
        "",
        f"- 판정일: {datetime.now(UTC).strftime('%Y-%m-%d')} (UTC)",
        f"- 노드: `{NODE_ID}` / 레인 {LANE}",
        f"- 게이트: `{GATE_VERSION}` (읽기만 함, 재동결 금지)",
        f"- 부모: `{DEPLOYED}` pooled Total {parent_pooled['total']:.6f}",
        "",
        "## 1. 왜 이 노드인가",
        "",
        "사이클 4 가 전역최적 `T0.6_G0.2` 가 배포를 pooled `+0.001041` 앞선다고 측정했으나",
        "그것은 same-fold 선택이었다. 동결 월별 게이트는 정확히 이런 후보를 판정하는 도구다.",
        "",
        "M251 은 pooled `+0.001223` 이었는데도 게이트에서 떨어진 전례가 있다. pooled 이득이",
        "커 보여도 게이트가 기각하면 기각이다.",
        "",
        "## 2. 결과 요약",
        "",
        f"- 시험한 정책 **{len(rows)}** 개",
        f"- pooled 델타가 양수인 정책 **{len(positive_pooled)}** 개",
        f"- **동결 게이트 통과 {len(passed)} 개**",
        f"- G1 alpha=0.10 만으로 순수 우연 기대치 약 {chance_g1:.1f} 개 "
        "(G1~G4 를 모두 요구하므로 실제 우연 통과는 이보다 훨씬 낮다)",
        "",
        "## 3. pooled 델타 상위 12",
        "",
        "| 정책 | pooled | 델타 | G1 | G2 | G3 | G4 | 통과 | 양수월 | p | q05 | 최악월 |",
        "|---|---:|---:|:--:|:--:|:--:|:--:|:--:|---:|---:|---:|---:|",
    ]
    for r in rows[:12]:
        g = r["gate"]
        flags = {k: ("O" if g.get(k) else "-") for k in ("G1", "G2", "G3", "G4")}
        lines.append(
            f"| `{r['policy']}` | {r['pooled_total']:.6f} | {r['pooled_delta']:+.6f} | "
            f"{flags['G1']} | {flags['G2']} | {flags['G3']} | {flags['G4']} | "
            f"{'**통과**' if r['passed'] else '기각'} | "
            f"{r['positive_months']}/{r['months_scored']} | {r['sign_test_p']:.4f} | "
            f"{r['bootstrap_q05']:+.6f} | {r['min_delta']:+.6f} |"
        )

    if passed:
        lines += ["", "## 4. 통과 후보", "", "| 정책 | pooled 델타 | 양수월 | p | q05 | 최악월 |",
                  "|---|---:|---:|---:|---:|---:|"]
        for r in passed:
            lines.append(
                f"| `{r['policy']}` | {r['pooled_delta']:+.6f} | "
                f"{r['positive_months']}/{r['months_scored']} | {r['sign_test_p']:.4f} | "
                f"{r['bootstrap_q05']:+.6f} | {r['min_delta']:+.6f} |"
            )
    else:
        lines += [
            "",
            "## 4. 통과 후보 없음",
            "",
            "pooled 델타가 양수인 정책이 있어도 월별 일관성·효과크기·최악월 조건을 함께",
            "만족하는 것이 없다. 결정정책 축은 닫힌다.",
        ]

    lines += [
        "",
        "## 5. 사전확약 대조",
        "",
        f"- H1 `{check['H1_expectation']}` -> **{check['H1_held']}**",
        "",
        f"판정: **{check['verdict']}**",
        "",
        "## 6. 다중비교",
        "",
        f"{len(rows)} 개를 한 게이트에 넣었다. G1 의 alpha=0.10 만 보면 우연 통과 기대치가",
        f"약 {chance_g1:.1f} 개지만, G1~G4 를 **모두** 요구하므로 실제 우연 통과 확률은 훨씬",
        "낮다. 그래도 통과 후보가 나오면 그 자체로 확정이 아니라 **다음 검증 단계의 입력**이며,",
        "최종 채택은 사용자 판단이다.",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    receipt = {
        "schema_version": 1,
        "stage": "M271_P4_CYCLE7_POLICY_GATE",
        "node": NODE_ID,
        "lane": LANE,
        "gate_version": GATE_VERSION,
        "gate_modified": False,
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

    print(f"[C7] 게이트 {GATE_VERSION}")
    print(f"[C7] 시험 {len(rows)}개 / pooled 양수 {len(positive_pooled)}개 / "
          f"게이트 통과 {len(passed)}개")
    for r in rows[:5]:
        g = r["gate"]
        flags = "".join("O" if g.get(k) else "-" for k in ("G1", "G2", "G3", "G4"))
        print(f"     {r['policy']:12s} 델타={r['pooled_delta']:+.6f} [{flags}] "
              f"{'통과' if r['passed'] else '기각'}")
    print(f"[C7] 판정: {check['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
