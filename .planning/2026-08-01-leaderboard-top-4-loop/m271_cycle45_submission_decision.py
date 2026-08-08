"""M271 P4 사이클 45 — 제출 파이프라인을 지을 가치가 있는가.

파이프라인을 짜기 전에 이 프로젝트의 규율을 그대로 적용한다: **기대값부터 잰다.**

사이클 35 는 고정정책 580 조합을 **배포 기준선 `M269@T0.5_G1.5` = 0.628605** 와 겨뤘다.
그런데 진짜 현직은 그것이 아니다. **이미 온라인에 제출된 M261 의 로컬이 0.629973** 이고
(`m270_second_anchor_receipt.json`, 표면 "2023 Q2-Q4 chronology-safe OOF" — 우리와 동일),
온라인 0.636527 을 받았다.

즉 사이클 35 의 기준선이 **현직보다 낮았다.** 사이클 33 에서 챔피언을 배포 정책하고만
겨뤘던 것과 **같은 오류 패턴**이다. 새 제출은 배포 기준선이 아니라 **이미 제출된 것**을
이겨야 의미가 있다.

이 노드는 그 대조를 하고 제출 파이프라인 착수 여부를 판정한다. **새 학습 없음.**

① 방법 리서치 (실행 전)
  - 새 방법 없음. 사이클 35 의 receipt 를 읽어 기준선만 바꿔 다시 순위를 매긴다.
  - 판정 도구는 사이클 9 가 실측한 **게이트 검출문턱 `+0.001013`** 이다. 그보다 작은
    차이는 동결 게이트가 구별하지 못하므로, 그 차이를 근거로 제출을 바꾸는 것은
    측정되지 않은 변화를 배포하는 것이다.

② 사양 동결

  현직   `M261` 로컬 **0.629973** (온라인 0.636527, 오프셋 +0.006554)
  후보   사이클 35 가 시험한 고정정책 조합 전부 (`m271_cycle35_fixed_policy_receipt.json`)
  문턱   `+0.001013` (사이클 9 실측)

  사전확약(실행 전 동결):
    H1  사이클 35 의 자격 후보(`M115_XGBOOST@T0.6_G0.2`, 0.630310)가 **현직 대비**로도
        검출문턱을 넘는다.
    H2  전 조합 중 현직을 검출문턱 이상 넘는 것이 **적어도 하나** 있다.
    H3  그런 후보가 있다면 그것이 사이클 35 에서 **게이트도 통과**했다.

  H1 이 기각되고 H2 도 기각되면 **제출 파이프라인 착수는 정당화되지 않는다.** 이미 제출된
  것과 구별되지 않는 산출물을 만드는 데 남은 시간을 쓰지 않는다.
  H2·H3 가 성립하면 그 후보를 지목하고 착수한다.

**새 학습·게이트 수정·lockbox 사용 없음.**
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
C35_RECEIPT = REPORTS / "m271_cycle35_fixed_policy_receipt.json"
REPORT_MD = REPORTS / "m271_cycle45_submission_decision.md"
RECEIPT = REPORTS / "m271_cycle45_submission_decision_receipt.json"

NODE_ID = "C1N45_SUBMISSION_DECISION"
LANE = "L5"
PARENT_NODE = "C1N44_SHARPENED_DECISION"
CORRECTS_BASELINE_OF = "C1N35_FIXED_POLICY_CORRECTION"

INCUMBENT = {
    "name": "M261_FULL_HISTORY_STRICT_TEMPORAL_TOP100",
    "local": 0.629973,
    "online": 0.6365274327,
    "offset": 0.006554,
    "surface": "2023 Q2-Q4 chronology-safe OOF",
    "source": "reports/m270_second_anchor_receipt.json",
}
DEPLOYED_BASELINE = 0.628605
GATE_DETECTION_THRESHOLD = 0.001013
C35_QUALIFIER = {"name": "M115_XGBOOST@T0.6_G0.2", "total": 0.630310}


def main() -> int:
    c35 = json.loads(C35_RECEIPT.read_text(encoding="utf-8"))["result"]
    rows = c35["top15"]
    qualified = c35["qualified"]

    incumbent_local = INCUMBENT["local"]
    ranked = []
    for r in rows:
        margin = r["total"] - incumbent_local
        ranked.append(
            {
                "model": r["model"], "policy": r["policy"], "total": r["total"],
                "margin_vs_incumbent": margin,
                "above_detection": bool(margin > GATE_DETECTION_THRESHOLD),
                "gate_passed_vs_deployed": r["gate_passed"],
                "fold_wins_vs_deployed": r["fold_wins"],
            }
        )
    ranked.sort(key=lambda x: -x["margin_vs_incumbent"])

    q_margin = C35_QUALIFIER["total"] - incumbent_local
    h1 = bool(q_margin > GATE_DETECTION_THRESHOLD)
    clearing = [r for r in ranked if r["above_detection"]]
    h2 = bool(clearing)
    h3 = bool(clearing and all(r["gate_passed_vs_deployed"] for r in clearing[:1]))

    build = bool(h2 and h3)
    verdict = (
        "BUILD_SUBMISSION_CANDIDATE_IDENTIFIED" if build
        else ("CLEARS_THRESHOLD_BUT_GATE_REJECTED" if h2
              else "NO_CANDIDATE_BEATS_INCUMBENT_DO_NOT_BUILD")
    )

    check = {
        "H1_expectation": f"사이클 35 자격 후보가 현직({incumbent_local}) 대비 "
                          f"{GATE_DETECTION_THRESHOLD} 초과",
        "H1_held": h1, "H1_margin": q_margin,
        "H2_expectation": "현직을 검출문턱 이상 넘는 조합이 적어도 하나",
        "H2_held": h2, "H2_count": len(clearing),
        "H3_expectation": "그 후보가 사이클 35 게이트도 통과",
        "H3_held": h3,
        "build_justified": build,
        "verdict": verdict,
    }

    payload = {
        "node": NODE_ID, "parent_node": PARENT_NODE,
        "corrects_baseline_of": CORRECTS_BASELINE_OF,
        "baseline_error": "사이클 35 는 배포 기준선(0.628605)과 겨뤘으나 진짜 현직은 "
                          f"이미 제출된 M261 의 로컬 {incumbent_local} 이다. "
                          "사이클 33 과 같은 오류 패턴",
        "incumbent": INCUMBENT,
        "deployed_baseline": DEPLOYED_BASELINE,
        "detection_threshold": GATE_DETECTION_THRESHOLD,
        "cycle35_qualifier": {**C35_QUALIFIER, "margin_vs_incumbent": q_margin},
        "reranked_top15": ranked,
        "clearing_detection_threshold": clearing,
        "cycle35_qualified_count": len(qualified),
        "no_training": True, "gate_touched": False, "lockbox_used": False,
        "predeclared_check": check,
    }

    lines = [
        "# M271 P4 사이클 45 — 제출 파이프라인을 지을 가치가 있는가",
        "",
        f"- 판정일: {datetime.now(UTC).strftime('%Y-%m-%d')} (UTC)",
        f"- 노드: `{NODE_ID}` / 레인 {LANE} / 부모 `{PARENT_NODE}`",
        f"- 기준선 정정 대상: `{CORRECTS_BASELINE_OF}`",
        "- **새 학습 없음.** 사이클 35 의 receipt 를 기준선만 바꿔 다시 읽는다",
        "",
        "## 1. 기준선 오류",
        "",
        payload["baseline_error"] + ".",
        "",
        "| 대상 | 로컬 | 비고 |",
        "|---|---:|---|",
        f"| 배포 기준선 `M269@T0.5_G1.5` | {DEPLOYED_BASELINE:.6f} | 사이클 35 가 쓴 것 |",
        f"| **현직 `{INCUMBENT['name']}`** | **{incumbent_local:.6f}** | "
        f"온라인 {INCUMBENT['online']:.6f} (오프셋 +{INCUMBENT['offset']:.6f}) |",
        "",
        f"현직의 표면은 `{INCUMBENT['surface']}` 로 우리와 동일하다 "
        f"(출처 `{INCUMBENT['source']}`).",
        "",
        "## 2. 현직 기준 재순위",
        "",
        f"검출문턱 **{GATE_DETECTION_THRESHOLD:+.6f}** (사이클 9 실측). 그보다 작은 차이는",
        "동결 게이트가 구별하지 못한다.",
        "",
        "| 모델 | 정책 | Total | **현직 대비** | 문턱초과 | 35 게이트 | fold |",
        "|---|---|---:|---:|:---:|:---:|:---:|",
    ]
    for r in ranked[:10]:
        lines.append(
            f"| `{r['model']}` | `{r['policy']}` | {r['total']:.6f} | "
            f"**{r['margin_vs_incumbent']:+.6f}** | "
            f"{'O' if r['above_detection'] else '**X**'} | "
            f"{'통과' if r['gate_passed_vs_deployed'] else '기각'} | "
            f"{r['fold_wins_vs_deployed']}/3 |"
        )
    lines += [
        "",
        f"사이클 35 의 유일 자격 후보 `{C35_QUALIFIER['name']}` "
        f"({C35_QUALIFIER['total']:.6f})는 현직 대비 **{q_margin:+.6f}** — "
        f"문턱의 {q_margin / GATE_DETECTION_THRESHOLD:.0%} 에 불과하다.",
        "",
        "## 3. 사전확약 대조",
        "",
        f"- H1 `{check['H1_expectation']}` -> **{h1}** ({q_margin:+.6f})",
        f"- H2 `{check['H2_expectation']}` -> **{h2}** ({len(clearing)} 개)",
        f"- H3 `{check['H3_expectation']}` -> **{h3}**",
        "",
        f"판정: **{verdict}**",
        "",
        f"제출 파이프라인 착수 정당화: **{build}**",
        "",
    ]
    if not build:
        lines += [
            "## 4. 왜 짓지 않는가",
            "",
            "새 제출물은 **이미 제출된 것**을 이겨야 의미가 있다. 문턱을 넘지 못하는 차이는",
            "동결 게이트가 구별하지 못하는 크기이고, 그것을 근거로 제출을 바꾸는 것은",
            "**측정되지 않은 변화를 배포하는 것**이다. 오프셋 산포(+0.0066~+0.0211)가",
            "그 차이의 20 배 이상이라는 점도 같은 방향을 가리킨다.",
            "",
            "남은 제출 관련 정당한 행동은 하나다: 현직 `M261` 이 이미 온라인에 올라가 있고",
            "로컬·온라인 앵커가 둘 다 기록돼 있으므로, **추가 제출 없이 현 상태를 유지**하는 것.",
        ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    receipt: dict[str, Any] = {
        "schema_version": 1,
        "stage": "M271_P4_CYCLE45_SUBMISSION_DECISION",
        "node": NODE_ID, "lane": LANE,
        "decided_utc": datetime.now(UTC).isoformat(),
        "result": payload,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "report_sha256": hashlib.sha256(REPORT_MD.read_bytes()).hexdigest(),
        "dacon_upload": False, "external_actions": [], "model_fits": 0,
        "lockbox_reopened": False, "new_2024_evaluation": False,
    }
    RECEIPT.write_text(
        json.dumps(receipt, indent=1, ensure_ascii=False, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

    print(f"[C45] 현직 {INCUMBENT['name']} 로컬 {incumbent_local:.6f} "
          f"(온라인 {INCUMBENT['online']:.6f})")
    print(f"[C45] 사이클 35 자격후보 {C35_QUALIFIER['total']:.6f} -> "
          f"현직 대비 {q_margin:+.6f} (문턱 {GATE_DETECTION_THRESHOLD:+.6f}) -> H1 {h1}")
    for r in ranked[:5]:
        print(f"[C45]   {r['model']:<22} {r['policy']:>12} {r['total']:.6f} "
              f"현직대비 {r['margin_vs_incumbent']:+.6f} "
              f"문턱초과 {r['above_detection']} 게이트 {r['gate_passed_vs_deployed']}")
    print(f"[C45] H1 {h1} | H2 {h2} ({len(clearing)}) | H3 {h3}")
    print(f"[C45] 판정: {verdict}  착수 {build}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
