"""M271 — C16 크기게이트 재검: 선언 관례를 측정값으로 바꾸고 합산을 본다.

C16 은 `expected_gain < 0.15 * remaining_gap` 이면 방향을 기각한다. **0.15 는 선언
관례**이고 보정된 바 없다(C1N78 이 그렇게 기록했다). 지금 증거 여섯 건이 전부 기각·종료로
갔으므로 그 값을 의심할 때다.

**설계 결함 둘.**

  (가) **개별 판정 대 합산 목표.** C16 은 방향을 하나씩 본다. 그런데 완료 기준은 합이다.
       다섯 방향이 각각 +0.002 면 합 +0.010 으로 격차의 1/3 인데 C16 은 다섯 다 기각한다.
       방향들이 서로 **대체재**라고 가정한 것인데 근거가 없다.
  (나) **근거 없는 상수.** 0.15 는 아무것도 재지 않은 값이다. 반면 이 프로젝트에는
       **측정된** 문턱이 있다 — C1N9 가 동결 게이트의 검출문턱을 **0.001013** 으로 쟀다.
       그 아래면 이득이 실재해도 게이트가 구분하지 못한다.

**교정 사양 — 두 문턱으로 나눈다**

  F1  **검출 가능성** (측정값). `expected_gain >= 0.001013`. 이보다 작으면 게이트가
      구분하지 못하므로 승격 자체가 불가능하다. 이것은 선언이 아니라 C1N9 의 계측이다.
  F2  **도달 가능성** (포트폴리오). 살아남은 방향들의 **합**이 남은 격차에 닿는가.
      개별 방향에 거는 것이 아니라 집합에 건다.

  **문턱을 낮추는 것이 아니라 성질을 바꾸는 것**이다. F1(0.001013)은 0.15 x 격차
  (0.004453)보다 낮지만, F2 가 새로 생겨 "작은 것 여럿" 이 실제로 합쳐지는지를 강제한다.
  둘 중 하나만 있으면 각각 다른 방식으로 틀린다.

  **결과를 보고 값을 고르지 않는다.** F1 은 C1N9 의 측정값을 그대로 쓰고, F2 는
  "합이 격차에 닿는가" 라는 산술이지 조정 가능한 상수가 아니다.

**사전확약**

  H1  F1 로 바꾸면 **C1N71(최적가중 +0.001238)이 살아난다**. 0.15 문턱(0.004453)이
      검출문턱 위의 유일한 양수 후보를 잘랐다면 그것이 C16 의 실제 피해다.
  H2  F1 을 통과하는 방향들의 **합이 남은 격차에 못 미친다**. 즉 문턱을 고쳐도
      결론이 안 바뀐다 — 그러면 폐쇄 판정이 문턱 선택의 인공물이 아님이 확인된다.
  H3  C1N84(teacher 용량 시간분할 +0.000160)는 F1 에서도 기각된다. 검출문턱의
      1/6 이므로 어떤 기준에서도 살릴 수 없다.
  H4  음수 방향(C1N77 -0.00092, C1N80 -0.00248)은 어떤 문턱에서도 기각된다.

  H1 이 참이고 H2 도 참이면 결론은 이렇게 갈린다 — **C16 은 실제로 한 후보를 잘못
  잘랐고(고쳐야 한다), 그럼에도 폐쇄 판정 자체는 문턱과 무관하게 성립한다.** 둘을
  분리해서 보고하는 것이 이 노드의 목적이다.

**진단·엔진 개선 전용.** 모델 미변경. 게이트 미수정. 제출 없음.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from m271_c10_loop_engine import MAGNITUDE_FLOOR

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
REPORT_MD = REPORTS / "m271_c16_recheck.md"
RECEIPT = REPORTS / "m271_c16_recheck_receipt.json"

NODE_ID = "C1N85_C16_RECHECK"
LANE = "L8"
PARENT_NODE = "C1N78_LOOP_ENGINE_C10"

DETECTION_THRESHOLD = 0.001013  # C1N9 측정값
CHAMPION_LOCAL = 0.630310
TARGET = 0.66
REMAINING_GAP = TARGET - CHAMPION_LOCAL
OLD_FLOOR = MAGNITUDE_FLOOR * REMAINING_GAP

# 측정된 방향들. 전부 receipt 에 근거가 있고 여기서 지어내지 않는다.
DIRECTIONS: tuple[dict[str, Any], ...] = (
    {"node": "C1N71_TEACHER_WEIGHT", "lane": "L2", "gain": 0.001238,
     "basis": "최적가중 fold-외, C1N69 곡선 환산", "surface": "풍속 잔차층"},
    {"node": "C1N84_TEACHER_CHRONOLOGICAL", "lane": "L6", "gain": 0.000160,
     "basis": "deep 255 잎, 시간분할 test 행", "surface": "test 행"},
    {"node": "C1N70_BASELINE_CORRECTION", "lane": "L2", "gain": 0.000098,
     "basis": "ECMWF 결합, teacher 기준선 교정 후", "surface": "풍속 잔차층"},
    {"node": "C1N77_PER_SOURCE_STACK", "lane": "L3", "gain": -0.000921,
     "basis": "소스 분리 스태킹", "surface": "개발 fold"},
    {"node": "C1N80_FUSION_ANOMALY", "lane": "L3", "gain": -0.002480,
     "basis": "정보량 맞춘 소스 분리", "surface": "개발 fold"},
    {"node": "C1N76_CIRCULAR_BLOCK", "lane": "L4", "gain": 0.0,
     "basis": "결정층 효과가 0 과 구분 불가", "surface": "일별 원형블록"},
)


def main() -> int:
    rows: list[dict[str, Any]] = []
    for d in DIRECTIONS:
        gain = float(d["gain"])
        rows.append({
            **d,
            "passes_old_c16": gain >= OLD_FLOOR,
            "passes_f1_detection": gain >= DETECTION_THRESHOLD,
            "gain_over_gap": gain / REMAINING_GAP,
        })

    survivors_old = [r for r in rows if r["passes_old_c16"]]
    survivors_f1 = [r for r in rows if r["passes_f1_detection"]]
    sum_f1 = sum(float(r["gain"]) for r in survivors_f1)
    reachable = bool(sum_f1 >= REMAINING_GAP)

    revived = [
        r["node"] for r in rows
        if r["passes_f1_detection"] and not r["passes_old_c16"]
    ]
    h1 = bool("C1N71_TEACHER_WEIGHT" in revived)
    h2 = bool(not reachable)
    h3 = bool(not next(
        r for r in rows if r["node"] == "C1N84_TEACHER_CHRONOLOGICAL"
    )["passes_f1_detection"])
    h4 = bool(all(
        not r["passes_f1_detection"] for r in rows if float(r["gain"]) < 0
    ))

    if h1 and h2:
        verdict = "C16_MISCUT_ONE_CANDIDATE_BUT_CLOSURE_HOLDS"
    elif h2:
        verdict = "CLOSURE_HOLDS_AND_C16_CUT_NOTHING_EXTRA"
    else:
        verdict = "PORTFOLIO_REACHES_GAP_REOPEN"

    payload: dict[str, Any] = {
        "node_id": NODE_ID,
        "lane": LANE,
        "parent": PARENT_NODE,
        "candidate": False,
        "diagnostic_only": True,
        "old_rule": {
            "form": "expected_gain < 0.15 * remaining_gap -> REJECT",
            "floor": OLD_FLOOR,
            "status": "선언 관례. 보정된 바 없다",
            "defects": [
                "방향을 개별 판정하는데 완료 기준은 합산이다 — 방향들을 대체재로 가정",
                "0.15 는 아무것도 재지 않은 값이다",
            ],
        },
        "new_rule": {
            "F1_detection": {
                "value": DETECTION_THRESHOLD,
                "source": "C1N9 가 잰 동결 게이트 검출문턱",
                "meaning": "이보다 작으면 이득이 실재해도 게이트가 구분하지 못한다",
            },
            "F2_reachability": {
                "form": "sum(F1 통과 방향) >= remaining_gap",
                "meaning": "개별이 아니라 집합에 건다",
            },
            "note": "문턱을 낮춘 것이 아니라 성질을 바꿨다. F1 은 측정값, F2 는 산술이다",
        },
        "remaining_gap": REMAINING_GAP,
        "directions": rows,
        "survivors_old_c16": [r["node"] for r in survivors_old],
        "survivors_f1": [r["node"] for r in survivors_f1],
        "revived_by_f1": revived,
        "sum_of_survivors": sum_f1,
        "reachable": reachable,
        "shortfall": REMAINING_GAP - sum_f1,
        "hypotheses": {
            "H1_c71_revived": h1,
            "H2_portfolio_still_short": h2,
            "H3_c84_rejected_under_f1_too": h3,
            "H4_negatives_rejected": h4,
        },
        "verdict": verdict,
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
        "# M271 — C16 크기게이트 재검",
        "",
        f"노드 `{NODE_ID}` / 레인 {LANE} / 부모 `{PARENT_NODE}` / **진단 전용**",
        "",
        "## 1. 옛 규칙의 결함",
        "",
        f"`expected_gain < 0.15 x 남은격차({OLD_FLOOR:.6f})` 면 기각.",
        "",
        "- **개별 판정 대 합산 목표** — 다섯 방향이 각각 +0.002 면 합 +0.010 인데 "
        "다섯 다 기각한다. 방향들을 대체재로 가정한 것이고 근거가 없다.",
        "- **근거 없는 상수** — 0.15 는 아무것도 재지 않았다.",
        "",
        "## 2. 교정 — 두 문턱으로 나눈다",
        "",
        f"- **F1 검출 가능성** `>= {DETECTION_THRESHOLD}` — C1N9 가 잰 **측정값**. "
        "이보다 작으면 게이트가 구분하지 못해 승격 자체가 불가능하다.",
        "- **F2 도달 가능성** — F1 통과 방향들의 **합**이 격차에 닿는가. 개별이 아니라 "
        "집합에 건다.",
        "",
        "문턱을 낮춘 것이 아니라 **성질을 바꿨다**. F1 은 0.15 x 격차보다 낮지만 F2 가 "
        "새로 생겨 '작은 것 여럿' 이 실제로 합쳐지는지를 강제한다.",
        "",
        "## 3. 측정된 방향",
        "",
        "| 노드 | 레인 | 이득 | 격차대비 | 옛 C16 | F1 검출 |",
        "|---|---|---:|---:|:---:|:---:|",
    ]
    for r in sorted(rows, key=lambda x: -float(x["gain"])):
        lines.append(
            f"| {r['node']} | {r['lane']} | {float(r['gain']):+.6f} | "
            f"{float(r['gain_over_gap']):.3f} | "
            f"{'O' if r['passes_old_c16'] else '-'} | "
            f"{'**O**' if r['passes_f1_detection'] else '-'} |"
        )
    lines += [
        "",
        f"**F1 이 되살린 후보**: `{', '.join(revived) or '없음'}`",
        "",
        f"F1 통과 방향의 합 **{sum_f1:+.6f}** / 남은 격차 **{REMAINING_GAP:.6f}** "
        f"-> 부족분 **{payload['shortfall']:.6f}**",
        "",
        "## 4. 사전확약",
        "",
        f"- H1 C1N71 이 F1 으로 살아난다 -> **{h1}**",
        f"- H2 포트폴리오 합이 격차에 못 미친다 -> **{h2}**",
        f"- H3 C1N84 는 F1 에서도 기각 -> **{h3}**",
        f"- H4 음수 방향은 어떤 문턱에서도 기각 -> **{h4}**",
        "",
        "## 5. 판정",
        "",
        f"**{verdict}**",
        "",
        "두 진술을 분리해서 읽어야 한다 — **C16 은 실제로 한 후보를 잘못 잘랐고**, "
        "**그럼에도 폐쇄 판정 자체는 문턱 선택과 무관하게 성립한다**.",
        "",
        f"digest `{payload['digest']}`",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("=== C16 재검 ===")
    print(f"[C16R] 옛 문턱 {OLD_FLOOR:.6f} (0.15 x 격차) / F1 검출문턱 "
          f"{DETECTION_THRESHOLD} (C1N9 측정)")
    for r in sorted(rows, key=lambda x: -float(x["gain"])):
        print(f"[C16R] {r['node']:32s} {float(r['gain']):+.6f}  "
              f"옛 {'통과' if r['passes_old_c16'] else '기각'} / "
              f"F1 {'통과' if r['passes_f1_detection'] else '기각'}")
    print(f"[C16R] F1 이 되살린 후보: {revived or '없음'}")
    print(f"[C16R] 합 {sum_f1:+.6f} / 격차 {REMAINING_GAP:.6f} -> 부족 "
          f"{payload['shortfall']:.6f} / 도달가능 {reachable}")
    print(f"[C16R] H1 {h1} / H2 {h2} / H3 {h3} / H4 {h4}")
    print(f"[C16R] 판정: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
