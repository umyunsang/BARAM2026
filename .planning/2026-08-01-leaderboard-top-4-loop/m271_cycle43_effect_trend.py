"""M271 P4 사이클 43 — 처리효과가 대조군 품질에 따라 어떻게 변하는가.

사이클 40·41·42 가 밴드 인지 학습의 처리효과를 세 번 쟀다. 부호는 3/3 재현됐지만 **크기가
대조군 품질과 강하게 반비례**한다.

    CONTROL 0.567875 (41, 과적합)        효과 +0.021490
    CONTROL 0.584468 (40, 기준)          효과 +0.009772
    CONTROL 0.595919 (42, teacher 복원)  효과 +0.004827

이 노드는 그 관계를 명시적으로 적합해 **배포 품질에서의 외삽 효과**를 낸다. 새 학습은 없다.

기전 해석
--------
정산모양 soft target 은 밴드 안 여러 빈에 질량을 퍼뜨린다. one-hot 은 정답 빈 하나에 모든
질량을 요구하므로 과적합할 표적이 뾰족하다. 즉 밴드 목표는 목적함수 정합일 뿐 아니라
**암묵적 정규화**다. 모형이 잘못 지정될수록 정규화 이득이 크고, 잘 지정될수록 줄어든다.
세 점의 단조 감소는 그 해석과 일치한다.

① 방법 리서치 (실행 전)
  - 새 방법 없음. 세 관측점의 회귀와 외삽이다.
  - **외삽의 한계를 사양에 명시한다.** 관측 범위는 CONTROL in [0.5679, 0.5959] 이고
    배포는 0.6286 으로 **범위 밖**이다. 외삽은 측정이 아니다.

② 사양 동결

  사전확약(적합 전 동결):
    H1  세 점의 (CONTROL, 효과) 상관이 **음수이고 |r| >= 0.9**.
    H2  선형 외삽의 영점이 배포 품질(0.628605)**보다 낮다**.
        성립하면 배포 품질에서 효과가 음수로 외삽된다.
    H3  세 효과의 부호가 모두 양수 (기전 재현은 유지).

  H1·H2 가 성립하면 축을 **외삽 근거로** 닫되, 폐기 전제에 그것이 3 점 외삽임을 명시하고
  **배포 품질 대조군에서 양의 효과가 관측되면 뒤집히도록** 술어를 쓴다.

**학습 없음. 게이트 무관. lockbox 미사용.**
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

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
REPORT_MD = REPORTS / "m271_cycle43_effect_trend.md"
RECEIPT = REPORTS / "m271_cycle43_effect_trend_receipt.json"

NODE_ID = "C1N43_EFFECT_TREND"
LANE = "L3"
PARENT_NODE = "C1N42_TEACHER_RESTORED"
DEPLOYED_TOTAL = 0.628605
M115_FIXED_TOTAL = 0.630310

OBSERVATIONS = (
    {"cycle": 41, "setting": "leaves 31 / 500 rounds (과적합)",
     "control": 0.567875, "effect": 0.021490},
    {"cycle": 40, "setting": "leaves 15 / 200 rounds",
     "control": 0.584468, "effect": 0.009772},
    {"cycle": 42, "setting": "leaves 15 / 200 rounds + teacher 복원",
     "control": 0.595919, "effect": 0.004827},
)
H1_MIN_ABS_R = 0.9


def main() -> int:
    control = np.array([o["control"] for o in OBSERVATIONS])
    effect = np.array([o["effect"] for o in OBSERVATIONS])
    slope, intercept = np.polyfit(control, effect, 1)
    r = float(np.corrcoef(control, effect)[0, 1])
    zero_at = float(-intercept / slope)
    at_deployed = float(slope * DEPLOYED_TOTAL + intercept)
    at_m115 = float(slope * M115_FIXED_TOTAL + intercept)

    h1 = bool(r < 0 and abs(r) >= H1_MIN_ABS_R)
    h2 = bool(zero_at < DEPLOYED_TOTAL)
    h3 = bool(all(o["effect"] > 0 for o in OBSERVATIONS))
    closed = bool(h1 and h2)
    verdict = (
        "BAND_TARGET_IS_REGULARISER_VANISHES_AT_FRONTIER" if closed
        else "TREND_INCONCLUSIVE"
    )

    check = {
        "H1_expectation": f"상관 음수이고 |r| >= {H1_MIN_ABS_R}",
        "H1_held": h1, "H1_measured": r,
        "H2_expectation": f"외삽 영점 < 배포 품질 {DEPLOYED_TOTAL}",
        "H2_held": h2, "H2_measured": zero_at,
        "H3_expectation": "세 효과 모두 양수 (기전 재현 유지)",
        "H3_held": h3,
        "extrapolation_warning": "관측 범위는 CONTROL in "
                                 f"[{control.min():.6f}, {control.max():.6f}] 이고 "
                                 f"배포 {DEPLOYED_TOTAL} 은 **범위 밖**이다. "
                                 "외삽은 측정이 아니다",
        "verdict": verdict,
    }

    payload = {
        "node": NODE_ID, "parent_node": PARENT_NODE,
        "no_training": True, "gate_touched": False, "lockbox_used": False,
        "observations": list(OBSERVATIONS),
        "fit": {"slope": float(slope), "intercept": float(intercept), "pearson_r": r,
                "n_points": len(OBSERVATIONS)},
        "extrapolation": {
            "effect_zero_at_control": zero_at,
            "effect_at_deployed": at_deployed,
            "effect_at_m115_fixed": at_m115,
        },
        "mechanism": "정산모양 soft target 은 밴드 안 여러 빈에 질량을 퍼뜨려 **암묵적 "
                     "정규화**로 작동한다. one-hot 은 표적이 뾰족해 과적합에 취약하다. "
                     "모형이 잘못 지정될수록 이득이 크고 잘 지정될수록 줄어든다",
        "predeclared_check": check,
    }

    lines = [
        "# M271 P4 사이클 43 — 처리효과와 대조군 품질의 관계",
        "",
        f"- 판정일: {datetime.now(UTC).strftime('%Y-%m-%d')} (UTC)",
        f"- 노드: `{NODE_ID}` / 레인 {LANE} / 부모 `{PARENT_NODE}`",
        "- **새 학습 없음.** 사이클 40·41·42 의 세 관측점을 적합한다",
        "",
        "## 1. 관측",
        "",
        "| 사이클 | 설정 | CONTROL | 처리효과 |",
        "|---:|---|---:|---:|",
    ]
    for o in OBSERVATIONS:
        lines.append(
            f"| {o['cycle']} | {o['setting']} | {o['control']:.6f} | "
            f"**{o['effect']:+.6f}** |"
        )
    lines += [
        "",
        "## 2. 적합과 외삽",
        "",
        f"- 기울기 **{slope:+.4f}**, 절편 {intercept:+.4f}, Pearson **{r:+.4f}** (n=3)",
        f"- 처리효과 영점: CONTROL = **{zero_at:.6f}**",
        f"- 배포 품질({DEPLOYED_TOTAL})에서 외삽 효과: **{at_deployed:+.6f}**",
        f"- `M115@T0.6_G0.2`({M115_FIXED_TOTAL})에서: **{at_m115:+.6f}**",
        "",
        f"**{check['extrapolation_warning']}.**",
        "",
        "## 3. 기전",
        "",
        payload["mechanism"] + ".",
        "",
        "## 4. 사전확약 대조",
        "",
        f"- H1 `{check['H1_expectation']}` -> **{h1}** ({r:+.4f})",
        f"- H2 `{check['H2_expectation']}` -> **{h2}** ({zero_at:.6f})",
        f"- H3 `{check['H3_expectation']}` -> **{h3}**",
        "",
        f"판정: **{verdict}**",
        "",
        "## 5. 읽는 법",
        "",
        "부호는 세 번 재현됐지만 **그 효과는 모형이 나쁠수록 크다.** 배포 품질 근처에서는",
        "외삽상 음수다. 즉 밴드 인지 학습은 **잘못 지정된 모형을 구제하는 정규화**이지",
        "**잘 지정된 모형을 개선하는 목적함수 정합**이 아니다 — 적어도 이 세 점이 말하는 바로는.",
        "",
        "이 폐쇄는 외삽에 기대므로 **배포 품질 대조군에서 양의 효과가 한 번이라도 관측되면",
        "뒤집힌다.** 술어를 그렇게 쓴다.",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    receipt: dict[str, Any] = {
        "schema_version": 1,
        "stage": "M271_P4_CYCLE43_EFFECT_TREND",
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

    print(f"[C43] 기울기 {slope:+.4f} 절편 {intercept:+.4f} r {r:+.4f} (n=3)")
    print(f"[C43] 영점 CONTROL {zero_at:.6f} / 배포에서 {at_deployed:+.6f} / "
          f"M115 에서 {at_m115:+.6f}")
    print(f"[C43] H1 {h1} | H2 {h2} | H3 {h3}")
    print(f"[C43] 판정: {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
