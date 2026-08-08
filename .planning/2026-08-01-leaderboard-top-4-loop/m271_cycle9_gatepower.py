"""M271 P4 사이클 9 — 동결 월별 게이트의 검출력.

사이클 7 은 정책 62 개가 **전부** 동결 게이트에 걸렸다고 보고했다. 사이클 4·8 은 실재하는
오라클 이득(+0.001~+0.005)이 있다고 측정했다. 그러면 질문이 뒤집힌다.

    **게이트가 이 크기의 진짜 개선을 탐지할 힘이 있는가?**

게이트는 9 개 월 블록의 부호검정이다. n=9 에서 G1(p<=0.10)을 통과하려면 최소 7 개월이
양수여야 한다. A1 이 측정한 대로 월 안의 행들은 lag-1 자기상관 0.95 이상으로 묶여 있어
월별 델타 자체의 잡음이 크다.

게이트가 +0.005 수준의 진짜 개선도 못 잡는다면 "0/62 기각" 은 정책에 대한 정보가 아니라
**게이트에 대한 정보**다. 그리고 이 프로젝트의 모든 후보가 그 게이트를 통과해야 하므로
검증전략(L4) 레인의 근본 문제가 된다.

측정 방법: 배포 예측을 실제값 쪽으로 배율 `k` 만큼 수축시켜 **알려진 크기**의 개선을
만들고, 그 후보를 동결 게이트에 넣는다. 게이트가 통과시키기 시작하는 효과크기가 곧
검출 문턱이다.

수축은 회수 방법이 아니라 **자** 다. 실제값을 쓰므로 운영 불가이며, 여기서는 게이트의
민감도를 재는 도구로만 쓴다. `m270_revised_verdicts.py` 가 요구량을 잴 때 쓴 것과 같은 장치다.

사전확약(실행 전 동결):
  H1  게이트는 +0.005 Total 개선을 통과시킨다.
  H1 이 기각되면 게이트가 실제 개선을 놓치고 있다는 뜻이고, 사이클 7 의 "0/62" 는
  재해석되어야 한다.

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
from m270_monthly_validation import load_predictions, score_frame

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
REPORT_MD = REPORTS / "m271_cycle9_gatepower.md"
RECEIPT = REPORTS / "m271_cycle9_gatepower_receipt.json"

NODE_ID = "C1N9_GATE_POWER"
LANE = "L4"
DEPLOYED = "T0.5_G1.5"
# 수축 배율. 1.0 = 개선 없음. 작을수록 큰 개선.
SHRINK_FACTORS = (1.000, 0.995, 0.990, 0.980, 0.970, 0.960, 0.950, 0.930, 0.900, 0.850)
PREDECLARED_DETECTABLE = 0.005  # H1 이 요구하는 검출 문턱


def shrink(frame, k: float):
    """예측을 실제값 쪽으로 k 배 수축. 알려진 크기의 개선을 만든다."""
    out = frame.copy()
    actual = out["actual_kwh"].to_numpy(float)
    pred = out["prediction_kwh"].to_numpy(float)
    out["prediction_kwh"] = actual + k * (pred - actual)
    return out


def main() -> int:
    parent = load_predictions(DEPLOYED)
    parent_pooled = score_frame(parent)

    rows: list[dict[str, Any]] = []
    for k in SHRINK_FACTORS:
        candidate = shrink(parent, k)
        pooled = score_frame(candidate)
        result = evaluate_gate(candidate, parent)
        stats = result.evidence
        gate = {label.split()[0]: bool(ok) for label, ok in result.conditions.items()}
        rows.append(
            {
                "k": k,
                "pooled_total": pooled["total"],
                "true_gain": pooled["total"] - parent_pooled["total"],
                "passed": bool(result.passed),
                "gate": gate,
                "positive_months": int(stats["positive_months"]),
                "months_scored": int(stats["months_scored"]),
                "sign_test_p": float(stats["sign_test_p_greater"]),
                "median_delta": float(stats["median_total_delta"]),
                "bootstrap_q05": float(stats["block_bootstrap_q05"]),
                "min_delta": float(stats["min_total_delta"]),
            }
        )

    passing = [r for r in rows if r["passed"] and r["true_gain"] > 0]
    threshold = min((r["true_gain"] for r in passing), default=float("nan"))
    # G1 이 언제부터 통과하는지 별도로 본다. 보통 여기가 병목이다.
    g1_pass = [r for r in rows if r["gate"].get("G1") and r["true_gain"] > 0]
    g1_threshold = min((r["true_gain"] for r in g1_pass), default=float("nan"))

    detectable_at_target = any(
        r["passed"] and abs(r["true_gain"] - PREDECLARED_DETECTABLE) < 0.0015 for r in rows
    )
    check = {
        "H1_expectation": f"게이트는 +{PREDECLARED_DETECTABLE} Total 개선을 통과시킨다",
        "H1_held": bool(
            threshold == threshold and threshold <= PREDECLARED_DETECTABLE
        ),
        "detection_threshold_total": threshold,
        "g1_detection_threshold_total": g1_threshold,
        "detectable_near_target": detectable_at_target,
        "verdict": (
            "GATE_HAS_POWER"
            if (threshold == threshold and threshold <= PREDECLARED_DETECTABLE)
            else "GATE_UNDERPOWERED"
        ),
    }
    payload = {
        "gate_version": GATE_VERSION,
        "gate_modified": False,
        "deployed_pooled": parent_pooled,
        "ladder": rows,
        "predeclared_check": check,
    }

    lines = [
        "# M271 P4 사이클 9 — 동결 월별 게이트의 검출력",
        "",
        f"- 판정일: {datetime.now(UTC).strftime('%Y-%m-%d')} (UTC)",
        f"- 노드: `{NODE_ID}` / 레인 {LANE}",
        f"- 게이트: `{GATE_VERSION}` (읽기만 함, 재동결 금지)",
        f"- 부모: `{DEPLOYED}` pooled {parent_pooled['total']:.6f}",
        "",
        "## 1. 질문",
        "",
        "사이클 7 은 정책 62 개가 전부 게이트에 걸렸다고 보고했다. 사이클 4·8 은 실재하는",
        "오라클 이득(+0.001~+0.005)이 있다고 측정했다. 게이트가 그 크기의 **진짜** 개선을",
        "탐지하지 못한다면 '0/62 기각' 은 정책이 아니라 게이트에 대한 정보다.",
        "",
        "예측을 실제값 쪽으로 배율 k 만큼 수축시켜 알려진 크기의 개선을 만들고 게이트에 넣는다.",
        "수축은 회수 방법이 아니라 **자** 다. 실제값을 쓰므로 운영 불가다.",
        "",
        "## 2. 검출력 사다리",
        "",
        "| k | pooled Total | **진짜 이득** | G1 | G2 | G3 | G4 | 통과 | 양수월 | 부호검정 p |",
        "|---:|---:|---:|:--:|:--:|:--:|:--:|:--:|---:|---:|",
    ]
    for r in rows:
        g = r["gate"]
        flags = {k: ("O" if g.get(k) else "-") for k in ("G1", "G2", "G3", "G4")}
        lines.append(
            f"| {r['k']:.3f} | {r['pooled_total']:.6f} | **{r['true_gain']:+.6f}** | "
            f"{flags['G1']} | {flags['G2']} | {flags['G3']} | {flags['G4']} | "
            f"{'**통과**' if r['passed'] else '기각'} | "
            f"{r['positive_months']}/{r['months_scored']} | {r['sign_test_p']:.4f} |"
        )

    lines += [
        "",
        "## 3. 검출 문턱",
        "",
        f"- 전체 게이트가 통과시키기 시작하는 진짜 이득: **{threshold:+.6f}**",
        f"- G1(일관성) 단독 문턱: **{g1_threshold:+.6f}**",
        "",
        "## 4. 사전확약 대조",
        "",
        f"- H1 `{check['H1_expectation']}` -> **{check['H1_held']}**",
        "",
        f"판정: **{check['verdict']}**",
        "",
        "## 5. 읽는 법",
        "",
        "`GATE_UNDERPOWERED` 이면 사이클 7 의 '0/62 기각' 은 **재해석되어야 한다**. 정책이",
        "나빠서가 아니라 게이트가 그 크기를 못 봐서일 수 있다. 그 경우 게이트를 고치는 것이",
        "아니라(재동결 금지) **더 큰 효과를 내는 후보를 찾거나 검증 표면을 늘리는 것**이",
        "대응이다.",
        "",
        "`GATE_HAS_POWER` 이면 사이클 7 의 판정은 그대로 유효하고 정책선택 축은 닫힌 채로 둔다.",
        "",
        "수축 사다리는 **요구량을 재는 자**이지 달성 가능량이 아니다. k 를 줄여 얻은 이득은",
        "실제값을 봐야 나오므로 운영에서 재현되지 않는다.",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    receipt = {
        "schema_version": 1,
        "stage": "M271_P4_CYCLE9_GATE_POWER",
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
        g = r["gate"]
        flags = "".join("O" if g.get(k) else "-" for k in ("G1", "G2", "G3", "G4"))
        print(f"[C9] k={r['k']:.3f} 이득={r['true_gain']:+.6f} [{flags}] "
              f"{r['positive_months']}/{r['months_scored']}월 "
              f"{'통과' if r['passed'] else '기각'}")
    print(f"[C9] 검출 문턱 = {threshold:+.6f} (G1 단독 {g1_threshold:+.6f})")
    print(f"[C9] 판정: {check['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
