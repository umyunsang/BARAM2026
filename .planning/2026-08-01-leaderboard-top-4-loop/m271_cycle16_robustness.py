"""M271 P4 사이클 16 — 검증된 후보의 견고성 감사.

사이클 14 의 `SHRINKBLEND_A05` 가 동결 게이트를 통과했다(Total 0.632505, 델타 +0.003900,
8/9 월). 게이트를 통과했다는 것이 곧 승격 가능하다는 뜻은 아니다. 취약한 후보는 2025 년에
재현되지 않는다.

감사 항목 넷.

  1. **그룹별 개선** — 셋 다 좋아지는가, 아니면 하나가 끌고 둘이 끌려가는가
  2. **월 하나 제거(leave-one-month-out)** — 특정 달에 기대고 있는가
  3. **재현성** — 사양만으로 재구성해 예측이 바이트 일치하는가
  4. **alpha 근방 안정성** — 사이클 14 사다리에서 0.25~0.75 가 모두 통과했다

사전확약(실행 전 동결):
  H1  세 그룹 **모두** FICR 이 개선된다.
  H2  9 개월 중 **어느 달을 빼도** 동결 게이트를 통과한다.
  H3  같은 사양으로 두 번 만든 예측이 **바이트 일치**한다.
  셋 다 성립해야 승격 후보로 볼 수 있다. 하나라도 기각되면 취약점을 명시한다.

**게이트를 수정하지 않는다.** 읽기만 한다.

읽기 전용. 모델을 적합하지 않고 2024 행을 읽지 않는다.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from m270_gate import GATE_VERSION, evaluate_gate
from m270_monthly_validation import load_predictions
from m271_cycle13_ensemble import ENSEMBLES, average
from m271_cycle14_shrinkblend import blend
from m271_evaluate_candidate import official

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
REPORT_MD = REPORTS / "m271_cycle16_robustness.md"
RECEIPT = REPORTS / "m271_cycle16_robustness_receipt.json"

NODE_ID = "C1N16_ROBUSTNESS_AUDIT"
LANE = "L4"
DEPLOYED = "T0.5_G1.5"
BASE_ENSEMBLE = "E3_FOUR_FAMILY"
ALPHA = 0.5


def build_candidate(parent: pd.DataFrame) -> pd.DataFrame:
    return blend(parent, average(ENSEMBLES[BASE_ENSEMBLE]), ALPHA)


def prediction_digest(frame: pd.DataFrame) -> str:
    ordered = frame.sort_values(["forecast_kst_dtm", "group_id", "forecast_id"])
    payload = ordered["prediction_kwh"].to_numpy(dtype="float64").tobytes()
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    parent = load_predictions(DEPLOYED)
    candidate = build_candidate(parent)

    # --- H1 그룹별 개선
    cand_score = official(candidate)
    parent_score = official(parent)
    per_group = []
    for g in (1, 2, 3):
        ficr_delta = cand_score["group_ficr"][g] - parent_score["group_ficr"][g]
        nmae_delta = cand_score["group_nmae"][g] - parent_score["group_nmae"][g]
        per_group.append(
            {
                "group": g,
                "ficr_parent": parent_score["group_ficr"][g],
                "ficr_candidate": cand_score["group_ficr"][g],
                "ficr_delta": ficr_delta,
                # NMAE 는 낮을수록 좋으므로 음수가 개선이다.
                "nmae_delta": nmae_delta,
                "ficr_improved": bool(ficr_delta > 0),
                "nmae_improved": bool(nmae_delta < 0),
            }
        )
    h1 = all(r["ficr_improved"] for r in per_group)

    # --- H2 월 하나 제거
    months = sorted(candidate["month"].unique())
    loo = []
    for drop in months:
        c = candidate.loc[candidate["month"] != drop]
        p = parent.loc[parent["month"] != drop]
        gate = evaluate_gate(c, p)
        stats = gate.evidence
        loo.append(
            {
                "dropped_month": drop,
                "passed": bool(gate.passed),
                "gate": {label.split()[0]: bool(ok) for label, ok in gate.conditions.items()},
                "positive_months": int(stats["positive_months"]),
                "months_scored": int(stats["months_scored"]),
                "sign_test_p": float(stats["sign_test_p_greater"]),
                "bootstrap_q05": float(stats["block_bootstrap_q05"]),
            }
        )
    h2 = all(r["passed"] for r in loo)
    failing = [r["dropped_month"] for r in loo if not r["passed"]]

    # --- H3 재현성
    d1 = prediction_digest(candidate)
    d2 = prediction_digest(build_candidate(load_predictions(DEPLOYED)))
    h3 = bool(d1 == d2)

    check = {
        "H1_expectation": "세 그룹 모두 FICR 개선",
        "H1_held": h1,
        "H2_expectation": "어느 달을 빼도 동결 게이트 통과",
        "H2_held": h2,
        "H2_failing_when_dropped": failing,
        "H3_expectation": "같은 사양으로 두 번 만든 예측이 바이트 일치",
        "H3_held": h3,
        "prediction_digest": d1,
        "verdict": (
            "PROMOTION_CANDIDATE" if (h1 and h2 and h3) else "FRAGILE_SEE_NOTES"
        ),
    }
    payload = {
        "gate_version": GATE_VERSION,
        "gate_modified": False,
        "candidate": {
            "name": "M271_SHRINKBLEND_A05",
            "recipe": f"0.5 * {DEPLOYED} + 0.5 * mean({', '.join(ENSEMBLES[BASE_ENSEMBLE])})",
            **cand_score,
            "delta_total": cand_score["total"] - parent_score["total"],
        },
        "parent": {"policy": DEPLOYED, **parent_score},
        "per_group": per_group,
        "leave_one_month_out": loo,
        "predeclared_check": check,
    }

    lines = [
        "# M271 P4 사이클 16 — 검증된 후보의 견고성 감사",
        "",
        f"- 판정일: {datetime.now(UTC).strftime('%Y-%m-%d')} (UTC)",
        f"- 노드: `{NODE_ID}` / 레인 {LANE}",
        f"- 후보: `M271_SHRINKBLEND_A05` — {payload['candidate']['recipe']}",
        f"- 게이트: `{GATE_VERSION}` (읽기만 함)",
        "",
        f"pooled Total **{cand_score['total']:.6f}** (배포 {parent_score['total']:.6f}, "
        f"델타 {payload['candidate']['delta_total']:+.6f})",
        "",
        "## 1. 그룹별 개선 (H1)",
        "",
        "| 그룹 | FICR 배포 | FICR 후보 | **FICR 델타** | NMAE 델타 | FICR 개선 | NMAE 개선 |",
        "|---:|---:|---:|---:|---:|:---:|:---:|",
    ]
    for r in per_group:
        lines.append(
            f"| {r['group']} | {r['ficr_parent']:.6f} | {r['ficr_candidate']:.6f} | "
            f"**{r['ficr_delta']:+.6f}** | {r['nmae_delta']:+.6f} | "
            f"{'O' if r['ficr_improved'] else '**X**'} | "
            f"{'O' if r['nmae_improved'] else '**X**'} |"
        )
    lines += ["", "NMAE 는 낮을수록 좋으므로 음수가 개선이다.", ""]

    lines += [
        "## 2. 월 하나 제거 (H2)",
        "",
        "특정 달에 기대고 있으면 그 달을 뺐을 때 게이트가 무너진다.",
        "",
        "| 제거한 달 | G1 G2 G3 G4 | 양수월 | p | q05 | 통과 |",
        "|---|:---:|---:|---:|---:|:---:|",
    ]
    for r in loo:
        f = "".join("O" if r["gate"].get(k) else "-" for k in ("G1", "G2", "G3", "G4"))
        lines.append(
            f"| {r['dropped_month']} | `{f}` | "
            f"{r['positive_months']}/{r['months_scored']} | {r['sign_test_p']:.4f} | "
            f"{r['bootstrap_q05']:+.6f} | {'**통과**' if r['passed'] else '**기각**'} |"
        )

    lines += [
        "",
        "## 3. 재현성 (H3)",
        "",
        f"- 예측 벡터 SHA-256: `{d1[:32]}...`",
        f"- 두 번 재구성 일치: **{h3}**",
        "",
        "## 4. 사전확약 대조",
        "",
        f"- H1 `{check['H1_expectation']}` -> **{h1}**",
        f"- H2 `{check['H2_expectation']}` -> **{h2}**"
        + (f" (실패: {failing})" if failing else ""),
        f"- H3 `{check['H3_expectation']}` -> **{h3}**",
        "",
        f"판정: **{check['verdict']}**",
        "",
        "## 5. 남는 한계",
        "",
        "1. 이것은 **로컬 2023 Q2~Q4** 판정이다. 온라인 이전은 별개이며 방법군별 오프셋이",
        "   3.2 배까지 벌어진 전례가 있다(M261 +0.0066 vs M252 +0.0211). 이 후보는 분류기와",
        "   analog 를 섞었으므로 오프셋이 어느 쪽을 따를지 모른다.",
        "2. 이득 `+0.0039` 는 격차 `0.0314` 의 12.4% 다. 목표까지 갈 길이 남았다.",
        "3. alpha=0.5 는 사전 선언점이다. 사이클 14 사다리에서 0.25~0.75 가 모두 게이트를",
        "   통과했으므로 alpha 선택에 민감하지 않다는 점은 견고성 근거가 된다.",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    receipt = {
        "schema_version": 1,
        "stage": "M271_P4_CYCLE16_ROBUSTNESS",
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

    for r in per_group:
        print(f"[C16] g{r['group']} FICR {r['ficr_delta']:+.6f} "
              f"NMAE {r['nmae_delta']:+.6f} "
              f"({'개선' if r['ficr_improved'] else '악화'})")
    print(f"[C16] H1 전그룹 FICR 개선 = {h1}")
    print(f"[C16] H2 월제거 견고성 = {h2}" + (f" (실패: {failing})" if failing else ""))
    print(f"[C16] H3 재현 = {h3}  digest {d1[:16]}")
    print(f"[C16] 판정: {check['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
