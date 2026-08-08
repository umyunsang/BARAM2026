"""M271 P4 사이클 14 — 앙상블을 배포 쪽으로 수축하면 일관성이 생기는가.

사이클 13 에서 E3(4 계열 앙상블)가 pooled `+0.003539` 를 냈다. 검출 문턱 `+0.001013` 의
3.5 배이고 G2·G3·G4 를 통과하는데 **G1(일관성)만** 걸린다 — 9 개월 중 6 개월 양수.

막힌 것이 크기가 아니라 일관성이라면 구조를 직접 봐야 한다.

가설: E3 를 배포 예측 쪽으로 수축하면(`pred = (1-a)*배포 + a*E3`) 이득과 손실이 둘 다
줄어든다. 손실이 모델 간 불일치가 큰 구간에서 나오므로 이득보다 빨리 줄 수 있고, 그러면
어느 a 에서 9 개월 전부 양수가 되면서 델타가 문턱을 넘을 수 있다.

**a 를 데이터 보고 고르면 same-fold 선택이다.** 사전 선언된 사다리로 전부 보고하고 최적
선택은 하지 않는다. 자연스러운 사전 선언점은 `a=0.5` 다.

사전확약(실행 전 동결):
  H1  E3 가 지는 달의 손실 크기가 이기는 달의 이득 크기보다 **작다** (중앙값 기준).
      작으면 수축이 통할 구조다.
  H2  **사전 선언점 a=0.5** 에서 9 개월 전부 양수이고 동결 게이트를 통과한다.
  H2 는 a=0.5 하나로만 판정한다. 사다리의 다른 값이 통과해도 그것은 사후 선택이므로
  H2 의 근거가 되지 않으며, 참고로만 보고한다.

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
from m270_monthly_validation import load_predictions, monthly_scores
from m271_cycle13_ensemble import ENSEMBLES, average
from m271_evaluate_candidate import ledger_diff, official

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
REPORT_MD = REPORTS / "m271_cycle14_shrinkblend.md"
RECEIPT = REPORTS / "m271_cycle14_shrinkblend_receipt.json"

NODE_ID = "C1N14_SHRINK_BLEND"
LANE = "L7"
DEPLOYED = "T0.5_G1.5"
BASE_ENSEMBLE = "E3_FOUR_FAMILY"
PREDECLARED_ALPHA = 0.5  # 사전 선언점. 이 값 하나로만 H2 를 판정한다.
ALPHA_LADDER = (0.25, 0.375, 0.5, 0.625, 0.75, 1.0)
GATE_DETECTION_THRESHOLD = 0.001013


def blend(parent, ensemble, alpha: float):
    keys = ["forecast_id", "forecast_kst_dtm", "group_id"]
    merged = parent.loc[:, [*keys, "actual_kwh", "prediction_kwh"]].merge(
        ensemble.loc[:, [*keys, "prediction_kwh"]].rename(
            columns={"prediction_kwh": "ens"}
        ),
        on=keys, how="inner",
    )
    merged["prediction_kwh"] = (
        (1.0 - alpha) * merged["prediction_kwh"] + alpha * merged["ens"]
    )
    out = merged.loc[:, [*keys, "actual_kwh", "prediction_kwh"]]
    out["month"] = out["forecast_kst_dtm"].dt.to_period("M").astype(str)
    return out


def monthly_delta(candidate, parent) -> dict[str, float]:
    left = monthly_scores(candidate).set_index("month")
    right = monthly_scores(parent).set_index("month")
    shared = [m for m in left.index if m in right.index]
    return {
        m: float(left.loc[m, "total"] - right.loc[m, "total"]) for m in shared
    }


def main() -> int:
    parent = load_predictions(DEPLOYED)
    parent_score = official(parent)
    ensemble = average(ENSEMBLES[BASE_ENSEMBLE])

    # H1: E3 의 월별 이득/손실 구조
    base_delta = monthly_delta(ensemble, parent)
    gains = [v for v in base_delta.values() if v > 0]
    losses = [-v for v in base_delta.values() if v < 0]
    h1 = bool(losses and gains and np.median(losses) < np.median(gains))

    rows: list[dict[str, Any]] = []
    for alpha in ALPHA_LADDER:
        candidate = blend(parent, ensemble, alpha)
        score = official(candidate)
        gate = evaluate_gate(candidate, parent)
        flags = {label.split()[0]: bool(ok) for label, ok in gate.conditions.items()}
        stats = gate.evidence
        diff = ledger_diff(candidate, parent)
        rows.append(
            {
                "alpha": alpha,
                "total": score["total"],
                "delta_total": score["total"] - parent_score["total"],
                "one_minus_nmae": score["one_minus_nmae"],
                "ficr": score["ficr"],
                "passed": bool(gate.passed),
                "gate": flags,
                "positive_months": int(stats["positive_months"]),
                "months_scored": int(stats["months_scored"]),
                "sign_test_p": float(stats["sign_test_p_greater"]),
                "bootstrap_q05": float(stats["block_bootstrap_q05"]),
                "min_delta": float(stats["min_total_delta"]),
                "cells_improved": diff["cells_improved"],
                "cells_worsened": diff["cells_worsened"],
                "above_detection": bool(
                    abs(score["total"] - parent_score["total"]) >= GATE_DETECTION_THRESHOLD
                ),
            }
        )

    predeclared = next(r for r in rows if r["alpha"] == PREDECLARED_ALPHA)
    h2 = bool(
        predeclared["passed"]
        and predeclared["positive_months"] == predeclared["months_scored"]
    )
    ladder_passers = [r["alpha"] for r in rows if r["passed"]]

    check = {
        "H1_expectation": "지는 달의 손실 크기가 이기는 달의 이득 크기보다 작다",
        "H1_held": h1,
        "gain_median": float(np.median(gains)) if gains else float("nan"),
        "loss_median": float(np.median(losses)) if losses else float("nan"),
        "H2_expectation": f"사전 선언점 a={PREDECLARED_ALPHA} 에서 전월 양수 + 게이트 통과",
        "H2_held": h2,
        "predeclared_alpha": PREDECLARED_ALPHA,
        "ladder_passers_for_reference_only": ladder_passers,
        "verdict": "SHRINK_BLEND_VALIDATED" if h2 else "SHRINK_BLEND_REJECTED",
    }
    payload = {
        "gate_version": GATE_VERSION,
        "gate_modified": False,
        "base_ensemble": BASE_ENSEMBLE,
        "parent": {"policy": DEPLOYED, **parent_score},
        "base_monthly_delta": base_delta,
        "ladder": rows,
        "predeclared_check": check,
    }

    lines = [
        "# M271 P4 사이클 14 — 앙상블 수축 블렌드",
        "",
        f"- 판정일: {datetime.now(UTC).strftime('%Y-%m-%d')} (UTC)",
        f"- 노드: `{NODE_ID}` / 레인 {LANE}",
        f"- 기반: `{BASE_ENSEMBLE}` (사이클 13, pooled +0.003539, G1 만 실패)",
        f"- 게이트: `{GATE_VERSION}` (읽기만 함)",
        "",
        "## 1. E3 의 월별 구조 (H1)",
        "",
        "| 월 | Total 델타 |",
        "|---|---:|",
    ]
    for month in sorted(base_delta):
        v = base_delta[month]
        mark = "" if v > 0 else " **(손실)**"
        lines.append(f"| {month} | {v:+.6f}{mark} |")

    lines += [
        "",
        f"- 이득 중앙값 **{check['gain_median']:+.6f}** / 손실 중앙값 "
        f"**{check['loss_median']:+.6f}**",
        f"- H1 `{check['H1_expectation']}` -> **{check['H1_held']}**",
        "",
        "손실이 이득보다 작으면 수축으로 손실을 먼저 지울 여지가 있다.",
        "",
        "## 2. 수축 사다리",
        "",
        "`pred = (1-a)*배포 + a*E3`. **a 를 데이터 보고 고르지 않는다** — 사전 선언점은",
        f"`a={PREDECLARED_ALPHA}` 이고 H2 는 그 값 하나로만 판정한다.",
        "",
        "| a | Total | 델타 | G1 G2 G3 G4 | 양수월 | p | q05 | 개선/악화 셀 | 문턱초과 | 통과 |",
        "|---:|---:|---:|:---:|---:|---:|---:|---:|:---:|:---:|",
    ]
    for r in rows:
        f = "".join("O" if r["gate"].get(k) else "-" for k in ("G1", "G2", "G3", "G4"))
        star = " **<- 사전선언**" if r["alpha"] == PREDECLARED_ALPHA else ""
        lines.append(
            f"| {r['alpha']:.3f}{star} | {r['total']:.6f} | {r['delta_total']:+.6f} | "
            f"`{f}` | {r['positive_months']}/{r['months_scored']} | "
            f"{r['sign_test_p']:.4f} | {r['bootstrap_q05']:+.6f} | "
            f"{r['cells_improved']}/{r['cells_worsened']} | "
            f"{'O' if r['above_detection'] else '-'} | "
            f"{'**통과**' if r['passed'] else '기각'} |"
        )

    lines += [
        "",
        "## 3. 사전확약 대조",
        "",
        f"- H1 -> **{check['H1_held']}**",
        f"- H2 `{check['H2_expectation']}` -> **{check['H2_held']}**",
        "",
        f"판정: **{check['verdict']}**",
        "",
        "## 4. 사다리의 다른 값에 대하여",
        "",
        f"게이트를 통과한 a 값: `{ladder_passers or '없음'}`",
        "",
        "**사전 선언점이 아닌 a 가 통과해도 그것은 근거가 되지 않는다.** 사다리를 보고 통과하는",
        "a 를 고르는 것은 이 프로젝트가 반복적으로 기각해 온 same-fold 선택이다. 참고로만",
        "보고하며, 채택하려면 별도의 시간순 안전 선택 절차가 필요하다.",
        "",
        "## 5. 한계",
        "",
        "여기서 잰 것은 **로컬** 판정이다. 온라인 이전은 별개 문제이며 방법군별 오프셋이",
        "3.2 배까지 벌어진 전례가 있다(M261 +0.0066 vs M252 +0.0211).",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    receipt = {
        "schema_version": 1,
        "stage": "M271_P4_CYCLE14_SHRINK_BLEND",
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

    print(f"[C14] E3 이득 중앙값 {check['gain_median']:+.6f} / "
          f"손실 중앙값 {check['loss_median']:+.6f} -> H1={h1}")
    for r in rows:
        f = "".join("O" if r["gate"].get(k) else "-" for k in ("G1", "G2", "G3", "G4"))
        star = " *사전선언*" if r["alpha"] == PREDECLARED_ALPHA else ""
        print(f"[C14] a={r['alpha']:.3f} Total={r['total']:.6f} 델타={r['delta_total']:+.6f} "
              f"[{f}] {r['positive_months']}/{r['months_scored']}월 "
              f"셀 +{r['cells_improved']}/-{r['cells_worsened']} "
              f"{'통과' if r['passed'] else '기각'}{star}")
    print(f"[C14] H2 (a={PREDECLARED_ALPHA} 만으로 판정) = {h2} -> {check['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
