"""M271 P4 사이클 23 — 모집단을 고친 MOS 재시행.

사이클 21·22 는 실패했지만 사유가 내가 적은 것과 다르다. 층별 이동이 전 대역에서 음이고
크기가 `-0.05 ~ -0.14` 용량이었다. 앙상블이 용량의 10% 만큼 과대예측한다는 뜻인데, 공식
지표는 **실측 >= 용량 10% 인 행만 채점**한다. 이동을 **전체 행**에서 추정한 것이 원인이다.
실측이 0 에 가까운 미채점 행(예측은 양수)들이 거대한 음의 잔차로 중앙값을 끌고 갔다.

**추정 모집단 != 평가 모집단.** 사양 결함이지 편향보정이 안 듣는다는 증거가 아니다.
여기서 축을 닫으면 잘못 닫는 것이다(계획 R10 이 경고한 실패 모드).

사이클 22 의 H1·H2 가 내가 예측한 **부호의 반대**로 기각된 것이 단서였다. 평균회귀
가설이면 대역이 오를수록 이동이 음으로 커져야 하는데 실제로는 **낮은 대역일수록 더 음**
이었다 — 저출력 행이 끌고 있다는 서명이다.

① 방법 리서치 (실행 전)
  - 새 방법 없음. 사이클 21 의 MOS 설계(Glahn & Lowry 1972)와 leave-one-fold-out 을
    그대로 쓰고 **추정 모집단만** 고친다. 방법이 아니라 사양이 틀렸기 때문이다.
  - 목적함수 우선순위를 **실행 전에** 다시 정한다. 평가가 공식 지표이므로 지표정합
    추정량이 선험적으로 옳다. `median_shift` 는 조건부 중앙값(NMAE)만 겨냥하는데 A7 이
    잰 손실 구성은 FICR 이 80.5% 다. 따라서 둘 다 자격을 얻으면 **`metric_shift` 우선**.
    (사이클 21 은 반대 우선순위였다. 근거가 바뀐 것이지 결과를 보고 바꾼 것이 아니다 —
    그때는 "덜 맞춘 표준 추정량" 이 근거였고, 지금은 "모집단을 고치면 지표정합 목적이
    비로소 제 모집단에서 계산된다" 가 근거다.)

② 사양 동결

  추정은 **학습 fold 의 유효행(실측 >= 용량 10%)** 에서만. 적용은 보류 fold 전체에
  (미채점 행은 어차피 점수에 들어가지 않는다).

  사전확약(실행 전 동결):
    H1  (진단) 유효행 기준 이동의 크기가 전체행 기준의 **절반 미만**이다.
        성립하면 사이클 21·22 의 실패 원인이 모집단 오설정으로 확정된다.
    H2  `median_shift` 가 R1·R2·R3 를 만족한다.
    H3  `metric_shift` 가 R1·R2·R3 를 만족한다.
  R1 pooled Total 이 `M271_MEDIAN4` 대비 개선 / R2 동결 게이트 통과(부모=M271_MEDIAN4)
  R3 층별 이동 부호가 세 분할에서 일치하는 층 >= 70%

  H2·H3 가 **둘 다 기각되면** 편향보정 계열을 닫는다. 이번엔 모집단이 맞으므로 그 폐기가
  유효하다.

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
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from m270_gate import GATE_VERSION, evaluate_gate
from m271_cycle21_mos import (
    FOLDS,
    R3_MIN_SIGN_AGREEMENT,
    apply_shifts,
    estimate_shifts,
)
from m271_cycle22_global_shift import build_base
from m271_evaluate_candidate import official

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
REPORT_MD = REPORTS / "m271_cycle23_eligible_mos.md"
RECEIPT = REPORTS / "m271_cycle23_eligible_mos_receipt.json"

NODE_ID = "C1N23_ELIGIBLE_MOS"
LANE = "L1"
PARENT_NODE = "C1N22_GLOBAL_SHIFT"
SUPERSEDES = ("C1N21_MOS_BIAS_CORRECTION", "C1N22_GLOBAL_SHIFT")
INCUMBENT = "M271_MEDIAN4"
ELIGIBLE_THRESHOLD = 0.10
FROZEN_PRIORITY = ("metric_shift", "median_shift")


def eligible_mask(frame: pd.DataFrame) -> pd.Series:
    return frame["actual_kwh"] >= ELIGIBLE_THRESHOLD * frame["capacity"]


def main() -> int:
    base = build_base()
    elig = eligible_mask(base)
    incumbent_score = official(base)

    # --- H1 진단: 모집단만 바꾼 같은 추정량
    diag = []
    for method in ("median_shift", "metric_shift"):
        all_rows, _ = estimate_shifts(base, method)
        elig_rows, _ = estimate_shifts(base.loc[elig], method)
        shared = sorted(set(all_rows) & set(elig_rows), key=lambda x: (x[0], str(x[1])))
        ratios = [
            abs(elig_rows[key]) / abs(all_rows[key])
            for key in shared
            if abs(all_rows[key]) > 1e-9
        ]
        diag.append(
            {
                "method": method,
                "strata_compared": len(shared),
                "median_abs_shift_all_rows": float(
                    np.median([abs(all_rows[k]) for k in shared])
                ),
                "median_abs_shift_eligible": float(
                    np.median([abs(elig_rows[k]) for k in shared])
                ),
                "median_magnitude_ratio": float(np.median(ratios)) if ratios else float("nan"),
                "examples": [
                    {"group": int(k[0]), "pred_band": k[1],
                     "all_rows": round(all_rows[k], 4), "eligible": round(elig_rows[k], 4)}
                    for k in shared[:12]
                ],
            }
        )
    h1 = all(d["median_magnitude_ratio"] < 0.5 for d in diag)

    # --- H2 · H3 유효행 추정 + LOO
    results: dict[str, Any] = {}
    for method in ("median_shift", "metric_shift"):
        pieces = []
        by_split: dict[Any, list[float]] = {}
        for held in FOLDS:
            train = base.loc[(base["fold"] != held) & elig]
            test = base.loc[base["fold"] == held].copy()
            strata, fallback = estimate_shifts(train, method)
            test["prediction_kwh"] = apply_shifts(test, strata, fallback)
            pieces.append(test)
            for key, value in strata.items():
                by_split.setdefault(key, []).append(value)
        corrected = pd.concat(pieces, ignore_index=True)
        assert len(corrected) == len(base), "LOO 이어붙이기에서 행 수가 바뀌었다"

        score = official(corrected)
        gate = evaluate_gate(corrected, base)
        stats = gate.evidence
        complete = {k: v for k, v in by_split.items() if len(v) == len(FOLDS)}
        agree = [
            k for k, v in complete.items()
            if len({np.sign(x) for x in v}) == 1 and np.sign(v[0]) != 0
        ]
        agreement = len(agree) / len(complete) if complete else 0.0

        r1 = bool(score["total"] > incumbent_score["total"])
        r2 = bool(gate.passed)
        r3 = bool(agreement >= R3_MIN_SIGN_AGREEMENT)
        results[method] = {
            **score,
            "delta_vs_incumbent": score["total"] - incumbent_score["total"],
            "gate": {
                "passed": r2,
                "flags": {la.split()[0]: bool(ok) for la, ok in gate.conditions.items()},
                "positive_months": int(stats["positive_months"]),
                "months_scored": int(stats["months_scored"]),
                "sign_test_p": float(stats["sign_test_p_greater"]),
                "bootstrap_q05": float(stats["block_bootstrap_q05"]),
            },
            "strata_with_all_splits": len(complete),
            "sign_agreement": agreement,
            "shift_examples": [
                {"group": int(k[0]), "pred_band": k[1],
                 "shifts": [round(x, 4) for x in complete[k]]}
                for k in sorted(complete, key=lambda x: (x[0], str(x[1])))[:12]
            ],
            "R1_improves": r1,
            "R2_gate": r2,
            "R3_sign_stable": r3,
            "qualifies": bool(r1 and r2 and r3),
        }

    qualified = [m for m in FROZEN_PRIORITY if results[m]["qualifies"]]
    chosen = qualified[0] if qualified else None
    promoted_total = results[chosen]["total"] if chosen else incumbent_score["total"]

    verdict = (
        f"ELIGIBLE_MOS_PROMOTED_{chosen.upper()}" if chosen
        else "BIAS_CORRECTION_FAMILY_CLOSED_ON_CORRECT_POPULATION"
    )
    check = {
        "H1_expectation": "유효행 기준 이동 크기가 전체행 기준의 절반 미만",
        "H1_held": h1,
        "H2_expectation": "median_shift 가 R1·R2·R3 만족",
        "H2_held": results["median_shift"]["qualifies"],
        "H3_expectation": "metric_shift 가 R1·R2·R3 만족",
        "H3_held": results["metric_shift"]["qualifies"],
        "frozen_priority": list(FROZEN_PRIORITY),
        "qualified_methods": qualified,
        "chosen": chosen,
        "verdict": verdict,
    }

    payload = {
        "node": NODE_ID,
        "parent_node": PARENT_NODE,
        "supersedes": list(SUPERSEDES),
        "supersession_reason": "추정 모집단 오설정 (전체행 -> 유효행)",
        "gate_version": GATE_VERSION,
        "gate_modified": False,
        "eligible_rows": int(elig.sum()),
        "total_rows": len(base),
        "eligible_fraction": float(elig.mean()),
        "incumbent": INCUMBENT,
        "incumbent_score": incumbent_score,
        "population_diagnosis": diag,
        "methods": results,
        "predeclared_check": check,
        "promoted_total": promoted_total,
        "gap_to_target": 0.66 - promoted_total,
    }

    lines = [
        "# M271 P4 사이클 23 — 모집단을 고친 MOS 재시행",
        "",
        f"- 판정일: {datetime.now(UTC).strftime('%Y-%m-%d')} (UTC)",
        f"- 노드: `{NODE_ID}` / 레인 {LANE} / 부모 `{PARENT_NODE}`",
        f"- 대체: `{', '.join(SUPERSEDES)}` — 사유 **추정 모집단 오설정**",
        f"- 전체 {len(base):,} 행 중 유효행 {int(elig.sum()):,} "
        f"({elig.mean():.1%}). 공식 지표는 유효행만 채점한다.",
        f"- 기존 승격후보 `{INCUMBENT}` Total **{incumbent_score['total']:.6f}**",
        f"- 게이트: `{GATE_VERSION}` (읽기만 함)",
        "",
        "## 1. 모집단 진단 (H1)",
        "",
        "같은 추정량을 모집단만 바꿔 돌린다. 미채점 저출력 행이 이동을 끌고 있었다면",
        "유효행 기준 크기가 훨씬 작아야 한다.",
        "",
        "| 방법 | 층 | 전체행 \\|이동\\| 중앙값 | 유효행 \\|이동\\| 중앙값 | **크기비** |",
        "|---|---:|---:|---:|---:|",
    ]
    for d in diag:
        lines.append(
            f"| `{d['method']}` | {d['strata_compared']} | "
            f"{d['median_abs_shift_all_rows']:.4f} | {d['median_abs_shift_eligible']:.4f} | "
            f"**{d['median_magnitude_ratio']:.3f}** |"
        )
    lines += ["", "층별 대조 (`median_shift`, 전체행 -> 유효행):", "",
              "| 그룹 | 예측대역 | 전체행 | 유효행 |", "|---:|---|---:|---:|"]
    for e in diag[0]["examples"]:
        lines.append(
            f"| {e['group']} | {e['pred_band']} | `{e['all_rows']:+.4f}` | "
            f"`{e['eligible']:+.4f}` |"
        )

    lines += [
        "",
        "## 2. 유효행 추정 + leave-one-fold-out (H2 · H3)",
        "",
        "| 방법 | Total | 1-NMAE | FICR | 기존대비 | G1G2G3G4 | 양수월 | q05 "
        "| R1 | R2 | R3 | 자격 |",
        "|---|---:|---:|---:|---:|:---:|---:|---:|:---:|:---:|:---:|:---:|",
        f"| `{INCUMBENT}` (기존) | {incumbent_score['total']:.6f} | "
        f"{incumbent_score['one_minus_nmae']:.6f} | {incumbent_score['ficr']:.6f} | "
        "— | — | — | — | — | — | — | — |",
    ]
    for method, r in results.items():
        g = r["gate"]
        flags = "".join("O" if g["flags"].get(x) else "-" for x in ("G1", "G2", "G3", "G4"))
        lines.append(
            f"| `{method}` | {r['total']:.6f} | {r['one_minus_nmae']:.6f} | {r['ficr']:.6f} | "
            f"{r['delta_vs_incumbent']:+.6f} | `{flags}` | "
            f"{g['positive_months']}/{g['months_scored']} | {g['bootstrap_q05']:+.6f} | "
            f"{'O' if r['R1_improves'] else 'X'} | {'O' if r['R2_gate'] else 'X'} | "
            f"{'O' if r['R3_sign_stable'] else 'X'} | "
            f"{'**자격**' if r['qualifies'] else '미달'} |"
        )

    lines += ["", "층별 이동의 분할간 안정성:", ""]
    for method, r in results.items():
        lines += [
            f"### `{method}` — 일치율 **{r['sign_agreement']:.3f}** "
            f"({r['strata_with_all_splits']} 층)",
            "",
            "| 그룹 | 예측대역 | 세 분할의 이동 |",
            "|---:|---|---|",
        ]
        for e in r["shift_examples"]:
            lines.append(
                f"| {e['group']} | {e['pred_band']} | "
                + " / ".join(f"`{x:+.4f}`" for x in e["shifts"]) + " |"
            )
        lines.append("")

    lines += [
        "## 3. 사전확약 대조",
        "",
        f"- H1 `{check['H1_expectation']}` -> **{h1}**",
        f"- H2 `{check['H2_expectation']}` -> **{check['H2_held']}**",
        f"- H3 `{check['H3_expectation']}` -> **{check['H3_held']}**",
        "",
        f"동결 우선순위 `{' > '.join(FROZEN_PRIORITY)}` / 자격 `{qualified or '없음'}` "
        f"-> 선택 `{chosen or '없음'}`",
        "",
        f"판정: **{verdict}**",
        "",
        f"승격 Total **{promoted_total:.6f}**, 목표 0.66 까지 **{0.66 - promoted_total:+.6f}**.",
        "",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    receipt: dict[str, Any] = {
        "schema_version": 1,
        "stage": "M271_P4_CYCLE23_ELIGIBLE_MOS",
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

    print(f"[C23] 유효행 {int(elig.sum()):,}/{len(base):,} ({elig.mean():.1%})")
    for d in diag:
        print(f"[C23] {d['method']:>14} 이동크기 전체행 {d['median_abs_shift_all_rows']:.4f} "
              f"-> 유효행 {d['median_abs_shift_eligible']:.4f} "
              f"(비 {d['median_magnitude_ratio']:.3f})")
    print(f"[C23] H1 모집단 오설정 확인 = {h1}")
    for method, r in results.items():
        print(f"[C23] {method:>14}  Total {r['total']:.6f}  "
              f"기존대비 {r['delta_vs_incumbent']:+.6f}  "
              f"게이트 {'통과' if r['R2_gate'] else '기각'}  "
              f"부호일치 {r['sign_agreement']:.3f}  자격 {r['qualifies']}")
    print(f"[C23] 판정: {verdict}  ->  Total {promoted_total:.6f} "
          f"(목표까지 {0.66 - promoted_total:+.6f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
