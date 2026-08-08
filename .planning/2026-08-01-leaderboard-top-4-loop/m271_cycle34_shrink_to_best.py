"""M271 P4 사이클 34 — 최고 단일에 결합이 무언가를 더하는가 (기준선 M115).

사이클 33 이 기준선을 바꿨다. `M115_XGBOOST` 단독이 로컬 `0.638410` 으로 4 모델 median
챔피언(`0.636597`)보다 높고, 배포 대비 게이트를 **9/9 월 전부 양수**로 통과한다. 세 fold
각각에서 최고 단일이기도 하다. 이제부터 기준선은 M115 다.

지금까지의 결합 실험은 전부 **배포 정책(0.628605)** 을 닻으로 놨다. M115 를 닻으로 놓고
나머지 쪽으로 수축시키는 것은 시험한 적이 없다. 그것이 Breiman (1996) 의 원래 질문이다 —
비음 제약 결합이 **최고 단일 선택** 에 무언가를 더하는가.

① 방법 리서치 (실행 전)
  - Breiman (1996) *Stacked Regressions* — 비음·합1 가중치는 결합을 보간적으로 만들고,
    그 제약이 최고 단일 예측기 선택보다 나은 정확도의 이유다. `(1-b)*M115 + b*X` 는
    정확히 그 형태이며 `b` 하나만 자유롭다.
  - 사이클 20 의 교훈을 사양에 반영한다: **결합자를 바꾸면 사다리를 다시 돌려야 한다.**
    닻을 바꾼 것도 같은 종류의 변경이므로 사다리를 끝점까지 판다.
  - forecast combination puzzle: 자유 가중치는 단순평균에 지곤 한다. 그래서 자유도를
    `b` **하나**로 묶고 `X` 는 사전 선언된 셋만 쓴다.

② 사양 동결

  닻    `M115_XGBOOST`
  상대  X1 `median(나머지 3)` (M102 · M244 · M113)
        X2 `median(4 전부)`   (= 구 챔피언 M271_MEDIAN4)
        X3 `mean(나머지 3)`
  사다리 b in {0.00, 0.15, 0.30, 0.50, 0.70, 1.00}. b=0 은 M115 자신이다.

  **다중검정 명시**: 3 x 5 = 15 개 조합을 본다. 통제는 동결 게이트 자체다 — 사이클 7 이
  62 개 정책을 전부 기각시킨 그 게이트이고, 월별 일관성(G1)·부트스트랩 하한(G3)·최악월
  하한(G4)을 동시에 요구한다. pooled 개선만으로는 통과하지 못한다.

  **승격 규칙 (실행 전 동결)** — 최고점을 고르지 않는다.
    R1  M115 대비 pooled Total 개선.
    R2  M115 를 부모로 한 **동결 게이트 통과**.
    R3  세 fold 각각에서 M115 보다 높다 (사이클 29 의 잣대).
  셋을 다 만족하는 조합이 복수면 **b 가 가장 작은 것**을 택한다(단순한 쪽 우선).
  b 동률이면 X1 < X2 < X3 순서로 끊는다.

  사전확약:
    H1  자격을 얻는 조합이 **적어도 하나** 있다.
    H2  그 이득이 게이트 검출문턱 `+0.001013` 을 넘는다.
    H3  최적 b 가 사다리 **내부**(0 < b < 1)다. 끝점이면 그건 결합이 아니라 교체다.
  H1 이 기각되면 **최고 단일에 결합이 더할 것이 없다** 가 확정되고, M115 단독이 최종
  후보로 남는다.

**게이트를 수정하지 않는다.** 2023 만 쓴다. lockbox 미사용.
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
from m271_cycle13_ensemble import ENSEMBLES
from m271_cycle17_combiner import stack_members
from m271_cycle21_mos import QUARTER_OF_MONTH
from m271_evaluate_candidate import official

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
REPORT_MD = REPORTS / "m271_cycle34_shrink_to_best.md"
RECEIPT = REPORTS / "m271_cycle34_shrink_to_best_receipt.json"

NODE_ID = "C1N34_SHRINK_TO_BEST"
LANE = "L7"
PARENT_NODE = "C1N33_SINGLE_VS_ENSEMBLE"
ANCHOR = "M115_XGBOOST"
BASE_ENSEMBLE = "E3_FOUR_FAMILY"
FOLDS = ("Q2", "Q3", "Q4")
BETAS = (0.00, 0.15, 0.30, 0.50, 0.70, 1.00)
GATE_DETECTION_THRESHOLD = 0.001013
PARTNER_ORDER = ("X1_median_rest3", "X2_median_all4", "X3_mean_rest3")


def build_parts(stacked: pd.DataFrame, members: tuple[str, ...]) -> dict[str, np.ndarray]:
    anchor_index = members.index(ANCHOR)
    others = [i for i in range(len(members)) if i != anchor_index]
    arr = stacked.loc[:, [f"m{i}" for i in range(len(members))]].to_numpy(dtype="float64")
    rest = arr[:, others]
    return {
        "anchor": arr[:, anchor_index],
        "X1_median_rest3": np.median(rest, axis=1),
        "X2_median_all4": np.median(arr, axis=1),
        "X3_mean_rest3": rest.mean(axis=1),
    }


def frame_from(stacked: pd.DataFrame, prediction: np.ndarray) -> pd.DataFrame:
    out = stacked.loc[
        :, ["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh", "month"]
    ].copy()
    out["prediction_kwh"] = prediction
    return out


def by_fold_total(frame: pd.DataFrame) -> dict[str, float]:
    f = frame.copy()
    f["fold"] = f["month"].map(QUARTER_OF_MONTH)
    return {
        fold: official(cell)["total"]
        for fold, cell in f.groupby("fold", observed=True)
        if fold in FOLDS
    }


def main() -> int:
    members = ENSEMBLES[BASE_ENSEMBLE]
    stacked = stack_members(members)
    parts = build_parts(stacked, members)

    anchor_frame = frame_from(stacked, parts["anchor"])
    anchor_score = official(anchor_frame)
    anchor_folds = by_fold_total(anchor_frame)

    rows: list[dict[str, Any]] = []
    for partner in PARTNER_ORDER:
        for beta in BETAS:
            prediction = (1.0 - beta) * parts["anchor"] + beta * parts[partner]
            frame = frame_from(stacked, prediction)
            score = official(frame)
            gate = evaluate_gate(frame, anchor_frame)
            stats = gate.evidence
            folds = by_fold_total(frame)
            per_fold_wins = {f: bool(folds[f] > anchor_folds[f]) for f in FOLDS}
            r1 = bool(score["total"] > anchor_score["total"])
            r2 = bool(gate.passed)
            r3 = all(per_fold_wins.values())
            rows.append(
                {
                    "partner": partner,
                    "beta": beta,
                    **score,
                    "delta_vs_anchor": score["total"] - anchor_score["total"],
                    "by_fold": folds,
                    "per_fold_wins": per_fold_wins,
                    "gate": {
                        "passed": r2,
                        "flags": {la.split()[0]: bool(ok)
                                  for la, ok in gate.conditions.items()},
                        "positive_months": int(stats["positive_months"]),
                        "months_scored": int(stats["months_scored"]),
                        "sign_test_p": float(stats["sign_test_p_greater"]),
                        "bootstrap_q05": float(stats["block_bootstrap_q05"]),
                        "min_delta": float(stats["min_total_delta"]),
                    },
                    "R1_improves": r1,
                    "R2_gate": r2,
                    "R3_all_folds": r3,
                    "qualifies": bool(r1 and r2 and r3 and beta > 0.0),
                }
            )

    qualified = [r for r in rows if r["qualifies"]]
    qualified.sort(key=lambda r: (r["beta"], PARTNER_ORDER.index(r["partner"])))
    chosen = qualified[0] if qualified else None

    h1 = bool(chosen)
    h2 = bool(chosen and chosen["delta_vs_anchor"] > GATE_DETECTION_THRESHOLD)
    h3 = bool(chosen and 0.0 < chosen["beta"] < 1.0)
    promoted_total = chosen["total"] if chosen else anchor_score["total"]
    promoted_name = (
        f"M271_SHRINK_{chosen['partner']}_b{chosen['beta']:.2f}" if chosen else ANCHOR
    )
    verdict = (
        "COMBINATION_ADDS_TO_BEST_SINGLE" if h1
        else "NOTHING_TO_ADD_BEST_SINGLE_STANDS"
    )

    check = {
        "H1_expectation": "자격을 얻는 (X, b) 조합이 적어도 하나",
        "H1_held": h1,
        "H2_expectation": f"이득이 검출문턱 {GATE_DETECTION_THRESHOLD} 초과",
        "H2_held": h2,
        "H3_expectation": "최적 b 가 사다리 내부 (0 < b < 1)",
        "H3_held": h3,
        "promotion_rule_frozen_before_run": [
            "R1 M115 대비 Total 개선", "R2 M115 부모 동결 게이트 통과",
            "R3 세 fold 각각 M115 초과", "복수 자격시 b 최소, 동률시 X1<X2<X3",
        ],
        "multiplicity": f"{len(PARTNER_ORDER)} x {len(BETAS)} = {len(rows)} 조합. "
                        "통제는 동결 게이트 자체(사이클 7 이 62 개 정책을 전부 기각)",
        "chosen": promoted_name if chosen else None,
        "verdict": verdict,
    }

    payload = {
        "node": NODE_ID, "parent_node": PARENT_NODE,
        "anchor": ANCHOR, "gate_version": GATE_VERSION, "gate_modified": False,
        "lockbox_used": False,
        "baseline_change": "기준선을 배포 정책(0.628605)에서 M115(0.638410)로 바꿨다. "
                           "지금까지의 결합 실험은 전부 낮은 닻에서 판정된 것이다",
        "anchor_score": anchor_score,
        "anchor_by_fold": anchor_folds,
        "ladder": rows,
        "qualified": [
            {"partner": r["partner"], "beta": r["beta"], "total": r["total"],
             "delta": r["delta_vs_anchor"]} for r in qualified
        ],
        "predeclared_check": check,
        "promoted": promoted_name,
        "promoted_total": promoted_total,
        "gap_to_local_target": 0.66 - promoted_total,
        "gap_to_offset_implied_local": 0.638881 - promoted_total,
    }

    lines = [
        "# M271 P4 사이클 34 — 최고 단일에 결합이 무언가를 더하는가",
        "",
        f"- 판정일: {datetime.now(UTC).strftime('%Y-%m-%d')} (UTC)",
        f"- 노드: `{NODE_ID}` / 레인 {LANE} / 부모 `{PARENT_NODE}`",
        f"- **닻 `{ANCHOR}`** Total **{anchor_score['total']:.6f}** "
        f"(1-NMAE {anchor_score['one_minus_nmae']:.6f}, FICR {anchor_score['ficr']:.6f})",
        f"- 게이트: `{GATE_VERSION}` (읽기만 함) / lockbox 미사용 / 2023 만",
        "",
        "## 0. 기준선 변경",
        "",
        payload["baseline_change"] + ".",
        "",
        f"다중검정: {check['multiplicity']}.",
        "",
        "## 1. 승격 규칙 (실행 전 동결)",
        "",
    ]
    for r in check["promotion_rule_frozen_before_run"]:
        lines.append(f"- {r}")
    lines += [
        "",
        "## 2. 수축 사다리",
        "",
        "`pred = (1-b) * M115 + b * X`. b=0 은 M115 자신이다.",
        "",
        "| X | b | Total | 닻 대비 | G1G2G3G4 | 양수월 | q05 | 최소월 | fold 3/3 | 자격 |",
        "|---|---:|---:|---:|:---:|---:|---:|---:|:---:|:---:|",
    ]
    for r in rows:
        g = r["gate"]
        flags = "".join("O" if g["flags"].get(x) else "-" for x in ("G1", "G2", "G3", "G4"))
        wins = sum(r["per_fold_wins"].values())
        mark = "**" if r["qualifies"] else ""
        lines.append(
            f"| `{r['partner']}` | {mark}{r['beta']:.2f}{mark} | "
            f"{mark}{r['total']:.6f}{mark} | {r['delta_vs_anchor']:+.6f} | `{flags}` | "
            f"{g['positive_months']}/{g['months_scored']} | {g['bootstrap_q05']:+.6f} | "
            f"{g['min_delta']:+.6f} | {wins}/3 | "
            f"{'**자격**' if r['qualifies'] else '-'} |"
        )

    lines += [
        "",
        "## 3. 사전확약 대조",
        "",
        f"- H1 `{check['H1_expectation']}` -> **{h1}** (자격 {len(qualified)} 개)",
        f"- H2 `{check['H2_expectation']}` -> **{h2}**",
        f"- H3 `{check['H3_expectation']}` -> **{h3}**",
        "",
        f"판정: **{verdict}**",
        "",
        f"승격: **`{promoted_name}`** Total **{promoted_total:.6f}**",
        f"- 로컬 목표 0.66 까지 {0.66 - promoted_total:+.6f}",
        f"- 오프셋 함의 로컬(온라인 0.66 낙관 요건 0.638881) 까지 "
        f"**{0.638881 - promoted_total:+.6f}**",
        "",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    receipt: dict[str, Any] = {
        "schema_version": 1,
        "stage": "M271_P4_CYCLE34_SHRINK_TO_BEST",
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

    print(f"[C34] 닻 {ANCHOR} {anchor_score['total']:.6f}")
    for r in rows:
        if r["beta"] == 0.0:
            continue
        g = r["gate"]
        print(f"[C34] {r['partner']:>16} b={r['beta']:.2f}  {r['total']:.6f}  "
              f"({r['delta_vs_anchor']:+.6f})  게이트 "
              f"{'통과' if g['passed'] else '기각'}  "
              f"fold {sum(r['per_fold_wins'].values())}/3  "
              f"자격 {r['qualifies']}")
    print(f"[C34] H1 {h1} | H2 {h2} | H3 {h3}")
    print(f"[C34] 판정: {verdict}  ->  `{promoted_name}` {promoted_total:.6f} "
          f"(오프셋 함의 로컬까지 {0.638881 - promoted_total:+.6f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
