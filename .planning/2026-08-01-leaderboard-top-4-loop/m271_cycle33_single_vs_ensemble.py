"""M271 P4 사이클 33 — 챔피언은 자기 앙상블의 최고 멤버를 이기는가.

사이클 32 가 lockbox 2 차 소비로 낸 결과가 예상 밖의 것을 가리켰다. 2024 에서 median 도
mean 도 **최고 단일 모델**에 크게 졌다(-0.0124 / -0.0116). 그래서 2023 을 다시 보니
같은 일이 작은 규모로 일어나 있었다.

    M244_ANALOG_Q234   0.605760
    M102_TOP100        0.630589
    mean(4)            0.632143
    M113_LGBM_DART     0.636376
    **median(4) 챔피언  0.636597**
    **M115_XGBOOST     0.638410**   <- 챔피언보다 +0.001813 높다

**나는 챔피언을 배포 정책(0.628605)하고만 비교했지 자기 앙상블의 최고 멤버와 비교한 적이
없다.** Breiman (1996) 의 스태킹 논거가 정확히 "비음 제약 결합은 **최고 단일 선택보다**
낫다" 인데 그 검사를 빠뜨렸다. 사이클 19·25 에서 Breiman 을 인용해놓고 그 주장의 핵심
대조를 안 한 것이다.

이 노드가 그 누락을 메운다.

① 방법 리서치 (실행 전)
  - Breiman (1996) — 비음·합1 제약이 결합을 보간적으로 만들고, 그 제약이 **최고 단일
    예측기 선택보다 나은 정확도**의 이유라고 논증한다. 이것이 대조의 근거다.
  - **선택 편향 대칭성**이 이 노드의 방법론적 쟁점이다. "최고 단일" 은 4 개 중 최댓값을
    같은 fold 에서 사후 선택한 값이라 **위로 편향**돼 있다. 결합은 그렇지 않다. 따라서
    단순 비교는 결합에 불리하다. 공정하게 하려면 두 가지를 함께 봐야 한다.
      (a) 사후 최고 단일 vs 챔피언  — 결합에 불리한 상한 대조
      (b) **fold 별로 최고 단일이 바뀌는가** — 바뀌면 "최고 단일" 은 배포 가능한 규칙이
          아니고, 그 편향 크기가 곧 결합의 존재 이유다
  - 게이트 검출문턱은 `+0.001013` 이고 관측 차이 `-0.001813` 은 그보다 크므로 검정 가능하다.

② 사양 동결

  사전확약(실행 전 동결):
    H1  `M115_XGBOOST` 가 챔피언을 부모로 한 **동결 게이트를 통과**한다.
        (통과하면 챔피언은 자기 멤버 하나에 유의하게 진다)
    H2  최고 단일 멤버가 **세 fold 에서 동일**하다. 동일하면 "최고 단일" 이 안정적 규칙이고,
        바뀌면 사후 선택의 산물이다.
    H3  챔피언이 **fold 별 최고 단일의 평균**보다 낮다. (사후 선택 편향을 fold 단위로
        완화한 대조 — 각 fold 에서 그 fold 의 최고를 골랐을 때의 성적)
    H4  챔피언이 **모든 개별 멤버**를 부모로 한 게이트를 통과한다. 즉 어느 멤버 대비로도
        유의하게 낫다.

  H1 이 성립하고 H2 도 성립하면 **챔피언을 M115 로 교체하는 것이 정당**하다.
  H1 이 성립하지만 H2 가 기각되면 "최고 단일" 은 배포 불가능한 사후 규칙이고 챔피언을
  유지하되 이 열세를 명시 기록한다.

**게이트를 수정하지 않는다.** 2023 데이터만 쓴다. lockbox 미사용.
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
from m270_monthly_validation import load_predictions
from m271_cycle13_ensemble import ENSEMBLES
from m271_cycle17_combiner import combine, stack_members
from m271_cycle21_mos import QUARTER_OF_MONTH
from m271_evaluate_candidate import official

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
REPORT_MD = REPORTS / "m271_cycle33_single_vs_ensemble.md"
RECEIPT = REPORTS / "m271_cycle33_single_vs_ensemble_receipt.json"

NODE_ID = "C1N33_SINGLE_VS_ENSEMBLE"
LANE = "L7"
PARENT_NODE = "C1N32_LOCKBOX_OPERATOR"
CHAMPION = "M271_MEDIAN4"
BASE_ENSEMBLE = "E3_FOUR_FAMILY"
DEPLOYED = "T0.5_G1.5"
FOLDS = ("Q2", "Q3", "Q4")
GATE_DETECTION_THRESHOLD = 0.001013


def member_frame(stacked: pd.DataFrame, index: int) -> pd.DataFrame:
    out = stacked.loc[
        :, ["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh", "month"]
    ].copy()
    out["prediction_kwh"] = stacked[f"m{index}"]
    return out


def by_fold_total(frame: pd.DataFrame) -> dict[str, float]:
    f = frame.copy()
    f["fold"] = f["month"].map(QUARTER_OF_MONTH)
    return {
        fold: official(cell)["total"]
        for fold, cell in f.groupby("fold", observed=True)
        if fold in FOLDS
    }


def gate_row(candidate: pd.DataFrame, reference: pd.DataFrame) -> dict[str, Any]:
    gate = evaluate_gate(candidate, reference)
    stats = gate.evidence
    return {
        "passed": bool(gate.passed),
        "flags": {la.split()[0]: bool(ok) for la, ok in gate.conditions.items()},
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
    champion = combine(stacked, k, "median")
    champion_score = official(champion)
    deployed = load_predictions(DEPLOYED)
    deployed_score = official(deployed)

    singles = {}
    for i, name in enumerate(members):
        frame = member_frame(stacked, i)
        singles[name] = {
            "frame": frame,
            "score": official(frame),
            "by_fold": by_fold_total(frame),
        }
    champion_folds = by_fold_total(champion)

    pooled_best = max(singles, key=lambda n: singles[n]["score"]["total"])
    best_by_fold = {
        fold: max(singles, key=lambda n: singles[n]["by_fold"][fold]) for fold in FOLDS
    }
    h2 = bool(len(set(best_by_fold.values())) == 1)

    # H3: 각 fold 에서 그 fold 의 최고를 고른 오라클 선택기
    oracle_fold_pick = {
        fold: singles[best_by_fold[fold]]["by_fold"][fold] for fold in FOLDS
    }
    champion_fold_mean = sum(champion_folds[f] for f in FOLDS) / len(FOLDS)
    oracle_fold_mean = sum(oracle_fold_pick[f] for f in FOLDS) / len(FOLDS)
    h3 = bool(champion_fold_mean < oracle_fold_mean)

    gates_vs_champion = {
        name: gate_row(singles[name]["frame"], champion) for name in members
    }
    gates_champion_vs = {
        name: gate_row(champion, singles[name]["frame"]) for name in members
    }
    h1 = bool(gates_vs_champion[pooled_best]["passed"])
    h4 = all(g["passed"] for g in gates_champion_vs.values())

    if h1 and h2:
        verdict = "REPLACE_CHAMPION_WITH_BEST_SINGLE"
    elif h1:
        verdict = "BEST_SINGLE_WINS_BUT_NOT_A_DEPLOYABLE_RULE"
    elif h4:
        verdict = "CHAMPION_DOMINATES_ALL_MEMBERS"
    else:
        verdict = "CHAMPION_HELD_NEITHER_SIDE_SIGNIFICANT"

    check = {
        "H1_expectation": f"최고단일({pooled_best})이 챔피언 부모 게이트를 통과",
        "H1_held": h1,
        "H2_expectation": "최고 단일 멤버가 세 fold 에서 동일",
        "H2_held": h2, "H2_measured": best_by_fold,
        "H3_expectation": "챔피언 < fold 별 최고단일 평균",
        "H3_held": h3,
        "H3_champion_fold_mean": champion_fold_mean,
        "H3_oracle_fold_mean": oracle_fold_mean,
        "H4_expectation": "챔피언이 모든 개별 멤버 부모 게이트를 통과",
        "H4_held": h4,
        "verdict": verdict,
    }

    payload = {
        "node": NODE_ID, "parent_node": PARENT_NODE, "champion": CHAMPION,
        "gate_version": GATE_VERSION, "gate_modified": False,
        "lockbox_used": False,
        "omission_being_corrected": "챔피언을 배포 정책하고만 비교했고 자기 앙상블의 "
                                    "최고 멤버와 비교한 적이 없다 (Breiman 1996 의 핵심 대조)",
        "deployed": {"policy": DEPLOYED, **deployed_score},
        "champion_score": champion_score,
        "champion_by_fold": champion_folds,
        "members": {
            name: {
                **singles[name]["score"],
                "by_fold": singles[name]["by_fold"],
                "delta_vs_champion": singles[name]["score"]["total"]
                - champion_score["total"],
                "gate_vs_champion": gates_vs_champion[name],
                "champion_gate_vs_it": gates_champion_vs[name],
            }
            for name in members
        },
        "pooled_best_single": pooled_best,
        "best_single_by_fold": best_by_fold,
        "oracle_fold_pick_mean": oracle_fold_mean,
        "champion_fold_mean": champion_fold_mean,
        "detection_threshold": GATE_DETECTION_THRESHOLD,
        "predeclared_check": check,
    }

    lines = [
        "# M271 P4 사이클 33 — 챔피언은 자기 앙상블의 최고 멤버를 이기는가",
        "",
        f"- 판정일: {datetime.now(UTC).strftime('%Y-%m-%d')} (UTC)",
        f"- 노드: `{NODE_ID}` / 레인 {LANE} / 부모 `{PARENT_NODE}`",
        f"- 게이트: `{GATE_VERSION}` (읽기만 함) / **lockbox 미사용, 2023 만**",
        "",
        "## 0. 메우는 누락",
        "",
        payload["omission_being_corrected"] + ".",
        "",
        "## 1. 2023 전체 (같은 행집합)",
        "",
        "| 대상 | Total | 1-NMAE | FICR | 챔피언 대비 |",
        "|---|---:|---:|---:|---:|",
        f"| 배포 `{DEPLOYED}` | {deployed_score['total']:.6f} | "
        f"{deployed_score['one_minus_nmae']:.6f} | {deployed_score['ficr']:.6f} | "
        f"{deployed_score['total'] - champion_score['total']:+.6f} |",
    ]
    for name in members:
        v = payload["members"][name]
        mark = " **(최고단일)**" if name == pooled_best else ""
        lines.append(
            f"| `{name}`{mark} | {v['total']:.6f} | {v['one_minus_nmae']:.6f} | "
            f"{v['ficr']:.6f} | **{v['delta_vs_champion']:+.6f}** |"
        )
    lines.append(
        f"| **`{CHAMPION}`** | **{champion_score['total']:.6f}** | "
        f"{champion_score['one_minus_nmae']:.6f} | {champion_score['ficr']:.6f} | — |"
    )

    lines += [
        "",
        "## 2. 개별 멤버 대비 동결 게이트 (H1 · H4)",
        "",
        "양방향으로 잰다. 어느 쪽이 유의한지가 판정을 가른다.",
        "",
        "| 멤버 | 멤버가 챔피언 부모로 | 챔피언이 멤버 부모로 |",
        "|---|:---:|:---:|",
    ]
    for name in members:
        a = gates_vs_champion[name]
        b = gates_champion_vs[name]
        fa = "".join("O" if a["flags"].get(x) else "-" for x in ("G1", "G2", "G3", "G4"))
        fb = "".join("O" if b["flags"].get(x) else "-" for x in ("G1", "G2", "G3", "G4"))
        lines.append(
            f"| `{name}` | `{fa}` {a['positive_months']}/{a['months_scored']} "
            f"q05={a['bootstrap_q05']:+.6f} **{'통과' if a['passed'] else '기각'}** "
            f"| `{fb}` {b['positive_months']}/{b['months_scored']} "
            f"q05={b['bootstrap_q05']:+.6f} **{'통과' if b['passed'] else '기각'}** |"
        )

    lines += [
        "",
        "## 3. 최고 단일은 배포 가능한 규칙인가 (H2 · H3)",
        "",
        "\"최고 단일\" 은 같은 fold 에서 최댓값을 사후 선택한 값이라 위로 편향돼 있다.",
        "fold 별로 승자가 바뀌면 그건 규칙이 아니라 사후 산물이다.",
        "",
        "| fold | 최고 단일 | 그 값 | 챔피언 |",
        "|---|---|---:|---:|",
    ]
    for fold in FOLDS:
        lines.append(
            f"| {fold} | `{best_by_fold[fold]}` | {oracle_fold_pick[fold]:.6f} | "
            f"{champion_folds[fold]:.6f} |"
        )
    lines += [
        "",
        f"- 세 fold 승자 동일: **{h2}** ({sorted(set(best_by_fold.values()))})",
        f"- fold 평균 — 오라클 선택 {oracle_fold_mean:.6f} vs 챔피언 "
        f"{champion_fold_mean:.6f} (차이 {oracle_fold_mean - champion_fold_mean:+.6f})",
        "",
        "## 4. 사전확약 대조",
        "",
        f"- H1 `{check['H1_expectation']}` -> **{h1}**",
        f"- H2 `{check['H2_expectation']}` -> **{h2}**",
        f"- H3 `{check['H3_expectation']}` -> **{h3}**",
        f"- H4 `{check['H4_expectation']}` -> **{h4}**",
        "",
        f"판정: **{verdict}**",
        "",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    receipt: dict[str, Any] = {
        "schema_version": 1,
        "stage": "M271_P4_CYCLE33_SINGLE_VS_ENSEMBLE",
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

    for name in members:
        v = payload["members"][name]
        a = gates_vs_champion[name]
        b = gates_champion_vs[name]
        print(f"[C33] {name:<42} {v['total']:.6f} "
              f"(챔피언대비 {v['delta_vs_champion']:+.6f})  "
              f"멤버>챔피언 {'통과' if a['passed'] else '기각'} / "
              f"챔피언>멤버 {'통과' if b['passed'] else '기각'}")
    print(f"[C33] 챔피언 {champion_score['total']:.6f} / 최고단일 {pooled_best}")
    print(f"[C33] fold 별 최고단일 {best_by_fold} -> H2 {h2}")
    print(f"[C33] H1 {h1} | H2 {h2} | H3 {h3} | H4 {h4}")
    print(f"[C33] 판정: {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
