"""M271 P4 사이클 21 — 측정된 조건부 편향의 fold-외 보정 (MOS).

사이클 19 의 순위 히스토그램이 결함을 하나 특정했다. 앙상블 median 이 **57.8% 의 행에서
과대예측**한다(무편향이면 50%). 이상치 비대칭 0.1245 는 Hamill 의 판별 기준으로 과소분산이
아니라 **조건부 편향**이다.

조건부 편향은 추측이 아니라 측정된 결함이고, 그것을 지우는 것은 표준 후처리다.

① 방법 리서치 (실행 전)
  - MOS (Glahn & Lowry 1972) — NWP 산출의 계통 편향을 과거 관측에 맞춰 보정하는 표준.
  - EMOS / NGR (Gneiting et al. 2005) — 앙상블판. 다만 **CRPS(적정 채점규칙)를 최적화**
    한다. 우리 손실은 밴드 히트율이라 적정 채점규칙이 아니다. 그대로 쓰면 목적함수가
    어긋난다.
  - 풍력 편향 보정 문헌은 복잡지형·해륙대비에서 편향이 커진다고 본다. A3 가 이미 이
    사이트에서 그 조건을 쟀다(17 기, 1000m 능선, 2km 사이트, 해안).
  - 채택: 층별 이동(shift) 보정을 **두 목적함수로** 각각 추정한다.
      M1 `median_shift`  잔차 중앙값 — 표준 MOS. 조건부 중앙값을 겨냥(NMAE 정합)
      M2 `metric_shift`  층의 공식 Total 기여를 최대화하는 이동 — 지표정합
    기각: CRPS 목적(손실 불일치), 같은 fold 추정(선택 편향).

② 사양 동결

  **fold 하나씩 빼고 추정한다.** 3 개 fold(2023 Q2/Q3/Q4) 중 둘에서 층별 이동을 추정하고
  나머지 하나에 적용한다. 세 조각을 이어붙이면 **전량이 fold-외** 예측이다.

  층 = 그룹 x **예측**대역. 예보시점 가용 변수만 쓴다. 학습 fold 에서 층의 행이 100 개
  미만이면 그룹 수준 이동으로, 그것도 없으면 0 으로 후퇴한다(사전 선언).

  **승격 규칙 (실행 전 동결)** — 두 방법 중 좋은 쪽을 고르지 않는다. 각 방법이 독립적으로
  다음을 모두 만족해야 승격 자격이 생긴다.
    R1  pooled Total 이 `M271_MEDIAN4` 대비 개선.
    R2  `M271_MEDIAN4` 를 부모로 한 **동결 게이트 통과**.
    R3  층별 이동의 **부호가 세 LOO 분할에서 일치**하는 층이 70% 이상.
        (부호가 분할마다 뒤집히면 편향이 아니라 잡음이다)
  둘 다 자격을 얻으면 **M1 을 택한다** — 더 표준적이고 덜 맞춘 추정량이기 때문이며,
  이 우선순위도 실행 전에 정한다.

  사전확약:
    H1  M1 이 세 조건을 모두 만족한다.
    H2  M2 가 세 조건을 모두 만족한다.
    H3  R3 부호 일치율이 두 방법 모두에서 0.70 이상이다.

**게이트를 수정하지 않는다.** 읽기만 한다.

읽기 전용. 모델을 적합하지 않고(층별 상수 이동은 모델 적합이 아니다) 2024 행을 읽지 않는다.
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
from m271_cycle17_combiner import combine, stack_members
from m271_evaluate_candidate import official
from m271_n0_deficit_init import Y_BAND_EDGES

from baram.evaluation.official import settlement_unit

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
REPORT_MD = REPORTS / "m271_cycle21_mos.md"
RECEIPT = REPORTS / "m271_cycle21_mos_receipt.json"

NODE_ID = "C1N21_MOS_BIAS_CORRECTION"
LANE = "L1"  # 데이터전처리/후처리
PARENT_NODE = "C1N20_ALPHA_ENDPOINT"
INCUMBENT = "M271_MEDIAN4"
BASE_ENSEMBLE = "E3_FOUR_FAMILY"
COMBINER = "median"

QUARTER_OF_MONTH = {
    "2023-04": "Q2", "2023-05": "Q2", "2023-06": "Q2",
    "2023-07": "Q3", "2023-08": "Q3", "2023-09": "Q3",
    "2023-10": "Q4", "2023-11": "Q4", "2023-12": "Q4",
}
FOLDS = ("Q2", "Q3", "Q4")
MIN_STRATUM_TRAIN_ROWS = 100
SHIFT_GRID = np.round(np.arange(-0.15, 0.15 + 1e-9, 0.005), 4)  # 용량 단위
R3_MIN_SIGN_AGREEMENT = 0.70

METHOD_SOURCES = (
    {
        "id": "glahn_lowry_1972_mos",
        "cite": "Glahn & Lowry (1972) — Model Output Statistics",
        "claim": "NWP 산출의 계통 편향을 과거 관측에 맞춰 통계적으로 보정하는 표준 절차",
        "applicability": "directly_supported",
        "use": "층별 이동 보정의 근거",
    },
    {
        "id": "gneiting_2005_emos",
        "cite": "Gneiting et al. (2005) — EMOS / NGR",
        "claim": "앙상블 후처리의 표준이나 **CRPS(적정 채점규칙)** 를 최적화한다",
        "applicability": "near_match_only",
        "use": "목적함수가 우리 손실(밴드 히트율)과 어긋나므로 CRPS 목적은 기각하고 "
               "지표정합 목적 M2 를 별도로 둔다",
    },
    {
        "id": "wind_bias_complex_terrain",
        "cite": "풍력 편향보정 문헌 — 복잡지형·해륙대비에서 편향 증대",
        "claim": "지형 복잡도와 해안 효과가 계통 편향을 키운다",
        "applicability": "directly_supported",
        "use": "A3 가 이 사이트에서 해당 조건을 이미 확인 — 편향 존재의 사전 개연성",
    },
)


def local_total(actual: np.ndarray, pred: np.ndarray, cap: np.ndarray) -> float:
    """층 안에서의 공식 Total 유사량. 지표정합 목적함수(M2)."""
    err_rate = np.abs(pred - actual) / cap
    unit = settlement_unit(err_rate)
    gen = actual.sum()
    if gen <= 0:
        return 0.0
    ficr = float((unit * actual).sum() / gen / 4.0)
    return 0.5 * ficr + 0.5 * (1.0 - float(err_rate.mean()))


def estimate_shifts(train: pd.DataFrame, method: str) -> tuple[dict[Any, float], dict[int, float]]:
    """층별 이동(용량 단위)과 그룹 수준 후퇴값을 추정한다."""
    strata: dict[Any, float] = {}
    for key, cell in train.groupby(["group_id", "pred_band"], observed=True):
        if len(cell) < MIN_STRATUM_TRAIN_ROWS:
            continue
        strata[key] = _shift_for(cell, method)
    fallback = {
        int(g): _shift_for(cell, method)
        for g, cell in train.groupby("group_id", observed=True)
    }
    return strata, fallback


def _shift_for(cell: pd.DataFrame, method: str) -> float:
    actual = cell["actual_kwh"].to_numpy(dtype="float64")
    pred = cell["prediction_kwh"].to_numpy(dtype="float64")
    cap = cell["capacity"].to_numpy(dtype="float64")
    if method == "median_shift":
        return float(np.median((actual - pred) / cap))
    scores = [local_total(actual, pred + s * cap, cap) for s in SHIFT_GRID]
    return float(SHIFT_GRID[int(np.argmax(scores))])


def apply_shifts(
    holdout: pd.DataFrame, strata: dict[Any, float], fallback: dict[int, float]
) -> np.ndarray:
    keys = list(zip(holdout["group_id"], holdout["pred_band"], strict=True))
    shifts = np.array(
        [strata.get(k, fallback.get(int(k[0]), 0.0)) for k in keys], dtype="float64"
    )
    return holdout["prediction_kwh"].to_numpy(dtype="float64") + shifts * holdout[
        "capacity"
    ].to_numpy(dtype="float64")


def main() -> int:
    members = ENSEMBLES[BASE_ENSEMBLE]
    k = len(members)
    stacked = stack_members(members)
    base = combine(stacked, k, COMBINER)
    base["capacity"] = stacked["capacity"].to_numpy()
    base["fold"] = base["month"].map(QUARTER_OF_MONTH)
    base["pred_band"] = pd.cut(
        base["prediction_kwh"] / base["capacity"], bins=list(Y_BAND_EDGES), right=True
    ).astype(str)
    assert base["fold"].notna().all(), "fold 매핑에 구멍이 있다"
    incumbent_score = official(base)

    results: dict[str, Any] = {}
    for method in ("median_shift", "metric_shift"):
        pieces = []
        shift_by_split: dict[Any, list[float]] = {}
        for held in FOLDS:
            train = base.loc[base["fold"] != held]
            test = base.loc[base["fold"] == held].copy()
            strata, fallback = estimate_shifts(train, method)
            test["prediction_kwh"] = apply_shifts(test, strata, fallback)
            pieces.append(test)
            for key, value in strata.items():
                shift_by_split.setdefault(key, []).append(value)
        corrected = pd.concat(pieces, ignore_index=True)
        assert len(corrected) == len(base), "LOO 이어붙이기에서 행 수가 바뀌었다"

        score = official(corrected)
        gate = evaluate_gate(corrected, base)
        stats = gate.evidence

        complete = {key: v for key, v in shift_by_split.items() if len(v) == len(FOLDS)}
        agree = [
            key for key, v in complete.items()
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
                "min_delta": float(stats["min_total_delta"]),
            },
            "strata_with_all_splits": len(complete),
            "sign_agreement": agreement,
            "shift_examples": [
                {"group": int(key[0]), "pred_band": key[1],
                 "shifts": [round(x, 4) for x in complete[key]]}
                for key in sorted(complete, key=lambda x: (x[0], str(x[1])))[:12]
            ],
            "R1_improves": r1,
            "R2_gate": r2,
            "R3_sign_stable": r3,
            "qualifies": bool(r1 and r2 and r3),
        }

    qualified = [m for m in ("median_shift", "metric_shift") if results[m]["qualifies"]]
    chosen = "median_shift" if "median_shift" in qualified else (
        qualified[0] if qualified else None
    )
    promoted_total = results[chosen]["total"] if chosen else incumbent_score["total"]

    check = {
        "H1_expectation": "median_shift 가 R1·R2·R3 를 모두 만족",
        "H1_held": results["median_shift"]["qualifies"],
        "H2_expectation": "metric_shift 가 R1·R2·R3 를 모두 만족",
        "H2_held": results["metric_shift"]["qualifies"],
        "H3_expectation": f"부호 일치율이 두 방법 모두 >= {R3_MIN_SIGN_AGREEMENT:.2f}",
        "H3_held": bool(
            results["median_shift"]["R3_sign_stable"] and results["metric_shift"]["R3_sign_stable"]
        ),
        "qualified_methods": qualified,
        "chosen_by_frozen_priority": chosen,
        "verdict": (
            f"MOS_PROMOTED_{chosen.upper()}" if chosen else "MOS_REJECTED_KEEP_MEDIAN4"
        ),
    }

    payload = {
        "node": NODE_ID,
        "parent_node": PARENT_NODE,
        "incumbent": INCUMBENT,
        "gate_version": GATE_VERSION,
        "gate_modified": False,
        "design": "leave-one-fold-out (2023 Q2/Q3/Q4), 층 = 그룹 x 예측대역",
        "method_sources": list(METHOD_SOURCES),
        "incumbent_score": incumbent_score,
        "methods": results,
        "promotion_rule_frozen_before_run": [
            "R1 pooled Total 개선", "R2 동결 게이트 통과(부모=M271_MEDIAN4)",
            f"R3 부호 일치 층 비율 >= {R3_MIN_SIGN_AGREEMENT}",
            "둘 다 자격시 median_shift 우선",
        ],
        "predeclared_check": check,
        "promoted_total": promoted_total,
        "gap_to_target": 0.66 - promoted_total,
    }

    lines = [
        "# M271 P4 사이클 21 — 측정된 조건부 편향의 fold-외 보정 (MOS)",
        "",
        f"- 판정일: {datetime.now(UTC).strftime('%Y-%m-%d')} (UTC)",
        f"- 노드: `{NODE_ID}` / 레인 {LANE} / 부모 `{PARENT_NODE}`",
        f"- 기존 승격후보 `{INCUMBENT}` Total **{incumbent_score['total']:.6f}**",
        "- 설계: **leave-one-fold-out** (2023 Q2/Q3/Q4). 전량이 fold-외 예측이다.",
        f"- 게이트: `{GATE_VERSION}` (읽기만 함)",
        "",
        "## 0. 방법 리서치 (실행 전)",
        "",
    ]
    for s in METHOD_SOURCES:
        lines.append(f"- **{s['cite']}** (`{s['applicability']}`)")
        lines.append(f"  - {s['claim']}")
        lines.append(f"  - 사용: {s['use']}")

    lines += [
        "",
        "## 1. 승격 규칙 — 실행 **전에** 동결",
        "",
        "두 방법 중 좋은 쪽을 고르지 않는다. 각자 독립적으로 세 조건을 만족해야 자격이 생기고,",
        "둘 다 자격을 얻으면 더 표준적이고 덜 맞춘 `median_shift` 를 택한다.",
        "",
        "- R1 pooled Total 개선",
        f"- R2 `{INCUMBENT}` 를 부모로 한 동결 게이트 통과",
        f"- R3 층별 이동 부호가 세 분할에서 일치하는 층 >= {R3_MIN_SIGN_AGREEMENT:.0%}",
        "",
        "## 2. 결과",
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

    lines += [
        "",
        "## 3. 층별 이동의 분할간 안정성 (R3)",
        "",
        "부호가 분할마다 뒤집히면 편향이 아니라 잡음이다.",
        "",
    ]
    for method, r in results.items():
        lines += [
            f"### `{method}` — 일치율 **{r['sign_agreement']:.3f}** "
            f"({r['strata_with_all_splits']} 층)",
            "",
            "| 그룹 | 예측대역 | 세 분할의 이동 (용량 단위) |",
            "|---:|---|---|",
        ]
        for e in r["shift_examples"]:
            lines.append(
                f"| {e['group']} | {e['pred_band']} | "
                + " / ".join(f"`{x:+.4f}`" for x in e["shifts"]) + " |"
            )
        lines.append("")

    lines += [
        "## 4. 사전확약 대조",
        "",
        f"- H1 `{check['H1_expectation']}` -> **{check['H1_held']}**",
        f"- H2 `{check['H2_expectation']}` -> **{check['H2_held']}**",
        f"- H3 `{check['H3_expectation']}` -> **{check['H3_held']}**",
        "",
        f"자격 방법: `{qualified or '없음'}` / 동결 우선순위로 선택: `{chosen or '없음'}`",
        "",
        f"판정: **{check['verdict']}**",
        "",
        f"승격 Total **{promoted_total:.6f}**, 목표 0.66 까지 **{0.66 - promoted_total:+.6f}**.",
        "",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    receipt: dict[str, Any] = {
        "schema_version": 1,
        "stage": "M271_P4_CYCLE21_MOS",
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

    print(f"[C21] 기존 {INCUMBENT} Total {incumbent_score['total']:.6f}")
    for method, r in results.items():
        print(f"[C21] {method:>14}  Total {r['total']:.6f}  "
              f"기존대비 {r['delta_vs_incumbent']:+.6f}  "
              f"게이트 {'통과' if r['R2_gate'] else '기각'}  "
              f"부호일치 {r['sign_agreement']:.3f}  "
              f"자격 {r['qualifies']}")
    print(f"[C21] 판정: {check['verdict']}  ->  Total {promoted_total:.6f} "
          f"(목표까지 {0.66 - promoted_total:+.6f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
