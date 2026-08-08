"""M271 P4 사이클 25 — 봉투 안 위치는 예보시점에 예측 가능한가.

사이클 19 가 봉투 오라클 `0.742380` 을 쟀다. Breiman (1996) 이 그 대상을 정확히 지목한다:
비음이고 합이 1 인 가중치는 스태킹 추정량을 **보간적** 으로 만들어 예측이 반드시 멤버
min~max 사이에 놓이게 하고, 그 제약이 스태킹을 "최고 단일 모델 선택" 보다 낫게 만든다.
따라서 봉투 오라클은 **Breiman 제약 스태킹의 천장** 이다.

그러면 질문이 하나로 환원된다. 봉투 안 위치

    lambda* = clip((actual - min) / (max - min), 0, 1)

를 **예보시점 문맥** 으로 예측할 수 있는가. 못 하면 어떤 스태킹 모델도 소용없다.

모델을 세우기 전에 도달 가능성부터 잰다. 이 프로젝트가 결정층·편향보정에서 쓴 순서와 같다.

① 방법 리서치 (실행 전)
  - Wolpert (1992) stacked generalization / Breiman (1996) *Stacked Regressions* —
    교차검증 예측 위에서 **비음 제약** 최소제곱으로 결합. 비음 제약이 결정적이다.
  - Breiman 의 보간 논거가 이 노드의 좌표계를 준다. 결합을 "가중치 4 개" 가 아니라
    **봉투 안 스칼라 위치 1 개** 로 다시 쓰면 자유도가 4 에서 1 로 줄고, 오라클이
    직접 정의된다.
  - forecast combination puzzle: 추정 가중치는 흔히 단순평균에 진다. 따라서 자유로운
    회귀가 아니라 **층별 상수 lambda** 만 추정한다 — 자유도를 최소로 유지한다.
  - 사이클 23 의 교훈을 사양에 못박는다: **추정은 유효행에서만.**

② 사양 동결

  문맥 층 = 그룹 x 예측대역 x 스프레드 3분위. 전부 예보시점 가용.
  추정량 둘, leave-one-fold-out.
    L1 `median_lambda`  층별 lambda* 조건부 중앙값
    L2 `metric_lambda`  층의 공식 Total 기여를 최대화하는 lambda (격자 탐색)

  **승격 규칙 (실행 전 동결)** — 각자 독립적으로
    R1 `M271_MEDIAN4` 대비 Total 개선 / R2 그것을 부모로 한 동결 게이트 통과.
  둘 다 자격시 **`metric_lambda` 우선** (평가가 공식 지표이므로 지표정합 추정량이
  선험적으로 옳다 — 사이클 23 과 같은 근거).

  사전확약(실행 전 동결):
    H1  lambda* 가 양극단에 몰린다: `P(lambda* <= 0.01 or >= 0.99) >= 0.60`.
        사이클 19 의 커버리지 0.226 에서 따라 나오는 예측이다.
    H2  층별 lambda* 중앙값의 **층간 범위 >= 0.15**. 문맥이 위치를 가른다는 뜻.
    H3  층 순위가 fold 간 안정: 세 fold 쌍 Spearman 평균 **>= 0.50**.
    H4  둘 중 적어도 하나가 R1·R2 를 만족한다.
  H2 나 H3 가 기각되면 lambda 는 문맥으로 예측되지 않고 **추출 축이 닫힌다** — 이 4 개
  멤버로는 목표에 갈 수 없고 새 기저모델이 필요하다는 결론이 된다.

**게이트를 수정하지 않는다.** 읽기만 한다.

읽기 전용. 층별 상수 추정이며 자유 회귀를 적합하지 않는다. 2024 행을 읽지 않는다.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))

from m270_gate import GATE_VERSION, evaluate_gate
from m271_cycle13_ensemble import ENSEMBLES
from m271_cycle17_combiner import combine, stack_members
from m271_cycle21_mos import FOLDS, QUARTER_OF_MONTH, local_total
from m271_evaluate_candidate import official
from m271_n0_deficit_init import Y_BAND_EDGES

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
REPORT_MD = REPORTS / "m271_cycle25_lambda.md"
RECEIPT = REPORTS / "m271_cycle25_lambda_receipt.json"

NODE_ID = "C1N25_ENVELOPE_LAMBDA"
LANE = "L3"  # 모델링 방법 — 스태킹 좌표계
PARENT_NODE = "C1N19_ENVELOPE_ORACLE"
INCUMBENT = "M271_MEDIAN4"
BASE_ENSEMBLE = "E3_FOUR_FAMILY"
ELIGIBLE_THRESHOLD = 0.10
DEGENERATE_WIDTH = 1e-9
LAMBDA_GRID = np.round(np.arange(0.0, 1.0 + 1e-9, 0.02), 3)
MIN_STRATUM_TRAIN_ROWS = 100
FROZEN_PRIORITY = ("metric_lambda", "median_lambda")

H1_MIN_EXTREME_MASS = 0.60
H2_MIN_LAMBDA_RANGE = 0.15
H3_MIN_MEAN_SPEARMAN = 0.50

METHOD_SOURCES = (
    {
        "id": "wolpert_1992_stacking",
        "cite": "Wolpert (1992) — Stacked Generalization",
        "claim": "교차검증 예측 위에 상위 학습기를 얹어 결합한다",
        "applicability": "directly_supported",
        "use": "fold-외 추정 설계의 근거",
    },
    {
        "id": "breiman_1996_stacked_regressions",
        "cite": "Breiman (1996), Machine Learning — Stacked Regressions",
        "claim": "비음·합1 가중치는 추정량을 **보간적**으로 만들어 min<=pred<=max 를 보장하고, "
                 "그 제약이 최고 단일 모델 선택보다 나은 정확도의 이유다",
        "applicability": "directly_supported",
        "use": "결합을 가중치 4 개가 아니라 **봉투 안 위치 lambda 1 개**로 재좌표화. "
               "사이클 19 의 봉투 오라클이 곧 이 제약 스태킹의 천장임을 확정",
    },
    {
        "id": "forecast_combination_puzzle",
        "cite": "forecast combination puzzle (Clemen 1989 이후 반복 확인)",
        "claim": "추정 가중치는 흔히 단순평균에 진다",
        "applicability": "contradicts_premise",
        "use": "자유 회귀를 기각하고 **층별 상수 lambda** 로 자유도를 최소화",
    },
)


def build_frame() -> pd.DataFrame:
    members = ENSEMBLES[BASE_ENSEMBLE]
    k = len(members)
    stacked = stack_members(members)
    arr = stacked.loc[:, [f"m{i}" for i in range(k)]].to_numpy(dtype="float64")
    med = combine(stacked, k, "median")["prediction_kwh"].to_numpy(dtype="float64")
    cap = stacked["capacity"].to_numpy(dtype="float64")
    actual = stacked["actual_kwh"].to_numpy(dtype="float64")
    lo, hi = arr.min(axis=1), arr.max(axis=1)
    width = hi - lo

    frame = stacked.loc[
        :, ["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh", "month"]
    ].copy()
    frame["capacity"] = cap
    frame["lo"] = lo
    frame["hi"] = hi
    frame["width"] = width
    frame["median_pred"] = med
    safe_width = np.where(width > 0, width, 1.0)
    frame["lambda_star"] = np.where(
        width > DEGENERATE_WIDTH, np.clip((actual - lo) / safe_width, 0.0, 1.0), 0.5
    )
    frame["lambda_median_combiner"] = np.where(
        width > DEGENERATE_WIDTH, (med - lo) / safe_width, 0.5
    )
    frame["fold"] = frame["month"].map(QUARTER_OF_MONTH)
    frame["pred_band"] = pd.cut(med / cap, bins=list(Y_BAND_EDGES), right=True).astype(str)
    frame["spread_tercile"] = pd.qcut(
        width / cap, 3, labels=["lo", "mid", "hi"], duplicates="drop"
    ).astype(str)
    frame["stratum"] = (
        frame["group_id"].astype(str) + "|" + frame["pred_band"] + "|" + frame["spread_tercile"]
    )
    frame["eligible"] = frame["actual_kwh"] >= ELIGIBLE_THRESHOLD * cap
    assert frame["fold"].notna().all(), "fold 매핑에 구멍이 있다"
    return frame


def lambda_for(cell: pd.DataFrame, estimator: str) -> float:
    if estimator == "median_lambda":
        return float(cell["lambda_star"].median())
    actual = cell["actual_kwh"].to_numpy(dtype="float64")
    cap = cell["capacity"].to_numpy(dtype="float64")
    lo = cell["lo"].to_numpy(dtype="float64")
    width = cell["width"].to_numpy(dtype="float64")
    scores = [local_total(actual, lo + lam * width, cap) for lam in LAMBDA_GRID]
    return float(LAMBDA_GRID[int(np.argmax(scores))])


def estimate(train: pd.DataFrame, estimator: str) -> tuple[dict[str, float], float]:
    table = {
        str(key): lambda_for(cell, estimator)
        for key, cell in train.groupby("stratum", observed=True)
        if len(cell) >= MIN_STRATUM_TRAIN_ROWS
    }
    return table, lambda_for(train, estimator)


def main() -> int:
    frame = build_frame()
    elig = frame["eligible"].to_numpy()
    incumbent = frame.loc[
        :, ["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh", "month"]
    ].copy()
    incumbent["prediction_kwh"] = frame["median_pred"].to_numpy()
    incumbent_score = official(incumbent)

    e = frame.loc[frame["eligible"]]
    lam = e["lambda_star"].to_numpy()
    extreme_mass = float(((lam <= 0.01) | (lam >= 0.99)).mean())
    h1 = bool(extreme_mass >= H1_MIN_EXTREME_MASS)

    # --- H2 층간 범위
    per_stratum = (
        e.groupby("stratum", observed=True)
        .agg(rows=("lambda_star", "size"), lambda_median=("lambda_star", "median"),
             lambda_mean=("lambda_star", "mean"))
        .reset_index()
    )
    thick = per_stratum.loc[per_stratum["rows"] >= MIN_STRATUM_TRAIN_ROWS]
    lam_range = float(thick["lambda_median"].max() - thick["lambda_median"].min())
    h2 = bool(lam_range >= H2_MIN_LAMBDA_RANGE)

    # --- H3 fold 간 순위 안정성
    by_fold = {
        f: e.loc[e["fold"] == f].groupby("stratum", observed=True)["lambda_star"].median()
        for f in FOLDS
    }
    pair_rhos = []
    for a, b in itertools.combinations(FOLDS, 2):
        shared = by_fold[a].index.intersection(by_fold[b].index)
        shared = [s for s in shared if s in set(thick["stratum"])]
        if len(shared) < 4:
            continue
        r, _ = spearmanr(by_fold[a].loc[shared].to_numpy(), by_fold[b].loc[shared].to_numpy())
        pair_rhos.append({"folds": f"{a}~{b}", "n_strata": len(shared), "spearman": float(r)})
    mean_rho = float(np.mean([p["spearman"] for p in pair_rhos])) if pair_rhos else 0.0
    h3 = bool(mean_rho >= H3_MIN_MEAN_SPEARMAN)

    # --- H4 fold-외 lambda 대체
    results: dict[str, Any] = {}
    for estimator in ("median_lambda", "metric_lambda"):
        pieces = []
        for held in FOLDS:
            train = frame.loc[(frame["fold"] != held) & frame["eligible"]]
            test = frame.loc[frame["fold"] == held].copy()
            table, fallback = estimate(train, estimator)
            lam_hat = test["stratum"].map(table).fillna(fallback).to_numpy(dtype="float64")
            out = test.loc[
                :, ["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh", "month"]
            ].copy()
            out["prediction_kwh"] = test["lo"].to_numpy() + lam_hat * test["width"].to_numpy()
            pieces.append(out)
        candidate = pd.concat(pieces, ignore_index=True)
        assert len(candidate) == len(frame), "LOO 이어붙이기에서 행 수가 바뀌었다"
        score = official(candidate)
        gate = evaluate_gate(candidate, incumbent)
        stats = gate.evidence
        r1 = bool(score["total"] > incumbent_score["total"])
        r2 = bool(gate.passed)
        results[estimator] = {
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
            "R1_improves": r1,
            "R2_gate": r2,
            "qualifies": bool(r1 and r2),
        }

    qualified = [m for m in FROZEN_PRIORITY if results[m]["qualifies"]]
    chosen = qualified[0] if qualified else None
    h4 = bool(chosen)
    promoted_total = results[chosen]["total"] if chosen else incumbent_score["total"]
    verdict = (
        f"LAMBDA_PROMOTED_{chosen.upper()}" if chosen
        else ("EXTRACTION_AXIS_CLOSED_LAMBDA_UNPREDICTABLE" if not (h2 and h3)
              else "LAMBDA_PREDICTABLE_BUT_NOT_EXPLOITABLE")
    )

    check = {
        "H1_expectation": f"lambda* 양극단 질량 >= {H1_MIN_EXTREME_MASS:.2f}",
        "H1_held": h1, "H1_measured": extreme_mass,
        "H2_expectation": f"층별 lambda 중앙값 층간 범위 >= {H2_MIN_LAMBDA_RANGE:.2f}",
        "H2_held": h2, "H2_measured": lam_range,
        "H3_expectation": f"fold 쌍 Spearman 평균 >= {H3_MIN_MEAN_SPEARMAN:.2f}",
        "H3_held": h3, "H3_measured": mean_rho,
        "H4_expectation": "적어도 하나가 Total 개선 + 게이트 통과",
        "H4_held": h4,
        "chosen": chosen,
        "verdict": verdict,
    }

    payload = {
        "node": NODE_ID, "parent_node": PARENT_NODE, "incumbent": INCUMBENT,
        "gate_version": GATE_VERSION, "gate_modified": False,
        "method_sources": list(METHOD_SOURCES),
        "incumbent_score": incumbent_score,
        "eligible_rows": int(elig.sum()),
        "lambda_of_median_combiner": {
            "mean": float(e["lambda_median_combiner"].mean()),
            "median": float(e["lambda_median_combiner"].median()),
        },
        "lambda_star_distribution": {
            "extreme_mass": extreme_mass,
            "at_zero": float((lam <= 0.01).mean()),
            "at_one": float((lam >= 0.99).mean()),
            "interior": float(((lam > 0.01) & (lam < 0.99)).mean()),
            "median": float(np.median(lam)),
        },
        "strata": [
            {"stratum": r.stratum, "rows": int(r.rows),
             "lambda_median": float(r.lambda_median), "lambda_mean": float(r.lambda_mean)}
            for r in thick.sort_values("lambda_median").itertuples()
        ],
        "lambda_range_across_strata": lam_range,
        "fold_rank_stability": pair_rhos,
        "mean_pairwise_spearman": mean_rho,
        "estimators": results,
        "predeclared_check": check,
        "promoted_total": promoted_total,
        "gap_to_target": 0.66 - promoted_total,
    }

    lines = [
        "# M271 P4 사이클 25 — 봉투 안 위치는 예보시점에 예측 가능한가",
        "",
        f"- 판정일: {datetime.now(UTC).strftime('%Y-%m-%d')} (UTC)",
        f"- 노드: `{NODE_ID}` / 레인 {LANE} / 부모 `{PARENT_NODE}`",
        f"- 기존 승격후보 `{INCUMBENT}` Total **{incumbent_score['total']:.6f}**",
        f"- 유효행 {int(elig.sum()):,}. 추정은 유효행에서만 (사이클 23 의 교훈)",
        f"- 게이트: `{GATE_VERSION}` (읽기만 함)",
        "",
        "## 0. 방법 리서치 (실행 전)",
        "",
    ]
    for s in METHOD_SOURCES:
        lines.append(f"- **{s['cite']}** (`{s['applicability']}`)")
        lines.append(f"  - {s['claim']}")
        lines.append(f"  - 사용: {s['use']}")

    d = payload["lambda_star_distribution"]
    lines += [
        "",
        "## 1. lambda* 분포 (H1)",
        "",
        "`lambda* = clip((actual - min) / (max - min), 0, 1)` — 봉투 안 최적 위치.",
        "",
        "| 위치 | 질량 |",
        "|---|---:|",
        f"| lambda* <= 0.01 (최저 멤버 이하) | {d['at_zero']:.4f} |",
        f"| 내부 0.01~0.99 | {d['interior']:.4f} |",
        f"| lambda* >= 0.99 (최고 멤버 이상) | {d['at_one']:.4f} |",
        f"| **양극단 합** | **{d['extreme_mass']:.4f}** |",
        "",
        f"참고로 현 `median` 결합자의 실효 위치는 평균 "
        f"{payload['lambda_of_median_combiner']['mean']:.4f}, "
        f"중앙값 {payload['lambda_of_median_combiner']['median']:.4f} 이다.",
        "",
        "## 2. 문맥이 위치를 가르는가 (H2 · H3)",
        "",
        f"굵은 층 {len(thick)} 개, lambda 중앙값 층간 범위 **{lam_range:.4f}**.",
        "",
        "| 층 (그룹\\|예측대역\\|스프레드) | 행 | lambda 중앙값 | lambda 평균 |",
        "|---|---:|---:|---:|",
    ]
    for r in payload["strata"][:8] + payload["strata"][-8:]:
        lines.append(
            f"| `{r['stratum']}` | {r['rows']:,} | **{r['lambda_median']:.4f}** | "
            f"{r['lambda_mean']:.4f} |"
        )
    lines += ["", "fold 간 층 순위 안정성:", "", "| fold 쌍 | 공통 층 | Spearman |",
              "|---|---:|---:|"]
    for p in pair_rhos:
        lines.append(f"| {p['folds']} | {p['n_strata']} | **{p['spearman']:+.4f}** |")
    lines += ["", f"평균 **{mean_rho:+.4f}**", ""]

    lines += [
        "## 3. fold-외 lambda 대체 (H4)",
        "",
        "| 추정량 | Total | 1-NMAE | FICR | 기존대비 | G1G2G3G4 | 양수월 | q05 | 자격 |",
        "|---|---:|---:|---:|---:|:---:|---:|---:|:---:|",
        f"| `{INCUMBENT}` (기존) | {incumbent_score['total']:.6f} | "
        f"{incumbent_score['one_minus_nmae']:.6f} | {incumbent_score['ficr']:.6f} | "
        "— | — | — | — | — |",
    ]
    for estimator, r in results.items():
        g = r["gate"]
        flags = "".join("O" if g["flags"].get(x) else "-" for x in ("G1", "G2", "G3", "G4"))
        lines.append(
            f"| `{estimator}` | {r['total']:.6f} | {r['one_minus_nmae']:.6f} | "
            f"{r['ficr']:.6f} | {r['delta_vs_incumbent']:+.6f} | `{flags}` | "
            f"{g['positive_months']}/{g['months_scored']} | {g['bootstrap_q05']:+.6f} | "
            f"{'**자격**' if r['qualifies'] else '미달'} |"
        )

    lines += [
        "",
        "## 4. 사전확약 대조",
        "",
        f"- H1 `{check['H1_expectation']}` -> **{h1}** (실측 {extreme_mass:.4f})",
        f"- H2 `{check['H2_expectation']}` -> **{h2}** (실측 {lam_range:.4f})",
        f"- H3 `{check['H3_expectation']}` -> **{h3}** (실측 {mean_rho:+.4f})",
        f"- H4 `{check['H4_expectation']}` -> **{h4}**",
        "",
        f"판정: **{verdict}**",
        "",
        f"승격 Total **{promoted_total:.6f}**, 목표 0.66 까지 **{0.66 - promoted_total:+.6f}**.",
        "",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    receipt: dict[str, Any] = {
        "schema_version": 1,
        "stage": "M271_P4_CYCLE25_LAMBDA",
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

    print(f"[C25] lambda* 양극단 {extreme_mass:.4f} (0 쪽 {d['at_zero']:.4f} / "
          f"1 쪽 {d['at_one']:.4f}) -> H1 {h1}")
    print(f"[C25] 층간 lambda 범위 {lam_range:.4f} ({len(thick)} 층) -> H2 {h2}")
    print(f"[C25] fold 쌍 Spearman 평균 {mean_rho:+.4f} -> H3 {h3}")
    for estimator, r in results.items():
        print(f"[C25] {estimator:>14}  Total {r['total']:.6f}  "
              f"기존대비 {r['delta_vs_incumbent']:+.6f}  "
              f"게이트 {'통과' if r['R2_gate'] else '기각'}  자격 {r['qualifies']}")
    print(f"[C25] 판정: {verdict}  ->  Total {promoted_total:.6f} "
          f"(목표까지 {0.66 - promoted_total:+.6f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
