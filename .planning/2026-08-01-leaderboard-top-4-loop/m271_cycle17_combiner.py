"""M271 P4 사이클 17 — 지표정합 결합 연산자와 앙상블 천장의 직접 측정.

사이클 16 이 `SHRINKBLEND_A05` 를 승격 후보로 확정했다(세 그룹 동시 개선, 월제거 9/9,
재현 일치). C7 발화 — 확인된 방향(분산 감소)으로 재진입한다. 그런데 "멤버를 더 넣는다" 는
이미 닫혔다: 사이클 13 에서 12 개 전부(E1)가 4 개(E3)보다 나빴다.

남은 질문은 **평균이 이 지표에 맞는 결합 연산인가** 다.

① 방법 리서치 (실행 전)
  - Elliott & Timmermann (2004), *J. Econometrics* 122(1):47-79 — 일반 손실 하에서
    최적 결합 가중치는 MSE 최적 가중치와 다르며, 손실의 비대칭성과 오차분포의 왜도가
    최적 가중치를 크게 바꾼다. 평균은 **제곱오차에 최적인 결합자**일 뿐이다.
  - FICR 은 `|err| <= 6% 용량` 지시함수의 발전량가중 합 — 계단 손실이다. 지시 손실의
    표준 처리는 매끄러운 대리함수(arXiv:2503.20082 은 Cauchy CDF)로 바꾼 뒤 제약
    최적화로 가중치를 적합하는 것이다.
  - **그 방법은 여기서 채택하지 않는다.** 가중치를 평가 fold 에서 적합하면 이 프로젝트가
    반복 기각해 온 선택 편향이고, forecast combination puzzle 은 추정 가중치가 대개
    단순평균에 진다고 말한다. 채택하는 것은 **모수 없는** 결합자뿐이며, 유일한 상수는
    데이터가 아니라 **공식 규칙의 6% 밴드**에서 읽는다.

② 사양 동결 — 두 부분

  A. 선결 측정. 멤버 4 개의 불일치 폭(max-min, 용량 단위)이 밴드에 미치는가. 동시에
     `스프레드 vs 앙상블 오차` 로 **분산감소 천장을 직접 측정**한다. 지금까지의 천장
     (rho=0.78 -> 오차 11.7% 감소)은 상관계수에서 **유도한** 값이지 잰 값이 아니다.

  B. 결합 연산자. 승격 후보의 다른 모든 것을 고정하고(같은 4 멤버, 같은 alpha=0.5, 같은
     부모) **결합 연산만** 바꾼다.
       O1 `mean`         승격 후보의 결합자 (대조군)
       O2 `median`       모수 없는 강건 결합자
       O3 `modal_window` 폭 12% 용량 창 중 멤버를 가장 많이 담는 창의 평균.
                         (= 최빈창 절사평균. 창 폭은 FICR 밴드 +-6% 에서 옴)
     멤버 4 개가 모두 한 창에 들면 O3 은 O1 과 항등이다 — 설계상 그렇다.

  사전확약(실행 전 동결):
    H1  멤버 스프레드가 6% 용량을 넘는 행이 **10% 이상**이다.
        기각되면 O3 은 구조적으로 평균과 같고 "더 똑똑한 결합" 계열 전체가 닫힌다.
    H2  O2 또는 O3 중 적어도 하나가 O1 대비 pooled Total 을 **개선**한다.
    H3  개선한 연산자가 **동결 게이트를 통과**한다.
  예상 부호: H1 은 **기각을 예상**한다. 멤버 스프레드는 모델 불확실성이고 오차는 NWP
  불확실성이 지배하므로 후자가 훨씬 크다. 예상대로면 이 축은 측정으로 닫힌다.

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
from m270_monthly_validation import load_predictions
from m271_cycle13_ensemble import ENSEMBLES, load_model
from m271_cycle14_shrinkblend import blend
from m271_evaluate_candidate import official

from baram.constants import CAPACITIES_KWH

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
REPORT_MD = REPORTS / "m271_cycle17_combiner.md"
RECEIPT = REPORTS / "m271_cycle17_combiner_receipt.json"

NODE_ID = "C1N17_METRIC_ALIGNED_COMBINER"
LANE = "L7"  # 모델 개선 전략
PARENT_NODE = "C1N14_SHRINKBLEND"
DEPLOYED = "T0.5_G1.5"
BASE_ENSEMBLE = "E3_FOUR_FAMILY"
ALPHA = 0.5

BAND_HALF_WIDTH = 0.06  # 공식 규칙: |err| <= 6% 용량 이면 정산단위 4
SPREAD_MATERIAL_FRACTION = 0.10  # H1 문턱

METHOD_SOURCES = (
    {
        "id": "elliott_timmermann_2004",
        "cite": "Elliott & Timmermann (2004), J. Econometrics 122(1):47-79",
        "claim": "일반 손실에서 최적 결합 가중치는 MSE 최적과 다르다",
        "applicability": "directly_supported",
        "use": "평균이 FICR 에 최적인 결합자가 아닐 수 있다는 근거",
    },
    {
        "id": "arxiv_2503_20082",
        "cite": "arXiv:2503.20082 — hit/win rate loss 결합 가중치 최적화",
        "claim": "0-1 지시 손실은 Cauchy CDF 대리함수 + 제약 최적화로 다룬다",
        "applicability": "contradicts_premise",
        "use": "방법은 유효하나 가중치 적합이 선택 편향이라 **기각**. 모수 없는 연산자만 채택",
    },
)


def stack_members(models: tuple[str, ...]) -> pd.DataFrame:
    """멤버별 예측을 한 행에 나란히 놓는다. 결합 연산자마다 다르게 접기 위함."""
    keys = ["forecast_id", "forecast_kst_dtm", "group_id"]
    frames = [load_model(m) for m in models]
    out = frames[0].loc[:, [*keys, "actual_kwh"]].copy()
    for i, frame in enumerate(frames):
        out = out.merge(
            frame.loc[:, [*keys, "prediction_kwh"]].rename(columns={"prediction_kwh": f"m{i}"}),
            on=keys,
            how="inner",
        )
    out["month"] = out["forecast_kst_dtm"].dt.to_period("M").astype(str)
    out["capacity"] = out["group_id"].map(CAPACITIES_KWH).astype(float)
    return out


def modal_window(members: np.ndarray, width: np.ndarray) -> np.ndarray:
    """폭 `width` 창 중 멤버를 가장 많이 담는 창의 평균 (최빈창 절사평균).

    동률이면 창 span 이 작은 쪽, 그래도 동률이면 낮은 시작 인덱스. 멤버가 모두 한 창에
    들면 결과는 단순평균과 항등이다.
    """
    ordered = np.sort(members, axis=1)
    n_rows, k = ordered.shape
    best_start = np.zeros(n_rows, dtype=int)
    best_stop = np.full(n_rows, k, dtype=int)  # 배타적
    best_count = np.zeros(n_rows, dtype=int)
    best_span = np.full(n_rows, np.inf)

    for i in range(k):
        for j in range(i, k):
            span = ordered[:, j] - ordered[:, i]
            fits = span <= width
            count = j - i + 1
            better = fits & (
                (count > best_count) | ((count == best_count) & (span < best_span))
            )
            best_start = np.where(better, i, best_start)
            best_stop = np.where(better, j + 1, best_stop)
            best_span = np.where(better, span, best_span)
            best_count = np.where(better, count, best_count)

    idx = np.arange(k)[None, :]
    mask = (idx >= best_start[:, None]) & (idx < best_stop[:, None])
    return (ordered * mask).sum(axis=1) / mask.sum(axis=1)


def combine(stacked: pd.DataFrame, k: int, operator: str) -> pd.DataFrame:
    cols = [f"m{i}" for i in range(k)]
    members = stacked.loc[:, cols].to_numpy(dtype="float64")
    if operator == "mean":
        combined = members.mean(axis=1)
    elif operator == "median":
        combined = np.median(members, axis=1)
    elif operator == "modal_window":
        width = 2.0 * BAND_HALF_WIDTH * stacked["capacity"].to_numpy(dtype="float64")
        combined = modal_window(members, width)
    else:  # pragma: no cover - 사양에 없는 연산자
        raise ValueError(f"unknown operator: {operator}")
    out = stacked.loc[
        :, ["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh", "month"]
    ].copy()
    out["prediction_kwh"] = combined
    return out


def main() -> int:
    members = ENSEMBLES[BASE_ENSEMBLE]
    k = len(members)
    stacked = stack_members(members)
    parent = load_predictions(DEPLOYED)
    parent_score = official(parent)

    cols = [f"m{i}" for i in range(k)]
    arr = stacked.loc[:, cols].to_numpy(dtype="float64")
    cap = stacked["capacity"].to_numpy(dtype="float64")
    spread = (arr.max(axis=1) - arr.min(axis=1)) / cap
    ens_mean = arr.mean(axis=1)
    err = np.abs(ens_mean - stacked["actual_kwh"].to_numpy(dtype="float64")) / cap

    # 유효행(실측 >= 10% 용량)만 채점 대상이다. 진단도 같은 모집단에서 본다.
    eligible = (stacked["actual_kwh"].to_numpy(dtype="float64") >= 0.10 * cap)
    sp_e, er_e = spread[eligible], err[eligible]

    frac_over_band = float((sp_e > BAND_HALF_WIDTH).mean())
    frac_over_full = float((sp_e > 2.0 * BAND_HALF_WIDTH).mean())
    h1 = bool(frac_over_band >= SPREAD_MATERIAL_FRACTION)

    # 분산감소 천장의 **직접** 측정. 멤버 오차의 평균분산 대비 앙상블 오차 분산.
    signed = (arr - stacked["actual_kwh"].to_numpy(dtype="float64")[:, None]) / cap[:, None]
    signed_e = signed[eligible]
    mean_member_mse = float((signed_e**2).mean())
    ens_mse = float(((signed_e.mean(axis=1)) ** 2).mean())
    # e_ens^2 = (1/K)(1 + (K-1) rho_bar) * e_member^2  =>  rho_bar 를 역산
    rho_bar = float((k * ens_mse / mean_member_mse - 1.0) / (k - 1))
    ceiling_ratio = float(np.sqrt(rho_bar)) if rho_bar > 0 else 0.0
    realised_ratio = float(np.sqrt(ens_mse / mean_member_mse))

    spread_diag = {
        "rows_eligible": int(eligible.sum()),
        "spread_median": float(np.median(sp_e)),
        "spread_p90": float(np.quantile(sp_e, 0.90)),
        "spread_p99": float(np.quantile(sp_e, 0.99)),
        "abs_err_median": float(np.median(er_e)),
        "spread_over_err_median": float(np.median(sp_e) / np.median(er_e)),
        "frac_spread_over_6pct": frac_over_band,
        "frac_spread_over_12pct": frac_over_full,
        "mean_member_mse_cap_units": mean_member_mse,
        "ensemble_mse_cap_units": ens_mse,
        "implied_mean_error_correlation": rho_bar,
        "realised_rmse_ratio_K4": realised_ratio,
        "asymptotic_rmse_ratio_ceiling": ceiling_ratio,
        "fraction_of_ceiling_captured": float(
            (1.0 - realised_ratio) / (1.0 - ceiling_ratio)
        ) if ceiling_ratio < 1.0 else float("nan"),
    }

    rows: list[dict[str, Any]] = []
    baseline_total: float | None = None
    for operator in ("mean", "median", "modal_window"):
        ensemble = combine(stacked, k, operator)
        candidate = blend(parent, ensemble, ALPHA)
        score = official(candidate)
        gate = evaluate_gate(candidate, parent)
        stats = gate.evidence
        if operator == "mean":
            baseline_total = score["total"]
        # O3 이 O1 과 항등인지 직접 확인한다.
        identical_to_mean = bool(
            np.allclose(
                combine(stacked, k, "mean")["prediction_kwh"].to_numpy(),
                ensemble["prediction_kwh"].to_numpy(),
                rtol=0.0,
                atol=1e-9,
            )
        )
        rows.append(
            {
                "operator": operator,
                "total": score["total"],
                "one_minus_nmae": score["one_minus_nmae"],
                "ficr": score["ficr"],
                "delta_vs_deployed": score["total"] - parent_score["total"],
                "delta_vs_mean": score["total"] - (baseline_total or score["total"]),
                "gate_passed": bool(gate.passed),
                "gate": {label.split()[0]: bool(ok) for label, ok in gate.conditions.items()},
                "positive_months": int(stats["positive_months"]),
                "sign_test_p": float(stats["sign_test_p_greater"]),
                "bootstrap_q05": float(stats["block_bootstrap_q05"]),
                "identical_to_mean": identical_to_mean,
            }
        )

    assert baseline_total is not None
    improved = [r for r in rows if r["operator"] != "mean" and r["delta_vs_mean"] > 0]
    h2 = bool(improved)
    h3 = bool(any(r["gate_passed"] for r in improved))

    check = {
        "H1_expectation": f"멤버 스프레드 > 6% 용량인 행 >= {SPREAD_MATERIAL_FRACTION:.0%}",
        "H1_held": h1,
        "H1_measured": frac_over_band,
        "H2_expectation": "median 또는 modal_window 가 mean 대비 Total 개선",
        "H2_held": h2,
        "H3_expectation": "개선한 연산자가 동결 게이트 통과",
        "H3_held": h3,
        "verdict": (
            "COMBINER_AXIS_OPEN" if (h2 and h3)
            else ("COMBINER_AXIS_CLOSED_STRUCTURAL" if not h1 else "COMBINER_AXIS_CLOSED_MEASURED")
        ),
    }

    payload = {
        "gate_version": GATE_VERSION,
        "gate_modified": False,
        "node": NODE_ID,
        "parent_node": PARENT_NODE,
        "members": list(members),
        "alpha": ALPHA,
        "method_sources": list(METHOD_SOURCES),
        "spread_diagnosis": spread_diag,
        "operators": rows,
        "parent": {"policy": DEPLOYED, **parent_score},
        "predeclared_check": check,
    }

    lines = [
        "# M271 P4 사이클 17 — 지표정합 결합 연산자와 앙상블 천장의 직접 측정",
        "",
        f"- 판정일: {datetime.now(UTC).strftime('%Y-%m-%d')} (UTC)",
        f"- 노드: `{NODE_ID}` / 레인 {LANE} / 부모 `{PARENT_NODE}`",
        f"- 멤버 {k} 개 고정, alpha={ALPHA} 고정. **결합 연산만** 바꾼다.",
        f"- 게이트: `{GATE_VERSION}` (읽기만 함)",
        "",
        "## 0. 방법 리서치 (실행 전)",
        "",
    ]
    for s in METHOD_SOURCES:
        lines.append(f"- **{s['cite']}** — {s['claim']} (`{s['applicability']}`)")
        lines.append(f"  - 사용: {s['use']}")
    lines += [
        "",
        "평균은 제곱오차에 최적인 결합자다. FICR 은 계단 손실이므로 최적 결합자가 다를 수",
        "있다. 다만 가중치를 **적합**하는 정통 해법은 평가 fold 선택 편향이라 기각하고,",
        "상수를 공식 규칙(6% 밴드)에서만 읽는 **모수 없는** 연산자로 제한했다.",
        "",
        "## 1. 선결 측정 — 멤버는 얼마나 불일치하는가 (H1)",
        "",
        f"유효행 {spread_diag['rows_eligible']:,} 개 기준.",
        "",
        "| 양 | 값 (용량 단위) |",
        "|---|---:|",
        f"| 멤버 스프레드 중앙값 | {spread_diag['spread_median']:.4f} |",
        f"| 멤버 스프레드 p90 | {spread_diag['spread_p90']:.4f} |",
        f"| 멤버 스프레드 p99 | {spread_diag['spread_p99']:.4f} |",
        f"| 앙상블 절대오차 중앙값 | {spread_diag['abs_err_median']:.4f} |",
        f"| **스프레드 / 오차** (중앙값비) | **{spread_diag['spread_over_err_median']:.3f}** |",
        f"| 스프레드 > 6% 용량인 행 비율 | {frac_over_band:.4f} |",
        f"| 스프레드 > 12% 용량인 행 비율 | {frac_over_full:.4f} |",
        "",
        "## 2. 분산감소 천장 — 유도값이 아니라 실측",
        "",
        "지금까지의 천장은 상관계수 `rho=0.78` 에서 유도한 값이었다. 여기서는 멤버 오차의",
        "평균 MSE 와 앙상블 MSE 의 비로 **직접** 잰다.",
        "",
        "| 양 | 값 |",
        "|---|---:|",
        f"| 멤버 평균 MSE | {mean_member_mse:.6f} |",
        f"| 앙상블(K={k}) MSE | {ens_mse:.6f} |",
        f"| 역산 평균 오차상관 `rho_bar` | **{rho_bar:.4f}** |",
        f"| K={k} 실현 RMSE 비 | {realised_ratio:.4f} |",
        f"| K->inf 천장 RMSE 비 | {ceiling_ratio:.4f} |",
        "| **천장 대비 이미 확보한 비율** | "
        f"**{spread_diag['fraction_of_ceiling_captured']:.1%}** |",
        "",
        "## 3. 결합 연산자 (H2 · H3)",
        "",
        "| 연산자 | Total | 1-NMAE | FICR | 배포대비 | mean대비 | G1G2G3G4 | 양수월 | q05 "
        "| 게이트 | mean과 동일 |",
        "|---|---:|---:|---:|---:|---:|:---:|---:|---:|:---:|:---:|",
    ]
    for r in rows:
        flags = "".join("O" if r["gate"].get(g) else "-" for g in ("G1", "G2", "G3", "G4"))
        lines.append(
            f"| `{r['operator']}` | {r['total']:.6f} | {r['one_minus_nmae']:.6f} | "
            f"{r['ficr']:.6f} | {r['delta_vs_deployed']:+.6f} | {r['delta_vs_mean']:+.6f} | "
            f"`{flags}` | {r['positive_months']}/9 | {r['bootstrap_q05']:+.6f} | "
            f"{'**통과**' if r['gate_passed'] else '기각'} | {r['identical_to_mean']} |"
        )

    lines += [
        "",
        "## 4. 사전확약 대조",
        "",
        f"- H1 `{check['H1_expectation']}` -> **{h1}** (실측 {frac_over_band:.4f})",
        f"- H2 `{check['H2_expectation']}` -> **{h2}**",
        f"- H3 `{check['H3_expectation']}` -> **{h3}**",
        "",
        f"판정: **{check['verdict']}**",
        "",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    receipt = {
        "schema_version": 1,
        "stage": "M271_P4_CYCLE17_COMBINER",
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

    print(f"[C17] 스프레드 중앙값 {spread_diag['spread_median']:.4f} vs "
          f"오차 중앙값 {spread_diag['abs_err_median']:.4f} "
          f"(비 {spread_diag['spread_over_err_median']:.3f})")
    print(f"[C17] 스프레드 > 6% 행 비율 {frac_over_band:.4f}  -> H1 {h1}")
    print(f"[C17] rho_bar {rho_bar:.4f}  실현 {realised_ratio:.4f}  "
          f"천장 {ceiling_ratio:.4f}  확보 {spread_diag['fraction_of_ceiling_captured']:.1%}")
    for r in rows:
        print(f"[C17] {r['operator']:>12}  Total {r['total']:.6f}  "
              f"mean대비 {r['delta_vs_mean']:+.6f}  "
              f"게이트 {'통과' if r['gate_passed'] else '기각'}  "
              f"동일 {r['identical_to_mean']}")
    print(f"[C17] 판정: {check['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
