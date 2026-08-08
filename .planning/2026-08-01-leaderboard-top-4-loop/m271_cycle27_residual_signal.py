"""M271 P4 사이클 27 — 챔피언의 잔차에 NWP 신호가 남아 있는가 (A2 전제 재검).

A2 가 미사용 NWP 컬럼을 닫은 근거는 `MI(y ; 컬럼)` — **라벨에 대한 주변 상호정보량** 이다.
개선에 필요한 질문은 다르다. 그 컬럼들이 우리가 **틀리고 있는 것** 에 대한 정보를 갖는가.

두 가지가 주변 스크린에서 빠진다.
  1. 표적이 y 가 아니라 **잔차** 여야 한다. y 를 잘 설명하는 컬럼은 이미 쓰고 있다.
  2. 주변 MI 는 **상보성** 을 못 본다. 쓰는 컬럼과 조건부로만 정보를 주는 변수는 단변량
     스크린에서 낮게 나온다 (mRMR 의 relevance vs redundancy).

**이 노드는 이 세션의 첫 모델 적합이다.** 지금까지 26 사이클이 `model_fits: 0` 이었던 것은
규칙이 아니라 홀드아웃 보호 규율이었고, 파라미터 없는 연산이 사이클 25·26 에서 소진됐다.
진단용 fold-외 적합이며 **2024 행과 lockbox 는 건드리지 않는다.**

① 방법 리서치 (실행 전)
  - Peng, Long & Ding (2005) mRMR — 최대관련성-최소중복성. 단변량 관련성 스크린은
    **상보적** 변수를 놓친다. A2 의 폐기 전제가 정확히 그 형태다.
  - 예보 검증의 표준 진단: 잔차에 구조가 남아 있으면 그 구조를 설명하는 공변량이 있다.
    남아 있지 않으면 어떤 피처공학도 소용없다.
  - 목적함수는 **L1(조건부 중앙값)** 으로 동결한다. 챔피언이 median 결합자이므로 제거
    대상은 조건부 중앙값 오프셋이다. 결과를 보고 바꾸지 않는다.
  - 사이클 23 의 교훈: **학습 모집단 = 평가 모집단**. 유효행(실측 >= 용량 10%)에서만
    학습한다.

② 사양 동결

  표적   `residual_rate = (actual - pred_M271_MEDIAN4) / capacity`, 유효행
  피처   GFS + LDAPS 의 **전 수치 컬럼** 을 격자평균한 것 + `group_id`
         (A2 와 같은 범위. 시각·리드타임 피처는 넣지 않는다 — A2 전제를 시험하는 것이
          목적이므로 NWP 컬럼 외 변수를 섞으면 판정이 흐려진다)
  분할   leave-one-fold-out (2023 Q2/Q3/Q4). 전량이 fold-외 예측
  학습기 LightGBM, `objective=regression_l1`, leaves 31, lr 0.05, 300 rounds,
         min_child_samples 100, feature_fraction 0.8, seed·스레드 고정, deterministic

  사전확약(실행 전 동결):
    H1  잔차 모형의 fold-외 `R^2 > 0.02`. 잔차에 실재하는 구조가 있다는 뜻.
    H2  보정 예측이 `M271_MEDIAN4` 대비 pooled Total 을 개선한다.
    H3  보정 예측이 **동결 게이트를 통과** 한다 (부모 = `M271_MEDIAN4`).
    H4  (A2 전제 재검) 이득 상위 20 피처 중 `spatial_v2` **미선언** 컬럼이 3 개 이상.
  H1 이 기각되면 격자평균 NWP 컬럼에는 남은 신호가 없고 이 축이 닫힌다.
  H4 가 성립하면 A2 의 폐기 전제가 뒤집힌다 — 미사용 컬럼이 **결합적으로는** 기여한다.

  **한계 명시**: 격자평균이므로 공간 세부는 지워진다. 음성 결과는 "격자평균 NWP 컬럼" 을
  닫는 것이지 "모든 공간 세부" 를 닫지 않는다.

**게이트를 수정하지 않는다.** 읽기만 한다. 2024 행·lockbox 미사용.
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
from lightgbm import LGBMRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))

from m270_gate import GATE_VERSION, evaluate_gate
from m271_cycle21_mos import FOLDS, QUARTER_OF_MONTH
from m271_cycle22_global_shift import build_base
from m271_evaluate_candidate import official
from m271_n0_columns import KEY_COLUMNS, declared_usage, grid_mean
from m271_n0_common import SEED, load_tables

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
REPORT_MD = REPORTS / "m271_cycle27_residual_signal.md"
RECEIPT = REPORTS / "m271_cycle27_residual_signal_receipt.json"

NODE_ID = "C1N27_RESIDUAL_SIGNAL"
LANE = "L2"  # 피처 구성
PARENT_NODE = "C1N20_ALPHA_ENDPOINT"
REOPENS = "AXIS_UNUSED_COLUMNS"
INCUMBENT = "M271_MEDIAN4"
ELIGIBLE_THRESHOLD = 0.10
THREADS = 1

H1_MIN_R2 = 0.02
H4_MIN_UNDECLARED_IN_TOP20 = 3

MODEL_PARAMS = {
    "objective": "regression_l1",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "n_estimators": 300,
    "min_child_samples": 100,
    "colsample_bytree": 0.8,
    "subsample": 0.8,
    "subsample_freq": 1,
    "random_state": SEED,
    "n_jobs": THREADS,
    "deterministic": True,
    "force_row_wise": True,
    "verbose": -1,
}

METHOD_SOURCES = (
    {
        "id": "peng_long_ding_2005_mrmr",
        "cite": "Peng, Long & Ding (2005) — mRMR (max-relevance min-redundancy)",
        "claim": "단변량 관련성 스크린은 이미 선택된 변수와 **조건부로만** 정보를 주는 "
                 "상보적 변수를 놓친다",
        "applicability": "directly_supported",
        "use": "A2 의 단변량 MI 폐기 전제를 재검할 근거",
    },
    {
        "id": "residual_diagnostic",
        "cite": "예보 검증의 잔차 진단 (표준 절차)",
        "claim": "잔차에 공변량 구조가 남아 있으면 설명 가능한 오차가 남은 것이고, "
                 "없으면 그 공변량으로는 개선이 불가능하다",
        "applicability": "directly_supported",
        "use": "표적을 y 가 아니라 챔피언 잔차로 두는 근거",
    },
)


def build_features() -> tuple[pd.DataFrame, set[str]]:
    tables, input_hashes = load_tables()
    usage = declared_usage()
    declared = set(usage["declared"]["gfs"]) | set(usage["declared"]["ldaps"])
    parts = []
    for source, frame in (("gfs", tables.gfs_train), ("ldaps", tables.ldaps_train)):
        numeric = [
            c for c in frame.columns
            if c not in KEY_COLUMNS and pd.api.types.is_numeric_dtype(frame[c])
        ]
        block = grid_mean(frame, numeric)
        block.columns = [f"{source}__{c}" for c in block.columns]
        parts.append(block)
    features = parts[0].join(parts[1], how="inner")
    features.attrs["input_hashes"] = input_hashes
    features.attrs["declared"] = declared
    return features, declared


def main() -> int:
    champion = build_base()
    champion = champion.rename(columns={"median_pred": "prediction_kwh"})
    champion = champion.loc[
        :, ["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh",
            "prediction_kwh", "month", "capacity", "fold"]
    ]
    incumbent_score = official(champion)

    features, declared = build_features()
    merged = champion.merge(
        features, left_on="forecast_kst_dtm", right_index=True, how="inner"
    )
    join_loss = len(champion) - len(merged)
    feature_cols = [c for c in features.columns]

    merged["residual_rate"] = (
        (merged["actual_kwh"] - merged["prediction_kwh"]) / merged["capacity"]
    )
    merged["eligible"] = merged["actual_kwh"] >= ELIGIBLE_THRESHOLD * merged["capacity"]
    merged["fold"] = merged["month"].map(QUARTER_OF_MONTH)

    x_cols = [*feature_cols, "group_id"]
    gains = np.zeros(len(x_cols))
    pieces = []
    fits = 0
    for held in FOLDS:
        train = merged.loc[(merged["fold"] != held) & merged["eligible"]]
        test = merged.loc[merged["fold"] == held].copy()
        model = LGBMRegressor(**MODEL_PARAMS)
        model.fit(
            train.loc[:, x_cols],
            train["residual_rate"],
            categorical_feature=["group_id"],
        )
        fits += 1
        test["residual_hat"] = model.predict(test.loc[:, x_cols])
        pieces.append(test)
        gains += model.booster_.feature_importance(importance_type="gain")
    oof = pd.concat(pieces, ignore_index=True)
    assert len(oof) == len(merged), "LOO 이어붙이기에서 행 수가 바뀌었다"

    e = oof.loc[oof["eligible"]]
    ss_res = float(((e["residual_rate"] - e["residual_hat"]) ** 2).sum())
    ss_tot = float(((e["residual_rate"] - e["residual_rate"].mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    corr = float(np.corrcoef(e["residual_rate"], e["residual_hat"])[0, 1])
    h1 = bool(r2 > H1_MIN_R2)

    corrected = oof.loc[
        :, ["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh", "month"]
    ].copy()
    corrected["prediction_kwh"] = (
        oof["prediction_kwh"] + oof["residual_hat"] * oof["capacity"]
    )
    base_for_gate = oof.loc[
        :, ["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh",
            "prediction_kwh", "month"]
    ].copy()
    corrected_score = official(corrected)
    base_score = official(base_for_gate)
    gate = evaluate_gate(corrected, base_for_gate)
    stats = gate.evidence
    h2 = bool(corrected_score["total"] > base_score["total"])
    h3 = bool(gate.passed)

    order = np.argsort(-gains)
    top20 = [
        {
            "feature": x_cols[i],
            "gain": float(gains[i]),
            "declared_in_spatial_v2": bool(
                x_cols[i].split("__", 1)[-1] in declared
            ) if "__" in x_cols[i] else None,
        }
        for i in order[:20]
    ]
    undeclared_in_top20 = sum(
        1 for f in top20 if f["declared_in_spatial_v2"] is False
    )
    h4 = bool(undeclared_in_top20 >= H4_MIN_UNDECLARED_IN_TOP20)

    promote = bool(h2 and h3)
    promoted_total = corrected_score["total"] if promote else base_score["total"]
    verdict = (
        "RESIDUAL_SIGNAL_EXPLOITED_PROMOTED" if promote
        else ("RESIDUAL_SIGNAL_PRESENT_NOT_EXPLOITABLE" if h1
              else "NO_RESIDUAL_SIGNAL_GRIDMEAN_NWP_CLOSED")
    )

    check = {
        "H1_expectation": f"잔차 모형 fold-외 R^2 > {H1_MIN_R2}",
        "H1_held": h1, "H1_measured": r2,
        "H2_expectation": f"보정이 {INCUMBENT} 대비 Total 개선",
        "H2_held": h2,
        "H3_expectation": "보정이 동결 게이트 통과",
        "H3_held": h3,
        "H4_expectation": f"이득 상위 20 중 미선언 컬럼 >= {H4_MIN_UNDECLARED_IN_TOP20}",
        "H4_held": h4, "H4_measured": undeclared_in_top20,
        "a2_premise_flipped": h4,
        "verdict": verdict,
    }

    payload = {
        "node": NODE_ID, "parent_node": PARENT_NODE, "reopens": REOPENS,
        "gate_version": GATE_VERSION, "gate_modified": False,
        "incumbent": INCUMBENT,
        "method_sources": list(METHOD_SOURCES),
        "model_params": MODEL_PARAMS,
        "design": "leave-one-fold-out (2023 Q2/Q3/Q4), 학습은 유효행에서만",
        "scope_limitation": "격자평균이므로 공간 세부는 지워진다. 음성 결과는 "
                            "'격자평균 NWP 컬럼'을 닫는 것이지 '모든 공간 세부'가 아니다",
        "features": {
            "n_nwp_columns": len(feature_cols),
            "n_declared": sum(
                1 for c in feature_cols if c.split("__", 1)[-1] in declared
            ),
            "join_rows_lost": join_loss,
            "rows_used": len(merged),
            "eligible_rows": int(merged["eligible"].sum()),
        },
        "residual_model": {
            "oof_r2": r2,
            "oof_pearson": corr,
            "fits": fits,
            "top20_by_gain": top20,
            "undeclared_in_top20": undeclared_in_top20,
        },
        "incumbent_full_score": incumbent_score,
        "scores": {
            "base_on_joined_rows": base_score,
            "corrected": corrected_score,
            "delta_total": corrected_score["total"] - base_score["total"],
        },
        "gate": {
            "passed": h3,
            "flags": {la.split()[0]: bool(ok) for la, ok in gate.conditions.items()},
            "positive_months": int(stats["positive_months"]),
            "months_scored": int(stats["months_scored"]),
            "sign_test_p": float(stats["sign_test_p_greater"]),
            "bootstrap_q05": float(stats["block_bootstrap_q05"]),
        },
        "predeclared_check": check,
        "promoted_total": promoted_total,
        "gap_to_target": 0.66 - promoted_total,
    }

    flags = "".join("O" if payload["gate"]["flags"].get(x) else "-"
                    for x in ("G1", "G2", "G3", "G4"))
    lines = [
        "# M271 P4 사이클 27 — 챔피언의 잔차에 NWP 신호가 남아 있는가",
        "",
        f"- 판정일: {datetime.now(UTC).strftime('%Y-%m-%d')} (UTC)",
        f"- 노드: `{NODE_ID}` / 레인 {LANE} / 부모 `{PARENT_NODE}`",
        f"- 재검 대상: `{REOPENS}` (A2 의 단변량 MI 폐기)",
        "- **이 세션의 첫 모델 적합.** 진단용 fold-외 적합이며 2024 행·lockbox 미사용",
        f"- 게이트: `{GATE_VERSION}` (읽기만 함)",
        "",
        "## 0. 방법 리서치 (실행 전)",
        "",
    ]
    for s in METHOD_SOURCES:
        lines.append(f"- **{s['cite']}** (`{s['applicability']}`)")
        lines.append(f"  - {s['claim']}")
        lines.append(f"  - 사용: {s['use']}")

    f = payload["features"]
    lines += [
        "",
        f"**한계**: {payload['scope_limitation']}",
        "",
        "## 1. 설정",
        "",
        f"- NWP 수치 컬럼 {f['n_nwp_columns']} 개 (그중 `spatial_v2` 선언 {f['n_declared']} 개)",
        f"- 조인 후 {f['rows_used']:,} 행 (유실 {f['join_rows_lost']:,}), "
        f"유효행 {f['eligible_rows']:,}",
        f"- 표적: `residual_rate` / 학습기: LightGBM `{MODEL_PARAMS['objective']}` "
        f"leaves {MODEL_PARAMS['num_leaves']} / {MODEL_PARAMS['n_estimators']} rounds",
        f"- 적합 {fits} 회 (fold 당 1 회)",
        "",
        "## 2. 잔차에 구조가 있는가 (H1)",
        "",
        "| 양 | 값 |",
        "|---|---:|",
        f"| fold-외 `R^2` | **{r2:+.4f}** |",
        f"| fold-외 Pearson | {corr:+.4f} |",
        "",
        "## 3. 그 구조가 점수로 바뀌는가 (H2 · H3)",
        "",
        "| | Total | 1-NMAE | FICR |",
        "|---|---:|---:|---:|",
        f"| `{INCUMBENT}` (조인된 행 기준) | {base_score['total']:.6f} | "
        f"{base_score['one_minus_nmae']:.6f} | {base_score['ficr']:.6f} |",
        f"| **잔차 보정** | **{corrected_score['total']:.6f}** | "
        f"{corrected_score['one_minus_nmae']:.6f} | {corrected_score['ficr']:.6f} |",
        "",
        f"차이 **{payload['scores']['delta_total']:+.6f}**. 동결 게이트 `{flags}` "
        f"{payload['gate']['positive_months']}/{payload['gate']['months_scored']}월 "
        f"p={payload['gate']['sign_test_p']:.4f} "
        f"q05={payload['gate']['bootstrap_q05']:+.6f} -> "
        f"**{'통과' if h3 else '기각'}**",
        "",
        "## 4. A2 전제 재검 — 미사용 컬럼이 결합적으로 기여하는가 (H4)",
        "",
        f"이득 상위 20 중 `spatial_v2` 미선언 **{undeclared_in_top20} 개**.",
        "",
        "| 순위 | 피처 | 이득 | v2 선언 |",
        "|---:|---|---:|:---:|",
    ]
    for rank, item in enumerate(top20, start=1):
        mark = {True: "O", False: "**X**", None: "-"}[item["declared_in_spatial_v2"]]
        lines.append(
            f"| {rank} | `{item['feature']}` | {item['gain']:,.0f} | {mark} |"
        )

    lines += [
        "",
        "## 5. 사전확약 대조",
        "",
        f"- H1 `{check['H1_expectation']}` -> **{h1}** (실측 {r2:+.4f})",
        f"- H2 `{check['H2_expectation']}` -> **{h2}**",
        f"- H3 `{check['H3_expectation']}` -> **{h3}**",
        f"- H4 `{check['H4_expectation']}` -> **{h4}** (실측 {undeclared_in_top20})",
        "",
        f"판정: **{verdict}**",
        "",
        f"승격 Total **{promoted_total:.6f}**, 목표 0.66 까지 **{0.66 - promoted_total:+.6f}**.",
        "",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    receipt: dict[str, Any] = {
        "schema_version": 1,
        "stage": "M271_P4_CYCLE27_RESIDUAL_SIGNAL",
        "node": NODE_ID, "lane": LANE,
        "decided_utc": datetime.now(UTC).isoformat(),
        "result": payload,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "report_sha256": hashlib.sha256(REPORT_MD.read_bytes()).hexdigest(),
        "dacon_upload": False, "external_actions": [],
        "model_fits": fits,
        "model_fit_note": "진단용 fold-외 잔차 모형. 제출 후보가 아니며 2024 행 미사용",
        "lockbox_reopened": False, "new_2024_evaluation": False,
    }
    RECEIPT.write_text(
        json.dumps(receipt, indent=1, ensure_ascii=False, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

    print(f"[C27] 피처 {f['n_nwp_columns']} (선언 {f['n_declared']}) / "
          f"행 {f['rows_used']:,} (유실 {f['join_rows_lost']:,}) / "
          f"유효 {f['eligible_rows']:,} / 적합 {fits} 회")
    print(f"[C27] 잔차 fold-외 R^2 {r2:+.4f}  Pearson {corr:+.4f}  -> H1 {h1}")
    print(f"[C27] 보정 Total {corrected_score['total']:.6f} "
          f"(기준 {base_score['total']:.6f}, "
          f"차이 {payload['scores']['delta_total']:+.6f}) "
          f"게이트 {'통과' if h3 else '기각'} -> H2 {h2} H3 {h3}")
    print(f"[C27] 상위20 중 미선언 {undeclared_in_top20} -> H4 {h4}")
    print(f"[C27] 판정: {verdict}  ->  Total {promoted_total:.6f} "
          f"(목표까지 {0.66 - promoted_total:+.6f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
