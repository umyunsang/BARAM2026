"""M271 P4 사이클 28 — 격자평균이 지운 공간 세부에 신호가 있는가.

사이클 27 이 격자평균 NWP 컬럼으로 잔차를 설명하지 못했다(fold-외 `R^2 = -0.0535`).
그 노드가 사양에 명시한 한계가 하나 있다: **격자평균은 공간 세부를 지운다.** A3 가 쟀듯
17 기 터빈이 GFS 격자 1 개를 공유하지만 LDAPS 는 16 격자를 준다. 평균내면 그 구조가 사라진다.

이 노드가 그 마지막 구멍을 막는다. 격자별로 풀어서 같은 질문을 다시 던진다.

또 사이클 27 의 **설계 결함** 을 고친다. H4 는 "이득 상위 20 중 미선언 컬럼 >= 3" 이었고
성립했지만(12 개), 그 모형의 fold-외 `R^2` 가 음수였다. **일반화하지 못하는 모형의 피처
중요도는 정보의 증거가 아니다.** 중요도 해석을 fold-외 성능에 **조건화** 한다.

① 방법 리서치 (실행 전)
  - 새 방법 없음. 사이클 27 의 설계를 그대로 쓰고 **피처의 공간 해상도만** 바꾼다.
    한 변수만 바꿔야 원인을 귀속할 수 있다.
  - 고차원 대비 정규화를 **실행 전에** 강화한다. 피처 약 770 개, 학습행 약 7,700 개로
    비율이 1:10 이다. leaves 31->15, colsample 0.8->0.3, min_child 100->200.
    결과를 보고 조정하지 않는다.

② 사양 동결

  피처   GFS 9 격자 x 수치변수 + LDAPS 16 격자 x 수치변수 (격자별로 편 것) + `group_id`
  그 외는 사이클 27 과 동일: 표적 `residual_rate`, leave-one-fold-out, 유효행 학습.

  사전확약(실행 전 동결):
    H1  격자별 모형의 fold-외 `R^2 > 0.02`.
    H2  격자별 `R^2` 가 격자평균 `R^2`(-0.0535)보다 **크다**. 공간 세부가 무언가를 더한다.
    H3  보정이 `M271_MEDIAN4` 대비 Total 개선 + **동결 게이트 통과**.
    H4  **H1 이 성립할 때만 판정한다.** 이득 상위 20 중 `spatial_v2` 미선언 >= 3.
        H1 이 기각되면 H4 는 `판정불가` 로 기록하고 A2 전제를 뒤집지 않는다.

  H1 이 기각되면 **격자평균이든 격자별이든 공급 NWP 로는 이 잔차를 설명할 수 없다** 가
  확정되고, L2(피처 구성) 레인 전체가 공급 데이터 범위에서 닫힌다.

**게이트를 수정하지 않는다.** 2024 행·lockbox 미사용.
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
from m271_n0_columns import KEY_COLUMNS, declared_usage
from m271_n0_common import SEED, load_tables

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
REPORT_MD = REPORTS / "m271_cycle28_pergrid.md"
RECEIPT = REPORTS / "m271_cycle28_pergrid_receipt.json"

NODE_ID = "C1N28_PERGRID_SIGNAL"
LANE = "L2"
PARENT_NODE = "C1N27_RESIDUAL_SIGNAL"
INCUMBENT = "M271_MEDIAN4"
ELIGIBLE_THRESHOLD = 0.10
THREADS = 1

GRIDMEAN_R2 = -0.0535  # 사이클 27 실측
H1_MIN_R2 = 0.02

MODEL_PARAMS = {
    "objective": "regression_l1",
    "num_leaves": 15,
    "learning_rate": 0.03,
    "n_estimators": 400,
    "min_child_samples": 200,
    "colsample_bytree": 0.3,
    "subsample": 0.8,
    "subsample_freq": 1,
    "random_state": SEED,
    "n_jobs": THREADS,
    "deterministic": True,
    "force_row_wise": True,
    "verbose": -1,
}


def per_grid_block(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    numeric = [
        c for c in frame.columns
        if c not in KEY_COLUMNS and pd.api.types.is_numeric_dtype(frame[c])
    ]
    wide = frame.pivot_table(
        index="forecast_kst_dtm", columns="grid_id", values=numeric, aggfunc="mean"
    ).sort_index()
    wide.columns = [f"{source}__{var}__g{grid}" for var, grid in wide.columns]
    return wide


def main() -> int:
    champion = build_base().rename(columns={"median_pred": "prediction_kwh"})
    champion = champion.loc[
        :, ["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh",
            "prediction_kwh", "month", "capacity"]
    ]

    tables, input_hashes = load_tables()
    declared = set(declared_usage()["declared"]["gfs"]) | set(
        declared_usage()["declared"]["ldaps"]
    )
    gfs = per_grid_block(tables.gfs_train, "gfs")
    ldaps = per_grid_block(tables.ldaps_train, "ldaps")
    features = gfs.join(ldaps, how="inner")
    feature_cols = list(features.columns)

    merged = champion.merge(
        features, left_on="forecast_kst_dtm", right_index=True, how="inner"
    )
    join_loss = len(champion) - len(merged)
    merged["residual_rate"] = (
        (merged["actual_kwh"] - merged["prediction_kwh"]) / merged["capacity"]
    )
    merged["eligible"] = merged["actual_kwh"] >= ELIGIBLE_THRESHOLD * merged["capacity"]
    merged["fold"] = merged["month"].map(QUARTER_OF_MONTH)
    assert merged["fold"].notna().all(), "fold 매핑에 구멍이 있다"

    x_cols = [*feature_cols, "group_id"]
    gains = np.zeros(len(x_cols))
    pieces = []
    fits = 0
    for held in FOLDS:
        train = merged.loc[(merged["fold"] != held) & merged["eligible"]]
        test = merged.loc[merged["fold"] == held].copy()
        model = LGBMRegressor(**MODEL_PARAMS)
        model.fit(
            train.loc[:, x_cols], train["residual_rate"],
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
    h2 = bool(r2 > GRIDMEAN_R2)

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
    h3 = bool(corrected_score["total"] > base_score["total"] and gate.passed)

    order = np.argsort(-gains)
    top20 = [
        {
            "feature": x_cols[i],
            "gain": float(gains[i]),
            "declared_in_spatial_v2": (
                bool(x_cols[i].split("__")[1] in declared)
                if x_cols[i].count("__") == 2 else None
            ),
        }
        for i in order[:20]
    ]
    undeclared_in_top20 = sum(1 for f in top20 if f["declared_in_spatial_v2"] is False)
    # 설계 교정: 중요도 해석은 fold-외 성능에 조건화한다.
    h4: bool | None = (undeclared_in_top20 >= 3) if h1 else None

    promote = h3
    promoted_total = corrected_score["total"] if promote else base_score["total"]
    verdict = (
        "PERGRID_SIGNAL_EXPLOITED_PROMOTED" if promote
        else ("PERGRID_SIGNAL_PRESENT_NOT_EXPLOITABLE" if h1
              else "L2_LANE_CLOSED_WITHIN_SUPPLIED_NWP")
    )

    check = {
        "H1_expectation": f"격자별 모형 fold-외 R^2 > {H1_MIN_R2}",
        "H1_held": h1, "H1_measured": r2,
        "H2_expectation": f"격자별 R^2 > 격자평균 R^2 ({GRIDMEAN_R2})",
        "H2_held": h2,
        "H3_expectation": "보정이 Total 개선 + 동결 게이트 통과",
        "H3_held": h3,
        "H4_expectation": "이득 상위 20 중 미선언 >= 3 — **H1 성립시에만 판정**",
        "H4_held": h4,
        "H4_note": (
            "H1 기각이므로 판정하지 않는다. 일반화하지 못하는 모형의 피처 중요도는 "
            "정보의 증거가 아니다 (사이클 27 의 설계 결함 교정)"
        ) if not h1 else "H1 성립하므로 판정 유효",
        "a2_premise_flipped": bool(h4) if h4 is not None else False,
        "verdict": verdict,
    }

    payload = {
        "node": NODE_ID, "parent_node": PARENT_NODE,
        "gate_version": GATE_VERSION, "gate_modified": False,
        "incumbent": INCUMBENT,
        "design_correction_of": "C1N27_RESIDUAL_SIGNAL H4",
        "model_params": MODEL_PARAMS,
        "input_hashes": input_hashes,
        "features": {
            "n_features": len(feature_cols),
            "gfs_grids": int(tables.gfs_train["grid_id"].nunique()),
            "ldaps_grids": int(tables.ldaps_train["grid_id"].nunique()),
            "join_rows_lost": join_loss,
            "rows_used": len(merged),
            "eligible_rows": int(merged["eligible"].sum()),
            "feature_to_train_row_ratio": len(feature_cols)
            / max(int(merged["eligible"].sum()) * 2 // 3, 1),
        },
        "residual_model": {
            "oof_r2": r2,
            "oof_pearson": corr,
            "gridmean_r2_reference": GRIDMEAN_R2,
            "fits": fits,
            "top20_by_gain": top20,
            "undeclared_in_top20": undeclared_in_top20,
        },
        "scores": {
            "base": base_score, "corrected": corrected_score,
            "delta_total": corrected_score["total"] - base_score["total"],
        },
        "gate": {
            "passed": bool(gate.passed),
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

    f = payload["features"]
    flags = "".join("O" if payload["gate"]["flags"].get(x) else "-"
                    for x in ("G1", "G2", "G3", "G4"))
    lines = [
        "# M271 P4 사이클 28 — 격자평균이 지운 공간 세부에 신호가 있는가",
        "",
        f"- 판정일: {datetime.now(UTC).strftime('%Y-%m-%d')} (UTC)",
        f"- 노드: `{NODE_ID}` / 레인 {LANE} / 부모 `{PARENT_NODE}`",
        "- 사이클 27 에서 **피처의 공간 해상도만** 바꿨다. 한 변수만 움직여야 귀속이 된다",
        f"- 게이트: `{GATE_VERSION}` (읽기만 함) / 2024 행·lockbox 미사용",
        "",
        "## 1. 설정",
        "",
        f"- GFS {f['gfs_grids']} 격자 + LDAPS {f['ldaps_grids']} 격자를 풀어 "
        f"**피처 {f['n_features']:,} 개**",
        f"- 행 {f['rows_used']:,} (유실 {f['join_rows_lost']:,}), "
        f"유효행 {f['eligible_rows']:,}, 피처/학습행 비 "
        f"**{f['feature_to_train_row_ratio']:.3f}**",
        f"- 정규화를 실행 전에 강화: leaves {MODEL_PARAMS['num_leaves']}, "
        f"colsample {MODEL_PARAMS['colsample_bytree']}, "
        f"min_child {MODEL_PARAMS['min_child_samples']}",
        "",
        "## 2. 공간 세부가 무언가를 더하는가 (H1 · H2)",
        "",
        "| 모형 | fold-외 R^2 | fold-외 Pearson |",
        "|---|---:|---:|",
        f"| 사이클 27 격자평균 (65 피처) | {GRIDMEAN_R2:+.4f} | +0.0699 |",
        f"| **사이클 28 격자별 ({f['n_features']:,} 피처)** | **{r2:+.4f}** | {corr:+.4f} |",
        "",
        "## 3. 점수 (H3)",
        "",
        "| | Total | 1-NMAE | FICR |",
        "|---|---:|---:|---:|",
        f"| `{INCUMBENT}` | {base_score['total']:.6f} | "
        f"{base_score['one_minus_nmae']:.6f} | {base_score['ficr']:.6f} |",
        f"| 잔차 보정 | {corrected_score['total']:.6f} | "
        f"{corrected_score['one_minus_nmae']:.6f} | {corrected_score['ficr']:.6f} |",
        "",
        f"차이 **{payload['scores']['delta_total']:+.6f}**, 게이트 `{flags}` "
        f"{payload['gate']['positive_months']}/{payload['gate']['months_scored']}월 -> "
        f"**{'통과' if payload['gate']['passed'] else '기각'}**",
        "",
        "## 4. 피처 중요도 — 조건부 판정 (H4)",
        "",
        f"{check['H4_note']}",
        "",
        f"참고로 이득 상위 20 중 미선언은 {undeclared_in_top20} 개다. "
        f"H1 이 기각된 상태에서 이 수치는 **모형이 어디서 잡음을 찾았는지**를 말할 뿐이다.",
        "",
        "| 순위 | 피처 | 이득 | v2 선언 |",
        "|---:|---|---:|:---:|",
    ]
    for rank, item in enumerate(top20, start=1):
        mark = {True: "O", False: "X", None: "-"}[item["declared_in_spatial_v2"]]
        lines.append(f"| {rank} | `{item['feature']}` | {item['gain']:,.0f} | {mark} |")

    lines += [
        "",
        "## 5. 사전확약 대조",
        "",
        f"- H1 `{check['H1_expectation']}` -> **{h1}** (실측 {r2:+.4f})",
        f"- H2 `{check['H2_expectation']}` -> **{h2}**",
        f"- H3 `{check['H3_expectation']}` -> **{h3}**",
        f"- H4 `{check['H4_expectation']}` -> **{h4 if h4 is not None else '판정불가'}**",
        "",
        f"판정: **{verdict}**",
        "",
        f"승격 Total **{promoted_total:.6f}**, 목표 0.66 까지 **{0.66 - promoted_total:+.6f}**.",
        "",
    ]
    if not h1:
        lines += [
            "## 6. 이것이 확정하는 것",
            "",
            "격자평균(사이클 27)이든 격자별(이 노드)이든 **공급 NWP 로는 챔피언의 잔차를**",
            "**설명할 수 없다.** 두 해상도 모두 fold-외 `R^2` 가 문턱 미달이다. L2(피처 구성)",
            "레인은 공급 데이터 범위 안에서 닫힌다.",
            "",
            "닫히지 **않는** 것: 외부 데이터(규칙상 허용, 다만 C1N10 이 블렌딩 이득을 "
            "0.10~0.37% 로 쟀다), 그리고 라벨 자체의 시계열 구조(NWP 가 아닌 경로).",
        ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    receipt: dict[str, Any] = {
        "schema_version": 1,
        "stage": "M271_P4_CYCLE28_PERGRID",
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

    print(f"[C28] 피처 {f['n_features']:,} (GFS {f['gfs_grids']}격자 + "
          f"LDAPS {f['ldaps_grids']}격자) / 유효행 {f['eligible_rows']:,} / 적합 {fits} 회")
    print(f"[C28] 격자별 R^2 {r2:+.4f} vs 격자평균 {GRIDMEAN_R2:+.4f}  "
          f"Pearson {corr:+.4f}  -> H1 {h1} H2 {h2}")
    print(f"[C28] 보정 Total {corrected_score['total']:.6f} "
          f"(차이 {payload['scores']['delta_total']:+.6f}) -> H3 {h3}")
    print(f"[C28] H4 {h4 if h4 is not None else '판정불가 (H1 기각)'}  "
          f"(상위20 미선언 {undeclared_in_top20})")
    print(f"[C28] 판정: {verdict}  ->  Total {promoted_total:.6f} "
          f"(목표까지 {0.66 - promoted_total:+.6f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
