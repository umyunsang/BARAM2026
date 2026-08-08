"""M271 P4 사이클 31 — 공간 x 시간 동시. 2x2 설계의 마지막 칸.

사이클 27·28·30 이 잔차 설명력을 세 점에서 쟀다.

    27  동시점 x 격자평균   65 피처   R^2 -0.0535
    28  동시점 x 격자별    795 피처   R^2 -0.0453   (공간 정제 +0.0082)
    30  시간문맥 x 격자평균 585 피처  R^2 -0.0322   (시간 정제 +0.0213)

두 정제가 각각 도움이 되지만 문턱 `+0.02` 에 못 미친다. 남은 칸은 **둘 다** 다.

가법을 가정하면 `-0.0535 + 0.0082 + 0.0213 = -0.0240` 으로 여전히 못 미친다. 그러나
**가법은 가정이지 측정이 아니다.** 공간 세부와 시간 미분은 상호작용할 수 있다 — 격자간
풍속차의 시간 변화는 전선 통과의 직접 신호다. 외삽으로 축을 닫지 않고 마지막 칸을 잰다.

① 방법 리서치 (실행 전)
  - 새 방법 없음. 27 의 설계에서 공간·시간 해상도를 **동시에** 올린다. 앞선 세 노드가
    한 축씩만 움직였으므로 이 노드의 증분은 정확히 **상호작용** 에 귀속된다.
  - 피처가 5,500 개대가 되므로 정규화를 한 단계 더 조인다. 실행 전에 정한다.

② 사양 동결

  피처   GFS 9 격자 + LDAPS 16 격자 (격자별) 전 컬럼에 대해
           contemporaneous, lag {1,3}h, lead {1,3}h, diff {1}h
         + `group_id`
         (시간 변환 폭은 30 의 {1,3,6}+{1,3} 에서 줄인다. 격자별로 곱해지면 5,500 개를
          넘기 때문이며, 30 의 이득 상위에서 lag/lead 1~3h 가 지배적이었는지는 보지 않고
          **차원 이유만으로** 줄인다 — 결과를 본 뒤 고르면 same-fold 선택이다)
  그 외는 동일: 표적 `residual_rate`, leave-one-fold-out, 유효행 학습.

  사전확약(실행 전 동결):
    H1  fold-외 `R^2 > 0.02`.
    H2  `R^2` 가 앞선 셋의 최고값(-0.0322)보다 크다.
    H3  두 정제가 **초가법(super-additive)** 이다: 증분이 `0.0082 + 0.0213 = 0.0295`
        보다 크다. 성립하면 공간과 시간이 상호작용한다는 직접 증거다.
    H4  보정이 `M271_MEDIAN4` 대비 Total 개선 + 동결 게이트 통과.

  H1 이 기각되면 **공급 NWP 로 챔피언 잔차를 설명하는 경로가 2x2 전 칸에서 닫힌다.**

**게이트를 수정하지 않는다. lockbox 를 열지 않는다.** 2024 행 미사용.
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
from m271_cycle28_pergrid import per_grid_block
from m271_cycle30_temporal import is_temporal
from m271_evaluate_candidate import official
from m271_n0_common import SEED, load_tables

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
REPORT_MD = REPORTS / "m271_cycle31_space_time.md"
RECEIPT = REPORTS / "m271_cycle31_space_time_receipt.json"

NODE_ID = "C1N31_SPACE_TIME"
LANE = "L2"
PARENT_NODE = "C1N30_TEMPORAL_SIGNAL"
INCUMBENT = "M271_MEDIAN4"
ELIGIBLE_THRESHOLD = 0.10
THREADS = 1

R2_CONTEMPORANEOUS_GRIDMEAN = -0.0535  # C27
R2_CONTEMPORANEOUS_PERGRID = -0.0453  # C28
R2_TEMPORAL_GRIDMEAN = -0.0322  # C30
SPACE_INCREMENT = R2_CONTEMPORANEOUS_PERGRID - R2_CONTEMPORANEOUS_GRIDMEAN
TIME_INCREMENT = R2_TEMPORAL_GRIDMEAN - R2_CONTEMPORANEOUS_GRIDMEAN
ADDITIVE_PREDICTION = R2_CONTEMPORANEOUS_GRIDMEAN + SPACE_INCREMENT + TIME_INCREMENT

H1_MIN_R2 = 0.02
LAGS = (1, 3)
LEADS = (1, 3)
DIFFS = (1,)

MODEL_PARAMS = {
    "objective": "regression_l1",
    "num_leaves": 15,
    "learning_rate": 0.03,
    "n_estimators": 400,
    "min_child_samples": 300,
    "colsample_bytree": 0.15,
    "subsample": 0.8,
    "subsample_freq": 1,
    "random_state": SEED,
    "n_jobs": THREADS,
    "deterministic": True,
    "force_row_wise": True,
    "verbose": -1,
}


def build_space_time() -> tuple[pd.DataFrame, dict[str, int]]:
    tables, _ = load_tables()
    spatial = per_grid_block(tables.gfs_train, "gfs").join(
        per_grid_block(tables.ldaps_train, "ldaps"), how="inner"
    )
    deltas = pd.Series(spatial.index).diff().dropna()
    assert (deltas == pd.Timedelta(hours=1)).all(), (
        "시간 인덱스가 1 시간 등간격이 아니다. shift 로 lag 을 만들 수 없다"
    )

    blocks = [spatial.astype("float32")]
    counts = {"contemporaneous": spatial.shape[1]}
    for lag in LAGS:
        b = spatial.shift(lag).astype("float32")
        b.columns = [f"{c}__lag{lag}h" for c in spatial.columns]
        blocks.append(b)
    counts["lag"] = len(LAGS) * spatial.shape[1]
    for lead in LEADS:
        b = spatial.shift(-lead).astype("float32")
        b.columns = [f"{c}__lead{lead}h" for c in spatial.columns]
        blocks.append(b)
    counts["lead"] = len(LEADS) * spatial.shape[1]
    for d in DIFFS:
        b = spatial.diff(d).astype("float32")
        b.columns = [f"{c}__diff{d}h" for c in spatial.columns]
        blocks.append(b)
    counts["diff"] = len(DIFFS) * spatial.shape[1]
    return pd.concat(blocks, axis=1), counts


def main() -> int:
    champion = build_base().rename(columns={"median_pred": "prediction_kwh"})
    champion = champion.loc[
        :, ["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh",
            "prediction_kwh", "month", "capacity"]
    ]

    features, counts = build_space_time()
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
    observed_increment = r2 - R2_CONTEMPORANEOUS_GRIDMEAN

    h1 = bool(r2 > H1_MIN_R2)
    h2 = bool(r2 > R2_TEMPORAL_GRIDMEAN)
    h3 = bool(observed_increment > SPACE_INCREMENT + TIME_INCREMENT)

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
    h4 = bool(corrected_score["total"] > base_score["total"] and gate.passed)

    order = np.argsort(-gains)
    top20 = [
        {"feature": x_cols[i], "gain": float(gains[i]),
         "temporal": is_temporal(x_cols[i])}
        for i in order[:20]
    ]

    promoted_total = corrected_score["total"] if h4 else base_score["total"]
    verdict = (
        "SPACE_TIME_EXPLOITED_PROMOTED" if h4
        else ("SPACE_TIME_SIGNAL_PRESENT_NOT_EXPLOITABLE" if h1
              else "SUPPLIED_NWP_CLOSED_ALL_FOUR_CELLS")
    )

    check = {
        "H1_expectation": f"fold-외 R^2 > {H1_MIN_R2}",
        "H1_held": h1, "H1_measured": r2,
        "H2_expectation": f"R^2 > 앞선 최고 ({R2_TEMPORAL_GRIDMEAN})",
        "H2_held": h2,
        "H3_expectation": f"증분이 가법 예측 ({SPACE_INCREMENT + TIME_INCREMENT:.4f}) 초과 "
                          "— 공간x시간 상호작용",
        "H3_held": h3, "H3_measured_increment": observed_increment,
        "H4_expectation": "보정이 Total 개선 + 동결 게이트 통과",
        "H4_held": h4,
        "verdict": verdict,
    }

    payload = {
        "node": NODE_ID, "parent_node": PARENT_NODE,
        "gate_version": GATE_VERSION, "gate_modified": False,
        "incumbent": INCUMBENT,
        "design": "2x2 (공간 해상도 x 시간 해상도) 의 마지막 칸",
        "grid_of_measurements": {
            "contemporaneous_gridmean": R2_CONTEMPORANEOUS_GRIDMEAN,
            "contemporaneous_pergrid": R2_CONTEMPORANEOUS_PERGRID,
            "temporal_gridmean": R2_TEMPORAL_GRIDMEAN,
            "temporal_pergrid": r2,
            "space_increment": SPACE_INCREMENT,
            "time_increment": TIME_INCREMENT,
            "additive_prediction": ADDITIVE_PREDICTION,
            "observed_increment": observed_increment,
            "interaction": observed_increment - (SPACE_INCREMENT + TIME_INCREMENT),
        },
        "temporal_transforms": {"lags_h": list(LAGS), "leads_h": list(LEADS),
                                "diffs_h": list(DIFFS)},
        "feature_counts": counts,
        "model_params": MODEL_PARAMS,
        "features": {
            "n_features": len(feature_cols),
            "join_rows_lost": join_loss,
            "rows_used": len(merged),
            "eligible_rows": int(merged["eligible"].sum()),
        },
        "residual_model": {"oof_r2": r2, "oof_pearson": corr, "fits": fits,
                           "top20_by_gain": top20},
        "scores": {"base": base_score, "corrected": corrected_score,
                   "delta_total": corrected_score["total"] - base_score["total"]},
        "gate": {
            "passed": bool(gate.passed),
            "flags": {la.split()[0]: bool(ok) for la, ok in gate.conditions.items()},
            "positive_months": int(stats["positive_months"]),
            "months_scored": int(stats["months_scored"]),
            "bootstrap_q05": float(stats["block_bootstrap_q05"]),
        },
        "predeclared_check": check,
        "promoted_total": promoted_total,
        "gap_to_target": 0.66 - promoted_total,
    }

    g = payload["grid_of_measurements"]
    f = payload["features"]
    flags = "".join("O" if payload["gate"]["flags"].get(x) else "-"
                    for x in ("G1", "G2", "G3", "G4"))
    lines = [
        "# M271 P4 사이클 31 — 공간 x 시간 동시 (2x2 마지막 칸)",
        "",
        f"- 판정일: {datetime.now(UTC).strftime('%Y-%m-%d')} (UTC)",
        f"- 노드: `{NODE_ID}` / 레인 {LANE} / 부모 `{PARENT_NODE}`",
        f"- 게이트: `{GATE_VERSION}` (읽기만 함) / lockbox 미개봉 / 2024 행 미사용",
        "",
        "## 1. 2x2 격자",
        "",
        "| | 격자평균 | 격자별 |",
        "|---|---:|---:|",
        f"| **동시점** | {R2_CONTEMPORANEOUS_GRIDMEAN:+.4f} (C27) | "
        f"{R2_CONTEMPORANEOUS_PERGRID:+.4f} (C28) |",
        f"| **시간문맥** | {R2_TEMPORAL_GRIDMEAN:+.4f} (C30) | **{r2:+.4f}** (이 노드) |",
        "",
        f"- 공간 정제 증분 `{SPACE_INCREMENT:+.4f}` / 시간 정제 증분 `{TIME_INCREMENT:+.4f}`",
        f"- 가법 예측 `{ADDITIVE_PREDICTION:+.4f}` / 실측 `{r2:+.4f}`",
        f"- **상호작용 `{g['interaction']:+.4f}`**",
        "",
        "## 2. 설정",
        "",
        f"- 피처 **{f['n_features']:,}** (동시점 {counts['contemporaneous']:,} + "
        f"lag {counts['lag']:,} + lead {counts['lead']:,} + diff {counts['diff']:,})",
        f"- 유효행 {f['eligible_rows']:,}, 적합 {fits} 회, "
        f"colsample {MODEL_PARAMS['colsample_bytree']}, "
        f"min_child {MODEL_PARAMS['min_child_samples']}",
        f"- fold-외 Pearson {corr:+.4f}",
        "",
        "## 3. 점수 (H4)",
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
        "## 4. 사전확약 대조",
        "",
        f"- H1 `{check['H1_expectation']}` -> **{h1}** (실측 {r2:+.4f})",
        f"- H2 `{check['H2_expectation']}` -> **{h2}**",
        f"- H3 `{check['H3_expectation']}` -> **{h3}** "
        f"(실측 증분 {observed_increment:+.4f})",
        f"- H4 `{check['H4_expectation']}` -> **{h4}**",
        "",
        f"판정: **{verdict}**",
        "",
        f"승격 Total **{promoted_total:.6f}**, 목표 0.66 까지 **{0.66 - promoted_total:+.6f}**.",
        "",
        "## 5. 이득 상위 20",
        "",
        "| 순위 | 피처 | 이득 | 시간변환 |",
        "|---:|---|---:|:---:|",
    ]
    for rank, item in enumerate(top20, start=1):
        lines.append(
            f"| {rank} | `{item['feature']}` | {item['gain']:,.0f} | "
            f"{'O' if item['temporal'] else '-'} |"
        )
    if not h1:
        lines += [
            "",
            "## 6. 이것이 확정하는 것",
            "",
            "공간 x 시간 2x2 **네 칸 모두** fold-외 `R^2` 가 음수다. 챔피언의 잔차는 공급",
            "NWP 의 어떤 해상도 조합으로도 설명되지 않는다.",
            "",
            "**같은 입력으로 새 기저모델을 만드는 경로가 닫힌다.** 새 모델이 고칠 수 있는",
            "오차가 있었다면 이 2x2 어딘가에서 잔차 구조로 나타났어야 한다.",
        ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    receipt: dict[str, Any] = {
        "schema_version": 1,
        "stage": "M271_P4_CYCLE31_SPACE_TIME",
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

    print(f"[C31] 피처 {f['n_features']:,} / 유효행 {f['eligible_rows']:,} / 적합 {fits} 회")
    print(f"[C31] 2x2:  동시점x격자평균 {R2_CONTEMPORANEOUS_GRIDMEAN:+.4f} | "
          f"동시점x격자별 {R2_CONTEMPORANEOUS_PERGRID:+.4f} | "
          f"시간x격자평균 {R2_TEMPORAL_GRIDMEAN:+.4f} | **시간x격자별 {r2:+.4f}**")
    print(f"[C31] 가법예측 {ADDITIVE_PREDICTION:+.4f} vs 실측 {r2:+.4f}  "
          f"상호작용 {g['interaction']:+.4f}")
    print(f"[C31] H1 {h1} | H2 {h2} | H3 초가법 {h3} | H4 {h4}")
    print(f"[C31] 판정: {verdict}  ->  Total {promoted_total:.6f} "
          f"(목표까지 {0.66 - promoted_total:+.6f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
