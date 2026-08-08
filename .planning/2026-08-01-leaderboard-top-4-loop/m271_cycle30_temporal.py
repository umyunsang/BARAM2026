"""M271 P4 사이클 30 — 잔차가 NWP 의 **시간 구조**로 설명되는가.

사이클 27·28 이 잔차를 NWP 로 설명하지 못했다(격자평균 R^2 -0.0535, 격자별 -0.0453).
그런데 두 노드 모두 **동시점 NWP 만** 썼다. 이건 누락이다.

풍력 예보 오차를 지배하는 것은 램프(ramp)이고 램프는 풍속의 **시간 미분** 이다. A1 이
라벨 lag-1 자기상관 0.951~0.962 를, A4 가 일주기 편향을 쟀는데 잔차 프로브에는 시간
문맥이 하나도 들어가지 않았다.

**유출 없음**: 대상일 NWP 는 09:00 KST 단일 초기화가 D+1 01:00~D+2 00:00 을 통째로 준다
(A6 감사). 같은 배치 안의 `t +- 3h` 는 전부 예보시점 가용이고, 배치 경계를 넘어가는 lag 은
**더 오래된** 예보이므로 더더욱 가용하다.

① 방법 리서치 (실행 전)
  - 새 학습 방법은 없다. 사이클 27 의 설계를 그대로 두고 **피처의 시간 해상도만** 바꾼다.
    사이클 28 이 공간 해상도만 바꿨듯, 한 축씩 움직여야 귀속이 된다.
  - 근거는 도메인 물리다. 발전량은 풍속의 3 승에 비례하고 파워커브 급경사 구간에서는
    작은 풍속 변화가 큰 출력 변화를 낳는다(A5 가 이 사이트에서 측정). 급경사 구간을
    가르는 것은 수준이 아니라 **변화율** 이며, 동시점 값만으로는 표현되지 않는다.
  - 시간 변환은 **모든 컬럼에 일률 적용** 한다. 어느 변수에 붙일지 고르면 same-fold
    선택이 된다.

② 사양 동결

  피처   격자평균 NWP 전 컬럼에 대해
           contemporaneous, lag {1,3,6}h, lead {1,3,6}h, diff {1,3}h
         + `group_id`. 변환은 예외 없이 전 컬럼에 적용.
  그 외는 사이클 27 과 동일: 표적 `residual_rate`, leave-one-fold-out, 유효행 학습.
  정규화는 사이클 28 의 고차원 설정을 쓴다(피처가 다시 수백 개가 되므로).

  사전확약(실행 전 동결):
    H1  시간문맥 모형의 fold-외 `R^2 > 0.02`.
    H2  시간문맥 `R^2` 가 동시점 `R^2`(-0.0535)보다 **크다**.
    H3  보정이 `M271_MEDIAN4` 대비 Total 개선 + **동결 게이트 통과**.
    H4  **H1 이 성립할 때만 판정한다** (사이클 28 의 교정된 설계). 이득 상위 20 중
        시간변환 피처(lag/lead/diff)가 5 개 이상.

  H1 이 기각되면 공급 NWP 의 **공간·시간 어느 해상도로도** 잔차가 설명되지 않는 것이
  확정되고, 새 기저모델을 같은 입력으로 만드는 경로가 닫힌다.

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
from m271_cycle27_residual_signal import build_features
from m271_cycle28_pergrid import MODEL_PARAMS
from m271_evaluate_candidate import official

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
REPORT_MD = REPORTS / "m271_cycle30_temporal.md"
RECEIPT = REPORTS / "m271_cycle30_temporal_receipt.json"

NODE_ID = "C1N30_TEMPORAL_SIGNAL"
LANE = "L2"
PARENT_NODE = "C1N27_RESIDUAL_SIGNAL"
INCUMBENT = "M271_MEDIAN4"
ELIGIBLE_THRESHOLD = 0.10

CONTEMPORANEOUS_R2 = -0.0535  # 사이클 27 실측
PERGRID_R2 = -0.0453  # 사이클 28 실측
H1_MIN_R2 = 0.02
H4_MIN_TEMPORAL_IN_TOP20 = 5

LAGS = (1, 3, 6)
LEADS = (1, 3, 6)
DIFFS = (1, 3)


def add_temporal(features: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """전 컬럼에 일률 적용. 어느 변수에 붙일지 고르지 않는다."""
    index = features.index
    deltas = pd.Series(index).diff().dropna()
    assert (deltas == pd.Timedelta(hours=1)).all(), (
        "시간 인덱스가 1 시간 등간격이 아니다. shift 로 lag 을 만들 수 없다"
    )
    blocks = [features]
    counts = {"contemporaneous": features.shape[1]}
    for lag in LAGS:
        b = features.shift(lag)
        b.columns = [f"{c}__lag{lag}h" for c in features.columns]
        blocks.append(b)
    counts["lag"] = len(LAGS) * features.shape[1]
    for lead in LEADS:
        b = features.shift(-lead)
        b.columns = [f"{c}__lead{lead}h" for c in features.columns]
        blocks.append(b)
    counts["lead"] = len(LEADS) * features.shape[1]
    for d in DIFFS:
        b = features.diff(d)
        b.columns = [f"{c}__diff{d}h" for c in features.columns]
        blocks.append(b)
    counts["diff"] = len(DIFFS) * features.shape[1]
    return pd.concat(blocks, axis=1), counts


def is_temporal(name: str) -> bool:
    return "__lag" in name or "__lead" in name or "__diff" in name


def main() -> int:
    champion = build_base().rename(columns={"median_pred": "prediction_kwh"})
    champion = champion.loc[
        :, ["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh",
            "prediction_kwh", "month", "capacity"]
    ]

    base_features, _ = build_features()
    features, counts = add_temporal(base_features)
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
    h2 = bool(r2 > CONTEMPORANEOUS_R2)

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
        {"feature": x_cols[i], "gain": float(gains[i]),
         "temporal": is_temporal(x_cols[i])}
        for i in order[:20]
    ]
    temporal_in_top20 = sum(1 for f in top20 if f["temporal"])
    h4: bool | None = (temporal_in_top20 >= H4_MIN_TEMPORAL_IN_TOP20) if h1 else None

    promote = h3
    promoted_total = corrected_score["total"] if promote else base_score["total"]
    verdict = (
        "TEMPORAL_SIGNAL_EXPLOITED_PROMOTED" if promote
        else ("TEMPORAL_SIGNAL_PRESENT_NOT_EXPLOITABLE" if h1
              else "SUPPLIED_NWP_CLOSED_IN_SPACE_AND_TIME")
    )

    check = {
        "H1_expectation": f"시간문맥 모형 fold-외 R^2 > {H1_MIN_R2}",
        "H1_held": h1, "H1_measured": r2,
        "H2_expectation": f"시간문맥 R^2 > 동시점 R^2 ({CONTEMPORANEOUS_R2})",
        "H2_held": h2,
        "H3_expectation": "보정이 Total 개선 + 동결 게이트 통과",
        "H3_held": h3,
        "H4_expectation": f"이득 상위 20 중 시간변환 >= {H4_MIN_TEMPORAL_IN_TOP20} "
                          "— **H1 성립시에만 판정**",
        "H4_held": h4,
        "H4_note": (
            "H1 기각이므로 판정하지 않는다 (사이클 28 에서 교정한 설계)"
        ) if not h1 else "H1 성립하므로 판정 유효",
        "verdict": verdict,
    }

    payload = {
        "node": NODE_ID, "parent_node": PARENT_NODE,
        "gate_version": GATE_VERSION, "gate_modified": False,
        "incumbent": INCUMBENT,
        "leakage_argument": "대상일 NWP 는 09:00 KST 단일 초기화가 D+1 01:00~D+2 00:00 을 "
                            "통째로 준다(A6). 같은 배치 안의 lead 는 가용하고, 배치 경계를 "
                            "넘는 lag 은 더 오래된 예보이므로 더더욱 가용하다",
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
        "residual_model": {
            "oof_r2": r2, "oof_pearson": corr,
            "contemporaneous_r2_reference": CONTEMPORANEOUS_R2,
            "pergrid_r2_reference": PERGRID_R2,
            "fits": fits, "top20_by_gain": top20,
            "temporal_in_top20": temporal_in_top20,
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
        "# M271 P4 사이클 30 — 잔차가 NWP 의 시간 구조로 설명되는가",
        "",
        f"- 판정일: {datetime.now(UTC).strftime('%Y-%m-%d')} (UTC)",
        f"- 노드: `{NODE_ID}` / 레인 {LANE} / 부모 `{PARENT_NODE}`",
        "- 사이클 27 에서 **피처의 시간 해상도만** 바꿨다 (28 이 공간을 바꿨듯)",
        f"- 게이트: `{GATE_VERSION}` (읽기만 함) / lockbox 미개봉 / 2024 행 미사용",
        "",
        "## 1. 유출 논거",
        "",
        payload["leakage_argument"],
        "",
        "## 2. 설정",
        "",
        f"- 변환: lag {LAGS} h / lead {LEADS} h / diff {DIFFS} h, **전 컬럼 일률 적용**",
        f"- 동시점 {counts['contemporaneous']} + lag {counts['lag']} + "
        f"lead {counts['lead']} + diff {counts['diff']} = **{f['n_features']:,} 피처**",
        f"- 행 {f['rows_used']:,} (유실 {f['join_rows_lost']:,}), "
        f"유효행 {f['eligible_rows']:,}, 적합 {fits} 회",
        "",
        "## 3. 시간 구조가 무언가 더하는가 (H1 · H2)",
        "",
        "| 모형 | 피처 | fold-외 R^2 | Pearson |",
        "|---|---:|---:|---:|",
        f"| 사이클 27 동시점 격자평균 | 65 | {CONTEMPORANEOUS_R2:+.4f} | +0.0699 |",
        f"| 사이클 28 동시점 격자별 | 795 | {PERGRID_R2:+.4f} | +0.0647 |",
        f"| **사이클 30 시간문맥 격자평균** | {f['n_features']:,} | **{r2:+.4f}** | "
        f"{corr:+.4f} |",
        "",
        "## 4. 점수 (H3)",
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
        "## 5. 피처 중요도 — 조건부 판정 (H4)",
        "",
        check["H4_note"],
        "",
        f"참고: 상위 20 중 시간변환 {temporal_in_top20} 개.",
        "",
        "| 순위 | 피처 | 이득 | 시간변환 |",
        "|---:|---|---:|:---:|",
    ]
    for rank, item in enumerate(top20, start=1):
        lines.append(
            f"| {rank} | `{item['feature']}` | {item['gain']:,.0f} | "
            f"{'O' if item['temporal'] else '-'} |"
        )

    lines += [
        "",
        "## 6. 사전확약 대조",
        "",
        f"- H1 `{check['H1_expectation']}` -> **{h1}** (실측 {r2:+.4f})",
        f"- H2 `{check['H2_expectation']}` -> **{h2}**",
        f"- H3 `{check['H3_expectation']}` -> **{h3}**",
        f"- H4 -> **{h4 if h4 is not None else '판정불가'}**",
        "",
        f"판정: **{verdict}**",
        "",
        f"승격 Total **{promoted_total:.6f}**, 목표 0.66 까지 **{0.66 - promoted_total:+.6f}**.",
        "",
    ]
    if not h1:
        lines += [
            "## 7. 이것이 확정하는 것",
            "",
            "공급 NWP 를 **공간(27·28)으로도 시간(30)으로도** 풀어봤고 세 해상도 모두",
            "fold-외 `R^2` 가 음수다. 챔피언의 잔차는 공급 NWP 로 설명되지 않는다.",
            "",
            "따라서 **같은 입력으로 새 기저모델을 만드는 경로가 닫힌다.** 새 모델이",
            "고칠 수 있는 오차가 있었다면 잔차-NWP 구조로 나타났어야 한다.",
            "",
            "닫히지 **않는** 것: 공급 밖 정보(외부 공개데이터 — 규칙상 허용), 그리고",
            "라벨 자체의 시계열 구조(단, 평가기간 라벨은 없다).",
        ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    receipt: dict[str, Any] = {
        "schema_version": 1,
        "stage": "M271_P4_CYCLE30_TEMPORAL",
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

    print(f"[C30] 피처 {f['n_features']:,} (동시점 {counts['contemporaneous']} + "
          f"lag {counts['lag']} + lead {counts['lead']} + diff {counts['diff']}) / "
          f"유효행 {f['eligible_rows']:,} / 적합 {fits} 회")
    print(f"[C30] 시간문맥 R^2 {r2:+.4f}  vs 동시점 {CONTEMPORANEOUS_R2:+.4f} / "
          f"격자별 {PERGRID_R2:+.4f}   Pearson {corr:+.4f}  -> H1 {h1} H2 {h2}")
    print(f"[C30] 보정 Total {corrected_score['total']:.6f} "
          f"(차이 {payload['scores']['delta_total']:+.6f}) -> H3 {h3}")
    print(f"[C30] H4 {h4 if h4 is not None else '판정불가 (H1 기각)'}  "
          f"(상위20 시간변환 {temporal_in_top20})")
    print(f"[C30] 판정: {verdict}  ->  Total {promoted_total:.6f} "
          f"(목표까지 {0.66 - promoted_total:+.6f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
