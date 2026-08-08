"""M271 P4 사이클 68 — 전파 모형을 버리고 같은 행에서 직접 분해한다.

사이클 65·66·67 이 `error = 파워커브(풍속오차) + 잔차` 로 적중률을 예측했고 세 번 다
저출력대에서 관측이 예측보다 +0.09~+0.12 좋았다. 등분산을 고쳐도(66), 선형화를
버려도(67) 그대로였다.

세 번 기각하고 나서야 결함이 보인다. **전파 모형이 가정하는 예측기가 우리 예측기가
아니다.** 그 모형은 "풍속 추정에 파워커브를 적용하는 기계" 다. 우리 것은 피처 100 개짜리
GBM 이고, 풍속 밖의 피처가 정보를 나르는 구간에서는 그 기계보다 잘한다. 그러면 관측이
예측보다 좋게 나온다 — 정확히 관측된 부호다.

거기에 조건화 불일치도 겹쳐 있었다. 전파는 `P(적중 | 풍속 v)` 를 재고 `E[P|v]` 로 대역을
배정했는데, 관측은 `P(적중 | **실제 출력** in B)` 다.

모형을 더 고치는 대신 없앤다. 캐시 v2 에 teacher 풍속과 관측 나셀풍속이 둘 다 있으므로
**같은 행에서 세 팔을 만들어 같은 조건으로** 비교하면 된다. 가우스 가정도, 선형화도,
조건화 불일치도 사라진다.

    MODEL          우리 Bayes 결정                     (실제 예측)
    CURVE_TEACHER  실측 커브를 **teacher 풍속**에 적용   (풍속-only 기계)
    CURVE_ORACLE   실측 커브를 **관측 나셀풍속**에 적용   (풍속을 완벽히 알 때)

분해가 그대로 나온다.

    CURVE_ORACLE - CURVE_TEACHER  =  풍속 오차가 물리는 비용
    MODEL - CURVE_TEACHER         =  우리 모형이 풍속-only 기계에 더하는 것
    CURVE_ORACLE - MODEL          =  풍속을 완벽히 알면 더 얻을 것 (**남은 여지**)

**① 방법 리서치**

  - 이건 예보 검증의 **참조예보 분해**다. 각 팔이 하나의 참조예보이고, 같은 표본에서
    같은 점수를 매겨 차이를 본다(Murphy 1988 의 스킬스코어 틀; Jolliffe & Stephenson
    2012 의 참조예보 선택 논의).
  - `CURVE_ORACLE` 은 **완전예보 참조**다. C59 가 같은 역할을 학습형으로 했으나
    나셀풍속을 **피처로 넣어** 오염됐다(1-NMAE 0.857 -> 0.946). 여기서는 학습이 없고
    **실측 커브 한 번 통과**뿐이라 그 경로가 막힌다 — 커브는 풍속만 받는다.
  - 나셀 풍속계가 로터 뒤라는 한계는 남는다(A5; IEC 61400-12-1). 따라서
    `CURVE_ORACLE` 은 여전히 **위쪽으로 편향된 상한**이고, 그렇게만 읽는다.
  - **채택**: 같은 행 3 팔 분해. 적합 없음. 분포 가정 없음.

**② 사양 동결**

  입력   확률면 캐시 v2. `scada_ws` 와 `sitewind` 가 **모두 있는 행만** 세 팔 공통으로
         쓴다(팔마다 다른 모집단을 쓰면 C22 결함 재발).
  커브   C57 실측 커브 + C67 이 동결한 cut-in `(0,0),(3,0)`. 그룹별. 위쪽 평탄.
  결정   MODEL 은 C60 GLOBAL fold-외 T. C64~C67 과 동일.
  대역   실제 출력 대역 4 개 x 그룹 3. 유효행(실제 >= 10% 용량)만.
  지표   대역별 적중률 `P(|예측 - 실제| <= 0.06)`.

  **타당성 가드**
    V1  `scada_ws` 보유 비율 >= 0.80. 낮으면 표본이 편향됐을 수 있다.
    V2  전 행 MODEL 적중률이 캐시 전체(제한 없는) 값의 ±0.02 이내. `scada_ws` 보유
        행으로 제한한 것이 MODEL 을 유리/불리하게 만들지 않았음을 확인한다.

  사전확약 (V1·V2 통과시에만 판정):
    H1  `CURVE_ORACLE > MODEL` 이 12 셀 **전부**에서 성립. 상한이 상한 노릇을 해야 한다.
        깨지면 커브 참조가 상한이 아니고 해석이 무너진다.
    H2  `MODEL > CURVE_TEACHER` 가 전체에서 성립. 우리 모형이 풍속-only 기계보다 낫다.
    H3  `MODEL - CURVE_TEACHER` 가 **저출력대에서 최대**다.  **핵심.**
        C65~C67 이 저출력대에서만 계속 과소예측한 이유가 이것이라면, 그 잉여가
        저출력대에 몰려 있어야 한다.
    H4  g2 중간대에서 `CURVE_ORACLE - MODEL` 이 12 셀 중 **하위 3 분위**.
        C63 이 라우팅한 셀이고 C65~C67 이 "이미 한계" 라 했으니 남은 여지가 작아야 한다.

  H3 이 참이면 C65~C67 의 결손이 설명되고 그 축이 닫힌다. H1 이 거짓이면 커브 참조
  자체가 무효라 이 노드의 나머지를 버린다.

  **부호를 예단하지 않는 곳**: `CURVE_ORACLE - MODEL` 의 절대 크기. 그것이 이 대회에서
  남은 여지의 **경험적** 상한이고, 크든 작든 그대로 보고한다.

**진단 전용.** 후보 아님. `scada_ws` 는 평가기간 부재라 피처가 될 수 없다(C1N39).
게이트 미수정. 제출 없음.
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

from m271_cycle40_band_classifier import bayes_decision
from m271_cycle60_level_temperature import sharpen_by_row
from m271_cycle64_band_avoidance import LABELS, band_of
from m271_cycle65_wind_limited_bound import BAND_HIT, ELIGIBLE, MIN_ROWS
from m271_cycle67_exact_curve_propagation import build_curve
from m271_decision_surface import load_surface

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
C57_RECEIPT = REPORTS / "m271_cycle57_ficr_ceiling_receipt.json"
C60_RECEIPT = REPORTS / "m271_cycle60_level_temperature_receipt.json"
REPORT_MD = REPORTS / "m271_cycle68_empirical_decomposition.md"
RECEIPT = REPORTS / "m271_cycle68_empirical_decomposition_receipt.json"

NODE_ID = "C1N68_EMPIRICAL_DECOMPOSITION"
LANE = "L6"
PARENT_NODE = "C1N67_EXACT_CURVE_PROPAGATION"

COVERAGE_FLOOR = 0.80
SELECTION_TOLERANCE = 0.02
ARMS = ("model", "curve_teacher", "curve_oracle")


def main() -> int:
    store, info = load_surface()
    c57 = json.loads(C57_RECEIPT.read_text(encoding="utf-8"))["result"]
    c60 = json.loads(C60_RECEIPT.read_text(encoding="utf-8"))
    assert info["probability_digest"] == c60["probability_digest"], "확률면 불일치"

    curves = {}
    for group in (1, 2, 3):
        bins = [b for b in c57["per_group"][str(group)]["bins"] if b["rows"] >= MIN_ROWS]
        curves[group] = build_curve(bins)

    parts: list[pd.DataFrame] = []
    for fold, entry in store.items():
        temperature = np.full(
            len(entry["capacity"]), float(c60["chosen"]["global"][fold])
        )
        frame = entry["meta"].copy()
        frame["group_id"] = entry["group"]
        frame["actual_rate"] = frame["actual_kwh"] / entry["capacity"]
        frame["model"] = bayes_decision(
            sharpen_by_row(entry["probability"], temperature)
        )
        frame["sitewind"] = entry["sitewind"]
        frame["scada_ws"] = entry["scada_ws"]
        parts.append(frame)
    data = pd.concat(parts, ignore_index=True)
    data = data.loc[data["actual_rate"] >= ELIGIBLE].reset_index(drop=True)
    data["band"] = band_of(data["actual_rate"].to_numpy(float))
    data = data.loc[data["band"] >= 0].reset_index(drop=True)

    unrestricted_hit = float(
        ((data["model"] - data["actual_rate"]).abs() <= BAND_HIT).mean()
    )
    coverage = float((data["sitewind"].notna() & data["scada_ws"].notna()).mean())
    v1 = bool(coverage >= COVERAGE_FLOOR)

    both = data.loc[data["sitewind"].notna() & data["scada_ws"].notna()].copy()
    for group, (cv, cp) in curves.items():
        mask = both["group_id"] == group
        both.loc[mask, "curve_teacher"] = np.interp(
            both.loc[mask, "sitewind"], cv, cp, left=0.0, right=cp[-1]
        )
        both.loc[mask, "curve_oracle"] = np.interp(
            both.loc[mask, "scada_ws"], cv, cp, left=0.0, right=cp[-1]
        )
    for arm in ARMS:
        both[f"hit_{arm}"] = (both[arm] - both["actual_rate"]).abs() <= BAND_HIT

    restricted_hit = float(both["hit_model"].mean())
    v2 = bool(abs(restricted_hit - unrestricted_hit) <= SELECTION_TOLERANCE)

    rows: list[dict[str, Any]] = []
    for group in (1, 2, 3):
        for index, label in enumerate(LABELS):
            block = both.loc[(both["group_id"] == group) & (both["band"] == index)]
            if not len(block):
                continue
            hits = {arm: float(block[f"hit_{arm}"].mean()) for arm in ARMS}
            rows.append({
                "group_id": group,
                "band": label,
                "band_index": index,
                "rows": int(len(block)),
                **hits,
                "model_minus_teacher": hits["model"] - hits["curve_teacher"],
                "oracle_minus_model": hits["curve_oracle"] - hits["model"],
                "oracle_minus_teacher": hits["curve_oracle"] - hits["curve_teacher"],
            })
    frame = pd.DataFrame(rows)

    h1 = bool((frame["oracle_minus_model"] > 0).all())
    h2 = bool(float(both["hit_model"].mean()) > float(both["hit_curve_teacher"].mean()))

    by_band = frame.groupby("band_index")["model_minus_teacher"].mean()
    h3 = bool(int(by_band.idxmax()) == 0)

    ranked = frame.sort_values("oracle_minus_model").reset_index(drop=True)
    g2_mid_rank = int(
        ranked.index[
            (ranked["group_id"] == 2) & (ranked["band_index"] == 2)
        ][0]
    )
    h4 = bool(g2_mid_rank < len(ranked) / 3.0)

    overall = {
        arm: float(both[f"hit_{arm}"].mean()) for arm in ARMS
    }

    if not v1:
        verdict = "SCADA_COVERAGE_TOO_LOW_RESULT_VOID"
    elif not v2:
        verdict = "RESTRICTION_BIASES_MODEL_RESULT_VOID"
    elif not h1:
        verdict = "CURVE_ORACLE_IS_NOT_AN_UPPER_BOUND"
    elif h3:
        verdict = "MODEL_SURPLUS_OVER_WIND_ONLY_CONCENTRATES_IN_LOW_BAND"
    else:
        verdict = "MODEL_SURPLUS_NOT_LOW_BAND_CONCENTRATED"

    payload: dict[str, Any] = {
        "node_id": NODE_ID,
        "lane": LANE,
        "parent": PARENT_NODE,
        "candidate": False,
        "diagnostic_only": True,
        "method": "REFERENCE_FORECAST_DECOMPOSITION (Murphy 1988; Jolliffe & Stephenson 2012)",
        "surface": info,
        "rows": int(len(both)),
        "coverage": coverage,
        "overall_hit": overall,
        "cells": frame.to_dict(orient="records"),
        "band_mean_model_minus_teacher": {int(k): float(v) for k, v in by_band.items()},
        "checks": {
            "V1_coverage": coverage, "V1_pass": v1,
            "V2_unrestricted_hit": unrestricted_hit,
            "V2_restricted_hit": restricted_hit, "V2_pass": v2,
        },
        "g2_mid_oracle_gap_rank": g2_mid_rank,
        "hypotheses": {
            "H1_oracle_bounds_model_everywhere": h1,
            "H2_model_beats_curve_teacher": h2,
            "H3_surplus_max_in_low_band": h3,
            "H4_g2_mid_little_room_left": h4,
        },
        "limitation": (
            "나셀 풍속계는 로터 뒤라 자유유입풍속이 아니다(A5; IEC 61400-12-1). "
            "`CURVE_ORACLE` 은 위쪽으로 편향된 상한이며 그렇게만 읽는다. 다만 C59 와 "
            "달리 학습이 없고 커브 한 번 통과이므로 운전상태가 **피처로** 새는 경로는 없다."
        ),
        "verdict": verdict,
        "no_training": True,
        "dacon_upload": False,
        "lockbox_used": False,
        "generated_at": datetime.now(UTC).isoformat(),
    }
    payload["digest"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]
    RECEIPT.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    lines = [
        "# M271 P4 사이클 68 — 같은 행 3 팔 경험적 분해",
        "",
        f"노드 `{NODE_ID}` / 레인 {LANE} / 부모 `{PARENT_NODE}` / **진단 전용**",
        "",
        f"확률면 prob `{info['probability_digest']}` / 대조행 {len(both):,} "
        f"(보유율 {coverage:.3f})",
        "",
        "적합 없음. 분포 가정 없음. 세 팔이 **같은 행**을 쓴다.",
        "",
        "## 1. 전체 적중률",
        "",
        "| 팔 | 적중률 |",
        "|---|---:|",
    ]
    for arm in ARMS:
        lines.append(f"| {arm} | {overall[arm]:.4f} |")
    lines += [
        "",
        "## 2. 셀별",
        "",
        "| 그룹 | 대역 | 행 | MODEL | CURVE_TEACHER | CURVE_ORACLE | "
        "모형잉여 | 남은여지 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in frame.itertuples():
        lines.append(
            f"| {row.group_id} | {row.band} | {row.rows} | {row.model:.3f} | "
            f"{row.curve_teacher:.3f} | {row.curve_oracle:.3f} | "
            f"{row.model_minus_teacher:+.3f} | {row.oracle_minus_model:+.3f} |"
        )
    lines += [
        "",
        "**모형잉여** = MODEL - CURVE_TEACHER (풍속-only 기계 대비 우리 모형의 더함)",
        "",
        "**남은여지** = CURVE_ORACLE - MODEL (풍속을 완벽히 알면 더 얻을 것, 상한)",
        "",
        "## 3. 대역별 모형잉여 평균",
        "",
        "| 대역 | 모형잉여 |",
        "|---|---:|",
    ]
    for index, value in by_band.items():
        lines.append(f"| {LABELS[int(index)]} | {value:+.4f} |")
    lines += [
        "",
        "## 4. 사전확약",
        "",
        f"- V1 보유율 {coverage:.3f} >= {COVERAGE_FLOOR} -> **{v1}**",
        f"- V2 제한 {restricted_hit:.4f} vs 전체 {unrestricted_hit:.4f} -> **{v2}**",
        f"- H1 CURVE_ORACLE 이 12 셀 전부에서 MODEL 상한 -> **{h1}**",
        f"- H2 MODEL > CURVE_TEACHER -> **{h2}** "
        f"({overall['model']:.4f} vs {overall['curve_teacher']:.4f})",
        f"- H3 모형잉여가 저출력대에서 최대 -> **{h3}**",
        f"- H4 g2 중간대 남은여지가 하위 3 분위 (순위 {g2_mid_rank + 1}/12) -> **{h4}**",
        "",
        "## 5. 판정",
        "",
        f"**{verdict}**",
        "",
        payload["limitation"],
        "",
        f"digest `{payload['digest']}`",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("=== 완료 ===")
    print(f"[C68] 보유율 {coverage:.3f} (V1 {v1}) / 선택편향 "
          f"{restricted_hit:.4f} vs {unrestricted_hit:.4f} (V2 {v2})")
    print(f"[C68] 전체  MODEL {overall['model']:.4f} / CURVE_TEACHER "
          f"{overall['curve_teacher']:.4f} / CURVE_ORACLE {overall['curve_oracle']:.4f}")
    for row in frame.itertuples():
        print(f"[C68] g{row.group_id} {row.band}  M {row.model:.3f} / T "
              f"{row.curve_teacher:.3f} / O {row.curve_oracle:.3f}  "
              f"잉여 {row.model_minus_teacher:+.3f}  여지 {row.oracle_minus_model:+.3f}")
    print(f"[C68] 대역별 모형잉여 " + " / ".join(
        f"{LABELS[int(k)]} {v:+.3f}" for k, v in by_band.items()))
    print(f"[C68] H1 {h1} / H2 {h2} / H3 {h3} / H4 {h4} (g2중간 순위 {g2_mid_rank+1}/12)")
    print(f"[C68] 판정: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
