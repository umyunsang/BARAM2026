"""M271 P4 사이클 65 — 우리는 이미 풍속제약 한계에 붙어 있는가.

사이클 64 가 비켜감 가설을 기각했다. 예측은 중간대를 오히려 **더 자주** 쓰고(주변비
1.15~1.38) 오차 부호는 대칭이다. 결정층 인공물이 아니라 대칭 산포다.

그러면 남는 설명은 하나다. 사이클 57 의 천장은 **관측 풍속을 조건으로** 계산됐으므로
풍속 오차를 포함하지 않는다. 우리 예보에는 풍속 오차가 있고, 그것이 파워커브의
**국소 기울기**를 통해 출력 오차로 증폭된다.

    출력오차 = 기울기 * 풍속오차  (+) 내재 잔차
    sigma_total^2 ~ (dP/dv * sigma_v)^2 + sigma_resid^2

기울기는 C57 의 구간표에서 직접 나온다. sigma_v 는 C58 이 쟀다. sigma_resid 도 C57 에
있다. 즉 **적중률을 예측할 수 있고**, 관측과 대조하면 우리가 그 한계에 붙어 있는지
아니면 아직 여유가 있는지 갈린다.

이 대조는 축을 닫거나 열거나 한다.

    관측 ~= 예측  ->  **풍속제약 한계에 도달**. 결정층·피처 작업으로는 더 못 간다.
                     남는 길은 풍속 예보뿐이고 C52 가 가용 외부소스로 필요량의
                     30~45% 만 얻는다고 측정했다.
    관측 << 예측  ->  모형에 여유가 있다. 한계가 아니라 우리가 못하고 있는 것이다.

**① 방법 리서치**

  - 오차 전파를 국소 선형화로 다루는 것은 계측학의 표준(GUM, JCGM 100:2008 의
    불확도 전파). 여기서는 `u(P) = |dP/dv| * u(v)` 가 1 차 항이다.
  - 풍력 분야에서 이 전파는 파워커브 민감도로 잘 알려져 있다 — IEC 61400-12-1 이
    풍속 불확도를 출력 불확도로 옮길 때 같은 식을 쓴다. 급경사 구간에서 불확도가
    증폭되는 것이 표준 지식이다.
  - 정규 가정을 쓰지 않는다. C55 가 그 가정으로 틀렸고 C57 이 경험적 분포로
    교정했다. 잔차는 C57 의 **경험적 구간 잔차**에서 뽑고, 풍속 오차만 정규로 둔다
    (풍속 오차의 경험적 분포는 평가기간에 없으므로).
  - **채택**: 구간별 몬테카를로. `error = slope * eps_v + r`, `eps_v ~ N(0, sigma_v)`,
    `r ~ 경험적 잔차`. 적중률 = `P(|error| <= 0.06)`. 적합 없음.

**② 사양 동결**

  기울기   C57 receipt 의 구간표에서 `np.gradient(mean_power, bin_center)`.
           `rows >= 200` 구간만.
  잔차     C57 구간별 `sigma_resid` 를 척도로 한 경험적 형태. C57 이 경험적 분포를
           썼으나 receipt 에는 요약통계만 있으므로, **잔차는 정규로 두고 그 사실을
           한계로 명시한다.** 이 근사는 적중률을 **과대**평가하는 쪽이므로
           (경험적 분포가 더 두꺼운 꼬리를 가짐) 상한 성격을 강화한다.
  sigma_v  **1.40 m/s 로 동결.** C58 이 잰 test 행 teacher 산포 1.37~1.50 의 하단이다.
           낮게 잡을수록 예측 적중률이 높아져 "우리가 한계에 못 미친다"는 결론이
           나오기 쉬우므로, **내 가설에 불리한 쪽**으로 고른다.
           민감도로 1.2 / 1.6 도 같이 보고한다.
  관측     `m271_decision_surface` 캐시 + C60 GLOBAL fold-외 T. C64 와 동일 결정.
  대역     (0.10,0.25] (0.25,0.45] (0.45,0.70] (0.70,1.10]. 그룹x대역 12 셀.

  **타당성 가드**
    V1  예측 적중률이 12 셀 모두에서 C57 천장(관측풍속 조건)보다 **낮다**.
        풍속 오차를 더했는데 천장보다 높으면 계산이 틀린 것이다.

  사전확약 (V1 통과시에만 판정):
    H1  12 셀 중 **8 개 이상**에서 |관측 - 예측| <= 0.05. 전파식이 관측을 재현한다.
    H2  어긋나는 셀은 **g3 에 몰린다**. g3 는 theta 0.775 의 곱셈 잡음이 추가로
        있으므로(C57b) 이 1 차 모형이 과대예측해야 한다.
    H3  대역 순위가 일치한다 — Spearman(관측, 예측) >= 0.7.
    H4  g2 중간대에서 |관측 - 예측| <= 0.05. C63 이 지목한 최대 여유 셀이
        전파식으로 설명되는지가 라우팅의 핵심이다.

  H1·H4 가 참이면 **그 셀들에서 우리는 이미 풍속제약 한계에 있고**, C63 이 계산한
  0.18133 의 여유 중 상당 부분은 결정층·피처로 회수할 수 없다. 거짓이면 여유가
  실재하고 계속 판다.

**진단 전용.** 후보 아님. 점수를 주장하지 않는다. 게이트 미수정. 제출 없음.
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
from m271_cycle64_band_avoidance import EDGES, LABELS, band_of
from m271_decision_surface import load_surface

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
C57_RECEIPT = REPORTS / "m271_cycle57_ficr_ceiling_receipt.json"
C57B_RECEIPT = REPORTS / "m271_cycle57b_variance_law_receipt.json"
C60_RECEIPT = REPORTS / "m271_cycle60_level_temperature_receipt.json"
REPORT_MD = REPORTS / "m271_cycle65_wind_limited_bound.md"
RECEIPT = REPORTS / "m271_cycle65_wind_limited_bound_receipt.json"

NODE_ID = "C1N65_WIND_LIMITED_BOUND"
LANE = "L6"
PARENT_NODE = "C1N64_BAND_AVOIDANCE"

SIGMA_V = 1.40
SIGMA_V_SENSITIVITY = (1.20, 1.60)
MIN_ROWS = 200
BAND_HIT = 0.06
ELIGIBLE = 0.10
DRAWS = 400_000
SEED = 20260805
TOLERANCE = 0.05
MIN_MATCHES = 8


def band_index(power: float) -> int:
    for i in range(len(EDGES) - 1):
        if EDGES[i] < power <= EDGES[i + 1]:
            return i
    return -1


def predicted_hit(bins: list[dict], slopes: np.ndarray, sigma_v: float,
                  rng: np.random.Generator) -> dict[int, dict[str, float]]:
    """구간별로 error = slope*eps_v + r 을 뽑아 대역으로 묶는다. 발전량 가중."""
    per_band: dict[int, list[tuple[float, float]]] = {}
    for b, slope in zip(bins, slopes, strict=True):
        index = band_index(float(b["mean_power"]))
        if index < 0:
            continue
        eps = rng.normal(0.0, sigma_v, DRAWS)
        resid = rng.normal(0.0, float(b["sigma_resid"]), DRAWS)
        hit = float((np.abs(slope * eps + resid) <= BAND_HIT).mean())
        per_band.setdefault(index, []).append((hit, float(b["gen_weight"])))
    out: dict[int, dict[str, float]] = {}
    for index, items in per_band.items():
        h = np.array([x for x, _ in items])
        w = np.array([w for _, w in items])
        out[index] = {
            "predicted_hit": float(np.average(h, weights=w)),
            "gen_weight": float(w.sum()),
        }
    return out


def main() -> int:
    c57 = json.loads(C57_RECEIPT.read_text(encoding="utf-8"))["result"]
    c57b = json.loads(C57B_RECEIPT.read_text(encoding="utf-8"))
    c60 = json.loads(C60_RECEIPT.read_text(encoding="utf-8"))
    rng = np.random.default_rng(SEED)

    ceilings = {
        (int(g), b["band"]): float(b["unit_over_4"])
        for g, blk in c57b["per_group"].items()
        for b in blk["bands"]
    }
    label_of_c57b = {
        "(0.10,0.25]": 0, "(0.25,0.45]": 1, "(0.45,0.70]": 2, "(0.70,1.10]": 3,
    }

    predicted: dict[int, dict[int, dict[str, float]]] = {}
    sensitivity: dict[str, dict[str, float]] = {}
    slopes_by_band: dict[int, dict[int, float]] = {}
    for group in (1, 2, 3):
        bins = [b for b in c57["per_group"][str(group)]["bins"] if b["rows"] >= MIN_ROWS]
        v = np.array([b["bin_center"] for b in bins], dtype="float64")
        p = np.array([b["mean_power"] for b in bins], dtype="float64")
        slopes = np.gradient(p, v)
        predicted[group] = predicted_hit(bins, slopes, SIGMA_V, rng)
        slopes_by_band[group] = {}
        for index in range(len(LABELS)):
            picked = [
                (s, b["gen_weight"]) for s, b in zip(slopes, bins, strict=True)
                if band_index(float(b["mean_power"])) == index
            ]
            if picked:
                s = np.array([x for x, _ in picked])
                w = np.array([w for _, w in picked])
                slopes_by_band[group][index] = float(np.average(s, weights=w))
        for alt in SIGMA_V_SENSITIVITY:
            alt_pred = predicted_hit(bins, slopes, alt, rng)
            for index, value in alt_pred.items():
                sensitivity.setdefault(f"sigma_v={alt}", {})[
                    f"g{group}|{LABELS[index]}"
                ] = value["predicted_hit"]

    store, info = load_surface()
    assert info["probability_digest"] == c60["probability_digest"], "확률면 불일치"
    parts: list[pd.DataFrame] = []
    for fold, entry in store.items():
        temperature = np.full(
            len(entry["capacity"]), float(c60["chosen"]["global"][fold])
        )
        prediction = bayes_decision(sharpen_by_row(entry["probability"], temperature))
        frame = entry["meta"].copy()
        frame["group_id"] = entry["group"]
        frame["actual_rate"] = frame["actual_kwh"] / entry["capacity"]
        frame["pred_rate"] = prediction
        parts.append(frame)
    data = pd.concat(parts, ignore_index=True)
    data = data.loc[data["actual_rate"] >= ELIGIBLE].reset_index(drop=True)
    data["band"] = band_of(data["actual_rate"].to_numpy(float))
    data = data.loc[data["band"] >= 0].reset_index(drop=True)
    data["hit"] = (data["pred_rate"] - data["actual_rate"]).abs() <= BAND_HIT

    rows: list[dict[str, Any]] = []
    for group in (1, 2, 3):
        for index, label in enumerate(LABELS):
            block = data.loc[(data["group_id"] == group) & (data["band"] == index)]
            pred = predicted[group].get(index)
            if pred is None or not len(block):
                continue
            observed = float(block["hit"].mean())
            rows.append({
                "group_id": group,
                "band": label,
                "rows": int(len(block)),
                "slope": slopes_by_band[group].get(index, float("nan")),
                "ceiling": ceilings[(group, list(label_of_c57b)[index])],
                "predicted_hit": pred["predicted_hit"],
                "observed_hit": observed,
                "residual": observed - pred["predicted_hit"],
            })
    frame = pd.DataFrame(rows)

    v1 = bool((frame["predicted_hit"] < frame["ceiling"]).all())

    frame["match"] = frame["residual"].abs() <= TOLERANCE
    matches = int(frame["match"].sum())
    h1 = bool(matches >= MIN_MATCHES)

    mismatched = frame.loc[~frame["match"]]
    h2 = bool(len(mismatched) == 0 or (mismatched["group_id"] == 3).mean() >= 0.5)

    rx = frame["observed_hit"].rank().to_numpy()
    ry = frame["predicted_hit"].rank().to_numpy()
    rx, ry = rx - rx.mean(), ry - ry.mean()
    rho = float((rx * ry).sum() / np.sqrt((rx * rx).sum() * (ry * ry).sum()))
    h3 = bool(rho >= 0.7)

    g2_mid = frame.loc[(frame["group_id"] == 2) & (frame["band"] == "(0.45,0.70]")]
    g2_mid_residual = float(g2_mid["residual"].iloc[0]) if len(g2_mid) else float("nan")
    h4 = bool(abs(g2_mid_residual) <= TOLERANCE)

    if not v1:
        verdict = "PROPAGATION_EXCEEDS_CEILING_CALCULATION_WRONG"
    elif h1 and h4:
        verdict = "AT_WIND_LIMITED_BOUND_DECISION_AND_FEATURE_AXES_EXHAUSTED"
    elif h1:
        verdict = "BOUND_REPRODUCED_OVERALL_BUT_NOT_AT_ROUTED_CELL"
    else:
        verdict = "SLACK_REMAINS_BELOW_WIND_LIMITED_BOUND"

    payload: dict[str, Any] = {
        "node_id": NODE_ID,
        "lane": LANE,
        "parent": PARENT_NODE,
        "candidate": False,
        "diagnostic_only": True,
        "method": "LOCAL_LINEAR_ERROR_PROPAGATION (JCGM 100:2008; IEC 61400-12-1)",
        "sigma_v": SIGMA_V,
        "sigma_v_sensitivity": sensitivity,
        "draws": DRAWS,
        "seed": SEED,
        "surface": info,
        "cells": frame.to_dict(orient="records"),
        "checks": {"V1_predicted_below_ceiling": v1},
        "matches": matches,
        "spearman": rho,
        "g2_mid_residual": g2_mid_residual,
        "hypotheses": {
            "H1_reproduces_at_least_8_of_12": h1,
            "H2_mismatch_concentrated_in_g3": h2,
            "H3_rank_agreement": h3,
            "H4_g2_mid_explained": h4,
        },
        "limitation": (
            "잔차를 정규로 근사했다. C57 의 경험적 분포는 꼬리가 더 두꺼우므로 "
            "예측 적중률은 **과대**평가 쪽이고, 따라서 '한계에 붙어 있다'는 결론에 "
            "**보수적**이다."
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
        "# M271 P4 사이클 65 — 풍속제약 한계",
        "",
        f"노드 `{NODE_ID}` / 레인 {LANE} / 부모 `{PARENT_NODE}` / **진단 전용**",
        "",
        f"`error = dP/dv * eps_v + r`, `eps_v ~ N(0, {SIGMA_V})`, 추첨 {DRAWS:,}, "
        f"시드 {SEED}",
        "",
        "## 1. 셀별",
        "",
        "| 그룹 | 대역 | 행 | 기울기 | 천장 | 예측 적중 | 관측 적중 | 차 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in frame.itertuples():
        mark = "" if row.match else " **X**"
        lines.append(
            f"| {row.group_id} | {row.band} | {row.rows} | {row.slope:.4f} | "
            f"{row.ceiling:.3f} | {row.predicted_hit:.3f} | {row.observed_hit:.3f} | "
            f"{row.residual:+.3f}{mark} |"
        )
    lines += [
        "",
        "## 2. 사전확약",
        "",
        f"- V1 예측 적중이 12 셀 모두 천장 미만 -> **{v1}**",
        f"- H1 {matches}/12 셀이 ±{TOLERANCE} 이내 (>= {MIN_MATCHES}) -> **{h1}**",
        f"- H2 어긋남이 g3 에 몰린다 -> **{h2}**",
        f"- H3 순위일치 Spearman {rho:+.3f} >= 0.7 -> **{h3}**",
        f"- H4 g2 중간대 차 {g2_mid_residual:+.3f} -> **{h4}**",
        "",
        "## 3. sigma_v 민감도",
        "",
        "| 셀 | " + " | ".join(f"sv={s}" for s in SIGMA_V_SENSITIVITY) + f" | sv={SIGMA_V} |",
        "|---|" + "---:|" * (len(SIGMA_V_SENSITIVITY) + 1),
    ]
    for row in frame.itertuples():
        key = f"g{row.group_id}|{row.band}"
        alts = " | ".join(
            f"{sensitivity[f'sigma_v={s}'].get(key, float('nan')):.3f}"
            for s in SIGMA_V_SENSITIVITY
        )
        lines.append(f"| {key} | {alts} | {row.predicted_hit:.3f} |")
    lines += [
        "",
        "## 4. 판정",
        "",
        f"**{verdict}**",
        "",
        payload["limitation"],
        "",
        f"`sigma_v` 를 1.40 으로 동결한 것은 C58 이 잰 test 행 산포 1.37~1.50 의 "
        "**하단**이다. 낮게 잡을수록 예측 적중률이 올라 '한계 미달' 결론이 나오기 쉬우므로 "
        "내 가설에 불리한 쪽을 골랐다.",
        "",
        f"digest `{payload['digest']}`",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("=== 완료 ===")
    for row in frame.itertuples():
        mark = "" if row.match else "  <-- 어긋남"
        print(f"[C65] g{row.group_id} {row.band}  기울기 {row.slope:.4f}  "
              f"예측 {row.predicted_hit:.3f}  관측 {row.observed_hit:.3f}  "
              f"차 {row.residual:+.3f}{mark}")
    print(f"[C65] V1 {v1} / H1 {matches}/12 {h1} / H2 {h2} / "
          f"H3 rho {rho:+.3f} {h3} / H4 g2중간 {g2_mid_residual:+.3f} {h4}")
    print(f"[C65] 판정: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
