"""M271 P4 사이클 66 — 풍속 오차는 등분산이 아니다: sigma_v(v) 를 직접 재서 전파를 다시 건다.

사이클 65 는 `error = dP/dv * eps_v + r` 로 적중률을 예측했고 12 셀 중 3 개만 맞혔다.
그런데 **어긋남이 잡음이 아니었다.**

    대역          g1       g2       g3
    (0.10,0.25]  +0.119   +0.113   +0.093    관측이 예측보다 **좋다**
    (0.25,0.45]  -0.001   -0.050   -0.025
    (0.45,0.70]  +0.073   -0.028   -0.053
    (0.70,1.10]  -0.134   -0.112   -0.093    관측이 예측보다 **나쁘다**

세 그룹 전부에서 대역에 대해 단조다. 용의자가 하나뿐이다 — `sigma_v = 1.40` 상수.
NWP 풍속 오차는 상대오차가 대략 일정해 **절대오차가 풍속을 따라 커진다**. 저풍속
구간에서는 1.40 이 과대라 우리가 예측보다 잘하고, 고풍속 구간에서는 과소라 못한다.
가정 하나가 양끝을 정확히 반대 방향으로 틀리게 만든다.

**관측 적중률에서 sigma_v 를 역산하면 순환논법**이다. 그래서 독립 계측한다 — teacher 가
편성한 풍속(`sitewind__mean`)과 관측 나셀풍속(`scada_ws`)의 fold-외 잔차를, **예측 풍속
구간별로** 나눠 본다. 이를 위해 확률면 캐시를 v2 로 넓혔다.

**① 방법 리서치**

  - 예보 오차를 예보값으로 층화해 보는 것은 Murphy & Winkler(1987) 의 분포지향
    검증틀 그대로다. 조건부 검증(conditional verification)이라 부른다.
  - 풍속 예보 오차의 이분산성과 그 모수화(비례형 `sigma ~ b*v`, 또는 spread-skill
    관계)는 확립돼 있다. 이 프로젝트는 Whitaker & Loughe(1998) 의 spread-skill 을
    이미 인용했다.
  - **주의**: 나셀 풍속계는 로터 뒤에 있어 자유유입풍속이 아니다(A5, IEC 61400-12-1).
    다만 teacher 는 **`scada_ws` 를 표적으로 학습**했으므로, 그 잔차는 우리가 실제로
    쓰는 사상의 오차를 재는 올바른 양이다. 자유유입풍속의 오차가 아니라는 점은
    한계로 남긴다.
  - **채택**: 예측풍속 구간별 잔차 표준편차 -> `sigma_v(v)` -> C65 몬테카를로 재실행.

**② 사양 동결**

  입력   확률면 캐시 v2 (`sitewind`, `scada_ws`). 두 값이 모두 있는 행만.
  구간   `sitewind` 를 0.5 m/s 폭으로. `rows >= 200` 구간만 사용(C57 과 같은 문턱).
  추정   구간별 `std(scada_ws - sitewind)`. 그룹별로 따로.
  적합   `sigma_v = a + b*v` 를 행수 가중 최소제곱으로. 상대형 `sigma_v/v` 도 보고.
  전파   C65 와 **완전히 동일한 몬테카를로**. 바뀌는 것은 각 C57 구간에서 쓰는
         `sigma_v` 뿐이며, 그 구간의 `bin_center` 에서 적합선을 평가한다.
         추첨 400,000 / 시드 20260805 — C65 와 동일.

  **타당성 가드**
    V1  전체 잔차 표준편차가 **1.2 ~ 1.7 m/s**. C58 이 잰 test 행 산포 1.37~1.50 과
        정합해야 한다. 벗어나면 계측이 틀린 것이고 나머지 판정을 버린다.
    V2  C65 의 상수 sigma_v 결과가 이 하네스에서 재현된다 — 상수 1.40 으로 돌렸을 때
        12 셀 적중률이 C65 값과 ±0.005 이내. 전파 코드가 바뀌지 않았음을 못박는다.

  사전확약 (V1·V2 통과시에만 판정):
    H1  `sigma_v` 가 풍속에 따라 **증가**한다 — 적합 기울기 b > 0 이고 구간별
        Spearman(v, sigma_v) >= 0.7.  주 가설의 전제.
    H2  이분산 전파에서 **8/12 이상**이 ±0.05 이내. C65 는 3/12 였다.  **핵심 판정.**
    H3  잔차의 대역 단조성이 사라진다 — |Spearman(대역 인덱스, 잔차)| < 0.5.
        C65 에서는 강한 음수였다.
    H4  g2 중간대가 여전히 설명된다 (|잔차| <= 0.05). C63 이 라우팅한 셀이다.

  H2 가 참이면 회계가 닫힌다: **관측 스킬 = 풍속오차를 파워커브로 전파한 것 + 내재
  산포.** 그러면 남는 지렛대는 `sigma_v` 하나뿐이고, C52 가 가용 외부소스로 필요량의
  30~45% 만 얻는다고 이미 측정했다. H2 가 거짓이면 풍속오차 밖에 구조가 더 있고
  계속 판다.

  **부호를 예단하지 않는 곳 없음** — H1 은 물리에서, H2·H3 는 H1 이 참일 때의 산술적
  귀결에서 나온다. 그래서 H1 이 참인데 H2 가 거짓이면 그것이 가장 유익한 결과다.

**진단 전용.** 후보 아님. `scada_ws` 는 평가기간 부재이므로 피처가 될 수 없다(C1N39).
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
from m271_cycle65_wind_limited_bound import (
    BAND_HIT,
    DRAWS,
    ELIGIBLE,
    MIN_ROWS,
    SEED,
    SIGMA_V,
    TOLERANCE,
    band_index,
)
from m271_decision_surface import load_surface

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
C57_RECEIPT = REPORTS / "m271_cycle57_ficr_ceiling_receipt.json"
C60_RECEIPT = REPORTS / "m271_cycle60_level_temperature_receipt.json"
C65_RECEIPT = REPORTS / "m271_cycle65_wind_limited_bound_receipt.json"
REPORT_MD = REPORTS / "m271_cycle66_heteroscedastic_wind.md"
RECEIPT = REPORTS / "m271_cycle66_heteroscedastic_wind_receipt.json"

NODE_ID = "C1N66_HETEROSCEDASTIC_WIND"
LANE = "L6"
PARENT_NODE = "C1N65_WIND_LIMITED_BOUND"

WIND_BIN = 0.5
SIGMA_RANGE = (1.2, 1.7)
MIN_MATCHES = 8
REPRO_TOLERANCE = 0.005
MONOTONE_FLOOR = 0.7
FLATNESS_CEILING = 0.5


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    rx, ry = rx - rx.mean(), ry - ry.mean()
    denom = float(np.sqrt((rx * rx).sum() * (ry * ry).sum()))
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


def propagate(bins: list[dict], slopes: np.ndarray, sigma_at: Any,
              rng: np.random.Generator) -> dict[int, float]:
    """C65 와 동일한 전파. `sigma_at` 만 상수에서 함수로 일반화했다."""
    per_band: dict[int, list[tuple[float, float]]] = {}
    for b, slope in zip(bins, slopes, strict=True):
        index = band_index(float(b["mean_power"]))
        if index < 0:
            continue
        sigma = sigma_at(float(b["bin_center"])) if callable(sigma_at) else float(sigma_at)
        eps = rng.normal(0.0, max(sigma, 1e-6), DRAWS)
        resid = rng.normal(0.0, float(b["sigma_resid"]), DRAWS)
        hit = float((np.abs(slope * eps + resid) <= BAND_HIT).mean())
        per_band.setdefault(index, []).append((hit, float(b["gen_weight"])))
    return {
        index: float(np.average([h for h, _ in items],
                                weights=[w for _, w in items]))
        for index, items in per_band.items()
    }


def main() -> int:
    store, info = load_surface()
    c57 = json.loads(C57_RECEIPT.read_text(encoding="utf-8"))["result"]
    c60 = json.loads(C60_RECEIPT.read_text(encoding="utf-8"))
    c65 = json.loads(C65_RECEIPT.read_text(encoding="utf-8"))
    assert info["probability_digest"] == c60["probability_digest"], (
        f"확률면 불일치: {info['probability_digest']} vs {c60['probability_digest']}"
    )

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
        frame["sitewind"] = entry["sitewind"]
        frame["scada_ws"] = entry["scada_ws"]
        parts.append(frame)
    data = pd.concat(parts, ignore_index=True)

    # --- sigma_v(v) 독립 계측 -------------------------------------------------
    wind = data.loc[data["sitewind"].notna() & data["scada_ws"].notna()].copy()
    wind["wind_err"] = wind["scada_ws"] - wind["sitewind"]
    overall_sigma = float(wind["wind_err"].std(ddof=1))
    v1 = bool(SIGMA_RANGE[0] <= overall_sigma <= SIGMA_RANGE[1])

    wind["wind_bin"] = (wind["sitewind"] / WIND_BIN).round() * WIND_BIN
    sigma_rows: list[dict[str, Any]] = []
    fits: dict[int, tuple[float, float]] = {}
    monotone: dict[int, float] = {}
    for group in (1, 2, 3):
        block = wind.loc[wind["group_id"] == group]
        grouped = block.groupby("wind_bin")["wind_err"].agg(["std", "size", "mean"])
        grouped = grouped.loc[grouped["size"] >= MIN_ROWS]
        v = grouped.index.to_numpy(dtype="float64")
        s = grouped["std"].to_numpy(dtype="float64")
        w = grouped["size"].to_numpy(dtype="float64")
        b, a = np.polyfit(v, s, 1, w=np.sqrt(w))
        fits[group] = (float(a), float(b))
        monotone[group] = spearman(v, s)
        for vi, si, wi, mi in zip(v, s, w, grouped["mean"].to_numpy(), strict=True):
            sigma_rows.append({
                "group_id": group, "wind_bin": float(vi), "sigma_v": float(si),
                "rows": int(wi), "bias": float(mi), "relative": float(si / vi),
            })

    h1 = bool(
        all(fits[g][1] > 0 for g in fits)
        and all(monotone[g] >= MONOTONE_FLOOR for g in monotone)
    )

    # --- 전파 재실행 ----------------------------------------------------------
    data = data.loc[data["actual_rate"] >= ELIGIBLE].reset_index(drop=True)
    data["band"] = band_of(data["actual_rate"].to_numpy(float))
    data = data.loc[data["band"] >= 0].reset_index(drop=True)
    data["hit"] = (data["pred_rate"] - data["actual_rate"]).abs() <= BAND_HIT

    observed = {
        (int(g), int(b)): float(blk["hit"].mean())
        for (g, b), blk in data.groupby(["group_id", "band"])
    }

    rows: list[dict[str, Any]] = []
    repro_gap = 0.0
    c65_lookup = {
        (int(c["group_id"]), LABELS.index(c["band"])): float(c["predicted_hit"])
        for c in c65["cells"]
    }
    for group in (1, 2, 3):
        bins = [b for b in c57["per_group"][str(group)]["bins"] if b["rows"] >= MIN_ROWS]
        v = np.array([b["bin_center"] for b in bins], dtype="float64")
        p = np.array([b["mean_power"] for b in bins], dtype="float64")
        slopes = np.gradient(p, v)
        a, b_slope = fits[group]

        constant = propagate(bins, slopes, SIGMA_V, np.random.default_rng(SEED))
        hetero = propagate(
            bins, slopes, lambda x: a + b_slope * x, np.random.default_rng(SEED)
        )
        for index, label in enumerate(LABELS):
            if index not in hetero or (group, index) not in observed:
                continue
            repro_gap = max(
                repro_gap,
                abs(constant[index] - c65_lookup.get((group, index), constant[index])),
            )
            rows.append({
                "group_id": group,
                "band": label,
                "band_index": index,
                "sigma_v_used": float(
                    a + b_slope * float(np.average(
                        [x["bin_center"] for x in bins
                         if band_index(float(x["mean_power"])) == index],
                        weights=[x["gen_weight"] for x in bins
                                 if band_index(float(x["mean_power"])) == index],
                    ))
                ),
                "constant_predicted": constant[index],
                "hetero_predicted": hetero[index],
                "observed": observed[(group, index)],
                "residual": observed[(group, index)] - hetero[index],
            })
    frame = pd.DataFrame(rows)
    v2 = bool(repro_gap <= REPRO_TOLERANCE)

    frame["match"] = frame["residual"].abs() <= TOLERANCE
    matches = int(frame["match"].sum())
    h2 = bool(matches >= MIN_MATCHES)

    band_rho = spearman(
        frame["band_index"].to_numpy(float), frame["residual"].to_numpy(float)
    )
    h3 = bool(abs(band_rho) < FLATNESS_CEILING)

    g2_mid = frame.loc[(frame["group_id"] == 2) & (frame["band_index"] == 2)]
    g2_mid_residual = float(g2_mid["residual"].iloc[0]) if len(g2_mid) else float("nan")
    h4 = bool(abs(g2_mid_residual) <= TOLERANCE)

    if not v1:
        verdict = "WIND_ERROR_MEASUREMENT_OUT_OF_RANGE_RESULT_VOID"
    elif not v2:
        verdict = "PROPAGATION_HARNESS_DRIFTED_RESULT_VOID"
    elif not h1:
        verdict = "WIND_ERROR_NOT_HETEROSCEDASTIC"
    elif h2 and h3:
        verdict = "ACCOUNTING_CLOSES_SKILL_IS_WIND_ERROR_THROUGH_POWER_CURVE"
    elif h2:
        verdict = "HETEROSCEDASTIC_FIT_IMPROVES_BUT_PATTERN_REMAINS"
    else:
        verdict = "STRUCTURE_BEYOND_WIND_ERROR_REMAINS"

    payload: dict[str, Any] = {
        "node_id": NODE_ID,
        "lane": LANE,
        "parent": PARENT_NODE,
        "candidate": False,
        "diagnostic_only": True,
        "method": "CONDITIONAL_VERIFICATION (Murphy & Winkler 1987; Whitaker & Loughe 1998)",
        "surface": info,
        "wind_rows": int(len(wind)),
        "overall_sigma_v": overall_sigma,
        "sigma_bins": sigma_rows,
        "fits": {str(g): {"intercept": fits[g][0], "slope": fits[g][1],
                          "spearman_v_sigma": monotone[g]} for g in fits},
        "cells": frame.to_dict(orient="records"),
        "checks": {
            "V1_sigma_in_range": v1, "V1_overall_sigma": overall_sigma,
            "V2_c65_reproduced": v2, "V2_max_gap": repro_gap,
        },
        "matches": matches,
        "c65_matches": int(c65["matches"]),
        "band_residual_spearman": band_rho,
        "g2_mid_residual": g2_mid_residual,
        "hypotheses": {
            "H1_sigma_increases_with_wind": h1,
            "H2_eight_of_twelve_match": h2,
            "H3_band_monotonicity_gone": h3,
            "H4_g2_mid_explained": h4,
        },
        "limitation": (
            "나셀 풍속계는 로터 뒤에 있어 자유유입풍속이 아니다. teacher 가 "
            "`scada_ws` 를 표적으로 학습했으므로 이 잔차는 **우리가 쓰는 사상의 오차**를 "
            "재며, 자유유입풍속의 예보오차와는 다르다."
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
        "# M271 P4 사이클 66 — 이분산 풍속오차",
        "",
        f"노드 `{NODE_ID}` / 레인 {LANE} / 부모 `{PARENT_NODE}` / **진단 전용**",
        "",
        f"확률면 prob `{info['probability_digest']}` / 풍속 대조행 {len(wind):,}",
        "",
        "## 1. sigma_v(v) 독립 계측",
        "",
        f"전체 잔차 표준편차 **{overall_sigma:.4f}** m/s "
        f"(C58 test 행 1.37~1.50 과 대조)",
        "",
        "| 그룹 | 절편 a | 기울기 b | Spearman(v, sigma) |",
        "|---:|---:|---:|---:|",
    ]
    for group in (1, 2, 3):
        a, b_slope = fits[group]
        lines.append(f"| {group} | {a:+.4f} | {b_slope:+.4f} | {monotone[group]:+.3f} |")
    lines += [
        "",
        "| 그룹 | 예측풍속 | sigma_v | 상대 | 편의 | 행 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sigma_rows:
        lines.append(
            f"| {row['group_id']} | {row['wind_bin']:.1f} | {row['sigma_v']:.3f} | "
            f"{row['relative']:.3f} | {row['bias']:+.3f} | {row['rows']} |"
        )
    lines += [
        "",
        "## 2. 전파 재실행",
        "",
        "| 그룹 | 대역 | sigma_v | C65 상수 | 이분산 | 관측 | 차 |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in frame.itertuples():
        mark = "" if row.match else " **X**"
        lines.append(
            f"| {row.group_id} | {row.band} | {row.sigma_v_used:.3f} | "
            f"{row.constant_predicted:.3f} | {row.hetero_predicted:.3f} | "
            f"{row.observed:.3f} | {row.residual:+.3f}{mark} |"
        )
    lines += [
        "",
        "## 3. 사전확약",
        "",
        f"- V1 전체 sigma_v {overall_sigma:.4f} in {SIGMA_RANGE} -> **{v1}**",
        f"- V2 C65 상수 결과 재현 (최대 차 {repro_gap:.5f}) -> **{v2}**",
        f"- H1 sigma_v 가 풍속따라 증가 -> **{h1}**",
        f"- H2 {matches}/12 이 ±{TOLERANCE} 이내 (C65 는 {c65['matches']}/12) -> **{h2}**",
        f"- H3 대역 단조성 소멸 (Spearman {band_rho:+.3f}) -> **{h3}**",
        f"- H4 g2 중간대 차 {g2_mid_residual:+.3f} -> **{h4}**",
        "",
        "## 4. 판정",
        "",
        f"**{verdict}**",
        "",
        payload["limitation"],
        "",
        f"digest `{payload['digest']}`",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("=== 완료 ===")
    print(f"[C66] 전체 sigma_v {overall_sigma:.4f} (V1 {v1}) / V2 재현 {v2} "
          f"(최대차 {repro_gap:.5f})")
    for group in (1, 2, 3):
        a, b_slope = fits[group]
        print(f"[C66] g{group} sigma_v = {a:+.4f} {b_slope:+.4f}*v  "
              f"Spearman {monotone[group]:+.3f}")
    for row in frame.itertuples():
        mark = "" if row.match else "  <-- 어긋남"
        print(f"[C66] g{row.group_id} {row.band}  sv {row.sigma_v_used:.3f}  "
              f"상수 {row.constant_predicted:.3f} -> 이분산 {row.hetero_predicted:.3f}  "
              f"관측 {row.observed:.3f}  차 {row.residual:+.3f}{mark}")
    print(f"[C66] H1 {h1} / H2 {matches}/12 {h2} / H3 rho {band_rho:+.3f} {h3} / "
          f"H4 {g2_mid_residual:+.3f} {h4}")
    print(f"[C66] 판정: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
