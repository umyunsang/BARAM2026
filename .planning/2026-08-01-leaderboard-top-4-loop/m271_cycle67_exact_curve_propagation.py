"""M271 P4 사이클 67 — 선형화를 버린다: 풍속 오차를 실측 파워커브에 그대로 통과시킨다.

사이클 66 이 이분산을 확인했고(Spearman +0.91~+0.98) 그 보정으로 **고출력대가 닫혔다**
— g2 고출력 -0.112 -> -0.006, g3 -0.093 -> -0.024. 그런데 저출력대는 꿈쩍도 안 했다.

    (0.10,0.25]   g1 +0.124   g2 +0.124   g3 +0.118    관측이 예측보다 **좋다**

세 그룹에서 같은 크기다. 잡음이 아니라 모형 결함이고, 어디인지도 좁혀진다.

1 차 선형화 `dP = (dP/dv) * eps_v` 는 커브가 국소적으로 직선이라고 본다. 저풍속
구간에서는 두 가지가 그 가정을 깬다.

    곡률   저풍속에서 파워커브는 볼록하다(P ~ v^3 영역). 같은 크기의 +-eps 가
           비대칭한 출력 변화를 낸다.
    절단   cut-in 아래로는 출력이 0 에 붙는다. 음의 풍속 오차가 출력을 음수로
           보내지 않는다. 선형화는 이 바닥을 모르므로 산포를 **과대**평가한다.

둘 다 저출력대에서만 세게 작동하고, 둘 다 선형화가 산포를 과대평가하는 방향이다.
관측이 예측보다 좋게 나오는 부호와 정확히 맞는다.

고칠 방법은 근사를 개선하는 게 아니라 **없애는** 것이다. C57 의 실측 커브가 있으니
`dP = P(v + eps_v) - P(v)` 를 직접 계산하면 곡률도 절단도 자동으로 들어간다.

**① 방법 리서치**

  - 비선형 함수를 통한 불확도 전파에서 1 차 근사가 부족할 때의 표준 대안은
    **몬테카를로 전파**다. JCGM 101:2008(GUM Supplement 1)이 정확히 이 경우를 위해
    쓰였고, 1 차 전파(JCGM 100)가 부적절한 조건으로 **강한 비선형성과 정의역
    경계**를 든다. 여기가 그 두 조건이다.
  - 이미 몬테카를로를 쓰고 있었다(C65·C66). 바뀌는 것은 표본을 **선형 근사**에
    통과시키느냐 **실측 커브**에 통과시키느냐 하나뿐이다.
  - **채택**: C57 의 그룹별 실측 커브를 보간해 `P(v+eps) - P(v)`. 새 데이터 없음.

**② 사양 동결**

  커브   C57 receipt 의 `(bin_center, mean_power)`, `rows >= 200` 구간.
         **cut-in 을 명시적으로 앞에 붙인다** — `(0.0, 0.0)` 과 `(3.0, 0.0)`.
         IEC 파워커브의 통상 cut-in 이 3 m/s 이고, A5 실측 최저 구간이 4.5~5.5 m/s 라
         그 아래를 외삽하지 않으려면 바닥을 명시해야 한다. 위쪽은 마지막 값으로 평탄.
         **이 두 점은 실행 전에 동결하며 결과를 보고 조정하지 않는다.**
  전파   `dP = interp(v + eps_v) - interp(v)`, `eps_v ~ N(0, sigma_v(v))`,
         `sigma_v` 는 C66 의 그룹별 적합 `a + b*v`. 잔차 `r` 은 C57 구간별
         `sigma_resid`. `error = dP + r`. 추첨·시드는 C65·C66 과 동일.
  대역   실제 출력 대역 4 개 x 그룹 3 = 12 셀. 관측은 C64·C65·C66 과 동일 결정
         (C60 GLOBAL fold-외 T).

  **타당성 가드**
    V1  선형 모드로 돌리면 C66 의 이분산 결과가 ±0.005 이내로 재현된다.
        전파 코드가 커브 교체 외에 바뀌지 않았음을 못박는다.
    V2  커브 전파의 예측 적중률이 12 셀 모두 C57 천장 미만. 풍속 오차를 더했는데
        천장을 넘으면 계산이 틀린 것이다.

  사전확약 (V1·V2 통과시에만 판정):
    H1  12 셀 중 **8 개 이상**이 ±0.05 이내. C65 상수 3/12, C66 이분산 7/12.
    H2  **저출력대 세 셀이 모두** ±0.05 이내. C66 이 남긴 정확한 결손이고,
        곡률·절단 진단이 옳다면 여기가 닫혀야 한다.  **핵심 판정.**
    H3  대역 단조성 소멸 — |Spearman(대역 인덱스, 잔차)| < 0.5.
        C65 이후 계속 강한 음수였다(C66 -0.756).
    H4  g2 중간대가 여전히 설명된다 (|잔차| <= 0.05).

  H1·H2·H3 가 함께 참이면 **회계가 닫힌다**: 관측 스킬 = 이분산 풍속오차를 실측
  파워커브로 전파한 것 + 내재 산포. 그러면 남는 지렛대는 `sigma_v` 하나이고,
  C52 가 가용 외부소스로 필요량의 30~45% 만 얻는다고 이미 측정했다.
  H2 가 거짓이면 저출력대에 곡률·절단 아닌 다른 것이 있고 계속 판다.

**진단 전용.** 후보 아님. 게이트 미수정. 제출 없음.
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
    TOLERANCE,
    band_index,
)
from m271_decision_surface import load_surface

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
C57_RECEIPT = REPORTS / "m271_cycle57_ficr_ceiling_receipt.json"
C57B_RECEIPT = REPORTS / "m271_cycle57b_variance_law_receipt.json"
C60_RECEIPT = REPORTS / "m271_cycle60_level_temperature_receipt.json"
C66_RECEIPT = REPORTS / "m271_cycle66_heteroscedastic_wind_receipt.json"
REPORT_MD = REPORTS / "m271_cycle67_exact_curve_propagation.md"
RECEIPT = REPORTS / "m271_cycle67_exact_curve_propagation_receipt.json"

NODE_ID = "C1N67_EXACT_CURVE_PROPAGATION"
LANE = "L6"
PARENT_NODE = "C1N66_HETEROSCEDASTIC_WIND"

CUT_IN = ((0.0, 0.0), (3.0, 0.0))  # 실행 전 동결
MIN_MATCHES = 8
LOW_BAND = 0
REPRO_TOLERANCE = 0.005
FLATNESS_CEILING = 0.5


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    rx, ry = rx - rx.mean(), ry - ry.mean()
    denom = float(np.sqrt((rx * rx).sum() * (ry * ry).sum()))
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


def build_curve(bins: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    v = [c[0] for c in CUT_IN] + [float(b["bin_center"]) for b in bins]
    p = [c[1] for c in CUT_IN] + [float(b["mean_power"]) for b in bins]
    order = np.argsort(v)
    return np.asarray(v)[order], np.asarray(p)[order]


def propagate(
    bins: list[dict],
    slopes: np.ndarray,
    sigma_at,
    mode: str,
    curve: tuple[np.ndarray, np.ndarray],
    rng: np.random.Generator,
) -> dict[int, float]:
    """`mode='linear'` 는 C66 과 동일. `mode='curve'` 는 실측 커브를 그대로 통과."""
    cv, cp = curve
    per_band: dict[int, list[tuple[float, float]]] = {}
    for b, slope in zip(bins, slopes, strict=True):
        index = band_index(float(b["mean_power"]))
        if index < 0:
            continue
        centre = float(b["bin_center"])
        sigma = max(float(sigma_at(centre)), 1e-6)
        eps = rng.normal(0.0, sigma, DRAWS)
        if mode == "linear":
            delta = slope * eps
        else:
            delta = np.interp(centre + eps, cv, cp, left=0.0, right=cp[-1]) - np.interp(
                centre, cv, cp, left=0.0, right=cp[-1]
            )
        resid = rng.normal(0.0, float(b["sigma_resid"]), DRAWS)
        hit = float((np.abs(delta + resid) <= BAND_HIT).mean())
        per_band.setdefault(index, []).append((hit, float(b["gen_weight"])))
    return {
        index: float(np.average([h for h, _ in items],
                                weights=[w for _, w in items]))
        for index, items in per_band.items()
    }


def main() -> int:
    store, info = load_surface()
    c57 = json.loads(C57_RECEIPT.read_text(encoding="utf-8"))["result"]
    c57b = json.loads(C57B_RECEIPT.read_text(encoding="utf-8"))
    c60 = json.loads(C60_RECEIPT.read_text(encoding="utf-8"))
    c66 = json.loads(C66_RECEIPT.read_text(encoding="utf-8"))
    assert info["probability_digest"] == c60["probability_digest"], "확률면 불일치"

    fits = {int(g): (blk["intercept"], blk["slope"]) for g, blk in c66["fits"].items()}
    ceilings = {
        (int(g), b["band"]): float(b["unit_over_4"])
        for g, blk in c57b["per_group"].items()
        for b in blk["bands"]
    }
    c57b_labels = ("(0.10,0.25]", "(0.25,0.45]", "(0.45,0.70]", "(0.70,1.10]")
    c66_lookup = {
        (int(c["group_id"]), LABELS.index(c["band"])): float(c["hetero_predicted"])
        for c in c66["cells"]
    }

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
    observed = {
        (int(g), int(b)): float(blk["hit"].mean())
        for (g, b), blk in data.groupby(["group_id", "band"])
    }

    rows: list[dict[str, Any]] = []
    repro_gap = 0.0
    for group in (1, 2, 3):
        bins = [b for b in c57["per_group"][str(group)]["bins"] if b["rows"] >= MIN_ROWS]
        v = np.array([b["bin_center"] for b in bins], dtype="float64")
        p = np.array([b["mean_power"] for b in bins], dtype="float64")
        slopes = np.gradient(p, v)
        curve = build_curve(bins)
        a, b_slope = fits[group]

        def sigma_at(x: float, a=a, b=b_slope) -> float:
            return a + b * x

        linear = propagate(bins, slopes, sigma_at, "linear", curve,
                           np.random.default_rng(SEED))
        exact = propagate(bins, slopes, sigma_at, "curve", curve,
                          np.random.default_rng(SEED))
        for index, label in enumerate(LABELS):
            if index not in exact or (group, index) not in observed:
                continue
            repro_gap = max(
                repro_gap, abs(linear[index] - c66_lookup.get((group, index), linear[index]))
            )
            rows.append({
                "group_id": group,
                "band": label,
                "band_index": index,
                "ceiling": ceilings[(group, c57b_labels[index])],
                "linear_predicted": linear[index],
                "curve_predicted": exact[index],
                "observed": observed[(group, index)],
                "residual": observed[(group, index)] - exact[index],
            })
    frame = pd.DataFrame(rows)

    v1 = bool(repro_gap <= REPRO_TOLERANCE)
    v2 = bool((frame["curve_predicted"] < frame["ceiling"]).all())

    frame["match"] = frame["residual"].abs() <= TOLERANCE
    matches = int(frame["match"].sum())
    h1 = bool(matches >= MIN_MATCHES)

    low = frame.loc[frame["band_index"] == LOW_BAND]
    h2 = bool(low["match"].all())

    band_rho = spearman(
        frame["band_index"].to_numpy(float), frame["residual"].to_numpy(float)
    )
    h3 = bool(abs(band_rho) < FLATNESS_CEILING)

    g2_mid = frame.loc[(frame["group_id"] == 2) & (frame["band_index"] == 2)]
    g2_mid_residual = float(g2_mid["residual"].iloc[0]) if len(g2_mid) else float("nan")
    h4 = bool(abs(g2_mid_residual) <= TOLERANCE)

    if not v1:
        verdict = "HARNESS_DRIFTED_RESULT_VOID"
    elif not v2:
        verdict = "CURVE_PROPAGATION_EXCEEDS_CEILING_CALCULATION_WRONG"
    elif h1 and h2 and h3:
        verdict = "ACCOUNTING_CLOSES_SKILL_IS_WIND_ERROR_THROUGH_MEASURED_CURVE"
    elif h2:
        verdict = "LOW_BAND_EXPLAINED_BY_CURVATURE_OTHERS_REMAIN"
    else:
        verdict = "CURVATURE_DOES_NOT_EXPLAIN_LOW_BAND"

    payload: dict[str, Any] = {
        "node_id": NODE_ID,
        "lane": LANE,
        "parent": PARENT_NODE,
        "candidate": False,
        "diagnostic_only": True,
        "method": "MONTE_CARLO_PROPAGATION (JCGM 101:2008 GUM Supplement 1)",
        "cut_in": [list(c) for c in CUT_IN],
        "sigma_v_fits": {str(g): {"intercept": fits[g][0], "slope": fits[g][1]}
                         for g in fits},
        "surface": info,
        "cells": frame.to_dict(orient="records"),
        "checks": {"V1_c66_reproduced": v1, "V1_max_gap": repro_gap,
                   "V2_below_ceiling": v2},
        "matches": matches,
        "match_history": {"c65_constant": 3, "c66_hetero": int(c66["matches"]),
                          "c67_curve": matches},
        "band_residual_spearman": band_rho,
        "g2_mid_residual": g2_mid_residual,
        "hypotheses": {
            "H1_eight_of_twelve": h1,
            "H2_low_band_closed": h2,
            "H3_band_monotonicity_gone": h3,
            "H4_g2_mid_explained": h4,
        },
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
        "# M271 P4 사이클 67 — 실측 커브 몬테카를로 전파",
        "",
        f"노드 `{NODE_ID}` / 레인 {LANE} / 부모 `{PARENT_NODE}` / **진단 전용**",
        "",
        f"`error = P(v + eps_v) - P(v) + r`, `eps_v ~ N(0, a + b*v)`, "
        f"cut-in {CUT_IN}, 추첨 {DRAWS:,}, 시드 {SEED}",
        "",
        "## 1. 셀별",
        "",
        "| 그룹 | 대역 | 천장 | C66 선형 | C67 커브 | 관측 | 차 |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in frame.itertuples():
        mark = "" if row.match else " **X**"
        lines.append(
            f"| {row.group_id} | {row.band} | {row.ceiling:.3f} | "
            f"{row.linear_predicted:.3f} | {row.curve_predicted:.3f} | "
            f"{row.observed:.3f} | {row.residual:+.3f}{mark} |"
        )
    lines += [
        "",
        "## 2. 사전확약",
        "",
        f"- V1 C66 선형 재현 (최대 차 {repro_gap:.5f}) -> **{v1}**",
        f"- V2 커브 예측이 12 셀 모두 천장 미만 -> **{v2}**",
        f"- H1 {matches}/12 이 ±{TOLERANCE} 이내 -> **{h1}** "
        f"(상수 3 -> 이분산 {c66['matches']} -> 커브 {matches})",
        f"- H2 저출력대 세 셀 모두 이내 -> **{h2}**",
        f"- H3 대역 단조성 소멸 (Spearman {band_rho:+.3f}) -> **{h3}**",
        f"- H4 g2 중간대 차 {g2_mid_residual:+.3f} -> **{h4}**",
        "",
        "## 3. 판정",
        "",
        f"**{verdict}**",
        "",
        f"digest `{payload['digest']}`",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("=== 완료 ===")
    print(f"[C67] V1 재현 {v1} (최대차 {repro_gap:.5f}) / V2 천장미만 {v2}")
    for row in frame.itertuples():
        mark = "" if row.match else "  <-- 어긋남"
        print(f"[C67] g{row.group_id} {row.band}  선형 {row.linear_predicted:.3f} -> "
              f"커브 {row.curve_predicted:.3f}  관측 {row.observed:.3f}  "
              f"차 {row.residual:+.3f}{mark}")
    print(f"[C67] 적중 추이  상수 3/12 -> 이분산 {c66['matches']}/12 -> 커브 {matches}/12")
    print(f"[C67] H1 {h1} / H2 저출력대 {h2} / H3 rho {band_rho:+.3f} {h3} / "
          f"H4 {g2_mid_residual:+.3f} {h4}")
    print(f"[C67] 판정: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
