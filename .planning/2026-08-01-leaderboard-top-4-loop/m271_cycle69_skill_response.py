"""M271 P4 사이클 69 — 풍속 오차를 얼마나 줄여야 하는가: 스킬 반응곡선을 직접 잰다.

사이클 68 이 회계를 열었다.

    MODEL 0.3067   CURVE_TEACHER 0.2897   CURVE_ORACLE 0.7592

모형 전체가 더하는 것이 **+0.017**, 완벽한 풍속이 더하는 것이 **+0.453**. 풍속이 전부다.

그러면 남는 질문은 크기다. **얼마나 좋은 풍속이면 목표에 닿는가.** 지금까지 이 질문은
사이클 46 이 역산한 요구치(오차 13.3% 감소)로 다뤄졌고 C52 가 가용 외부소스를 재서
그 요구의 30~45% 라고 판정했다. 그런데 13.3% 는 **역산값**이지 측정값이 아니다.

이제 직접 잴 수 있다. 관측 나셀풍속에 **C66 이 측정한 오차 형태**를 배율 `k` 로 넣어
실측 커브에 통과시키면, `k` 를 0(완벽) 에서 1(현재) 까지 훑어 스킬 반응곡선이 나온다.
전파 근사도, 역산도 없다.

    pred(k) = 커브( scada_ws + k * eps ),   eps ~ N(0, sigma_v(v))

`k=1` 이 현재를 재현하는지가 이 노드의 타당성 가드다. 재현하면 곡선 전체를 믿을 수 있다.

**① 방법 리서치**

  - 이건 **섭동 민감도 분석**이다. 예보 검증에서 "예측인자 오차를 인위로 조절해 스킬
    반응을 보는" 절차는 관측시스템 실험(OSE/OSSE)의 축소판이고, 표준 도구다.
  - 이 프로젝트는 이미 같은 뼈대를 썼다 — 사이클 46·48 의 **오차 스케일링 반사실**
    `pred_k = actual + k(pred - actual)` 로 상위권 격차를 설명했고 k 불일치 0.0057 로
    교차검증됐다. 거기서는 **출력** 공간에서 스케일했고, 여기서는 **풍속** 공간에서
    스케일한다. 풍속 공간이 옳은 이유는 개선 가능한 대상이 풍속 예보이기 때문이다.
  - 오차 형태는 C66 이 측정한 이분산 `sigma_v(v) = a + b*v` 를 쓴다. 가우스 가정은
    남지만, `k=1` 재현 가드가 그 가정의 적정성을 **직접 검정**한다.
  - **채택**: `k` 격자 섭동 + 공식 산식. 적합 없음.

**② 사양 동결**

  입력   확률면 캐시 v2. `scada_ws` 보유 행 전체(유효행 제한 **없이** — 공식 산식이
         스스로 유효행을 가른다).
  커브   C57 실측 커브 + C67 동결 cut-in. 그룹별.
  섭동   `k` 격자 (0.0, 0.1, ..., 1.0) 실행 전 동결. `eps ~ N(0, a + b*v)` 에서
         `v = scada_ws`. 시드 20260805. 잡음 실현 **30 회 평균**으로 몬테카를로
         오차를 줄인다.
  점수   공식 산식 Total / 1-NMAE / FICR. 대역별 적중률도 같이.

  **타당성 가드**
    V1  `k=1` 의 적중률이 C68 의 CURVE_TEACHER **0.2897 의 ±0.02 이내**.
        측정된 오차 형태를 다시 넣으면 관측된 기계를 재현해야 한다. 벗어나면
        오차 모형(가우스·이분산선형)이 부적절하고 곡선을 못 믿는다.
    V2  `k=0` 의 적중률이 C68 의 CURVE_ORACLE **0.7592 의 ±0.01 이내**.
        섭동을 끄면 오라클 팔과 같아야 한다.

  사전확약 (V1·V2 통과시에만 판정):
    H1  반응곡선이 `k` 에 **단조 감소**한다.
    H2  Total 이 목표 0.66 에 닿는 `k*` 가 존재하고 `k* > 0` 이다.
        즉 완벽한 풍속까지 안 가도 된다.
    H3  요구 감소율 `1 - k*` 가 C52 가 측정한 가용 감소 **3.9~6.1% 를 초과**한다.
        참이면 가용 외부소스로는 못 닿는다는 C52 판정이 재확인되고, 이번엔
        역산이 아니라 측정으로 확인된다.
    H4  반응곡선이 `k` 에 대해 **오목**하다(수확체증). 즉 초기 감소가 후기 감소보다
        많이 준다. 부호 예단 없음 — 볼록이면 작은 개선이 거의 값을 못 하고,
        오목이면 부분 개선도 값을 한다. 이것이 외부소스 축의 경제성을 정한다.

  **`k*` 는 보간으로 읽고 격자점으로 반올림하지 않는다.**

**진단 전용.** 후보 아님. `scada_ws` 는 평가기간 부재라 피처가 될 수 없다(C1N39) —
이 노드는 그것을 **반사실의 기준점**으로만 쓴다. 게이트 미수정. 제출 없음.
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
from m271_cycle65_wind_limited_bound import BAND_HIT, ELIGIBLE, MIN_ROWS, SEED
from m271_cycle67_exact_curve_propagation import build_curve
from m271_decision_surface import load_surface
from m271_evaluate_candidate import official

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
C57_RECEIPT = REPORTS / "m271_cycle57_ficr_ceiling_receipt.json"
C60_RECEIPT = REPORTS / "m271_cycle60_level_temperature_receipt.json"
C66_RECEIPT = REPORTS / "m271_cycle66_heteroscedastic_wind_receipt.json"
C68_RECEIPT = REPORTS / "m271_cycle68_empirical_decomposition_receipt.json"
REPORT_MD = REPORTS / "m271_cycle69_skill_response.md"
RECEIPT = REPORTS / "m271_cycle69_skill_response_receipt.json"

NODE_ID = "C1N69_SKILL_RESPONSE"
LANE = "L6"
PARENT_NODE = "C1N68_EMPIRICAL_DECOMPOSITION"

K_GRID = tuple(round(0.1 * i, 1) for i in range(11))
REALISATIONS = 30
TARGET = 0.66
V1_TOLERANCE = 0.02
V2_TOLERANCE = 0.01
C52_AVAILABLE = (0.039, 0.061)


def main() -> int:
    store, info = load_surface()
    c57 = json.loads(C57_RECEIPT.read_text(encoding="utf-8"))["result"]
    c60 = json.loads(C60_RECEIPT.read_text(encoding="utf-8"))
    c66 = json.loads(C66_RECEIPT.read_text(encoding="utf-8"))
    c68 = json.loads(C68_RECEIPT.read_text(encoding="utf-8"))
    assert info["probability_digest"] == c60["probability_digest"], "확률면 불일치"

    curves = {
        group: build_curve(
            [b for b in c57["per_group"][str(group)]["bins"] if b["rows"] >= MIN_ROWS]
        )
        for group in (1, 2, 3)
    }
    fits = {int(g): (blk["intercept"], blk["slope"]) for g, blk in c66["fits"].items()}

    parts: list[pd.DataFrame] = []
    for fold, entry in store.items():
        temperature = np.full(
            len(entry["capacity"]), float(c60["chosen"]["global"][fold])
        )
        frame = entry["meta"].copy()
        frame["group_id"] = entry["group"]
        frame["capacity"] = entry["capacity"]
        frame["actual_rate"] = frame["actual_kwh"] / entry["capacity"]
        frame["model_rate"] = bayes_decision(
            sharpen_by_row(entry["probability"], temperature)
        )
        frame["scada_ws"] = entry["scada_ws"]
        parts.append(frame)
    data = pd.concat(parts, ignore_index=True)
    data = data.loc[data["scada_ws"].notna()].reset_index(drop=True)
    data["month"] = data["forecast_kst_dtm"].dt.to_period("M").astype(str)

    sigma = np.zeros(len(data), dtype="float64")
    base = np.zeros(len(data), dtype="float64")
    wind = data["scada_ws"].to_numpy(dtype="float64")
    for group, (cv, cp) in curves.items():
        mask = (data["group_id"] == group).to_numpy()
        a, b = fits[group]
        sigma[mask] = a + b * wind[mask]
        base[mask] = np.interp(wind[mask], cv, cp, left=0.0, right=cp[-1])
    sigma = np.clip(sigma, 1e-6, None)

    eligible = data["actual_rate"] >= ELIGIBLE
    band = band_of(data["actual_rate"].to_numpy(float))

    rng = np.random.default_rng(SEED)
    rows: list[dict[str, Any]] = []
    band_hits: dict[float, dict[tuple[int, int], float]] = {}
    for k in K_GRID:
        totals, nmaes, ficrs, hits = [], [], [], []
        accumulated: dict[tuple[int, int], list[float]] = {}
        for _ in range(REALISATIONS if k > 0 else 1):
            noisy = wind + k * rng.normal(0.0, 1.0, len(wind)) * sigma
            pred = np.zeros(len(data), dtype="float64")
            for group, (cv, cp) in curves.items():
                mask = (data["group_id"] == group).to_numpy()
                pred[mask] = np.interp(noisy[mask], cv, cp, left=0.0, right=cp[-1])
            scored = data.loc[:, ["forecast_id", "forecast_kst_dtm", "group_id",
                                  "actual_kwh", "month"]].copy()
            scored["prediction_kwh"] = pred * data["capacity"].to_numpy(float)
            result = official(scored)
            totals.append(result["total"])
            nmaes.append(result["one_minus_nmae"])
            ficrs.append(result["ficr"])
            hit = np.abs(pred - data["actual_rate"].to_numpy(float)) <= BAND_HIT
            hits.append(float(hit[eligible.to_numpy()].mean()))
            for group in (1, 2, 3):
                for index in range(len(LABELS)):
                    cell = (data["group_id"].to_numpy() == group) & (band == index)
                    if cell.any():
                        accumulated.setdefault((group, index), []).append(
                            float(hit[cell].mean())
                        )
        rows.append({
            "k": k,
            "total": float(np.mean(totals)),
            "total_sd": float(np.std(totals, ddof=1)) if len(totals) > 1 else 0.0,
            "one_minus_nmae": float(np.mean(nmaes)),
            "ficr": float(np.mean(ficrs)),
            "hit": float(np.mean(hits)),
        })
        band_hits[k] = {key: float(np.mean(v)) for key, v in accumulated.items()}
    frame = pd.DataFrame(rows)

    hit_at_1 = float(frame.loc[frame["k"] == 1.0, "hit"].iloc[0])
    hit_at_0 = float(frame.loc[frame["k"] == 0.0, "hit"].iloc[0])
    v1 = bool(abs(hit_at_1 - c68["overall_hit"]["curve_teacher"]) <= V1_TOLERANCE)
    v2 = bool(abs(hit_at_0 - c68["overall_hit"]["curve_oracle"]) <= V2_TOLERANCE)

    h1 = bool((np.diff(frame["total"].to_numpy()) < 0).all())

    totals = frame["total"].to_numpy()
    ks = frame["k"].to_numpy()
    k_star = float("nan")
    if totals.min() <= TARGET <= totals.max():
        order = np.argsort(totals)
        k_star = float(np.interp(TARGET, totals[order], ks[order]))
    h2 = bool(np.isfinite(k_star) and k_star > 0.0)

    required_reduction = 1.0 - k_star if np.isfinite(k_star) else float("nan")
    h3 = bool(np.isfinite(required_reduction)
              and required_reduction > C52_AVAILABLE[1])

    # 오목성 — 이웃 삼중항의 2 계 차분 부호.
    second = np.diff(totals, 2)
    h4 = bool((second <= 0).mean() >= 0.5)

    if not v1:
        verdict = "ERROR_MODEL_DOES_NOT_REPRODUCE_CURRENT_RESULT_VOID"
    elif not v2:
        verdict = "ORACLE_ENDPOINT_MISMATCH_RESULT_VOID"
    elif not h2:
        verdict = "TARGET_UNREACHABLE_EVEN_WITH_PERFECT_WIND"
    elif h3:
        verdict = "REQUIRED_WIND_REDUCTION_EXCEEDS_MEASURED_AVAILABLE"
    else:
        verdict = "REQUIRED_WIND_REDUCTION_WITHIN_MEASURED_AVAILABLE"

    model_scored = data.loc[:, ["forecast_id", "forecast_kst_dtm", "group_id",
                               "actual_kwh", "month"]].copy()
    model_scored["prediction_kwh"] = (
        data["model_rate"].to_numpy(float) * data["capacity"].to_numpy(float)
    )
    model_result = official(model_scored)

    payload: dict[str, Any] = {
        "node_id": NODE_ID,
        "lane": LANE,
        "parent": PARENT_NODE,
        "candidate": False,
        "diagnostic_only": True,
        "method": "PERTURBATION_SENSITIVITY (OSE/OSSE 축소판; C1N46 오차스케일링의 풍속판)",
        "surface": info,
        "rows": int(len(data)),
        "k_grid": list(K_GRID),
        "realisations": REALISATIONS,
        "seed": SEED,
        "sigma_v_fits": {str(g): {"intercept": fits[g][0], "slope": fits[g][1]}
                         for g in fits},
        "response": frame.to_dict(orient="records"),
        "band_hit": {
            str(k): {f"g{g}|{LABELS[i]}": v for (g, i), v in cells.items()}
            for k, cells in band_hits.items()
        },
        "model_on_same_rows": model_result,
        "checks": {
            "V1_k1_matches_curve_teacher": v1,
            "V1_hit_at_k1": hit_at_1,
            "V1_reference": c68["overall_hit"]["curve_teacher"],
            "V2_k0_matches_curve_oracle": v2,
            "V2_hit_at_k0": hit_at_0,
            "V2_reference": c68["overall_hit"]["curve_oracle"],
        },
        "k_star": k_star,
        "required_reduction": required_reduction,
        "c52_available_reduction": list(C52_AVAILABLE),
        "hypotheses": {
            "H1_monotone_decreasing": h1,
            "H2_target_reachable_short_of_perfect": h2,
            "H3_required_exceeds_available": h3,
            "H4_concave": h4,
        },
        "limitation": (
            "잡음을 가우스로 뒀다. V1 이 그 적정성을 현재 지점에서 검정하지만 "
            "`k` 가 작은 영역에서는 검정되지 않는다. `scada_ws` 는 로터 뒤 관측이라 "
            "`k=0` 은 위쪽으로 편향된 상한이다."
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
        "# M271 P4 사이클 69 — 풍속 오차에 대한 스킬 반응곡선",
        "",
        f"노드 `{NODE_ID}` / 레인 {LANE} / 부모 `{PARENT_NODE}` / **진단 전용**",
        "",
        f"`pred(k) = 커브(scada_ws + k * eps)`, `eps ~ N(0, a + b*v)` (C66 측정), "
        f"실현 {REALISATIONS} 회 평균, 시드 {SEED}, 행 {len(data):,}",
        "",
        "## 1. 반응곡선",
        "",
        "| k | Total | (표준편차) | 1-NMAE | FICR | 적중률 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in frame.itertuples():
        lines.append(
            f"| {row.k:.1f} | **{row.total:.6f}** | {row.total_sd:.6f} | "
            f"{row.one_minus_nmae:.6f} | {row.ficr:.6f} | {row.hit:.4f} |"
        )
    lines += [
        "",
        f"같은 행에서 MODEL 의 공식 점수: Total **{model_result['total']:.6f}** "
        f"(1-NMAE {model_result['one_minus_nmae']:.6f} / FICR {model_result['ficr']:.6f})",
        "",
        "## 2. 타당성 가드",
        "",
        f"- V1 `k=1` 적중 {hit_at_1:.4f} vs C68 CURVE_TEACHER "
        f"{c68['overall_hit']['curve_teacher']:.4f} -> **{v1}**",
        f"- V2 `k=0` 적중 {hit_at_0:.4f} vs C68 CURVE_ORACLE "
        f"{c68['overall_hit']['curve_oracle']:.4f} -> **{v2}**",
        "",
        "## 3. 사전확약",
        "",
        f"- H1 단조 감소 -> **{h1}**",
        f"- H2 목표 {TARGET} 에 닿는 `k*` 존재 (`k*` = **{k_star:.4f}**) -> **{h2}**",
        f"- H3 요구 감소율 **{required_reduction:.1%}** > C52 가용 "
        f"{C52_AVAILABLE[0]:.1%}~{C52_AVAILABLE[1]:.1%} -> **{h3}**",
        f"- H4 오목(수확체증) -> **{h4}**",
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
    print(f"[C69] V1 k=1 적중 {hit_at_1:.4f} vs {c68['overall_hit']['curve_teacher']:.4f} "
          f"-> {v1}")
    print(f"[C69] V2 k=0 적중 {hit_at_0:.4f} vs {c68['overall_hit']['curve_oracle']:.4f} "
          f"-> {v2}")
    for row in frame.itertuples():
        print(f"[C69] k={row.k:.1f}  Total {row.total:.6f}  1-NMAE "
              f"{row.one_minus_nmae:.6f}  FICR {row.ficr:.6f}  적중 {row.hit:.4f}")
    print(f"[C69] 같은 행 MODEL Total {model_result['total']:.6f}")
    print(f"[C69] k* = {k_star:.4f} -> 요구 감소율 {required_reduction:.1%} "
          f"(C52 가용 {C52_AVAILABLE[0]:.1%}~{C52_AVAILABLE[1]:.1%})")
    print(f"[C69] H1 {h1} / H2 {h2} / H3 {h3} / H4 {h4}")
    print(f"[C69] 판정: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
