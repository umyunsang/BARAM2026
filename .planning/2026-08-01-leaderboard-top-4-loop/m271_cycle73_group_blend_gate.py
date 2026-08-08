"""M271 P4 사이클 73 — 그룹별 결합이 게이트를 통과하는가.

사이클 72 는 V2 가드로 무효다. 내가 박아 둔 참조 상수가 틀렸기 때문이다 — C69 의
`k=1` 팔은 `curve(scada_ws + 시뮬레이션 잡음)` 이지 `curve(sitewind)` 가 아니었다.
실제 커브 직독은 **0.594183** 으로 MODEL(0.604043)에 진다.

**팔 값 자체는 유효하다.** 틀린 것은 참조 상수뿐이고 계산은 옳았다. 그 값들이 이것이다.

    MODEL         0.604043
    GLOBAL_ALPHA  0.603794   (모형보다 **낮다** — 전역 결합은 도움이 안 된다)
    GROUP_ALPHA   0.608974   (모형보다 **+0.004931**)

C72 는 GLOBAL 만 게이트에 걸었고 `[-O--]` 5/9 월로 기각됐다. **GROUP_ALPHA 의 게이트는
아직 모른다.** 그것이 이 노드의 질문이다.

그리고 이 결과 자체가 이상하다. C60 에서는 그룹 자유도를 준 LEVELGROUP(9 파라미터)이
LEVEL(3 파라미터)에 **졌다** — 과적합 서명이었다. 여기서는 그룹 자유도(3 파라미터)가
전역(1 파라미터)을 **이긴다**. 둘 다 fold-외 선택인데 방향이 반대다.

설명 후보가 둘 있고 게이트가 그것을 가른다.

    실체   그룹마다 커브 직독의 상대 가치가 실제로 다르다(C57b 의 theta 가 다르므로
           그럴 근거가 있다). 그러면 월별로도 일관돼 게이트를 통과한다.
    잡음   좌표상승이 fold 두 개에서 우연을 주웠다. 그러면 C60 처럼 월별 비일관으로
           게이트가 기각한다.

**① 방법 리서치**

  - 새 방법 없음. 동결 게이트(`M270_MONTHLY_GATE_v1_frozen_2026-08-04`)가
    정확히 이 구분을 위해 존재한다 — 효과 크기가 아니라 **월별 일관성**을 본다.
  - C60 이 같은 형태의 결과(큰 평균, 비일관)를 냈고 C62 가 2 개월 집중임을 밝혔다.
    같은 진단을 여기서도 붙인다 — 월별 델타 분포를 함께 보고한다.
  - **채택**: 동결 게이트 + 월별 델타 분해. 적합 없음(캐시).

**② 사양 동결**

  입력·절차   C72 와 **완전히 동일**. 확률면 캐시 v3, C60 GLOBAL fold-외 T,
              C57 실측 커브 + C67 cut-in, alpha 격자 0.0~1.0, fold-외 좌표상승.
              **`scada_ws` 미사용** — 두 팔 모두 배포 가능하다.
  참조        MODEL 0.604043 / CURVE **0.594183** (C72 가 실측한 값. 이번엔 맞다)

  **타당성 가드**
    V1  alpha=1 이 MODEL 0.604043 을 ±0.0005 로 재현.
    V2  alpha=0 이 CURVE 0.594183 을 ±0.0005 로 재현.
    V3  GLOBAL_ALPHA 와 GROUP_ALPHA 가 C72 의 값(0.603794 / 0.608974)을 ±0.0005 로
        재현. 절차가 같으므로 같은 값이 나와야 한다.

  사전확약 (V1~V3 통과시에만 판정):
    H1  GROUP_ALPHA 가 MODEL 대비 **동결 게이트 통과**.  **핵심 미결 질문.**
    H2  월별 델타의 **중앙값이 양수**. C60 은 평균 +0.00899 인데 중앙값 +0.00048 로
        2 개월 집중이었다. 여기서 중앙값이 양수면 그 병리가 아니다.
    H3  최대 기여 1 개월을 제거해도 월평균이 양수로 남는다. C60 은 여기서 90% 가
        사라졌다.
    H4  선택된 alpha 가 그룹마다 **다르다**(적어도 두 값). 전부 같으면 GROUP 팔이
        사실상 GLOBAL 이고 이득이 우연이다.
    H5  이득이 FICR 쪽에서 우세.

  H1·H2·H3 가 함께 참이면 **배포 후보**다. H1 이 거짓이면 C60 과 같은 병리이고
  그룹 결합 축을 닫는다.

  **부호를 예단하지 않는다** — C60 전례가 있어 비일관 쪽이 오히려 유력하다.
  예측하는 것은 게이트가 그 둘을 **가른다**는 것이지 어느 쪽인지가 아니다.

  챔피언 로컬 0.630310 은 다른 표면(M115 고정정책)이므로 절대값 비교를 하지 않는다.

게이트 미수정. lockbox·외부데이터·2024 행 미사용. 제출 없음.
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
from m270_monthly_validation import paired_monthly_delta
from m271_cycle40_band_classifier import bayes_decision
from m271_cycle60_level_temperature import sharpen_by_row
from m271_cycle65_wind_limited_bound import MIN_ROWS
from m271_cycle67_exact_curve_propagation import build_curve
from m271_decision_surface import load_surface
from m271_evaluate_candidate import official

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
C57_RECEIPT = REPORTS / "m271_cycle57_ficr_ceiling_receipt.json"
C60_RECEIPT = REPORTS / "m271_cycle60_level_temperature_receipt.json"
REPORT_MD = REPORTS / "m271_cycle73_group_blend_gate.md"
RECEIPT = REPORTS / "m271_cycle73_group_blend_gate_receipt.json"

NODE_ID = "C1N73_GROUP_BLEND_GATE"
LANE = "L7"
PARENT_NODE = "C1N72_CURVE_BLEND"

MODEL_REFERENCE = 0.604043
CURVE_REFERENCE = 0.594183
GLOBAL_REFERENCE = 0.603794
GROUP_REFERENCE = 0.608974
TOLERANCE = 0.0005
ALPHA_GRID = tuple(round(0.1 * i, 1) for i in range(11))
GROUPS = (1, 2, 3)


def main() -> int:
    store, info = load_surface()
    c57 = json.loads(C57_RECEIPT.read_text(encoding="utf-8"))["result"]
    c60 = json.loads(C60_RECEIPT.read_text(encoding="utf-8"))
    assert info["probability_digest"] == c60["probability_digest"], "확률면 불일치"

    curves = {
        group: build_curve(
            [b for b in c57["per_group"][str(group)]["bins"] if b["rows"] >= MIN_ROWS]
        )
        for group in GROUPS
    }

    folds = sorted(store)
    prepared: dict[str, pd.DataFrame] = {}
    for fold in folds:
        entry = store[fold]
        temperature = np.full(
            len(entry["capacity"]), float(c60["chosen"]["global"][fold])
        )
        frame = entry["meta"].copy()
        frame["group_id"] = entry["group"]
        frame["capacity"] = entry["capacity"]
        frame["model_rate"] = bayes_decision(
            sharpen_by_row(entry["probability"], temperature)
        )
        curve_rate = np.zeros(len(frame), dtype="float64")
        for group, (cv, cp) in curves.items():
            mask = (frame["group_id"] == group).to_numpy()
            curve_rate[mask] = np.interp(
                entry["sitewind"][mask], cv, cp, left=0.0, right=cp[-1]
            )
        frame["curve_rate"] = curve_rate
        frame["month"] = frame["forecast_kst_dtm"].dt.to_period("M").astype(str)
        prepared[fold] = frame

    def scored(fold: str, alpha: Any) -> pd.DataFrame:
        frame = prepared[fold]
        a = (np.full(len(frame), float(alpha)) if np.isscalar(alpha)
             else np.asarray([alpha[int(g)] for g in frame["group_id"]], dtype="float64"))
        blended = a * frame["model_rate"] + (1.0 - a) * frame["curve_rate"]
        out = frame.loc[:, ["forecast_id", "forecast_kst_dtm", "group_id",
                            "actual_kwh", "month"]].copy()
        out["prediction_kwh"] = blended * frame["capacity"].to_numpy(float)
        return out

    def pooled(fold_list: list[str], alpha: Any) -> float:
        return float(official(
            pd.concat([scored(f, alpha) for f in fold_list], ignore_index=True)
        )["total"])

    endpoints = {}
    for name, alpha in (("model", 1.0), ("curve", 0.0)):
        frame = pd.concat([scored(f, alpha) for f in folds], ignore_index=True)
        endpoints[name] = {"frame": frame, "score": official(frame)}

    chosen: dict[str, dict[str, Any]] = {"global": {}, "group": {}}
    pieces: dict[str, list[pd.DataFrame]] = {"global": [], "group": []}
    for held in folds:
        others = [f for f in folds if f != held]
        best_a, best_score = ALPHA_GRID[0], -np.inf
        for alpha in ALPHA_GRID:
            score = pooled(others, alpha)
            if score > best_score:
                best_a, best_score = alpha, score
        chosen["global"][held] = float(best_a)
        pieces["global"].append(scored(held, float(best_a)))

        table = dict.fromkeys(GROUPS, float(best_a))
        incumbent = pooled(others, table)
        for group in GROUPS:
            current = table[group]
            for alpha in ALPHA_GRID:
                table[group] = float(alpha)
                score = pooled(others, table)
                if score > incumbent:
                    current, incumbent = float(alpha), score
            table[group] = current
        chosen["group"][held] = {str(k): v for k, v in table.items()}
        pieces["group"].append(scored(held, table))

    frames = {n: pd.concat(p, ignore_index=True) for n, p in pieces.items()}
    results = {n: official(frames[n]) for n in frames}

    v1 = bool(abs(endpoints["model"]["score"]["total"] - MODEL_REFERENCE) <= TOLERANCE)
    v2 = bool(abs(endpoints["curve"]["score"]["total"] - CURVE_REFERENCE) <= TOLERANCE)
    v3 = bool(
        abs(results["global"]["total"] - GLOBAL_REFERENCE) <= TOLERANCE
        and abs(results["group"]["total"] - GROUP_REFERENCE) <= TOLERANCE
    )

    gate = evaluate_gate(frames["group"], endpoints["model"]["frame"])
    gd = gate.evidence
    h1 = bool(gate.passed)
    flags = {label.split()[0]: bool(ok) for label, ok in gate.conditions.items()}
    signature = "[" + "".join("O" if flags[g] else "-" for g in sorted(flags)) + "]"

    stats = paired_monthly_delta(frames["group"], endpoints["model"]["frame"])
    monthly = pd.DataFrame(
        [{"month": m, "delta": part["total"], "ficr": part["ficr"],
          "nmae": part["one_minus_nmae"]}
         for m, part in stats["per_month"].items()]
    ).sort_values("delta")
    deltas = monthly["delta"].to_numpy(dtype="float64")
    h2 = bool(float(np.median(deltas)) > 0.0)
    h3 = bool(float(np.sort(deltas)[:-1].mean()) > 0.0)

    distinct = max(len(set(t.values())) for t in chosen["group"].values())
    h4 = bool(distinct >= 2)

    ficr_contrib = 0.5 * (
        results["group"]["ficr"] - endpoints["model"]["score"]["ficr"]
    )
    nmae_contrib = 0.5 * (
        results["group"]["one_minus_nmae"]
        - endpoints["model"]["score"]["one_minus_nmae"]
    )
    h5 = bool(ficr_contrib > nmae_contrib)

    if not (v1 and v2 and v3):
        verdict = "REPRODUCTION_FAILED_RESULT_VOID"
    elif h1 and h2 and h3:
        verdict = "GROUP_BLEND_IS_A_DEPLOYABLE_CANDIDATE"
    elif h1:
        verdict = "GATE_PASSED_BUT_GAIN_CONCENTRATED"
    else:
        verdict = "GROUP_BLEND_GAIN_NOT_MONTHLY_CONSISTENT"

    payload: dict[str, Any] = {
        "node_id": NODE_ID,
        "lane": LANE,
        "parent": PARENT_NODE,
        "gate_version": GATE_VERSION,
        "surface": info,
        "endpoints": {n: endpoints[n]["score"] for n in endpoints},
        "arms": results,
        "chosen": chosen,
        "checks": {
            "V1_model": v1, "V2_curve": v2, "V3_arms_reproduce_c72": v3,
            "model_total": endpoints["model"]["score"]["total"],
            "curve_total": endpoints["curve"]["score"]["total"],
        },
        "monthly": monthly.to_dict(orient="records"),
        "monthly_summary": {
            "mean": float(deltas.mean()), "median": float(np.median(deltas)),
            "min": float(deltas.min()), "max": float(deltas.max()),
            "drop_best_one": float(np.sort(deltas)[:-1].mean()),
            "drop_best_two": float(np.sort(deltas)[:-2].mean()),
            "positive": int((deltas > 0).sum()), "n": int(len(deltas)),
        },
        "gate": {
            "signature": signature, "flags": flags,
            "positive_months": int(gd["positive_months"]),
            "months_scored": int(gd["months_scored"]),
            "sign_test_p": float(gd["sign_test_p_greater"]),
            "median_delta": float(gd["median_total_delta"]),
            "bootstrap_q05": float(gd["block_bootstrap_q05"]),
            "min_delta": float(gd["min_total_delta"]),
        },
        "contributions": {"ficr": float(ficr_contrib), "nmae": float(nmae_contrib)},
        "hypotheses": {
            "H1_gate_passed": h1,
            "H2_median_positive": h2,
            "H3_survives_dropping_best_month": h3,
            "H4_alphas_differ_across_groups": h4,
            "H5_ficr_dominant": h5,
        },
        "deployable": True,
        "deployability_note": (
            "두 팔 모두 배포 가능한 입력만 쓴다 — teacher 풍속은 NWP 피처에서 나오고 "
            "실측 커브는 학습기간 적합 고정 함수다. `scada_ws` 미사용."
        ),
        "verdict": verdict,
        "dacon_upload": False,
        "external_actions": [],
        "lockbox_used": False,
        "generated_at": datetime.now(UTC).isoformat(),
    }
    payload["digest"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]
    RECEIPT.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    summary = payload["monthly_summary"]
    lines = [
        "# M271 P4 사이클 73 — 그룹별 커브 결합의 게이트",
        "",
        f"노드 `{NODE_ID}` / 레인 {LANE} / 부모 `{PARENT_NODE}`",
        "",
        f"확률면 prob `{info['probability_digest']}`. **`scada_ws` 미사용 — 배포 가능**",
        "",
        "## 1. 팔",
        "",
        "| 팔 | Total | 1-NMAE | FICR |",
        "|---|---:|---:|---:|",
        f"| MODEL (alpha=1) | {endpoints['model']['score']['total']:.6f} | "
        f"{endpoints['model']['score']['one_minus_nmae']:.6f} | "
        f"{endpoints['model']['score']['ficr']:.6f} |",
        f"| CURVE (alpha=0) | {endpoints['curve']['score']['total']:.6f} | "
        f"{endpoints['curve']['score']['one_minus_nmae']:.6f} | "
        f"{endpoints['curve']['score']['ficr']:.6f} |",
        f"| GLOBAL_ALPHA | {results['global']['total']:.6f} | "
        f"{results['global']['one_minus_nmae']:.6f} | {results['global']['ficr']:.6f} |",
        f"| **GROUP_ALPHA** | **{results['group']['total']:.6f}** | "
        f"{results['group']['one_minus_nmae']:.6f} | {results['group']['ficr']:.6f} |",
        "",
        "## 2. 선택된 alpha (fold-외, 모형 쪽 가중)",
        "",
        "```",
        json.dumps(chosen["group"], indent=1, ensure_ascii=False),
        "```",
        "",
        "## 3. 월별 델타 (GROUP_ALPHA - MODEL)",
        "",
        "| 월 | 델타 | FICR | 1-NMAE |",
        "|---|---:|---:|---:|",
    ]
    for row in monthly.itertuples():
        lines.append(
            f"| {row.month} | {row.delta:+.6f} | {row.ficr:+.6f} | {row.nmae:+.6f} |"
        )
    lines += [
        "",
        f"평균 {summary['mean']:+.6f} / 중앙값 {summary['median']:+.6f} / "
        f"양수 {summary['positive']}/{summary['n']}",
        "",
        f"최대 1 개월 제거 {summary['drop_best_one']:+.6f} / "
        f"2 개월 제거 {summary['drop_best_two']:+.6f}",
        "",
        "## 4. 타당성 가드",
        "",
        f"- V1 alpha=1 = MODEL {MODEL_REFERENCE} -> **{v1}**",
        f"- V2 alpha=0 = CURVE {CURVE_REFERENCE} -> **{v2}**",
        f"- V3 두 팔이 C72 재현 -> **{v3}**",
        "",
        "## 5. 사전확약",
        "",
        f"- H1 게이트 통과 -> **{h1}** {signature} "
        f"({gd['positive_months']}/{gd['months_scored']} 월, "
        f"p={gd['sign_test_p_greater']:.4f}, q05={gd['block_bootstrap_q05']:+.6f}, "
        f"최악월={gd['min_total_delta']:+.6f})",
        f"- H2 월별 중앙값 양수 -> **{h2}**",
        f"- H3 최대 1 개월 제거해도 양수 -> **{h3}**",
        f"- H4 그룹별 alpha 가 다르다 -> **{h4}**",
        f"- H5 FICR 우세 (FICR {ficr_contrib:+.6f} / 1-NMAE {nmae_contrib:+.6f}) -> **{h5}**",
        "",
        "## 6. 판정",
        "",
        f"**{verdict}**",
        "",
        payload["deployability_note"],
        "",
        f"digest `{payload['digest']}`",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("=== 완료 ===")
    print(f"[C73] V1 {v1} / V2 {v2} / V3 {v3}")
    print(f"[C73] MODEL {endpoints['model']['score']['total']:.6f} / CURVE "
          f"{endpoints['curve']['score']['total']:.6f} / GLOBAL "
          f"{results['global']['total']:.6f} / **GROUP {results['group']['total']:.6f}**")
    print(f"[C73] alpha {json.dumps(chosen['group'], ensure_ascii=False)}")
    print(f"[C73] 월별  평균 {summary['mean']:+.6f} / 중앙값 {summary['median']:+.6f} / "
          f"양수 {summary['positive']}/{summary['n']}")
    print(f"[C73] 최대1개월 제거 {summary['drop_best_one']:+.6f} / "
          f"2개월 {summary['drop_best_two']:+.6f}")
    print(f"[C73] H1 게이트 {h1} {signature} / H2 {h2} / H3 {h3} / H4 {h4} / H5 {h5}")
    print(f"[C73] 판정: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
