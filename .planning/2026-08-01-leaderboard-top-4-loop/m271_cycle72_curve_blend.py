"""M271 P4 사이클 72 — 커브 직독이 파이프라인을 이긴다: 두 예측을 섞는다.

사이클 69 가 같은 행에서 둘을 채점했다.

    MODEL          Total 0.604043   1-NMAE 0.856870   FICR 0.351216
    CURVE_TEACHER  Total 0.612331   1-NMAE 0.856354   FICR 0.368308
    차                  **+0.008288**       -0.000516      **+0.017091**

"실측 파워커브를 teacher 풍속에 그냥 적용" 이 GBM + 46 구간 분포 + Bayes 결정층 +
온도보정을 다 얹은 것보다 **높다**. 조건부 평균은 비기고 차이는 전부 FICR 이다.

이건 모순처럼 보인다 — Bayes 결정은 **모형 자신의 분포 하에서** 기대 정산단위를
최대화하므로, 분포가 잘 보정돼 있으면 어떤 다른 규칙도 이길 수 없다. 졌다는 것은
분포가 오보정이라는 뜻이고, C44·C60 이 전역 T 로 격자 최대치 2.2(최대 평탄화)를 고른
것이 같은 증상이다. 평탄화를 끝까지 밀어도 부족했다.

**C56 이 이것을 놓친 이유도 분명하다.** C56 은 실측 커브를 **피처로** 넣어 -0.000039 를
얻었다. 커브를 피처로 갖는 것과 **예측으로 직접 쓰는 것**은 다르다 — 전자는 결정층을
통과하고 후자는 우회한다.

둘 다 배포 가능하다. teacher 풍속은 NWP 피처만으로 만들고, 실측 커브는 학습기간에서
적합한 고정 함수다(평가기간 데이터가 아니다).

**① 방법 리서치**

  - 두 예측을 섞는 표준은 **수축 결합**이고 이 프로젝트가 이미 검증했다 —
    C1N14 `SHRINKBLEND`, C1N20 `ALPHA_ENDPOINT` 가 같은 사다리를 썼다.
  - 가중은 **fold-외**로 고른다(C44 가 온도에서, C71 이 teacher 가중에서 쓴 절차).
    평가하는 fold 에서 고르면 표본내 최적이라 이득이 부풀려진다.
  - Bates & Granger(1969) 이래 예측 결합의 표준 결과는 "결합이 최선 단일을 이긴다" 이나,
    **C1N34 가 이 프로젝트에서 그것이 성립하지 않는 후보군을 이미 봤다**
    (`NOTHING_TO_ADD_TO_BEST_SINGLE`). 그래서 H1 을 예단하지 않는다.
  - **채택**: fold-외 alpha 사다리. 새 런타임·새 데이터 없음. 캐시만 읽는다.

**② 사양 동결**

  입력   확률면 캐시 v3. `probability`(MODEL 용), `sitewind`(teacher 풍속),
         `capacity`, `actual_kwh`. **`scada_ws` 미사용** — 이 노드의 두 팔은
         모두 배포 가능해야 하므로 관측 나셀풍속이 들어가면 안 된다.
  커브   C57 실측 커브 + C67 동결 cut-in `(0,0),(3,0)`. 그룹별. 위쪽 평탄.
  MODEL  C60 GLOBAL fold-외 T 로 sharpen 후 Bayes 결정. C64~C69 와 동일.
  섞기   `pred(alpha) = alpha * MODEL + (1 - alpha) * CURVE_TEACHER` (정격비 공간).
         `alpha` 격자 = 0.0, 0.1, ..., 1.0 — **실행 전 동결**.
  선택   fold-외. 보류 fold 의 alpha 는 나머지 두 fold 의 pooled Total 을 최대화.
  팔     GLOBAL_ALPHA (alpha 하나) / GROUP_ALPHA (그룹별 alpha 3 개)
         두 번째는 C60 의 LEVELGROUP 이 과적합으로 진 전례가 있어 대조군이다.

  **타당성 가드**
    V1  `alpha = 1` 고정이 MODEL 0.604043 을 ±0.0005 로 재현.
    V2  `alpha = 0` 고정이 CURVE_TEACHER 0.612331 을 ±0.0005 로 재현.
        두 끝점이 맞아야 사다리 중간을 믿을 수 있다.

  사전확약 (V1·V2 통과시에만 판정):
    H1  fold-외 GLOBAL_ALPHA 가 **두 끝점을 모두** 넘는다. 부호 예단 없음 —
        C1N34 전례가 있어 결합이 최선단일에 못 더할 수 있다.
    H2  GLOBAL_ALPHA 가 MODEL 대비 **동결 게이트 통과**.
    H3  선택된 alpha 가 0.5 미만(커브 쪽). 커브가 더 나으므로 그쪽에 무게가
        실려야 앞뒤가 맞는다.
    H4  이득이 **FICR** 쪽에서 우세.
    H5  GROUP_ALPHA 가 GLOBAL_ALPHA 를 넘지 **못한다**. C60 LEVELGROUP 의
        과적합 서명이 재현되는지 본다. 넘으면 그룹 자유도에 실체가 있는 것이다.

  **이 노드는 승격 판정을 낼 수 있다.** 두 팔 모두 배포 가능한 입력만 쓰고
  fold-외 선택이므로, 게이트를 통과하면 후보가 된다. 다만 챔피언 로컬 0.630310 은
  **다른 표면**(M115 고정정책)이므로 이 노드의 절대값과 직접 비교하지 않는다 —
  C33·C45 에서 두 번 틀린 비교다.

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
REPORT_MD = REPORTS / "m271_cycle72_curve_blend.md"
RECEIPT = REPORTS / "m271_cycle72_curve_blend_receipt.json"

NODE_ID = "C1N72_CURVE_BLEND"
LANE = "L7"
PARENT_NODE = "C1N69_SKILL_RESPONSE"

MODEL_REFERENCE = 0.604043
CURVE_REFERENCE = 0.612331
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
        sitewind = entry["sitewind"]
        for group, (cv, cp) in curves.items():
            mask = (frame["group_id"] == group).to_numpy()
            curve_rate[mask] = np.interp(
                sitewind[mask], cv, cp, left=0.0, right=cp[-1]
            )
        frame["curve_rate"] = curve_rate
        frame["month"] = frame["forecast_kst_dtm"].dt.to_period("M").astype(str)
        prepared[fold] = frame

    def scored(fold: str, alpha: Any) -> pd.DataFrame:
        frame = prepared[fold]
        if np.isscalar(alpha):
            a = np.full(len(frame), float(alpha))
        else:
            a = np.asarray(
                [alpha[int(g)] for g in frame["group_id"]], dtype="float64"
            )
        blended = a * frame["model_rate"] + (1.0 - a) * frame["curve_rate"]
        out = frame.loc[:, ["forecast_id", "forecast_kst_dtm", "group_id",
                            "actual_kwh", "month"]].copy()
        out["prediction_kwh"] = blended * frame["capacity"].to_numpy(float)
        return out

    def pooled(fold_list: list[str], alpha: Any) -> float:
        frame = pd.concat([scored(f, alpha) for f in fold_list], ignore_index=True)
        return float(official(frame)["total"])

    endpoints = {}
    for name, alpha in (("model", 1.0), ("curve", 0.0)):
        frame = pd.concat([scored(f, alpha) for f in folds], ignore_index=True)
        endpoints[name] = {"frame": frame, "score": official(frame)}
    v1 = bool(abs(endpoints["model"]["score"]["total"] - MODEL_REFERENCE) <= TOLERANCE)
    v2 = bool(abs(endpoints["curve"]["score"]["total"] - CURVE_REFERENCE) <= TOLERANCE)

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

    frames = {name: pd.concat(parts, ignore_index=True)
              for name, parts in pieces.items()}
    results = {name: official(frames[name]) for name in frames}

    h1 = bool(
        results["global"]["total"] > endpoints["model"]["score"]["total"]
        and results["global"]["total"] > endpoints["curve"]["score"]["total"]
    )
    gate = evaluate_gate(frames["global"], endpoints["model"]["frame"])
    gd = gate.evidence
    h2 = bool(gate.passed)
    flags = {label.split()[0]: bool(ok) for label, ok in gate.conditions.items()}
    signature = "[" + "".join("O" if flags[g] else "-" for g in sorted(flags)) + "]"

    mean_alpha = float(np.mean(list(chosen["global"].values())))
    h3 = bool(mean_alpha < 0.5)

    ficr_contrib = 0.5 * (
        results["global"]["ficr"] - endpoints["model"]["score"]["ficr"]
    )
    nmae_contrib = 0.5 * (
        results["global"]["one_minus_nmae"]
        - endpoints["model"]["score"]["one_minus_nmae"]
    )
    h4 = bool(ficr_contrib > nmae_contrib)
    h5 = bool(results["group"]["total"] <= results["global"]["total"])

    if not v1 or not v2:
        verdict = "ENDPOINT_REPRODUCTION_FAILED_RESULT_VOID"
    elif h1 and h2:
        verdict = "BLEND_BEATS_BOTH_ENDPOINTS_AND_PASSES_GATE"
    elif h1:
        verdict = "BLEND_BEATS_BOTH_BUT_GATE_REJECTS"
    elif results["global"]["total"] > endpoints["model"]["score"]["total"]:
        verdict = "CURVE_DOMINATES_BLEND_ADDS_NOTHING"
    else:
        verdict = "BLEND_DOES_NOT_HELP"

    payload: dict[str, Any] = {
        "node_id": NODE_ID,
        "lane": LANE,
        "parent": PARENT_NODE,
        "gate_version": GATE_VERSION,
        "method": "SHRINK_BLEND fold-out (Bates & Granger 1969; C1N14/C1N20 계보)",
        "surface": info,
        "alpha_grid": list(ALPHA_GRID),
        "endpoints": {name: endpoints[name]["score"] for name in endpoints},
        "arms": results,
        "chosen": chosen,
        "mean_alpha": mean_alpha,
        "checks": {
            "V1_model_endpoint": v1,
            "V1_gap": abs(endpoints["model"]["score"]["total"] - MODEL_REFERENCE),
            "V2_curve_endpoint": v2,
            "V2_gap": abs(endpoints["curve"]["score"]["total"] - CURVE_REFERENCE),
        },
        "hypotheses": {
            "H1_blend_beats_both": h1,
            "H2_gate_passed": h2,
            "H3_alpha_below_half": h3,
            "H4_ficr_dominant": h4,
            "H5_group_alpha_does_not_help": h5,
        },
        "contributions": {"ficr": float(ficr_contrib), "nmae": float(nmae_contrib)},
        "gate": {
            "signature": signature, "flags": flags,
            "positive_months": int(gd["positive_months"]),
            "months_scored": int(gd["months_scored"]),
            "sign_test_p": float(gd["sign_test_p_greater"]),
            "median_delta": float(gd["median_total_delta"]),
            "bootstrap_q05": float(gd["block_bootstrap_q05"]),
            "min_delta": float(gd["min_total_delta"]),
        },
        "deployable": True,
        "deployability_note": (
            "두 팔 모두 배포 가능한 입력만 쓴다 — teacher 풍속은 NWP 피처에서 나오고 "
            "실측 커브는 학습기간에서 적합한 고정 함수다. `scada_ws` 미사용."
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

    lines = [
        "# M271 P4 사이클 72 — 커브 직독과 모형의 수축 결합",
        "",
        f"노드 `{NODE_ID}` / 레인 {LANE} / 부모 `{PARENT_NODE}`",
        "",
        f"확률면 prob `{info['probability_digest']}`. **`scada_ws` 미사용 — 두 팔 모두 배포 가능**",
        "",
        "## 1. 끝점과 결합",
        "",
        "| 팔 | Total | 1-NMAE | FICR |",
        "|---|---:|---:|---:|",
        f"| MODEL (alpha=1) | {endpoints['model']['score']['total']:.6f} | "
        f"{endpoints['model']['score']['one_minus_nmae']:.6f} | "
        f"{endpoints['model']['score']['ficr']:.6f} |",
        f"| CURVE (alpha=0) | {endpoints['curve']['score']['total']:.6f} | "
        f"{endpoints['curve']['score']['one_minus_nmae']:.6f} | "
        f"{endpoints['curve']['score']['ficr']:.6f} |",
        f"| **GLOBAL_ALPHA** | **{results['global']['total']:.6f}** | "
        f"{results['global']['one_minus_nmae']:.6f} | {results['global']['ficr']:.6f} |",
        f"| GROUP_ALPHA | {results['group']['total']:.6f} | "
        f"{results['group']['one_minus_nmae']:.6f} | {results['group']['ficr']:.6f} |",
        "",
        "## 2. 선택된 alpha (fold-외, 모형 쪽 가중)",
        "",
        "```",
        json.dumps(chosen, indent=1, ensure_ascii=False),
        "```",
        "",
        "## 3. 타당성 가드",
        "",
        f"- V1 alpha=1 이 MODEL {MODEL_REFERENCE} 재현 -> **{v1}**",
        f"- V2 alpha=0 이 CURVE {CURVE_REFERENCE} 재현 -> **{v2}**",
        "",
        "## 4. 사전확약",
        "",
        f"- H1 결합이 두 끝점을 모두 넘는다 -> **{h1}**",
        f"- H2 게이트 통과 -> **{h2}** {signature} "
        f"({gd['positive_months']}/{gd['months_scored']} 월, "
        f"p={gd['sign_test_p_greater']:.4f}, q05={gd['block_bootstrap_q05']:+.6f}, "
        f"최악월={gd['min_total_delta']:+.6f})",
        f"- H3 평균 alpha {mean_alpha:.2f} < 0.5 -> **{h3}**",
        f"- H4 FICR 우세 (FICR {ficr_contrib:+.6f} / 1-NMAE {nmae_contrib:+.6f}) -> **{h4}**",
        f"- H5 GROUP_ALPHA 가 GLOBAL 을 못 넘는다 -> **{h5}**",
        "",
        "## 5. 판정",
        "",
        f"**{verdict}**",
        "",
        payload["deployability_note"],
        "",
        "챔피언 로컬 0.630310 은 **다른 표면**(M115 고정정책)이므로 이 노드의 절대값과 "
        "직접 비교하지 않는다 — C33·C45 에서 두 번 틀린 비교다.",
        "",
        f"digest `{payload['digest']}`",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("=== 완료 ===")
    print(f"[C72] V1 {v1} / V2 {v2}")
    print(f"[C72] MODEL  {endpoints['model']['score']['total']:.6f}  "
          f"CURVE  {endpoints['curve']['score']['total']:.6f}")
    print(f"[C72] GLOBAL_ALPHA {results['global']['total']:.6f}  "
          f"(1-NMAE {results['global']['one_minus_nmae']:.6f} / "
          f"FICR {results['global']['ficr']:.6f})")
    print(f"[C72] GROUP_ALPHA  {results['group']['total']:.6f}")
    print(f"[C72] 선택 alpha {chosen['global']} (평균 {mean_alpha:.2f})")
    print(f"[C72] H1 {h1} / H2 게이트 {h2} {signature} "
          f"{gd['positive_months']}/{gd['months_scored']}월 / H3 {h3} / H4 {h4} / H5 {h5}")
    print(f"[C72] 판정: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
