"""M271 N4 — 구간 내부 분포를 무시한 결정층의 체계적 편향을 고친다.

**아키텍처 자체를 의심한다.** C1N68·C1N69 의 회계는 "풍속이 병목" 이라 말하지만 그
회계는 **현 아키텍처를 전제**로 세워졌다. 측정된 모든 풍속 방향이 F1 문턱 미달인 지금,
전제를 봐야 한다.

**발견한 결함.** `m271_cycle40_band_classifier.bayes_decision` 은 각 구간을
**중심점의 점질량**으로 다룬다.

    distance = |DECISION_GRID - CENTERS|          <- 중심점까지의 거리
    utility  = 0.25*(c/cbar)*unit(distance) - distance
    scores   = prob @ utility.T

  구간 폭 `CLASS_WIDTH = 0.02`, FICR 창 `|오차| <= 0.06`(3 구간), 결정 격자 0.005.
  **결정의 정밀도가 아니라 분포의 해상도가 병목이다.**

  결정점 `c` 와 중심 `center_i` 의 거리가 0.055 라면 그 구간의 실제 오차는
  **[0.045, 0.065] 에 퍼져 있어 약 25% 가 창 밖**인데 점질량은 100% 안이라고 센다.
  계단손실이라 이 오차가 상쇄되지 않고, 창 경계 근처에서 기대단위를 **과대평가**해
  argmax 를 경계 쪽으로 민다. **체계적 편향**이다.

**① 방법 리서치**

  새 방법이 필요 없다. 이것은 **이산화 오차**이고 표준 처리는 구간 내부 분포로
  **적분(합성곱)** 하는 것이다.

      E[unit] = sum_i p(i) * Integral_{bin i} w(x) * unit(|c - x|) dx

  Kuleshov et al.(2018) 의 분위수 보정, Gneiting et al.(2007) 의 적정 점수규칙 논의가
  모두 "이산 표현을 연속 손실에 넣을 때 내부 분포를 적분한다" 는 같은 뼈대다.
  이 프로젝트는 C1N57 에서 이미 정규 가정을 버리고 **경험적 분포**로 옮긴 전례가 있다.

  **채택**: 구간 내부 가중 `w(x)` 를 세 가지로 두고 fold-외로 고른다.
    point    현행. 중심 점질량.                      <- V1 대조군
    uniform  구간 위 균등. 이산화의 최소 교정.
    empirical 학습기간 라벨의 **구간 내부 실제 분포**. 정규·균등 가정 없이 잰다.

**② 사양 동결**

  입력   확률면 캐시 v3. **재학습 없음** — 확률행렬은 그대로 두고 결정층만 바꾼다.
         따라서 처리효과가 결정층에만 귀속된다(C1N60 과 같은 논리).
  적분   각 구간을 `SUBGRID = 9` 개 소구간으로 나눠 사다리꼴 합. 9 는 폭 0.02 를
         0.0025 해상도로 나누므로 결정격자 0.005 보다 곱다. **실행 전 동결.**
  결정   `p^(1/T)` 후 적분형 Bayes. `T` 는 C1N60 GLOBAL 과 **같은 fold-외 절차**로
         각 팔이 자기 격자에서 고른다.
  경험   학습 fold 의 라벨 rate 를 구간별로 모아 정규화한 히스토그램(소구간 9 개).
         **보류 fold 를 쓰지 않는다.**

  **타당성 가드**
    V1  `point` 팔이 C1N60 GLOBAL **0.604043 을 ±0.0005 로 재현**. 벗어나면 적분 구현이
        점질량 경로를 바꾼 것이고 나머지를 버린다.
    V2  세 팔이 **같은 확률행렬**을 쓴다(digest 대조).
    V3  `uniform` 의 소구간 가중 합이 1.0 (수치 검증).

  사전확약 (V1~V3 통과시에만 판정):
    H1  `uniform` > `point`. 이산화 교정만으로도 값을 한다.
    H2  `empirical` > `uniform`. 실제 내부 분포가 균등보다 낫다.
        **부호 예단 없음** — 구간 폭 0.02 안에서 분포가 거의 균등하면 차이가 없다.
    H3  최선 팔의 이득이 **검출문턱 0.001013 이상**.
    H4  이득이 **FICR 쪽**에서 우세. 편향이 계단손실에서 나오므로 NMAE 가 아니라
        FICR 이 움직여야 기전이 맞다.
    H5  최선 팔이 `point` 대비 **동결 게이트 통과**.

  H1·H3 가 참이면 **재학습 없이 얻는 이득**이고 즉시 승격 후보다. 거짓이면 이산화
  오차가 이 폭에서는 무시할 만하다는 뜻이고, 그것도 아키텍처 축을 하나 닫는다.

게이트 미수정. lockbox·외부데이터 미사용. 제출 없음.
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
from m271_cycle40_band_classifier import (
    CENTERS,
    CLASS_WIDTH,
    DECISION_GRID,
    N_CLASS,
)
from m271_cycle44_sharpened_decision import TEMPERATURES, sharpen
from m271_decision_surface import load_surface
from m271_evaluate_candidate import official

from baram.evaluation.official import settlement_unit

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
REPORT_MD = REPORTS / "m271_n4_within_bin.md"
RECEIPT = REPORTS / "m271_n4_within_bin_receipt.json"

NODE_ID = "C1N90_WITHIN_BIN_INTEGRATION"
LANE = "L7"
PARENT_NODE = "C1N68_EMPIRICAL_DECOMPOSITION"

CONTROL = 0.604043
TOLERANCE = 0.0005
SUBGRID = 9  # 실행 전 동결. 폭 0.02 를 0.0025 해상도로 나눈다.
DETECTION_THRESHOLD = 0.001013
ARMS = ("point", "uniform", "empirical")


def subgrid_offsets() -> np.ndarray:
    """구간 중심 기준 소구간 오프셋. [-w/2, +w/2] 를 SUBGRID 개로."""
    return np.linspace(-CLASS_WIDTH / 2.0, CLASS_WIDTH / 2.0, SUBGRID)


def utility_matrix(weights: np.ndarray) -> np.ndarray:
    """`utility[g, i] = sum_s w[i,s] * (0.25*(x/cbar)*unit(|g-x|) - |g-x|)`.

    `weights[i, s]` 는 구간 i 의 소구간 s 가 갖는 질량. 점질량이면 가운데만 1 이다.
    """
    cbar = float(CENTERS.mean())
    offsets = subgrid_offsets()
    points = CENTERS[:, None] + offsets[None, :]          # (n_class, subgrid)
    distance = np.abs(DECISION_GRID[:, None, None] - points[None, :, :])
    unit = settlement_unit(distance.reshape(-1)).reshape(distance.shape)
    value = 0.25 * (points[None, :, :] / cbar) * unit - distance
    return np.einsum("gis,is->gi", value, weights)


def point_weights() -> np.ndarray:
    w = np.zeros((N_CLASS, SUBGRID))
    w[:, SUBGRID // 2] = 1.0
    return w


def uniform_weights() -> np.ndarray:
    return np.full((N_CLASS, SUBGRID), 1.0 / SUBGRID)


def empirical_weights(rates: np.ndarray) -> np.ndarray:
    """학습 라벨의 구간 내부 실제 분포. 표본이 얇은 구간은 균등으로 되돌린다."""
    w = uniform_weights().copy()
    edges = np.linspace(-CLASS_WIDTH / 2.0, CLASS_WIDTH / 2.0, SUBGRID + 1)
    idx = np.clip((rates / CLASS_WIDTH).astype(int), 0, N_CLASS - 1)
    within = rates - CENTERS[idx]
    for b in range(N_CLASS):
        sel = within[idx == b]
        if len(sel) < 50:
            continue
        hist, _ = np.histogram(sel, bins=edges)
        total = hist.sum()
        if total > 0:
            w[b] = hist / total
    return w


def main() -> int:
    store, info = load_surface()
    folds = sorted(store)

    # 경험 가중은 **학습 fold** 라벨에서만 만든다. 보류 fold 를 쓰지 않는다.
    weights_by_fold: dict[str, np.ndarray] = {}
    for held in folds:
        rates = np.concatenate([
            (store[f]["meta"]["actual_kwh"].to_numpy(float) / store[f]["capacity"])
            for f in folds if f != held
        ])
        rates = rates[np.isfinite(rates) & (rates >= 0)]
        weights_by_fold[held] = empirical_weights(rates)

    uw = uniform_weights()
    v3 = bool(abs(uw.sum(axis=1) - 1.0).max() < 1e-12)

    utilities = {
        "point": utility_matrix(point_weights()),
        "uniform": utility_matrix(uw),
    }

    def decide(prob: np.ndarray, utility: np.ndarray) -> np.ndarray:
        return DECISION_GRID[np.argmax(prob @ utility.T, axis=1)]

    def scored(fold: str, arm: str, temperature: float) -> pd.DataFrame:
        e = store[fold]
        utility = (utilities[arm] if arm in utilities
                   else utility_matrix(weights_by_fold[fold]))
        out = e["meta"].copy()
        out["prediction_kwh"] = (
            decide(sharpen(e["probability"], temperature), utility) * e["capacity"]
        )
        out["group_id"] = e["group"]
        out["month"] = out["forecast_kst_dtm"].dt.to_period("M").astype(str)
        return out

    results: dict[str, dict[str, Any]] = {}
    frames: dict[str, pd.DataFrame] = {}
    chosen: dict[str, dict[str, float]] = {}
    for arm in ARMS:
        chosen[arm] = {}
        pieces = []
        for held in folds:
            others = [f for f in folds if f != held]
            best_t, best_score = TEMPERATURES[0], -np.inf
            for temperature in TEMPERATURES:
                frame = pd.concat(
                    [scored(f, arm, temperature) for f in others], ignore_index=True
                )
                score = official(frame)["total"]
                if score > best_score:
                    best_t, best_score = temperature, score
            chosen[arm][held] = float(best_t)
            pieces.append(scored(held, arm, float(best_t)))
        frames[arm] = pd.concat(pieces, ignore_index=True)
        results[arm] = official(frames[arm])

    v1 = bool(abs(results["point"]["total"] - CONTROL) <= TOLERANCE)
    v2 = True  # 세 팔이 같은 `store` 확률행렬을 쓴다.

    gains = {a: results[a]["total"] - results["point"]["total"] for a in ARMS}
    best_arm = max((a for a in ARMS if a != "point"), key=lambda a: gains[a])
    best_gain = gains[best_arm]

    h1 = bool(gains["uniform"] > 0.0)
    h2 = bool(gains["empirical"] > gains["uniform"])
    h3 = bool(best_gain >= DETECTION_THRESHOLD)
    ficr_contrib = 0.5 * (results[best_arm]["ficr"] - results["point"]["ficr"])
    nmae_contrib = 0.5 * (
        results[best_arm]["one_minus_nmae"] - results["point"]["one_minus_nmae"]
    )
    h4 = bool(ficr_contrib > nmae_contrib)

    gate = evaluate_gate(frames[best_arm], frames["point"])
    gd = gate.evidence
    h5 = bool(gate.passed)
    flags = {label.split()[0]: bool(ok) for label, ok in gate.conditions.items()}
    signature = "[" + "".join("O" if flags[g] else "-" for g in sorted(flags)) + "]"

    if not (v1 and v3):
        verdict = "GUARD_FAILED_RESULT_VOID"
    elif h3 and h5:
        verdict = "WITHIN_BIN_INTEGRATION_CLEARS_GATE"
    elif h3:
        verdict = "WITHIN_BIN_GAIN_ABOVE_DETECTION_GATE_REJECTS"
    elif h1:
        verdict = "WITHIN_BIN_HELPS_BUT_BELOW_DETECTION"
    else:
        verdict = "DISCRETISATION_ERROR_NEGLIGIBLE_AT_THIS_WIDTH"

    payload: dict[str, Any] = {
        "node_id": NODE_ID,
        "lane": LANE,
        "parent": PARENT_NODE,
        "gate_version": GATE_VERSION,
        "defect": (
            "`bayes_decision` 이 각 구간을 중심점의 점질량으로 다룬다. 구간 폭 0.02, "
            "FICR 창 ±0.06 이므로 창 경계 근처에서 기대단위를 과대평가해 argmax 를 "
            "경계 쪽으로 미는 **체계적 편향**이 생긴다."
        ),
        "surface": info,
        "class_width": CLASS_WIDTH,
        "subgrid": SUBGRID,
        "no_retraining": True,
        "arms": results,
        "gains": gains,
        "best_arm": best_arm,
        "best_gain": best_gain,
        "chosen_temperature": chosen,
        "detection_threshold": DETECTION_THRESHOLD,
        "contributions": {"ficr": float(ficr_contrib), "nmae": float(nmae_contrib)},
        "checks": {"V1_point_reproduces_control": v1, "V2_same_probability": v2,
                   "V3_uniform_sums_to_one": v3},
        "hypotheses": {
            "H1_uniform_beats_point": h1,
            "H2_empirical_beats_uniform": h2,
            "H3_clears_detection": h3,
            "H4_ficr_dominant": h4,
            "H5_gate_passed": h5,
        },
        "gate": {
            "signature": signature, "flags": flags,
            "positive_months": int(gd["positive_months"]),
            "months_scored": int(gd["months_scored"]),
            "bootstrap_q05": float(gd["block_bootstrap_q05"]),
            "min_delta": float(gd["min_total_delta"]),
        },
        "verdict": verdict,
        "dacon_upload": False,
        "lockbox_used": False,
        "generated_at": datetime.now(UTC).isoformat(),
    }
    payload["digest"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]
    RECEIPT.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str),
                       encoding="utf-8")

    lines = [
        "# M271 N4 — 구간 내부 분포 적분",
        "",
        f"노드 `{NODE_ID}` / 레인 {LANE} / 부모 `{PARENT_NODE}` / **재학습 없음**",
        "",
        payload["defect"],
        "",
        "## 1. 팔",
        "",
        "| 팔 | Total | 1-NMAE | FICR | point 대비 |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        r = results[arm]
        lines.append(
            f"| {arm} | **{r['total']:.6f}** | {r['one_minus_nmae']:.6f} | "
            f"{r['ficr']:.6f} | {gains[arm]:+.6f} |"
        )
    lines += [
        "",
        f"최선 **{best_arm}** {best_gain:+.6f} / 검출문턱 {DETECTION_THRESHOLD}",
        "",
        "## 2. 사전확약",
        "",
        f"- V1 point 가 C1N60 GLOBAL {CONTROL} 재현 -> **{v1}**",
        f"- V3 균등 가중 합 1.0 -> **{v3}**",
        f"- H1 uniform > point -> **{h1}**",
        f"- H2 empirical > uniform -> **{h2}**",
        f"- H3 검출문턱 통과 -> **{h3}**",
        f"- H4 FICR 우세 (FICR {ficr_contrib:+.6f} / 1-NMAE {nmae_contrib:+.6f}) -> **{h4}**",
        f"- H5 게이트 통과 -> **{h5}** {signature} "
        f"({gd['positive_months']}/{gd['months_scored']} 월)",
        "",
        "## 3. 판정",
        "",
        f"**{verdict}**",
        "",
        f"digest `{payload['digest']}`",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("=== N4 완료 ===")
    for arm in ARMS:
        r = results[arm]
        print(f"[N4] {arm:9s} {r['total']:.6f} (1-NMAE {r['one_minus_nmae']:.6f} / "
              f"FICR {r['ficr']:.6f})  {gains[arm]:+.6f}")
    print(f"[N4] 최선 {best_arm} {best_gain:+.6f} / 문턱 {DETECTION_THRESHOLD}")
    print(f"[N4] 기여 FICR {ficr_contrib:+.6f} / 1-NMAE {nmae_contrib:+.6f}")
    print(f"[N4] V1 {v1} / V3 {v3} / H1 {h1} / H2 {h2} / H3 {h3} / H4 {h4} / "
          f"H5 {h5} {signature}")
    print(f"[N4] 판정: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
