"""M271 N5 — 손실 축 재개: C1N43 의 폐쇄 전제를 현 대조군에서 직접 시험한다.

**손실 축은 닫힌 것이 아니라 미판정이다.**

    C1N37  구현 결함(학습 모집단 오설정 + 헤시안 폭주)으로 **무효**
    C1N38  D1·D2 교정했으나 **V1 대조군 가드 발화**로 손실 축 미판정
    C1N39  구조 격차 0.086202 실측. "손실 축은 **미판정**" 이라 명시
    C1N43  처리효과 vs 대조군 품질 r=-0.9922(**n=3**) 외삽으로 폐쇄

**C1N43 의 폐쇄가 가장 약하다.** 스스로 이렇게 기록해 뒀다 —

    "이 폐쇄는 3 점 외삽에 기대며 **관측 범위 밖**이다. 배포 품질 대조군에서 양의
     효과가 한 번이라도 관측되면 뒤집힌다."
    부활 술어: `positive_band_effect_at_frontier_control`

  그 3 점은 대조군 품질 **0.5679 / 0.5845 / 0.5959** 에서 쟀다. 현 대조군은
  **0.604043** 으로 그 범위 **밖**이다. 여기서 재는 것은 외삽이 아니라 **관측**이고,
  C1N43 이 명시한 부활 조건을 정면으로 검정한다.

**왜 지금 이 축인가.** 상위권 1-NMAE 0.87964 대 우리 0.85789 로 격차 0.0218 이고,
Total 격차의 29% 다. 우리 목적함수는 **순수 교차엔트로피**다 —

    grad = p - target ;  hess = max(2*p*(1-p), 1e-6)

소프트 표적이 밴드 모양이라 분포는 밴드를 닮게 학습되지만, **손실 자체는 공식 지표와
무관**하다. 절대오차도 정산단위도 목적함수에 없다.

**① 방법 리서치**

  이것은 **손실-지표 정합**(loss-metric alignment) 문제다. 표준 처리는 지표의 대리
  손실을 쓰는 것이고, 계단형 지표에서는 **기대 효용을 직접 최대화**하는 것이 옳다.
  Gneiting & Raftery(2007)의 적정 점수규칙 논의가 뼈대이며, 이 프로젝트는 결정층에서
  이미 그 기대 효용을 쓰고 있다(`bayes_decision` 의 `utility`). **학습에는 안 쓴다.**

  **채택**: 결정층이 쓰는 **바로 그 효용**을 학습 목적함수에 넣는다. 새 개념이 아니라
  이미 있는 함수를 반대편에도 쓰는 것이다.

      L(p) = -sum_i p(i) * E_util(i)   (기대 효용의 음수)
      E_util(i) = 라벨 rate 에서 본 구간 i 의 효용 = 0.25*(y/cbar)*unit(|c_i - y|) - |c_i - y|

  여기서 `c_i` 는 구간 중심, `y` 는 실제 rate. 즉 **각 구간을 예측했을 때 얻을 효용**을
  라벨로부터 계산해 그 기대값을 최대화한다. 그러면 grad = -E_util 이고 hess 는
  상수로 둘 수 없으므로(C1N37 의 D2 결함) **교차엔트로피 헤시안을 유지**한다.

**② 사양 동결**

  하네스   C1N60 과 **동일**(teacher 복원, generic 기저, leaves 15, lr 0.1, 200 rounds).
           바뀌는 것은 **목적함수 하나**다.
  팔
    ce        현행 순수 교차엔트로피.                       <- V1 대조군 0.604043
    util      `alpha * CE + (1-alpha) * (-기대효용)`, alpha=0.5
    util_soft `alpha=0.8` (CE 쪽으로 치우침, 정규화 유지)
  혼합     순수 효용 손실은 분포를 붕괴시킬 수 있으므로 CE 와 섞는다. `alpha` 두 값을
           **실행 전 동결**하고 결과를 보고 늘리지 않는다.
  헤시안   **교차엔트로피 헤시안을 그대로 쓴다.** C1N37 이 상수 헤시안(1.0)에 기울기가
           100 배 큰 조합으로 폭주했다(D2). 같은 실수를 반복하지 않는다.
  결정층   C1N60 GLOBAL fold-외 T. 세 팔에 동일.

  **타당성 가드**
    V1  `ce` 가 C1N60 GLOBAL **0.604043 을 ±0.0005 로 재현**. C1N38 이 여기서 발화해
        미판정이 됐으므로 이번엔 하네스를 C1N60 과 동일하게 맞췄다.
    V2  세 팔의 기울기 규모가 같은 자릿수 — `|grad|` 평균 비가 0.1~10 배.
        벗어나면 C1N37 의 D2 재발이다.
    V3  모든 팔의 확률행렬이 정상(행합 1, NaN 없음).

  사전확약 (V1~V3 통과시에만 판정):
    H1  `util` 또는 `util_soft` 가 `ce` 를 넘는다. **이것이 C1N43 의 부활 술어
        `positive_band_effect_at_frontier_control` 를 정면으로 검정한다.**
    H2  이득이 **1-NMAE 쪽**에서 나온다. 효용에 `-|c-y|` 항이 있으므로 절대오차가
        움직여야 기전이 맞다. C1N60·C1N73·C1N90 은 전부 FICR 쪽이었다.
    H3  최선 팔의 이득이 **검출문턱 0.001013 이상**.
    H4  `util`(alpha 0.5)이 `util_soft`(0.8)보다 낫다 — 효용 비중이 클수록 좋다.
        **부호 예단 없음.** 반대면 정규화가 지배한다는 뜻이고 C1N43 의 결론이
        관측 범위 안에서도 성립하는 것이다.

  H1 이 참이면 **C1N43 의 폐쇄가 뒤집히고** 손실 축이 열린다. 거짓이면 C1N43 이
  외삽으로 내린 결론이 관측으로 확인되고, 손실 축이 **처음으로 제대로 닫힌다**.

게이트 미수정. lockbox·외부데이터 미사용. 제출 없음.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from m270_gate import GATE_VERSION, evaluate_gate
from m271_cycle37_band_loss import KEYS, PROBE, fold_rows
from m271_cycle40_band_classifier import (
    CENTERS,
    CLASS_WIDTH,
    N_CLASS,
    PARAMS,
    ROUNDS,
    bayes_decision,
    one_hot_targets,
)
from m271_cycle42_teacher_restored import all_weather_columns, teach
from m271_cycle44_sharpened_decision import TEMPERATURES, sharpen
from m271_cycle56_measured_powercurve import add_sitewind_with_basis, measured_curves
from m271_evaluate_candidate import official
from run_sequence_classifier import _surface

from baram.constants import CAPACITIES_KWH
from baram.evaluation.official import settlement_unit

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
REPORT_MD = REPORTS / "m271_n5_metric_aligned_loss.md"
RECEIPT = REPORTS / "m271_n5_metric_aligned_loss_receipt.json"

NODE_ID = "C1N92_METRIC_ALIGNED_LOSS"
LANE = "L3"
PARENT_NODE = "C1N39_ARCHITECTURE_GAP"

CONTROL = 0.604043
TOLERANCE = 0.0005
DETECTION_THRESHOLD = 0.001013
ARMS = {"ce": None, "util": 0.5, "util_soft": 0.8}


def expected_utility(rate: np.ndarray) -> np.ndarray:
    """`E_util[n, i]` = 구간 i 를 예측했을 때 라벨 rate[n] 에서 얻는 효용.

    `bayes_decision` 이 쓰는 바로 그 효용식이다. 결정층에만 쓰던 것을 학습에도 쓴다.
    """
    cbar = float(CENTERS.mean())
    distance = np.abs(CENTERS[None, :] - rate[:, None])
    unit = settlement_unit(distance.reshape(-1)).reshape(distance.shape)
    return 0.25 * (rate[:, None] / cbar) * unit - distance


def make_mixed_objective(target: np.ndarray, util: np.ndarray, alpha: float | None):
    """`alpha` 가 None 이면 순수 교차엔트로피(현행)."""
    # 효용을 확률 규모로 정규화한다. 행마다 최대 1 이 되게 하면 기울기 규모가
    # 교차엔트로피와 같은 자릿수에 놓인다 — C1N37 의 D2(헤시안 대비 기울기 폭주)를 피한다.
    scaled = util - util.max(axis=1, keepdims=True)
    scaled = scaled / max(float(np.abs(scaled).max()), 1e-9)

    def objective(preds: np.ndarray, dataset: lgb.Dataset):
        raw = preds if preds.ndim == 2 else preds.reshape(N_CLASS, -1).T
        shifted = raw - raw.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        p = exp / exp.sum(axis=1, keepdims=True)
        grad = p - target
        if alpha is not None:
            grad = alpha * grad + (1.0 - alpha) * (-scaled)
        hess = np.maximum(2.0 * p * (1.0 - p), 1e-6)
        if preds.ndim == 1:
            return grad.T.reshape(-1), hess.T.reshape(-1)
        return grad, hess

    return objective


def main() -> int:
    curves = measured_curves()
    surface, _base, auxiliary = _surface()
    surface["forecast_kst_dtm"] = pd.to_datetime(surface["forecast_kst_dtm"])
    surface["capacity"] = surface["group_id"].map(CAPACITIES_KWH).astype(float)
    surface["rate"] = surface["actual_kwh"] / surface["capacity"]
    surface = surface.loc[surface["rate"].notna()].reset_index(drop=True)

    wanted = json.loads(
        (PROBE / "M115_XGBOOST-dev-2023-Q3.json").read_text(encoding="utf-8")
    )["selected_feature_names"]
    base_features = [c for c in wanted if c in surface.columns and c != "scada_ws"]
    aux_cols = [c for c in auxiliary if c in surface.columns and c != "scada_ws"]
    aw_cols = all_weather_columns(surface)

    store: dict[str, dict[str, Any]] = {}
    grad_scale: dict[str, list[float]] = {a: [] for a in ARMS}
    fits = 0
    for probe_fold, meta in fold_rows().items():
        train = surface.loc[surface["forecast_kst_dtm"] < meta["start"]].copy()
        test = surface.loc[
            np.array([
                (fid, gid) in meta["keys"]
                for fid, gid in zip(surface["forecast_id"], surface["group_id"],
                                    strict=True)
            ])
        ].copy()
        legacy_tr, legacy_te = teach(train, test, aux_cols)
        aw_tr, aw_te = teach(train, test, aw_cols)

        rate = np.clip(train["rate"].to_numpy(dtype="float64"), 0.0, None)
        target = one_hot_targets(rate)
        util = expected_utility(rate)

        names = add_sitewind_with_basis(train, legacy_tr, aw_tr, "generic", curves)
        add_sitewind_with_basis(test, legacy_te, aw_te, "generic", curves)
        features = [*base_features, *names]
        dataset_x = train.loc[:, features].astype("float32")
        label = np.clip((rate / CLASS_WIDTH).astype(int), 0, N_CLASS - 1)

        entry: dict[str, Any] = {
            "meta": test.loc[:, [*KEYS, "actual_kwh"]].copy(),
            "capacity": test["capacity"].to_numpy(dtype="float64"),
            "group": test["group_id"].to_numpy(),
        }
        for arm, alpha in ARMS.items():
            dataset = lgb.Dataset(dataset_x, label=label, free_raw_data=False)
            params = dict(PARAMS)
            params["objective"] = make_mixed_objective(target, util, alpha)
            booster = lgb.train(params, dataset, num_boost_round=ROUNDS)
            fits += 1
            raw = np.asarray(
                booster.predict(test.loc[:, features].astype("float32"))
            ).reshape(len(test), N_CLASS)
            exp = np.exp(raw - raw.max(axis=1, keepdims=True))
            prob = exp / exp.sum(axis=1, keepdims=True)
            entry[arm] = prob
            # 기울기 규모 대리 — 초기 균등확률에서의 |grad| 평균.
            p0 = np.full_like(target, 1.0 / N_CLASS)
            g = p0 - target
            if alpha is not None:
                scaled = util - util.max(axis=1, keepdims=True)
                scaled = scaled / max(float(np.abs(scaled).max()), 1e-9)
                g = alpha * g + (1.0 - alpha) * (-scaled)
            grad_scale[arm].append(float(np.abs(g).mean()))
        store[probe_fold] = entry

    folds = sorted(store)
    ref = float(np.mean(grad_scale["ce"]))
    ratios = {a: float(np.mean(grad_scale[a])) / ref for a in ARMS}
    v2 = bool(all(0.1 <= r <= 10.0 for r in ratios.values()))
    v3 = bool(all(
        np.isfinite(store[f][a]).all()
        and abs(store[f][a].sum(axis=1) - 1.0).max() < 1e-9
        for f in folds for a in ARMS
    ))

    def scored(fold: str, arm: str, temperature: float) -> pd.DataFrame:
        e = store[fold]
        out = e["meta"].copy()
        out["prediction_kwh"] = (
            bayes_decision(sharpen(e[arm], temperature)) * e["capacity"]
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

    v1 = bool(abs(results["ce"]["total"] - CONTROL) <= TOLERANCE)

    gains = {a: results[a]["total"] - results["ce"]["total"] for a in ARMS}
    best_arm = max((a for a in ARMS if a != "ce"), key=lambda a: gains[a])
    best_gain = gains[best_arm]

    h1 = bool(best_gain > 0.0)
    nmae_contrib = 0.5 * (
        results[best_arm]["one_minus_nmae"] - results["ce"]["one_minus_nmae"]
    )
    ficr_contrib = 0.5 * (results[best_arm]["ficr"] - results["ce"]["ficr"])
    h2 = bool(nmae_contrib > ficr_contrib)
    h3 = bool(best_gain >= DETECTION_THRESHOLD)
    h4 = bool(gains["util"] > gains["util_soft"])

    gate = evaluate_gate(frames[best_arm], frames["ce"])
    gd = gate.evidence
    flags = {label.split()[0]: bool(ok) for label, ok in gate.conditions.items()}
    signature = "[" + "".join("O" if flags[g] else "-" for g in sorted(flags)) + "]"

    if not (v1 and v2 and v3):
        verdict = "GUARD_FAILED_RESULT_VOID"
    elif h1 and h3:
        verdict = "LOSS_ALIGNMENT_REVIVES_C1N43_PREMISE_FLIPPED"
    elif h1:
        verdict = "LOSS_ALIGNMENT_POSITIVE_BUT_BELOW_DETECTION"
    else:
        verdict = "C1N43_CONFIRMED_BY_OBSERVATION_LOSS_AXIS_CLOSES"

    payload: dict[str, Any] = {
        "node_id": NODE_ID,
        "lane": LANE,
        "parent": PARENT_NODE,
        "gate_version": GATE_VERSION,
        "tests_premise": {
            "node": "C1N43_EFFECT_TREND",
            "premise": "BAND_TARGET_VANISHES_AT_FRONTIER",
            "revival_predicate": "positive_band_effect_at_frontier_control",
            "why_now": (
                "C1N43 은 대조군 품질 0.5679/0.5845/0.5959 의 3 점 외삽으로 닫았고 "
                "스스로 '관측 범위 밖' 이라 기록했다. 현 대조군 0.604043 은 그 범위 "
                "밖이므로 여기서 재는 것은 외삽이 아니라 관측이다."
            ),
        },
        "objective": (
            "L = alpha*CE + (1-alpha)*(-기대효용). 효용은 `bayes_decision` 이 쓰는 "
            "바로 그 식이며 결정층에만 쓰던 것을 학습에도 쓴다. 헤시안은 교차엔트로피 "
            "것을 유지한다(C1N37 의 D2 재발 방지)."
        ),
        "model_fits": fits,
        "arms": results,
        "gains": gains,
        "best_arm": best_arm,
        "best_gain": best_gain,
        "gradient_scale_ratio": ratios,
        "chosen_temperature": chosen,
        "contributions": {"nmae": float(nmae_contrib), "ficr": float(ficr_contrib)},
        "detection_threshold": DETECTION_THRESHOLD,
        "checks": {"V1_ce_reproduces_control": v1, "V1_gap": abs(results["ce"]["total"] - CONTROL),
                   "V2_gradient_scale_ok": v2, "V3_probabilities_valid": v3},
        "hypotheses": {
            "H1_alignment_positive": h1,
            "H2_gain_is_nmae_side": h2,
            "H3_clears_detection": h3,
            "H4_more_utility_is_better": h4,
        },
        "gate": {
            "signature": signature, "flags": flags,
            "positive_months": int(gd["positive_months"]),
            "months_scored": int(gd["months_scored"]),
            "bootstrap_q05": float(gd["block_bootstrap_q05"]),
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
        "# M271 N5 — 손실 축 재개: 지표 정합 목적함수",
        "",
        f"노드 `{NODE_ID}` / 레인 {LANE} / 부모 `{PARENT_NODE}`",
        "",
        f"**C1N43 의 부활 술어를 정면으로 검정한다.** {payload['tests_premise']['why_now']}",
        "",
        payload["objective"],
        "",
        "## 1. 팔",
        "",
        "| 팔 | Total | 1-NMAE | FICR | ce 대비 |",
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
        f"- V1 ce 가 {CONTROL} 재현 -> **{v1}**",
        f"- V2 기울기 규모 비 {[round(v, 3) for v in ratios.values()]} -> **{v2}**",
        f"- V3 확률 정상 -> **{v3}**",
        f"- H1 정합이 양수 -> **{h1}**",
        f"- H2 이득이 1-NMAE 쪽 (1-NMAE {nmae_contrib:+.6f} / FICR {ficr_contrib:+.6f}) "
        f"-> **{h2}**",
        f"- H3 검출문턱 통과 -> **{h3}**",
        f"- H4 효용 비중이 클수록 낫다 -> **{h4}**",
        "",
        "## 3. 판정",
        "",
        f"**{verdict}**  게이트 {signature} "
        f"({gd['positive_months']}/{gd['months_scored']} 월)",
        "",
        f"digest `{payload['digest']}`",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("=== N5 완료 ===")
    print(f"[N5] 적합 {fits} / 기울기 규모 비 "
          f"{ {k: round(v,3) for k, v in ratios.items()} }")
    for arm in ARMS:
        r = results[arm]
        print(f"[N5] {arm:10s} {r['total']:.6f} (1-NMAE {r['one_minus_nmae']:.6f} / "
              f"FICR {r['ficr']:.6f})  {gains[arm]:+.6f}")
    print(f"[N5] 최선 {best_arm} {best_gain:+.6f} / 문턱 {DETECTION_THRESHOLD}")
    print(f"[N5] 기여 1-NMAE {nmae_contrib:+.6f} / FICR {ficr_contrib:+.6f}")
    print(f"[N5] V1 {v1} / V2 {v2} / V3 {v3} / H1 {h1} / H2 {h2} / H3 {h3} / H4 {h4}")
    print(f"[N5] 게이트 {signature} {gd['positive_months']}/{gd['months_scored']}월")
    print(f"[N5] 판정: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
