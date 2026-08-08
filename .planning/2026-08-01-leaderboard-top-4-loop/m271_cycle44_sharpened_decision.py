"""M271 P4 사이클 44 — 마지막 결손 원인 제거: 결정규칙에 날카로움 자유도를 준다.

사이클 42 에서 teacher 복원이 성공했고(V3 통과, 대조군 +0.011451) V1 은 **0.002685 차**로
기각됐다. 사이클 40 이 실행 전에 명명한 결손 원인 셋 중 둘은 제거됐다(용량 → 41 이 무관함을
보임, teacher → 42 가 제거). 남은 하나는 이것이다.

    내 Bayes 결정규칙  vs  배포의 튜닝된 (T, G) 정책

사이클 8 이 46-bin 분포가 **과소확신** 이라고 쟀다. `T0.5_G1.5` 같은 정책은 분포를
**날카롭게** 만들어 그것을 보정한다. 내 결정규칙에는 그 자유도가 없었다 — 원 분포를 그대로
Bayes 결정에 넣었다. 같은 자유도를 준다.

    p_T(i) ∝ p(i)^(1/T),   그 다음 사이클 40 의 Bayes 결정

**선택 편향은 fold-외 선택으로 제거한다.** 보류 fold 에 쓸 T 는 **나머지 두 fold** 에서
고른다. 평가하는 데이터로 T 를 고르지 않는다. 사이클 35 가 정책 선택에서 게이트+fold 3/3
으로 다중검정을 통제했다면, 여기서는 애초에 **선택을 fold 밖으로 뺀다.**

이것이 마지막 명명된 결손이다. 이후에도 V1 이 기각되면 이 축의 대조군 개선 시도를 종료한다.

① 방법 리서치 (실행 전)
  - 새 방법 없음. 온도 조절(temperature scaling)은 분류기 보정의 표준(Guo et al. 2017)
    이고, 배포 정책의 `T` 도 같은 역할이다.
  - 두 팔에 **동일한 절차**를 적용한다. 각 팔이 자기 분포에 맞는 T 를 fold-외로 고른다.

② 사양 동결

  학습   사이클 42 와 **완전 동일** (teacher 복원, leaves 15, lr 0.1, rounds 200, 101 피처)
  결정   `p^(1/T)` 정규화 후 사이클 40 의 Bayes 결정.
         T 격자 = {0.3, 0.4, 0.5, 0.6, 0.75, 1.0, 1.3, 1.7, 2.2} (실행 전 동결)
  선택   **fold-외**. 보류 fold f 의 T = 나머지 두 fold 의 pooled Total 을 최대화하는 값

  **타당성 가드**
    V1  CONTROL 이 배포(0.628605)의 `-0.03` 이내.
    V4  날카로움 자유도가 CONTROL 을 개선한다 (사이클 42 CONTROL 0.595919 초과).

  사전확약(V1 통과시에만 판정):
    H1  BAND > CONTROL.
    H2  BAND 가 배포 대비 개선 + **동결 게이트 통과**.
    H3  BAND > `M115@T0.6_G0.2` (0.630310).
    H4  이득이 **FICR 쪽**에서 나온다.
    H5  처리효과 부호가 사이클 40·41·42 와 동일 (네 번째 재현).
    H6  (추세 검증) 처리효과가 사이클 43 의 회귀선이 예측한 값의 **±0.006 이내**다.
        사이클 43 은 3 점 외삽이었다. 네 번째 점이 그 선 위에 오면 외삽 근거가 강해지고,
        벗어나면 그 폐쇄를 재고해야 한다.

**게이트를 수정하지 않는다.** 2024 행·lockbox 미사용. `scada_ws` 는 teacher 표적으로만.
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
from m270_monthly_validation import load_predictions
from m271_cycle37_band_loss import KEYS, PROBE, fold_rows
from m271_cycle40_band_classifier import (
    CLASS_WIDTH,
    DEPLOYED,
    FOLDS,
    M115_FIXED_TOTAL,
    N_CLASS,
    PARAMS,
    ROUNDS,
    V1_TOLERANCE,
    bayes_decision,
    by_fold_total,
    make_objective,
    one_hot_targets,
    soft_targets,
)
from m271_cycle42_teacher_restored import add_sitewind, all_weather_columns, teach
from m271_evaluate_candidate import official
from run_sequence_classifier import _surface

from baram.constants import CAPACITIES_KWH

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
REPORT_MD = REPORTS / "m271_cycle44_sharpened_decision.md"
RECEIPT = REPORTS / "m271_cycle44_sharpened_decision_receipt.json"

NODE_ID = "C1N44_SHARPENED_DECISION"
LANE = "L3"
PARENT_NODE = "C1N42_TEACHER_RESTORED"
DEPLOYED_TOTAL = 0.628605
C42_CONTROL = 0.595919
PRIOR_DELTAS = (0.009772, 0.021490, 0.004827)
TREND_SLOPE = -0.6022
TREND_INTERCEPT = 0.3630
H6_TOLERANCE = 0.006

TEMPERATURES = (0.3, 0.4, 0.5, 0.6, 0.75, 1.0, 1.3, 1.7, 2.2)


def sharpen(prob: np.ndarray, temperature: float) -> np.ndarray:
    powered = np.power(np.clip(prob, 1e-12, None), 1.0 / temperature)
    return powered / powered.sum(axis=1, keepdims=True)


def main() -> int:
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

    # fold 별로 각 팔의 확률과 메타를 보관한다. T 선택은 뒤에서 fold-외로 한다.
    store: dict[str, dict[str, Any]] = {}
    fits = 0
    for probe_fold, meta in fold_rows().items():
        train = surface.loc[surface["forecast_kst_dtm"] < meta["start"]].copy()
        test = surface.loc[
            np.array(
                [
                    (fid, gid) in meta["keys"]
                    for fid, gid in zip(surface["forecast_id"], surface["group_id"],
                                        strict=True)
                ]
            )
        ].copy()
        legacy_tr, legacy_te = teach(train, test, aux_cols)
        aw_tr, aw_te = teach(train, test, aw_cols)
        sitewind = add_sitewind(train, legacy_tr, aw_tr)
        add_sitewind(test, legacy_te, aw_te)
        features = [*base_features, *sitewind]

        x_tr = train.loc[:, features].astype("float32")
        rate = np.clip(train["rate"].to_numpy(dtype="float64"), 0.0, None)
        label = np.clip((rate / CLASS_WIDTH).astype(int), 0, N_CLASS - 1)
        entry: dict[str, Any] = {
            "meta": test.loc[:, [*KEYS, "actual_kwh"]].copy(),
            "capacity": test["capacity"].to_numpy(dtype="float64"),
        }
        for name, target in (
            ("BAND", soft_targets(rate)),
            ("CONTROL", one_hot_targets(rate)),
        ):
            dataset = lgb.Dataset(x_tr, label=label, free_raw_data=False)
            params = dict(PARAMS)
            params["objective"] = make_objective(target)
            booster = lgb.train(params, dataset, num_boost_round=ROUNDS)
            fits += 1
            raw = np.asarray(
                booster.predict(test.loc[:, features].astype("float32"))
            ).reshape(len(test), N_CLASS)
            exp = np.exp(raw - raw.max(axis=1, keepdims=True))
            entry[name] = exp / exp.sum(axis=1, keepdims=True)
        store[probe_fold] = entry

    def scored(fold: str, arm: str, temperature: float) -> pd.DataFrame:
        e = store[fold]
        out = e["meta"].copy()
        out["prediction_kwh"] = (
            bayes_decision(sharpen(e[arm], temperature)) * e["capacity"]
        )
        out["month"] = out["forecast_kst_dtm"].dt.to_period("M").astype(str)
        return out

    chosen_t: dict[str, dict[str, float]] = {}
    pieces: dict[str, list[pd.DataFrame]] = {"BAND": [], "CONTROL": []}
    for arm in ("BAND", "CONTROL"):
        chosen_t[arm] = {}
        for held in store:
            others = [f for f in store if f != held]
            best_t, best_score = None, -np.inf
            for temperature in TEMPERATURES:
                frame = pd.concat(
                    [scored(f, arm, temperature) for f in others], ignore_index=True
                )
                total = official(frame)["total"]
                if total > best_score:
                    best_t, best_score = temperature, total
            chosen_t[arm][held] = float(best_t)
            pieces[arm].append(scored(held, arm, float(best_t)))

    frames, results = {}, {}
    for arm, parts in pieces.items():
        frame = pd.concat(parts, ignore_index=True)
        frames[arm] = frame
        results[arm] = official(frame)

    parent = load_predictions(DEPLOYED)
    parent_score = official(parent)
    v1_gap = results["CONTROL"]["total"] - parent_score["total"]
    v1 = bool(v1_gap >= -V1_TOLERANCE)
    v4 = bool(results["CONTROL"]["total"] > C42_CONTROL)

    delta = results["BAND"]["total"] - results["CONTROL"]["total"]
    ficr_contrib = 0.5 * (results["BAND"]["ficr"] - results["CONTROL"]["ficr"])
    nmae_contrib = 0.5 * (
        results["BAND"]["one_minus_nmae"] - results["CONTROL"]["one_minus_nmae"]
    )
    gate = evaluate_gate(frames["BAND"], parent)
    gd = gate.evidence
    h5 = bool(all(np.sign(delta) == np.sign(d) for d in PRIOR_DELTAS))
    predicted = TREND_SLOPE * results["CONTROL"]["total"] + TREND_INTERCEPT
    h6 = bool(abs(delta - predicted) <= H6_TOLERANCE)

    if v1:
        h1: bool | None = bool(delta > 0)
        h2: bool | None = bool(
            results["BAND"]["total"] > parent_score["total"] and gate.passed
        )
        h3: bool | None = bool(results["BAND"]["total"] > M115_FIXED_TOTAL)
        h4: bool | None = bool(ficr_contrib > nmae_contrib)
        verdict = (
            "BAND_CLASSIFIER_PROMOTED" if h2
            else ("BAND_CLASSIFIER_HELPS_INTERNALLY_ONLY" if h1
                  else "BAND_AWARE_TRAINING_CLOSED_BY_MEASUREMENT")
        )
    else:
        h1 = h2 = h3 = h4 = None
        verdict = "ALL_NAMED_DEFECTS_REMOVED_STILL_UNJUDGED"

    by_fold = {arm: by_fold_total(frame) for arm, frame in frames.items()}
    check = {
        "V1_expectation": f"CONTROL 이 배포의 -{V1_TOLERANCE} 이내",
        "V1_held": v1, "V1_measured_gap": v1_gap,
        "V4_expectation": f"날카로움 자유도가 CONTROL 개선 (> {C42_CONTROL})",
        "V4_held": v4, "V4_measured": results["CONTROL"]["total"] - C42_CONTROL,
        "H1_expectation": "BAND > CONTROL", "H1_held": h1, "H1_measured": delta,
        "H2_expectation": "BAND 가 배포 대비 개선 + 게이트 통과", "H2_held": h2,
        "H3_expectation": f"BAND > {M115_FIXED_TOTAL}", "H3_held": h3,
        "H4_expectation": "이득이 FICR 쪽", "H4_held": h4,
        "H5_expectation": "처리효과 부호가 40·41·42 와 동일", "H5_held": h5,
        "H6_expectation": f"처리효과가 사이클 43 회귀선 예측의 ±{H6_TOLERANCE} 이내",
        "H6_held": h6, "H6_predicted": predicted, "H6_residual": delta - predicted,
        "judged": v1, "verdict": verdict,
    }

    payload = {
        "node": NODE_ID, "parent_node": PARENT_NODE,
        "gate_version": GATE_VERSION, "gate_modified": False, "lockbox_used": False,
        "changed_vs_cycle42": "결정규칙에 온도 자유도 추가. **T 는 fold-외 선택**",
        "temperature_grid": list(TEMPERATURES),
        "chosen_temperature_out_of_fold": chosen_t,
        "features_used": len(base_features) + 14,
        "classifier_fits": fits,
        "scores": {**results, "deployed": parent_score},
        "by_fold": by_fold,
        "delta_band_minus_control": delta,
        "contribution": {"ficr": ficr_contrib, "one_minus_nmae": nmae_contrib},
        "prior_deltas": list(PRIOR_DELTAS),
        "trend_check": {"slope": TREND_SLOPE, "intercept": TREND_INTERCEPT,
                        "predicted": predicted, "observed": delta,
                        "residual": delta - predicted},
        "gate_vs_deployed": {
            "passed": bool(gate.passed),
            "flags": {la.split()[0]: bool(ok) for la, ok in gate.conditions.items()},
            "positive_months": int(gd["positive_months"]),
            "months_scored": int(gd["months_scored"]),
            "sign_test_p": float(gd["sign_test_p_greater"]),
            "bootstrap_q05": float(gd["block_bootstrap_q05"]),
            "min_delta": float(gd["min_total_delta"]),
        },
        "predeclared_check": check,
    }

    g = payload["gate_vs_deployed"]
    flags = "".join("O" if g["flags"].get(x) else "-" for x in ("G1", "G2", "G3", "G4"))
    lines = [
        "# M271 P4 사이클 44 — 결정규칙에 날카로움 자유도 (마지막 명명 결손)",
        "",
        f"- 판정일: {datetime.now(UTC).strftime('%Y-%m-%d')} (UTC)",
        f"- 노드: `{NODE_ID}` / 레인 {LANE} / 부모 `{PARENT_NODE}`",
        f"- 사이클 42 대비: **{payload['changed_vs_cycle42']}**",
        f"- 게이트: `{GATE_VERSION}` (읽기만 함) / lockbox 미사용",
        "",
        "## 1. 온도 선택 (fold-외)",
        "",
        "보류 fold 에 쓸 T 는 **나머지 두 fold** 에서 골랐다. 평가 데이터로 고르지 않는다.",
        "",
        "| 팔 | " + " | ".join(store) + " |",
        "|---|" + "---|" * len(store),
    ]
    for arm in ("CONTROL", "BAND"):
        lines.append(
            f"| `{arm}` | " + " | ".join(f"{chosen_t[arm][f]:.2f}" for f in store) + " |"
        )
    lines += [
        "",
        "## 2. 가드",
        "",
        f"- V1 CONTROL {results['CONTROL']['total']:.6f} vs 배포 "
        f"{parent_score['total']:.6f} -> **{v1_gap:+.6f}** -> **{v1}**",
        f"- V4 온도 자유도 효과 **{results['CONTROL']['total'] - C42_CONTROL:+.6f}** "
        f"(사이클 42 {C42_CONTROL:.6f}) -> **{v4}**",
        "",
        "## 3. 결과",
        "",
        "| 모델 | Total | 1-NMAE | FICR |",
        "|---|---:|---:|---:|",
        f"| 배포 | {parent_score['total']:.6f} | "
        f"{parent_score['one_minus_nmae']:.6f} | {parent_score['ficr']:.6f} |",
        f"| `CONTROL` | {results['CONTROL']['total']:.6f} | "
        f"{results['CONTROL']['one_minus_nmae']:.6f} | {results['CONTROL']['ficr']:.6f} |",
        f"| **`BAND`** | **{results['BAND']['total']:.6f}** | "
        f"{results['BAND']['one_minus_nmae']:.6f} | {results['BAND']['ficr']:.6f} |",
        "",
        f"BAND - CONTROL = **{delta:+.6f}** "
        f"(FICR {ficr_contrib:+.6f} / 1-NMAE {nmae_contrib:+.6f})",
        "",
        "## 4. 추세선 검증 (H6)",
        "",
        f"사이클 43 회귀선 예측 **{predicted:+.6f}**, 실측 **{delta:+.6f}**, "
        f"잔차 **{delta - predicted:+.6f}** (허용 ±{H6_TOLERANCE}) -> **{h6}**",
        "",
        "네 번째 점이 선 위에 오면 사이클 43 의 외삽 근거가 강해진다.",
        "",
        "| fold | CONTROL | BAND |",
        "|---|---:|---:|",
    ]
    for fold in FOLDS:
        lines.append(
            f"| {fold} | {by_fold['CONTROL'].get(fold, float('nan')):.6f} | "
            f"{by_fold['BAND'].get(fold, float('nan')):.6f} |"
        )
    lines += [
        "",
        f"BAND 대 배포 게이트: `{flags}` {g['positive_months']}/{g['months_scored']}월 -> "
        f"**{'통과' if g['passed'] else '기각'}**",
        "",
        "## 5. 사전확약 대조",
        "",
        f"- V1 -> **{v1}** ({v1_gap:+.6f})",
        f"- V4 -> **{v4}**",
        f"- H1 -> **{h1 if h1 is not None else '판정안함'}** ({delta:+.6f})",
        f"- H2 -> **{h2 if h2 is not None else '판정안함'}**",
        f"- H3 -> **{h3 if h3 is not None else '판정안함'}**",
        f"- H4 -> **{h4 if h4 is not None else '판정안함'}**",
        f"- H5 -> **{h5}**",
        f"- H6 -> **{h6}**",
        "",
        f"판정: **{verdict}**",
        "",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    receipt: dict[str, Any] = {
        "schema_version": 1,
        "stage": "M271_P4_CYCLE44_SHARPENED_DECISION",
        "node": NODE_ID, "lane": LANE,
        "decided_utc": datetime.now(UTC).isoformat(),
        "result": payload,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "report_sha256": hashlib.sha256(REPORT_MD.read_bytes()).hexdigest(),
        "dacon_upload": False, "external_actions": [], "model_fits": fits,
        "lockbox_reopened": False, "new_2024_evaluation": False,
    }
    RECEIPT.write_text(
        json.dumps(receipt, indent=1, ensure_ascii=False, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

    print(f"[C44] 분류기 적합 {fits} / 온도(fold-외) {chosen_t}")
    print(f"[C44] CONTROL {results['CONTROL']['total']:.6f} "
          f"(배포 대비 {v1_gap:+.6f}, 사이클42 대비 "
          f"{results['CONTROL']['total'] - C42_CONTROL:+.6f}) -> V1 {v1} V4 {v4}")
    print(f"[C44] BAND    {results['BAND']['total']:.6f} (CONTROL 대비 {delta:+.6f})")
    print(f"[C44] 추세선 예측 {predicted:+.6f} 실측 {delta:+.6f} "
          f"잔차 {delta - predicted:+.6f} -> H6 {h6}")
    print(f"[C44] 게이트 [{flags}] {g['positive_months']}/{g['months_scored']}월 -> "
          f"{'통과' if g['passed'] else '기각'}")
    print(f"[C44] 판정: {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
