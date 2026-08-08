"""M271 P4 사이클 60 — 수준조건부 온도: 하나의 T 로는 10 배 변하는 산포를 못 맞춘다.

사이클 57 의 receipt 를 y 대역으로 다시 읽으니 산포가 출력수준을 따라 **단조로, 그리고
아주 매끄럽게** 커진다.

    group 1  sigma 0.0224 -> 0.0376 -> 0.0502 -> 0.0578 -> 0.0730   log-log 기울기 +0.551 (r 0.973)
    group 2  sigma 0.0067 -> 0.0228 -> 0.0368 -> 0.0461 -> 0.0508   log-log 기울기 +0.540 (r 0.967)
    group 3  sigma 0.0135 -> 0.0395 -> 0.0657 -> 0.0870 -> 0.1492   log-log 기울기 +0.818 (r 0.983)

r 0.97~0.98 이다. 조건부 산포는 잡음이 아니라 **출력수준의 매끈한 함수**다. 그런데 결정층은
fold 당 **전역 T 하나**를 쓴다(C44). sigma 가 구간에 걸쳐 약 10 배 변하는데 날카로움
자유도는 상수다. 그러면 한쪽 끝에서는 과하게 날카롭고 반대쪽에서는 무디다 — 그것도
**체계적으로**.

이건 병목 재특정이 아니라 결정층 안의 미사용 자유도다. 더 나은 예보가 필요없다.

**① 방법 리서치**

  - 온도 스케일링은 Guo et al.(2017) 의 분류기 보정 표준이고 C44 가 이미 채택했다.
    같은 논문이 **단일 스칼라 T** 를 쓰는 이유는 신경망 로짓의 오보정이 전역적으로
    균질하다고 봤기 때문이다. 여기서는 그 전제가 측정으로 깨졌다(위 기울기).
  - 조건부 보정의 표준 확장은 **벡터/구간별 스케일링**이다. Guo et al. 자신이 vector
    scaling / matrix scaling 을 같은 표에서 비교했고, Kuleshov et al.(2018) 의
    분위수 보정, Song et al.(2019) 의 distribution calibration 이 모두 "보정 파라미터를
    입력의 함수로 둔다"는 같은 뼈대다.
  - 이 대회 손실은 FICR — |오차| 의 계단함수다. 계단폭이 **용량 고정**(6%/8%)인데
    조건부 산포는 수준을 따라 변하므로, 산포 대 계단폭의 비가 구간마다 다르다.
    그 비가 결정의 날카로움을 지배한다. 따라서 T 를 수준의 함수로 두는 것이
    이 손실함수에 대해 정확히 맞는 확장이다.
  - **채택**: 수준별 T. 새 런타임 없음. C44 하네스의 `sharpen` 을 행별 T 로 일반화만 한다.

**② 사양 동결**

  하네스   사이클 56·59 와 동일 (teacher 복원, generic 기저, leaves 15, lr 0.1, 200 rounds).
           확률행렬까지는 **완전히 같고** 결정층만 갈린다. 따라서 적합은 3 회뿐이고
           팔 사이에 학습 차이가 없다 — 처리효과가 결정층에만 귀속된다.
  수준     예비 점추정 yhat0 = T=1.0 Bayes 결정 / 용량. 경계 **(0.25, 0.70)** 로 3 구간.
           C57 대역표에서 산포가 뚜렷이 갈리는 지점이고 실행 전에 동결한다.
  팔 셋    GLOBAL      T 하나            (C44 재현. 1 파라미터)
           LEVEL       T 3 개 (수준별)    (3 파라미터)
           LEVELGROUP  T 9 개 (수준x그룹) (9 파라미터)
  선택     **fold-외**. 보류 fold 의 T 들은 나머지 두 fold 에서 고른다. 다파라미터 팔은
           전역 최적에서 출발해 **좌표상승 2 회전**(격자 TEMPERATURES 고정, 순서 고정).
           평가하는 데이터로 고르지 않는다.

  **타당성 가드**
    V1  GLOBAL 이 C44 대조군 0.604043 의 ±0.0005 이내. 벗어나면 결정층 외 무언가가
        바뀐 것이고 나머지 판정을 버린다.
    V2  세 팔의 확률행렬이 **바이트 동일**. 갈리면 처리효과가 결정층 것이 아니다.

  사전확약 (V1·V2 통과시에만 판정):
    H1  LEVEL > GLOBAL.  ← 핵심. 수준별 T 족은 전역 T 족을 **포함**하므로 표본내에서는
        반드시 우세하다. fold-외 선택이므로 이 비교는 진짜 검정이다. 자유도가 값을
        못 하면 여기서 진다.
    H2  LEVEL 이 GLOBAL 대비 **동결 게이트 통과**.
    H3  이득이 FICR 쪽에서 우세.
    H4  LEVELGROUP > LEVEL. 부호 예상 **불확실** — g3 기울기(+0.818)가 g1/g2(+0.54)와
        달라 그룹별 자유도에 근거가 있지만, 9 파라미터를 fold 두 개로 고르면 과적합이
        더 클 수 있다. 지면 과적합 서명이고 그 자체가 정보다.
    H5  선택된 T 가 수준에 따라 **단조**다. 무작위면 자유도가 잡음을 먹은 것이다.

  **부호를 예단하지 않는다.** 산포가 크면 T 를 올려야 하는지 내려야 하는지는 이론에서
  안 나온다. 계단손실의 argmax 는 분포 모양에 달렸다. 예측하는 것은 **수준의존성의
  존재**이지 방향이 아니다. H5 가 그 존재를 검정한다.

게이트 미수정. lockbox·외부데이터·2024 행·`scada_ws` 미사용. 제출 없음.
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
    CLASS_WIDTH,
    N_CLASS,
    PARAMS,
    ROUNDS,
    bayes_decision,
    make_objective,
    one_hot_targets,
)
from m271_cycle42_teacher_restored import all_weather_columns, teach
from m271_cycle44_sharpened_decision import TEMPERATURES, sharpen
from m271_cycle56_measured_powercurve import add_sitewind_with_basis, measured_curves
from m271_evaluate_candidate import official
from run_sequence_classifier import _surface

from baram.constants import CAPACITIES_KWH

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
REPORT_MD = REPORTS / "m271_cycle60_level_temperature.md"
RECEIPT = REPORTS / "m271_cycle60_level_temperature_receipt.json"

NODE_ID = "C1N60_LEVEL_TEMPERATURE"
LANE = "L7"
PARENT_NODE = "C1N57_FICR_CEILING"

C44_CONTROL = 0.604043
V1_TOLERANCE = 0.0005
CHAMPION_LOCAL = 0.630310
LEVEL_EDGES = (0.25, 0.70)
N_LEVEL = len(LEVEL_EDGES) + 1
GROUPS = (1, 2, 3)
SWEEPS = 2


def level_of(rate: np.ndarray) -> np.ndarray:
    return np.digitize(rate, np.asarray(LEVEL_EDGES, dtype="float64"))


def sharpen_by_row(probability: np.ndarray, temperature: np.ndarray) -> np.ndarray:
    """행마다 다른 T 로 p^(1/T) 를 적용한다. T 가 상수면 `sharpen` 과 동치."""
    t = np.asarray(temperature, dtype="float64").reshape(-1, 1)
    powered = np.power(np.clip(probability, 1e-12, None), 1.0 / t)
    return powered / powered.sum(axis=1, keepdims=True)


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

        rate = np.clip(train["rate"].to_numpy(dtype="float64"), 0.0, None)
        label = np.clip((rate / CLASS_WIDTH).astype(int), 0, N_CLASS - 1)
        target = one_hot_targets(rate)

        names = add_sitewind_with_basis(train, legacy_tr, aw_tr, "generic", curves)
        add_sitewind_with_basis(test, legacy_te, aw_te, "generic", curves)
        features = [*base_features, *names]
        dataset = lgb.Dataset(
            train.loc[:, features].astype("float32"), label=label, free_raw_data=False
        )
        params = dict(PARAMS)
        params["objective"] = make_objective(target)
        booster = lgb.train(params, dataset, num_boost_round=ROUNDS)
        fits += 1
        raw = np.asarray(
            booster.predict(test.loc[:, features].astype("float32"))
        ).reshape(len(test), N_CLASS)
        exp = np.exp(raw - raw.max(axis=1, keepdims=True))
        probability = exp / exp.sum(axis=1, keepdims=True)

        preliminary = bayes_decision(sharpen(probability, 1.0))
        store[probe_fold] = {
            "meta": test.loc[:, [*KEYS, "actual_kwh"]].copy(),
            "capacity": test["capacity"].to_numpy(dtype="float64"),
            "group": test["group_id"].to_numpy(),
            "probability": probability,
            "level": level_of(preliminary),
        }

    # V2 — 팔 사이 확률행렬이 하나뿐임을 해시로 못박는다.
    prob_digest = hashlib.sha256(
        b"".join(store[f]["probability"].tobytes() for f in sorted(store))
    ).hexdigest()[:16]

    def scored(fold: str, temperature: np.ndarray | float) -> pd.DataFrame:
        e = store[fold]
        t = (
            np.full(len(e["level"]), float(temperature))
            if np.isscalar(temperature)
            else np.asarray(temperature, dtype="float64")
        )
        out = e["meta"].copy()
        out["prediction_kwh"] = (
            bayes_decision(sharpen_by_row(e["probability"], t)) * e["capacity"]
        )
        out["group_id"] = e["group"]
        out["month"] = out["forecast_kst_dtm"].dt.to_period("M").astype(str)
        return out

    def expand(fold: str, arm: str, table: Any) -> np.ndarray:
        e = store[fold]
        if arm == "global":
            return np.full(len(e["level"]), float(table))
        if arm == "level":
            return np.asarray([table[int(v)] for v in e["level"]], dtype="float64")
        return np.asarray(
            [table[(int(g), int(v))] for g, v in zip(e["group"], e["level"], strict=True)],
            dtype="float64",
        )

    def pooled(folds: list[str], arm: str, table: Any) -> float:
        frame = pd.concat(
            [scored(f, expand(f, arm, table)) for f in folds], ignore_index=True
        )
        return float(official(frame)["total"])

    def select(folds: list[str], arm: str) -> Any:
        best_t, best_score = TEMPERATURES[0], -np.inf
        for temperature in TEMPERATURES:
            score = pooled(folds, "global", temperature)
            if score > best_score:
                best_t, best_score = temperature, score
        if arm == "global":
            return float(best_t)
        keys = (
            list(range(N_LEVEL))
            if arm == "level"
            else [(g, v) for g in GROUPS for v in range(N_LEVEL)]
        )
        table = dict.fromkeys(keys, float(best_t))
        for _ in range(SWEEPS):
            for key in keys:
                current, incumbent = table[key], best_score
                for temperature in TEMPERATURES:
                    table[key] = float(temperature)
                    score = pooled(folds, arm, table)
                    if score > incumbent:
                        current, incumbent = float(temperature), score
                table[key], best_score = current, incumbent
        return table

    arms = ("global", "level", "levelgroup")
    chosen: dict[str, dict[str, Any]] = {}
    pieces: dict[str, list[pd.DataFrame]] = {arm: [] for arm in arms}
    for arm in arms:
        chosen[arm] = {}
        for held in sorted(store):
            others = [f for f in sorted(store) if f != held]
            table = select(others, arm)
            chosen[arm][held] = table if arm == "global" else {
                str(k): v for k, v in table.items()
            }
            pieces[arm].append(scored(held, expand(held, arm, table)))

    frames = {arm: pd.concat(parts, ignore_index=True) for arm, parts in pieces.items()}
    results = {arm: official(frames[arm]) for arm in arms}

    v1_gap = abs(results["global"]["total"] - C44_CONTROL)
    v1 = bool(v1_gap <= V1_TOLERANCE)

    level_gain = results["level"]["total"] - results["global"]["total"]
    h1 = bool(level_gain > 0.0)
    gate = evaluate_gate(frames["level"], frames["global"])
    gd = gate.evidence
    h2 = bool(gate.passed)
    flags = {label.split()[0]: bool(ok) for label, ok in gate.conditions.items()}
    signature = "[" + "".join("O" if flags[g] else "-" for g in sorted(flags)) + "]"
    ficr_contrib = 0.5 * (results["level"]["ficr"] - results["global"]["ficr"])
    nmae_contrib = 0.5 * (
        results["level"]["one_minus_nmae"] - results["global"]["one_minus_nmae"]
    )
    h3 = bool(ficr_contrib > nmae_contrib)
    h4 = bool(results["levelgroup"]["total"] > results["level"]["total"])

    monotone = []
    for held, table in chosen["level"].items():
        series = [float(table[str(v)]) for v in range(N_LEVEL)]
        pairs = list(zip(series, series[1:], strict=False))
        monotone.append(
            all(a <= b for a, b in pairs) or all(a >= b for a, b in pairs)
        )
    h5 = bool(all(monotone))

    if not v1:
        verdict = "HARNESS_DRIFT_RESULT_VOID"
    elif not h1:
        verdict = "LEVEL_FREEDOM_DOES_NOT_PAY_OUT_OF_FOLD"
    elif not h2:
        verdict = "LEVEL_GAIN_NOT_MONTHLY_CONSISTENT"
    elif results["level"]["total"] > CHAMPION_LOCAL:
        verdict = "LEVEL_TEMPERATURE_BEATS_CHAMPION"
    else:
        verdict = "LEVEL_TEMPERATURE_HELPS_BELOW_CHAMPION"

    payload: dict[str, Any] = {
        "node_id": NODE_ID,
        "lane": LANE,
        "parent": PARENT_NODE,
        "gate_version": GATE_VERSION,
        "method": "LEVEL_CONDITIONAL_TEMPERATURE (Guo et al. 2017 vector scaling)",
        "motivation_slopes": {"g1": 0.551, "g2": 0.540, "g3": 0.818},
        "level_edges": list(LEVEL_EDGES),
        "model_fits": fits,
        "probability_digest": prob_digest,
        "chosen": chosen,
        "arms": {arm: results[arm] for arm in arms},
        "checks": {"V1_global_reproduces_c44": v1, "V1_gap": float(v1_gap),
                   "V2_single_probability_matrix": True},
        "level_gain": float(level_gain),
        "hypotheses": {
            "H1_level_beats_global": h1,
            "H2_gate_passed": h2,
            "H3_ficr_dominant": h3,
            "H4_levelgroup_beats_level": h4,
            "H5_temperature_monotone_in_level": h5,
        },
        "contributions": {"ficr": float(ficr_contrib), "nmae": float(nmae_contrib)},
        "gate": {
            "signature": signature,
            "flags": flags,
            "positive_months": int(gd["positive_months"]),
            "months_scored": int(gd["months_scored"]),
            "sign_test_p": float(gd["sign_test_p_greater"]),
            "median_delta": float(gd["median_total_delta"]),
            "bootstrap_q05": float(gd["block_bootstrap_q05"]),
            "min_delta": float(gd["min_total_delta"]),
        },
        "champion_local": CHAMPION_LOCAL,
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
        "# M271 P4 사이클 60 — 수준조건부 온도",
        "",
        f"노드 `{NODE_ID}` / 레인 {LANE} / 부모 `{PARENT_NODE}`",
        "",
        "동기: C57 receipt 의 대역별 산포가 출력수준에 대해 매끄러운 멱함수다 "
        "(log-log 기울기 g1 +0.551 / g2 +0.540 / g3 +0.818, r 0.97~0.98). "
        "그런데 결정층 T 는 fold 당 하나였다.",
        "",
        "## 1. 팔별 점수",
        "",
        "| 팔 | 파라미터 | Total | 1-NMAE | FICR |",
        "|---|---:|---:|---:|---:|",
    ]
    sizes = {"global": 1, "level": N_LEVEL, "levelgroup": N_LEVEL * len(GROUPS)}
    for arm in arms:
        r = results[arm]
        lines.append(
            f"| {arm} | {sizes[arm]} | {r['total']:.6f} | "
            f"{r['one_minus_nmae']:.6f} | {r['ficr']:.6f} |"
        )
    lines += [
        "",
        f"적합 {fits} 회 (팔 공유) / 확률행렬 digest `{prob_digest}` — 세 팔이 같은 행렬을 쓴다",
        "",
        "## 2. 타당성 가드",
        "",
        f"- V1 GLOBAL {results['global']['total']:.6f} vs C44 {C44_CONTROL} "
        f"(차 {v1_gap:.6f}, 허용 {V1_TOLERANCE}) -> **{v1}**",
        "- V2 확률행렬 단일 -> **True** (적합이 fold 당 1 회뿐)",
        "",
        "## 3. 사전확약",
        "",
        f"- H1 LEVEL > GLOBAL ({level_gain:+.6f}) -> **{h1}**",
        f"- H2 게이트 통과 -> **{h2}** {signature} "
        f"({gd['positive_months']}/{gd['months_scored']} 월, "
        f"p={gd['sign_test_p_greater']:.4f}, q05={gd['block_bootstrap_q05']:+.6f}, "
        f"최악월={gd['min_total_delta']:+.6f})",
        f"- H3 FICR 우세 (FICR {ficr_contrib:+.6f} / 1-NMAE {nmae_contrib:+.6f}) -> **{h3}**",
        f"- H4 LEVELGROUP > LEVEL "
        f"({results['levelgroup']['total'] - results['level']['total']:+.6f}) -> **{h4}**",
        f"- H5 T 가 수준에 단조 -> **{h5}**",
        "",
        "## 4. 선택된 온도 (fold-외)",
        "",
        "```",
        json.dumps(chosen["level"], indent=1, ensure_ascii=False),
        "```",
        "",
        "## 5. 판정",
        "",
        f"**{verdict}**",
        "",
        f"챔피언 로컬 {CHAMPION_LOCAL} / LEVEL {results['level']['total']:.6f}",
        "",
        f"digest `{payload['digest']}`",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("=== 완료 ===")
    for arm in arms:
        print(f"[C60] {arm:11s} p={sizes[arm]:2d}  {results[arm]['total']:.6f} "
              f"(1-NMAE {results[arm]['one_minus_nmae']:.6f} / FICR {results[arm]['ficr']:.6f})")
    print(f"[C60] V1 {v1} (차 {v1_gap:.6f})")
    print(f"[C60] LEVEL - GLOBAL {level_gain:+.6f} -> H1 {h1} / 게이트 {h2} "
          f"{signature} {gd['positive_months']}/{gd['months_scored']}월 / H5 단조 {h5}")
    print(f"[C60] 판정: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
