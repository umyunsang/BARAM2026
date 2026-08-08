"""M271 N6 — 서수성(ordinality) 축. C1N40 이 하네스 결함으로 **버린 +0.009772** 를 되찾는다.

**이 노드의 출발점은 새 착상이 아니라 폐기된 측정이다.**

  C1N40 은 정산규칙이 모양을 정한 소프트 표적(`q_i ∝ u(|c_i - y|)`)을 재고 이렇게 남겼다 —

      | 대상                    | Total    | 1-NMAE   | FICR     |
      | 배포 `M269@T0.5_G1.5`   | 0.628605 | 0.854745 | 0.402464 |
      | `CONTROL` (one-hot)     | 0.584468 | 0.840843 | 0.328093 |
      | `BAND`   (정산모양 표적) | 0.594240 | 0.843939 | 0.344541 |

      BAND - CONTROL = **+0.009772**  (FICR 기여 +0.008224 / 1-NMAE 기여 +0.001548)
      V1 `CONTROL 이 배포의 -0.03 이내` -> **False** (-0.044136)
      판정: **CLASSIFIER_RECONSTRUCTION_INVALID_AXIS_NOT_JUDGED**

  V1 이 깨졌으므로 H1~H4 를 판정하지 않은 것은 옳다. 그러나 **효과 자체가 사라진 것은
  아니다.** +0.009772 는 검출문턱 0.001013 의 **9.6 배**이고 남은 격차 0.029690 의
  **33%** 다. 지금까지 통과한 모든 축의 합(+0.001238)보다 **8 배** 크다.
  그리고 **유효한 하네스에서 다시 측정된 적이 한 번도 없다.**

  C1N40 의 하네스는 M115 기저피처만 썼다(teacher·sitewind 없음). C1N60 하네스는
  대조군이 **0.604043** 으로 배포의 -0.0246 이므로 같은 V1 형식을 통과한다.
  이 노드는 **같은 실험을 성립하는 하네스에서 다시 한다.**

**① 방법 리서치 — 기전을 바로 세운다.**

  처음에 "46-class softmax 가 표본부족"이라는 기전을 세웠다가 **실행 전에 데이터로
  반증했다.** 학습행은 fold 별 23,807 / 30,359 / 36,978 이고 클래스당 최소 표본은
  257 / 326 / 378, 중앙값 341 / 415 / 468 이다. **50 행 미만 클래스가 0 개**다.
  표본부족이 아니다. 그 기전은 폐기한다.

  실제 기전은 **서수성**이다. one-hot softmax CE 는 46 클래스를 **순서 없는 범주**로
  다룬다. 정답이 22 번인데 23 번을 예측한 것과 5 번을 예측한 것을 **똑같이** 벌한다.
  그런데 정산규칙은 순서에 전적으로 의존한다 — `|err| <= 6%` 면 어느 빈이든 단위 4 다.
  **손실이 지표의 순서구조를 전혀 모른다.**

  서수구조를 넣는 검증된 방법 두 가지를 대조한다.
  - **표적 재형성**: `q_i ∝ u(|c_i - y|)`. 정산규칙이 모양을 정하므로 임의 평활상수가
    없다. C1N40 이 이미 구현했고 +0.009772 를 냈다. (비용민감 다중분류의 표준 처리)
  - **누적분포 추정**: Koenker & Bassett (1978) pinball 손실. 각 분위가 적정 점수규칙
    으로 추정되고 CDF 는 본질적으로 순서구조를 갖는다. 교차는
    Chernozhukov, Fernandez-Val & Galichon (2010) 의 **재배열**로 교정한다(필수).
    HEFTCom2024 우승 해법(arXiv 2505.10367)이 풍력 확률예측에 쓴 방법이다.

  **벤치마크 수치는 참조값이지 요구사항이 아니다.**

**② 사양 동결**

  하네스   C1N60 과 동일. `m271_harness_cache` 로 앞단(teacher 2 종 + sitewind generic)을
           캐시하되 digest 가 재현 계약을 유지한다.
  결정층   **네 팔 모두 동일** — `sharpen(T)` 후 `bayes_decision`, T 는 fold-외 선택
           (C1N60 GLOBAL 규약). 결정층이 같으므로 차이는 **분포 표현에만** 귀속된다.
  팔
    onehot  현행 one-hot softmax CE.                              <- V1 대조군 0.604043
    band    `q_i ∝ u(|c_i - y|)` 정산모양 소프트 표적 (C1N40 재측정)
    mq19    분위 19 개(0.05..0.95) -> 재배열 -> CDF 보간 -> 46 빈
    blend   0.5*band + 0.5*mq19 (두 서수 표현의 상보성)
  분위교정 재배열 후 [0, 1] clip. 교차가 남으면 V2 발화.
  CDF변환 구간별 선형보간. **꼬리는 최외곽 구간의 밀도로 외삽**하고 [0, 1] 절단.
           (0,0)·(1,1) 앵커를 쓰면 참 sd 0.05 가 0.126 으로 복원돼 2.5 배 과대분산되고,
           과대분산은 `bayes_decision` 을 보수적으로 만들어 mq 팔을 부당하게 불리하게
           한다 — 실행 전 단위검정에서 잡아 교정했다.
  엔진     LGBM leaves 15 / lr 0.1 / 200 rounds / seed·threads 고정. 네 팔 동일.

  **타당성 가드 (사전확약보다 먼저)**
    V1  `onehot` 이 C1N60 대조군 **0.604043 을 ±0.0005 로 재현**. C1N40 이 바로 여기서
        깨져 판정을 못 했다. 벗어나면 H1~H4 를 **판정하지 않는다**.
    V2  재배열 후 분위 교차 0 건.
    V3  네 팔의 확률행렬이 정상(행합 1 오차 1e-9, NaN 없음).
    V4  네 팔이 **동일 행집합**에서 평가됨.
    V5  `band` 의 처리효과가 **C1N44 의 -0.001015 를 ±0.002 로 재현**. 이 하네스가
        C1N44 의 틀과 같음을 확인하는 **재현 대조군**이지 가설이 아니다.

  사전확약 (V1~V5 통과시에만 판정):
    H1  `mq19` > `onehot`. **이것만이 이 노드의 가설이다.**
    H2  최선 팔(`mq19`/`blend`)의 이득이 **검출문턱 0.001013 이상**.
    H3  이득이 **FICR 쪽**.
    H4  `blend` > max(`onehot`, `mq19`). 두 추정이 상보적이다.

  **`band` 를 가설에서 강등한 이유 — 이미 측정돼 있다.**
  처음에는 C1N40 의 **+0.009772** 를 유효 하네스에서 회수하는 것을 주가설로 잡았다.
  그러나 원장의 `BAND_TARGET_AND_TEMPERATURE_ARE_SUBSTITUTES` 가 이렇게 기록한다 —

      "V1 가드가 통과한 유일한 실행(사이클 44, CONTROL 0.604043)에서 정산모양 soft
       target 의 처리효과가 **-0.001015 로 음수**였다. 두 팔이 fold-외로 고른 온도가
       T>1(BAND 1.7, CONTROL 2.2) — 즉 **평활**이며 밴드 목표가 하던 일과 같다.
       대조군에 명시적 평활 자유도를 주면 밴드 목표의 암묵적 평활은 **중복**이 되고
       약간 해롭다. 둘은 **대체재**다."

  이 노드의 결정층은 바로 그 fold-외 온도선택을 쓴다. 따라서 `band` 는 **C1N44 와 같은
  조건**이고 새로 아는 것이 없다. 재현 대조군 V5 로만 남긴다. C1N40 의 +0.009772 는
  대조군에 온도 자유도가 없던 하네스의 **인공물**이다.

  **남은 진짜 가설은 `mq19` 다.** 서수구조를 표적 재형성이 아니라 **CDF 추정**으로 넣는
  경로는 이 프로젝트에서 한 번도 측정된 적이 없다. 온도 평활과 대체재가 아닌 이유는,
  평활이 분포의 **모양**을 바꾸는 데 비해 분위수회귀는 분포를 **다른 손실로 처음부터
  다시 추정**하기 때문이다. H1 이 거짓이면 CDF 추정 축이 근거를 갖고 닫힌다.

게이트 미수정. lockbox·외부데이터·`scada_ws` 미사용. 제출 없음.
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
from m271_cycle37_band_loss import KEYS
from m271_cycle40_band_classifier import (
    CLASS_WIDTH,
    N_CLASS,
    PARAMS,
    ROUNDS,
    bayes_decision,
    make_objective,
    one_hot_targets,
    soft_targets,
)
from m271_cycle44_sharpened_decision import TEMPERATURES, sharpen
from m271_evaluate_candidate import official
from m271_harness_cache import fold_frames, resolved_base_features

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
REPORT_MD = REPORTS / "m271_n6_ordinal_representation.md"
RECEIPT = REPORTS / "m271_n6_ordinal_representation_receipt.json"

NODE_ID = "C1N93_ORDINAL_REPRESENTATION"
LANE = "L3"
PARENT_NODE = "C1N40_BAND_CLASSIFIER"

DEPLOYED = "T0.5_G1.5"  # `load_predictions` 는 정책 컬럼명을 받는다
CONTROL = 0.604043
TOLERANCE = 0.0005
DETECTION_THRESHOLD = 0.001013
C1N40_BAND_EFFECT = 0.009772
C1N44_BAND_EFFECT = -0.001015
BAND_REPRO_TOLERANCE = 0.002

ARMS = ("onehot", "band", "mq19", "blend")
LEVELS = np.round(np.arange(0.05, 0.96, 0.05), 4)
EDGES = np.round(np.arange(0, N_CLASS + 1) * CLASS_WIDTH, 6)  # 0.00 .. 0.92


def quantiles_to_bins(qmat: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """분위수 행렬 -> 46 빈 확률. 재배열 -> 꼬리 외삽 -> CDF 보간 -> 빈 질량."""
    q = np.maximum.accumulate(np.clip(qmat, 0.0, 1.0), axis=1)
    lv = levels.astype("float64")
    w_lo = lv[0] * (q[:, 1] - q[:, 0]) / (lv[1] - lv[0])
    w_hi = (1.0 - lv[-1]) * (q[:, -1] - q[:, -2]) / (lv[-1] - lv[-2])
    lo = np.clip(q[:, 0] - w_lo, 0.0, None)
    hi = np.clip(q[:, -1] + w_hi, None, 1.0)
    xs = np.maximum.accumulate(
        np.concatenate([lo[:, None], q, hi[:, None]], axis=1), axis=1
    )
    ys = np.concatenate([np.zeros(1), lv, np.ones(1)])
    cuts = EDGES[:-1]  # 0.00 .. 0.90
    n = q.shape[0]
    cdf = np.empty((n, len(cuts)), dtype="float64")
    for row in range(n):
        cdf[row] = np.interp(cuts, xs[row], ys, left=0.0, right=1.0)
    prob = np.empty((n, N_CLASS), dtype="float64")
    prob[:, :-1] = np.diff(cdf, axis=1)
    prob[:, -1] = 1.0 - cdf[:, -1]  # 최상단 빈이 잔여 전부 (라벨 clip 규약과 일치)
    prob = np.clip(prob, 0.0, None)
    prob /= prob.sum(axis=1, keepdims=True)
    return prob


def main() -> int:
    # **비싼 적합 전에 외부 의존을 전부 건드린다.** 첫 실행에서 이 조회가 적합 63 회
    # 뒤에 있어 10 분치 계산을 버렸다. 실패할 수 있는 I/O 는 앞에 온다.
    parent = load_predictions(DEPLOYED)
    parent_score = official(parent)

    store_raw, harness_digest, cache_hit = fold_frames()
    folds = sorted(store_raw)
    base_features = resolved_base_features(
        store_raw[folds[0]]["train"], store_raw[folds[0]]["sitewind_names"]
    )

    store: dict[str, dict[str, Any]] = {}
    fits = 0
    crossings = 0

    for fold in folds:
        entry = store_raw[fold]
        train, test = entry["train"], entry["test"]
        features = [*base_features, *entry["sitewind_names"]]
        x_tr = train.loc[:, features].astype("float32")
        x_te = test.loc[:, features].astype("float32")
        rate = np.clip(train["rate"].to_numpy(dtype="float64"), 0.0, None)
        label = np.clip((rate / CLASS_WIDTH).astype(int), 0, N_CLASS - 1)

        cell: dict[str, Any] = {
            "meta": test.loc[:, [*KEYS, "actual_kwh"]].copy(),
            "capacity": test["capacity"].to_numpy(dtype="float64"),
            "group": test["group_id"].to_numpy(),
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
        }

        # --- 분류기 두 팔: 표적만 다르고 나머지는 전부 동일 ---
        for arm, target in (
            ("onehot", one_hot_targets(rate)),
            ("band", soft_targets(rate)),
        ):
            dataset = lgb.Dataset(x_tr, label=label, free_raw_data=False)
            params = dict(PARAMS)
            params["objective"] = make_objective(target)
            booster = lgb.train(params, dataset, num_boost_round=ROUNDS)
            fits += 1
            raw = np.asarray(booster.predict(x_te)).reshape(len(test), N_CLASS)
            exp = np.exp(raw - raw.max(axis=1, keepdims=True))
            cell[arm] = exp / exp.sum(axis=1, keepdims=True)

        # --- 분위수회귀 19 개 ---
        qcols: list[np.ndarray] = []
        for tau in LEVELS:
            qparams = {
                k: v for k, v in PARAMS.items() if k not in ("objective", "num_class")
            }
            qparams["objective"] = "quantile"
            qparams["alpha"] = float(tau)
            qb = lgb.train(
                qparams,
                lgb.Dataset(x_tr, label=rate, free_raw_data=False),
                num_boost_round=ROUNDS,
            )
            fits += 1
            qcols.append(np.asarray(qb.predict(x_te), dtype="float64"))
        q19 = np.column_stack(qcols)
        crossings += int((np.diff(q19, axis=1) < -1e-12).sum())
        cell["mq19"] = quantiles_to_bins(q19, LEVELS)
        cell["blend"] = 0.5 * cell["onehot"] + 0.5 * cell["mq19"]
        store[fold] = cell

    v3 = bool(
        all(
            np.isfinite(store[f][a]).all()
            and abs(store[f][a].sum(axis=1) - 1.0).max() < 1e-9
            for f in folds
            for a in ARMS
        )
    )

    def scored(fold: str, arm: str, temperature: float) -> pd.DataFrame:
        cell = store[fold]
        out = cell["meta"].copy()
        out["prediction_kwh"] = (
            bayes_decision(sharpen(cell[arm], temperature)) * cell["capacity"]
        )
        out["group_id"] = cell["group"]
        out["month"] = out["forecast_kst_dtm"].dt.to_period("M").astype(str)
        return out

    # C1N60 GLOBAL 규약: 보류 fold 의 T 를 나머지 fold 에서 고른다.
    results: dict[str, dict[str, Any]] = {}
    frames: dict[str, pd.DataFrame] = {}
    chosen: dict[str, dict[str, float]] = {}
    per_fold: dict[str, dict[str, Any]] = {
        f: {
            "train_rows": store[f]["train_rows"],
            "test_rows": store[f]["test_rows"],
            "total": {},
        }
        for f in folds
    }
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
            piece = scored(held, arm, float(best_t))
            per_fold[held]["total"][arm] = float(official(piece)["total"])
            pieces.append(piece)
        frames[arm] = pd.concat(pieces, ignore_index=True)
        results[arm] = official(frames[arm])
    for f in folds:
        per_fold[f]["delta_vs_onehot"] = {
            a: per_fold[f]["total"][a] - per_fold[f]["total"]["onehot"] for a in ARMS
        }

    v1 = bool(abs(results["onehot"]["total"] - CONTROL) <= TOLERANCE)
    v2 = crossings == 0
    key_sets = {
        a: sorted(map(tuple, frames[a].loc[:, KEYS].to_numpy().tolist())) for a in ARMS
    }
    v4 = all(key_sets[a] == key_sets["onehot"] for a in ARMS)
    valid = v1 and v2 and v3 and v4

    gains = {a: results[a]["total"] - results["onehot"]["total"] for a in ARMS}
    # `band` 는 재현 대조군이므로 최선 팔 후보에서 뺀다.
    best_arm = max(("mq19", "blend"), key=lambda a: gains[a])
    best_gain = gains[best_arm]
    ficr_contrib = 0.5 * (results[best_arm]["ficr"] - results["onehot"]["ficr"])
    nmae_contrib = 0.5 * (
        results[best_arm]["one_minus_nmae"] - results["onehot"]["one_minus_nmae"]
    )

    gate = evaluate_gate(frames[best_arm], frames["onehot"])
    gd = gate.evidence
    flags = {label.split()[0]: bool(ok) for label, ok in gate.conditions.items()}
    signature = "[" + "".join("O" if flags[g] else "-" for g in sorted(flags)) + "]"

    # V5 는 재현 대조군이다. C1N44 가 같은 대조군 0.604043 에서 밴드 표적의 처리효과를
    # -0.001015 로 관측했다. 재현되면 이 하네스가 C1N44 의 틀과 같음이 확인된다.
    v5 = bool(abs(gains["band"] - C1N44_BAND_EFFECT) <= BAND_REPRO_TOLERANCE)
    valid_local = valid and v5

    if valid_local:
        h1: bool | None = bool(gains["mq19"] > 0.0)
        h2: bool | None = bool(best_gain >= DETECTION_THRESHOLD)
        h3: bool | None = bool(ficr_contrib > nmae_contrib)
        h4: bool | None = bool(
            results["blend"]["total"]
            > max(results["onehot"]["total"], results["mq19"]["total"])
        )
        if h2 and (h1 or h4):
            verdict = "CDF_ESTIMATOR_AXIS_OPEN"
        elif h1 or h4:
            verdict = "CDF_ESTIMATOR_SUBTHRESHOLD"
        else:
            verdict = "SOFTMAX_BEATS_QUANTILE_CDF_ESTIMATOR_AXIS_CLOSES"
    else:
        h1 = h2 = h3 = h4 = None
        verdict = "GUARD_FAILED_RESULT_VOID"

    check = {
        "V1_expectation": f"onehot 이 C1N60 대조군 {CONTROL} 을 +-{TOLERANCE} 로 재현",
        "V1_held": v1,
        "V1_measured": results["onehot"]["total"],
        "V2_expectation": "재배열 후 분위 교차 0 건",
        "V2_held": v2,
        "V2_measured": crossings,
        "V3_expectation": "네 팔 확률행렬 정상",
        "V3_held": v3,
        "V4_expectation": "네 팔 동일 행집합",
        "V4_held": v4,
        "V5_expectation": f"band 가 C1N44 의 {C1N44_BAND_EFFECT} 를 +-{BAND_REPRO_TOLERANCE} 로 재현",
        "V5_held": v5,
        "V5_measured": gains["band"],
        "H1_expectation": "mq19 > onehot",
        "H1_held": h1,
        "H1_measured": gains["mq19"],
        "H2_expectation": f"최선 팔 이득 >= {DETECTION_THRESHOLD}",
        "H2_held": h2,
        "H2_measured": best_gain,
        "H3_expectation": "이득이 FICR 쪽",
        "H3_held": h3,
        "H3_ficr_contrib": ficr_contrib,
        "H3_nmae_contrib": nmae_contrib,
        "H4_expectation": "blend > max(onehot, mq19)",
        "H4_held": h4,
        "H4_measured": gains["blend"],
        "judged": valid_local,
        "verdict": verdict,
    }

    receipt: dict[str, Any] = {
        "node_id": NODE_ID,
        "lane": LANE,
        "parent": PARENT_NODE,
        "judged_at": datetime.now(UTC).isoformat(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "harness_digest": harness_digest,
        "harness_cache_hit": cache_hit,
        "gate_version": GATE_VERSION,
        "fits": fits,
        "folds": folds,
        "feature_count": len(base_features)
        + len(store_raw[folds[0]]["sitewind_names"]),
        "levels": LEVELS.tolist(),
        "temperatures_chosen": chosen,
        "deployed": {"candidate": DEPLOYED, **parent_score},
        "arms": {a: dict(results[a]) for a in ARMS},
        "gains_vs_onehot": gains,
        "per_fold": per_fold,
        "best_arm": best_arm,
        "gate": {"passed": gate.passed, "signature": signature, "evidence": gd},
        "precommitment": check,
    }
    RECEIPT.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )

    arm_rows = "\n".join(
        f"| `{a}` | {results[a]['total']:.6f} | {results[a]['one_minus_nmae']:.6f} "
        f"| {results[a]['ficr']:.6f} | {gains[a]:+.6f} |"
        for a in ARMS
    )
    fold_rows_md = "\n".join(
        f"| `{f}` | {per_fold[f]['train_rows']:,} | {per_fold[f]['test_rows']:,} "
        f"| {per_fold[f]['total']['onehot']:.6f} "
        f"| {per_fold[f]['delta_vs_onehot']['band']:+.6f} "
        f"| {per_fold[f]['delta_vs_onehot']['mq19']:+.6f} "
        f"| {per_fold[f]['delta_vs_onehot']['blend']:+.6f} |"
        for f in folds
    )
    REPORT_MD.write_text(
        f"""# M271 N6 — 서수성 축: 분포를 CDF 로 추정하면 달라지는가

- 판정일: {receipt['judged_at']}
- 노드: `{NODE_ID}` / 레인 {LANE} / 부모 `{PARENT_NODE}`
- 하네스 digest `{harness_digest}` (캐시 적중 {cache_hit}) / 적합 {fits} 회
- **결정층은 네 팔 모두 `sharpen(T)` + `bayes_decision`, T 는 fold-외 선택.**
  차이는 분포 표현에만 귀속된다.
- `band` 는 **가설이 아니라 재현 대조군(V5)** 이다. C1N44 가 같은 대조군에서
  {C1N44_BAND_EFFECT} 를 관측했고, C1N40 의 +{C1N40_BAND_EFFECT} 는 대조군에 온도
  자유도가 없던 하네스의 인공물이다.

## 1. 측정

| 팔 | Total | 1-NMAE | FICR | onehot 대비 |
|---|---:|---:|---:|---:|
{arm_rows}

배포 `{DEPLOYED}` = {parent_score['total']:.6f} / C1N60 대조군 {CONTROL}

## 2. fold 별

| fold | 학습행 | 평가행 | onehot | band | mq19 | blend |
|---|---:|---:|---:|---:|---:|---:|
{fold_rows_md}

## 3. 타당성 가드

| 가드 | 기대 | 실측 | 판정 |
|---|---|---|:---:|
| V1 | onehot 이 {CONTROL} 재현 | {results['onehot']['total']:.6f} | {'O' if v1 else 'X'} |
| V2 | 분위 교차 0 건 | {crossings} | {'O' if v2 else 'X'} |
| V3 | 확률행렬 정상 | — | {'O' if v3 else 'X'} |
| V4 | 동일 행집합 | — | {'O' if v4 else 'X'} |
| V5 | band 가 C1N44 의 {C1N44_BAND_EFFECT} 재현 | {gains['band']:+.6f} | {'O' if v5 else 'X'} |

## 4. 사전확약 대조

- H1 `mq19 > onehot` -> **{h1}** (실측 {gains['mq19']:+.6f})
- H2 `최선 팔 이득 >= {DETECTION_THRESHOLD}` -> **{h2}** (실측 {best_gain:+.6f}, 팔 `{best_arm}`)
- H3 `이득이 FICR 쪽` -> **{h3}** (FICR {ficr_contrib:+.6f} / 1-NMAE {nmae_contrib:+.6f})
- H4 `blend > max(onehot, mq19)` -> **{h4}** (실측 {gains['blend']:+.6f})

동결 게이트(최선 팔 `{best_arm}` vs `onehot`): {signature} **{'통과' if gate.passed else '기각'}**

판정: **{verdict}**
""",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "verdict": verdict,
                "arms": {a: results[a]["total"] for a in ARMS},
                "gains": gains,
                "guards": {"V1": v1, "V2": v2, "V3": v3, "V4": v4},
                "gate": signature,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
