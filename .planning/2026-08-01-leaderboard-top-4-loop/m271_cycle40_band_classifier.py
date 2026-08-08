"""M271 P4 사이클 40 — 밴드 인지 학습을 **분류기 틀 안에서**.

사이클 39 가 사이클 37·38 의 오배치를 특정했다. 직접 점회귀는 분류기+정책보다
**0.086202** 열세이므로, 그 위에 손실을 얹으면 손실이 아무리 좋아도 못 따라잡는다.
올바른 실험은 분류기 틀 **안에서** 밴드 인지 학습이다.

착상
----
표준 multiclass CE 는 정답 빈에 one-hot 을 놓고 **모든 오분류를 똑같이** 벌한다. 그런데
정산 규칙은 다르다 — 밴드 안(`|err| <= 6%`)이면 어느 빈이든 정산단위 4 로 **같은 값**이고,
6~8% 면 3, 밖이면 0 이다. 그러면 목표분포를 그 모양으로 놓는 것이 정산 효용 하의 Bayes
목표다.

    q_i  ∝  u(|c_i - y|),   u(d) = 4 if d<=0.06,  3 if d<=0.08,  0 otherwise

이것은 **정산 규칙이 모양을 정한 label smoothing** 이다. 임의의 평활 상수가 없다.
구현도 사소하다: softmax CE 의 경사는 one-hot 이든 soft 든 `p - q` 로 같은 꼴이다.

결정 규칙도 역공학하지 않는다. 예측 분포에서 **공식 지표 기여 기대값을 최대화하는 점**을
격자 탐색으로 고른다(사이클 37 에서 유도한 상대 가중치 그대로).

    x* = argmax_x  sum_i p_i * [ (1/4)*(c_i/cbar)*u(|x - c_i|) - |x - c_i| ]

두 팔(CONTROL/BAND)에 **같은 결정 규칙**을 쓰므로 차이는 학습 목표에만 귀속된다.

① 방법 리서치 (실행 전)
  - 비용민감 다중분류(cost-sensitive multiclass): 오분류 비용이 균일하지 않으면 손실에
    비용을 반영해야 한다. 여기서는 비용이 **공식 규칙에서 그대로** 나온다.
  - 사이클 17 의 기각 사유(평가 fold 에서 결합 가중치 적합)는 여기 적용되지 않는다.
    학습은 fold-외이고 목표분포는 규칙에서 유도된다.
  - 사이클 38 의 V1 가드를 유지·강화한다. **대조군이 온전하지 않으면 판정하지 않는다.**

② 사양 동결

  표현   46 class, 폭 0.02 (배포와 동일). 빈 중심 `c_i = (i+0.5)*0.02`
  목표   CONTROL = one-hot / BAND = `q_i ∝ u(|c_i - y|)` (정규화, 전부 0 이면 one-hot 후퇴)
  손실   softmax CE. `grad = p - q`, `hess = max(2*p*(1-p), 1e-6)`. 두 팔 동일
  피처   M115 의 100 개 중 캐시 가용 87 개 (사이클 39 가 커버리지 무관함을 확인: 87 vs
         1,347 차이 0.000726). **`scada_ws` 는 누출이므로 어떤 경우에도 제외**
  결정   위 Bayes 규칙, 격자 `x in [0, 1]` 간격 0.005. 두 팔 동일
  분할   fold 별 chronology-safe (사이클 37·38 과 동일)

  **타당성 가드 (사전확약보다 먼저)**
    V1  CONTROL 의 pooled Total 이 배포(0.628605)의 `-0.03` 이내.
        기각되면 분류기 재구성이 깨진 것이므로 **H1~H4 를 판정하지 않는다.**

  사전확약(실행 전 동결, V1 통과시에만 판정):
    H1  BAND > CONTROL. (목표분포만 다르므로 순수 귀속)
    H2  BAND 가 배포 대비 개선 + **동결 게이트 통과**.
    H3  BAND > `M115@T0.6_G0.2` (0.630310).
    H4  이득이 **FICR 쪽**에서 나온다.

**게이트를 수정하지 않는다.** 2024 행·lockbox·`scada_ws` 미사용.
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
from m271_cycle21_mos import QUARTER_OF_MONTH
from m271_cycle37_band_loss import BAND_HIT, BAND_PARTIAL, KEYS, PROBE, fold_rows
from m271_evaluate_candidate import official
from m271_n0_common import SEED
from run_sequence_classifier import _surface

from baram.constants import CAPACITIES_KWH

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
REPORT_MD = REPORTS / "m271_cycle40_band_classifier.md"
RECEIPT = REPORTS / "m271_cycle40_band_classifier_receipt.json"

NODE_ID = "C1N40_BAND_CLASSIFIER"
LANE = "L3"
PARENT_NODE = "C1N39_ARCHITECTURE_GAP"
DEPLOYED = "T0.5_G1.5"
DEPLOYED_TOTAL = 0.628605
M115_FIXED_TOTAL = 0.630310
FOLDS = ("Q2", "Q3", "Q4")
LEAKY_COLUMNS = ("scada_ws",)

N_CLASS = 46
CLASS_WIDTH = 0.02
CENTERS = (np.arange(N_CLASS) + 0.5) * CLASS_WIDTH
DECISION_GRID = np.round(np.arange(0.0, 1.0 + 1e-9, 0.005), 4)
V1_TOLERANCE = 0.03

PARAMS = {
    "objective": "multiclass",
    "num_class": N_CLASS,
    "num_leaves": 15,
    "learning_rate": 0.1,
    "min_child_samples": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "seed": SEED,
    "num_threads": 1,
    "deterministic": True,
    "force_row_wise": True,
    "verbose": -1,
}
ROUNDS = 200


def settlement_unit_vec(distance: np.ndarray) -> np.ndarray:
    return np.where(distance <= BAND_HIT, 4.0, np.where(distance <= BAND_PARTIAL, 3.0, 0.0))


def soft_targets(rate: np.ndarray) -> np.ndarray:
    """q_i ∝ u(|c_i - y|). 정산 규칙이 모양을 정한 label smoothing."""
    distance = np.abs(CENTERS[None, :] - rate[:, None])
    weight = settlement_unit_vec(distance)
    total = weight.sum(axis=1, keepdims=True)
    onehot = np.zeros_like(weight)
    idx = np.clip((rate / CLASS_WIDTH).astype(int), 0, N_CLASS - 1)
    onehot[np.arange(len(rate)), idx] = 1.0
    return np.where(total > 0, weight / np.maximum(total, 1e-12), onehot)


def one_hot_targets(rate: np.ndarray) -> np.ndarray:
    out = np.zeros((len(rate), N_CLASS))
    idx = np.clip((rate / CLASS_WIDTH).astype(int), 0, N_CLASS - 1)
    out[np.arange(len(rate)), idx] = 1.0
    return out


def make_objective(target: np.ndarray):
    def objective(preds: np.ndarray, dataset: lgb.Dataset):
        raw = preds if preds.ndim == 2 else preds.reshape(N_CLASS, -1).T
        shifted = raw - raw.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        p = exp / exp.sum(axis=1, keepdims=True)
        grad = p - target
        hess = np.maximum(2.0 * p * (1.0 - p), 1e-6)
        if preds.ndim == 1:
            return grad.T.reshape(-1), hess.T.reshape(-1)
        return grad, hess

    return objective


def bayes_decision(prob: np.ndarray) -> np.ndarray:
    """분포에서 공식 지표 기여 기대값을 최대화하는 점. 두 팔에 동일 적용."""
    cbar = float(CENTERS.mean())
    # utility[g, i] = (1/4)*(c_i/cbar)*u(|x_g - c_i|) - |x_g - c_i|
    distance = np.abs(DECISION_GRID[:, None] - CENTERS[None, :])
    utility = 0.25 * (CENTERS[None, :] / cbar) * settlement_unit_vec(distance) - distance
    scores = prob @ utility.T  # (n_rows, n_grid)
    return DECISION_GRID[np.argmax(scores, axis=1)]


def by_fold_total(frame: pd.DataFrame) -> dict[str, float]:
    f = frame.copy()
    f["fold"] = f["month"].map(QUARTER_OF_MONTH)
    return {
        fold: official(cell)["total"]
        for fold, cell in f.groupby("fold", observed=True)
        if fold in FOLDS
    }


def main() -> int:
    surface, _, _ = _surface()
    surface["forecast_kst_dtm"] = pd.to_datetime(surface["forecast_kst_dtm"])
    wanted = json.loads(
        (PROBE / "M115_XGBOOST-dev-2023-Q3.json").read_text(encoding="utf-8")
    )["selected_feature_names"]
    features = [c for c in wanted if c in surface.columns and c not in LEAKY_COLUMNS]
    assert all(c not in features for c in LEAKY_COLUMNS), "누출 컬럼이 섞였다"

    surface["capacity"] = surface["group_id"].map(CAPACITIES_KWH).astype(float)
    surface["rate"] = surface["actual_kwh"] / surface["capacity"]
    surface = surface.loc[surface["rate"].notna()].reset_index(drop=True)

    pieces: dict[str, list[pd.DataFrame]] = {"BAND": [], "CONTROL": []}
    fits = 0
    for _probe_fold, meta in fold_rows().items():
        train = surface.loc[surface["forecast_kst_dtm"] < meta["start"]]
        test_mask = np.array(
            [
                (fid, gid) in meta["keys"]
                for fid, gid in zip(surface["forecast_id"], surface["group_id"],
                                    strict=True)
            ]
        )
        test = surface.loc[test_mask]
        assert len(test) > 0

        x_tr = train.loc[:, features].astype("float32")
        rate = np.clip(train["rate"].to_numpy(dtype="float64"), 0.0, None)
        label = np.clip((rate / CLASS_WIDTH).astype(int), 0, N_CLASS - 1)

        for name, target in (
            ("BAND", soft_targets(rate)),
            ("CONTROL", one_hot_targets(rate)),
        ):
            dataset = lgb.Dataset(x_tr, label=label, free_raw_data=False)
            params = dict(PARAMS)
            params["objective"] = make_objective(target)
            booster = lgb.train(params, dataset, num_boost_round=ROUNDS)
            fits += 1
            raw = booster.predict(test.loc[:, features].astype("float32"))
            raw = np.asarray(raw).reshape(len(test), N_CLASS)
            shifted = raw - raw.max(axis=1, keepdims=True)
            exp = np.exp(shifted)
            prob = exp / exp.sum(axis=1, keepdims=True)
            out = test.loc[:, [*KEYS, "actual_kwh"]].copy()
            out["prediction_kwh"] = bayes_decision(prob) * test["capacity"].to_numpy()
            pieces[name].append(out)

    frames, results = {}, {}
    for name, parts in pieces.items():
        frame = pd.concat(parts, ignore_index=True)
        frame["month"] = frame["forecast_kst_dtm"].dt.to_period("M").astype(str)
        frames[name] = frame
        results[name] = official(frame)

    parent = load_predictions(DEPLOYED)
    parent_score = official(parent)
    v1_gap = results["CONTROL"]["total"] - parent_score["total"]
    v1 = bool(v1_gap >= -V1_TOLERANCE)

    delta = results["BAND"]["total"] - results["CONTROL"]["total"]
    ficr_contrib = 0.5 * (results["BAND"]["ficr"] - results["CONTROL"]["ficr"])
    nmae_contrib = 0.5 * (
        results["BAND"]["one_minus_nmae"] - results["CONTROL"]["one_minus_nmae"]
    )
    gate = evaluate_gate(frames["BAND"], parent)
    gd = gate.evidence

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
                  else "BAND_AWARE_TRAINING_CLOSED")
        )
    else:
        h1 = h2 = h3 = h4 = None
        verdict = "CLASSIFIER_RECONSTRUCTION_INVALID_AXIS_NOT_JUDGED"

    by_fold = {name: by_fold_total(frame) for name, frame in frames.items()}
    check = {
        "V1_expectation": f"CONTROL 이 배포의 -{V1_TOLERANCE} 이내",
        "V1_held": v1, "V1_measured_gap": v1_gap,
        "H1_expectation": "BAND > CONTROL", "H1_held": h1, "H1_measured": delta,
        "H2_expectation": "BAND 가 배포 대비 개선 + 게이트 통과", "H2_held": h2,
        "H3_expectation": f"BAND > {M115_FIXED_TOTAL}", "H3_held": h3,
        "H4_expectation": "이득이 FICR 쪽", "H4_held": h4,
        "judged": v1, "verdict": verdict,
    }

    payload = {
        "node": NODE_ID, "parent_node": PARENT_NODE,
        "gate_version": GATE_VERSION, "gate_modified": False, "lockbox_used": False,
        "leaky_columns_excluded": list(LEAKY_COLUMNS),
        "design": {
            "representation": f"{N_CLASS} class, width {CLASS_WIDTH}",
            "control_target": "one-hot",
            "band_target": "q_i ∝ u(|c_i - y|), 정산 규칙이 모양을 정한 label smoothing",
            "loss": "softmax CE, grad = p - q, hess = max(2p(1-p), 1e-6)",
            "decision": "Bayes: argmax_x E_p[(1/4)(c/cbar)u(|x-c|) - |x-c|]",
            "same_decision_both_arms": True,
        },
        "features_used": len(features),
        "params": PARAMS, "rounds": ROUNDS, "fits": fits,
        "scores": {**results, "deployed": parent_score},
        "by_fold": by_fold,
        "delta_band_minus_control": delta,
        "contribution": {"ficr": ficr_contrib, "one_minus_nmae": nmae_contrib},
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
        "# M271 P4 사이클 40 — 밴드 인지 학습, 분류기 틀 안에서",
        "",
        f"- 판정일: {datetime.now(UTC).strftime('%Y-%m-%d')} (UTC)",
        f"- 노드: `{NODE_ID}` / 레인 {LANE} / 부모 `{PARENT_NODE}`",
        f"- 게이트: `{GATE_VERSION}` (읽기만 함) / lockbox·`scada_ws` 미사용",
        "",
        "## 1. 설계",
        "",
        f"- 표현: {payload['design']['representation']}",
        f"- CONTROL 목표: {payload['design']['control_target']}",
        f"- BAND 목표: {payload['design']['band_target']}",
        f"- 손실: `{payload['design']['loss']}`",
        f"- 결정: `{payload['design']['decision']}` — **두 팔 동일**",
        f"- 피처 {len(features)} 개, 적합 {fits} 회 ({ROUNDS} rounds x {N_CLASS} class)",
        "",
        "## 2. 타당성 가드 (V1)",
        "",
        f"CONTROL {results['CONTROL']['total']:.6f} vs 배포 {parent_score['total']:.6f} "
        f"-> **{v1_gap:+.6f}**, 허용 `-{V1_TOLERANCE}` -> **{v1}**",
        "",
        "## 3. 결과",
        "",
        "| 모델 | Total | 1-NMAE | FICR |",
        "|---|---:|---:|---:|",
        f"| 배포 `M269@{DEPLOYED}` | {parent_score['total']:.6f} | "
        f"{parent_score['one_minus_nmae']:.6f} | {parent_score['ficr']:.6f} |",
        f"| `CONTROL` (one-hot) | {results['CONTROL']['total']:.6f} | "
        f"{results['CONTROL']['one_minus_nmae']:.6f} | {results['CONTROL']['ficr']:.6f} |",
        f"| **`BAND`** (정산모양 목표) | **{results['BAND']['total']:.6f}** | "
        f"{results['BAND']['one_minus_nmae']:.6f} | {results['BAND']['ficr']:.6f} |",
        "",
        f"BAND - CONTROL = **{delta:+.6f}** "
        f"(FICR 기여 {ficr_contrib:+.6f} / 1-NMAE 기여 {nmae_contrib:+.6f})",
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
        f"BAND 대 배포 게이트: `{flags}` {g['positive_months']}/{g['months_scored']}월 "
        f"p={g['sign_test_p']:.4f} q05={g['bootstrap_q05']:+.6f} -> "
        f"**{'통과' if g['passed'] else '기각'}**",
        "",
        "## 4. 사전확약 대조",
        "",
        f"- V1 `{check['V1_expectation']}` -> **{v1}** ({v1_gap:+.6f})",
        f"- H1 `{check['H1_expectation']}` -> **{h1 if h1 is not None else '판정안함'}**",
        f"- H2 `{check['H2_expectation']}` -> **{h2 if h2 is not None else '판정안함'}**",
        f"- H3 `{check['H3_expectation']}` -> **{h3 if h3 is not None else '판정안함'}**",
        f"- H4 `{check['H4_expectation']}` -> **{h4 if h4 is not None else '판정안함'}**",
        "",
        f"판정: **{verdict}**",
        "",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    receipt: dict[str, Any] = {
        "schema_version": 1,
        "stage": "M271_P4_CYCLE40_BAND_CLASSIFIER",
        "node": NODE_ID, "lane": LANE,
        "decided_utc": datetime.now(UTC).isoformat(),
        "result": payload,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "report_sha256": hashlib.sha256(REPORT_MD.read_bytes()).hexdigest(),
        "dacon_upload": False, "external_actions": [],
        "model_fits": fits,
        "lockbox_reopened": False, "new_2024_evaluation": False,
    }
    RECEIPT.write_text(
        json.dumps(receipt, indent=1, ensure_ascii=False, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

    print(f"[C40] 피처 {len(features)} / 적합 {fits} 회 / {N_CLASS} class x {ROUNDS} rounds")
    print(f"[C40] 배포    {parent_score['total']:.6f}")
    print(f"[C40] CONTROL {results['CONTROL']['total']:.6f} "
          f"(배포 대비 {v1_gap:+.6f}) -> V1 {v1}")
    print(f"[C40] BAND    {results['BAND']['total']:.6f} "
          f"(CONTROL 대비 {delta:+.6f})")
    print(f"[C40] 기여 FICR {ficr_contrib:+.6f} / 1-NMAE {nmae_contrib:+.6f}")
    print(f"[C40] 게이트 [{flags}] {g['positive_months']}/{g['months_scored']}월 -> "
          f"{'통과' if g['passed'] else '기각'}")
    print(f"[C40] 판정: {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
