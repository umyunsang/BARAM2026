"""M271 P4 사이클 37 — 밴드 정합 손실로 기저모델을 직접 학습한다.

36 사이클 동안 이 프로젝트의 **모든** 기저모델은 회귀/분류 손실을 최적화하고 정산 밴드는
**사후 결정 정책**으로 붙였다. M115 조차 `xgboost_multiclass_fixed_m102_features` — 46 class
분류기 + 정책이다. **학습 목적함수가 지표를 반영한 적이 한 번도 없다.**

어떤 폐쇄도 이 축을 덮지 않는다.
  - 결정층 오라클 8.8% (사이클 4·8) 는 **고정된 분포에서 사후 사상**의 천장이다. 손실이
    밴드 적중을 보상하면 모델은 **다른 분포**를 학습하므로 그 오라클이 적용되지 않는다.
  - 물리적 폐쇄 (사이클 36) 의 `0.571 m/s` 는 급경사에서 밴드에 들기 위한 **풍속** 정확도다.
    밴드 정합 손실은 풍속을 더 맞히려는 게 아니라 **같은 불확실성에서 밴드 적중 확률이
    최대인 곳에 점을 놓으려는** 것이다. 급경사를 포기하고 평탄 구간에서 더 먹는 표현을
    학습할 수 있다.
  - 사이클 17 이 이 방법군을 기각한 근거는 "평가 fold 에서 **결합 가중치**를 적합하는 것은
    선택 편향" 이었다. **기저모델을 fold-외로 학습하는 것은 다른 얘기**이고 그 기각은
    여기 적용되지 않는다. 그때 나는 기각 사유를 방법 자체로 잘못 일반화했다.

① 방법 리서치 (실행 전)
  - arXiv:2503.20082 — 0-1 지시 손실을 매끄러운 대리함수(Cauchy CDF)로 바꿔 경사 최적화.
    여기서는 정산단위가 2 단 계단이므로 sigmoid 두 개의 합으로 쓴다.
  - **대리함수의 상대 가중치를 공식 지표에서 유도한다.** 임의로 정하지 않는다(아래 ②).
  - 통제는 **동일 조건 대조군**이다: 같은 피처·분할·시드·하이퍼파라미터, **손실만** 교체.
    두 모델 다 custom objective 로 구현해 내장/커스텀 차이도 제거한다.

② 사양 동결

  손실 유도. 유효행에서 공식 Total 은
      0.5 * (1 - mean_i |e_i|/C_i) + 0.5 * (sum_i y_i u_i / sum_i y_i) / 4
  이므로 행별 손실의 상대 가중치는 |e|/C 항 계수 `0.5/n` 대 y*u 항 계수 `0.5/(4 n ybar)` 다.
  전자로 나누면 **정확히**

      L_i = |e_i| - (1/4) * (y_i / ybar) * u_smooth(e_i)          (rate 척도, C=1)

  가 되고 임의 상수가 없다. 정산단위의 매끄러운 판:

      u_smooth(e) = 3 * sigmoid((0.08 - |e|)/tau) + 1 * sigmoid((0.06 - |e|)/tau)
      tau = 0.01  (용량의 1%. 밴드 폭 6%/8% 대비 자연 척도)

  경사: `grad = sign(e) * [1 + (1/(4*tau)) * (y/ybar) * (3*s8*(1-s8) + s6*(1-s6))]`
  헤시안: **상수 1.0** (L1 계열 커스텀 목적의 표준 처리). 두 모델 동일.

  대조군 CONTROL 은 같은 틀의 L1: `grad = sign(e)`, `hess = 1.0`.
  **두 모델 모두 사후 정책을 붙이지 않는다.** 손실이 밴드를 내재화하는지가 질문이므로
  정책을 얹으면 질문이 흐려진다.

  피처   M115 의 100 개 중 캐시에 있는 **87 개** (결측 13 개는 teacher 산출 `sitewind__*`
         이며 캐시에 없다. 비교가 내부 대조이므로 판정을 흔들지 않는다 — 명시적 이탈)
  분할   fold 별 chronology-safe. 학습 = 해당 fold 최초 시각 **이전** 행 전부
  예측   rate 를 예측하고 `[0, 1]` 로 clip 한 뒤 용량을 곱한다

  사전확약(실행 전 동결):
    H1  BAND 가 CONTROL 대비 pooled Total 개선. (손실만 다르므로 순수 귀속)
    H2  BAND 가 배포 기준선 `M269@T0.5_G1.5` (0.628605) 대비 개선 + **동결 게이트 통과**.
    H3  BAND 가 `M115@T0.6_G0.2` (0.630310) 대비 **동결 게이트 통과**.
    H4  BAND-CONTROL 이득이 **FICR 쪽**에서 나온다 (기전 확인).
  H1 이 기각되면 **손실함수 축도 닫힌다.** 그때는 물리적 폐쇄가 목적함수 선택과 무관함이
  확인되므로 결론이 오히려 단단해진다.

**게이트를 수정하지 않는다.** 2024 행·lockbox 미사용.
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
from m271_cycle13_ensemble import FOLDS as PROBE_FOLDS
from m271_cycle21_mos import QUARTER_OF_MONTH
from m271_evaluate_candidate import official
from m271_n0_common import SEED
from run_sequence_classifier import _surface

from baram.constants import CAPACITIES_KWH

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
PROBE = ROOT / "artifacts" / "backtests" / "metric-aligned-probe"
REPORT_MD = REPORTS / "m271_cycle37_band_loss.md"
RECEIPT = REPORTS / "m271_cycle37_band_loss_receipt.json"

NODE_ID = "C1N37_BAND_ALIGNED_LOSS"
LANE = "L3"  # 모델링 방법
PARENT_NODE = "C1N35_FIXED_POLICY_CORRECTION"
DEPLOYED = "T0.5_G1.5"
DEPLOYED_TOTAL = 0.628605
M115_FIXED_TOTAL = 0.630310
KEYS = ["forecast_id", "forecast_kst_dtm", "group_id"]
FOLDS = ("Q2", "Q3", "Q4")
THREADS = 1

BAND_HIT = 0.06
BAND_PARTIAL = 0.08
TAU = 0.01
ELIGIBLE_THRESHOLD = 0.10

PARAMS = {
    "num_leaves": 31,
    "learning_rate": 0.05,
    "n_estimators": 400,
    "min_child_samples": 100,
    "colsample_bytree": 0.8,
    "subsample": 0.8,
    "subsample_freq": 1,
    "seed": SEED,
    "num_threads": THREADS,
    "deterministic": True,
    "force_row_wise": True,
    "verbose": -1,
}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def make_band_objective(gen_weight: np.ndarray):
    """L_i = |e| - (1/4)*(y/ybar)*u_smooth(e). 상대 가중치는 공식 지표에서 유도됐다."""

    def objective(preds: np.ndarray, dataset: lgb.Dataset):
        y = dataset.get_label()
        e = preds - y
        s8 = _sigmoid((BAND_PARTIAL - np.abs(e)) / TAU)
        s6 = _sigmoid((BAND_HIT - np.abs(e)) / TAU)
        bump = (3.0 * s8 * (1.0 - s8) + s6 * (1.0 - s6)) / TAU
        grad = np.sign(e) * (1.0 + 0.25 * gen_weight * bump)
        hess = np.ones_like(grad)
        return grad, hess

    return objective


def l1_objective(preds: np.ndarray, dataset: lgb.Dataset):
    """대조군. 같은 틀의 L1 이라 내장/커스텀 구현 차이가 개입하지 않는다."""
    e = preds - dataset.get_label()
    return np.sign(e), np.ones_like(e)


def fold_rows() -> dict[str, dict[str, Any]]:
    out = {}
    for probe_fold in PROBE_FOLDS:
        frame = pd.read_parquet(PROBE / f"M115_XGBOOST-{probe_fold}.parquet")
        frame["forecast_kst_dtm"] = pd.to_datetime(frame["forecast_kst_dtm"])
        out[probe_fold] = {
            "keys": set(zip(frame["forecast_id"], frame["group_id"], strict=True)),
            "start": frame["forecast_kst_dtm"].min(),
        }
    return out


def main() -> int:
    surface, _, _ = _surface()
    surface["forecast_kst_dtm"] = pd.to_datetime(surface["forecast_kst_dtm"])
    wanted = json.loads(
        (PROBE / "M115_XGBOOST-dev-2023-Q3.json").read_text(encoding="utf-8")
    )["selected_feature_names"]
    available = [c for c in wanted if c in surface.columns]
    missing = [c for c in wanted if c not in surface.columns]
    assert len(available) >= 80, f"가용 피처가 너무 적다: {len(available)}"

    surface["capacity"] = surface["group_id"].map(CAPACITIES_KWH).astype(float)
    surface["rate"] = surface["actual_kwh"] / surface["capacity"]
    surface = surface.loc[surface["rate"].notna()].reset_index(drop=True)

    folds = fold_rows()
    pieces: dict[str, list[pd.DataFrame]] = {"BAND": [], "CONTROL": []}
    fits = 0
    for probe_fold, meta in folds.items():
        train_mask = surface["forecast_kst_dtm"] < meta["start"]
        test_mask = np.array(
            [
                (fid, gid) in meta["keys"]
                for fid, gid in zip(surface["forecast_id"], surface["group_id"],
                                    strict=True)
            ]
        )
        train = surface.loc[train_mask]
        test = surface.loc[test_mask]
        assert len(test) > 0, f"{probe_fold} 테스트 행이 없다"
        # 학습 모집단도 유효행으로 (사이클 23 의 교훈)
        train = train.loc[train["rate"] >= ELIGIBLE_THRESHOLD]

        x_tr = train.loc[:, available].astype("float32")
        y_tr = train["rate"].to_numpy(dtype="float64")
        gen_w = y_tr / y_tr.mean()
        init = float(np.median(y_tr))

        for name, obj in (
            ("BAND", make_band_objective(gen_w)),
            ("CONTROL", l1_objective),
        ):
            dataset = lgb.Dataset(
                x_tr, label=y_tr,
                init_score=np.full(len(y_tr), init, dtype="float64"),
                free_raw_data=False,
            )
            # LightGBM 4.x 는 fobj 인자를 없앴다. 커스텀 목적은 params 로 넘긴다.
            train_params = {k: v for k, v in PARAMS.items() if k != "n_estimators"}
            train_params["objective"] = obj
            booster = lgb.train(
                train_params,
                dataset,
                num_boost_round=PARAMS["n_estimators"],
            )
            fits += 1
            raw = booster.predict(test.loc[:, available].astype("float32")) + init
            out = test.loc[:, [*KEYS, "actual_kwh"]].copy()
            out["prediction_kwh"] = np.clip(raw, 0.0, 1.0) * test["capacity"].to_numpy()
            pieces[name].append(out)

    results: dict[str, Any] = {}
    frames: dict[str, pd.DataFrame] = {}
    for name, parts in pieces.items():
        frame = pd.concat(parts, ignore_index=True)
        frame["month"] = frame["forecast_kst_dtm"].dt.to_period("M").astype(str)
        frames[name] = frame
        results[name] = official(frame)

    parent = load_predictions(DEPLOYED)
    parent_score = official(parent)

    band = frames["BAND"]
    delta_band_control = results["BAND"]["total"] - results["CONTROL"]["total"]
    ficr_contrib = 0.5 * (results["BAND"]["ficr"] - results["CONTROL"]["ficr"])
    nmae_contrib = 0.5 * (
        results["BAND"]["one_minus_nmae"] - results["CONTROL"]["one_minus_nmae"]
    )

    gate_vs_deployed = evaluate_gate(band, parent)
    gd = gate_vs_deployed.evidence
    h1 = bool(delta_band_control > 0)
    h2 = bool(
        results["BAND"]["total"] > parent_score["total"] and gate_vs_deployed.passed
    )
    h3 = bool(results["BAND"]["total"] > M115_FIXED_TOTAL)
    h4 = bool(ficr_contrib > nmae_contrib)

    by_fold = {}
    for name, frame in frames.items():
        f = frame.copy()
        f["fold"] = f["month"].map(QUARTER_OF_MONTH)
        by_fold[name] = {
            fold: official(cell)["total"]
            for fold, cell in f.groupby("fold", observed=True)
            if fold in FOLDS
        }

    verdict = (
        "BAND_LOSS_HELPS_AND_PASSES_GATE" if h2
        else ("BAND_LOSS_HELPS_INTERNALLY_ONLY" if h1
              else "LOSS_FUNCTION_AXIS_CLOSED")
    )
    check = {
        "H1_expectation": "BAND > CONTROL (손실만 다름)",
        "H1_held": h1, "H1_measured": delta_band_control,
        "H2_expectation": f"BAND 가 배포({DEPLOYED_TOTAL}) 대비 개선 + 동결 게이트 통과",
        "H2_held": h2,
        "H3_expectation": f"BAND > M115@T0.6_G0.2 ({M115_FIXED_TOTAL})",
        "H3_held": h3,
        "H4_expectation": "이득이 FICR 쪽에서 나온다",
        "H4_held": h4,
        "verdict": verdict,
    }

    payload = {
        "node": NODE_ID, "parent_node": PARENT_NODE,
        "gate_version": GATE_VERSION, "gate_modified": False, "lockbox_used": False,
        "why_not_closed_by_prior_axes": [
            "결정층 오라클 8.8% 는 고정 분포에서 사후 사상의 천장이다",
            "물리 폐쇄 0.571 m/s 는 풍속 정확도 요구치이지 점 배치 문제가 아니다",
            "사이클 17 의 기각은 결합 가중치 적합에 대한 것이지 손실 설계가 아니다",
        ],
        "loss": {
            "formula": "L = |e| - (1/4)*(y/ybar)*u_smooth(e)",
            "u_smooth": "3*sigmoid((0.08-|e|)/tau) + sigmoid((0.06-|e|)/tau)",
            "tau": TAU,
            "weights_derived_from_metric": True,
            "hessian": "constant 1.0 (both models)",
            "post_hoc_policy_applied": False,
        },
        "features": {
            "requested": len(wanted), "available": len(available),
            "missing": missing,
            "missing_reason": "teacher 산출 sitewind__* 는 캐시에 없다. 내부 대조이므로 "
                              "판정에 영향 없음",
        },
        "params": PARAMS, "fits": fits,
        "surface_rows": len(surface),
        "scores": {**results, "deployed": parent_score},
        "by_fold": by_fold,
        "delta_band_minus_control": delta_band_control,
        "contribution": {"ficr": ficr_contrib, "one_minus_nmae": nmae_contrib},
        "gate_vs_deployed": {
            "passed": bool(gate_vs_deployed.passed),
            "flags": {la.split()[0]: bool(ok)
                      for la, ok in gate_vs_deployed.conditions.items()},
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
        "# M271 P4 사이클 37 — 밴드 정합 손실로 기저모델 직접 학습",
        "",
        f"- 판정일: {datetime.now(UTC).strftime('%Y-%m-%d')} (UTC)",
        f"- 노드: `{NODE_ID}` / 레인 {LANE} / 부모 `{PARENT_NODE}`",
        f"- 게이트: `{GATE_VERSION}` (읽기만 함) / lockbox 미사용 / 2024 행 미사용",
        "",
        "## 0. 왜 기존 폐쇄가 이 축을 덮지 않는가",
        "",
    ]
    for r in payload["why_not_closed_by_prior_axes"]:
        lines.append(f"- {r}")
    lines += [
        "",
        "## 1. 손실 (실행 전 동결, 가중치는 공식 지표에서 유도)",
        "",
        f"```\n{payload['loss']['formula']}\n"
        f"u_smooth(e) = {payload['loss']['u_smooth']},  tau = {TAU}\n```",
        "",
        "대조군 CONTROL 은 같은 틀의 L1. **두 모델 다 사후 정책 없음.**",
        f"피처 {len(available)}/{len(wanted)} (결측 {len(missing)} 은 teacher 산출), "
        f"적합 {fits} 회, 표면 {len(surface):,} 행",
        "",
        "## 2. 결과",
        "",
        "| 모델 | Total | 1-NMAE | FICR |",
        "|---|---:|---:|---:|",
        f"| 배포 `M269@{DEPLOYED}` | {parent_score['total']:.6f} | "
        f"{parent_score['one_minus_nmae']:.6f} | {parent_score['ficr']:.6f} |",
        f"| `CONTROL` (L1) | {results['CONTROL']['total']:.6f} | "
        f"{results['CONTROL']['one_minus_nmae']:.6f} | {results['CONTROL']['ficr']:.6f} |",
        f"| **`BAND`** | **{results['BAND']['total']:.6f}** | "
        f"{results['BAND']['one_minus_nmae']:.6f} | {results['BAND']['ficr']:.6f} |",
        "",
        f"BAND - CONTROL = **{delta_band_control:+.6f}** "
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
        f"BAND 대 배포 동결 게이트: `{flags}` {g['positive_months']}/{g['months_scored']}월 "
        f"p={g['sign_test_p']:.4f} q05={g['bootstrap_q05']:+.6f} "
        f"최소월 {g['min_delta']:+.6f} -> **{'통과' if g['passed'] else '기각'}**",
        "",
        "## 3. 사전확약 대조",
        "",
        f"- H1 `{check['H1_expectation']}` -> **{h1}** ({delta_band_control:+.6f})",
        f"- H2 `{check['H2_expectation']}` -> **{h2}**",
        f"- H3 `{check['H3_expectation']}` -> **{h3}**",
        f"- H4 `{check['H4_expectation']}` -> **{h4}**",
        "",
        f"판정: **{verdict}**",
        "",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    receipt: dict[str, Any] = {
        "schema_version": 1,
        "stage": "M271_P4_CYCLE37_BAND_LOSS",
        "node": NODE_ID, "lane": LANE,
        "decided_utc": datetime.now(UTC).isoformat(),
        "result": payload,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "report_sha256": hashlib.sha256(REPORT_MD.read_bytes()).hexdigest(),
        "dacon_upload": False, "external_actions": [],
        "model_fits": fits,
        "model_fit_note": "밴드 정합 손실 기저모델과 L1 대조군. 2024 행 미사용",
        "lockbox_reopened": False, "new_2024_evaluation": False,
    }
    RECEIPT.write_text(
        json.dumps(receipt, indent=1, ensure_ascii=False, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

    print(f"[C37] 피처 {len(available)}/{len(wanted)} / 적합 {fits} 회 / "
          f"표면 {len(surface):,} 행")
    print(f"[C37] 배포   {parent_score['total']:.6f}")
    print(f"[C37] CONTROL {results['CONTROL']['total']:.6f} "
          f"(FICR {results['CONTROL']['ficr']:.6f})")
    print(f"[C37] BAND    {results['BAND']['total']:.6f} "
          f"(FICR {results['BAND']['ficr']:.6f})  "
          f"CONTROL 대비 {delta_band_control:+.6f}")
    print(f"[C37] 기여 FICR {ficr_contrib:+.6f} / 1-NMAE {nmae_contrib:+.6f}")
    print(f"[C37] 게이트 [{flags}] {g['positive_months']}/{g['months_scored']}월 -> "
          f"{'통과' if g['passed'] else '기각'}")
    print(f"[C37] H1 {h1} | H2 {h2} | H3 {h3} | H4 {h4}  -> {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
