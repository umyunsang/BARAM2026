"""M271 P4 사이클 38 — 밴드 정합 손실 재시행 (사이클 37 의 구현 결함 교정).

사이클 37 은 `LOSS_FUNCTION_AXIS_CLOSED` 를 냈지만 **그 판정은 무효다.** 두 모델이 배포
기준선(0.628605)보다 한참 아래였다 — CONTROL 0.539702, BAND 0.366308. 87 개 좋은 피처
위의 평범한 L1 회귀기가 0.09 나 뒤처질 리 없다. 구현 결함이다.

결함 둘, 전부 내 것이다.

  D1  **학습 모집단을 유효행으로 걸렀다.** 사이클 23 의 교훈("보정량 추정은 유효행에서")을
      **모델 학습**에 잘못 옮겼다. 채점만 유효행(실측 >= 용량 10%)이고 모델은 저출력
      구간까지 배워야 보정된다. 약 40% 행을 빼서 두 모델 다 위로 편향됐다.
  D2  **헤시안 상수 1.0 이 BAND 에서 파탄난다.** `bump = (3*s8*(1-s8)+s6*(1-s6))/tau` 는
      tau=0.01 에서 최대 약 100 이라 경사가 L1 항의 100 배가 되는데 헤시안이 1 이면
      Newton 스텝 `-G/H` 가 폭발한다. BAND 만 망가뜨리는 비대칭 결함이다.

사이클 37 에 없던 가드도 추가한다: **처리군을 판정하기 전에 대조군의 온전성을 먼저 검사**
한다. 대조군이 기준선 근처에 못 오면 재구성이 깨진 것이므로 손실 축을 판정하지 않는다.

① 방법 리서치 (실행 전)
  - 새 방법 없음. 사이클 37 의 손실 유도(공식 지표에서 상대 가중치를 유도한 것)는 그대로
    쓴다. 그 부분은 틀리지 않았다.
  - 헤시안 처리만 바로잡는다. `hess = 1 + k*bump` 로 **경사와 같은 척도**를 주면
    `-grad/hess = -sign(e)` 로 L1 과 같은 스텝 크기를 유지하면서, 분할 이득
    `sum(g)^2/sum(h)` 에서는 밴드 임계 행이 더 무겁게 반영된다. 재가중 효과는 남기고
    스텝 폭주만 제거하는 표준 처리다.

② 사양 동결

  손실 (사이클 37 과 동일, 유도 그대로)
      L_i = |e_i| - (1/4) * (y_i/ybar) * u_smooth(e_i)
      u_smooth(e) = 3*sigmoid((0.08-|e|)/tau) + sigmoid((0.06-|e|)/tau),  tau = 0.01
      grad = sign(e) * (1 + k*bump),  k = 0.25*(y/ybar)
      hess = 1 + k*bump                      <- **D2 교정**
  대조군 CONTROL: grad = sign(e), hess = 1  (동일 틀의 L1)

  학습 모집단: **전 행** (D1 교정). 채점은 공식 산식이 알아서 유효행만 본다.
  그 외는 사이클 37 과 동일: 같은 피처 87 개, 같은 fold, 같은 시드·파라미터, 사후 정책 없음.

  **타당성 가드 (사전확약보다 먼저 판정)**
    V1  CONTROL 의 pooled Total 이 배포 기준선의 `-0.03` 이내다.
        기각되면 재구성이 여전히 깨진 것이므로 **H1~H4 를 판정하지 않는다.**

  사전확약(실행 전 동결, V1 통과시에만 판정):
    H1  BAND 가 CONTROL 대비 pooled Total 개선. (손실만 다르므로 순수 귀속)
    H2  BAND 가 배포(0.628605) 대비 개선 + **동결 게이트 통과**.
    H3  BAND 가 `M115@T0.6_G0.2` (0.630310) 초과.
    H4  BAND-CONTROL 이득이 **FICR 쪽**에서 나온다.

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
from m271_cycle21_mos import QUARTER_OF_MONTH
from m271_cycle37_band_loss import (
    BAND_HIT,
    BAND_PARTIAL,
    KEYS,
    PARAMS,
    PROBE,
    TAU,
    _sigmoid,
    fold_rows,
)
from m271_evaluate_candidate import official
from run_sequence_classifier import _surface

from baram.constants import CAPACITIES_KWH

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
REPORT_MD = REPORTS / "m271_cycle38_band_loss_fixed.md"
RECEIPT = REPORTS / "m271_cycle38_band_loss_fixed_receipt.json"

NODE_ID = "C1N38_BAND_LOSS_FIXED"
LANE = "L3"
PARENT_NODE = "C1N37_BAND_ALIGNED_LOSS"
SUPERSEDES = "C1N37_BAND_ALIGNED_LOSS"
DEPLOYED = "T0.5_G1.5"
DEPLOYED_TOTAL = 0.628605
M115_FIXED_TOTAL = 0.630310
FOLDS = ("Q2", "Q3", "Q4")
V1_TOLERANCE = 0.03


def band_terms(preds: np.ndarray, y: np.ndarray, gen_weight: np.ndarray):
    e = preds - y
    s8 = _sigmoid((BAND_PARTIAL - np.abs(e)) / TAU)
    s6 = _sigmoid((BAND_HIT - np.abs(e)) / TAU)
    bump = (3.0 * s8 * (1.0 - s8) + s6 * (1.0 - s6)) / TAU
    k = 0.25 * gen_weight
    return e, 1.0 + k * bump


def make_band_objective(gen_weight: np.ndarray):
    def objective(preds: np.ndarray, dataset: lgb.Dataset):
        e, scale = band_terms(preds, dataset.get_label(), gen_weight)
        # D2 교정: 헤시안을 경사와 같은 척도로. 스텝은 L1 과 같고 분할 이득만 재가중된다.
        return np.sign(e) * scale, scale

    return objective


def l1_objective(preds: np.ndarray, dataset: lgb.Dataset):
    e = preds - dataset.get_label()
    return np.sign(e), np.ones_like(e)


def main() -> int:
    surface, _, _ = _surface()
    surface["forecast_kst_dtm"] = pd.to_datetime(surface["forecast_kst_dtm"])
    wanted = json.loads(
        (PROBE / "M115_XGBOOST-dev-2023-Q3.json").read_text(encoding="utf-8")
    )["selected_feature_names"]
    available = [c for c in wanted if c in surface.columns]
    surface["capacity"] = surface["group_id"].map(CAPACITIES_KWH).astype(float)
    surface["rate"] = surface["actual_kwh"] / surface["capacity"]
    surface = surface.loc[surface["rate"].notna()].reset_index(drop=True)

    folds = fold_rows()
    pieces: dict[str, list[pd.DataFrame]] = {"BAND": [], "CONTROL": []}
    fits = 0
    train_sizes = {}
    for probe_fold, meta in folds.items():
        train = surface.loc[surface["forecast_kst_dtm"] < meta["start"]]  # D1 교정: 전 행
        test_mask = np.array(
            [
                (fid, gid) in meta["keys"]
                for fid, gid in zip(surface["forecast_id"], surface["group_id"],
                                    strict=True)
            ]
        )
        test = surface.loc[test_mask]
        assert len(test) > 0, f"{probe_fold} 테스트 행이 없다"
        train_sizes[probe_fold] = {"train": len(train), "test": len(test)}

        x_tr = train.loc[:, available].astype("float32")
        y_tr = train["rate"].to_numpy(dtype="float64")
        gen_w = y_tr / max(y_tr.mean(), 1e-9)
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
            params = {k: v for k, v in PARAMS.items() if k != "n_estimators"}
            params["objective"] = obj
            booster = lgb.train(params, dataset, num_boost_round=PARAMS["n_estimators"])
            fits += 1
            raw = booster.predict(test.loc[:, available].astype("float32")) + init
            out = test.loc[:, [*KEYS, "actual_kwh"]].copy()
            out["prediction_kwh"] = np.clip(raw, 0.0, 1.0) * test["capacity"].to_numpy()
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
            "BAND_LOSS_HELPS_AND_PASSES_GATE" if h2
            else ("BAND_LOSS_HELPS_INTERNALLY_ONLY" if h1
                  else "LOSS_FUNCTION_AXIS_CLOSED")
        )
    else:
        h1 = h2 = h3 = h4 = None
        verdict = "RECONSTRUCTION_INVALID_AXIS_NOT_JUDGED"

    by_fold = {}
    for name, frame in frames.items():
        f = frame.copy()
        f["fold"] = f["month"].map(QUARTER_OF_MONTH)
        by_fold[name] = {
            fold: official(cell)["total"]
            for fold, cell in f.groupby("fold", observed=True)
            if fold in FOLDS
        }

    check = {
        "V1_expectation": f"CONTROL 이 배포의 -{V1_TOLERANCE} 이내",
        "V1_held": v1, "V1_measured_gap": v1_gap,
        "H1_expectation": "BAND > CONTROL", "H1_held": h1, "H1_measured": delta,
        "H2_expectation": "BAND 가 배포 대비 개선 + 게이트 통과", "H2_held": h2,
        "H3_expectation": f"BAND > M115@T0.6_G0.2 ({M115_FIXED_TOTAL})", "H3_held": h3,
        "H4_expectation": "이득이 FICR 쪽", "H4_held": h4,
        "judged": v1,
        "verdict": verdict,
    }

    payload = {
        "node": NODE_ID, "parent_node": PARENT_NODE, "supersedes": SUPERSEDES,
        "gate_version": GATE_VERSION, "gate_modified": False, "lockbox_used": False,
        "defects_fixed": {
            "D1": "학습 모집단을 유효행으로 거른 것 -> 전 행으로 교정",
            "D2": "헤시안 상수 1.0 -> 경사와 같은 척도 (1 + k*bump)",
        },
        "guard_added": "V1 대조군 온전성. 기각시 손실 축을 판정하지 않는다",
        "features_used": len(available),
        "params": PARAMS, "fits": fits, "fold_sizes": train_sizes,
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
        "# M271 P4 사이클 38 — 밴드 정합 손실 재시행 (구현 결함 교정)",
        "",
        f"- 판정일: {datetime.now(UTC).strftime('%Y-%m-%d')} (UTC)",
        f"- 노드: `{NODE_ID}` / 레인 {LANE} / 부모 `{PARENT_NODE}` / 대체 `{SUPERSEDES}`",
        f"- 게이트: `{GATE_VERSION}` (읽기만 함) / lockbox 미사용 / 2024 행 미사용",
        "",
        "## 0. 교정한 결함",
        "",
        f"- **D1** {payload['defects_fixed']['D1']}",
        f"- **D2** {payload['defects_fixed']['D2']}",
        f"- **추가 가드** {payload['guard_added']}",
        "",
        "## 1. 타당성 가드 (V1)",
        "",
        f"CONTROL {results['CONTROL']['total']:.6f} vs 배포 {parent_score['total']:.6f} "
        f"-> 차이 **{v1_gap:+.6f}**, 허용 `-{V1_TOLERANCE}` -> **{v1}**",
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
        f"q05={g['bootstrap_q05']:+.6f} -> **{'통과' if g['passed'] else '기각'}**",
        "",
        "## 3. 사전확약 대조",
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
        "stage": "M271_P4_CYCLE38_BAND_LOSS_FIXED",
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

    print(f"[C38] 적합 {fits} 회 / 피처 {len(available)}")
    print(f"[C38] 배포    {parent_score['total']:.6f}")
    print(f"[C38] CONTROL {results['CONTROL']['total']:.6f} "
          f"(배포 대비 {v1_gap:+.6f}) -> V1 {v1}")
    print(f"[C38] BAND    {results['BAND']['total']:.6f} "
          f"(CONTROL 대비 {delta:+.6f})")
    print(f"[C38] 기여 FICR {ficr_contrib:+.6f} / 1-NMAE {nmae_contrib:+.6f}")
    print(f"[C38] 게이트 [{flags}] {g['positive_months']}/{g['months_scored']}월 -> "
          f"{'통과' if g['passed'] else '기각'}")
    print(f"[C38] 판정: {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
