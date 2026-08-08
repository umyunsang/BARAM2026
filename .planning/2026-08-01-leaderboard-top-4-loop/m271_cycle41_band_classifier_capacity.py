"""M271 P4 사이클 41 — 대조군 증량 1 회. 밴드 인지 학습을 판정 가능한 상태로 만든다.

사이클 40 에서 V1 가드가 발화했다(CONTROL 0.584468, 배포 대비 -0.044136). 형식상 손실 축은
미판정이지만 내부 대조가 강한 것을 말했다.

    BAND - CONTROL = **+0.009772**,  기여 FICR +0.008224 / 1-NMAE +0.001548

이득의 84% 가 FICR 에서 나온다 — 설계 시 예측한 기전 그대로다. 이번 세션에서 측정된
**가장 큰 단일 처리효과**이기도 하다(앞선 최고 +0.008 은 fold내 정책 선택 프리미엄으로
부풀려진 것이었고, 정직한 최선은 +0.001705 였다).

대조군이 뒤처진 원인은 특정 가능하다: 87 피처(최상위 13 개 teacher 피처 결측),
200 rounds x 15 leaves 무튜닝, 그리고 내 Bayes 결정규칙 대 배포의 튜닝된 T/G 정책.
앞의 둘은 증량으로 좁힐 수 있다.

**사전확약은 완화하지 않는다.** V1 문턱 `-0.03` 을 그대로 두고 **대조군만 강화**한다.
다만 "가드가 통과할 때까지 튜닝" 은 그 자체가 낚시이므로 **증량은 이 1 회로 제한**한다고
실행 전에 못박는다. 이번에도 V1 이 기각되면 내부 처리효과를 **시사적이나 미판정**으로
기록하고 이 축의 추가 증량을 중단한다.

① 방법 리서치 (실행 전)
  - 새 방법 없음. 사이클 40 의 설계를 그대로 두고 **모형 용량만** 올린다. 한 변수만
    움직이는 원칙(사이클 28->30->31, 37->38)을 유지한다.
  - 증량 폭도 실행 전에 고정한다: rounds 200->500, leaves 15->31, lr 0.1->0.06.
    결과를 보고 조정하지 않는다.

② 사양 동결

  사이클 40 과 **동일**: 46 class x 폭 0.02, CONTROL=one-hot / BAND=정산모양 soft target,
  softmax CE (`grad = p - q`), 두 팔 같은 Bayes 결정규칙, 87 피처, `scada_ws` 제외,
  fold 별 chronology-safe.
  **다른 것은 용량뿐**: `num_leaves 31, learning_rate 0.06, rounds 500`.

  **타당성 가드**
    V1  CONTROL 이 배포(0.628605)의 `-0.03` 이내. (사이클 40 과 동일 문턱)
    V2  증량이 CONTROL 을 실제로 개선한다 (사이클 40 CONTROL 0.584468 초과).
        기각되면 증량이 원인 진단을 틀렸다는 뜻이므로 함께 기록한다.

  사전확약(V1 통과시에만 판정):
    H1  BAND > CONTROL.
    H2  BAND 가 배포 대비 개선 + **동결 게이트 통과**.
    H3  BAND > `M115@T0.6_G0.2` (0.630310).
    H4  이득이 **FICR 쪽**에서 나온다.
    H5  (기전 재현) BAND-CONTROL 처리효과의 부호가 사이클 40 과 **같다**.
        용량이 바뀌어도 기전이 유지되는지 본다.

**게이트를 수정하지 않는다.** 2024 행·lockbox·`scada_ws` 미사용. **증량 1 회 제한.**
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
    LEAKY_COLUMNS,
    M115_FIXED_TOTAL,
    N_CLASS,
    V1_TOLERANCE,
    bayes_decision,
    by_fold_total,
    make_objective,
    one_hot_targets,
    soft_targets,
)
from m271_cycle40_band_classifier import (
    PARAMS as BASE_PARAMS,
)
from m271_evaluate_candidate import official
from run_sequence_classifier import _surface

from baram.constants import CAPACITIES_KWH

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
REPORT_MD = REPORTS / "m271_cycle41_band_classifier_capacity.md"
RECEIPT = REPORTS / "m271_cycle41_band_classifier_capacity_receipt.json"

NODE_ID = "C1N41_BAND_CLASSIFIER_CAPACITY"
LANE = "L3"
PARENT_NODE = "C1N40_BAND_CLASSIFIER"
DEPLOYED_TOTAL = 0.628605
C40_CONTROL = 0.584468
C40_DELTA = 0.009772

PARAMS = {**BASE_PARAMS, "num_leaves": 31, "learning_rate": 0.06}
ROUNDS = 500
ESCALATION_BUDGET = "1 회 제한 (실행 전 선언)"


def main() -> int:
    surface, _, _ = _surface()
    surface["forecast_kst_dtm"] = pd.to_datetime(surface["forecast_kst_dtm"])
    wanted = json.loads(
        (PROBE / "M115_XGBOOST-dev-2023-Q3.json").read_text(encoding="utf-8")
    )["selected_feature_names"]
    features = [c for c in wanted if c in surface.columns and c not in LEAKY_COLUMNS]
    surface["capacity"] = surface["group_id"].map(CAPACITIES_KWH).astype(float)
    surface["rate"] = surface["actual_kwh"] / surface["capacity"]
    surface = surface.loc[surface["rate"].notna()].reset_index(drop=True)

    pieces: dict[str, list[pd.DataFrame]] = {"BAND": [], "CONTROL": []}
    fits = 0
    for _fold, meta in fold_rows().items():
        train = surface.loc[surface["forecast_kst_dtm"] < meta["start"]]
        test = surface.loc[
            np.array(
                [
                    (fid, gid) in meta["keys"]
                    for fid, gid in zip(surface["forecast_id"], surface["group_id"],
                                        strict=True)
                ]
            )
        ]
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
            raw = np.asarray(
                booster.predict(test.loc[:, features].astype("float32"))
            ).reshape(len(test), N_CLASS)
            exp = np.exp(raw - raw.max(axis=1, keepdims=True))
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
    v2 = bool(results["CONTROL"]["total"] > C40_CONTROL)

    delta = results["BAND"]["total"] - results["CONTROL"]["total"]
    ficr_contrib = 0.5 * (results["BAND"]["ficr"] - results["CONTROL"]["ficr"])
    nmae_contrib = 0.5 * (
        results["BAND"]["one_minus_nmae"] - results["CONTROL"]["one_minus_nmae"]
    )
    gate = evaluate_gate(frames["BAND"], parent)
    gd = gate.evidence
    h5 = bool(np.sign(delta) == np.sign(C40_DELTA))

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
        verdict = "SUGGESTIVE_BUT_UNJUDGED_ESCALATION_BUDGET_SPENT"

    by_fold = {name: by_fold_total(frame) for name, frame in frames.items()}
    check = {
        "V1_expectation": f"CONTROL 이 배포의 -{V1_TOLERANCE} 이내",
        "V1_held": v1, "V1_measured_gap": v1_gap,
        "V2_expectation": f"증량이 CONTROL 을 개선 (> {C40_CONTROL})",
        "V2_held": v2, "V2_measured": results["CONTROL"]["total"] - C40_CONTROL,
        "H1_expectation": "BAND > CONTROL", "H1_held": h1, "H1_measured": delta,
        "H2_expectation": "BAND 가 배포 대비 개선 + 게이트 통과", "H2_held": h2,
        "H3_expectation": f"BAND > {M115_FIXED_TOTAL}", "H3_held": h3,
        "H4_expectation": "이득이 FICR 쪽", "H4_held": h4,
        "H5_expectation": f"처리효과 부호가 사이클 40({C40_DELTA:+.6f})과 동일",
        "H5_held": h5,
        "escalation_budget": ESCALATION_BUDGET,
        "judged": v1, "verdict": verdict,
    }

    payload = {
        "node": NODE_ID, "parent_node": PARENT_NODE,
        "gate_version": GATE_VERSION, "gate_modified": False, "lockbox_used": False,
        "changed_vs_parent": "모형 용량만 (leaves 15->31, lr 0.1->0.06, rounds 200->500)",
        "cycle40": {"control": C40_CONTROL, "delta": C40_DELTA},
        "features_used": len(features), "params": PARAMS, "rounds": ROUNDS, "fits": fits,
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
        "# M271 P4 사이클 41 — 대조군 증량 1 회",
        "",
        f"- 판정일: {datetime.now(UTC).strftime('%Y-%m-%d')} (UTC)",
        f"- 노드: `{NODE_ID}` / 레인 {LANE} / 부모 `{PARENT_NODE}`",
        f"- 부모 대비 바뀐 것: **{payload['changed_vs_parent']}**",
        f"- 증량 예산: **{ESCALATION_BUDGET}**",
        f"- 게이트: `{GATE_VERSION}` (읽기만 함) / lockbox·`scada_ws` 미사용",
        "",
        "## 1. 가드",
        "",
        f"- V1 CONTROL {results['CONTROL']['total']:.6f} vs 배포 "
        f"{parent_score['total']:.6f} -> **{v1_gap:+.6f}** (허용 -{V1_TOLERANCE}) "
        f"-> **{v1}**",
        f"- V2 증량 효과 {results['CONTROL']['total'] - C40_CONTROL:+.6f} "
        f"(사이클 40 CONTROL {C40_CONTROL:.6f}) -> **{v2}**",
        "",
        "## 2. 결과",
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
        f"(FICR {ficr_contrib:+.6f} / 1-NMAE {nmae_contrib:+.6f}) — "
        f"사이클 40 은 {C40_DELTA:+.6f}",
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
        "## 3. 사전확약 대조",
        "",
        f"- V1 -> **{v1}** ({v1_gap:+.6f})",
        f"- V2 `{check['V2_expectation']}` -> **{v2}**",
        f"- H1 -> **{h1 if h1 is not None else '판정안함'}** ({delta:+.6f})",
        f"- H2 -> **{h2 if h2 is not None else '판정안함'}**",
        f"- H3 -> **{h3 if h3 is not None else '판정안함'}**",
        f"- H4 -> **{h4 if h4 is not None else '판정안함'}**",
        f"- H5 `{check['H5_expectation']}` -> **{h5}**",
        "",
        f"판정: **{verdict}**",
        "",
    ]
    if not v1:
        lines += [
            "## 4. 여기서 멈추는 이유",
            "",
            "증량 예산 1 회를 실행 전에 선언했고 소진했다. 가드가 통과할 때까지 용량을",
            "올리는 것은 **가드를 무력화하는 낚시**다. 내부 처리효과는 **시사적이나",
            "미판정**으로 남기고 이 축의 추가 증량을 중단한다.",
            "",
            "남은 정당한 경로는 용량이 아니라 **대조군의 결손 원인 제거**다 — 결측된 13 개",
            "teacher 피처(`sitewind__*`)를 복원하는 것. 그건 teacher 러너를 돌리는 별개 작업이다.",
        ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    receipt: dict[str, Any] = {
        "schema_version": 1,
        "stage": "M271_P4_CYCLE41_BAND_CLASSIFIER_CAPACITY",
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

    print(f"[C41] 적합 {fits} 회 / leaves {PARAMS['num_leaves']} / rounds {ROUNDS}")
    print(f"[C41] CONTROL {results['CONTROL']['total']:.6f} "
          f"(배포 대비 {v1_gap:+.6f}, 사이클40 대비 "
          f"{results['CONTROL']['total'] - C40_CONTROL:+.6f}) -> V1 {v1} V2 {v2}")
    print(f"[C41] BAND    {results['BAND']['total']:.6f} (CONTROL 대비 {delta:+.6f})")
    print(f"[C41] 기여 FICR {ficr_contrib:+.6f} / 1-NMAE {nmae_contrib:+.6f} | H5 {h5}")
    print(f"[C41] 게이트 [{flags}] {g['positive_months']}/{g['months_scored']}월 -> "
          f"{'통과' if g['passed'] else '기각'}")
    print(f"[C41] 판정: {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
