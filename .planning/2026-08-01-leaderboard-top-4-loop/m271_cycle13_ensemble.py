"""M271 P4 사이클 13 — 분산 감소로서의 앙상블.

하네스 검증이 제약을 하나 드러냈다. 동결 게이트를 통과하려면 **악화 셀이 거의 0** 이어야
한다. 정책 재배분은 47 셀을 악화시켜 떨어지고(사이클 7), 진짜 개선은 0 셀 악화로 통과한다.

"거의 모든 곳에서 좋아지고 아무 데도 나빠지지 않는" 연산은 **분산 감소** 다. 서로 다른
예측기를 평균하면 개별 모델의 잡음이 상쇄된다.

사이클 7 은 62 개 (T,G) 정책을 **개별로만** 판정했다. 평균은 다른 연산이고 테스트된 적이
없다. probe 디렉터리에는 3 개 fold 전부에 예측을 가진 계열이 다른 모델이 13 개 있다 —
분류기(M102), analog(M244), DART, XGBoost, ordinal 등.

분산 감소는 이 프로젝트가 닫은 11 개 축 어디에도 걸리지 않는다.

**same-fold 선택 금지.** 어느 모델을 넣을지 같은 데이터로 고르면 이 프로젝트가 반복적으로
기각해 온 선택 편향이다. 따라서 **사전 선언된 규칙**만 쓴다.

  E1  가용한 13 개 전부 등가중 평균          (선택 없음)
  E2  배포 계열 + 계열이 다른 analog 2 개 등가중  (M102 + M244, 사전 선언)
  E3  배포 + analog + DART + XGBoost 4 개 등가중  (계열 다양성, 사전 선언)

사전확약(실행 전 동결):
  H1  적어도 하나의 앙상블이 배포 대비 **악화 셀 수가 정책 재배분(47)보다 적다**.
  H2  적어도 하나의 앙상블이 **동결 게이트를 통과**한다.
  H2 가 성립하면 12 사이클 만의 첫 검증된 이득이다.

읽기 전용. 모델을 적합하지 않고 2024 행을 읽지 않는다.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from m270_gate import GATE_VERSION, evaluate_gate
from m270_monthly_validation import load_predictions
from m271_evaluate_candidate import ledger_diff, official

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
PROBE = ROOT / "artifacts" / "backtests" / "metric-aligned-probe"
REPORT_MD = REPORTS / "m271_cycle13_ensemble.md"
RECEIPT = REPORTS / "m271_cycle13_ensemble_receipt.json"

NODE_ID = "C1N13_ENSEMBLE_VARIANCE"
LANE = "L7"  # 모델 개선 전략
FOLDS = ("dev-2023-Q2", "dev-2023-Q3", "dev-2023-Q4")
DEPLOYED = "T0.5_G1.5"

ALL_MODELS = (
    "M102_TOP100", "M113_LGBM_DART", "M115_XGBOOST", "M129_GROUP_FINETUNE",
    "M244_RARE_EVENT_CORRECTED_ANALOG_Q234", "M64B_ALLWEATHER_SITEWIND_CLASS",
    "M68_SITEWIND_CLASS_ITER", "M72_BIN020", "M84_LEAVES031", "M93_POWER_QUANTILE",
    "M96_ORDINAL_CUMULATIVE", "M98_ORDINAL_BIN025",
)

# 사전 선언된 조합. 데이터를 보고 고르지 않는다.
ENSEMBLES: dict[str, tuple[str, ...]] = {
    "E1_ALL_EQUAL": ALL_MODELS,
    "E2_CLASSIFIER_PLUS_ANALOG": ("M102_TOP100", "M244_RARE_EVENT_CORRECTED_ANALOG_Q234"),
    "E3_FOUR_FAMILY": (
        "M102_TOP100", "M244_RARE_EVENT_CORRECTED_ANALOG_Q234",
        "M113_LGBM_DART", "M115_XGBOOST",
    ),
}
POLICY_REDISTRIBUTION_WORSENED = 47  # 사이클 7 의 T0.6_G0.2 기준선


def load_model(model: str) -> pd.DataFrame:
    parts = []
    for fold in FOLDS:
        path = PROBE / f"{model}-{fold}.parquet"
        frame = pd.read_parquet(path)
        parts.append(
            frame.loc[
                :, ["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh",
                    "prediction_kwh"]
            ]
        )
    out = pd.concat(parts, ignore_index=True)
    out["forecast_kst_dtm"] = pd.to_datetime(out["forecast_kst_dtm"])
    return out


def average(models: tuple[str, ...]) -> pd.DataFrame:
    frames = [load_model(m) for m in models]
    base = frames[0].loc[
        :, ["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh"]
    ].copy()
    keys = ["forecast_id", "forecast_kst_dtm", "group_id"]
    stacked = base.copy()
    for i, frame in enumerate(frames):
        stacked = stacked.merge(
            frame.loc[:, [*keys, "prediction_kwh"]].rename(
                columns={"prediction_kwh": f"p{i}"}
            ),
            on=keys, how="inner",
        )
    cols = [f"p{i}" for i in range(len(frames))]
    stacked["prediction_kwh"] = stacked.loc[:, cols].mean(axis=1)
    out = stacked.loc[:, [*keys, "actual_kwh", "prediction_kwh"]]
    out["month"] = out["forecast_kst_dtm"].dt.to_period("M").astype(str)
    return out


def main() -> int:
    parent = load_predictions(DEPLOYED)
    parent_score = official(parent)

    rows: list[dict[str, Any]] = []
    for name, members in ENSEMBLES.items():
        candidate = average(members)
        score = official(candidate)
        gate = evaluate_gate(candidate, parent)
        flags = {label.split()[0]: bool(ok) for label, ok in gate.conditions.items()}
        stats = gate.evidence
        diff = ledger_diff(candidate, parent)
        rows.append(
            {
                "name": name,
                "members": list(members),
                "n_members": len(members),
                "rows": len(candidate),
                "total": score["total"],
                "one_minus_nmae": score["one_minus_nmae"],
                "ficr": score["ficr"],
                "delta_total": score["total"] - parent_score["total"],
                "passed": bool(gate.passed),
                "gate": flags,
                "positive_months": int(stats["positive_months"]),
                "months_scored": int(stats["months_scored"]),
                "sign_test_p": float(stats["sign_test_p_greater"]),
                "bootstrap_q05": float(stats["block_bootstrap_q05"]),
                "min_delta": float(stats["min_total_delta"]),
                "cells_improved": diff["cells_improved"],
                "cells_worsened": diff["cells_worsened"],
                "net_loss_removed": diff["net_loss_removed"],
            }
        )

    fewer_worsened = [r for r in rows if r["cells_worsened"] < POLICY_REDISTRIBUTION_WORSENED]
    passed = [r for r in rows if r["passed"]]
    check = {
        "H1_expectation": f"악화 셀 수가 정책 재배분({POLICY_REDISTRIBUTION_WORSENED})보다 적다",
        "H1_held": bool(fewer_worsened),
        "H1_which": [r["name"] for r in fewer_worsened],
        "H2_expectation": "적어도 하나가 동결 게이트를 통과한다",
        "H2_held": bool(passed),
        "H2_which": [r["name"] for r in passed],
        "verdict": "FIRST_VALIDATED_GAIN" if passed else "ENSEMBLE_AXIS_CLOSED",
    }
    payload = {
        "gate_version": GATE_VERSION,
        "gate_modified": False,
        "parent": {"policy": DEPLOYED, **parent_score},
        "ensembles": rows,
        "predeclared_check": check,
    }

    lines = [
        "# M271 P4 사이클 13 — 분산 감소로서의 앙상블",
        "",
        f"- 판정일: {datetime.now(UTC).strftime('%Y-%m-%d')} (UTC)",
        f"- 노드: `{NODE_ID}` / 레인 {LANE}",
        f"- 부모: `{DEPLOYED}` Total {parent_score['total']:.6f}",
        f"- 게이트: `{GATE_VERSION}` (읽기만 함)",
        "",
        "## 1. 왜 이 노드인가",
        "",
        "하네스 검증이 제약을 드러냈다. 게이트를 통과하려면 **악화 셀이 거의 0** 이어야 한다.",
        "정책 재배분은 47 셀 악화로 떨어지고 진짜 개선은 0 셀 악화로 통과한다.",
        "",
        "'거의 모든 곳에서 좋아지고 아무 데도 나빠지지 않는' 연산은 **분산 감소** 다.",
        "사이클 7 은 62 개 정책을 개별로만 판정했고 평균은 테스트된 적이 없다.",
        "",
        "**same-fold 선택을 하지 않는다.** 어느 모델을 넣을지 데이터를 보고 고르면 선택",
        "편향이므로 사전 선언된 조합만 쓴다.",
        "",
        "## 2. 결과",
        "",
        "| 앙상블 | 구성 | Total | 델타 | G1 G2 G3 G4 | 양수월 | **개선/악화 셀** | 통과 |",
        "|---|---:|---:|---:|:---:|---:|---:|:---:|",
    ]
    for r in rows:
        f = "".join("O" if r["gate"].get(k) else "-" for k in ("G1", "G2", "G3", "G4"))
        lines.append(
            f"| `{r['name']}` | {r['n_members']}개 | {r['total']:.6f} | "
            f"{r['delta_total']:+.6f} | `{f}` | "
            f"{r['positive_months']}/{r['months_scored']} | "
            f"**{r['cells_improved']}/{r['cells_worsened']}** | "
            f"{'**통과**' if r['passed'] else '기각'} |"
        )

    lines += [
        "",
        "| 앙상블 | 1-NMAE | FICR | 부호검정 p | q05 | 최악월 | 순 손실제거 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| `{r['name']}` | {r['one_minus_nmae']:.6f} | {r['ficr']:.6f} | "
            f"{r['sign_test_p']:.4f} | {r['bootstrap_q05']:+.6f} | "
            f"{r['min_delta']:+.6f} | {r['net_loss_removed']:+.6f} |"
        )

    lines += [
        "",
        "## 3. 사전확약 대조",
        "",
        f"- H1 `{check['H1_expectation']}` -> **{check['H1_held']}** "
        f"({check['H1_which'] or '없음'})",
        f"- H2 `{check['H2_expectation']}` -> **{check['H2_held']}** "
        f"({check['H2_which'] or '없음'})",
        "",
        f"판정: **{check['verdict']}**",
        "",
        "## 4. 읽는 법",
        "",
        "`FIRST_VALIDATED_GAIN` 이면 12 사이클 만의 첫 검증된 이득이다. 다만 이것도 **로컬**",
        "이며 온라인 이전은 별개 문제다 — 방법군별 오프셋이 3.2 배까지 벌어진 전례가 있다.",
        "",
        "`ENSEMBLE_AXIS_CLOSED` 이면 분산 감소로도 안 된다는 뜻이고, 개별 모델의 오차가",
        "상관되어 평균으로 상쇄되지 않는다는 의미다.",
        "",
        "구성 모델은 전부 사전 선언됐다. 결과를 본 뒤 조합을 바꾸지 않는다.",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    receipt = {
        "schema_version": 1,
        "stage": "M271_P4_CYCLE13_ENSEMBLE",
        "node": NODE_ID,
        "lane": LANE,
        "decided_utc": datetime.now(UTC).isoformat(),
        "result": payload,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "report_sha256": hashlib.sha256(REPORT_MD.read_bytes()).hexdigest(),
        "dacon_upload": False,
        "external_actions": [],
        "model_fits": 0,
        "lockbox_reopened": False,
        "new_2024_evaluation": False,
    }
    RECEIPT.write_text(
        json.dumps(receipt, indent=1, ensure_ascii=False, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

    for r in rows:
        f = "".join("O" if r["gate"].get(k) else "-" for k in ("G1", "G2", "G3", "G4"))
        print(f"[C13] {r['name']:26s} n={r['n_members']:2d} Total={r['total']:.6f} "
              f"델타={r['delta_total']:+.6f} [{f}] "
              f"{r['positive_months']}/{r['months_scored']}월 "
              f"셀 +{r['cells_improved']}/-{r['cells_worsened']} "
              f"{'통과' if r['passed'] else '기각'}")
    print(f"[C13] H1={check['H1_held']} H2={check['H2_held']} -> {check['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
