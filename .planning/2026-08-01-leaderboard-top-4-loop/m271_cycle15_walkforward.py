"""M271 P4 사이클 15 — 시간순 안전 walk-forward 앙상블 가중.

사이클 13 에서 E3(4 개 모델)가 E1(12 개 전부)보다 나았다. 구성에 구조가 있다는 뜻인데,
구성을 데이터 보고 고르면 same-fold 선택이다.

정당한 방법은 **walk-forward 가중** 이다.

    Q2: 선행 fold 없음  -> 등가중 (사전 선언 기본값)
    Q3: Q2 로 가중 산출
    Q4: Q2+Q3 로 가중 산출

각 fold 의 가중이 **그 fold 를 보지 않고** 정해지므로 same-fold 선택이 아니다. 나쁜 모델이
자동으로 내려가므로 12 개를 다 넣어도 E1 처럼 끌려내려가지 않을 수 있다.

가중 규칙도 사전 선언한다: **선행 fold 의 MSE 역수에 비례**, 정규화. 온도나 상위 K 컷 같은
자유도를 두지 않는다.

사이클 14 가 확인한 수축(a=0.5, 사전 선언)을 그대로 얹은 변형도 함께 본다.

사전확약(실행 전 동결):
  H1  walk-forward 가중 앙상블이 **동결 게이트를 통과**한다.
  H2  walk-forward 가중이 등가중(E1)보다 pooled Total 이 **높다**.
      (가중이 실제로 나쁜 모델을 내리는지)
  두 변형(수축 없음 / a=0.5 수축) 모두 사전 선언이므로 둘 다 보고하되, 어느 쪽이 좋은지
  보고 고르는 것은 same-fold 선택이므로 하지 않는다.

읽기 전용. 모델을 적합하지 않고 2024 행을 읽지 않는다.
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
from m270_monthly_validation import load_predictions
from m271_cycle13_ensemble import ALL_MODELS, PROBE
from m271_cycle14_shrinkblend import blend
from m271_evaluate_candidate import ledger_diff, official

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
REPORT_MD = REPORTS / "m271_cycle15_walkforward.md"
RECEIPT = REPORTS / "m271_cycle15_walkforward_receipt.json"

NODE_ID = "C1N15_WALKFORWARD_ENSEMBLE"
LANE = "L4"  # 검증 전략 — 시간순 안전 가중
DEPLOYED = "T0.5_G1.5"
# fold 는 시간순이다. 각 fold 의 가중은 그 앞의 fold 들로만 만든다.
FOLD_ORDER = ("dev-2023-Q2", "dev-2023-Q3", "dev-2023-Q4")
PREDECLARED_ALPHA = 0.5  # 사이클 14 의 사전 선언점을 그대로 쓴다
KEYS = ["forecast_id", "forecast_kst_dtm", "group_id"]


def load_fold(model: str, fold: str) -> pd.DataFrame:
    frame = pd.read_parquet(PROBE / f"{model}-{fold}.parquet")
    out = frame.loc[:, [*KEYS, "actual_kwh", "prediction_kwh"]].copy()
    out["forecast_kst_dtm"] = pd.to_datetime(out["forecast_kst_dtm"])
    return out


def fold_mse(model: str, folds: tuple[str, ...]) -> float:
    errors = []
    for fold in folds:
        frame = load_fold(model, fold)
        errors.append(
            (frame["prediction_kwh"].to_numpy(float) - frame["actual_kwh"].to_numpy(float))
            ** 2
        )
    return float(np.concatenate(errors).mean())


def weights_for(fold: str) -> dict[str, float]:
    """그 fold **앞** 의 fold 들로만 가중을 만든다. 선행이 없으면 등가중."""
    index = FOLD_ORDER.index(fold)
    preceding = FOLD_ORDER[:index]
    if not preceding:
        return dict.fromkeys(ALL_MODELS, 1.0 / len(ALL_MODELS))
    inverse = {m: 1.0 / fold_mse(m, preceding) for m in ALL_MODELS}
    total = sum(inverse.values())
    return {m: v / total for m, v in inverse.items()}


def walkforward_ensemble() -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    parts, used = [], {}
    for fold in FOLD_ORDER:
        w = weights_for(fold)
        used[fold] = w
        base = load_fold(ALL_MODELS[0], fold).loc[:, [*KEYS, "actual_kwh"]].copy()
        stacked = base.copy()
        for i, model in enumerate(ALL_MODELS):
            frame = load_fold(model, fold)
            stacked = stacked.merge(
                frame.loc[:, [*KEYS, "prediction_kwh"]].rename(
                    columns={"prediction_kwh": f"p{i}"}
                ),
                on=KEYS, how="inner",
            )
        weighted = np.zeros(len(stacked), dtype=float)
        for i, model in enumerate(ALL_MODELS):
            weighted += w[model] * stacked[f"p{i}"].to_numpy(float)
        out = stacked.loc[:, [*KEYS, "actual_kwh"]].copy()
        out["prediction_kwh"] = weighted
        parts.append(out)
    frame = pd.concat(parts, ignore_index=True)
    frame["month"] = frame["forecast_kst_dtm"].dt.to_period("M").astype(str)
    return frame, used


def judge(candidate: pd.DataFrame, parent: pd.DataFrame, name: str) -> dict[str, Any]:
    score = official(candidate)
    parent_score = official(parent)
    gate = evaluate_gate(candidate, parent)
    stats = gate.evidence
    diff = ledger_diff(candidate, parent)
    return {
        "name": name,
        "total": score["total"],
        "one_minus_nmae": score["one_minus_nmae"],
        "ficr": score["ficr"],
        "delta_total": score["total"] - parent_score["total"],
        "passed": bool(gate.passed),
        "gate": {label.split()[0]: bool(ok) for label, ok in gate.conditions.items()},
        "positive_months": int(stats["positive_months"]),
        "months_scored": int(stats["months_scored"]),
        "sign_test_p": float(stats["sign_test_p_greater"]),
        "bootstrap_q05": float(stats["block_bootstrap_q05"]),
        "min_delta": float(stats["min_total_delta"]),
        "cells_improved": diff["cells_improved"],
        "cells_worsened": diff["cells_worsened"],
    }


def main() -> int:
    parent = load_predictions(DEPLOYED)
    ensemble, used_weights = walkforward_ensemble()

    variants = [
        judge(ensemble, parent, "WF_RAW"),
        judge(blend(parent, ensemble, PREDECLARED_ALPHA), parent,
              f"WF_SHRINK_a{PREDECLARED_ALPHA}"),
    ]

    # H2 대조: 사이클 13 의 등가중 E1
    from m271_cycle13_ensemble import ENSEMBLES, average

    e1 = judge(average(ENSEMBLES["E1_ALL_EQUAL"]), parent, "E1_ALL_EQUAL_reference")

    passed = [v for v in variants if v["passed"]]
    beats_equal = any(v["total"] > e1["total"] for v in variants)
    check = {
        "H1_expectation": "walk-forward 가중 앙상블이 동결 게이트를 통과한다",
        "H1_held": bool(passed),
        "H1_which": [v["name"] for v in passed],
        "H2_expectation": "walk-forward 가중이 등가중(E1)보다 pooled Total 이 높다",
        "H2_held": bool(beats_equal),
        "e1_reference_total": e1["total"],
        "verdict": "WALKFORWARD_VALIDATED" if passed else "WALKFORWARD_REJECTED",
    }
    payload = {
        "gate_version": GATE_VERSION,
        "gate_modified": False,
        "fold_order": list(FOLD_ORDER),
        "weights_per_fold": used_weights,
        "variants": variants,
        "reference_e1": e1,
        "predeclared_check": check,
    }

    lines = [
        "# M271 P4 사이클 15 — 시간순 안전 walk-forward 앙상블 가중",
        "",
        f"- 판정일: {datetime.now(UTC).strftime('%Y-%m-%d')} (UTC)",
        f"- 노드: `{NODE_ID}` / 레인 {LANE}",
        f"- 게이트: `{GATE_VERSION}` (읽기만 함)",
        "",
        "## 1. 가중 규칙",
        "",
        "각 fold 의 가중은 **그 fold 앞** 의 fold 들로만 만든다. 선행이 없으면 등가중.",
        "가중은 선행 fold MSE 의 역수에 비례하며 정규화한다. 온도나 상위 K 컷 같은 자유도를",
        "두지 않는다 — 전부 사전 선언이다.",
        "",
        "| fold | 가중 출처 | 상위 3 모델 (가중) |",
        "|---|---|---|",
    ]
    for fold in FOLD_ORDER:
        w = used_weights[fold]
        top = sorted(w.items(), key=lambda kv: -kv[1])[:3]
        source = "등가중 (선행 없음)" if fold == FOLD_ORDER[0] else ", ".join(
            FOLD_ORDER[: FOLD_ORDER.index(fold)]
        )
        detail = ", ".join(f"`{m}` {v:.3f}" for m, v in top)
        lines.append(f"| `{fold}` | {source} | {detail} |")

    lines += [
        "",
        "## 2. 판정",
        "",
        "| 후보 | Total | 델타 | 1-NMAE | FICR | G1 G2 G3 G4 | 양수월 | 개선/악화 | 통과 |",
        "|---|---:|---:|---:|---:|:---:|---:|---:|:---:|",
    ]
    for v in [*variants, e1]:
        f = "".join("O" if v["gate"].get(k) else "-" for k in ("G1", "G2", "G3", "G4"))
        lines.append(
            f"| `{v['name']}` | {v['total']:.6f} | {v['delta_total']:+.6f} | "
            f"{v['one_minus_nmae']:.6f} | {v['ficr']:.6f} | `{f}` | "
            f"{v['positive_months']}/{v['months_scored']} | "
            f"{v['cells_improved']}/{v['cells_worsened']} | "
            f"{'**통과**' if v['passed'] else '기각'} |"
        )

    lines += [
        "",
        "## 3. 사전확약 대조",
        "",
        f"- H1 `{check['H1_expectation']}` -> **{check['H1_held']}** "
        f"({check['H1_which'] or '없음'})",
        f"- H2 `{check['H2_expectation']}` -> **{check['H2_held']}** "
        f"(E1 기준 {e1['total']:.6f})",
        "",
        f"판정: **{check['verdict']}**",
        "",
        "## 4. 한계",
        "",
        "Q2 는 선행 fold 가 없어 등가중을 쓴다. 즉 전체 행의 1/3 은 가중의 이점을 못 받는다.",
        "실제 운영에서는 2022 이력이 있으므로 이 제약이 완화되지만, 여기서는 probe 산출물이",
        "2023 Q2~Q4 만 있어 그렇게 하지 못한다.",
        "",
        "그리고 이것은 **로컬** 판정이다. 온라인 이전은 별개이며 방법군별 오프셋이 3.2 배까지",
        "벌어진 전례가 있다.",
        "",
        "두 변형 모두 사전 선언이다. 어느 쪽이 좋은지 보고 고르는 것은 same-fold 선택이므로",
        "하지 않는다.",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    receipt = {
        "schema_version": 1,
        "stage": "M271_P4_CYCLE15_WALKFORWARD",
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

    for v in [*variants, e1]:
        f = "".join("O" if v["gate"].get(k) else "-" for k in ("G1", "G2", "G3", "G4"))
        print(f"[C15] {v['name']:24s} Total={v['total']:.6f} 델타={v['delta_total']:+.6f} "
              f"[{f}] {v['positive_months']}/{v['months_scored']}월 "
              f"셀 +{v['cells_improved']}/-{v['cells_worsened']} "
              f"{'통과' if v['passed'] else '기각'}")
    print(f"[C15] H1={check['H1_held']} H2={check['H2_held']} -> {check['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
