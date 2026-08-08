"""M271 P4 사이클 8 — 정책선택 조건화의 오라클 사다리.

사이클 4 는 대역조건부(4 구간)에서 오라클 이득 `+0.002761` 을 얻었다. 사이클 7 은 전역
정책 62 개 전부가 동결 게이트를 못 넘는다고 확인했다. 남은 질문은 하나다.

    **더 풍부한 조건화가 얼마나 더 잡는가?**

이 노드는 조건화 세밀도를 올려가며 오라클 상한의 사다리를 만든다.

    0. 전역 (배포)                      = 조건화 없음
    1. 전역 최적                         = 후보 중 하나를 전 행에
    2. 예측 대역 4 구간 (사이클 4)
    3. 예측 대역 x 그룹 (12 구간)
    4. 예측 대역 x 그룹 x 월 (108 구간)
    5. **행 단위 오라클**                = 어떤 규칙도 넘을 수 없는 절대 상한

전부 same-fold 선택이므로 **달성 가능량이 아니라 상한**이다. 사다리의 모양이 결정을 준다.

  * 5 번이 작으면 정책선택 축 자체가 닫힌다. 조건화를 아무리 정교하게 해도 얻을 게 없다.
  * 5 번은 큰데 2~4 번이 작으면 얻을 것은 있으나 **거친 조건화로는 못 잡는다**는 뜻이고,
    그때만 모델 적합이 정당화된다.
  * 2~4 번이 5 번에 빠르게 접근하면 값싼 조건화로 대부분 잡힌다는 뜻이다.

사전확약(실행 전 동결):
  H1  행 단위 오라클 이득이 격차 `0.031395` 의 20% 를 **넘는다**.
  H2  대역x그룹x월 조건화가 행 단위 오라클의 절반을 **넘지 못한다**
      (거친 조건화로는 안 잡힌다 = 모델 적합이 정당화된다)
  H1 이 기각되면 정책선택 축 전체가 닫힌다.

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

from baram.constants import CAPACITIES_KWH, METRIC_COLUMNS
from baram.evaluation.official import evaluate_official, settlement_unit

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
PROBE = ROOT / "artifacts" / "backtests" / "metric-aligned-probe"
REPORT_MD = REPORTS / "m271_cycle8_oracleladder.md"
RECEIPT = REPORTS / "m271_cycle8_oracleladder_receipt.json"

NODE_ID = "C1N8_ORACLE_LADDER"
LANE = "L6"
FOLDS = ("dev-2023-Q2", "dev-2023-Q3", "dev-2023-Q4")
DEPLOYED = "T0.5_G1.5"
TARGET_GAP = 0.031395
PRED_BAND_EDGES = (0.0, 0.25, 0.45, 0.70, 1.20)


def load() -> tuple[pd.DataFrame, list[str]]:
    parts = [pd.read_parquet(PROBE / f"M269_PROBE_TOP100-{f}-policies.parquet") for f in FOLDS]
    frame = pd.concat(parts, ignore_index=True)
    frame["forecast_kst_dtm"] = pd.to_datetime(frame["forecast_kst_dtm"])
    frame["capacity"] = frame["group_id"].map(CAPACITIES_KWH).astype(float)
    frame["month"] = frame["forecast_kst_dtm"].dt.to_period("M").astype(str)
    ratio = frame[DEPLOYED].to_numpy(float) / frame["capacity"].to_numpy(float)
    frame["pred_band"] = pd.cut(ratio, bins=list(PRED_BAND_EDGES), right=True).astype(str)
    return frame, sorted(c for c in frame.columns if c.startswith("T"))


def score(frame: pd.DataFrame, prediction: np.ndarray) -> float:
    metric = frame.loc[:, ["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh"]].copy()
    metric["prediction_kwh"] = prediction
    return float(evaluate_official(metric.loc[:, sorted(METRIC_COLUMNS)], CAPACITIES_KWH).total)


def conditional_oracle(frame: pd.DataFrame, policies: list[str], keys: list[str],
                       base: np.ndarray) -> tuple[np.ndarray, int]:
    """구간별로 최선 정책을 골라 결합한다. same-fold 선택이다."""
    combined = base.copy()
    matrix = frame.loc[:, policies].to_numpy(float)
    groups = 0
    for _, part in frame.groupby(keys, observed=True):
        idx = part.index.to_numpy()
        best_value, best_col = -np.inf, None
        for j, _p in enumerate(policies):
            trial = combined.copy()
            trial[idx] = matrix[idx, j]
            value = score(frame, trial)
            if value > best_value:
                best_value, best_col = value, j
        combined[idx] = matrix[idx, best_col]
        groups += 1
    return combined, groups


def row_oracle(frame: pd.DataFrame, policies: list[str]) -> np.ndarray:
    """행마다 정산단위가 가장 큰 정책을 고른다. 동률이면 오차가 작은 쪽.

    이것이 어떤 정책선택 규칙도 넘을 수 없는 절대 상한이다.
    """
    matrix = frame.loc[:, policies].to_numpy(float)
    actual = frame["actual_kwh"].to_numpy(float)[:, None]
    capacity = frame["capacity"].to_numpy(float)[:, None]
    err = np.abs(matrix - actual) / capacity
    units = settlement_unit(err.ravel()).reshape(err.shape)
    # 정산단위 최대, 동률이면 절대오차 최소.
    rank = units * 1e6 - err
    return matrix[np.arange(len(frame)), rank.argmax(axis=1)]


def main() -> int:
    frame, policies = load()
    base_deployed = frame[DEPLOYED].to_numpy(float)
    deployed_total = score(frame, base_deployed)

    global_scores = {p: score(frame, frame[p].to_numpy(float)) for p in policies}
    global_best = max(global_scores, key=global_scores.get)
    base_global = frame[global_best].to_numpy(float)

    rungs: list[dict[str, Any]] = [
        {"level": 0, "name": "전역 (배포)", "cells": 1, "total": deployed_total},
        {"level": 1, "name": "전역 최적", "cells": 1, "total": global_scores[global_best]},
    ]

    for level, keys, label in (
        (2, ["pred_band"], "예측 대역"),
        (3, ["pred_band", "group_id"], "예측 대역 x 그룹"),
        (4, ["pred_band", "group_id", "month"], "예측 대역 x 그룹 x 월"),
    ):
        combined, cells = conditional_oracle(frame, policies, keys, base_global)
        rungs.append(
            {"level": level, "name": label, "cells": cells, "total": score(frame, combined)}
        )

    oracle = row_oracle(frame, policies)
    rungs.append(
        {"level": 5, "name": "행 단위 오라클", "cells": len(frame), "total": score(frame, oracle)}
    )

    for rung in rungs:
        rung["gain_vs_deployed"] = rung["total"] - deployed_total
        rung["share_of_gap"] = rung["gain_vs_deployed"] / TARGET_GAP

    row_gain = rungs[-1]["gain_vs_deployed"]
    coarse_gain = rungs[-2]["gain_vs_deployed"]
    check = {
        "H1_expectation": "행 단위 오라클 이득이 격차의 20% 를 넘는다",
        "H1_value": row_gain / TARGET_GAP,
        "H1_held": bool(row_gain / TARGET_GAP > 0.20),
        "H2_expectation": "거친 조건화(대역x그룹x월)가 행 오라클의 절반을 넘지 못한다",
        "H2_value": coarse_gain / row_gain if row_gain > 0 else float("nan"),
        "H2_held": bool(row_gain > 0 and coarse_gain / row_gain < 0.5),
        "is_oracle": True,
        "verdict": (
            "MODELLING_JUSTIFIED"
            if (row_gain / TARGET_GAP > 0.20 and coarse_gain / row_gain < 0.5)
            else ("COARSE_CONDITIONING_SUFFICES"
                  if row_gain / TARGET_GAP > 0.20 else "POLICY_SELECTION_AXIS_CLOSED")
        ),
    }
    payload = {
        "deployed": DEPLOYED,
        "global_best": global_best,
        "target_gap": TARGET_GAP,
        "rungs": rungs,
        "predeclared_check": check,
    }

    lines = [
        "# M271 P4 사이클 8 — 정책선택 조건화의 오라클 사다리",
        "",
        f"- 판정일: {datetime.now(UTC).strftime('%Y-%m-%d')} (UTC)",
        f"- 노드: `{NODE_ID}` / 레인 {LANE}",
        f"- 배포 `{DEPLOYED}` = {deployed_total:.6f} / 격차 {TARGET_GAP:.6f}",
        f"- 후보 정책 {len(policies)} 개, 행 {len(frame):,}",
        "",
        "## 1. 사다리",
        "",
        "전부 same-fold 선택이므로 **상한**이다. 사다리의 모양이 결정을 준다.",
        "",
        "| 단계 | 조건화 | 구간 수 | Total | 배포 대비 | **격차 대비** |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for r in rungs:
        lines.append(
            f"| {r['level']} | {r['name']} | {r['cells']:,} | {r['total']:.6f} | "
            f"{r['gain_vs_deployed']:+.6f} | **{r['share_of_gap']:.1%}** |"
        )

    lines += [
        "",
        "## 2. 사전확약 대조",
        "",
        f"- H1 `{check['H1_expectation']}` -> 행 오라클 {check['H1_value']:.1%} -> "
        f"**{check['H1_held']}**",
        f"- H2 `{check['H2_expectation']}` -> 거친/행 비율 {check['H2_value']:.1%} -> "
        f"**{check['H2_held']}**",
        "",
        f"판정: **{check['verdict']}**",
        "",
        "## 3. 판정이 뜻하는 것",
        "",
        "- `POLICY_SELECTION_AXIS_CLOSED` — 행 단위 오라클조차 작다. 조건화를 아무리 정교하게",
        "  해도 얻을 게 없으므로 이 축 전체를 닫는다.",
        "- `COARSE_CONDITIONING_SUFFICES` — 얻을 것이 있고 거친 조건화로 대부분 잡힌다.",
        "  그런데 사이클 7 이 그 거친 후보들을 동결 게이트로 이미 전부 기각했다.",
        "- `MODELLING_JUSTIFIED` — 얻을 것이 있으나 거친 조건화로는 못 잡는다. 이때만 피처",
        "  조건부 정책을 학습하는 모델 적합이 정당화된다.",
        "",
        "## 4. 이 값의 지위",
        "",
        "행 단위 오라클은 각 행에서 **실제값을 보고** 최선 정책을 고른 것이다. 예보 시점에",
        "불가능하며 어떤 규칙으로도 도달할 수 없다. 상한으로만 읽는다.",
        "",
        "사이클 7 이 확인한 대로 전역 정책 62 개는 전부 동결 게이트에서 기각됐다. 사다리가",
        "크더라도 그것을 **검증된 이득으로 바꾸는 것은 별개 문제**이며 이 노드는 하지 않는다.",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    receipt = {
        "schema_version": 1,
        "stage": "M271_P4_CYCLE8_ORACLE_LADDER",
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

    for r in rungs:
        print(f"[C8] {r['level']} {r['name']:22s} 구간={r['cells']:>6,} "
              f"Total={r['total']:.6f} 이득={r['gain_vs_deployed']:+.6f} "
              f"({r['share_of_gap']:.1%})")
    print(f"[C8] H1={check['H1_held']} H2={check['H2_held']} -> {check['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
