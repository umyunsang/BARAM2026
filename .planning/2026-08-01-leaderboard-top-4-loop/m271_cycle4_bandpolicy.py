"""M271 P4 사이클 4 — 대역조건부 결정정책의 오라클 상한.

사이클 3 이 표적을 정했다. 대역별 회수가능질량은 `(0.25, 0.45]` 가 0.0236 으로 단일 최대이며
격차 0.0314 의 75% 다. 이 노드는 그 대역의 초과손실이 **결정정책의 전역성** 때문인지 잰다.

가설: 배포 정책 `T0.5_G1.5` 는 전역 스칼라 2 개다. 정산 보상이 계단형이면 최적 행동은
조건부 분포에 따라 달라지므로, 조건부 분산이 다른 대역들에서 같은 정책이 최적일 이유가 없다.
M269 는 "결정층이 자기 계열의 최적점" 이라 했으나 그 계열이 전역 정책이었다면 대역조건부는
계열 밖이다.

**이것은 오라클 측정이다.** 같은 데이터에서 대역별 최적 정책을 고르는 것은 이 프로젝트가
반복적으로 경계해 온 same-fold 선택이다. 따라서 여기서 나오는 값은 **달성 가능량이 아니라
상한**이며, 상한이 유의미할 때만 제대로 검증된 실험을 연다. 그 검증은 이 노드가 하지 않는다.

대역은 **예측값**으로 정한다. 실측값으로 정하면 예보 시점에 알 수 없어 운영 불가이고,
사이클 2 가 확인한 collider 조건화에도 걸린다.

사전확약(실행 전 동결):
  H1  대역별 최적 정책이 전역 최적 정책과 **다르다** (하나 이상의 대역에서)
  H2  오라클 대역조건부 Total 이 전역 최적 Total 보다 **크다**      -> 이득 > 0
  H1 이 기각되면 전역 정책이 이미 대역별로도 최적이므로 이 축은 C8 로 닫는다.

읽기 전용. 모델을 적합하지 않고 2024 행을 읽지 않는다.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from baram.constants import CAPACITIES_KWH, METRIC_COLUMNS
from baram.evaluation.official import evaluate_official

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
PROBE = ROOT / "artifacts" / "backtests" / "metric-aligned-probe"
REPORT_MD = REPORTS / "m271_cycle4_bandpolicy.md"
RECEIPT = REPORTS / "m271_cycle4_bandpolicy_receipt.json"

NODE_ID = "C1N4_BAND_CONDITIONAL_POLICY_ORACLE"
LANE = "L6"  # 문제 해결 접근 방식 — 결정정책의 구조를 묻는다
FOLDS = ("dev-2023-Q2", "dev-2023-Q3", "dev-2023-Q4")
DEPLOYED = "T0.5_G1.5"
# 대역은 **예측값** 기준. 예보 시점에 알 수 있어야 운영 가능하다.
PRED_BAND_EDGES = (0.0, 0.25, 0.45, 0.70, 1.20)


def load_policies() -> tuple[pd.DataFrame, list[str]]:
    parts = []
    for fold in FOLDS:
        parts.append(pd.read_parquet(PROBE / f"M269_PROBE_TOP100-{fold}-policies.parquet"))
    frame = pd.concat(parts, ignore_index=True)
    policies = [c for c in frame.columns if c.startswith("T")]
    frame["forecast_kst_dtm"] = pd.to_datetime(frame["forecast_kst_dtm"])
    frame["capacity"] = frame["group_id"].map(CAPACITIES_KWH).astype(float)
    return frame, sorted(policies)


def score(frame: pd.DataFrame, prediction: np.ndarray) -> float:
    metric = frame.loc[:, ["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh"]].copy()
    metric["prediction_kwh"] = prediction
    metric = metric.loc[:, sorted(METRIC_COLUMNS)]
    return float(evaluate_official(metric, CAPACITIES_KWH).total)


def main() -> int:
    frame, policies = load_policies()

    # 전역 최적: 정책 하나를 전 행에 적용했을 때의 Total.
    global_scores = {p: score(frame, frame[p].to_numpy(dtype=float)) for p in policies}
    global_best = max(global_scores, key=global_scores.get)
    deployed_total = global_scores[DEPLOYED]
    global_best_total = global_scores[global_best]

    # 대역: 배포 정책의 **예측값**으로 정한다.
    pred_ratio = frame[DEPLOYED].to_numpy(dtype=float) / frame["capacity"].to_numpy(dtype=float)
    frame["pred_band"] = pd.cut(pred_ratio, bins=list(PRED_BAND_EDGES), right=True).astype(str)

    # 오라클: 각 대역에서 그 대역의 행만 보고 최적 정책을 고른다. same-fold 선택이다.
    band_choice: dict[str, str] = {}
    band_rows: dict[str, int] = {}
    combined = np.empty(len(frame), dtype=float)
    for band, part in frame.groupby("pred_band", observed=True):
        idx = part.index.to_numpy()
        band_rows[band] = len(idx)
        best_policy, best_value = None, -np.inf
        for p in policies:
            # 대역 부분집합만으로 채점하면 공식 산식의 그룹 구조가 깨질 수 있으므로,
            # 그 대역만 후보 정책으로 바꾸고 나머지는 전역 최적으로 두어 전체를 채점한다.
            trial = frame[global_best].to_numpy(dtype=float).copy()
            trial[idx] = part[p].to_numpy(dtype=float)
            value = score(frame, trial)
            if value > best_value:
                best_policy, best_value = p, value
        band_choice[band] = best_policy
        combined[idx] = part[best_policy].to_numpy(dtype=float)

    oracle_total = score(frame, combined)
    gain_vs_global_best = oracle_total - global_best_total
    gain_vs_deployed = oracle_total - deployed_total
    differs = sorted(b for b, p in band_choice.items() if p != global_best)

    check = {
        "H1_expectation": "대역별 최적 정책이 전역 최적과 다르다",
        "H1_bands_differing": differs,
        "H1_held": bool(differs),
        "H2_expectation": "오라클 대역조건부 Total 이 전역 최적보다 크다",
        "H2_gain": gain_vs_global_best,
        "H2_held": bool(gain_vs_global_best > 0),
        "is_oracle": True,
        "oracle_caveat": (
            "대역별 최적 정책을 같은 데이터에서 골랐다. 달성 가능량이 아니라 상한이다."
        ),
    }

    payload = {
        "policies_considered": len(policies),
        "deployed": {"policy": DEPLOYED, "total": deployed_total},
        "global_best": {"policy": global_best, "total": global_best_total},
        "band_edges_on_prediction": list(PRED_BAND_EDGES),
        "band_choice": band_choice,
        "band_rows": band_rows,
        "oracle_total": oracle_total,
        "gain_vs_global_best": gain_vs_global_best,
        "gain_vs_deployed": gain_vs_deployed,
        "predeclared_check": check,
    }

    lines = [
        "# M271 P4 사이클 4 — 대역조건부 결정정책의 오라클 상한",
        "",
        f"- 판정일: {datetime.now(UTC).strftime('%Y-%m-%d')} (UTC)",
        f"- 노드: `{NODE_ID}` / 레인 {LANE}",
        "- 표적: 사이클 3 이 정한 급경사 대역 (회수가능질량 격차의 75%)",
        "",
        "## 1. 가설",
        "",
        "배포 정책 `T0.5_G1.5` 는 전역 스칼라 2 개다. 정산 보상이 계단형이면 최적 행동은",
        "조건부 분포에 따라 달라지므로, 조건부 분산이 다른 대역들에서 같은 정책이 최적일",
        "이유가 없다. M269 의 '결정층은 자기 계열의 최적점' 은 그 계열이 전역 정책이었다면",
        "대역조건부를 포함하지 않는다.",
        "",
        "대역은 **예측값**으로 정했다. 실측값으로 정하면 예보 시점에 알 수 없고, 사이클 2 가",
        "확인한 collider 조건화에도 걸린다.",
        "",
        "## 2. 측정",
        "",
        f"후보 정책 {len(policies)} 개, 결합 행 {len(frame):,}",
        "",
        "| 항목 | 정책 | Total |",
        "|---|---|---:|",
        f"| 배포 | `{DEPLOYED}` | {deployed_total:.6f} |",
        f"| 전역 최적 | `{global_best}` | {global_best_total:.6f} |",
        f"| **오라클 대역조건부** | 대역별 상이 | **{oracle_total:.6f}** |",
        "",
        "| 예측 대역 | 행수 | 오라클 최적 정책 | 전역 최적과 다름 |",
        "|---|---:|---|:---:|",
    ]
    for band in sorted(band_choice):
        lines.append(
            f"| `{band}` | {band_rows[band]:,} | `{band_choice[band]}` | "
            f"{'예' if band_choice[band] != global_best else '아니오'} |"
        )

    lines += [
        "",
        "## 3. 사전확약 대조",
        "",
        f"- H1 `{check['H1_expectation']}` -> 다른 대역: `{differs}` -> **{check['H1_held']}**",
        f"- H2 `{check['H2_expectation']}` -> 이득 **{gain_vs_global_best:+.6f}** -> "
        f"**{check['H2_held']}**",
        "",
        f"배포 대비 이득: **{gain_vs_deployed:+.6f}**",
        "",
        "## 4. 이 값의 지위",
        "",
        f"**{check['oracle_caveat']}**",
        "",
        "대역별 최적 정책을 같은 fold 에서 골랐으므로 이 이득에는 선택 편향이 전부 들어 있다.",
        "이 프로젝트가 반복적으로 기각해 온 same-fold 오라클과 같은 종류다. 상한이 크면",
        "제대로 검증된 실험(시간순 안전 선택 + 동결 월별 게이트)을 열 근거가 되고, 상한이",
        "작으면 그 자체로 축을 닫는 근거가 된다.",
        "",
        "이 노드는 상한만 잰다. 회수 실험은 별도 노드이며 여기서 하지 않는다.",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    receipt = {
        "schema_version": 1,
        "stage": "M271_P4_CYCLE4_BAND_POLICY_ORACLE",
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

    print(f"[C4] 정책 {len(policies)}개, 행 {len(frame):,}")
    print(f"[C4] 배포 {DEPLOYED} = {deployed_total:.6f}")
    print(f"[C4] 전역최적 {global_best} = {global_best_total:.6f}")
    print(f"[C4] 오라클 대역조건부 = {oracle_total:.6f}")
    print(
        f"[C4] 이득: 전역최적 대비 {gain_vs_global_best:+.6f} / "
        f"배포 대비 {gain_vs_deployed:+.6f}"
    )
    print(f"[C4] H1(대역별 정책 상이)={check['H1_held']} {differs}")
    print(f"[C4] H2(이득>0)={check['H2_held']}  ** 오라클 상한, 달성 가능량 아님 **")
    return 0


if __name__ == "__main__":
    sys.exit(main())
