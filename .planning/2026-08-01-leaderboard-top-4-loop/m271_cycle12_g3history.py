"""M271 P4 사이클 12 — 그룹3 열세가 학습 이력 부족으로 설명되는가.

그룹3 에 이상 신호가 겹친다.

    손실 비중 36.8% (균등 33.3%)   |  unit_0 발전비중 64.2% (g1 56.8, g2 51.7)
    유휴터빈 발생률 21.2% (g1·g2 8%) |  라벨 결측 33.3% (2022 전무)
    유일하게 두 단지(가덕산+원동)에 걸침

핵심은 **g3 만 2022 라벨이 없다**는 것이다. OOF fold 는 셋 다 2023 Q2~Q4 로 같지만, 각 fold
직전까지의 학습 이력은 g3 가 1 년 짧다. 그리고 fold 가 진행될수록 g3 의 이력이 상대적으로
채워진다 — Q2 는 2023 Q1 만, Q4 는 2023 Q1~Q3.

이력 부족이 원인이라면 **g3 의 열세가 Q2 -> Q4 로 가면서 줄어야** 한다. 다른 원인
(두 단지 공간 이질성, 유휴터빈)이라면 fold 에 따라 체계적으로 변하지 않는다.

재적합 없이 기존 fold 산출물로 잰다.

사전확약(실행 전 동결):
  H1  g3 의 FICR 열세(g1·g2 평균 대비)가 Q2 > Q3 > Q4 순으로 **단조 감소**한다.
  H2  Q4 의 열세가 Q2 의 절반 **이하**다.
  둘 다 성립하면 이력 부족이 주원인이고, 그때는 전이·풀링 강화가 회수 경로다.
  기각되면 이력이 아니라 구조(공간 이질성·가용성)가 원인이고, 그 축들은 이미 닫혀 있다.

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

from baram.constants import CAPACITIES_KWH, METRIC_COLUMNS
from baram.evaluation.official import evaluate_official

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
PROBE = ROOT / "artifacts" / "backtests" / "metric-aligned-probe"
REPORT_MD = REPORTS / "m271_cycle12_g3history.md"
RECEIPT = REPORTS / "m271_cycle12_g3history_receipt.json"

NODE_ID = "C1N12_G3_HISTORY"
LANE = "L1"
FOLDS = ("dev-2023-Q2", "dev-2023-Q3", "dev-2023-Q4")
DEPLOYED = "T0.5_G1.5"
# 각 fold 직전까지 g3 가 가진 이력 길이 (2023-01 부터). g1·g2 는 여기에 2022 년이 더 붙는다.
G3_HISTORY_MONTHS = {"dev-2023-Q2": 3, "dev-2023-Q3": 6, "dev-2023-Q4": 9}


def fold_scores(fold: str) -> dict[str, Any]:
    frame = pd.read_parquet(PROBE / f"M269_PROBE_TOP100-{fold}-policies.parquet")
    metric = frame.loc[:, ["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh"]].copy()
    metric["prediction_kwh"] = frame[DEPLOYED].to_numpy(dtype=float)
    result = evaluate_official(metric.loc[:, sorted(METRIC_COLUMNS)], CAPACITIES_KWH)
    ficr = {int(k): float(v) for k, v in result.group_ficr.items()}
    nmae = {int(k): float(v) for k, v in result.group_nmae.items()}
    peers = (ficr[1] + ficr[2]) / 2.0
    return {
        "fold": fold,
        "g3_history_months": G3_HISTORY_MONTHS[fold],
        "group_ficr": ficr,
        "group_nmae": nmae,
        "peer_mean_ficr": peers,
        # 열세 = 동료 평균 - g3. 양수면 g3 가 뒤진다.
        "g3_ficr_shortfall": peers - ficr[3],
        "g3_nmae_excess": nmae[3] - (nmae[1] + nmae[2]) / 2.0,
        "rows": len(frame),
    }


def main() -> int:
    folds = [fold_scores(f) for f in FOLDS]
    shortfalls = [f["g3_ficr_shortfall"] for f in folds]

    monotone = all(
        shortfalls[i] > shortfalls[i + 1] for i in range(len(shortfalls) - 1)
    )
    halved = bool(shortfalls[-1] <= shortfalls[0] / 2.0)

    check = {
        "H1_expectation": "g3 FICR 열세가 Q2 > Q3 > Q4 로 단조 감소",
        "H1_held": bool(monotone),
        "H2_expectation": "Q4 열세가 Q2 의 절반 이하",
        "H2_held": halved,
        "shortfalls": shortfalls,
        "verdict": (
            "HISTORY_LENGTH_EXPLAINS_G3"
            if (monotone and halved)
            else "HISTORY_LENGTH_DOES_NOT_EXPLAIN_G3"
        ),
    }
    payload = {"folds": folds, "predeclared_check": check}

    lines = [
        "# M271 P4 사이클 12 — 그룹3 열세와 학습 이력 길이",
        "",
        f"- 판정일: {datetime.now(UTC).strftime('%Y-%m-%d')} (UTC)",
        f"- 노드: `{NODE_ID}` / 레인 {LANE} / 정책 `{DEPLOYED}`",
        "",
        "## 1. 구조",
        "",
        "g3 만 2022 라벨이 없다. OOF fold 는 셋 다 2023 Q2~Q4 로 같지만 각 fold 직전까지의",
        "학습 이력은 g3 가 1 년 짧고, fold 가 진행될수록 상대적으로 채워진다.",
        "",
        "| fold | g3 가용 이력(개월, 2023 기준) | g1·g2 추가 이력 |",
        "|---|---:|---|",
    ]
    for f in folds:
        lines.append(f"| `{f['fold']}` | {f['g3_history_months']} | 2022 년 12 개월 |")

    lines += [
        "",
        "## 2. fold 별 그룹 점수",
        "",
        "| fold | g1 FICR | g2 FICR | **g3 FICR** | 동료 평균 | **g3 열세** | g3 NMAE 초과 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for f in folds:
        g = f["group_ficr"]
        lines.append(
            f"| `{f['fold']}` | {g[1]:.4f} | {g[2]:.4f} | **{g[3]:.4f}** | "
            f"{f['peer_mean_ficr']:.4f} | **{f['g3_ficr_shortfall']:+.4f}** | "
            f"{f['g3_nmae_excess']:+.4f} |"
        )

    lines += [
        "",
        "## 3. 사전확약 대조",
        "",
        f"- H1 `{check['H1_expectation']}` -> **{check['H1_held']}** "
        f"(`{[round(v, 4) for v in shortfalls]}`)",
        f"- H2 `{check['H2_expectation']}` -> **{check['H2_held']}**",
        "",
        f"판정: **{check['verdict']}**",
        "",
        "## 4. 읽는 법",
        "",
        "`HISTORY_LENGTH_EXPLAINS_G3` 이면 g3 의 열세가 데이터 양 문제이고, 전이·풀링 강화가",
        "회수 경로가 된다. g3 는 전체 손실의 36.8% 를 차지하므로 표적으로 충분히 크다.",
        "",
        "`HISTORY_LENGTH_DOES_NOT_EXPLAIN_G3` 이면 원인은 구조(두 단지 공간 이질성, 유휴터빈",
        "21.2%)이고 그 축들은 이미 닫혀 있다.",
        "",
        "주의: fold 는 3 개뿐이므로 단조성 관측은 약한 증거다. 계절 효과와 이력 길이가",
        "완전히 교락되어 있다 — Q2/Q3/Q4 는 이력이 다르면서 계절도 다르다. 이 노드는 그 둘을",
        "분리하지 못한다.",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    receipt = {
        "schema_version": 1,
        "stage": "M271_P4_CYCLE12_G3_HISTORY",
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

    for f in folds:
        g = f["group_ficr"]
        print(f"[C12] {f['fold']} (g3 이력 {f['g3_history_months']}개월)  "
              f"g1={g[1]:.4f} g2={g[2]:.4f} g3={g[3]:.4f}  "
              f"열세={f['g3_ficr_shortfall']:+.4f}")
    print(f"[C12] H1 단조감소={check['H1_held']}  H2 절반이하={check['H2_held']}")
    print(f"[C12] 판정: {check['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
