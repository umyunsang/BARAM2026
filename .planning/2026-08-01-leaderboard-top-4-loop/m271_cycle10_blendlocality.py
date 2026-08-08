"""M271 P4 사이클 10 — 다중소스 블렌딩 이득이 어디에 떨어지는가.

직전 세션은 외부 NWP 축을 이렇게 닫았다.

    소스간 오차상관 rho ~= 0.78 이므로 무한 소스의 sigma_v 하한은 sigma*sqrt(rho) 이고,
    최선 블렌드는 2.159 m/s 로 4.6% 개선에 그친다. 필요한 것은 11% 다.

그 11% 는 **전 구간 균일 개선** 가정에서 나왔다. 사이클 3 이 그 프레임을 정정했다 — 요구는
균일 개선이 아니라 **평균 미달 셀의 평준화**이고, 격차는 가용 초과손실의 64% 다.

정정된 질문: **그 4.6% 가 어디에 떨어지는가.**

  * 초과비율이 높은 셀에 몰린다 -> 균일 분석이 과소평가했다. 외부 소스 수집이 정당화된다.
  * 고르게 퍼진다             -> 폐쇄가 정정된 프레임에서도 유지된다. 수집이 불필요하다.

새 데이터가 필요 없다. GFS 와 LDAPS 가 이미 2 개 소스이므로 블렌딩 이득의 **국소성**을
지금 잴 수 있다.

사전확약(실행 전 동결):
  H1  블렌딩의 풍속오차 감소가 **고초과 셀에서 더 크다**.
      (고초과 셀 평균 감소율 > 저초과 셀 평균 감소율)
  H1 이 기각되면 이득이 고르게 퍼진다는 뜻이고, 외부 NWP 축은 정정된 프레임에서도 닫힌다.

블렌드 가중은 그룹별로 나셀 풍속에 대해 적합한다. same-fold 적합이므로 이득은 **상한**이다.
상한조차 국소적이지 않으면 실제는 더 그렇다.

읽기 전용. 발전량 모델을 적합하지 않고 2024 행을 읽지 않는다.
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

from m271_cycle1_toprate import turbine_hourly
from m271_deficit import DeficitLedger
from m271_n0_common import load_tables
from m271_n0_deficit_init import annotate, load_deployed

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
REPORT_MD = REPORTS / "m271_cycle10_blendlocality.md"
RECEIPT = REPORTS / "m271_cycle10_blendlocality_receipt.json"

NODE_ID = "C1N10_BLEND_LOCALITY"
LANE = "L2"
U10, V10 = "heightAboveGround_10_10u", "heightAboveGround_10_10v"
MIN_CELL_ROWS = 40


def source_speed(weather: pd.DataFrame, name: str) -> pd.DataFrame:
    hourly = (
        weather.loc[:, ["forecast_kst_dtm", U10, V10]]
        .groupby("forecast_kst_dtm", as_index=False)
        .mean()
    )
    speed = np.hypot(hourly[U10].to_numpy(float), hourly[V10].to_numpy(float))
    return pd.DataFrame({"forecast_kst_dtm": hourly["forecast_kst_dtm"], name: speed})


def main() -> int:
    tables, input_hashes = load_tables()
    ledger = DeficitLedger.from_a7()
    ledger.compute_efficiency()
    excess = {c["key"]: c["excess_ratio"] for c in ledger.cells.values()}

    nacelle = (
        turbine_hourly(tables)
        .groupby(["group_id", "hour"], as_index=False)["ws"]
        .mean()
        .rename(columns={"ws": "nacelle_ws"})
    )
    ldaps = source_speed(tables.ldaps_train, "ldaps_ws")
    gfs = source_speed(tables.gfs_train, "gfs_ws")

    deployed = annotate(load_deployed())
    deployed["cell"] = (
        "group_id=" + deployed["group_id"].astype(str)
        + "|month=" + deployed["month"]
        + "|y_band=" + deployed["y_band"]
    )

    surface = (
        nacelle.merge(ldaps, left_on="hour", right_on="forecast_kst_dtm", how="inner")
        .merge(gfs, on="forecast_kst_dtm", how="inner")
        .merge(
            deployed.loc[:, ["group_id", "forecast_kst_dtm", "cell"]],
            on=["group_id", "forecast_kst_dtm"], how="inner",
        )
        .dropna()
    )

    rows: list[dict[str, Any]] = []
    for group, part in surface.groupby("group_id"):
        target = part["nacelle_ws"].to_numpy(float)
        design_l = np.column_stack([part["ldaps_ws"].to_numpy(float), np.ones(len(part))])
        design_b = np.column_stack(
            [part["ldaps_ws"].to_numpy(float), part["gfs_ws"].to_numpy(float),
             np.ones(len(part))]
        )
        # 그룹별 선형 보정. same-fold 적합이므로 이득은 상한이다.
        coef_l, *_ = np.linalg.lstsq(design_l, target, rcond=None)
        coef_b, *_ = np.linalg.lstsq(design_b, target, rcond=None)
        err_l = np.abs(target - design_l @ coef_l)
        err_b = np.abs(target - design_b @ coef_b)

        frame = pd.DataFrame(
            {"cell": part["cell"].to_numpy(), "err_ldaps": err_l, "err_blend": err_b}
        )
        agg = frame.groupby("cell").agg(
            rows=("err_ldaps", "size"),
            mae_ldaps=("err_ldaps", "mean"),
            mae_blend=("err_blend", "mean"),
        ).reset_index()
        agg = agg.loc[agg["rows"] >= MIN_CELL_ROWS]
        agg["reduction"] = 1.0 - agg["mae_blend"] / agg["mae_ldaps"]
        agg["excess"] = agg["cell"].map(excess)
        agg = agg.dropna(subset=["excess"])
        agg["group"] = int(group)
        rows.append(
            {
                "group": int(group),
                "cells": len(agg),
                "overall_reduction": float(
                    1.0 - err_b.mean() / err_l.mean()
                ),
                "detail": agg.to_dict("records"),
            }
        )

    all_cells = pd.DataFrame([c for r in rows for c in r["detail"]])
    high = all_cells.loc[all_cells["excess"] > 1.0]
    low = all_cells.loc[all_cells["excess"] <= 1.0]
    high_mean = float(high["reduction"].mean()) if len(high) else float("nan")
    low_mean = float(low["reduction"].mean()) if len(low) else float("nan")
    corr = float(all_cells["excess"].corr(all_cells["reduction"]))
    se = float(1.0 / np.sqrt(max(len(all_cells) - 3, 1)))

    check = {
        "H1_expectation": "블렌딩 이득이 고초과 셀에서 더 크다",
        "high_excess_mean_reduction": high_mean,
        "low_excess_mean_reduction": low_mean,
        "difference": high_mean - low_mean,
        "excess_vs_reduction_corr": corr,
        "approx_se": se,
        "sigma_multiple": abs(corr) / se if se else float("nan"),
        "H1_held": bool(high_mean > low_mean),
        "verdict": (
            "BLEND_GAIN_IS_LOCALISED"
            if high_mean > low_mean
            else "BLEND_GAIN_IS_UNIFORM_CLOSURE_HOLDS"
        ),
    }
    payload = {
        "per_group": [{k: v for k, v in r.items() if k != "detail"} for r in rows],
        "cells_analysed": len(all_cells),
        "high_excess_cells": len(high),
        "low_excess_cells": len(low),
        "predeclared_check": check,
        "input_hashes": input_hashes,
    }

    lines = [
        "# M271 P4 사이클 10 — 다중소스 블렌딩 이득의 국소성",
        "",
        f"- 판정일: {datetime.now(UTC).strftime('%Y-%m-%d')} (UTC)",
        f"- 노드: `{NODE_ID}` / 레인 {LANE}",
        "",
        "## 1. 정정된 질문",
        "",
        "직전 세션은 '블렌딩 4.6% vs 필요 11%' 로 외부 NWP 축을 닫았다. 그 11% 는 **전 구간",
        "균일 개선** 가정에서 나왔고 사이클 3 이 그 프레임을 정정했다 — 요구는 균일 개선이",
        "아니라 평균 미달 셀의 평준화다.",
        "",
        "그래서 질문이 바뀐다: **그 이득이 어디에 떨어지는가.**",
        "",
        "GFS 와 LDAPS 가 이미 2 개 소스이므로 새 데이터 없이 잴 수 있다.",
        "",
        "## 2. 그룹별 전체 감소율",
        "",
        "| 그룹 | 셀 수 | 블렌딩 오차 감소율 |",
        "|---:|---:|---:|",
    ]
    for r in payload["per_group"]:
        lines.append(f"| {r['group']} | {r['cells']} | {r['overall_reduction']:.2%} |")

    lines += [
        "",
        "블렌드 가중은 그룹별로 나셀 풍속에 대해 적합했다. same-fold 적합이므로 **상한**이다.",
        "",
        "## 3. 초과비율별 이득",
        "",
        f"분석 셀 {len(all_cells)} 개 (고초과 {len(high)}, 저초과 {len(low)})",
        "",
        "| 구분 | 셀 수 | 평균 오차 감소율 |",
        "|---|---:|---:|",
        f"| 초과비율 > 1 (평균 미달 셀) | {len(high)} | **{high_mean:.2%}** |",
        f"| 초과비율 <= 1 (평균 이상 셀) | {len(low)} | {low_mean:.2%} |",
        f"| 차이 | | **{check['difference']:+.2%}** |",
        "",
        f"초과비율과 감소율의 상관: **{corr:+.4f}** (근사 SE {se:.4f}, "
        f"{check['sigma_multiple']:.1f}배)",
        "",
        "## 4. 사전확약 대조",
        "",
        f"- H1 `{check['H1_expectation']}` -> **{check['H1_held']}**",
        "",
        f"판정: **{check['verdict']}**",
        "",
        "## 5. 읽는 법",
        "",
        "`BLEND_GAIN_IS_UNIFORM_CLOSURE_HOLDS` 이면 이득이 고르게 퍼진다는 뜻이고, 직전",
        "세션의 외부 NWP 폐쇄가 **정정된 프레임에서도 유지된다**. 6~8 시간의 외부 수집이",
        "정당화되지 않는다.",
        "",
        "`BLEND_GAIN_IS_LOCALISED` 이면 균일 분석이 과소평가한 것이고, 외부 소스 수집이",
        "정당화된다.",
        "",
        "어느 쪽이든 여기서 잰 것은 **풍속 오차**이지 공식 점수가 아니다. 풍속이 좋아져도",
        "정산 밴드 적중으로 얼마나 옮겨지는지는 별도 문제다.",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    receipt = {
        "schema_version": 1,
        "stage": "M271_P4_CYCLE10_BLEND_LOCALITY",
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

    for r in payload["per_group"]:
        print(f"[C10] g{r['group']} 셀 {r['cells']:>3} 전체 감소율 {r['overall_reduction']:.2%}")
    print(f"[C10] 고초과 셀 {len(high)}개 평균 감소 {high_mean:.2%} / "
          f"저초과 {len(low)}개 {low_mean:.2%} / 차이 {check['difference']:+.2%}")
    print(f"[C10] 초과비율~감소율 상관 {corr:+.4f} ({check['sigma_multiple']:.1f}x SE)")
    print(f"[C10] 판정: {check['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
