"""M271 P4 사이클 3 — 손실 질량 축 규명과 풍향 재분해 (C12 REFINE_AXIS).

단일 셀(최대 1.2%)을 계속 파는 것은 질량 기준이 아니다. 이 노드는 두 가지를 한다.

  1. 기존 축(group / 월 / y대역)별로 손실 질량을 **집계**해 어디에 몰리는지 본다.
  2. 큐에 대기 중이던 C12 를 실행해 결손 원장을 **풍향 섹터**로 재분해한다.

풍향을 고른 이유는 질량이 아니라 **회수 가능성**이다. 사이클 2 가 확인한 가용성 기전은
예보 시점에 알 수 없다(평가기간 SCADA 부재). 풍향은 NWP 에서 예보 시점에 알 수 있으므로
피처로 만들 수 있다. A4 는 240 도 섹터가 행의 43%·최고 풍속·최악 오차를 동시에 차지함을
이미 측정했다.

사전확약(실행 전 동결):
  H1  240 도 섹터의 손실 질량 비중이 그 섹터의 발전량 비중보다 **크다**.
      (즉 단위 발전량당 손실이 평균 이상)                     -> 초과 비율 > 1
  H2  섹터별 손실 집중도가 y대역별 집중도보다 **높지 않다**.
      기존 축이 이미 잘 잡고 있다면 새 축의 이득이 없다는 뜻이다.
  H1 이 성립하면 풍향은 회수 대상 축이다. 성립하지 않으면 C8 로 닫는다.

가법 항등식은 재분해 후에도 유지되어야 한다. 깨지면 분해가 틀린 것이다.

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

from m271_deficit import DeficitLedger
from m271_n0_common import load_tables
from m271_n0_deficit_init import annotate, build_ledger, load_deployed

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
REPORT_MD = REPORTS / "m271_cycle3_massaxis.md"
RECEIPT = REPORTS / "m271_cycle3_massaxis_receipt.json"

NODE_ID = "C1N3_MASS_AXIS_WIND_SECTOR"
LANE = "L2"
SECTOR_WIDTH_DEG = 30.0
IDENTITY_TOLERANCE = 1e-9


def nwp_direction_hourly(ldaps: pd.DataFrame) -> pd.DataFrame:
    """격자평균 10m 바람의 방향(불어오는 쪽)과 섹터. 예보 시점에 알 수 있는 값이다."""
    keys = ["forecast_kst_dtm"]
    cols = ["heightAboveGround_10_10u", "heightAboveGround_10_10v"]
    hourly = ldaps.loc[:, [*keys, *cols]].groupby(keys, as_index=False).mean()
    u = hourly[cols[0]].to_numpy(dtype=float)
    v = hourly[cols[1]].to_numpy(dtype=float)
    hourly["nwp_ws10"] = np.hypot(u, v)
    hourly["nwp_wd10"] = (np.degrees(np.arctan2(-u, -v)) + 360.0) % 360.0
    hourly["wind_sector"] = (
        (hourly["nwp_wd10"] // SECTOR_WIDTH_DEG).astype(int) * int(SECTOR_WIDTH_DEG)
    ).astype(str)
    return hourly.loc[:, ["forecast_kst_dtm", "nwp_ws10", "wind_sector"]]


def aggregate_existing_axes(ledger: DeficitLedger) -> dict[str, list[dict[str, Any]]]:
    """기존 축별 손실 질량 집계. 어디에 몰리는지 먼저 본다."""
    rows = [
        {**c["axes"], "loss": c["loss_share"], "ficr": c["ficr_loss"], "nmae": c["nmae_loss"]}
        for c in ledger.cells.values()
    ]
    frame = pd.DataFrame(rows)
    total = frame["loss"].sum()
    out: dict[str, list[dict[str, Any]]] = {}
    for axis in ("group_id", "month", "y_band"):
        agg = (
            frame.groupby(axis)[["loss", "ficr", "nmae"]]
            .sum()
            .sort_values("loss", ascending=False)
            .reset_index()
        )
        agg["share"] = agg["loss"] / total
        out[axis] = agg.to_dict("records")
    return out


def refine_by_sector(tables: Any) -> tuple[pd.DataFrame, dict[str, Any]]:
    """결손 원장을 (group x 월 x y대역 x 풍향섹터) 로 재분해한다."""
    deployed = annotate(load_deployed())
    direction = nwp_direction_hourly(tables.ldaps_train)
    merged = deployed.merge(direction, on="forecast_kst_dtm", how="inner")

    cells, summary = build_ledger(merged, ["month", "y_band", "wind_sector"])

    # 발전량 비중 대비 손실 비중을 본다. 초과 비율 > 1 이면 단위 발전량당 손실이 평균 이상.
    total_loss = cells["total_loss"].sum()
    sector = (
        cells.groupby("wind_sector")
        .agg(loss=("total_loss", "sum"), rows=("rows", "sum"), gen=("gen_sum", "sum"))
        .reset_index()
    )
    sector["loss_share"] = sector["loss"] / total_loss
    sector["gen_share"] = sector["gen"] / sector["gen"].sum()
    sector["excess_ratio"] = sector["loss_share"] / sector["gen_share"].clip(lower=1e-9)
    sector = sector.sort_values("loss", ascending=False).reset_index(drop=True)
    return cells, {"sector": sector.to_dict("records"), **summary}


def concentration(shares: list[float]) -> float:
    """상위 25% 축 수준이 차지하는 손실 비중. 집중도가 높을수록 표적이 뚜렷하다."""
    ordered = sorted(shares, reverse=True)
    take = max(1, len(ordered) // 4)
    return float(sum(ordered[:take]))


def main() -> int:
    ledger = DeficitLedger.from_a7()
    by_axis = aggregate_existing_axes(ledger)

    tables, input_hashes = load_tables()
    refined, sector_summary = refine_by_sector(tables)

    implied = 1.0 - ledger.total
    residual = sector_summary["total_loss_sum"] - implied
    identity_ok = abs(residual) <= IDENTITY_TOLERANCE

    sectors = sector_summary["sector"]
    top_sector = sectors[0]
    sector_conc = concentration([s["loss_share"] for s in sectors])
    band_conc = concentration([r["share"] for r in by_axis["y_band"]])

    check = {
        "H1_expectation": "240 도 섹터의 손실 비중이 발전량 비중보다 크다 (초과 비율 > 1)",
        "H1_target_sector": "240",
        "H1_excess_ratio": next(
            (s["excess_ratio"] for s in sectors if s["wind_sector"] == "240"), float("nan")
        ),
        "H1_held": bool(
            next((s["excess_ratio"] for s in sectors if s["wind_sector"] == "240"), 0.0) > 1.0
        ),
        "H2_expectation": "섹터 집중도가 y대역 집중도보다 높지 않다",
        "sector_top25_concentration": sector_conc,
        "y_band_top25_concentration": band_conc,
        "H2_held": bool(sector_conc <= band_conc),
        "identity_residual": residual,
        "identity_ok": identity_ok,
    }

    payload = {
        "ledger_total": ledger.total,
        "gap_to_target": ledger.gap_to_target(),
        "loss_by_existing_axis": by_axis,
        "refined_cells": len(refined),
        "sector_summary": sectors,
        "predeclared_check": check,
        "input_hashes": input_hashes,
    }

    lines = [
        "# M271 P4 사이클 3 — 손실 질량 축과 풍향 재분해",
        "",
        f"- 판정일: {datetime.now(UTC).strftime('%Y-%m-%d')} (UTC)",
        f"- 노드: `{NODE_ID}` / 레인 {LANE} / 라우팅 근거: C12 REFINE_AXIS",
        f"- 로컬 Total {ledger.total:.6f} / 목표 {ledger.target} / 격차 "
        f"{ledger.gap_to_target():.6f}",
        "",
        "단일 셀(최대 1.2%)을 계속 파는 것은 질량 기준이 아니다. 먼저 기존 축별로 질량을",
        "집계하고, 그다음 **예보 시점에 알 수 있는** 풍향 축으로 재분해한다.",
        "",
        "풍향을 고른 이유는 질량이 아니라 회수 가능성이다. 사이클 2 가 확인한 가용성 기전은",
        "평가기간 SCADA 가 없어 예보 시점에 알 수 없다. 풍향은 NWP 에서 알 수 있다.",
        "",
        "## 1. 기존 축별 손실 질량",
        "",
    ]
    for axis, label in (("y_band", "y 대역"), ("month", "월"), ("group_id", "그룹")):
        lines += [
            f"### {label}",
            "",
            f"| {label} | 손실 | 비중 | FICR측 | NMAE측 |",
            "|---|---:|---:|---:|---:|",
        ]
        for r in by_axis[axis]:
            lines.append(
                f"| `{r[axis]}` | {r['loss']:.5f} | **{r['share']:.1%}** | "
                f"{r['ficr']:.5f} | {r['nmae']:.5f} |"
            )
        lines.append("")

    lines += [
        "## 2. 풍향 섹터 재분해 (C12)",
        "",
        f"재분해 후 셀 {payload['refined_cells']:,} 개. 가법 항등식 잔차 "
        f"`{residual:.3e}` (허용 `{IDENTITY_TOLERANCE:.0e}`) -> **{identity_ok}**",
        "",
        "초과 비율 = 손실 비중 / 발전량 비중. 1 보다 크면 단위 발전량당 손실이 평균 이상이다.",
        "",
        "| 섹터 | 행수 | 손실 | 손실 비중 | 발전량 비중 | **초과 비율** |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for s in sectors:
        lines.append(
            f"| {s['wind_sector']} | {int(s['rows']):,} | {s['loss']:.5f} | "
            f"{s['loss_share']:.1%} | {s['gen_share']:.1%} | "
            f"**{s['excess_ratio']:.3f}** |"
        )

    lines += [
        "",
        "## 3. 사전확약 대조",
        "",
        f"- H1 `{check['H1_expectation']}`",
        f"  - 240 도 초과 비율 **{check['H1_excess_ratio']:.3f}** -> 유지: "
        f"**{check['H1_held']}**",
        f"- H2 `{check['H2_expectation']}`",
        f"  - 섹터 상위25% 집중도 {sector_conc:.1%} vs y대역 {band_conc:.1%} -> 유지: "
        f"**{check['H2_held']}**",
        "",
        "## 4. 읽는 법",
        "",
        "H1 이 성립하면 풍향은 회수 대상 축이다 — 그 섹터에서 단위 발전량당 손실이 크고,",
        "풍향은 예보 시점에 알 수 있으므로 조건부 처리가 가능하다.",
        "",
        "H1 이 성립하지 않으면 손실이 풍향에 대해 균일하다는 뜻이고, 그 경우 이 축은 회수",
        "표적이 아니므로 C8 로 닫는 것이 맞다. A4 가 잰 '240 도가 최악 오차' 는 **행 단위**",
        "평균 오차였고, 여기서 재는 것은 **발전량 가중 손실**이다. 둘은 다를 수 있다.",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    receipt = {
        "schema_version": 1,
        "stage": "M271_P4_CYCLE3_MASS_AXIS",
        "node": NODE_ID,
        "lane": LANE,
        "routed_by": "C12_refine_axis",
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

    print("[C3] 기존 축별 손실 비중 (상위)")
    for axis in ("y_band", "month", "group_id"):
        head = by_axis[axis][0]
        print(f"     {axis:9s} 최대 `{head[axis]}` {head['share']:.1%}")
    print(
        f"[C3] 재분해 셀 {payload['refined_cells']:,}, "
        f"항등식 잔차 {residual:.3e} -> {identity_ok}"
    )
    print(f"[C3] 최대 손실 섹터 {top_sector['wind_sector']}도: 손실비중 "
          f"{top_sector['loss_share']:.1%} / 발전비중 {top_sector['gen_share']:.1%} "
          f"/ 초과비율 {top_sector['excess_ratio']:.3f}")
    print(f"[C3] H1(240도 초과비율>1)={check['H1_held']} ({check['H1_excess_ratio']:.3f})  "
          f"H2={check['H2_held']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
