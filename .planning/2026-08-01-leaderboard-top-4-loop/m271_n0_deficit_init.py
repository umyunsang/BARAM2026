"""M271 N0 자식 A7 — 결손 원장 초기화.

동결 사양(`m271_n0_method.SPECS['A7_deficit_init']`)의 3개 산출만 만든다. 세대 3 이며
A1·A4 를 fan-in 한다.

방법 리서치(①)에서 유도한 주장을 여기서 **수치로 검증**한다.

    공식 Total 은 임의의 행 분할에 대해 정확히 가법 분해된다.

      FICR_g = sum_i(a_i * u_i) / sum_i(4 * a_i)
             = sum_C w_C * (ubar_C / 4),   w_C = sum_{i in C} a_i / sum_all a_i
      NMAE_g = mean_i(|e_i|) = sum_C (n_C / n_g) * meanabs_C
      Total  = 0.5*(1 - mean_g NMAE_g) + 0.5*mean_g FICR_g   (둘의 선형결합)

    완전예측 대비 손실을 셀별로 쪼개면 그 합이 (1 - Total) 과 정확히 같아야 한다.

사전확약(실행 전 동결): *잔차가 부동소수 오차 범위(1e-9) 안이어야 한다. 이를 넘으면 그
주장을 철회하고 계획 R9 를 복구한다.*

**분할 축의 성질 구분**: group/month/y대역 은 실측값에만 의존하므로 후보를 바꿔도 셀과
가중치가 불변이다. 반면 정산단위(4/3/0)는 예측에 의존하므로 후보마다 달라진다. 전자를
주 원장으로 쓰고 후자는 진단용 하위 분해로만 둔다.

읽기 전용. 모델을 적합하지 않고 2024 행을 읽지 않는다.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from m271_n0_common import fmt, load_tables, write_node_artifacts

from baram.constants import CAPACITIES_KWH
from baram.evaluation.official import evaluate_official, settlement_unit

NODE = "A7_deficit_init"
ROOT = Path(__file__).resolve().parents[2]
PROBE = ROOT / "artifacts" / "backtests" / "metric-aligned-probe"
FOLDS = ("dev-2023-Q2", "dev-2023-Q3", "dev-2023-Q4")
DEPLOYED_POLICY = "T0.5_G1.5"
TARGET_TOTAL = 0.66
Y_BAND_EDGES = (0.10, 0.25, 0.45, 0.70, 1.10)
RESIDUAL_TOLERANCE = 1e-9


def load_deployed() -> pd.DataFrame:
    parts = []
    for fold in FOLDS:
        path = PROBE / f"M269_PROBE_TOP100-{fold}-policies.parquet"
        frame = pd.read_parquet(path)
        part = frame.loc[:, ["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh"]].copy()
        part["prediction_kwh"] = frame[DEPLOYED_POLICY].to_numpy(dtype=float)
        parts.append(part)
    out = pd.concat(parts, ignore_index=True)
    out["forecast_kst_dtm"] = pd.to_datetime(out["forecast_kst_dtm"])
    return out


def annotate(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["capacity"] = out["group_id"].map(CAPACITIES_KWH).astype(float)
    out["y"] = out["actual_kwh"] / out["capacity"]
    out["abs_err_rate"] = (out["prediction_kwh"] - out["actual_kwh"]).abs() / out["capacity"]
    out["unit"] = settlement_unit(out["abs_err_rate"].to_numpy(dtype=float))
    out["month"] = out["forecast_kst_dtm"].dt.to_period("M").astype(str)
    out["y_band"] = pd.cut(out["y"], bins=list(Y_BAND_EDGES), right=True).astype(str)
    # 공식 산식의 유효행: 실측값에만 의존한다.
    return out.loc[out["actual_kwh"] >= 0.10 * out["capacity"]].reset_index(drop=True)


def build_ledger(eligible: pd.DataFrame, axes: list[str]) -> tuple[pd.DataFrame, dict[str, float]]:
    """셀별 손실 기여를 만든다. 합이 (1 - Total) 과 같아야 한다.

    그룹별 분모: FICR 은 발전량 가중, NMAE 는 행수 가중이다. 최종 Total 은 세 그룹의
    산술평균이므로 각 그룹에 1/3 이 곱해진다.
    """
    totals = eligible.groupby("group_id").agg(
        gen_sum=("actual_kwh", "sum"), rows=("actual_kwh", "size")
    )
    cells = (
        eligible.groupby(["group_id", *axes], observed=True)
        .agg(
            rows=("actual_kwh", "size"),
            gen_sum=("actual_kwh", "sum"),
            gen_weighted_unit=("unit", lambda s: 0.0),
            mean_abs_err_rate=("abs_err_rate", "mean"),
        )
        .reset_index()
    )
    # 발전량 가중 평균 단위를 별도로 계산한다.
    weighted = (
        eligible.assign(_wu=eligible["actual_kwh"] * eligible["unit"])
        .groupby(["group_id", *axes], observed=True)["_wu"]
        .sum()
        .reset_index(name="gen_unit_sum")
    )
    cells = cells.drop(columns=["gen_weighted_unit"]).merge(
        weighted, on=["group_id", *axes], how="left"
    )
    cells["ubar"] = cells["gen_unit_sum"] / cells["gen_sum"]

    cells["w_gen"] = cells.apply(
        lambda r: r["gen_sum"] / totals.loc[r["group_id"], "gen_sum"], axis=1
    )
    cells["w_rows"] = cells.apply(
        lambda r: r["rows"] / totals.loc[r["group_id"], "rows"], axis=1
    )
    # 완전예측 대비 손실 기여. 세 그룹 평균이므로 1/3, Total 의 절반씩이므로 0.5.
    cells["ficr_loss"] = 0.5 * (1.0 / 3.0) * cells["w_gen"] * (1.0 - cells["ubar"] / 4.0)
    cells["nmae_loss"] = 0.5 * (1.0 / 3.0) * cells["w_rows"] * cells["mean_abs_err_rate"]
    cells["total_loss"] = cells["ficr_loss"] + cells["nmae_loss"]
    cells = cells.sort_values("total_loss", ascending=False).reset_index(drop=True)

    summary = {
        "ficr_loss_sum": float(cells["ficr_loss"].sum()),
        "nmae_loss_sum": float(cells["nmae_loss"].sum()),
        "total_loss_sum": float(cells["total_loss"].sum()),
    }
    return cells, summary


def run(tables: Any, input_hashes: dict[str, str]) -> dict[str, Any]:
    deployed = load_deployed()
    metric_frame = deployed.loc[
        :, ["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh", "prediction_kwh"]
    ]
    official = evaluate_official(metric_frame, CAPACITIES_KWH)

    eligible = annotate(deployed)
    axes = ["month", "y_band"]
    cells, summary = build_ledger(eligible, axes)

    implied_loss = 1.0 - official.total
    residual = summary["total_loss_sum"] - implied_loss
    held = abs(residual) <= RESIDUAL_TOLERANCE

    gap_to_target = TARGET_TOTAL - official.total

    # 정산단위 하위분해는 예측 의존이므로 진단용으로만 둔다.
    tier = (
        eligible.assign(tier=eligible["unit"].map({4.0: "unit_4", 3.0: "unit_3", 0.0: "unit_0"}))
        .groupby(["group_id", "tier"], observed=True)
        .agg(rows=("actual_kwh", "size"), gen_sum=("actual_kwh", "sum"))
        .reset_index()
    )
    gen_total = eligible.groupby("group_id")["actual_kwh"].sum()
    tier["gen_share"] = tier.apply(lambda r: r["gen_sum"] / gen_total[r["group_id"]], axis=1)

    top_cells = cells.head(20).to_dict("records")
    payload: dict[str, Any] = {
        "source": {
            "policy": DEPLOYED_POLICY,
            "folds": list(FOLDS),
            "rows_all": len(deployed),
            "rows_eligible": len(eligible),
        },
        "official": {
            "total": official.total,
            "one_minus_nmae": official.one_minus_nmae,
            "ficr": official.ficr,
            "group_ficr": {int(k): float(v) for k, v in official.group_ficr.items()},
            "group_nmae": {int(k): float(v) for k, v in official.group_nmae.items()},
        },
        "target": TARGET_TOTAL,
        "gap_to_target": gap_to_target,
        "decomposition": {
            "axes": ["group_id", *axes],
            "cells": len(cells),
            "implied_loss_1_minus_total": implied_loss,
            **summary,
            "residual": residual,
            "tolerance": RESIDUAL_TOLERANCE,
            "predeclared_held": held,
            "verdict": "ADDITIVE_EXACT" if held else "WITHDRAWN_RESTORE_R9",
        },
        # 전 셀을 남긴다. 상위 20 만 남기면 결손 원장이 나머지를 하나로 접게 되고, 그 접힌
        # 덩어리가 최대 셀이 되어 우선순위를 왜곡한다(P4 부트스트랩이 실측한 결함).
        "cells": cells.to_dict("records"),
        "top_cells": top_cells,
        "settlement_tier_diagnostic": tier.to_dict("records"),
        "axis_note": (
            "group/month/y_band 는 실측값에만 의존하므로 후보를 바꿔도 셀과 가중치가 불변이다. "
            "정산단위(4/3/0)는 예측 의존이라 후보마다 달라지므로 주 원장 축에 넣지 않았다."
        ),
    }

    dec = payload["decomposition"]
    off = payload["official"]
    lines = [
        "## 1. 기준선",
        "",
        f"- 정책: `{DEPLOYED_POLICY}` / fold: {', '.join(FOLDS)}",
        f"- 전체 행 {payload['source']['rows_all']:,} 중 **유효행 "
        f"{payload['source']['rows_eligible']:,}**",
        "",
        "| 지표 | 값 |",
        "|---|---:|",
        f"| 공식 Total | **{fmt(off['total'], 6)}** |",
        f"| 1-NMAE | {fmt(off['one_minus_nmae'], 6)} |",
        f"| FICR | {fmt(off['ficr'], 6)} |",
        f"| 목표 | {TARGET_TOTAL} |",
        f"| **격차** | **{fmt(payload['gap_to_target'], 6)}** |",
        "",
        "## 2. 가법 분해 검증 (사전확약 대조)",
        "",
        "방법 리서치에서 유도한 주장을 수치로 검증한다. 완전예측 대비 손실을 셀별로 쪼갠",
        "합이 `1 - Total` 과 같아야 한다.",
        "",
        "| 항목 | 값 |",
        "|---|---:|",
        f"| 셀 수 (`{' x '.join(dec['axes'])}`) | {dec['cells']:,} |",
        f"| 셀 손실 합 | `{dec['total_loss_sum']:.15f}` |",
        f"| `1 - Total` | `{dec['implied_loss_1_minus_total']:.15f}` |",
        f"| **잔차** | **`{dec['residual']:.3e}`** |",
        f"| 허용 | `{dec['tolerance']:.0e}` |",
        "",
        f"→ 사전확약 유지: **{dec['predeclared_held']}** / 판정: **{dec['verdict']}**",
        "",
        f"- FICR 측 손실 합: {fmt(dec['ficr_loss_sum'], 6)}",
        f"- NMAE 측 손실 합: {fmt(dec['nmae_loss_sum'], 6)}",
        "",
        payload["axis_note"],
        "",
        "## 3. 결손 상위 20 셀",
        "",
        "`total_loss` 는 완전예측 대비 이 셀이 잃는 Total 의 양이다. 전 셀 합이 `1 - Total`",
        "이므로 셀 간 비교가 그대로 우선순위가 된다.",
        "",
        "| 그룹 | 월 | y대역 | 행수 | 발전량비중 | 평균단위 | FICR손실 | NMAE손실 | **총손실** |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for c in top_cells:
        lines.append(
            f"| {int(c['group_id'])} | {c['month']} | `{c['y_band']}` | {int(c['rows']):,} | "
            f"{c['w_gen']:.1%} | {fmt(c['ubar'], 3)} | {fmt(c['ficr_loss'], 5)} | "
            f"{fmt(c['nmae_loss'], 5)} | **{fmt(c['total_loss'], 5)}** |"
        )

    lines += [
        "",
        "## 4. 정산단위 분해 (진단 전용, 예측 의존)",
        "",
        "| 그룹 | 단위 | 행수 | 발전량 비중 |",
        "|---:|---|---:|---:|",
    ]
    for t in payload["settlement_tier_diagnostic"]:
        lines.append(
            f"| {int(t['group_id'])} | `{t['tier']}` | {int(t['rows']):,} | {t['gen_share']:.1%} |"
        )

    lines += [
        "",
        "## 5. 원장으로서의 성질",
        "",
        "이 표가 M271 루프의 연료다. 셀에 미설명 손실이 남아 있는 한 C1(기전 없음)이 발화해",
        "새 노드를 낳는다. 분해축을 추가하거나 잔차를 재귀속하면 새 셀이 생기므로 원장은",
        "고갈되지 않는다.",
        "",
        "다만 이 원장은 **현재 후보의 손실 분포**이지 '회수 가능한 양'이 아니다. 각 셀의",
        "회수 가능량은 그 셀의 기전이 밝혀진 뒤에야 추정할 수 있고, 그것이 방향 리서치의 일이다.",
    ]

    write_node_artifacts(
        node=NODE,
        title="M271 N0/A7 — 결손 원장 초기화",
        report_lines=lines,
        payload=payload,
        input_hashes=input_hashes,
        parents=["A1_labels", "A4_error"],
        script_path=Path(__file__),
    )
    return payload


def main() -> int:
    tables, hashes = load_tables()
    payload = run(tables, hashes)
    dec = payload["decomposition"]
    off = payload["official"]
    print(f"[A7] 로컬 Total={off['total']:.6f} 1-NMAE={off['one_minus_nmae']:.6f} "
          f"FICR={off['ficr']:.6f}")
    print(f"[A7] 목표 {payload['target']} 격차={payload['gap_to_target']:.6f}")
    print(f"[A7] 셀 {dec['cells']:,}개, 손실합={dec['total_loss_sum']:.12f} "
          f"vs 1-Total={dec['implied_loss_1_minus_total']:.12f}")
    print(f"[A7] 잔차={dec['residual']:.3e} (허용 {dec['tolerance']:.0e}) -> {dec['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
