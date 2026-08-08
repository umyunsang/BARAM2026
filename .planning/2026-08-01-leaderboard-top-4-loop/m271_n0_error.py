"""M271 N0 자식 A4 — 층화 예보검증.

동결 사양(`m271_n0_method.SPECS['A4_error']`)의 5개 산출만 만든다. 세대 2 이며 A1·A3 를
fan-in 한다.

방법 리서치(①) 결과: 층화 예보검증을 채택하되 `xskillscore`/`scores` 는 미채택이다(격자
xarray 대상, 우리는 tabular). 필요한 것은 지표 구현이 아니라 층화 규약이다.

**모델을 적합하지 않는다.** 측정 대상은 NWP 풍속과 나셀 풍속의 차이이며, 회귀 보정 대신
그룹별 **중앙값 편향 제거**만 적용한다. 나셀 풍속계는 로터 뒤에 있어 실제 유입풍속이
아니므로 여기서 나온 오차는 예보 오차의 **상한**이다.

사전확약(실행 전 동결): *max-min 스프레드가 오차와 양의 관계를 가질 것으로 예상한다.
부호가 반대면 C5(anomaly)로 라우팅된다.*

읽기 전용. 2024 행을 읽지 않는다.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from m271_n0_common import fmt, load_tables, write_node_artifacts
from m271_n0_scada import scada_long

NODE = "A4_error"
SECTOR_WIDTH_DEG = 30.0  # 12 섹터
MIN_STRATUM_ROWS = 100
SPREAD_PAIRS = {
    "spread_u50": ("heightAboveGround_50_50MUmax", "heightAboveGround_50_50MUmin"),
    "spread_v50": ("heightAboveGround_50_50MVmax", "heightAboveGround_50_50MVmin"),
}


def nwp_hourly(ldaps: pd.DataFrame) -> pd.DataFrame:
    """격자평균 10m 바람과 50m max-min 스프레드를 시간별로 만든다."""
    columns = ["heightAboveGround_10_10u", "heightAboveGround_10_10v"]
    for pair in SPREAD_PAIRS.values():
        columns.extend(pair)
    missing = [c for c in columns if c not in ldaps.columns]
    if missing:
        raise KeyError(f"LDAPS missing columns: {missing}")

    keys = ["forecast_kst_dtm", "lead_hour"]
    hourly = ldaps.loc[:, [*keys, *columns]].groupby(keys, as_index=False).mean()

    u = hourly["heightAboveGround_10_10u"].to_numpy(dtype=float)
    v = hourly["heightAboveGround_10_10v"].to_numpy(dtype=float)
    hourly["nwp_ws10"] = np.hypot(u, v)
    # 기상 관례: 바람이 불어오는 방향. 0/360 = 북, 90 = 동.
    hourly["nwp_wd10"] = (np.degrees(np.arctan2(-u, -v)) + 360.0) % 360.0
    hourly["sector"] = (hourly["nwp_wd10"] // SECTOR_WIDTH_DEG).astype(int) * int(
        SECTOR_WIDTH_DEG
    )
    for name, (hi, lo) in SPREAD_PAIRS.items():
        hourly[name] = hourly[hi] - hourly[lo]
    hourly["spread_vec"] = np.hypot(hourly["spread_u50"], hourly["spread_v50"])
    return hourly


def nacelle_hourly(scada: pd.DataFrame, turbines: pd.DataFrame) -> pd.DataFrame:
    key = turbines.loc[:, ["manufacturer", "turbine_number", "group_id"]]
    joined = scada.merge(key, on=["manufacturer", "turbine_number"], how="inner")
    joined["hour"] = joined["kst_dtm"].dt.floor("h")
    return (
        joined.groupby(["group_id", "hour"], as_index=False)["ws"]
        .mean()
        .rename(columns={"ws": "nacelle_ws"})
    )


def build_surface(ldaps: pd.DataFrame, scada: pd.DataFrame, turbines: pd.DataFrame) -> pd.DataFrame:
    nwp = nwp_hourly(ldaps)
    nac = nacelle_hourly(scada, turbines)
    merged = nac.merge(nwp, left_on="hour", right_on="forecast_kst_dtm", how="inner")
    merged["error_raw"] = merged["nwp_ws10"] - merged["nacelle_ws"]
    # 회귀를 적합하지 않는다. 그룹별 중앙값 편향만 제거한다.
    bias = merged.groupby("group_id")["error_raw"].transform("median")
    merged["error"] = merged["error_raw"] - bias
    merged["abs_error"] = merged["error"].abs()
    merged["hour_of_day"] = merged["hour"].dt.hour
    merged["month"] = merged["hour"].dt.month
    return merged


def stratify(frame: pd.DataFrame, axis: str) -> list[dict[str, Any]]:
    out = []
    for (group, level), part in frame.groupby(["group_id", axis], observed=True):
        if len(part) < MIN_STRATUM_ROWS:
            continue
        out.append(
            {
                "group": int(group),
                "axis": axis,
                "level": int(level),
                "rows": len(part),
                "mean_abs_error": float(part["abs_error"].mean()),
                "median_abs_error": float(part["abs_error"].median()),
                "bias": float(part["error"].mean()),
                "std_error": float(part["error"].std()),
                "mean_nacelle_ws": float(part["nacelle_ws"].mean()),
            }
        )
    return out


def _residualise(y: pd.Series, x: pd.Series) -> np.ndarray:
    """2차 다항 잔차화. 통제변수와의 비선형 관계를 남기지 않기 위함."""
    xv, yv = x.to_numpy(dtype=float), y.to_numpy(dtype=float)
    coef = np.polyfit(xv, yv, 2)
    return yv - np.polyval(coef, xv)


def spread_relation(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """스프레드와 |오차| 의 관계. 풍속을 통제한 편상관을 함께 낸다."""
    out = []
    for group, part in frame.groupby("group_id"):
        entry: dict[str, Any] = {"group": int(group), "rows": len(part)}
        resid_err = _residualise(part["abs_error"], part["nwp_ws10"])
        for name in [*SPREAD_PAIRS, "spread_vec"]:
            resid_spread = _residualise(part[name], part["nwp_ws10"])
            entry[f"{name}_corr"] = float(part[name].corr(part["abs_error"]))
            entry[f"{name}_partial"] = float(np.corrcoef(resid_err, resid_spread)[0, 1])
        entry["approx_se"] = float(1.0 / np.sqrt(max(len(part) - 3, 1)))
        out.append(entry)
    return out


def run(tables: Any, input_hashes: dict[str, str]) -> dict[str, Any]:
    scada = scada_long(tables.scada_vestas, tables.scada_unison)
    surface = build_surface(tables.ldaps_train, scada, tables.turbines)

    by_lead = stratify(surface, "lead_hour")
    by_hour = stratify(surface, "hour_of_day")
    by_month = stratify(surface, "month")
    by_sector = stratify(surface, "sector")
    spread = spread_relation(surface)

    all_positive = all(e["spread_vec_partial"] > 0 for e in spread)
    any_beyond_noise = any(
        abs(e["spread_vec_partial"]) > 2 * e["approx_se"] for e in spread
    )

    payload: dict[str, Any] = {
        "surface_rows": len(surface),
        "period": [str(surface["hour"].min()), str(surface["hour"].max())],
        "by_lead_hour": by_lead,
        "by_hour_of_day": by_hour,
        "by_month": by_month,
        "by_sector": by_sector,
        "spread_relation": spread,
        "predeclared_check": {
            "expectation": "max-min 스프레드는 |오차| 와 양의 관계",
            "spread_vec_partial_by_group": {e["group"]: e["spread_vec_partial"] for e in spread},
            "all_positive": all_positive,
            "any_beyond_2se": any_beyond_noise,
            "verdict": "HELD" if all_positive else "SIGN_REVERSED_ROUTE_C5",
        },
        "caveat": (
            "나셀 풍속계는 로터 뒤에 있고 10분 값을 시간평균했다. 여기서 나온 오차는 예보 "
            "오차의 상한이지 예보 오차 자체가 아니다."
        ),
    }

    lines = [
        "## 1. 표면",
        "",
        f"- 결합 행: **{payload['surface_rows']:,}** (그룹 x 시간)",
        f"- 기간: {payload['period'][0][:10]} ~ {payload['period'][1][:10]}",
        "",
        "오차 = (LDAPS 격자평균 10m 풍속) - (그룹 나셀 풍속 시간평균), 그룹별 중앙값 편향 제거.",
        "**회귀를 적합하지 않았다.**",
        "",
        payload["caveat"],
        "",
        "## 2. 리드타임별",
        "",
        "| 그룹 | 리드(h) | 행수 | 평균 \\|오차\\| | 편향 | 표준편차 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for r in by_lead:
        lines.append(
            f"| {r['group']} | {r['level']} | {r['rows']:,} | {fmt(r['mean_abs_error'], 3)} | "
            f"{fmt(r['bias'], 3)} | {fmt(r['std_error'], 3)} |"
        )

    lines += [
        "",
        "## 3. 시각별",
        "",
        "| 그룹 | 시각 | 행수 | 평균 \\|오차\\| | 편향 | 평균 나셀풍속 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for r in by_hour:
        lines.append(
            f"| {r['group']} | {r['level']:02d} | {r['rows']:,} | {fmt(r['mean_abs_error'], 3)} | "
            f"{fmt(r['bias'], 3)} | {fmt(r['mean_nacelle_ws'], 3)} |"
        )

    lines += [
        "",
        "## 4. 월별",
        "",
        "| 그룹 | 월 | 행수 | 평균 \\|오차\\| | 편향 | 평균 나셀풍속 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for r in by_month:
        lines.append(
            f"| {r['group']} | {r['level']:02d} | {r['rows']:,} | {fmt(r['mean_abs_error'], 3)} | "
            f"{fmt(r['bias'], 3)} | {fmt(r['mean_nacelle_ws'], 3)} |"
        )

    lines += [
        "",
        "## 5. 풍향 섹터별",
        "",
        f"섹터 폭 {SECTOR_WIDTH_DEG:.0f}도, NWP 10m 바람이 불어오는 방향 기준.",
        "",
        "| 그룹 | 섹터 | 행수 | 평균 \\|오차\\| | 편향 | 평균 나셀풍속 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for r in by_sector:
        lines.append(
            f"| {r['group']} | {r['level']:03d} | {r['rows']:,} | "
            f"{fmt(r['mean_abs_error'], 3)} | {fmt(r['bias'], 3)} | "
            f"{fmt(r['mean_nacelle_ws'], 3)} |"
        )

    lines += [
        "",
        "## 6. LDAPS 50m max-min 스프레드와 오차",
        "",
        "직전 세션은 이 신호를 찾으려고 외부 GEFS 앙상블 스프레드를 수집했다. 공급 데이터의",
        "`50MUmax/min`·`50MVmax/min` 자체가 변동성 공변량 후보이므로 여기서 먼저 잰다.",
        "",
        "| 그룹 | 행수 | vec 상관 | **vec 편상관** | u50 편상관 | v50 편상관 | 근사 SE |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for e in spread:
        lines.append(
            f"| {e['group']} | {e['rows']:,} | {fmt(e['spread_vec_corr'], 4)} | "
            f"**{fmt(e['spread_vec_partial'], 4)}** | {fmt(e['spread_u50_partial'], 4)} | "
            f"{fmt(e['spread_v50_partial'], 4)} | {fmt(e['approx_se'], 4)} |"
        )

    check = payload["predeclared_check"]
    partials = {k: round(v, 4) for k, v in check["spread_vec_partial_by_group"].items()}
    lines += [
        "",
        "편상관은 풍속(`nwp_ws10`)에 대해 2차 다항 잔차화한 뒤 계산했다. 스프레드와 오차가",
        "둘 다 풍속에 딸려 커지는 교란을 제거하기 위함이다.",
        "",
        "## 7. 사전확약 대조",
        "",
        f"동결된 기대: *{check['expectation']}. 부호가 반대면 C5(anomaly)로 라우팅된다.*",
        "",
        f"- 그룹별 vec 편상관: `{partials}`",
        f"- 전 그룹 양수: **{check['all_positive']}**",
        f"- 2xSE 를 넘는 그룹 존재: **{check['any_beyond_2se']}**",
        "",
        f"판정: **{check['verdict']}**",
        "",
        "부호가 유지되더라도 크기가 잡음대 안이면 신호로 쓸 수 없다. 임계 판정은 라우터 표의",
        "몫이며 이 노드는 하지 않는다.",
    ]

    write_node_artifacts(
        node=NODE,
        title="M271 N0/A4 — 층화 예보검증",
        report_lines=lines,
        payload=payload,
        input_hashes=input_hashes,
        parents=["A1_labels", "A3_spatial"],
        script_path=Path(__file__),
    )
    return payload


def main() -> int:
    tables, hashes = load_tables()
    payload = run(tables, hashes)
    check = payload["predeclared_check"]
    print(f"[A4] 표면 {payload['surface_rows']:,}행")
    for e in payload["spread_relation"]:
        ratio = abs(e["spread_vec_partial"]) / e["approx_se"]
        print(
            f"[A4] g{e['group']} vec편상관={e['spread_vec_partial']:+.4f} "
            f"(SE {e['approx_se']:.4f}, {ratio:.1f}x)"
        )
    print(f"[A4] 사전확약 판정: {check['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
