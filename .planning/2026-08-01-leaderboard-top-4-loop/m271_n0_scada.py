"""M271 N0 자식 A5 — SCADA 경험 파워커브와 나셀풍속-NWP 사상.

동결 사양(`m271_n0_method.SPECS['A5_scada']`)의 4개 산출만 만든다.

방법 리서치(①) 결과: IEC 61400-12-1 의 **풍속 bin 방법과 공기밀도 정규화**를 채택하되
태그는 `near_match_only` 다. 표준은 기상탑을 갖춘 통제된 단일터빈 계측 프로토콜인데
우리는 로터 뒤 나셀 풍속계, 10분 데이터, **운전로그 없음**이다. 따라서 표준의 로그 기반
필터링(가용성·출력제한·고장 제외)을 통계적 이상치 제거로 **대체**하며, 이는 표준 준수가
아니고 그 사실을 리포트에 명시한다.

제약: SCADA 는 진단 전용이다. 평가기간에 존재하지 않으므로 추론 피처를 만들지 않는다.

읽기 전용. 모델을 적합하지 않고 2024 행을 읽지 않는다.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from m271_n0_common import fmt, load_tables, write_node_artifacts

NODE = "A5_scada"
BIN_WIDTH = 0.5  # IEC 61400-12-1 의 풍속 bin 폭
BIN_EDGES = np.arange(0.0, 30.0 + BIN_WIDTH, BIN_WIDTH)
RHO_REFERENCE = 1.225  # kg/m^3, IEC 기준 공기밀도
R_SPECIFIC = 287.05  # J/(kg*K), 건조공기 기체상수
SIGMA_K = 3.0  # 통계적 이상치 제거의 3-sigma 계수
IQR_K = 1.5  # 사분위 기준 계수
MIN_BIN_ROWS = 30  # bin 대표값을 신뢰하기 위한 최소 표본


def scada_long(vestas: pd.DataFrame, unison: pd.DataFrame) -> pd.DataFrame:
    """10분 SCADA 를 (시각, 터빈, power, ws, wd) 롱 포맷으로 편다."""
    parts = []
    for prefix, frame, count in (("vestas", vestas, 12), ("unison", unison, 5)):
        for n in range(1, count + 1):
            tag = f"{prefix}_wtg{n:02d}"
            cols = {f"{tag}_power_kw10m": "power_kw", f"{tag}_ws": "ws", f"{tag}_wd": "wd"}
            missing = [c for c in cols if c not in frame.columns]
            if missing:
                raise KeyError(f"SCADA missing columns: {missing}")
            part = frame.loc[:, ["kst_dtm", *cols]].rename(columns=cols)
            part["scada_key"] = tag
            part["manufacturer"] = prefix.upper()
            part["turbine_number"] = n
            parts.append(part)
    out = pd.concat(parts, ignore_index=True)
    return out.dropna(subset=["ws", "power_kw"])


def air_density(ldaps: pd.DataFrame) -> pd.DataFrame:
    """LDAPS 격자평균 기압·기온으로 시간별 공기밀도를 만든다.

    격자 간 밀도 차이는 무시한다. A3 가 보인 대로 단지 전체가 2 km 안에 들어가고 고도
    편차도 130 m 수준이라, 밀도 정규화 목적에는 격자평균이면 충분하다.
    """
    need = ["surface_0_sp", "heightAboveGround_2_t"]
    missing = [c for c in need if c not in ldaps.columns]
    if missing:
        raise KeyError(f"LDAPS missing columns for density: {missing}")
    hourly = (
        ldaps.loc[:, ["forecast_kst_dtm", *need]]
        .groupby("forecast_kst_dtm", as_index=False)
        .mean()
    )
    temp = hourly["heightAboveGround_2_t"]
    # 켈빈이 아니면 섭씨로 보고 변환한다.
    if float(temp.median()) < 100.0:
        temp = temp + 273.15
    hourly["rho"] = hourly["surface_0_sp"] / (R_SPECIFIC * temp)
    return hourly.loc[:, ["forecast_kst_dtm", "rho"]]


def remove_outliers(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """bin 안에서 3-sigma 와 사분위 기준을 함께 적용한다.

    운전로그가 없으므로 출력제한·정지·고장 구간을 로그로 배제할 수 없다. 표준의 로그 기반
    필터링을 이 통계적 규칙으로 대체하며, 표준 준수가 아님을 명시한다.
    """
    kept = []
    before = len(frame)
    for _, part in frame.groupby("ws_bin", observed=True):
        if len(part) < 3:
            continue
        power = part["power_kw"]
        mu, sigma = power.mean(), power.std()
        q1, q3 = power.quantile(0.25), power.quantile(0.75)
        iqr = q3 - q1
        mask = (power - mu).abs() <= SIGMA_K * sigma
        mask &= (power >= q1 - IQR_K * iqr) & (power <= q3 + IQR_K * iqr)
        kept.append(part.loc[mask])
    result = pd.concat(kept, ignore_index=True) if kept else frame.iloc[0:0]
    return result, {
        "rows_before": before,
        "rows_after": len(result),
        "removed": before - len(result),
        "removed_fraction": float((before - len(result)) / before) if before else 0.0,
    }


def power_curve(frame: pd.DataFrame, speed_col: str) -> pd.DataFrame:
    bins = pd.cut(frame[speed_col], bins=BIN_EDGES, right=False)
    grouped = frame.assign(_bin=bins).groupby("_bin", observed=True)["power_kw"]
    curve = grouped.agg(["size", "mean", "std"]).reset_index()
    curve = curve.loc[curve["size"] >= MIN_BIN_ROWS]
    curve["bin_center"] = curve["_bin"].apply(lambda b: float(b.left) + BIN_WIDTH / 2.0)
    return curve.loc[:, ["bin_center", "size", "mean", "std"]].reset_index(drop=True)


def nacelle_vs_nwp(scada: pd.DataFrame, ldaps: pd.DataFrame, turbines: pd.DataFrame) -> list[dict]:
    """그룹별 나셀풍속 시간평균과 NWP 풍속의 사상 관계."""
    speed = np.hypot(
        ldaps["heightAboveGround_10_10u"].to_numpy(dtype=float),
        ldaps["heightAboveGround_10_10v"].to_numpy(dtype=float),
    )
    nwp = (
        pd.DataFrame({"forecast_kst_dtm": ldaps["forecast_kst_dtm"], "nwp_ws10": speed})
        .groupby("forecast_kst_dtm", as_index=False)
        .mean()
    )
    key = turbines.loc[:, ["manufacturer", "turbine_number", "group_id"]]
    joined = scada.merge(key, on=["manufacturer", "turbine_number"], how="inner")
    joined["hour"] = joined["kst_dtm"].dt.floor("h")
    hourly = (
        joined.groupby(["group_id", "hour"], as_index=False)["ws"].mean().rename(
            columns={"ws": "nacelle_ws"}
        )
    )
    merged = hourly.merge(nwp, left_on="hour", right_on="forecast_kst_dtm", how="inner")
    out = []
    for group, part in merged.groupby("group_id"):
        if len(part) < 100:
            continue
        slope, intercept = np.polyfit(part["nwp_ws10"], part["nacelle_ws"], 1)
        out.append(
            {
                "group": int(group),
                "rows": len(part),
                "pearson": float(part["nwp_ws10"].corr(part["nacelle_ws"])),
                "slope_nacelle_per_nwp": float(slope),
                "intercept": float(intercept),
                "mean_nacelle_ws": float(part["nacelle_ws"].mean()),
                "mean_nwp_ws10": float(part["nwp_ws10"].mean()),
                "residual_std": float(
                    (part["nacelle_ws"] - (slope * part["nwp_ws10"] + intercept)).std()
                ),
            }
        )
    return out


def run(tables: Any, input_hashes: dict[str, str]) -> dict[str, Any]:
    scada = scada_long(tables.scada_vestas, tables.scada_unison)
    density = air_density(tables.ldaps_train)

    scada["hour"] = scada["kst_dtm"].dt.floor("h")
    scada = scada.merge(density, left_on="hour", right_on="forecast_kst_dtm", how="left")
    scada["ws_normalised"] = scada["ws"] * (scada["rho"] / RHO_REFERENCE) ** (1.0 / 3.0)
    scada["ws_bin"] = pd.cut(scada["ws"], bins=BIN_EDGES, right=False)

    cleaned, removal = remove_outliers(scada)

    curves: list[dict[str, Any]] = []
    for key, part in cleaned.groupby("scada_key"):
        raw = power_curve(part, "ws")
        norm = power_curve(part.dropna(subset=["ws_normalised"]), "ws_normalised")
        merged = raw.merge(norm, on="bin_center", how="inner", suffixes=("_raw", "_norm"))
        peak = float(raw["mean"].max()) if len(raw) else float("nan")
        shift = (
            float((merged["mean_norm"] - merged["mean_raw"]).abs().mean())
            if len(merged)
            else float("nan")
        )
        # 정격 상한: VESTAS 3.6 MW -> 600, UNISON 4.2 MW(21.0/5) -> 700 kWh/10분
        rated = 600.0 if key.startswith("vestas") else 700.0
        curves.append(
            {
                "scada_key": key,
                "rows": len(part),
                "bins": len(merged),
                "max_bin_mean_kwh_per_10min": peak,
                "rated_kwh_per_10min": rated,
                "peak_over_rated": peak / rated,
                "mean_abs_shift_kwh_per_10min": shift,
                "curve": merged.to_dict("records"),
            }
        )

    mapping = nacelle_vs_nwp(cleaned, tables.ldaps_train, tables.turbines)

    payload: dict[str, Any] = {
        "scada_rows_long": len(scada),
        "turbines_covered": sorted(scada["scada_key"].unique().tolist()),
        "density": {
            "rho_min": float(density["rho"].min()),
            "rho_median": float(density["rho"].median()),
            "rho_max": float(density["rho"].max()),
            "reference": RHO_REFERENCE,
        },
        "outlier_removal": removal,
        "curves": curves,
        "nacelle_vs_nwp": mapping,
        "standard_deviation_note": (
            "운전로그가 없어 IEC 61400-12-1 의 로그 기반 필터링을 3-sigma + 사분위 규칙으로 "
            "대체했다. 표준 준수가 아니다."
        ),
    }

    d = payload["density"]
    lines = [
        "## 1. 표본과 공기밀도",
        "",
        f"- SCADA 롱 포맷 행: **{payload['scada_rows_long']:,}** (터빈 "
        f"{len(payload['turbines_covered'])}기, 10분 간격)",
        f"- 공기밀도 rho: {fmt(d['rho_min'], 4)} ~ {fmt(d['rho_max'], 4)} kg/m³ "
        f"(중앙값 {fmt(d['rho_median'], 4)}, IEC 기준 {d['reference']})",
        "",
        "밀도는 LDAPS 격자평균 기압·기온에서 `rho = p / (R * T)` 로 구했다. A3 가 보인 대로",
        "단지 전체가 2 km 안에 들어가므로 격자 간 밀도 차이는 무시했다.",
        "",
        "IEC 정규화: `v_n = v * (rho / 1.225)^(1/3)`",
        "",
        "## 2. 이상치 제거 (표준 대체)",
        "",
        f"- 제거 전 {removal['rows_before']:,} 행 → 제거 후 **{removal['rows_after']:,}** 행",
        f"- 제거율 **{removal['removed_fraction']:.2%}** ({removal['removed']:,} 행)",
        "",
        payload["standard_deviation_note"],
        "",
        "## 3. 터빈별 파워커브",
        "",
        f"풍속 bin 폭 {BIN_WIDTH} m/s, bin 당 최소 {MIN_BIN_ROWS} 행.",
        "",
        "**단위 주의.** `*_power_kw10m` 은 순시 kW 가 아니라 **10분당 kWh** 다. 명세서가 "
        "'10분 단위 power 값, 원천 컬럼명 기준 단위 kW10m' 이라고 기술한다. 정격 대비 포화값이",
        "이를 확인해 준다 — VESTAS 3.6 MW 는 600 kWh/10분, UNISON 4.2 MW(21.0/5)는 700 "
        "kWh/10분이 상한이다. kW 로 보려면 6 을 곱한다.",
        "",
        "| 터빈 | 정제행 | bin | 최대 bin 평균 | 정격 | 포화비 | 밀도정규화 이동 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for c in curves:
        lines.append(
            f"| `{c['scada_key']}` | {c['rows']:,} | {c['bins']} | "
            f"{fmt(c['max_bin_mean_kwh_per_10min'], 1)} | "
            f"{c['rated_kwh_per_10min']:.0f} | {fmt(c['peak_over_rated'], 3)} | "
            f"{fmt(c['mean_abs_shift_kwh_per_10min'], 2)} |"
        )

    lines += [
        "",
        "밀도정규화 평균이동이 작다는 것은 이 사이트에서 밀도 보정이 파워커브를 크게 바꾸지",
        "않는다는 뜻이고, 크다면 밀도가 실질적 공변량이라는 뜻이다.",
        "",
        "## 4. 나셀풍속 대 NWP 풍속",
        "",
        "| 그룹 | 시간행 | Pearson | 기울기 | 절편 | 평균 나셀 | 평균 NWP 10m | 잔차 표준편차 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for m in mapping:
        lines.append(
            f"| {m['group']} | {m['rows']:,} | {fmt(m['pearson'], 4)} | "
            f"{fmt(m['slope_nacelle_per_nwp'], 4)} | {fmt(m['intercept'], 3)} | "
            f"{fmt(m['mean_nacelle_ws'], 3)} | {fmt(m['mean_nwp_ws10'], 3)} | "
            f"**{fmt(m['residual_std'], 3)}** |"
        )

    lines += [
        "",
        "나셀 풍속계는 로터 뒤에 있어 실제 유입풍속이 아니고, 10분 값을 시간으로 평균했다.",
        "따라서 잔차 표준편차는 예보 오차의 **상한**이지 예보 오차 자체가 아니다.",
        "",
        "## 5. 사전확약 대조",
        "",
        "동결된 기대: *운전로그가 없으므로 표준의 로그 기반 필터링을 통계적 대체로 수행한다.",
        "이 대체는 표준 준수가 아니며 그 사실을 리포트에 명시한다.*",
        "",
        "→ 유지. 2절에 대체 사실과 제거율을 기록했다.",
        "",
        "## 6. 제약",
        "",
        "SCADA 는 평가기간에 존재하지 않는다. 이 노드의 산출은 **진단 전용**이며 추론 피처를",
        "만들지 않는다. 학습 시 보조 신호로 쓸 수 있는지는 별개 판단이고 여기서 하지 않는다.",
    ]

    write_node_artifacts(
        node=NODE,
        title="M271 N0/A5 — SCADA 파워커브와 나셀-NWP 사상",
        report_lines=lines,
        payload=payload,
        input_hashes=input_hashes,
        parents=[],
        script_path=Path(__file__),
    )
    return payload


def main() -> int:
    tables, hashes = load_tables()
    payload = run(tables, hashes)
    r = payload["outlier_removal"]
    d = payload["density"]
    print(f"[A5] SCADA {payload['scada_rows_long']:,}행, 터빈 {len(payload['turbines_covered'])}기")
    print(f"[A5] rho {d['rho_min']:.4f}~{d['rho_max']:.4f} (기준 {d['reference']})")
    print(f"[A5] 이상치 제거율 {r['removed_fraction']:.2%} ({r['removed']:,}행)")
    for m in payload["nacelle_vs_nwp"]:
        print(
            f"[A5] g{m['group']} 나셀~NWP r={m['pearson']:.4f} "
            f"기울기={m['slope_nacelle_per_nwp']:.4f} 잔차sd={m['residual_std']:.3f} m/s"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
