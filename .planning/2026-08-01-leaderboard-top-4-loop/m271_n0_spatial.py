"""M271 N0 자식 A3 — 격자·터빈 기하 구조 규명.

동결 사양(`m271_n0_method.SPECS['A3_spatial']`)의 4개 산출만 만든다. 기하 기술이며
사전확약 기대가 없다.

기존 `baram.features.spatial.haversine_km` 를 재사용한다. 거리 계산을 새로 쓰지 않는다.

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

from baram.features.spatial import haversine_km

NODE = "A3_spatial"
# 격자별 정적 필드. 시간에 따라 변하지 않으므로 첫 시각의 값을 쓴다.
LDAPS_STATIC = ("surface_0_h", "surface_0_lsm")


def grid_geometry(weather: pd.DataFrame, source: str) -> pd.DataFrame:
    geo = (
        weather.loc[:, ["grid_id", "latitude", "longitude"]]
        .drop_duplicates("grid_id")
        .sort_values("grid_id")
        .reset_index(drop=True)
    )
    geo["source"] = source
    return geo


def static_fields(weather: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    present = [c for c in columns if c in weather.columns]
    if not present:
        return pd.DataFrame({"grid_id": []})
    first_ts = weather["forecast_kst_dtm"].min()
    snapshot = weather.loc[weather["forecast_kst_dtm"] == first_ts, ["grid_id", *present]]
    varies = {
        c: float(weather.groupby("grid_id")[c].nunique().max()) for c in present
    }
    snapshot = snapshot.drop_duplicates("grid_id").sort_values("grid_id").reset_index(drop=True)
    snapshot.attrs["distinct_values_per_grid"] = varies
    return snapshot


def distance_matrix(turbines: pd.DataFrame, geo: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, turbine in turbines.iterrows():
        distances = haversine_km(
            float(turbine["latitude"]),
            float(turbine["longitude"]),
            geo["latitude"].to_numpy(dtype=float),
            geo["longitude"].to_numpy(dtype=float),
        )
        distances = np.asarray(distances, dtype=float)
        order = np.argsort(distances)
        rows.append(
            {
                "turbine_id": turbine["turbine_id"],
                "group_id": int(turbine["group_id"]),
                "nearest_grid_id": int(geo.iloc[order[0]]["grid_id"]),
                "nearest_km": float(distances[order[0]]),
                "second_km": float(distances[order[1]]) if len(order) > 1 else float("nan"),
                "mean_km": float(distances.mean()),
                "max_km": float(distances.max()),
            }
        )
    return pd.DataFrame(rows)


def group_layout(turbines: pd.DataFrame) -> list[dict[str, Any]]:
    out = []
    for group, part in turbines.groupby("group_id"):
        lat, lon = part["latitude"].to_numpy(), part["longitude"].to_numpy()
        spans = []
        for i in range(len(part)):
            for j in range(i + 1, len(part)):
                spans.append(float(haversine_km(lat[i], lon[i], lat[j], lon[j])))
        out.append(
            {
                "group": int(group),
                "turbines": len(part),
                "sites": sorted(part["site_name"].unique().tolist()),
                "models": sorted(part["model"].unique().tolist()),
                "hub_height_m": sorted(part["hub_height_m"].unique().tolist()),
                "rotor_diameter_m": sorted(part["rotor_diameter_m"].unique().tolist()),
                "group_capacity_mw": float(part["group_capacity_mw"].iloc[0]),
                "lat_range": [float(lat.min()), float(lat.max())],
                "lon_range": [float(lon.min()), float(lon.max())],
                "max_pair_km": float(max(spans)) if spans else 0.0,
                "mean_pair_km": float(np.mean(spans)) if spans else 0.0,
            }
        )
    return out


def run(tables: Any, input_hashes: dict[str, str]) -> dict[str, Any]:
    turbines = tables.turbines
    geo_gfs = grid_geometry(tables.gfs_train, "gfs")
    geo_ldaps = grid_geometry(tables.ldaps_train, "ldaps")
    static = static_fields(tables.ldaps_train, LDAPS_STATIC)

    dist_gfs = distance_matrix(turbines, geo_gfs)
    dist_ldaps = distance_matrix(turbines, geo_ldaps)

    payload: dict[str, Any] = {
        "turbines": turbines.drop(columns=["coordinate_dms"]).to_dict("records"),
        "grid_counts": {"gfs": len(geo_gfs), "ldaps": len(geo_ldaps)},
        "grid_geometry": {
            "gfs": geo_gfs.to_dict("records"),
            "ldaps": geo_ldaps.to_dict("records"),
        },
        "ldaps_static_fields": {
            "columns": list(static.columns),
            "rows": static.to_dict("records"),
            "distinct_values_per_grid": static.attrs.get("distinct_values_per_grid", {}),
        },
        "distance_gfs": dist_gfs.to_dict("records"),
        "distance_ldaps": dist_ldaps.to_dict("records"),
        "group_layout": group_layout(turbines),
        "grid_spacing_km": {
            "gfs_min_pair": _min_pair_km(geo_gfs),
            "ldaps_min_pair": _min_pair_km(geo_ldaps),
        },
    }

    layout = payload["group_layout"]
    lines = [
        "## 1. 그룹별 배치",
        "",
        "| 그룹 | 터빈 | 단지 | 모델 | 허브고(m) | 로터경(m) | 설비(MW) | 최대 터빈간 거리 |",
        "|---:|---:|---|---|---|---|---:|---:|",
    ]
    for g in layout:
        lines.append(
            f"| {g['group']} | {g['turbines']} | {', '.join(g['sites'])} | "
            f"{', '.join(g['models'])} | {g['hub_height_m']} | {g['rotor_diameter_m']} | "
            f"{g['group_capacity_mw']:.1f} | {fmt(g['max_pair_km'], 2)} km |"
        )

    lines += [
        "",
        "| 그룹 | 위도 범위 | 경도 범위 | 평균 터빈간 거리 |",
        "|---:|---|---|---:|",
    ]
    for g in layout:
        lines.append(
            f"| {g['group']} | {g['lat_range'][0]:.4f} ~ {g['lat_range'][1]:.4f} | "
            f"{g['lon_range'][0]:.4f} ~ {g['lon_range'][1]:.4f} | {fmt(g['mean_pair_km'], 2)} km |"
        )

    spacing = payload["grid_spacing_km"]
    lines += [
        "",
        "## 2. 격자 해상도 대조",
        "",
        f"- GFS 격자 {payload['grid_counts']['gfs']}개, 최소 격자간 거리 "
        f"**{fmt(spacing['gfs_min_pair'], 2)} km**",
        f"- LDAPS 격자 {payload['grid_counts']['ldaps']}개, 최소 격자간 거리 "
        f"**{fmt(spacing['ldaps_min_pair'], 2)} km**",
        "",
        "격자 간격을 터빈 배치 규모와 견주어 읽는다. 격자 간격이 단지 크기보다 크면 한 격자가",
        "여러 터빈을 덮으므로 공간 보간이 실질적으로 구분력을 갖지 못한다.",
        "",
        "## 3. 터빈-격자 거리",
        "",
        "| 그룹 | 소스 | 최근접 거리 최소~최대 | 2번째 격자까지 | 격자 커버리지 |",
        "|---:|---|---|---|---|",
    ]
    for source, frame in (("GFS", dist_gfs), ("LDAPS", dist_ldaps)):
        for group, part in frame.groupby("group_id"):
            unique_nearest = sorted(part["nearest_grid_id"].unique().tolist())
            lines.append(
                f"| {group} | {source} | {fmt(part['nearest_km'].min(), 2)} ~ "
                f"{fmt(part['nearest_km'].max(), 2)} km | "
                f"{fmt(part['second_km'].min(), 2)} ~ {fmt(part['second_km'].max(), 2)} km | "
                f"격자 {unique_nearest} |"
            )

    static_rows = payload["ldaps_static_fields"]["rows"]
    varies = payload["ldaps_static_fields"]["distinct_values_per_grid"]
    lines += [
        "",
        "## 4. LDAPS 정적 필드 (지형고도·육해 마스크)",
        "",
        f"격자별 고유값 개수: `{varies}` (1 이면 시간에 대해 불변)",
        "",
        "| 격자 | " + " | ".join(c for c in static.columns if c != "grid_id") + " |",
        "|---:|" + "---:|" * (len(static.columns) - 1),
    ]
    for row in static_rows:
        values = " | ".join(
            f"{row[c]:.2f}" if isinstance(row[c], int | float) else str(row[c])
            for c in static.columns
            if c != "grid_id"
        )
        lines.append(f"| {int(row['grid_id'])} | {values} |")

    lines += [
        "",
        "`surface_0_lsm` 은 육지/해양 마스크다. 값이 1 이면 육지, 0 이면 해양이며 중간값은",
        "혼합 격자를 뜻한다. `surface_0_h` 는 격자 대표 고도이며 터빈 허브고와 함께 읽어야",
        "풍속 외삽의 기준 높이를 정할 수 있다.",
        "",
        "## 5. 사전확약 대조",
        "",
        "이 노드는 기하 기술이므로 사전확약 기대가 없다. 매핑과 거리 행렬이 산출되었으므로",
        "중단 조건을 충족한다.",
    ]

    write_node_artifacts(
        node=NODE,
        title="M271 N0/A3 — 격자·터빈 기하",
        report_lines=lines,
        payload=payload,
        input_hashes=input_hashes,
        parents=[],
        script_path=Path(__file__),
    )
    return payload


def _min_pair_km(geo: pd.DataFrame) -> float:
    lat, lon = geo["latitude"].to_numpy(), geo["longitude"].to_numpy()
    best = float("inf")
    for i in range(len(geo)):
        for j in range(i + 1, len(geo)):
            best = min(best, float(haversine_km(lat[i], lon[i], lat[j], lon[j])))
    return best


def main() -> int:
    tables, hashes = load_tables()
    payload = run(tables, hashes)
    s = payload["grid_spacing_km"]
    print(f"[A3] 격자 GFS={payload['grid_counts']['gfs']} (간격 {s['gfs_min_pair']:.2f}km) "
          f"LDAPS={payload['grid_counts']['ldaps']} (간격 {s['ldaps_min_pair']:.2f}km)")
    for g in payload["group_layout"]:
        print(f"[A3] g{g['group']}: 터빈{g['turbines']} {g['sites']} 허브{g['hub_height_m']}m "
              f"최대터빈간 {g['max_pair_km']:.2f}km")
    return 0


if __name__ == "__main__":
    sys.exit(main())
