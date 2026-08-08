"""M271 P4 사이클 5 — GFS 공간 보간이 신호인가 오염인가 (C13 구조적 결과).

A3 가 측정한 구조적 사실: 17 기 전 터빈의 최근접 GFS 격자가 **하나**(격자 5)이고 두 번째
격자는 18~20 km 밖이다. 격자 간격 22 km 인데 단지 전체가 2 km 안에 들어간다.

`configs/features/spatial_v2.yaml` 은 `inverse_distance_power: 2` 로 GFS 9 격자를 IDW
보간한다. A3 의 실제 거리로 가중치를 계산하면 최근접이 약 74%, 18~20 km 밖 8 개가 약 26%
를 차지한다. 1,000 m 산악 능선에서 20 km 떨어진 격자는 다른 기상이다.

이 노드는 **보간이 신호를 넣는지 오염을 넣는지** 를 직접 잰다. 설정에 `nearest` 모드가 이미
있으므로 결론이 나오면 설정 변경만으로 적용된다.

사전확약(실행 전 동결):
  H1  GFS 는 nearest 가 IDW 보다 나셀 풍속과 **더 강하게** 상관한다 (보간이 오염)
  H2  LDAPS 는 그 반대이거나 차이가 없다 (격자 간격 1.5 km 이므로 보간이 정당)
  H1 이 기각되면 보간이 정당한 것이므로 이 축은 C8 로 닫는다.

비교 대상은 **나셀 풍속**이다. 발전량으로 비교하면 파워커브 비선형과 가용성이 섞이고,
사이클 2 가 확인한 collider 문제도 생긴다. 풍속 대 풍속이 가장 깨끗한 비교다.

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

from m271_cycle1_toprate import turbine_hourly
from m271_n0_common import load_tables

from baram.features.spatial import haversine_km

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
REPORT_MD = REPORTS / "m271_cycle5_spatial.md"
RECEIPT = REPORTS / "m271_cycle5_spatial_receipt.json"

NODE_ID = "C1N5_SPATIAL_INTERPOLATION"
LANE = "L2"
IDW_POWER = 2  # configs/features/spatial_v2.yaml 과 동일
DISTANCE_FLOOR_KM = 0.1  # 동일


def group_weights(turbines: pd.DataFrame, geo: pd.DataFrame) -> dict[int, dict[str, Any]]:
    """그룹별 IDW / nearest 가중치. 프로젝트 설정과 같은 규칙을 쓴다."""
    out: dict[int, dict[str, Any]] = {}
    for group, part in turbines.groupby("group_id"):
        distances = []
        for _, turbine in part.iterrows():
            d = np.asarray(
                haversine_km(
                    float(turbine["latitude"]), float(turbine["longitude"]),
                    geo["latitude"].to_numpy(dtype=float),
                    geo["longitude"].to_numpy(dtype=float),
                ),
                dtype=float,
            )
            distances.append(d)
        mean_distance = np.mean(distances, axis=0)
        raw = 1.0 / np.maximum(mean_distance, DISTANCE_FLOOR_KM) ** IDW_POWER
        idw = raw / raw.sum()
        nearest = np.zeros_like(idw)
        nearest[int(np.argmin(mean_distance))] = 1.0
        out[int(group)] = {
            "grid_ids": geo["grid_id"].to_numpy(),
            "distance_km": mean_distance,
            "idw": idw,
            "nearest": nearest,
            "nearest_grid": int(geo["grid_id"].to_numpy()[int(np.argmin(mean_distance))]),
            "idw_weight_on_nearest": float(idw[int(np.argmin(mean_distance))]),
            "idw_weight_beyond_10km": float(idw[mean_distance > 10.0].sum()),
        }
    return out


def weighted_speed(weather: pd.DataFrame, weights: np.ndarray, grid_ids: np.ndarray,
                   u_col: str, v_col: str) -> pd.DataFrame:
    """격자 가중 평균으로 벡터 합성한 뒤 속력을 낸다."""
    pivot_u = weather.pivot_table(
        index="forecast_kst_dtm", columns="grid_id", values=u_col, observed=True
    ).reindex(columns=grid_ids)
    pivot_v = weather.pivot_table(
        index="forecast_kst_dtm", columns="grid_id", values=v_col, observed=True
    ).reindex(columns=grid_ids)
    u = pivot_u.to_numpy(dtype=float) @ weights
    v = pivot_v.to_numpy(dtype=float) @ weights
    return pd.DataFrame({"forecast_kst_dtm": pivot_u.index, "speed": np.hypot(u, v)})


def main() -> int:
    tables, input_hashes = load_tables()

    geo_gfs = (
        tables.gfs_train.loc[:, ["grid_id", "latitude", "longitude"]]
        .drop_duplicates("grid_id").sort_values("grid_id").reset_index(drop=True)
    )
    geo_ldaps = (
        tables.ldaps_train.loc[:, ["grid_id", "latitude", "longitude"]]
        .drop_duplicates("grid_id").sort_values("grid_id").reset_index(drop=True)
    )
    w_gfs = group_weights(tables.turbines, geo_gfs)
    w_ldaps = group_weights(tables.turbines, geo_ldaps)

    nacelle = (
        turbine_hourly(tables)
        .groupby(["group_id", "hour"], as_index=False)["ws"]
        .mean()
        .rename(columns={"ws": "nacelle_ws"})
    )

    rows: list[dict[str, Any]] = []
    for group in (1, 2, 3):
        target = nacelle.loc[nacelle["group_id"] == group]
        for source, weather, weights, u_col, v_col in (
            ("gfs", tables.gfs_train, w_gfs[group],
             "heightAboveGround_10_10u", "heightAboveGround_10_10v"),
            ("ldaps", tables.ldaps_train, w_ldaps[group],
             "heightAboveGround_10_10u", "heightAboveGround_10_10v"),
        ):
            entry: dict[str, Any] = {
                "group": group,
                "source": source,
                "nearest_grid": weights["nearest_grid"],
                "idw_weight_on_nearest": weights["idw_weight_on_nearest"],
                "idw_weight_beyond_10km": weights["idw_weight_beyond_10km"],
            }
            for mode in ("idw", "nearest"):
                speed = weighted_speed(
                    weather, weights[mode], weights["grid_ids"], u_col, v_col
                )
                merged = target.merge(
                    speed, left_on="hour", right_on="forecast_kst_dtm", how="inner"
                ).dropna()
                entry[f"{mode}_rows"] = len(merged)
                entry[f"{mode}_pearson"] = float(
                    merged["speed"].corr(merged["nacelle_ws"])
                )
                # 그룹별 선형 보정 후 잔차 표준편차. 크기 비교에 공정하다.
                slope, intercept = np.polyfit(merged["speed"], merged["nacelle_ws"], 1)
                entry[f"{mode}_residual_std"] = float(
                    (merged["nacelle_ws"] - (slope * merged["speed"] + intercept)).std()
                )
            entry["nearest_minus_idw_pearson"] = (
                entry["nearest_pearson"] - entry["idw_pearson"]
            )
            entry["nearest_minus_idw_residual"] = (
                entry["nearest_residual_std"] - entry["idw_residual_std"]
            )
            rows.append(entry)

    gfs_rows = [r for r in rows if r["source"] == "gfs"]
    ldaps_rows = [r for r in rows if r["source"] == "ldaps"]
    h1 = all(r["nearest_minus_idw_pearson"] > 0 for r in gfs_rows)
    h2 = all(r["nearest_minus_idw_pearson"] <= 0 for r in ldaps_rows)

    check = {
        "H1_expectation": "GFS 는 nearest 가 IDW 보다 나셀 풍속과 더 강하게 상관",
        "H1_held": bool(h1),
        "H1_deltas": [r["nearest_minus_idw_pearson"] for r in gfs_rows],
        "H2_expectation": "LDAPS 는 그 반대이거나 차이 없음",
        "H2_held": bool(h2),
        "H2_deltas": [r["nearest_minus_idw_pearson"] for r in ldaps_rows],
        "verdict": (
            "GFS_INTERPOLATION_IS_CONTAMINATION" if h1 else "GFS_INTERPOLATION_JUSTIFIED"
        ),
    }
    payload = {"comparisons": rows, "predeclared_check": check, "input_hashes": input_hashes}

    lines = [
        "# M271 P4 사이클 5 — GFS 공간 보간은 신호인가 오염인가",
        "",
        f"- 판정일: {datetime.now(UTC).strftime('%Y-%m-%d')} (UTC)",
        f"- 노드: `{NODE_ID}` / 레인 {LANE} / 라우팅 근거: C13 구조적 결과 (A3)",
        "",
        "## 1. 구조",
        "",
        "A3 측정: 17 기 전 터빈의 최근접 GFS 격자가 하나이고 두 번째는 18~20 km 밖이다.",
        "`spatial_v2.yaml` 은 그 9 격자를 `inverse_distance_power: 2` 로 보간한다.",
        "",
        "| 그룹 | 소스 | 최근접 격자 | 최근접 IDW 가중 | **10km 밖 IDW 가중** |",
        "|---:|---|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['group']} | {r['source'].upper()} | {r['nearest_grid']} | "
            f"{r['idw_weight_on_nearest']:.1%} | **{r['idw_weight_beyond_10km']:.1%}** |"
        )

    lines += [
        "",
        "## 2. 나셀 풍속과의 비교",
        "",
        "발전량이 아니라 **풍속 대 풍속**으로 비교한다. 발전량으로 하면 파워커브 비선형과",
        "가용성이 섞이고 사이클 2 가 확인한 collider 문제도 생긴다.",
        "",
        "| 그룹 | 소스 | IDW r | nearest r | **차이** | IDW 잔차sd | nearest 잔차sd | 차이 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['group']} | {r['source'].upper()} | {r['idw_pearson']:.4f} | "
            f"{r['nearest_pearson']:.4f} | **{r['nearest_minus_idw_pearson']:+.4f}** | "
            f"{r['idw_residual_std']:.3f} | {r['nearest_residual_std']:.3f} | "
            f"{r['nearest_minus_idw_residual']:+.3f} |"
        )

    lines += [
        "",
        "## 3. 사전확약 대조",
        "",
        f"- H1 `{check['H1_expectation']}` -> **{check['H1_held']}** "
        f"(`{[round(v, 4) for v in check['H1_deltas']]}`)",
        f"- H2 `{check['H2_expectation']}` -> **{check['H2_held']}** "
        f"(`{[round(v, 4) for v in check['H2_deltas']]}`)",
        "",
        f"판정: **{check['verdict']}**",
        "",
        "## 4. 읽는 법",
        "",
        "H1 이 성립하면 GFS 보간이 오염이므로 `spatial_v2.yaml` 의 GFS 모드를 `nearest` 로",
        "바꾸는 것만으로 조건부 예보의 입력 품질이 오른다. 다만 **여기서 잰 것은 풍속 상관**",
        "이지 공식 점수가 아니다. 점수 효과는 동결 월별 게이트 아래 별도 실험으로 확인해야",
        "하며 이 노드는 그것을 하지 않는다.",
        "",
        "H1 이 기각되면 20 km 밖 격자가 실제로 정보를 준다는 뜻이고, 그 경우 보간은 정당하다.",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    receipt = {
        "schema_version": 1,
        "stage": "M271_P4_CYCLE5_SPATIAL",
        "node": NODE_ID,
        "lane": LANE,
        "routed_by": "C13_structural_consequence",
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

    for r in rows:
        print(
            f"[C5] g{r['group']} {r['source']:5s} 10km밖가중={r['idw_weight_beyond_10km']:.1%} "
            f"IDW r={r['idw_pearson']:.4f} nearest r={r['nearest_pearson']:.4f} "
            f"차이={r['nearest_minus_idw_pearson']:+.4f}"
        )
    print(f"[C5] H1(GFS nearest 우세)={check['H1_held']}  H2(LDAPS 반대)={check['H2_held']}")
    print(f"[C5] 판정: {check['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
