"""M271 P4 사이클 6 — LDAPS 50 m 중점 바람이라는 피처 갭.

`src/baram/features/weather.py:_LDAPS_VECTOR_PAIRS` 는 `wind50max` 와 `wind50min` 을 각각
**별도 벡터**로 처리해 `_speed / _dir_sin / _dir_cos` 를 만든다. 중점은 만들지 않는다.

두 문제가 겹친다.

  1. 현재 피처가 물리적으로 이상하다. `wind50max_speed = |(Umax, Vmax)|` 는 "최대 U 성분과
     최대 V 성분으로 만든 벡터의 크기" 인데, 둘이 같은 시점·지점에서 일어난다는 보장이 없다.
     실재하지 않는 바람이다.
  2. 의미 있는 조합이 빠져 있다. 중점 `((Umax+Umin)/2, (Vmax+Vmin)/2)` 는 50 m 평균 바람의
     최선 추정이고, 차이 `(Umax-Umin, Vmax-Vmin)` 은 변동폭이다. A3 가 측정한 허브고는
     117 m 이므로 LDAPS 가 주는 높이 중 50 m 가 가장 가깝다(10 m 보다 훨씬).

트리는 `(a+b)/2` 를 잘 표현하지 못한다. 두 피처의 평균을 근사하려면 분할이 아주 많이 든다.
명시적으로 만들어 주면 공짜로 얻는 구조다.

사전확약(실행 전 동결):
  H1  중점 속력이 나셀 풍속과 `wind10_speed` 보다 **더 강하게** 상관한다
  H2  중점 속력이 `wind50max_speed` / `wind50min_speed` 각각보다 **더 강하게** 상관한다
  H1 과 H2 가 모두 성립해야 피처 갭이 실재한다. 하나라도 기각되면 C8 로 닫는다.

비교는 **나셀 풍속** 대상이다. 발전량으로 하면 파워커브 비선형과 가용성이 섞이고 사이클 2 가
확인한 collider 문제가 생긴다.

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

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
REPORT_MD = REPORTS / "m271_cycle6_wind50mid.md"
RECEIPT = REPORTS / "m271_cycle6_wind50mid_receipt.json"

NODE_ID = "C1N6_WIND50_MIDPOINT"
LANE = "L2"

UMAX, UMIN = "heightAboveGround_50_50MUmax", "heightAboveGround_50_50MUmin"
VMAX, VMIN = "heightAboveGround_50_50MVmax", "heightAboveGround_50_50MVmin"
U10, V10 = "heightAboveGround_10_10u", "heightAboveGround_10_10v"


def candidate_features(ldaps: pd.DataFrame) -> pd.DataFrame:
    """격자평균 후 후보 속력들을 만든다. 기존 피처와 제안 피처를 나란히 둔다."""
    cols = [UMAX, UMIN, VMAX, VMIN, U10, V10]
    hourly = (
        ldaps.loc[:, ["forecast_kst_dtm", *cols]]
        .groupby("forecast_kst_dtm", as_index=False)
        .mean()
    )
    umax, umin = hourly[UMAX].to_numpy(float), hourly[UMIN].to_numpy(float)
    vmax, vmin = hourly[VMAX].to_numpy(float), hourly[VMIN].to_numpy(float)

    out = pd.DataFrame({"forecast_kst_dtm": hourly["forecast_kst_dtm"]})
    # 기존 파이프라인이 만드는 것
    out["wind10_speed"] = np.hypot(hourly[U10].to_numpy(float), hourly[V10].to_numpy(float))
    out["wind50max_speed"] = np.hypot(umax, vmax)
    out["wind50min_speed"] = np.hypot(umin, vmin)
    # 제안: 중점(50 m 평균 바람 추정)과 변동폭
    out["wind50mid_speed"] = np.hypot((umax + umin) / 2.0, (vmax + vmin) / 2.0)
    out["wind50_range"] = np.hypot(umax - umin, vmax - vmin)
    # 대조: 기존 두 속력의 단순 평균. 트리가 근사하려 할 법한 형태.
    out["wind50_speed_avg"] = 0.5 * (out["wind50max_speed"] + out["wind50min_speed"])
    return out


def main() -> int:
    tables, input_hashes = load_tables()
    features = candidate_features(tables.ldaps_train)
    nacelle = (
        turbine_hourly(tables)
        .groupby(["group_id", "hour"], as_index=False)["ws"]
        .mean()
        .rename(columns={"ws": "nacelle_ws"})
    )

    candidates = [
        "wind10_speed",
        "wind50max_speed",
        "wind50min_speed",
        "wind50_speed_avg",
        "wind50mid_speed",
        "wind50_range",
    ]
    rows: list[dict[str, Any]] = []
    for group in (1, 2, 3):
        target = nacelle.loc[nacelle["group_id"] == group]
        merged = target.merge(
            features, left_on="hour", right_on="forecast_kst_dtm", how="inner"
        ).dropna()
        entry: dict[str, Any] = {"group": group, "rows": len(merged)}
        for name in candidates:
            entry[f"{name}__r"] = float(merged[name].corr(merged["nacelle_ws"]))
            slope, intercept = np.polyfit(merged[name], merged["nacelle_ws"], 1)
            entry[f"{name}__residual_std"] = float(
                (merged["nacelle_ws"] - (slope * merged[name] + intercept)).std()
            )
        rows.append(entry)

    mid = [r["wind50mid_speed__r"] for r in rows]
    h1 = all(
        r["wind50mid_speed__r"] > r["wind10_speed__r"] for r in rows
    )
    h2 = all(
        r["wind50mid_speed__r"] > max(r["wind50max_speed__r"], r["wind50min_speed__r"])
        for r in rows
    )
    check = {
        "H1_expectation": "중점 속력이 wind10_speed 보다 나셀과 더 강하게 상관",
        "H1_held": bool(h1),
        "H2_expectation": "중점 속력이 wind50max / wind50min 각각보다 더 강하게 상관",
        "H2_held": bool(h2),
        "midpoint_r": mid,
        "verdict": "FEATURE_GAP_CONFIRMED" if (h1 and h2) else "FEATURE_GAP_REJECTED",
    }
    payload = {"comparisons": rows, "predeclared_check": check, "input_hashes": input_hashes}

    lines = [
        "# M271 P4 사이클 6 — LDAPS 50 m 중점 바람 피처 갭",
        "",
        f"- 판정일: {datetime.now(UTC).strftime('%Y-%m-%d')} (UTC)",
        f"- 노드: `{NODE_ID}` / 레인 {LANE}",
        "",
        "## 1. 갭",
        "",
        "`weather.py:_LDAPS_VECTOR_PAIRS` 는 `wind50max` 와 `wind50min` 을 각각 별도 벡터로",
        "처리하고 중점을 만들지 않는다.",
        "",
        "- `wind50max_speed = |(Umax, Vmax)|` 는 최대 U 와 최대 V 로 만든 벡터의 크기다.",
        "  둘이 같은 시점·지점에서 일어난다는 보장이 없으므로 실재하지 않는 바람이다.",
        "- 중점 `((Umax+Umin)/2, (Vmax+Vmin)/2)` 는 50 m 평균 바람의 최선 추정이다.",
        "- A3 측정 허브고는 117 m 이므로 LDAPS 가 주는 높이 중 50 m 가 가장 가깝다.",
        "",
        "트리는 두 피처의 평균을 잘 표현하지 못하므로 명시 피처의 값이 있다.",
        "",
        "## 2. 나셀 풍속과의 상관",
        "",
        "| 그룹 | 행수 | `wind10` | `wind50max` | `wind50min` | 두 속력 평균 | "
        "**`wind50mid`** | `wind50_range` |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['group']} | {r['rows']:,} | {r['wind10_speed__r']:.4f} | "
            f"{r['wind50max_speed__r']:.4f} | {r['wind50min_speed__r']:.4f} | "
            f"{r['wind50_speed_avg__r']:.4f} | **{r['wind50mid_speed__r']:.4f}** | "
            f"{r['wind50_range__r']:.4f} |"
        )

    lines += [
        "",
        "선형 보정 후 잔차 표준편차 (작을수록 좋다):",
        "",
        "| 그룹 | `wind10` | `wind50max` | `wind50min` | **`wind50mid`** |",
        "|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['group']} | {r['wind10_speed__residual_std']:.3f} | "
            f"{r['wind50max_speed__residual_std']:.3f} | "
            f"{r['wind50min_speed__residual_std']:.3f} | "
            f"**{r['wind50mid_speed__residual_std']:.3f}** |"
        )

    lines += [
        "",
        "## 3. 사전확약 대조",
        "",
        f"- H1 `{check['H1_expectation']}` -> **{check['H1_held']}**",
        f"- H2 `{check['H2_expectation']}` -> **{check['H2_held']}**",
        "",
        f"판정: **{check['verdict']}**",
        "",
        "## 4. 읽는 법",
        "",
        "여기서 잰 것은 **나셀 풍속과의 상관**이지 공식 점수가 아니다. 상관이 좋아도 점수가",
        "오른다는 보장은 없다 — 모델이 이미 다른 경로로 같은 정보를 얻고 있을 수 있다.",
        "",
        "판정이 `FEATURE_GAP_CONFIRMED` 이면 다음 단계는 이 피처를 파이프라인에 넣고 동결",
        "월별 게이트 아래 실제 점수 효과를 재는 것이다. 그것은 모델 적합이 들어가는 별도",
        "노드이며 이 노드는 하지 않는다.",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    receipt = {
        "schema_version": 1,
        "stage": "M271_P4_CYCLE6_WIND50_MIDPOINT",
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

    for r in rows:
        print(
            f"[C6] g{r['group']} wind10={r['wind10_speed__r']:.4f} "
            f"50max={r['wind50max_speed__r']:.4f} 50min={r['wind50min_speed__r']:.4f} "
            f"**50mid={r['wind50mid_speed__r']:.4f}** "
            f"(잔차 {r['wind50mid_speed__residual_std']:.3f} vs "
            f"{r['wind10_speed__residual_std']:.3f})"
        )
    print(f"[C6] H1={check['H1_held']} H2={check['H2_held']} -> {check['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
