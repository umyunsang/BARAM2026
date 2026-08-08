"""M271 N0 자식 A6 — 시간 규약 감사.

공급 데이터의 가용시각과 공식 규칙의 예측기준시점이 일치하는지, 그리고 규칙상 허용되지만
현재 쓰이지 않는 정보창이 있는지 측정한다.

공식 규칙(2026-08-03 17:00 FAQ 추가분):
  각 예측 대상일의 발전량은 해당 일자의 **전일 14:00 KST** 를 예측기준시점으로 한다.
  같은 대상일의 24개 시간대는 모두 같은 기준시점을 공유한다.
  예: 2025-01-13 00:00~23:00 을 예측하면 모든 행의 기준시점은 2025-01-12 14:00.

공급 데이터 명세서 §6:
  매일 09:00 KST 초기화 예보에서 다음날 01:00 ~ 그다음날 00:00 의 24시간만 추출.
  각 예보자료는 해당일 **13:00 KST** 부터 사용 가능한 것으로 간주.

두 규약의 하루 경계가 한 시간 어긋난다. 공급은 01:00~익일 00:00 을 한 묶음으로 보고,
공식은 00:00~23:00 을 대상일로 본다. 그 결과가 무엇인지 측정으로 확정한다.

프로젝트 내부 규약도 함께 감사한다. `baram.data.canonical.add_operating_period` 는
`operating_day = (t - 1h).normalize()` 로 정의하므로 공식 대상일과 다르다.

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

NODE = "A6_timing"
SPEC_AVAILABLE_HOUR = 13  # 명세서가 기술한 가용시각
OFFICIAL_REFERENCE_HOUR = 14  # 공식 규칙의 예측기준시점 (대상일 전일)


def official_reference(ts: pd.Series) -> pd.Series:
    """공식 예측기준시점 = (대상일 - 1일) + 14:00 KST."""
    offset = np.timedelta64(OFFICIAL_REFERENCE_HOUR - 24, "h")
    return ts.dt.normalize() + offset


def audit_frame(frame: pd.DataFrame, label: str) -> dict[str, Any]:
    """한 예보 소스의 시간 규약을 감사한다. 격자 중복은 제거한다."""
    keys = (
        frame.loc[:, ["forecast_kst_dtm", "data_available_kst_dtm", "lead_hour"]]
        .drop_duplicates("forecast_kst_dtm")
        .sort_values("forecast_kst_dtm")
        .reset_index(drop=True)
    )
    avail = keys["data_available_kst_dtm"]
    ref = official_reference(keys["forecast_kst_dtm"])
    slack_h = (ref - avail).dt.total_seconds() / 3600.0
    hour = keys["forecast_kst_dtm"].dt.hour

    by_hour = (
        pd.DataFrame({"hour": hour, "slack_h": slack_h, "lead_hour": keys["lead_hour"]})
        .groupby("hour")
        .agg(
            rows=("slack_h", "size"),
            slack_min=("slack_h", "min"),
            slack_max=("slack_h", "max"),
            lead_min=("lead_hour", "min"),
            lead_max=("lead_hour", "max"),
        )
        .reset_index()
    )

    return {
        "source": label,
        "rows_unique_forecast_ts": len(keys),
        "period": [str(keys["forecast_kst_dtm"].min()), str(keys["forecast_kst_dtm"].max())],
        "available_hour_values": sorted(avail.dt.hour.unique().tolist()),
        "available_hour_matches_spec": bool((avail.dt.hour == SPEC_AVAILABLE_HOUR).all()),
        "lead_hour_min": int(keys["lead_hour"].min()),
        "lead_hour_max": int(keys["lead_hour"].max()),
        "slack_h_min": float(slack_h.min()),
        "slack_h_max": float(slack_h.max()),
        "slack_h_unique": sorted(np.unique(np.round(slack_h.to_numpy(), 6)).tolist()),
        "slack_negative_rows": int((slack_h < 0).sum()),
        "by_hour": by_hour.to_dict("records"),
    }


def audit_operating_day(frame: pd.DataFrame) -> dict[str, Any]:
    """프로젝트 내부 operating_day 와 공식 대상일이 어긋나는 행을 센다."""
    keys = frame.loc[:, ["forecast_kst_dtm", "operating_day"]].drop_duplicates("forecast_kst_dtm")
    official_day = keys["forecast_kst_dtm"].dt.normalize()
    mismatch = official_day != keys["operating_day"]
    mismatch_hours = sorted(keys.loc[mismatch, "forecast_kst_dtm"].dt.hour.unique().tolist())
    return {
        "rows": len(keys),
        "mismatched_rows": int(mismatch.sum()),
        "mismatch_fraction": float(mismatch.mean()),
        "mismatched_hours": mismatch_hours,
        "note": (
            "operating_day = (t - 1h).normalize() 이므로 00:00 행만 공식 대상일보다 하루 "
            "이르다. 24분의 1 이 어긋나는 것이 정상이며, 문제는 어긋남 자체가 아니라 "
            "어긋난 행의 기준시점을 어느 쪽으로 잡느냐다."
        ),
    }


def run(tables: Any, input_hashes: dict[str, str]) -> dict[str, Any]:
    audits = {
        "gfs_train": audit_frame(tables.gfs_train, "gfs_train"),
        "ldaps_train": audit_frame(tables.ldaps_train, "ldaps_train"),
        "gfs_test": audit_frame(tables.gfs_test, "gfs_test"),
        "ldaps_test": audit_frame(tables.ldaps_test, "ldaps_test"),
    }
    op_day = audit_operating_day(tables.gfs_test)

    all_slack_min = min(a["slack_h_min"] for a in audits.values())
    any_negative = any(a["slack_negative_rows"] for a in audits.values())
    spec_ok = all(a["available_hour_matches_spec"] for a in audits.values())

    payload: dict[str, Any] = {
        "audits": audits,
        "operating_day_audit": op_day,
        "verdict": {
            "available_hour_matches_spec_all_sources": spec_ok,
            "min_slack_hours_any_row": all_slack_min,
            "any_row_violates_official_rule": any_negative,
            "predeclared_expectation_held": bool(spec_ok and not any_negative),
        },
    }

    a = audits["gfs_test"]
    lines = [
        "## 1. 판정",
        "",
        f"- 명세서 가용시각 13:00 KST 와 실측 일치: **{spec_ok}**",
        f"- 공식 기준시점을 위반하는 행: **{'있음' if any_negative else '없음'}**",
        f"- 모든 소스·모든 행의 최소 여유: **{fmt(all_slack_min, 2)} 시간**",
        "",
        "여유 = (공식 예측기준시점) - (공급 가용시각). 양수면 공급 데이터가 규칙보다",
        "보수적이라는 뜻이고, 그만큼이 규칙상 허용되지만 쓰이지 않는 정보창이다.",
        "",
        "## 2. 소스별 감사",
        "",
        "| 소스 | 고유 예보시각 | 기간 | 가용시각 | 리드(h) | 여유(h) 최소~최대 |",
        "|---|---:|---|---|---|---|",
    ]
    for name, aud in audits.items():
        lines.append(
            f"| `{name}` | {aud['rows_unique_forecast_ts']:,} | "
            f"{aud['period'][0][:10]} ~ {aud['period'][1][:10]} | "
            f"{aud['available_hour_values']} | {aud['lead_hour_min']}~{aud['lead_hour_max']} | "
            f"{fmt(aud['slack_h_min'], 1)} ~ {fmt(aud['slack_h_max'], 1)} |"
        )

    lines += [
        "",
        f"관측된 여유 값 집합 (`gfs_test`): `{a['slack_h_unique']}`",
        "",
        "## 3. 시각별 구조 (`gfs_test`)",
        "",
        "| 예보시각 | 행수 | 여유(h) | 리드(h) |",
        "|---:|---:|---:|---:|",
    ]
    for row in a["by_hour"]:
        slack = (
            fmt(row["slack_min"], 1)
            if row["slack_min"] == row["slack_max"]
            else f"{fmt(row['slack_min'], 1)}~{fmt(row['slack_max'], 1)}"
        )
        lead = (
            str(row["lead_min"])
            if row["lead_min"] == row["lead_max"]
            else f"{row['lead_min']}~{row['lead_max']}"
        )
        lines.append(f"| {row['hour']:02d}:00 | {row['rows']:,} | {slack} | {lead} |")

    lines += [
        "",
        "## 4. 프로젝트 내부 규약 대조",
        "",
        f"- `operating_day` 가 공식 대상일과 다른 행: **{op_day['mismatched_rows']:,}** "
        f"({op_day['mismatch_fraction']:.1%})",
        f"- 어긋나는 시각: `{op_day['mismatched_hours']}`",
        "",
        op_day["note"],
        "",
        "## 5. 사전확약 대조",
        "",
        "동결된 기대: *모든 행이 13:00 가용이고 공식 기준시점은 14:00 이므로 최소 1시간 "
        "여유가 예상된다. 00:00 행은 대상일이 하루 뒤라 여유가 25시간으로 예상된다.*",
        "",
        f"→ 기대 유지: **{payload['verdict']['predeclared_expectation_held']}**",
        "",
        "## 6. 읽는 법",
        "",
        "여유가 양수라는 것은 공급 데이터가 규칙보다 이르게 확정된다는 뜻이지, 그 창을 채울",
        "다른 데이터가 존재한다는 뜻이 아니다. 실제로 쓸 수 있는지는 별개 문제이며 이 노드는",
        "판단하지 않는다. 그 판단은 증거가 라우터를 통과한 뒤 방향 리서치가 맡는다.",
    ]

    write_node_artifacts(
        node=NODE,
        title="M271 N0/A6 — 시간 규약 감사",
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
    v = payload["verdict"]
    print(f"[A6] 명세일치={v['available_hour_matches_spec_all_sources']} "
          f"최소여유={v['min_slack_hours_any_row']}h 위반={v['any_row_violates_official_rule']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
