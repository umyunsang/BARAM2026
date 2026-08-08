"""M270 finding 2, step 1: does GEFS ensemble spread predict this model's error?

VALUE OF INFORMATION
Collecting the full development period costs roughly 7 GB and 45-90 blocking minutes.
The literature supports a spread-skill relationship in general, but whether it holds for
THIS data, THIS lead, and THESE sites is a separate question. This probe samples about 60
days, joins spread to the champion's own errors, and measures the relationship. A null
result closes the lane for minutes instead of hours.

AVAILABILITY-TIME COMPLIANCE
The prediction reference time is 13:00 KST (04:00 UTC) daily. Only the PREVIOUS DAY'S 18Z
cycle is used, issued 18:00 UTC on D-1, about 10 hours before the reference instant. The
issuance cycle is part of the S3 key (`gefs.YYYYMMDD/18/`), so availability is provable
from the path rather than asserted.

Read-only public HTTPS. No model is fitted, no 2024 row is read, no submission is built.
"""

from __future__ import annotations

import json
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROBE_DIR = ROOT / "artifacts" / "backtests" / "metric-aligned-probe"
REPORTS = ROOT / "reports"

BUCKET = "https://noaa-gefs-pds.s3.amazonaws.com"
PRODUCT = "atmos/pgrb2sp25"
CYCLE = "18"
FOLDS = ("dev-2023-Q2", "dev-2023-Q3", "dev-2023-Q4")
PARENT_PATHS = {fold: f"M269_PROBE_TOP100-{fold}-policies.parquet" for fold in FOLDS}
DEPLOYED = "T0.5_G1.5"

# Leads from the 18Z D-1 cycle that land inside the 01:00-24:00 KST D+1 target window.
# 18Z + 24h = 18:00 UTC D = 03:00 KST D+1, and so on every 6 hours.
LEADS = (24, 30, 36, 42)
WANTED = ("UGRD:10 m above ground", "VGRD:10 m above ground", "GUST:surface")

# The supplied competition grid: 3x3 at 0.25 degree spacing on the Korean east coast.
GRID_POINTS = [(lat, lon) for lat in (37.50, 37.25, 37.00) for lon in (128.75, 129.00, 129.25)]

SAMPLE_DAYS = 60
TIMEOUT = 60


def _get(url: str, byte_range: tuple[int, int] | None = None) -> bytes:
    request = urllib.request.Request(url)
    if byte_range is not None:
        request.add_header("Range", f"bytes={byte_range[0]}-{byte_range[1]}")
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        return response.read()


def index_entries(date: str, lead: int) -> list[tuple[int, int, str]]:
    """Parse the .idx sidecar into (start, end, description) byte ranges."""
    stem = f"{BUCKET}/gefs.{date}/{CYCLE}/{PRODUCT}/gespr.t{CYCLE}z.pgrb2s.0p25.f{lead:03d}"
    text = _get(stem + ".idx").decode("utf-8", errors="replace")
    rows = []
    for line in text.strip().split("\n"):
        parts = line.split(":")
        if len(parts) < 5:
            continue
        rows.append((int(parts[1]), ":".join(parts[3:5]), line))
    out = []
    for i, (offset, description, _) in enumerate(rows):
        end = rows[i + 1][0] - 1 if i + 1 < len(rows) else -1
        out.append((offset, end, description))
    return out


def spread_at_sites(date: str, lead: int) -> dict[str, float] | None:
    """Mean ensemble standard deviation over the nine supplied grid points."""
    import eccodes

    stem = f"{BUCKET}/gefs.{date}/{CYCLE}/{PRODUCT}/gespr.t{CYCLE}z.pgrb2s.0p25.f{lead:03d}"
    try:
        entries = index_entries(date, lead)
    except Exception:
        return None
    values: dict[str, float] = {}
    for offset, end, description in entries:
        if description not in WANTED:
            continue
        try:
            raw = _get(stem, (offset, end if end > 0 else offset + 4_000_000))
            handle = eccodes.codes_new_from_message(raw)
            try:
                point_values = [
                    eccodes.codes_grib_find_nearest(handle, lat, lon)[0]["value"]
                    for lat, lon in GRID_POINTS
                ]
            finally:
                eccodes.codes_release(handle)
        except Exception:
            return None
        values[description.split(":")[0]] = float(np.mean(point_values))
    return values if len(values) == len(WANTED) else None


def load_parent() -> pd.DataFrame:
    parts = []
    for fold in FOLDS:
        frame = pd.read_parquet(PROBE_DIR / PARENT_PATHS[fold])
        part = frame.loc[:, ["forecast_kst_dtm", "group_id", "actual_kwh"]].copy()
        part["prediction_kwh"] = frame[DEPLOYED].to_numpy(dtype=float)
        parts.append(part)
    out = pd.concat(parts, ignore_index=True)
    out["forecast_kst_dtm"] = pd.to_datetime(out["forecast_kst_dtm"])
    from baram.constants import CAPACITIES_KWH

    capacity = out["group_id"].map(CAPACITIES_KWH).astype(float)
    out["abs_err"] = ((out["prediction_kwh"] - out["actual_kwh"]).abs() / capacity)
    out["y"] = out["actual_kwh"] / capacity
    out["hit"] = (out["abs_err"] <= 0.06).astype(int)
    return out.loc[out["y"] >= 0.10]


def main() -> None:
    parent = load_parent()
    first = parent["forecast_kst_dtm"].min().normalize()
    last = parent["forecast_kst_dtm"].max().normalize()
    span = (last - first).days
    sample = [
        (first + timedelta(days=round(i * span / (SAMPLE_DAYS - 1)))).to_pydatetime()
        for i in range(SAMPLE_DAYS)
    ]

    records = []
    failures = 0
    for target_day in sample:
        # Targets on day T come from the 18Z cycle of T-2 (reference 13:00 KST on T-1).
        cycle_date = (target_day - timedelta(days=2)).strftime("%Y%m%d")
        for lead in LEADS:
            valid_utc = datetime.strptime(cycle_date, "%Y%m%d") + timedelta(
                hours=18 + lead
            )
            valid_kst = valid_utc + timedelta(hours=9)
            values = spread_at_sites(cycle_date, lead)
            if values is None:
                failures += 1
                continue
            records.append(
                {
                    "cycle_date": cycle_date,
                    "lead": lead,
                    "forecast_kst_dtm": pd.Timestamp(valid_kst),
                    "spread_u10": values["UGRD"],
                    "spread_v10": values["VGRD"],
                    "spread_gust": values["GUST"],
                }
            )
    spread = pd.DataFrame(records)
    if spread.empty:
        raise RuntimeError("no GEFS spread rows were retrieved")
    spread["spread_vec"] = np.hypot(spread["spread_u10"], spread["spread_v10"])

    merged = parent.merge(spread, on="forecast_kst_dtm", how="inner")

    lines: list[str] = []
    lines.append("# M270 발견 2 — GEFS 앙상블 스프레드 상관 탐침\n")
    lines.append(
        f"- 표본 일수 {SAMPLE_DAYS}, 스프레드 행 {len(spread)}, 실패 {failures}"
    )
    lines.append(f"- 사용 사이클: 전일 18Z, 리드 {list(LEADS)}")
    lines.append(f"- 결합된 예측 행: **{len(merged)}** (유효 발전 행만)\n")

    if merged.empty:
        lines.append("결합 결과가 비어 있어 상관을 계산할 수 없다.")
    else:
        lines.append("## 스프레드와 오차의 관계\n")
        lines.append("| 스프레드 변수 | corr(|오차|) | Spearman | corr(적중) |")
        lines.append("|---|---:|---:|---:|")
        for column in ("spread_vec", "spread_u10", "spread_v10", "spread_gust"):
            pear = merged[column].corr(merged["abs_err"])
            spear = merged[column].corr(merged["abs_err"], method="spearman")
            hit = merged[column].corr(merged["hit"])
            lines.append(f"| `{column}` | {pear:+.4f} | {spear:+.4f} | {hit:+.4f} |")

        lines.append("\n## 스프레드 사분위별 실제 성능\n")
        merged["spread_q"] = pd.qcut(
            merged["spread_vec"], 4, labels=["Q1저", "Q2", "Q3", "Q4고"]
        )
        lines.append("| 스프레드 사분위 | 행수 | 평균\\|오차\\| | 4단위 적중률 | 평균 y |")
        lines.append("|---|---:|---:|---:|---:|")
        for label, block in merged.groupby("spread_q", observed=True):
            lines.append(
                f"| {label} | {len(block)} | {block['abs_err'].mean():.4f} | "
                f"{block['hit'].mean():.4f} | {block['y'].mean():.4f} |"
            )

    (REPORTS / "m270_gefs_probe.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    receipt = {
        "schema_version": 1,
        "stage": "M270_GEFS_SPREAD_CORRELATION_PROBE",
        "bucket": BUCKET,
        "product": PRODUCT,
        "cycle": f"previous-day {CYCLE}Z",
        "leads": list(LEADS),
        "variables": list(WANTED),
        "grid_points": GRID_POINTS,
        "sample_days_requested": SAMPLE_DAYS,
        "spread_rows": len(spread),
        "fetch_failures": failures,
        "merged_rows": len(merged),
        "model_fits": 0,
        "new_2024_evaluation": False,
        "lockbox_reopened": False,
        "dacon_upload": False,
        "external_actions": ["read-only anonymous HTTPS GET against NOAA public S3"],
    }
    (REPORTS / "m270_gefs_probe_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=1, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    spread.to_parquet(PROBE_DIR / "M270_GEFS_SPREAD_PROBE.parquet", index=False)

    print(f"spread rows={len(spread)} failures={failures} merged={len(merged)}")
    if not merged.empty:
        for column in ("spread_vec", "spread_gust"):
            print(
                f"  {column}: pearson(|err|)={merged[column].corr(merged['abs_err']):+.4f} "
                f"spearman={merged[column].corr(merged['abs_err'], method='spearman'):+.4f} "
                f"corr(hit)={merged[column].corr(merged['hit']):+.4f}"
            )


if __name__ == "__main__":
    main()
