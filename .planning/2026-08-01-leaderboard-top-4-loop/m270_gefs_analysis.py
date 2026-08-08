"""M270 finding 2: does the gust-spread signal survive a larger sample?

The 60-day probe found the naive hypothesis dead (10 m wind-vector spread, partial r about
`+0.027`) and one narrower signal alive (surface gust spread, partial r about `+0.186` on
414 rows). This re-runs the same controlled analysis on the expanded sample.

The confound handled here is generation level: mean generation falls monotonically across
spread quartiles, and low generation has the highest band-hit rate, so an uncontrolled
correlation reads backwards. Both a within-level breakdown and a quadratic partial
correlation are reported.

Read-only. No model is fitted, no 2024 row is read.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from baram.constants import CAPACITIES_KWH

ROOT = Path(__file__).resolve().parents[2]
PROBE_DIR = ROOT / "artifacts" / "backtests" / "metric-aligned-probe"
REPORTS = ROOT / "reports"
STORE = PROBE_DIR / "M270_GEFS_SPREAD_PROBE.parquet"

FOLDS = ("dev-2023-Q2", "dev-2023-Q3", "dev-2023-Q4")
PARENT_PATHS = {fold: f"M269_PROBE_TOP100-{fold}-policies.parquet" for fold in FOLDS}
DEPLOYED = "T0.5_G1.5"
LEVEL_EDGES = (0.10, 0.25, 0.45, 0.70, 1.10)
SPREAD_COLUMNS = ("spread_vec", "spread_u10", "spread_v10", "spread_gust")
MIN_CELL = 25


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_parent() -> pd.DataFrame:
    parts = []
    for fold in FOLDS:
        frame = pd.read_parquet(PROBE_DIR / PARENT_PATHS[fold])
        part = frame.loc[:, ["forecast_kst_dtm", "group_id", "actual_kwh"]].copy()
        part["prediction_kwh"] = frame[DEPLOYED].to_numpy(dtype=float)
        parts.append(part)
    out = pd.concat(parts, ignore_index=True)
    out["forecast_kst_dtm"] = pd.to_datetime(out["forecast_kst_dtm"])
    capacity = out["group_id"].map(CAPACITIES_KWH).astype(float)
    out["abs_err"] = (out["prediction_kwh"] - out["actual_kwh"]).abs() / capacity
    out["y"] = out["actual_kwh"] / capacity
    out["hit"] = (out["abs_err"] <= 0.06).astype(int)
    return out.loc[out["y"] >= 0.10]


def partial_corr(x: np.ndarray, e: np.ndarray, z: np.ndarray) -> float:
    """Correlation of x and e after removing a quadratic fit on z from both."""
    rx = x - np.poly1d(np.polyfit(z, x, 2))(z)
    re = e - np.poly1d(np.polyfit(z, e, 2))(z)
    return float(np.corrcoef(rx, re)[0, 1])


def main() -> None:
    spread = pd.read_parquet(STORE)
    merged = load_parent().merge(spread, on="forecast_kst_dtm", how="inner")
    n = len(merged)
    se = 1.0 / np.sqrt(max(n - 3, 1))
    noise_band = 2.0 * se

    merged["ylev"] = pd.cut(merged["y"], list(LEVEL_EDGES), include_lowest=True)

    partials = {c: partial_corr(
        merged[c].to_numpy(), merged["abs_err"].to_numpy(), merged["y"].to_numpy()
    ) for c in SPREAD_COLUMNS}

    lines: list[str] = []
    lines.append("# M270 발견 2 — 확대 표본 스프레드 분석\n")
    lines.append(f"- 스프레드 행 {len(spread)} (사이클 {spread['cycle_date'].nunique()}개)")
    lines.append(f"- 결합된 유효 예측 행: **{n}**")
    lines.append(f"- 근사 SE = `{se:.4f}`, 잡음대 = `±{noise_band:.4f}`\n")

    lines.append("## 1. 발전량 통제 후 편상관 (2차 잔차화)\n")
    lines.append("| 스프레드 변수 | 편상관 r(|오차|) | SE 배수 | 판정 |")
    lines.append("|---|---:|---:|---|")
    for column, value in partials.items():
        verdict = "**실재**" if abs(value) > noise_band else "구분 불가"
        lines.append(f"| `{column}` | {value:+.4f} | {abs(value) / se:.1f} | {verdict} |")

    lines.append("\n## 2. 발전량 구간별 상관 (교란 직접 통제)\n")
    lines.append("| y 구간 | n | r(vec,\\|e\\|) | r(gust,\\|e\\|) | 적중률 | 평균 vec |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    per_level = {}
    for level, block in merged.groupby("ylev", observed=True):
        if len(block) < MIN_CELL:
            lines.append(f"| {level} | {len(block)} | — | — | — | — | (표본 부족) |")
            continue
        rv = block["spread_vec"].corr(block["abs_err"])
        rg = block["spread_gust"].corr(block["abs_err"])
        per_level[str(level)] = {"n": len(block), "r_vec": float(rv), "r_gust": float(rg)}
        lines.append(
            f"| {level} | {len(block)} | {rv:+.4f} | {rg:+.4f} | "
            f"{block['hit'].mean():.4f} | {block['spread_vec'].mean():.4f} |"
        )

    lines.append("\n## 3. 60일 표본 대비 안정성\n")
    lines.append("| 변수 | 60일 편상관 | 확대 편상관 | 변화 |")
    lines.append("|---|---:|---:|---:|")
    baseline = {
        "spread_vec": 0.0268, "spread_u10": 0.0232,
        "spread_v10": 0.0341, "spread_gust": 0.1860,
    }
    for column, before in baseline.items():
        after = partials[column]
        lines.append(f"| `{column}` | {before:+.4f} | {after:+.4f} | {after - before:+.4f} |")
    lines.append(
        "\n확대 표본에서 추정치가 크게 흔들리면 60일 결과는 표본 요행이었다는 뜻이고, "
        "유지되면 약하지만 실재하는 신호라는 뜻이다."
    )

    (REPORTS / "m270_gefs_expanded.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    receipt = {
        "schema_version": 1,
        "stage": "M270_GEFS_EXPANDED_ANALYSIS",
        "spread_store_sha256": sha256_file(STORE),
        "spread_rows": len(spread),
        "cycles": int(spread["cycle_date"].nunique()),
        "merged_rows": n,
        "standard_error": se,
        "noise_band": noise_band,
        "partial_correlations": partials,
        "per_level": per_level,
        "baseline_60day": baseline,
        "model_fits": 0,
        "new_2024_evaluation": False,
        "lockbox_reopened": False,
        "dacon_upload": False,
        "external_actions": [],
    }
    (REPORTS / "m270_gefs_expanded_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=1, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

    print(f"merged={n} se={se:.4f} noise_band=±{noise_band:.4f}")
    for column, value in partials.items():
        mark = "REAL" if abs(value) > noise_band else "null"
        print(f"  {column:<14} partial={value:+.4f}  ({abs(value) / se:.1f} SE)  {mark}"
              f"   [60d was {baseline[column]:+.4f}]")


if __name__ == "__main__":
    main()
