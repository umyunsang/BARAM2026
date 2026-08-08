"""M270 round 3c: how much does year-to-year difficulty move the score at fixed skill?

WHY
Twenty-plus teams exceed Total 0.66 while this project's local development sits near 0.628
and the wind-accuracy wall appears to bind everyone. One untested explanation is that the
2025 evaluation period is structurally easier than the 2023 development period. The 2025
labels are unobservable, so difficulty there cannot be measured - but the SPREAD across the
two observable years can be, and that is exactly the quantity R7 needs: how much a
2023-derived estimate can move when carried to another year.

TWO MODEL-FREE DIFFICULTY AXES
  * composition - where the generation mass sits on the power curve. Because the hit rate
    is U-shaped, an identical model scores differently on different mass distributions.
  * input - sigma_v measured against SCADA, per year.

Holding the 2023-measured per-band settlement ratios fixed and applying them to each year's
composition isolates the composition effect alone.

Read-only. SCADA diagnostic only. No model is fitted, no 2024 row is read.
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PLAN_DIR = Path(__file__).resolve().parent
PROBE_DIR = ROOT / "artifacts" / "backtests" / "metric-aligned-probe"
REPORTS = ROOT / "reports"
ARCHIVE = Path("/Users/um-yunsang/Downloads/open.zip")
sys.path.insert(0, str(PLAN_DIR))

FOLDS = ("dev-2023-Q2", "dev-2023-Q3", "dev-2023-Q4")
PARENT_PATHS = {fold: f"M269_PROBE_TOP100-{fold}-policies.parquet" for fold in FOLDS}
DEPLOYED = "T0.5_G1.5"
GROUP_TURBINES = {
    1: [f"vestas_wtg{i:02d}_ws" for i in range(1, 7)],
    2: [f"vestas_wtg{i:02d}_ws" for i in range(7, 13)],
    3: [f"unison_wtg{i:02d}_ws" for i in range(1, 6)],
}
GFS_COL = "gfs_spatial__idw__wind100_speed"
LDAPS_COL = "ldaps_spatial__idw__wind50max_speed"
LEVEL_EDGES = (0.10, 0.25, 0.45, 0.70, 1.10)


def load_scada() -> pd.DataFrame:
    frames = []
    with zipfile.ZipFile(ARCHIVE) as archive:
        for name in ("train/scada_unison_train.csv", "train/scada_vestas_train.csv"):
            with archive.open(name) as handle:
                frames.append(pd.read_csv(handle))
    merged = frames[0].merge(frames[1], on="kst_dtm", how="outer")
    merged["kst_dtm"] = pd.to_datetime(merged["kst_dtm"])
    rows = []
    for group, columns in GROUP_TURBINES.items():
        present = [c for c in columns if c in merged.columns]
        part = pd.DataFrame(
            {
                "forecast_kst_dtm": merged["kst_dtm"],
                "group_id": group,
                "scada_ws": merged[present].mean(axis=1, skipna=True),
            }
        )
        rows.append(part.dropna(subset=["scada_ws"]))
    return pd.concat(rows, ignore_index=True)


def band_settlement_2023() -> pd.Series:
    """Generation-weighted settlement ratio per level, measured on the 2023 dev folds."""
    from baram.constants import CAPACITIES_KWH

    parts = []
    for fold in FOLDS:
        frame = pd.read_parquet(PROBE_DIR / PARENT_PATHS[fold])
        part = frame.loc[:, ["forecast_kst_dtm", "group_id", "actual_kwh"]].copy()
        part["prediction_kwh"] = frame[DEPLOYED].to_numpy(dtype=float)
        parts.append(part)
    out = pd.concat(parts, ignore_index=True)
    capacity = out["group_id"].map(CAPACITIES_KWH).astype(float)
    out["y"] = out["actual_kwh"] / capacity
    err = (out["prediction_kwh"] - out["actual_kwh"]).abs() / capacity
    out["units"] = np.select([err <= 0.06, err <= 0.08], [4.0, 3.0], 0.0)
    out = out.loc[out["y"] >= 0.10]
    out["level"] = pd.cut(out["y"], list(LEVEL_EDGES), include_lowest=True)
    grouped = out.groupby("level", observed=True)
    return grouped.apply(
        lambda b: float((b["y"] * b["units"]).sum() / (b["y"] * 4.0).sum()), include_groups=False
    )


def main() -> None:
    from strict_dev_surface import development_surface

    from baram.constants import CAPACITIES_KWH

    surface, _ = development_surface()
    frame = surface.loc[
        :, ["forecast_kst_dtm", "group_id", "actual_kwh", GFS_COL, LDAPS_COL]
    ].copy()
    frame["forecast_kst_dtm"] = pd.to_datetime(frame["forecast_kst_dtm"])
    frame["year"] = frame["forecast_kst_dtm"].dt.year
    capacity = frame["group_id"].map(CAPACITIES_KWH).astype(float)
    frame["y"] = frame["actual_kwh"] / capacity
    frame = frame.dropna(subset=["y"])
    eligible = frame.loc[frame["y"] >= 0.10].copy()
    eligible["level"] = pd.cut(eligible["y"], list(LEVEL_EDGES), include_lowest=True)

    settlement = band_settlement_2023()

    # Composition-implied FICR: hold the 2023 per-band settlement ratio fixed and reweight
    # by each year's generation mass. Any difference is pure composition.
    rows = []
    for year, part in eligible.groupby("year", sort=True):
        mass = part.groupby("level", observed=True)["y"].sum()
        share = mass / mass.sum()
        implied = float((share * settlement.reindex(share.index)).sum())
        rows.append(
            {
                "year": int(year),
                "eligible_rows": len(part),
                "eligible_rate": float((frame["year"] == year).pipe(
                    lambda m: (frame.loc[m, "y"] >= 0.10).mean()
                )),
                "mean_y": float(part["y"].mean()),
                "implied_ficr": implied,
                **{f"share_{k!s}": float(v) for k, v in share.items()},
            }
        )
    year_table = pd.DataFrame(rows)

    # Input difficulty: sigma_v per year, per group, after per-group-year calibration.
    scada = load_scada()
    joined = frame.merge(scada, on=["forecast_kst_dtm", "group_id"], how="inner")
    joined = joined.loc[joined["scada_ws"].between(0.5, 30.0)].dropna(
        subset=[GFS_COL, LDAPS_COL]
    )
    sigma_rows = []
    for (year, group), part in joined.groupby(["year", "group_id"], sort=True):
        if len(part) < 500:
            continue
        truth = part["scada_ws"].to_numpy(dtype=float)
        best = np.vstack([part[GFS_COL], part[LDAPS_COL]]).mean(axis=0).astype(float)
        slope, intercept = np.polyfit(best, truth, 1)
        sigma_rows.append(
            {
                "year": int(year),
                "group_id": int(group),
                "n": len(part),
                "sigma_v": float((best * slope + intercept - truth).std()),
            }
        )
    sigma_table = pd.DataFrame(sigma_rows)

    lines: list[str] = []
    lines.append("# M270 — 연도 간 난이도 편차\n")
    lines.append("- 2025 라벨은 관측 불가하므로 **관측 가능한 두 해의 편차**를 측정한다")
    lines.append("- 실력 고정: 2023에서 측정한 구간별 정산비를 각 해 구성에 적용\n")

    lines.append("## 1. 구성 난이도 (실력 고정 시 함의 FICR)\n")
    lines.append("| year | 유효행 | 유효율 | 평균 y | **함의 FICR** |")
    lines.append("|---|---:|---:|---:|---:|")
    for r in year_table.itertuples(index=False):
        lines.append(
            f"| {r.year} | {r.eligible_rows} | {r.eligible_rate:.3f} | "
            f"{r.mean_y:.4f} | **{r.implied_ficr:.4f}** |"
        )
    if len(year_table) == 2:
        delta = float(year_table["implied_ficr"].iloc[1] - year_table["implied_ficr"].iloc[0])
        lines.append(
            f"\n두 해 차이: **{delta:+.4f} FICR** = Total 기준 **{delta / 2:+.4f}** "
            "(FICR는 Total에 0.5 가중)"
        )

    lines.append("\n## 2. 발전량 구간 질량 구성\n")
    share_cols = [c for c in year_table.columns if c.startswith("share_")]
    lines.append("| year | " + " | ".join(c.replace("share_", "") for c in share_cols) + " |")
    lines.append("|---" * (len(share_cols) + 1) + "|")
    # itertuples mangles column names containing brackets, so use dict records here.
    for record in year_table.to_dict(orient="records"):
        cells = " | ".join(f"{record[c]:.3f}" for c in share_cols)
        lines.append(f"| {record['year']} | " + cells + " |")

    lines.append("\n## 3. 2023 기준 구간별 정산비 (실력 척도)\n")
    lines.append("| 구간 | 정산비 |")
    lines.append("|---|---:|")
    for level, value in settlement.items():
        lines.append(f"| {level} | {value:.4f} |")

    lines.append("\n## 4. 입력 난이도 (연도별 sigma_v)\n")
    lines.append("| year | group | n | sigma_v (m/s) |")
    lines.append("|---|---:|---:|---:|")
    for r in sigma_table.itertuples(index=False):
        lines.append(f"| {r.year} | {r.group_id} | {r.n} | {r.sigma_v:.3f} |")

    (REPORTS / "m270_year_difficulty.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    receipt = {
        "schema_version": 1,
        "stage": "M270_YEAR_DIFFICULTY",
        "skill_reference": "2023 dev folds, deployed policy",
        "level_edges": list(LEVEL_EDGES),
        "band_settlement_2023": {str(k): float(v) for k, v in settlement.items()},
        "year_table": year_table.to_dict(orient="records"),
        "sigma_by_year": sigma_table.to_dict(orient="records"),
        "note": "2025 difficulty is unobservable; this measures spread across observable years",
        "model_fits": 0,
        "new_2024_evaluation": False,
        "lockbox_reopened": False,
        "dacon_upload": False,
        "external_actions": [],
    }
    (REPORTS / "m270_year_difficulty_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=1, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

    print(year_table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print()
    print(sigma_table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
