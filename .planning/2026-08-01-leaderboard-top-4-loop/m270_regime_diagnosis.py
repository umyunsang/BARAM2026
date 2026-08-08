"""M270 finding 3b: why do some months collapse while others already clear the target?

The monthly block view showed deployed FICR ranging from 0.310503 in July to 0.464753 in
December, with December already above the 0.44675 rank-20 requirement. The quarterly view
hid this by averaging October against December inside Q4.

This diagnostic asks what is different about the weak months, using only persisted
predictions, actuals, and timestamps. No refit, no new features, no 2024 access.

Decomposition axes:
  * settlement tier counts and generation-weighted mass (is it more misses, or shifted mass?)
  * per group (is the collapse group-specific?)
  * generation level (does it break at high or low output?)
  * ramp magnitude (is it a variability regime?)
  * error scale versus generation scale (is the band simply harder when output is high?)
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from baram.constants import CAPACITIES_KWH, METRIC_COLUMNS
from baram.evaluation.official import evaluate_official

ROOT = Path(__file__).resolve().parents[2]
PROBE = ROOT / "artifacts" / "backtests" / "metric-aligned-probe"
REPORTS = ROOT / "reports"

FOLDS = ("dev-2023-Q2", "dev-2023-Q3", "dev-2023-Q4")
PARENT_PATHS = {fold: f"M269_PROBE_TOP100-{fold}-policies.parquet" for fold in FOLDS}
DEPLOYED = "T0.5_G1.5"

WEAK_MONTHS = ("2023-07", "2023-10")
STRONG_MONTHS = ("2023-11", "2023-12")
LEVEL_EDGES = (0.10, 0.25, 0.45, 0.70, 1.10)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load() -> pd.DataFrame:
    parts = []
    for fold in FOLDS:
        frame = pd.read_parquet(PROBE / PARENT_PATHS[fold])
        part = frame.loc[:, ["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh"]].copy()
        part["prediction_kwh"] = frame[DEPLOYED].to_numpy(dtype=float)
        parts.append(part)
    out = pd.concat(parts, ignore_index=True)
    out["forecast_kst_dtm"] = pd.to_datetime(out["forecast_kst_dtm"])
    out["month"] = out["forecast_kst_dtm"].dt.to_period("M").astype(str)
    out["capacity"] = out["group_id"].map(CAPACITIES_KWH).astype(float)
    out["y"] = out["actual_kwh"] / out["capacity"]
    out["yhat"] = out["prediction_kwh"] / out["capacity"]
    out["abs_err"] = (out["yhat"] - out["y"]).abs()
    out["signed_err"] = out["yhat"] - out["y"]
    out["eligible"] = out["y"] >= 0.10
    out["units"] = np.select([out["abs_err"] <= 0.06, out["abs_err"] <= 0.08], [4.0, 3.0], 0.0)
    return out.sort_values(["group_id", "forecast_kst_dtm"], kind="stable").reset_index(drop=True)


def add_ramp(frame: pd.DataFrame) -> pd.DataFrame:
    """Hour-to-hour absolute change in normalized actual, within a group."""
    out = frame.copy()
    out["ramp"] = out.groupby("group_id", sort=False)["y"].diff().abs()
    return out


def month_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for month, part in frame.groupby("month", sort=True):
        eligible = part.loc[part["eligible"]]
        if len(eligible) < 150:
            continue
        scored = part.loc[:, sorted(METRIC_COLUMNS)].copy()
        official = evaluate_official(scored, CAPACITIES_KWH)
        gen = eligible["y"].to_numpy()
        units = eligible["units"].to_numpy()
        rows.append(
            {
                "month": month,
                "ficr": official.ficr,
                "one_minus_nmae": official.one_minus_nmae,
                "eligible_rate": float(part["eligible"].mean()),
                "mean_y": float(gen.mean()),
                "std_y": float(gen.std()),
                "mean_abs_err": float(eligible["abs_err"].mean()),
                "mean_signed_err": float(eligible["signed_err"].mean()),
                "hit4_rate": float((units == 4.0).mean()),
                "hit3_rate": float((units == 3.0).mean()),
                "miss_rate": float((units == 0.0).mean()),
                "genw_hit4": float((gen * (units == 4.0)).sum() / gen.sum()),
                "genw_miss": float((gen * (units == 0.0)).sum() / gen.sum()),
                "mean_ramp": float(eligible["ramp"].abs().mean(skipna=True)),
                "err_over_y": float(eligible["abs_err"].mean() / gen.mean()),
            }
        )
    return pd.DataFrame(rows)


def group_month_ficr(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (month, group), part in frame.groupby(["month", "group_id"], sort=True):
        eligible = part.loc[part["eligible"]]
        if len(eligible) < 50:
            continue
        gen = eligible["y"].to_numpy()
        units = eligible["units"].to_numpy()
        rows.append(
            {
                "month": month,
                "group_id": int(group),
                "ficr": float((gen * units).sum() / (gen * 4.0).sum()),
                "mean_abs_err": float(eligible["abs_err"].mean()),
                "mean_y": float(gen.mean()),
            }
        )
    return pd.DataFrame(rows)


def level_table(frame: pd.DataFrame, months: tuple[str, ...], label: str) -> pd.DataFrame:
    part = frame.loc[frame["month"].isin(months) & frame["eligible"]].copy()
    part["level"] = pd.cut(part["y"], bins=list(LEVEL_EDGES), include_lowest=True)
    rows = []
    for level, block in part.groupby("level", observed=True, sort=True):
        gen = block["y"].to_numpy()
        units = block["units"].to_numpy()
        rows.append(
            {
                "regime": label,
                "level": str(level),
                "rows": len(block),
                "gen_share": float(gen.sum()),
                "mean_abs_err": float(block["abs_err"].mean()),
                "hit4_rate": float((units == 4.0).mean()),
                "genw_units": float((gen * units).sum() / (gen * 4.0).sum()),
            }
        )
    out = pd.DataFrame(rows)
    out["gen_share"] = out["gen_share"] / out["gen_share"].sum()
    return out


def main() -> None:
    frame = add_ramp(load())
    months = month_table(frame)
    groups = group_month_ficr(frame)
    weak = level_table(frame, WEAK_MONTHS, "약월(7,10)")
    strong = level_table(frame, STRONG_MONTHS, "강월(11,12)")

    lines: list[str] = []
    lines.append("# M270 — 계절 체제별 붕괴 진단\n")
    lines.append("- 배포 정책 고정, 재적합 없음, 신규 피처 없음, 2024 미접근")
    lines.append(f"- 약월 {list(WEAK_MONTHS)} vs 강월 {list(STRONG_MONTHS)}\n")

    lines.append("## 1. 월별 분해\n")
    header = (
        "| month | FICR | 유효율 | 평균y | 표준편차y | 평균\\|오차\\| | 평균부호오차 "
        "| 4단위율 | miss율 | 발전량가중 miss | 평균램프 | 오차/y |"
    )
    lines.append(header)
    lines.append("|---" * 12 + "|")
    for r in months.itertuples(index=False):
        lines.append(
            f"| {r.month} | {r.ficr:.4f} | {r.eligible_rate:.3f} | {r.mean_y:.3f} | "
            f"{r.std_y:.3f} | {r.mean_abs_err:.4f} | {r.mean_signed_err:+.4f} | "
            f"{r.hit4_rate:.3f} | {r.miss_rate:.3f} | {r.genw_miss:.3f} | "
            f"{r.mean_ramp:.4f} | {r.err_over_y:.3f} |"
        )

    lines.append("\n## 2. 그룹별 월별 FICR\n")
    pivot = groups.pivot(index="month", columns="group_id", values="ficr")
    lines.append("| month | g1 | g2 | g3 |")
    lines.append("|---|---:|---:|---:|")
    for month, row in pivot.iterrows():
        cells = " | ".join(f"{row.get(g, float('nan')):.4f}" for g in (1, 2, 3))
        lines.append(f"| {month} | {cells} |")

    lines.append("\n## 3. 발전량 수준별 (약월 vs 강월)\n")
    lines.append(
        "| 체제 | 수준 | 행수 | 발전량비중 | 평균오차 | 4단위율 | 가중정산비 |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for table in (weak, strong):
        for r in table.itertuples(index=False):
            lines.append(
                f"| {r.regime} | {r.level} | {r.rows} | {r.gen_share:.3f} | "
                f"{r.mean_abs_err:.4f} | {r.hit4_rate:.3f} | {r.genw_units:.4f} |"
            )

    weak_rows = months.loc[months["month"].isin(WEAK_MONTHS)]
    strong_rows = months.loc[months["month"].isin(STRONG_MONTHS)]
    lines.append("\n## 4. 약월 대 강월 요약\n")
    lines.append("| 지표 | 약월 평균 | 강월 평균 | 차이 |")
    lines.append("|---|---:|---:|---:|")
    for column in (
        "ficr", "eligible_rate", "mean_y", "std_y", "mean_abs_err",
        "mean_signed_err", "hit4_rate", "genw_miss", "mean_ramp", "err_over_y",
    ):
        a = float(weak_rows[column].mean())
        b = float(strong_rows[column].mean())
        lines.append(f"| {column} | {a:.4f} | {b:.4f} | {b - a:+.4f} |")

    (REPORTS / "m270_regime_diagnosis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    receipt = {
        "schema_version": 1,
        "stage": "M270_REGIME_COLLAPSE_DIAGNOSIS",
        "policy": DEPLOYED,
        "weak_months": list(WEAK_MONTHS),
        "strong_months": list(STRONG_MONTHS),
        "level_edges": list(LEVEL_EDGES),
        "inputs_sha256": {p: sha256_file(PROBE / p) for p in PARENT_PATHS.values()},
        "monthly": months.to_dict(orient="records"),
        "group_month": groups.to_dict(orient="records"),
        "model_fits": 0,
        "new_2024_evaluation": False,
        "lockbox_reopened": False,
        "dacon_upload": False,
        "external_actions": [],
    }
    (REPORTS / "m270_regime_diagnosis_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=1, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

    pd.set_option("display.width", 200)
    print(months.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nweak vs strong:")
    summary_columns = (
        "ficr", "mean_y", "std_y", "mean_abs_err", "hit4_rate", "mean_ramp", "err_over_y",
    )
    for column in summary_columns:
        a = float(weak_rows[column].mean())
        b = float(strong_rows[column].mean())
        print(f"  {column:<16} weak={a:.4f}  strong={b:.4f}  diff={b - a:+.4f}")


if __name__ == "__main__":
    main()
