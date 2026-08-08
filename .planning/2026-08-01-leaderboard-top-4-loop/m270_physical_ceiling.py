"""M270 round 3: is the steep-segment error already at its physical ceiling?

QUESTION
The regime diagnosis located the deficit on the steep part of the power curve, where mean
absolute error is about 0.17 of capacity against a 0.06 settlement band. If that error is
explained by NWP wind-speed error propagated through the power-curve slope,

    sigma_P  ~=  |dP/dv| * sigma_v

then no modelling on the supplied NWP can close it and the ceiling is set by wind-forecast
accuracy. If observed error EXCEEDS the physical prediction, recoverable modelling loss
remains and this quantifies it.

WHY THIS IS NOT A DUPLICATE
The record notes that "the inference-safe NWP-to-turbine-wind error collapses the physical
stack" (2026-08-03), but that is an observation that a MODEL failed. M165 built an
empirical site-wind power curve and it failed promotion. Neither quantified the implied
ceiling. Building a physical stack and measuring whether its failure is irreducible are
different operations.

SCADA USE
SCADA is train-only and unavailable at inference. It is used here purely as a DIAGNOSTIC
measurement of wind-error magnitude, exactly as the project previously used it to check
that turbine power sums reproduce official labels. No inference feature is created.

Read-only. No model is fitted, no 2024 row is read.
"""

from __future__ import annotations

import hashlib
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
# Gamma 0 is the pure-NMAE action, the closest available thing to a point forecast. The
# deployed policy deliberately shifts away from the conditional centre to chase settlement,
# so comparing IT against a propagated-error prediction would confound decision shift with
# forecast error.
POINT_POLICY = "T1_G0"

GROUP_TURBINES = {
    1: [f"vestas_wtg{i:02d}_ws" for i in range(1, 7)],
    2: [f"vestas_wtg{i:02d}_ws" for i in range(7, 13)],
    3: [f"unison_wtg{i:02d}_ws" for i in range(1, 6)],
}
LEVEL_EDGES = (0.10, 0.25, 0.45, 0.70, 1.10)
WIND_BIN = 0.5
MIN_BIN_ROWS = 30


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        if not present:
            continue
        part = pd.DataFrame(
            {
                "forecast_kst_dtm": merged["kst_dtm"],
                "group_id": group,
                "scada_ws": merged[present].mean(axis=1, skipna=True),
            }
        )
        rows.append(part.dropna(subset=["scada_ws"]))
    return pd.concat(rows, ignore_index=True)


def load_surface() -> pd.DataFrame:
    from strict_dev_surface import development_surface

    surface, _ = development_surface()
    out = surface.loc[
        :, ["forecast_kst_dtm", "group_id", "actual_kwh", "phys_v2__hub117_speed"]
    ].copy()
    out["forecast_kst_dtm"] = pd.to_datetime(out["forecast_kst_dtm"])
    return out


def load_point_predictions() -> pd.DataFrame:
    parts = []
    for fold in FOLDS:
        frame = pd.read_parquet(PROBE_DIR / PARENT_PATHS[fold])
        part = frame.loc[:, ["forecast_kst_dtm", "group_id"]].copy()
        part["point_kwh"] = frame[POINT_POLICY].to_numpy(dtype=float)
        parts.append(part)
    out = pd.concat(parts, ignore_index=True)
    out["forecast_kst_dtm"] = pd.to_datetime(out["forecast_kst_dtm"])
    return out


def power_curve_slope(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    """Empirical normalized-power curve and its finite-difference slope, per group."""
    rows = []
    slopes: dict[int, np.ndarray] = {}
    for group, part in frame.groupby("group_id", sort=True):
        binned = (part["scada_ws"] / WIND_BIN).round().astype(int) * WIND_BIN
        table = part.assign(vbin=binned).groupby("vbin").agg(
            n=("y", "size"), mean_y=("y", "mean")
        )
        table = table.loc[table["n"] >= MIN_BIN_ROWS].sort_index()
        if len(table) < 3:
            continue
        centres = table.index.to_numpy(dtype=float)
        means = table["mean_y"].to_numpy(dtype=float)
        slope = np.gradient(means, centres)
        slopes[int(group)] = np.vstack([centres, slope])
        for centre, value, gradient, count in zip(
            centres, means, slope, table["n"].to_numpy(), strict=True
        ):
            rows.append(
                {
                    "group_id": int(group),
                    "wind_bin": float(centre),
                    "n": int(count),
                    "mean_y": float(value),
                    "slope_per_ms": float(gradient),
                }
            )
    return pd.DataFrame(rows), slopes


def main() -> None:
    surface = load_surface()
    scada = load_scada()
    points = load_point_predictions()

    from baram.constants import CAPACITIES_KWH

    merged = surface.merge(scada, on=["forecast_kst_dtm", "group_id"], how="inner")
    merged = merged.merge(points, on=["forecast_kst_dtm", "group_id"], how="inner")
    capacity = merged["group_id"].map(CAPACITIES_KWH).astype(float)
    merged["y"] = merged["actual_kwh"] / capacity
    merged["point_err"] = (merged["point_kwh"] - merged["actual_kwh"]).abs() / capacity
    merged = merged.loc[
        merged["y"].between(0.10, 1.10) & merged["scada_ws"].between(0.5, 30.0)
    ].copy()

    # Wind error, bias removed per group: the systematic offset between a 117 m NWP proxy
    # and nacelle anemometry is not forecast error and would inflate sigma_v.
    merged["v_err"] = merged["phys_v2__hub117_speed"] - merged["scada_ws"]
    bias = merged.groupby("group_id")["v_err"].transform("median")
    merged["v_err_debiased"] = merged["v_err"] - bias
    sigma_v = merged.groupby("group_id")["v_err_debiased"].std()

    curve, slopes = power_curve_slope(merged)
    merged["slope"] = [
        float(np.interp(ws, slopes[g][0], slopes[g][1])) if g in slopes else np.nan
        for ws, g in zip(merged["scada_ws"], merged["group_id"], strict=True)
    ]
    merged["sigma_p_pred"] = merged["slope"].abs() * merged["group_id"].map(sigma_v)
    # Half-normal mean: E|X| = sigma * sqrt(2/pi) for zero-mean Gaussian error.
    merged["pred_abs_err"] = merged["sigma_p_pred"] * np.sqrt(2.0 / np.pi)
    merged["ylev"] = pd.cut(merged["y"], list(LEVEL_EDGES), include_lowest=True)

    lines: list[str] = []
    lines.append("# M270 — 급경사 구간 물리 천장 분석\n")
    lines.append("- 질문: 관측 오차가 `|dP/dv| x sigma_v`로 설명되는가")
    lines.append(f"- 비교 대상: `{POINT_POLICY}` (순수 NMAE 행동 = 점예측 근사)")
    lines.append("- SCADA는 **진단 전용**이며 추론 피처를 만들지 않는다")
    lines.append(f"- 결합 행: **{len(merged)}**\n")

    lines.append("## 1. 풍속 오차 (그룹별 편향 제거 후)\n")
    lines.append("| group | n | 중앙 편향 (m/s) | sigma_v (m/s) |")
    lines.append("|---|---:|---:|---:|")
    for group in sorted(sigma_v.index):
        part = merged.loc[merged["group_id"].eq(group)]
        lines.append(
            f"| {group} | {len(part)} | {part['v_err'].median():+.3f} | {sigma_v[group]:.3f} |"
        )

    lines.append("\n## 2. 발전량 구간별 — 관측 오차 대 물리 예측\n")
    lines.append("| y 구간 | n | 평균 \\|기울기\\| | 예측 평균오차 | 관측 평균오차 | 관측/예측 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    summary = {}
    for level, block in merged.groupby("ylev", observed=True):
        valid = block.dropna(subset=["pred_abs_err"])
        if len(valid) < MIN_BIN_ROWS:
            continue
        pred = float(valid["pred_abs_err"].mean())
        obs = float(valid["point_err"].mean())
        summary[str(level)] = {
            "n": len(valid), "pred": pred, "obs": obs,
            "ratio": obs / pred if pred > 0 else float("nan"),
        }
        lines.append(
            f"| {level} | {len(valid)} | {valid['slope'].abs().mean():.4f} | "
            f"{pred:.4f} | {obs:.4f} | **{obs / pred:.2f}** |"
        )

    lines.append("\n## 3. 경험적 파워커브 기울기\n")
    lines.append("| group | 풍속 bin | n | 평균 y | 기울기 (1/(m/s)) |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in curve.itertuples(index=False):
        lines.append(
            f"| {row.group_id} | {row.wind_bin:.1f} | {row.n} | {row.mean_y:.4f} | "
            f"{row.slope_per_ms:+.4f} |"
        )

    lines.append("\n## 4. 읽는 법\n")
    lines.append(
        "관측/예측 비율이 **1에 가까우면** 그 구간의 오차는 풍속 예보 오차가 파워커브를 통해 "
        "증폭된 결과이며, 공급된 NWP 위에서는 모델링으로 줄일 수 없다. **1보다 크면** 그 초과분이 "
        "회수 가능한 모델링 손실이다."
    )

    (REPORTS / "m270_physical_ceiling.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    receipt = {
        "schema_version": 1,
        "stage": "M270_PHYSICAL_CEILING_ANALYSIS",
        "point_policy": POINT_POLICY,
        "archive_sha256": sha256_file(ARCHIVE),
        "merged_rows": len(merged),
        "sigma_v_ms": {int(k): float(v) for k, v in sigma_v.items()},
        "level_summary": summary,
        "scada_use": "diagnostic only; no inference feature created",
        "model_fits": 0,
        "new_2024_evaluation": False,
        "lockbox_reopened": False,
        "dacon_upload": False,
        "external_actions": [],
    }
    (REPORTS / "m270_physical_ceiling_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=1, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

    print(f"merged={len(merged)}")
    print("sigma_v (m/s):", {int(k): round(float(v), 3) for k, v in sigma_v.items()})
    for level, values in summary.items():
        print(
            f"  {level:<16} n={values['n']:>5}  pred={values['pred']:.4f}  "
            f"obs={values['obs']:.4f}  obs/pred={values['ratio']:.2f}"
        )


if __name__ == "__main__":
    main()
