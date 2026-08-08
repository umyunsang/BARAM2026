"""M270 round 3b: can ANY additional NWP source close the wind-accuracy gap?

CONTEXT
The ceiling analysis measured sigma_v about 3 m/s against a requirement near 0.4-0.6 m/s
to land inside the settlement band on the steep power-curve segment. The only lever that
targets that mechanism is a more accurate wind forecast. Before collecting an external
source, this decomposes the error already present in the two SUPPLIED sources.

DECISIVE QUANTITY
The correlation between GFS and LDAPS wind errors. Independent errors average down roughly
as 1/sqrt(n); correlated errors do not average down at all. With rho measured, the best
sigma_v that ANY number of equally good, equally correlated sources could reach is
bounded analytically - no collection required.

Each source is given a per-group linear calibration before comparison so that differing
measurement heights (GFS 100 m, LDAPS 50 m, nacelle anemometry) do not masquerade as
forecast error. Fitting two parameters per group on thousands of rows is mildly optimistic
and is recorded as such.

Read-only. SCADA is diagnostic only. No model is fitted for inference, no 2024 row is read.
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
REPORTS = ROOT / "reports"
ARCHIVE = Path("/Users/um-yunsang/Downloads/open.zip")
sys.path.insert(0, str(PLAN_DIR))

GROUP_TURBINES = {
    1: [f"vestas_wtg{i:02d}_ws" for i in range(1, 7)],
    2: [f"vestas_wtg{i:02d}_ws" for i in range(7, 13)],
    3: [f"unison_wtg{i:02d}_ws" for i in range(1, 6)],
}
GFS_COL = "gfs_spatial__idw__wind100_speed"
LDAPS_COL = "ldaps_spatial__idw__wind50max_speed"

# Wind accuracy the settlement band demands, from the empirical power-curve slope:
# 0.06 capacity / slope. Peak slope 0.152 per m/s and typical steep-band slope 0.105.
REQUIRED_SIGMA_PEAK = 0.06 / 0.152
REQUIRED_SIGMA_TYPICAL = 0.06 / 0.105


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


def calibrate(source: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Per-group linear calibration, so height differences are not counted as error."""
    slope, intercept = np.polyfit(source, truth, 1)
    return source * slope + intercept


def main() -> None:
    from strict_dev_surface import development_surface

    surface, _ = development_surface()
    frame = surface.loc[
        :, ["forecast_kst_dtm", "group_id", GFS_COL, LDAPS_COL]
    ].copy()
    frame["forecast_kst_dtm"] = pd.to_datetime(frame["forecast_kst_dtm"])
    merged = frame.merge(load_scada(), on=["forecast_kst_dtm", "group_id"], how="inner")
    merged = merged.dropna(subset=[GFS_COL, LDAPS_COL, "scada_ws"])
    merged = merged.loc[merged["scada_ws"].between(0.5, 30.0)]

    rows = []
    for group, part in merged.groupby("group_id", sort=True):
        truth = part["scada_ws"].to_numpy(dtype=float)
        gfs = calibrate(part[GFS_COL].to_numpy(dtype=float), truth)
        ldaps = calibrate(part[LDAPS_COL].to_numpy(dtype=float), truth)
        err_g, err_l = gfs - truth, ldaps - truth
        blend = calibrate(
            np.vstack([part[GFS_COL], part[LDAPS_COL]]).mean(axis=0).astype(float), truth
        )
        rho = float(np.corrcoef(err_g, err_l)[0, 1])
        rows.append(
            {
                "group_id": int(group),
                "n": len(part),
                "sigma_gfs": float(err_g.std()),
                "sigma_ldaps": float(err_l.std()),
                "sigma_mean_blend": float((blend - truth).std()),
                "rho_errors": rho,
            }
        )
    table = pd.DataFrame(rows)

    # Analytic bound: n equicorrelated sources of equal sigma average to
    # sigma * sqrt((1 + (n-1)*rho) / n), which tends to sigma*sqrt(rho) as n grows.
    table["sigma_floor_infinite_sources"] = table[["sigma_gfs", "sigma_ldaps"]].mean(
        axis=1
    ) * np.sqrt(table["rho_errors"].clip(lower=0.0))
    table["sigma_three_sources"] = table[["sigma_gfs", "sigma_ldaps"]].mean(axis=1) * np.sqrt(
        (1 + 2 * table["rho_errors"]) / 3
    )

    lines: list[str] = []
    lines.append("# M270 — 소스별 풍속 오차 분해와 다중소스 하한\n")
    lines.append(f"- GFS 대표 컬럼 `{GFS_COL}` (100 m), LDAPS `{LDAPS_COL}` (50 m)")
    lines.append("- 각 소스에 그룹별 선형 보정 후 잔차 표준편차를 비교 (높이 차이 제거)")
    lines.append("- SCADA는 진단 전용, 추론 피처 생성 없음\n")

    lines.append("## 1. 소스별 오차와 상관\n")
    lines.append("| group | n | sigma GFS | sigma LDAPS | 평균혼합 | 오차상관 rho |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for r in table.itertuples(index=False):
        lines.append(
            f"| {r.group_id} | {r.n} | {r.sigma_gfs:.3f} | {r.sigma_ldaps:.3f} | "
            f"{r.sigma_mean_blend:.3f} | **{r.rho_errors:.3f}** |"
        )

    lines.append("\n## 2. 소스를 더 넣으면 어디까지 가는가 (해석적 하한)\n")
    lines.append(
        "동일 품질·동일 상관 소스 n개의 평균 오차는 `sigma * sqrt((1+(n-1)rho)/n)`이며, "
        "n을 무한히 늘려도 `sigma * sqrt(rho)` 아래로는 내려가지 않는다.\n"
    )
    lines.append(
        "| group | 현재최선 | 3소스 | **무한소스 하한** | 필요치 | 하한/필요치 |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")
    for r in table.itertuples(index=False):
        best = min(r.sigma_gfs, r.sigma_ldaps, r.sigma_mean_blend)
        lines.append(
            f"| {r.group_id} | {best:.3f} | {r.sigma_three_sources:.3f} | "
            f"**{r.sigma_floor_infinite_sources:.3f}** | {REQUIRED_SIGMA_TYPICAL:.3f} | "
            f"**{r.sigma_floor_infinite_sources / REQUIRED_SIGMA_TYPICAL:.1f}x** |"
        )

    lines.append(
        f"\n필요 풍속 정확도는 최대기울기 기준 `{REQUIRED_SIGMA_PEAK:.3f}` m/s, "
        f"급경사 전형 기울기 기준 `{REQUIRED_SIGMA_TYPICAL:.3f}` m/s다."
    )

    lines.append("\n## 3. 읽는 법\n")
    lines.append(
        "오차상관이 높으면 두 소스가 **같은 방향으로 틀린다**는 뜻이고, 소스를 더 넣어도 "
        "평균화로 상쇄되지 않는다. 무한소스 하한이 필요치보다 여전히 크면 **외부 NWP 소스를 "
        "아무리 추가해도 정산 밴드에 필요한 풍속 정확도에 도달할 수 없다**는 결론이 수집 없이 "
        "확정된다."
    )

    (REPORTS / "m270_sigma_decomposition.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    receipt = {
        "schema_version": 1,
        "stage": "M270_SIGMA_SOURCE_DECOMPOSITION",
        "gfs_column": GFS_COL,
        "ldaps_column": LDAPS_COL,
        "required_sigma_peak_slope": REQUIRED_SIGMA_PEAK,
        "required_sigma_typical_slope": REQUIRED_SIGMA_TYPICAL,
        "per_group": table.to_dict(orient="records"),
        "calibration_note": "two parameters per group fitted in-sample; mildly optimistic",
        "scada_use": "diagnostic only; no inference feature created",
        "model_fits": 0,
        "new_2024_evaluation": False,
        "lockbox_reopened": False,
        "dacon_upload": False,
        "external_actions": [],
    }
    (REPORTS / "m270_sigma_decomposition_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=1, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nrequired sigma_v: peak-slope {REQUIRED_SIGMA_PEAK:.3f}, "
          f"typical-steep {REQUIRED_SIGMA_TYPICAL:.3f} m/s")


if __name__ == "__main__":
    main()
