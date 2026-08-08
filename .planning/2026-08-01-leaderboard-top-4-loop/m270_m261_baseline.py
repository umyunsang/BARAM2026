"""M270: the local counterpart of submission_M261, predeclared BEFORE its online result.

WHY
No same-lineage local-to-online anchor exists. The two earliest online submissions predate
this repository and M252 never had a local score computed. Every "we are at 0.628" statement
this session therefore rests on an uncalibrated ruler.

`submission_M261.csv` is the full-history deployment of exactly the M107/M102 Q2-frozen
champion policy, so its chronology-safe local counterpart IS computable. Computing it now,
before the online result is known, makes the resulting offset a measurement rather than a
post-hoc rationalisation.

WHAT IS COMPUTED
The M269 probe reproduced the M102 classifier on the CORRECTED surface (post
`_strict_preceding_mask`). Applying the exact frozen M107 temporal transform to those
predictions reproduces the M107-equivalent locally, which is the policy M261 deploys.

PREDECLARED EXPECTATION
Online should EXCEED this local figure, because the deployed model is fitted on all
2022-2024 history while each local fold trains only on data preceding it. The magnitude is
unknown; whatever it turns out to be is recorded as measured.

Read-only. No model is fitted, no 2024 row is read, no upload is performed.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from baram.constants import CAPACITIES_KWH, METRIC_COLUMNS
from baram.evaluation.official import evaluate_official

ROOT = Path(__file__).resolve().parents[2]
PROBE_DIR = ROOT / "artifacts" / "backtests" / "metric-aligned-probe"
REPORTS = ROOT / "reports"
SUBMISSION = ROOT / "artifacts" / "submissions" / "submission_M261.csv"

FOLDS = ("dev-2023-Q2", "dev-2023-Q3", "dev-2023-Q4")
PARENT_PATHS = {fold: f"M269_PROBE_TOP100-{fold}-policies.parquet" for fold in FOLDS}
DEPLOYED = "T0.5_G1.5"

# Exact frozen M107 transform, copied from build_full_history_strict_temporal_champion.py.
TEMPORAL_POLICY = {
    1: {"shift_hours": -1, "original_weight": 0.7},
    2: {"shift_hours": -2, "original_weight": 0.8},
    3: {"shift_hours": -2, "original_weight": 0.8},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_probe() -> pd.DataFrame:
    parts = []
    for fold in FOLDS:
        frame = pd.read_parquet(PROBE_DIR / PARENT_PATHS[fold])
        columns = ["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh"]
        part = frame.loc[:, columns].copy()
        part["prediction_kwh"] = frame[DEPLOYED].to_numpy(dtype=float)
        parts.append(part)
    out = pd.concat(parts, ignore_index=True)
    out["forecast_kst_dtm"] = pd.to_datetime(out["forecast_kst_dtm"])
    # forecast_id keys the FORECAST timestamp, not the issuance batch, so grouping by it
    # would leave one row per group and the shift would silently do nothing. The real
    # issuance key lives on the development surface.
    from strict_dev_surface import development_surface

    surface, _ = development_surface()
    issuance = surface.loc[
        :, ["forecast_kst_dtm", "group_id", "data_available_kst_dtm"]
    ].copy()
    issuance["forecast_kst_dtm"] = pd.to_datetime(issuance["forecast_kst_dtm"])
    out = out.merge(issuance, on=["forecast_kst_dtm", "group_id"], how="left")
    # The probe carries a 2024-01 boundary block that the physically pre-2024 development
    # surface excludes. Those rows get no issuance key; they are dropped from the transform
    # surface rather than silently transformed under a wrong grouping.
    missing = int(out["data_available_kst_dtm"].isna().sum())
    if missing:
        print(f"  dropping {missing} boundary rows without an issuance key", flush=True)
        out = out.loc[out["data_available_kst_dtm"].notna()].copy()
    out["issuance"] = out["data_available_kst_dtm"]
    return out


def apply_temporal(frame: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for group, selection in TEMPORAL_POLICY.items():
        part = (
            frame.loc[frame["group_id"].eq(group)]
            .sort_values(["issuance", "forecast_kst_dtm"], kind="stable")
            .copy()
        )
        shifted = (
            part.groupby("issuance", sort=False)["prediction_kwh"]
            .shift(int(selection["shift_hours"]))
            .fillna(part["prediction_kwh"])
        )
        weight = float(selection["original_weight"])
        part["prediction_kwh"] = weight * part["prediction_kwh"] + (1.0 - weight) * shifted
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def score(frame: pd.DataFrame) -> dict[str, float]:
    scored = frame.loc[:, sorted(METRIC_COLUMNS)].copy()
    result = evaluate_official(scored, CAPACITIES_KWH)
    return {
        "total": result.total,
        "one_minus_nmae": result.one_minus_nmae,
        "ficr": result.ficr,
        "group_ficr": {int(k): float(v) for k, v in result.group_ficr.items()},
    }


def main() -> None:
    probe = load_probe()
    m102_level = score(probe)
    m107_level = score(apply_temporal(probe))

    monthly = []
    transformed = apply_temporal(probe)
    transformed["month"] = transformed["forecast_kst_dtm"].dt.to_period("M").astype(str)
    for month, block in transformed.groupby("month", sort=True):
        if len(block) < 500:
            continue
        monthly.append({"month": month, **{
            k: v for k, v in score(block).items() if k != "group_ficr"
        }})

    lines: list[str] = []
    lines.append("# M270 — submission_M261의 로컬 대응값 (온라인 결과 전 사전 확정)\n")
    lines.append(f"- 제출 파일 SHA-256: `{sha256_file(SUBMISSION)}`")
    lines.append("- 대응 정책: M107/M102 Q2-동결 챔피언 (M261이 배포하는 바로 그 정책)")
    lines.append("- 표면: M269 수정 표면 (`_strict_preceding_mask` 적용 후)\n")

    lines.append("## 1. 로컬 시간순 안전 pooled OOF\n")
    lines.append("| 수준 | Total | 1-NMAE | FICR |")
    lines.append("|---|---:|---:|---:|")
    lines.append(
        f"| M102 (변환 전) | {m102_level['total']:.6f} | "
        f"{m102_level['one_minus_nmae']:.6f} | {m102_level['ficr']:.6f} |"
    )
    lines.append(
        f"| **M107 (변환 후 = M261 대응)** | **{m107_level['total']:.6f}** | "
        f"{m107_level['one_minus_nmae']:.6f} | {m107_level['ficr']:.6f} |"
    )
    groups = m107_level["group_ficr"]
    lines.append(
        f"\n그룹별 FICR: g1 `{groups[1]:.6f}` / g2 `{groups[2]:.6f}` "
        f"/ g3 `{groups[3]:.6f}`"
    )

    lines.append("\n## 2. 월별 (온라인은 연간 집계이므로 대조용)\n")
    lines.append("| month | Total | 1-NMAE | FICR |")
    lines.append("|---|---:|---:|---:|")
    for row in monthly:
        lines.append(
            f"| {row['month']} | {row['total']:.6f} | "
            f"{row['one_minus_nmae']:.6f} | {row['ficr']:.6f} |"
        )

    lines.append("\n## 3. 사전 선언\n")
    lines.append(
        "온라인 점수는 이 로컬 값을 **초과할 것으로 예상**한다. 배포 모델은 전체 이력으로 "
        "적합되지만 로컬 fold는 그 이전 데이터만 쓰기 때문이다. 크기는 알 수 없다.\n"
    )
    lines.append("측정될 오프셋 = (온라인 Total) - (로컬 Total). 결과가 무엇이든 기록한다.")
    lines.append(
        "\n이 값을 결과를 본 뒤 재계산하거나 재해석하지 않는다. 사후 합리화를 막기 위해 "
        "지금 고정한다."
    )

    (REPORTS / "m270_m261_local_baseline.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    receipt = {
        "schema_version": 1,
        "stage": "M270_M261_LOCAL_BASELINE_PREDECLARED",
        "submission_path": str(SUBMISSION),
        "submission_sha256": sha256_file(SUBMISSION),
        "policy": "M107/M102 Q2-frozen champion, deployed full-history as M261",
        "surface": "M269 corrected probe surface",
        "temporal_policy": TEMPORAL_POLICY,
        "local_m102_level": m102_level,
        "local_m107_level": m107_level,
        "monthly": monthly,
        "predeclared_expectation": "online Total should exceed local Total; magnitude unknown",
        "online_result": None,
        "measured_offset": None,
        "model_fits": 0,
        "new_2024_evaluation": False,
        "lockbox_reopened": False,
        "dacon_upload": False,
        "external_actions": [],
    }
    (REPORTS / "m270_m261_local_baseline_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=1, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

    print(f"submission sha256: {sha256_file(SUBMISSION)}")
    print(f"local M102 level : Total={m102_level['total']:.6f} "
          f"1-NMAE={m102_level['one_minus_nmae']:.6f} FICR={m102_level['ficr']:.6f}")
    print(f"local M107 level : Total={m107_level['total']:.6f} "
          f"1-NMAE={m107_level['one_minus_nmae']:.6f} "
          f"FICR={m107_level['ficr']:.6f}   <-- M261 counterpart")


if __name__ == "__main__":
    main()
