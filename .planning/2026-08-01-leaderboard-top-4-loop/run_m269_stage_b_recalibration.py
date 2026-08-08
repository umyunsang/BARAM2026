"""M269 T1 Stage B: learnable conditional-bias (calibration) headroom, chronology-safe.

Measures how much FICR a stratified constant action offset recovers on top of the frozen
deployed champion policy. Two protocols are reported:

  * same-fold oracle  - offsets fitted on the fold being scored (upper bound, not promotable)
  * strict prequential - offsets fitted only on strictly preceding folds (promotable estimate)

Strata are group x predicted-level bin, so the offset is a function of the model's own
output only. This isolates conditional bias that a recalibration layer could learn without
any new feature, model family, or refit.

Read-only inputs. No model is fitted, no 2024 row is read, no submission is built.
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
PARENT_PATHS = {
    fold: f"M269_PROBE_TOP100-{fold}-policies.parquet" for fold in FOLDS
}
DEPLOYED = "T0.5_G1.5"

# Frozen recalibration contract, declared before any Stage B score is inspected.
LEVEL_EDGES = (0.0, 0.10, 0.20, 0.30, 0.45, 0.60, 0.80, 1.10)
OFFSET_GRID = np.round(np.arange(-0.10, 0.100001, 0.0025), 6)
MIN_STRATUM_ROWS = 200


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_fold(fold: str) -> pd.DataFrame:
    frame = pd.read_parquet(PROBE / PARENT_PATHS[fold])
    out = frame.loc[
        :, ["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh"]
    ].copy()
    out["prediction_kwh"] = frame[DEPLOYED].to_numpy(dtype=float)
    out["fold_id"] = fold
    capacity = out["group_id"].map(CAPACITIES_KWH).to_numpy(dtype=float)
    out["capacity_kwh"] = capacity
    out["level_bin"] = np.clip(
        np.digitize(out["prediction_kwh"].to_numpy(dtype=float) / capacity, LEVEL_EDGES) - 1,
        0,
        len(LEVEL_EDGES) - 2,
    )
    return out


def settlement_value(prediction: np.ndarray, actual: np.ndarray, capacity: np.ndarray) -> float:
    """Exact FICR numerator contribution: sum(actual * units) over eligible rows."""
    eligible = actual >= 0.10 * capacity
    if not eligible.any():
        return 0.0
    error = np.abs(prediction[eligible] - actual[eligible]) / capacity[eligible]
    units = np.select([error <= 0.06, error <= 0.08], [4.0, 3.0], default=0.0)
    return float((actual[eligible] * units).sum())


def fit_offsets(frame: pd.DataFrame) -> dict[tuple[int, int], float]:
    """Choose one capacity-relative offset per (group, level bin) maximizing settlement value."""
    offsets: dict[tuple[int, int], float] = {}
    for (group, level), part in frame.groupby(["group_id", "level_bin"], sort=True):
        if len(part) < MIN_STRATUM_ROWS:
            continue
        prediction = part["prediction_kwh"].to_numpy(dtype=float)
        actual = part["actual_kwh"].to_numpy(dtype=float)
        capacity = part["capacity_kwh"].to_numpy(dtype=float)
        values = [
            settlement_value(prediction + delta * capacity, actual, capacity)
            for delta in OFFSET_GRID
        ]
        best = int(np.argmax(values))
        # Deterministic tie-break: prefer the smallest absolute offset among ties.
        top = float(values[best])
        tied = [i for i, value in enumerate(values) if value == top]
        best = min(tied, key=lambda i: (abs(OFFSET_GRID[i]), OFFSET_GRID[i]))
        offsets[(int(group), int(level))] = float(OFFSET_GRID[best])
    return offsets


def apply_offsets(frame: pd.DataFrame, offsets: dict[tuple[int, int], float]) -> pd.DataFrame:
    out = frame.copy()
    delta = np.asarray(
        [
            offsets.get((int(group), int(level)), 0.0)
            for group, level in zip(out["group_id"], out["level_bin"], strict=True)
        ]
    )
    out["prediction_kwh"] = np.maximum(
        out["prediction_kwh"].to_numpy(dtype=float)
        + delta * out["capacity_kwh"].to_numpy(dtype=float),
        0.0,
    )
    return out


def score(frame: pd.DataFrame) -> dict[str, object]:
    scored = frame.loc[:, sorted(METRIC_COLUMNS)].copy()
    result = evaluate_official(scored, CAPACITIES_KWH)
    return {
        "total": result.total,
        "one_minus_nmae": result.one_minus_nmae,
        "ficr": result.ficr,
        "group_ficr": dict(result.group_ficr),
        "tier_counts": {g: dict(c) for g, c in result.settlement_tier_counts.items()},
    }


def main() -> None:
    inputs = {
        PARENT_PATHS[fold]: sha256_file(PROBE / PARENT_PATHS[fold]) for fold in FOLDS
    }
    folds = {fold: load_fold(fold) for fold in FOLDS}

    baseline = {fold: score(folds[fold]) for fold in FOLDS}
    baseline["pooled"] = score(pd.concat(list(folds.values()), ignore_index=True))

    # Same-fold oracle: fit and apply on the identical rows (upper bound, not promotable).
    oracle_frames = {
        fold: apply_offsets(folds[fold], fit_offsets(folds[fold])) for fold in FOLDS
    }
    oracle = {fold: score(oracle_frames[fold]) for fold in FOLDS}
    oracle["pooled"] = score(pd.concat(list(oracle_frames.values()), ignore_index=True))

    # Strict prequential: Q2 stays exact parent; Q3 uses Q2; Q4 uses Q2+Q3.
    prequential_frames = {FOLDS[0]: folds[FOLDS[0]]}
    fitted_offsets: dict[str, dict[str, float]] = {}
    for index in (1, 2):
        history = pd.concat([folds[f] for f in FOLDS[:index]], ignore_index=True)
        offsets = fit_offsets(history)
        fitted_offsets[FOLDS[index]] = {
            f"g{group}_L{level}": value for (group, level), value in sorted(offsets.items())
        }
        prequential_frames[FOLDS[index]] = apply_offsets(folds[FOLDS[index]], offsets)
    prequential = {fold: score(prequential_frames[fold]) for fold in FOLDS}
    prequential["pooled"] = score(
        pd.concat(list(prequential_frames.values()), ignore_index=True)
    )

    lines: list[str] = []
    lines.append("# M269 T1 Stage B — 조건부 편향(보정) 여유 측정\n")
    lines.append(f"- 기준 정책: 동결 배포 챔피언 `{DEPLOYED}` (Stage A에서 T/G 최적 확인)")
    lines.append(f"- 층화: group x 예측수준 {len(LEVEL_EDGES) - 1}구간 (경계 {list(LEVEL_EDGES)})")
    lines.append(
        f"- 오프셋 격자: {OFFSET_GRID[0]:+.4f} ~ {OFFSET_GRID[-1]:+.4f} 용량비, "
        f"{len(OFFSET_GRID)}점 / 최소 지지 {MIN_STRATUM_ROWS}행"
    )
    lines.append("- 모델 적합 없음, 2024 접근 없음, 신규 피처 없음, 제출물 없음\n")

    lines.append("## 1. FICR 비교\n")
    lines.append(
        "| fold | 기준 | 동일fold오라클 | 시간순안전 | 오라클이득 | 안전이득 |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")
    for fold in (*FOLDS, "pooled"):
        base = baseline[fold]["ficr"]
        orc = oracle[fold]["ficr"]
        pre = prequential[fold]["ficr"]
        lines.append(
            f"| {fold} | {base:.6f} | {orc:.6f} | {pre:.6f} | "
            f"{orc - base:+.6f} | {pre - base:+.6f} |"
        )

    lines.append("\n## 2. Total 비교\n")
    lines.append("| fold | 기준(배포) | 동일fold 오라클 | 시간순안전 | 오라클 이득 | 안전 이득 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for fold in (*FOLDS, "pooled"):
        base = baseline[fold]["total"]
        orc = oracle[fold]["total"]
        pre = prequential[fold]["total"]
        lines.append(
            f"| {fold} | {base:.6f} | {orc:.6f} | {pre:.6f} | "
            f"{orc - base:+.6f} | {pre - base:+.6f} |"
        )

    lines.append("\n## 3. 적합된 시간순안전 오프셋 (용량비)\n")
    for fold, offsets in fitted_offsets.items():
        rendered = ", ".join(f"`{k}={v:+.4f}`" for k, v in offsets.items())
        lines.append(f"- **{fold}**: {rendered}")

    lines.append("\n## 4. 정산 tier 분포 (pooled)\n")
    lines.append("| 프로토콜 | g1 4/3/0 | g2 4/3/0 | g3 4/3/0 |")
    lines.append("|---|---|---|---|")
    for label, block in (("기준", baseline), ("오라클", oracle), ("시간순안전", prequential)):
        tiers = block["pooled"]["tier_counts"]
        cells = [
            f"{tiers[g]['unit_4']}/{tiers[g]['unit_3']}/{tiers[g]['unit_0']}"
            for g in (1, 2, 3)
        ]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    (REPORTS / "m269_stage_b_recalibration_corrected.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    receipt = {
        "schema_version": 1,
        "stage": "M269_T1_STAGE_B_CONDITIONAL_BIAS_HEADROOM",
        "base_policy": DEPLOYED,
        "level_edges": list(LEVEL_EDGES),
        "offset_grid_points": len(OFFSET_GRID),
        "min_stratum_rows": MIN_STRATUM_ROWS,
        "inputs_sha256": inputs,
        "baseline": baseline,
        "same_fold_oracle": oracle,
        "strict_prequential": prequential,
        "fitted_prequential_offsets": fitted_offsets,
        "model_fits": 0,
        "new_2024_evaluation": False,
        "lockbox_reopened": False,
        "dacon_upload": False,
        "external_actions": [],
    }
    (REPORTS / "m269_stage_b_recalibration_corrected_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=1, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

    for fold in (*FOLDS, "pooled"):
        print(
            f"{fold:>12} base FICR={baseline[fold]['ficr']:.6f} | "
            f"oracle={oracle[fold]['ficr']:.6f} "
            f"({oracle[fold]['ficr'] - baseline[fold]['ficr']:+.6f}) | "
            f"prequential={prequential[fold]['ficr']:.6f} "
            f"({prequential[fold]['ficr'] - baseline[fold]['ficr']:+.6f})"
        )


if __name__ == "__main__":
    main()
