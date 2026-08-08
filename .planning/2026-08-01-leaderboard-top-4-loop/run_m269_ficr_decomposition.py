"""M269 T1 Stage A: decision-layer FICR loss from archived M102 policy artifacts.

Read-only diagnostic. Scores every archived (temperature, gamma) policy on the exact
official metric to separate deployed-policy loss from exact-metric-policy loss and to
expose the calibration shape of the 46-bin conditional distribution.

No model is fitted, no 2024 row is read, no submission is built, no external action occurs.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd

from baram.constants import CAPACITIES_KWH, METRIC_COLUMNS
from baram.evaluation.official import evaluate_official

ROOT = Path(__file__).resolve().parents[2]
PROBE = ROOT / "artifacts" / "backtests" / "metric-aligned-probe"
REPORTS = ROOT / "reports"

MODEL = "M102_TOP100"
FOLDS = ("dev-2023-Q2", "dev-2023-Q3", "dev-2023-Q4")

# Exact champion parent mapping, copied from build_strict_feature_champion.py:14-16.
# The M103/M107 lineage uses the I60 iteration variant on Q3, not the plain surface.
PARENT_PATHS = {
    fold: f"M269_PROBE_TOP100-{fold}-policies.parquet" for fold in FOLDS
}

DEPLOYED = "T0.5_G1.5"          # frozen champion policy (ACTION_TEMPERATURE=0.5, ACTION_GAMMA=1.5)
EXACT_METRIC = "T1_G1"          # official objective: no sharpening, unit FICR coefficient
POLICY_RE = re.compile(r"^T(?P<t>[\d.]+)_G(?P<g>[\d.]+)$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def policy_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if POLICY_RE.match(column)]


def score_policy(frame: pd.DataFrame, policy: str) -> dict[str, object]:
    scored = frame.loc[:, ["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh"]].copy()
    scored["prediction_kwh"] = frame[policy].to_numpy(dtype=float)
    if set(scored.columns) != METRIC_COLUMNS:
        raise RuntimeError("metric frame columns diverged from the official contract")
    result = evaluate_official(scored, CAPACITIES_KWH)
    return {
        "policy": policy,
        "total": result.total,
        "one_minus_nmae": result.one_minus_nmae,
        "ficr": result.ficr,
        "group_ficr": dict(result.group_ficr),
        "valid_rows": dict(result.valid_rows),
        "tier_counts": {
            group: dict(counts) for group, counts in result.settlement_tier_counts.items()
        },
    }


def main() -> None:
    inputs: dict[str, str] = {}
    per_fold: dict[str, pd.DataFrame] = {}
    for fold in FOLDS:
        path = PROBE / PARENT_PATHS[fold]
        inputs[path.name] = sha256_file(path)
        per_fold[fold] = pd.read_parquet(path)

    columns = policy_columns(per_fold[FOLDS[0]])
    for fold, frame in per_fold.items():
        if policy_columns(frame) != columns:
            raise RuntimeError(f"policy grid diverged on {fold}")
    if DEPLOYED not in columns or EXACT_METRIC not in columns:
        raise RuntimeError("required policies are absent from the archived grid")

    pooled_frame = pd.concat(list(per_fold.values()), ignore_index=True)
    surfaces: dict[str, list[dict[str, object]]] = {}
    for fold in (*FOLDS, "pooled"):
        frame = pooled_frame if fold == "pooled" else per_fold[fold]
        surfaces[fold] = [score_policy(frame, policy) for policy in columns]

    report: list[str] = []
    report.append("# M269 T1 Stage A — 결정계층 FICR 손실 분해 (아카이브 전용, 재적합 없음)\n")
    report.append(f"- 모델 계보: `{MODEL}` (46-bin, class width 0.02, 100 frozen features)")
    report.append(f"- 배포 정책: `{DEPLOYED}` / 정확-지표 정책: `{EXACT_METRIC}`")
    report.append(f"- 정책 격자: {len(columns)}개 (7 temperature x 9 gamma)")
    report.append("- 2024 접근 없음, 모델 적합 없음, 제출물 없음, 외부 행위 없음\n")

    summary: dict[str, dict[str, object]] = {}
    report.append("## 1. 배포 vs 정확-지표 vs 동일-fold 최적\n")
    report.append(
        "| fold | 정책 | Total | 1-NMAE | FICR | g1 FICR | g2 FICR | g3 FICR |"
    )
    report.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for fold in (*FOLDS, "pooled"):
        rows = {row["policy"]: row for row in surfaces[fold]}
        best = max(surfaces[fold], key=lambda row: row["total"])
        best_ficr = max(surfaces[fold], key=lambda row: row["ficr"])
        summary[fold] = {
            "deployed": rows[DEPLOYED],
            "exact_metric": rows[EXACT_METRIC],
            "best_total_same_fold": best,
            "best_ficr_same_fold": best_ficr,
        }
        for label, row in (
            ("배포", rows[DEPLOYED]),
            ("정확지표", rows[EXACT_METRIC]),
            (f"최적Total({best['policy']})", best),
            (f"최적FICR({best_ficr['policy']})", best_ficr),
        ):
            groups = row["group_ficr"]
            report.append(
                f"| {fold} | {label} | {row['total']:.6f} | {row['one_minus_nmae']:.6f} | "
                f"{row['ficr']:.6f} | {groups[1]:.6f} | {groups[2]:.6f} | {groups[3]:.6f} |"
            )

    report.append("\n## 2. 성분 (a) 결정 손실\n")
    report.append(
        "| fold | a2 = (정확지표 - 배포) FICR | a1 = (동일fold최적 - 배포) FICR | a1 Total |"
    )
    report.append("|---|---:|---:|---:|")
    for fold in (*FOLDS, "pooled"):
        deployed = summary[fold]["deployed"]
        exact = summary[fold]["exact_metric"]
        best_ficr = summary[fold]["best_ficr_same_fold"]
        best_total = summary[fold]["best_total_same_fold"]
        report.append(
            f"| {fold} | {exact['ficr'] - deployed['ficr']:+.6f} | "
            f"{best_ficr['ficr'] - deployed['ficr']:+.6f} | "
            f"{best_total['total'] - deployed['total']:+.6f} |"
        )

    report.append("\n## 3. 보정 형상 진단 — pooled Total 표면 (행 = temperature, 열 = gamma)\n")
    pooled_rows = {row["policy"]: row for row in surfaces["pooled"]}
    temperatures = sorted(
        {POLICY_RE.match(c).group("t") for c in columns}, key=float, reverse=True
    )
    gammas = sorted({POLICY_RE.match(c).group("g") for c in columns}, key=float)
    report.append("| T \\ G | " + " | ".join(gammas) + " |")
    report.append("|---" * (len(gammas) + 1) + "|")
    for temperature in temperatures:
        cells = []
        for gamma in gammas:
            key = f"T{temperature}_G{gamma}"
            cells.append(f"{pooled_rows[key]['total']:.5f}" if key in pooled_rows else "—")
        report.append(f"| **T={temperature}** | " + " | ".join(cells) + " |")

    report.append("\n## 4. pooled FICR 표면 (행 = temperature, 열 = gamma)\n")
    report.append("| T \\ G | " + " | ".join(gammas) + " |")
    report.append("|---" * (len(gammas) + 1) + "|")
    for temperature in temperatures:
        cells = []
        for gamma in gammas:
            key = f"T{temperature}_G{gamma}"
            cells.append(f"{pooled_rows[key]['ficr']:.5f}" if key in pooled_rows else "—")
        report.append(f"| **T={temperature}** | " + " | ".join(cells) + " |")

    report.append("\n## 5. 정산 tier 분포 (pooled)\n")
    report.append("| 정책 | g1 4/3/0 | g2 4/3/0 | g3 4/3/0 |")
    report.append("|---|---|---|---|")
    for label, policy in (("배포", DEPLOYED), ("정확지표", EXACT_METRIC)):
        tiers = pooled_rows[policy]["tier_counts"]
        cells = [
            f"{tiers[g]['unit_4']}/{tiers[g]['unit_3']}/{tiers[g]['unit_0']}" for g in (1, 2, 3)
        ]
        report.append(f"| {label} | " + " | ".join(cells) + " |")

    (REPORTS / "m269_ficr_decomposition_corrected.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )

    receipt = {
        "schema_version": 1,
        "stage": "M269_T1_STAGE_A_DECISION_LOSS",
        "model_lineage": MODEL,
        "folds": list(FOLDS),
        "deployed_policy": DEPLOYED,
        "exact_metric_policy": EXACT_METRIC,
        "policy_grid_size": len(columns),
        "inputs_sha256": inputs,
        "summary": summary,
        "model_fits": 0,
        "score_calls_on_2024": 0,
        "new_2024_evaluation": False,
        "lockbox_reopened": False,
        "dacon_upload": False,
        "external_actions": [],
    }
    (REPORTS / "m269_ficr_decomposition_corrected_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=1, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

    for fold in (*FOLDS, "pooled"):
        deployed = summary[fold]["deployed"]
        exact = summary[fold]["exact_metric"]
        best_ficr = summary[fold]["best_ficr_same_fold"]
        print(
            f"{fold:>12} deployed Total={deployed['total']:.6f} FICR={deployed['ficr']:.6f} | "
            f"exact Total={exact['total']:.6f} FICR={exact['ficr']:.6f} | "
            f"bestFICR={best_ficr['policy']} {best_ficr['ficr']:.6f}"
        )


if __name__ == "__main__":
    main()
