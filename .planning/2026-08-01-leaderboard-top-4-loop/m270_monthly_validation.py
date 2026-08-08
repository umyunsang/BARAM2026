"""M270 finding 3: monthly block evaluation for transfer stability.

WHAT THIS FIXES
The project's promotion gate compares a candidate to its parent on two quarterly folds
(Q3 and Q4). Two paired observations cannot separate "the transfer is structurally
unstable" from "the two folds happened to disagree". M269 Stage B is the clearest case:
its chronology-safe recalibration scored `+0.009074` on Q3 and `-0.006229` on Q4, and the
existing gate had no way to interpret that.

WHAT THIS IS
Re-slicing the SAME chronology-safe OOF rows into monthly blocks, which raises paired
transfer observations from 2 to about 9 and makes the delta distribution measurable.

WHAT THIS IS NOT
This is not a rolling-origin refit. The training origin stays quarterly, so the three
months inside one quarter share a model. Calling it "rolling origin" would overclaim.
The question it answers is whether the candidate-minus-parent delta is stable over time,
which is exactly the question the quarterly gate could not answer.

Read-only over persisted predictions. No model is fitted, no 2024 row is read.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from baram.constants import CAPACITIES_KWH, METRIC_COLUMNS
from baram.evaluation.official import evaluate_official

ROOT = Path(__file__).resolve().parents[2]
PROBE = ROOT / "artifacts" / "backtests" / "metric-aligned-probe"
REPORTS = ROOT / "reports"

FOLDS = ("dev-2023-Q2", "dev-2023-Q3", "dev-2023-Q4")
PARENT_PATHS = {fold: f"M269_PROBE_TOP100-{fold}-policies.parquet" for fold in FOLDS}

# Minimum eligible rows per group for a month to be scorable. Below this the official
# metric is too thin to be meaningful and the block is reported as excluded, never
# silently dropped.
MIN_ELIGIBLE_ROWS = 50
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 20260804


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_predictions(policy: str) -> pd.DataFrame:
    """Concatenate the chronology-safe OOF rows for one policy across all dev folds."""
    parts = []
    for fold in FOLDS:
        frame = pd.read_parquet(PROBE / PARENT_PATHS[fold])
        part = frame.loc[:, ["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh"]].copy()
        part["prediction_kwh"] = frame[policy].to_numpy(dtype=float)
        part["fold_id"] = fold
        parts.append(part)
    out = pd.concat(parts, ignore_index=True)
    out["month"] = pd.to_datetime(out["forecast_kst_dtm"]).dt.to_period("M").astype(str)
    return out


def score_frame(frame: pd.DataFrame) -> dict[str, float] | None:
    """Official score for one block, or None when any group is too thin to score."""
    for group, capacity in CAPACITIES_KWH.items():
        part = frame.loc[frame["group_id"].eq(group)]
        if int((part["actual_kwh"] >= 0.10 * capacity).sum()) < MIN_ELIGIBLE_ROWS:
            return None
    scored = frame.loc[:, sorted(METRIC_COLUMNS)].copy()
    result = evaluate_official(scored, CAPACITIES_KWH)
    return {
        "total": result.total,
        "one_minus_nmae": result.one_minus_nmae,
        "ficr": result.ficr,
    }


def monthly_scores(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for month, part in frame.groupby("month", sort=True):
        scored = score_frame(part)
        rows.append(
            {
                "month": month,
                "rows": len(part),
                "scorable": scored is not None,
                **(scored or {"total": np.nan, "one_minus_nmae": np.nan, "ficr": np.nan}),
            }
        )
    return pd.DataFrame(rows)


def paired_monthly_delta(candidate: pd.DataFrame, parent: pd.DataFrame) -> dict[str, object]:
    """Distribution of the candidate-minus-parent delta across monthly blocks."""
    left = monthly_scores(candidate).set_index("month")
    right = monthly_scores(parent).set_index("month")
    shared = [m for m in left.index if m in right.index and left.loc[m, "scorable"]
              and right.loc[m, "scorable"]]
    deltas = {
        metric: np.asarray(
            [float(left.loc[m, metric] - right.loc[m, metric]) for m in shared], dtype=float
        )
        for metric in ("total", "one_minus_nmae", "ficr")
    }
    total_delta = deltas["total"]
    positive = int((total_delta > 0).sum())
    n = len(total_delta)
    sign_test = stats.binomtest(positive, n, 0.5, alternative="greater") if n else None

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    if n:
        draws = rng.choice(total_delta, size=(BOOTSTRAP_DRAWS, n), replace=True).mean(axis=1)
        boot_positive = float((draws > 0).mean())
        boot_q05 = float(np.quantile(draws, 0.05))
    else:
        boot_positive, boot_q05 = float("nan"), float("nan")

    return {
        "months_scored": n,
        "months_excluded": [m for m in left.index if m not in shared],
        "positive_months": positive,
        "positive_fraction": positive / n if n else float("nan"),
        "mean_total_delta": float(total_delta.mean()) if n else float("nan"),
        "median_total_delta": float(np.median(total_delta)) if n else float("nan"),
        "min_total_delta": float(total_delta.min()) if n else float("nan"),
        "max_total_delta": float(total_delta.max()) if n else float("nan"),
        "sign_test_p_greater": float(sign_test.pvalue) if sign_test else float("nan"),
        "block_bootstrap_positive_fraction": boot_positive,
        "block_bootstrap_q05": boot_q05,
        "per_month": {
            m: {metric: float(deltas[metric][i]) for metric in deltas}
            for i, m in enumerate(shared)
        },
    }


def main() -> None:
    deployed = load_predictions("T0.5_G1.5")
    exact = load_predictions("T1_G1")

    per_month = monthly_scores(deployed)
    comparison = paired_monthly_delta(exact, deployed)

    lines: list[str] = []
    lines.append("# M270 발견 3 — 월 블록 전이 안정성 평가\n")
    lines.append("- 목적: 전이 델타 관측을 늘려 안정성을 **측정 가능**하게 만든다")
    lines.append("- 대상: 동일한 시간순 안전 OOF 행을 재분할한 것이며 **재적합이 아니다**")
    lines.append("- 학습 원점은 여전히 분기 단위이므로 한 분기 내 3개월은 같은 모델을 공유한다\n")

    lines.append("## 1. 배포 정책의 월별 점수\n")
    lines.append("| month | rows | 채점가능 | Total | 1-NMAE | FICR |")
    lines.append("|---|---:|---|---:|---:|---:|")
    for row in per_month.itertuples(index=False):
        mark = "예" if row.scorable else "**아니오**"
        if row.scorable:
            lines.append(
                f"| {row.month} | {row.rows} | {mark} | {row.total:.6f} | "
                f"{row.one_minus_nmae:.6f} | {row.ficr:.6f} |"
            )
        else:
            lines.append(f"| {row.month} | {row.rows} | {mark} | — | — | — |")

    lines.append("\n## 2. 프로토콜 자체 검증 — 정확지표 정책 vs 배포 정책\n")
    lines.append(
        "M269 Stage A는 pooled 단일 수치로만 보고했다. 월 블록에서는 분포가 드러난다.\n"
    )
    lines.append(f"- 채점된 월: **{comparison['months_scored']}개** (분기 프로토콜은 2개)")
    lines.append(f"- 제외된 월: {comparison['months_excluded'] or '없음'}")
    lines.append(
        f"- 양의 델타 월: **{comparison['positive_months']}"
        f"/{comparison['months_scored']}**"
    )
    lines.append(f"- 중앙값 Total 델타: `{comparison['median_total_delta']:+.6f}`")
    lines.append(f"- 평균 Total 델타: `{comparison['mean_total_delta']:+.6f}`")
    lines.append(
        f"- 범위: `{comparison['min_total_delta']:+.6f}` ~ `{comparison['max_total_delta']:+.6f}`"
    )
    lines.append(f"- 부호검정 p(단측, 우월): `{comparison['sign_test_p_greater']:.4f}`")
    lines.append(
        f"- 월 블록 부트스트랩 양수 비율: `{comparison['block_bootstrap_positive_fraction']:.4f}`, "
        f"5% 분위: `{comparison['block_bootstrap_q05']:+.6f}`"
    )

    lines.append("\n### 월별 델타\n")
    lines.append("| month | Total | 1-NMAE | FICR |")
    lines.append("|---|---:|---:|---:|")
    for month, values in comparison["per_month"].items():  # type: ignore[union-attr]
        lines.append(
            f"| {month} | {values['total']:+.6f} | {values['one_minus_nmae']:+.6f} | "
            f"{values['ficr']:+.6f} |"
        )

    lines.append("\n## 3. 제안 게이트 (예고·수용 전까지 확정 아님)\n")
    lines.append(
        "분기 게이트 `Q3/Q4 둘 다 양수 + Q4 bootstrap > 0.50`은 2-표본 검정이다. 월 블록에서는 "
        "델타 **분포**를 직접 쓸 수 있다. 다만 임계값을 결과를 본 뒤 고르면 이 프로젝트가 "
        "반복적으로 경계해 온 실패를 되풀이한다. 아래는 **제안이며 수용 전까지 확정이 아니다.**"
    )
    lines.append("")
    lines.append("- 양의 월 비율이 사전 지정 수준 이상")
    lines.append("- 중앙값 Total 델타가 양수")
    lines.append("- 월 블록 부트스트랩 5% 분위가 양수")
    lines.append("- 최악 월 델타가 사전 지정 하한 이상 (붕괴 방지)")

    (REPORTS / "m270_monthly_validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    receipt = {
        "schema_version": 1,
        "stage": "M270_FINDING3_MONTHLY_BLOCK_VALIDATION",
        "is_refit": False,
        "training_origin": "quarterly (unchanged)",
        "min_eligible_rows_per_group": MIN_ELIGIBLE_ROWS,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "inputs_sha256": {p: sha256_file(PROBE / p) for p in PARENT_PATHS.values()},
        "deployed_monthly": per_month.to_dict(orient="records"),
        "exact_metric_vs_deployed": comparison,
        "model_fits": 0,
        "new_2024_evaluation": False,
        "lockbox_reopened": False,
        "dacon_upload": False,
        "external_actions": [],
    }
    (REPORTS / "m270_monthly_validation_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=1, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

    print(f"months scored: {comparison['months_scored']} (quarterly protocol gives 2)")
    print(f"excluded: {comparison['months_excluded']}")
    print(
        f"positive {comparison['positive_months']}/{comparison['months_scored']} "
        f"median={comparison['median_total_delta']:+.6f} "
        f"range=[{comparison['min_total_delta']:+.6f}, {comparison['max_total_delta']:+.6f}]"
    )
    print(
        f"sign-test p={comparison['sign_test_p_greater']:.4f} "
        f"boot_positive={comparison['block_bootstrap_positive_fraction']:.4f} "
        f"boot_q05={comparison['block_bootstrap_q05']:+.6f}"
    )


if __name__ == "__main__":
    main()
