"""M270: re-examine this session's closures against the newly calibrated baseline.

WHY
Every closure this session was argued against "local 0.628, target 0.66, gap 0.032". The
M261 anchor measured the offset directly: online exceeds local strict OOF by `+0.006554`
Total and `+0.012027` FICR. The real online position is `0.6365274` and the real gap is
`0.023473`, about 27% smaller than assumed. Verdicts reached on the old numbers must be
rechecked rather than left standing.

METHOD
Instead of scaling the old figures by hand, this simulates directly: shrink the prediction
error toward the actual by a factor k and score the result with the official metric. That
answers "how much more accurate would forecasts need to be" in the metric's own units.

This is a SENSITIVITY analysis, not an achievable result - it uses the actuals to shrink
error, which no forecaster can do. Its purpose is to size the requirement, then compare
that size against what the source-decomposition showed is achievable.

Read-only. No model is fitted, no 2024 row is read.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from m270_m261_baseline import apply_temporal, load_probe

from baram.constants import CAPACITIES_KWH, METRIC_COLUMNS
from baram.evaluation.official import evaluate_official

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"

# Measured on the M261 anchor, 2026-08-04.
OFFSET_TOTAL = 0.006554
OFFSET_FICR = 0.012027
ONLINE_M261 = {"total": 0.6365274327, "one_minus_nmae": 0.857885373, "ficr": 0.4151694923}
RANK20 = {"total": 0.66010, "one_minus_nmae": 0.87345, "ficr": 0.44675}
TARGET_TOTAL = 0.66

# From the source decomposition: best supplied blend sigma_v against the infinite-source
# floor, i.e. the most any number of equally correlated sources could remove.
SIGMA_BEST = 2.159
SIGMA_FLOOR = 2.059

# From M269 Stage D, on the local surface.
LOCAL_ORACLE_FICR = 0.4227

# Fine near the crossing point; a coarse grid overstates the requirement.
SHRINK_GRID = np.round(
    np.concatenate([np.arange(1.00, 0.879, -0.01), np.arange(0.85, 0.49, -0.05)]), 3
)


def score(frame: pd.DataFrame) -> dict[str, float]:
    scored = frame.loc[:, sorted(METRIC_COLUMNS)].copy()
    result = evaluate_official(scored, CAPACITIES_KWH)
    return {
        "total": result.total,
        "one_minus_nmae": result.one_minus_nmae,
        "ficr": result.ficr,
    }


def main() -> None:
    base = apply_temporal(load_probe())
    actual = base["actual_kwh"].to_numpy(dtype=float)
    predicted = base["prediction_kwh"].to_numpy(dtype=float)

    rows = []
    for k in SHRINK_GRID:
        shrunk = base.copy()
        shrunk["prediction_kwh"] = actual + float(k) * (predicted - actual)
        values = score(shrunk)
        rows.append(
            {
                "k": float(k),
                **values,
                "online_equiv_total": values["total"] + OFFSET_TOTAL,
                "online_equiv_ficr": values["ficr"] + OFFSET_FICR,
            }
        )
    table = pd.DataFrame(rows)

    reach = table.loc[table["online_equiv_total"] >= TARGET_TOTAL]
    k_needed = float(reach["k"].max()) if not reach.empty else float("nan")
    achievable_k = SIGMA_FLOOR / SIGMA_BEST

    lines: list[str] = []
    lines.append("# M270 — 새 기준선에 따른 종결 판정 재검토\n")
    lines.append(
        f"- 측정 오프셋(M261): Total `+{OFFSET_TOTAL:.6f}`, FICR `+{OFFSET_FICR:.6f}`"
    )
    lines.append(f"- 실제 온라인 위치: `{ONLINE_M261['total']:.6f}` / 목표 `{TARGET_TOTAL}`")
    lines.append(
        f"- 실제 격차: **`{TARGET_TOTAL - ONLINE_M261['total']:.6f}`** (기존 가정 `0.032`)\n"
    )

    lines.append("## 1. 오차 축소 민감도 (시뮬레이션)\n")
    lines.append(
        "예측을 실제값 쪽으로 `k`배 수축시킨 뒤 공식 산식으로 채점한다. "
        "**달성 가능한 결과가 아니라 요구량을 재는 자**다.\n"
    )
    lines.append(
        "| k | 로컬 Total | 로컬 FICR | 온라인환산 Total | 온라인환산 FICR |"
    )
    lines.append("|---:|---:|---:|---:|---:|")
    for r in table.itertuples(index=False):
        mark = " **←목표달성**" if r.online_equiv_total >= TARGET_TOTAL else ""
        lines.append(
            f"| {r.k:.3f} | {r.total:.6f} | {r.ficr:.6f} | "
            f"{r.online_equiv_total:.6f} | {r.online_equiv_ficr:.6f} |{mark}"
        )

    lines.append("\n## 2. 요구량 대 달성 가능량\n")
    lines.append("| 항목 | 값 |")
    lines.append("|---|---:|")
    lines.append(f"| 목표 도달에 필요한 오차 배율 k | **{k_needed:.3f}** |")
    lines.append(f"| 필요 오차 감소율 | **{(1 - k_needed) * 100:.1f}%** |")
    lines.append(
        f"| 다중소스로 달성 가능한 배율 (sigma {SIGMA_BEST:.3f} -> {SIGMA_FLOOR:.3f}) | "
        f"{achievable_k:.3f} |"
    )
    lines.append(f"| 달성 가능한 감소율 | {(1 - achievable_k) * 100:.1f}% |")
    lines.append(f"| 배수 부족 | **{(1 - k_needed) / (1 - achievable_k):.1f}x** |")

    lines.append("\n## 3. M269 표현 천장 재검토\n")
    ceiling_online = LOCAL_ORACLE_FICR + OFFSET_FICR
    lines.append("| 항목 | 값 |")
    lines.append("|---|---:|")
    lines.append(f"| 로컬 오라클 천장 FICR | {LOCAL_ORACLE_FICR:.4f} |")
    lines.append(f"| 온라인 환산 천장 | **{ceiling_online:.4f}** |")
    lines.append(f"| rank-20 필요 FICR | {RANK20['ficr']:.5f} |")
    lines.append(f"| 남은 차이 | **{RANK20['ficr'] - ceiling_online:+.4f}** |")
    lines.append(
        "\n로컬 기준으로는 천장이 필요치보다 `0.0240` 낮았으나, 오프셋 반영 시 `"
        f"{RANK20['ficr'] - ceiling_online:.4f}`로 좁혀진다. 판정 유지, 여유는 절반."
    )
    lines.append(
        "\n**중요한 단서**: 오프셋은 *배포 정책*에서 측정됐고 *오라클*에서 측정된 것이 아니다. "
        "오프셋이 정책과 무관하다는 가정은 검증되지 않았으므로 이 환산은 시사적 수치다."
    )

    lines.append("\n## 4. 판정 변화 요약\n")
    lines.append("| 레인 | 기존 근거 | 재검토 후 |")
    lines.append("|---|---|---|")
    lines.append(
        f"| 외부 NWP 소스 | 필요 ~18% vs 가능 4.6% | 필요 **{(1 - k_needed) * 100:.1f}%** vs "
        f"가능 {(1 - achievable_k) * 100:.1f}% -> **유지** |"
    )
    lines.append(
        f"| M269 표현 천장 | 로컬 `0.4227` < 필요 `0.44675` | 온라인 환산 `{ceiling_online:.4f}` "
        "< 필요 -> **유지, 여유 축소** |"
    )
    lines.append("| 후처리·밴드적중·그룹3 보강 | 중복/소진 | 기준선과 무관 -> **유지** |")
    lines.append("| 월 블록 게이트 | 비교 프로토콜 | 기준선과 무관 -> **유지** |")

    (REPORTS / "m270_revised_verdicts.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    receipt = {
        "schema_version": 1,
        "stage": "M270_REVISED_VERDICTS_AFTER_ANCHOR",
        "measured_offset_total": OFFSET_TOTAL,
        "measured_offset_ficr": OFFSET_FICR,
        "online_m261": ONLINE_M261,
        "real_gap_online": TARGET_TOTAL - ONLINE_M261["total"],
        "shrink_table": table.to_dict(orient="records"),
        "k_needed_for_target": k_needed,
        "achievable_k_multisource": achievable_k,
        "shortfall_multiple": (1 - k_needed) / (1 - achievable_k),
        "ceiling_local_ficr": LOCAL_ORACLE_FICR,
        "ceiling_online_equivalent": ceiling_online,
        "ceiling_shortfall_vs_rank20": RANK20["ficr"] - ceiling_online,
        "caveat": "offset measured on the deployed policy, not the oracle; "
                  "policy-independence is unverified. Shrink simulation uses actuals and is "
                  "a requirement-sizing device, not an achievable result.",
        "model_fits": 0,
        "new_2024_evaluation": False,
        "lockbox_reopened": False,
        "dacon_upload": False,
        "external_actions": [],
    }
    (REPORTS / "m270_revised_verdicts_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=1, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

    print(table.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    print(f"\nk needed for target: {k_needed:.3f}  (error reduction {(1 - k_needed) * 100:.1f}%)")
    print(f"achievable k from multi-source: {achievable_k:.3f} "
          f"(reduction {(1 - achievable_k) * 100:.1f}%)")
    print(f"shortfall multiple: {(1 - k_needed) / (1 - achievable_k):.1f}x")
    print(f"ceiling online-equivalent FICR: {ceiling_online:.4f} vs required {RANK20['ficr']:.5f}")


if __name__ == "__main__":
    main()
