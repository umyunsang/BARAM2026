"""M271 P4 사이클 19 — 답이 이미 4 개 모델 안에 있는가 (봉투 오라클).

사이클 18 이 결정층을 이중으로 닫았다. 결정층 오라클은 격차의 8.8%, 그중 일반화 가능한
조건화는 16.5% — 곱하면 Total `0.00045` 로 게이트 검출 문턱 `0.001013` 보다도 작다.
남은 격차 `0.0257` 은 결정층에서 나올 수 없다.

그러면 질문이 하나로 좁혀진다. **4 개 모델이 이미 답을 담고 있는데 못 꺼내는 것인가,
애초에 담고 있지 않은가.** 이것은 어떤 결합자를 고르느냐의 문제가 아니라, 결합이라는
연산 자체의 천장을 묻는 것이다.

① 방법 리서치 (실행 전)
  - Talagrand 순위 히스토그램 — 실측이 정렬된 멤버들 사이 어느 순위에 떨어지는가.
    보정된 앙상블이면 평탄, 과소분산이면 U 자.
  - Hamill (2001), *Mon. Wea. Rev.* 129(3):550 — **집계 히스토그램은 조건부 편향을
    가린다.** U 자는 과소분산의 징표로 통용되지만 조건부 편향으로도 생긴다. 평탄해도
    보정을 뜻하지 않는다.
  - 따라서 채택: (a) 집계 + **층화**(그룹 x 예측대역) 히스토그램, (b) **이상치 대칭성**
    으로 과소분산과 조건부 편향을 가른다. 기각: 집계 히스토그램 단독 판정.

② 사양 동결

  **봉투 오라클** — 실측이 멤버 최소~최대 안에 있으면 어떤 행별 볼록결합이 정확히
  맞출 수 있으므로 오차 0(정산단위 4), 밖이면 최선은 가장 가까운 멤버다. 이 오라클
  점수가 **4 개 모델의 어떤 행별 볼록결합으로도 넘을 수 없는 천장**이다. 라벨을 쓰므로
  후보가 아니라 천장 측정이다.

  봉투 둘을 잰다: 멤버 4 개, 그리고 4 개 + 배포정책(승격 후보가 실제로 섞는 5 개).

  사전확약(실행 전 동결):
    H1  커버리지(실측이 봉투 안) **>= 0.50**.
    H2  4 멤버 봉투 오라클 Total **>= 0.66**. 성립하면 목표는 이미 이 4 개 안에 있고
        남은 일은 **추출**이다. 기각되면 어떤 결합으로도 목표에 못 간다.
    H3  이상치가 **비대칭**이다 (`|P(rank=0) - P(rank=K)| >= 0.10`). 성립하면 조건부
        편향, 기각되면 순수 과소분산 — 처방이 완전히 다르다.
    H4  층화해도 커버리지 부호가 유지된다 (모든 굵은 셀에서 커버리지 >= 0.30).
        Hamill 의 경고에 대한 대조.

**게이트를 수정하지 않는다. 정책도 바꾸지 않는다.** 천장 측정만 한다.

읽기 전용. 모델을 적합하지 않고 2024 행을 읽지 않는다.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from m270_monthly_validation import load_predictions
from m271_cycle13_ensemble import ENSEMBLES
from m271_cycle17_combiner import combine, stack_members
from m271_evaluate_candidate import official
from m271_n0_deficit_init import Y_BAND_EDGES

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
REPORT_MD = REPORTS / "m271_cycle19_envelope.md"
RECEIPT = REPORTS / "m271_cycle19_envelope_receipt.json"

NODE_ID = "C1N19_ENVELOPE_ORACLE"
LANE = "L5"  # 예측성능 우수성 — 천장 측정
PARENT_NODE = "C1N18_SPREAD_SKILL"
BASE_ENSEMBLE = "E3_FOUR_FAMILY"
DEPLOYED = "T0.5_G1.5"
TARGET = 0.66

H1_MIN_COVERAGE = 0.50
H2_MIN_ORACLE_TOTAL = 0.66
H3_MIN_OUTLIER_ASYMMETRY = 0.10
H4_MIN_CELL_COVERAGE = 0.30
CELL_MIN_ROWS = 100

METHOD_SOURCES = (
    {
        "id": "talagrand_rank_histogram",
        "cite": "Talagrand 순위 히스토그램 (Talagrand 1997/1999)",
        "claim": "실측의 앙상블 내 순위 분포로 분산 보정을 진단한다. 과소분산이면 U 자",
        "applicability": "directly_supported",
        "use": "커버리지와 이상치 비율의 표준 정의",
    },
    {
        "id": "hamill_2001",
        "cite": "Hamill (2001), Mon. Wea. Rev. 129(3):550 — Interpretation of Rank Histograms",
        "claim": "집계 히스토그램은 조건부 편향을 가린다. U 자는 과소분산과 조건부 편향에 "
                 "모두서 나오고, 평탄해도 보정을 뜻하지 않는다",
        "applicability": "directly_supported",
        "use": "층화 히스토그램(H4)과 이상치 대칭성 검사(H3)를 추가하는 근거",
    },
)


def envelope_oracle(members: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """행별 볼록결합의 천장. 봉투 안이면 실측, 밖이면 가장 가까운 경계."""
    lo = members.min(axis=1)
    hi = members.max(axis=1)
    return np.clip(actual, lo, hi)


def score_frame(base: pd.DataFrame, prediction: np.ndarray) -> dict[str, Any]:
    frame = base.loc[
        :, ["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh"]
    ].copy()
    frame["prediction_kwh"] = prediction
    return official(frame)


def main() -> int:
    member_names = ENSEMBLES[BASE_ENSEMBLE]
    k = len(member_names)
    stacked = stack_members(member_names)

    cap = stacked["capacity"].to_numpy(dtype="float64")
    actual = stacked["actual_kwh"].to_numpy(dtype="float64")
    arr4 = stacked.loc[:, [f"m{i}" for i in range(k)]].to_numpy(dtype="float64")

    # 배포 정책을 5 번째 멤버로. 승격 후보가 실제로 섞는 집합이다.
    keys = ["forecast_id", "forecast_kst_dtm", "group_id"]
    deployed = load_predictions(DEPLOYED).loc[:, [*keys, "prediction_kwh"]].rename(
        columns={"prediction_kwh": "deployed"}
    )
    merged = stacked.merge(deployed, on=keys, how="inner")
    assert len(merged) == len(stacked), "배포 정책 조인에서 행이 유실됐다"
    arr5 = merged.loc[:, [*[f"m{i}" for i in range(k)], "deployed"]].to_numpy(dtype="float64")

    current = combine(stacked, k, "median")["prediction_kwh"].to_numpy(dtype="float64")
    current_score = score_frame(stacked, current)

    envelopes = {}
    for label, arr in (("members_4", arr4), ("members_4_plus_deployed", arr5)):
        oracle = envelope_oracle(arr, actual)
        score = score_frame(stacked, oracle)
        lo, hi = arr.min(axis=1), arr.max(axis=1)
        inside = (actual >= lo) & (actual <= hi)
        eligible = actual >= 0.10 * cap
        envelopes[label] = {
            "n_members": arr.shape[1],
            **score,
            "coverage_all_rows": float(inside.mean()),
            "coverage_eligible": float(inside[eligible].mean()),
            "median_envelope_width_cap": float(np.median(((hi - lo) / cap)[eligible])),
            "headroom_vs_current": score["total"] - current_score["total"],
            "reaches_target": bool(score["total"] >= TARGET),
        }

    # --- 순위 히스토그램 (4 멤버 -> 순위 0..4)
    eligible = actual >= 0.10 * cap
    ranks = (arr4 < actual[:, None]).sum(axis=1)
    ranks_e = ranks[eligible]
    hist = [float((ranks_e == r).mean()) for r in range(k + 1)]
    p_below, p_above = hist[0], hist[k]
    coverage = float(1.0 - p_below - p_above)
    asymmetry = float(abs(p_below - p_above))
    uniform = 1.0 / (k + 1)

    h1 = bool(coverage >= H1_MIN_COVERAGE)
    h2 = bool(envelopes["members_4"]["total"] >= H2_MIN_ORACLE_TOTAL)
    h3 = bool(asymmetry >= H3_MIN_OUTLIER_ASYMMETRY)

    # --- H4 층화 (Hamill 경고 대조)
    strat = pd.DataFrame(
        {
            "group_id": stacked["group_id"].to_numpy(),
            "pred_band": pd.cut(
                pd.Series(current / cap), bins=list(Y_BAND_EDGES), right=True
            ).astype(str),
            "rank": ranks,
            "inside": (ranks > 0) & (ranks < k),
        }
    ).loc[eligible]
    cells = []
    for (g_id, band), cell in strat.groupby(["group_id", "pred_band"], observed=True):
        if len(cell) < CELL_MIN_ROWS:
            continue
        cells.append(
            {
                "group": int(g_id),
                "pred_band": band,
                "rows": len(cell),
                "coverage": float(cell["inside"].mean()),
                "p_below": float((cell["rank"] == 0).mean()),
                "p_above": float((cell["rank"] == k).mean()),
            }
        )
    h4 = bool(cells) and all(c["coverage"] >= H4_MIN_CELL_COVERAGE for c in cells)

    if h2:
        verdict = "ANSWER_INSIDE_ENSEMBLE_EXTRACTION_AXIS_OPENS"
    elif envelopes["members_4_plus_deployed"]["reaches_target"]:
        verdict = "ANSWER_INSIDE_ONLY_WITH_DEPLOYED_MEMBER"
    else:
        verdict = "ANSWER_NOT_INSIDE_ENSEMBLE_NEED_NEW_BASE_MODEL"

    check = {
        "H1_expectation": f"커버리지 >= {H1_MIN_COVERAGE:.2f}",
        "H1_held": h1,
        "H1_measured": coverage,
        "H2_expectation": f"4 멤버 봉투 오라클 Total >= {H2_MIN_ORACLE_TOTAL}",
        "H2_held": h2,
        "H2_measured": envelopes["members_4"]["total"],
        "H3_expectation": f"이상치 비대칭 >= {H3_MIN_OUTLIER_ASYMMETRY:.2f}",
        "H3_held": h3,
        "H3_measured": asymmetry,
        "H3_reading": "조건부 편향" if h3 else "순수 과소분산",
        "H4_expectation": f"모든 굵은 셀에서 커버리지 >= {H4_MIN_CELL_COVERAGE:.2f}",
        "H4_held": h4,
        "verdict": verdict,
    }

    payload = {
        "node": NODE_ID,
        "parent_node": PARENT_NODE,
        "gate_modified": False,
        "policy_modified": False,
        "uses_labels": True,
        "is_oracle_not_candidate": True,
        "members": list(member_names),
        "method_sources": list(METHOD_SOURCES),
        "current_combiner": {"operator": "median", **current_score},
        "target": TARGET,
        "envelopes": envelopes,
        "rank_histogram": {
            "bins": hist,
            "uniform_expectation": uniform,
            "p_below_all": p_below,
            "p_above_all": p_above,
            "coverage": coverage,
            "outlier_asymmetry": asymmetry,
        },
        "stratified_cells": cells,
        "predeclared_check": check,
    }

    lines = [
        "# M271 P4 사이클 19 — 답이 이미 4 개 모델 안에 있는가 (봉투 오라클)",
        "",
        f"- 판정일: {datetime.now(UTC).strftime('%Y-%m-%d')} (UTC)",
        f"- 노드: `{NODE_ID}` / 레인 {LANE} / 부모 `{PARENT_NODE}`",
        "- **라벨을 쓴다. 후보가 아니라 천장 측정이다.** 게이트·정책 미변경",
        "",
        "## 0. 방법 리서치 (실행 전)",
        "",
    ]
    for s in METHOD_SOURCES:
        lines.append(f"- **{s['cite']}** (`{s['applicability']}`)")
        lines.append(f"  - {s['claim']}")
        lines.append(f"  - 사용: {s['use']}")

    lines += [
        "",
        "## 1. 봉투 오라클 — 행별 볼록결합의 천장",
        "",
        "실측이 멤버 최소~최대 안이면 어떤 행별 볼록결합이 정확히 맞출 수 있다. 밖이면",
        "최선은 가장 가까운 경계다. **이 점수를 넘는 결합자는 존재하지 않는다.**",
        "",
        "| 봉투 | 멤버 | 커버리지(유효행) | 봉투폭 중앙값 | **오라클 Total** | 1-NMAE | FICR "
        "| 현재대비 | 목표도달 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|:---:|",
        f"| 현재 결합자 `median` | {k} | — | — | **{current_score['total']:.6f}** | "
        f"{current_score['one_minus_nmae']:.6f} | {current_score['ficr']:.6f} | — | "
        f"{'O' if current_score['total'] >= TARGET else 'X'} |",
    ]
    for label, e in envelopes.items():
        lines.append(
            f"| `{label}` | {e['n_members']} | {e['coverage_eligible']:.4f} | "
            f"{e['median_envelope_width_cap']:.4f} | **{e['total']:.6f}** | "
            f"{e['one_minus_nmae']:.6f} | {e['ficr']:.6f} | "
            f"{e['headroom_vs_current']:+.6f} | {'**O**' if e['reaches_target'] else 'X'} |"
        )

    lines += [
        "",
        "## 2. 순위 히스토그램 (H1 · H3)",
        "",
        f"멤버 {k} 개 -> 순위 {k + 1} 칸. 평탄하면 각 칸 {uniform:.4f}.",
        "",
        "| 순위 | 뜻 | 비율 | 평탄대비 |",
        "|---:|---|---:|---:|",
    ]
    meaning = ["실측이 **모든 멤버 아래**"] + [
        f"멤버 {r} 개 아래" for r in range(1, k)
    ] + ["실측이 **모든 멤버 위**"]
    for r, (h, m) in enumerate(zip(hist, meaning, strict=True)):
        lines.append(f"| {r} | {m} | {h:.4f} | {h / uniform:.2f}x |")
    lines += [
        "",
        f"커버리지 **{coverage:.4f}**, 아래로 이탈 {p_below:.4f}, 위로 이탈 {p_above:.4f},",
        f"이상치 비대칭 **{asymmetry:.4f}** -> 읽기: **{check['H3_reading']}**.",
        "",
        "## 3. 층화 커버리지 (H4 — Hamill 경고 대조)",
        "",
        "| 그룹 | 예측대역 | 행 | 커버리지 | 아래이탈 | 위이탈 |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for c in sorted(cells, key=lambda x: -x["rows"])[:14]:
        lines.append(
            f"| {c['group']} | {c['pred_band']} | {c['rows']:,} | **{c['coverage']:.4f}** | "
            f"{c['p_below']:.4f} | {c['p_above']:.4f} |"
        )

    lines += [
        "",
        "## 4. 사전확약 대조",
        "",
        f"- H1 `{check['H1_expectation']}` -> **{h1}** (실측 {coverage:.4f})",
        f"- H2 `{check['H2_expectation']}` -> **{h2}** "
        f"(실측 {envelopes['members_4']['total']:.6f})",
        f"- H3 `{check['H3_expectation']}` -> **{h3}** (실측 {asymmetry:.4f})",
        f"- H4 `{check['H4_expectation']}` -> **{h4}** ({len(cells)} 셀)",
        "",
        f"판정: **{verdict}**",
        "",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    receipt: dict[str, Any] = {
        "schema_version": 1,
        "stage": "M271_P4_CYCLE19_ENVELOPE",
        "node": NODE_ID,
        "lane": LANE,
        "decided_utc": datetime.now(UTC).isoformat(),
        "result": payload,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "report_sha256": hashlib.sha256(REPORT_MD.read_bytes()).hexdigest(),
        "dacon_upload": False,
        "external_actions": [],
        "model_fits": 0,
        "lockbox_reopened": False,
        "new_2024_evaluation": False,
    }
    RECEIPT.write_text(
        json.dumps(receipt, indent=1, ensure_ascii=False, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

    print(f"[C19] 현재 median 결합자 Total {current_score['total']:.6f}")
    for label, e in envelopes.items():
        print(f"[C19] {label:>24}  오라클 {e['total']:.6f}  "
              f"커버리지 {e['coverage_eligible']:.4f}  "
              f"여유 {e['headroom_vs_current']:+.6f}  "
              f"목표도달 {e['reaches_target']}")
    print(f"[C19] 순위 히스토그램 {[round(h, 4) for h in hist]} (평탄 {uniform:.4f})")
    print(f"[C19] 커버리지 {coverage:.4f} -> H1 {h1} | 비대칭 {asymmetry:.4f} -> H3 {h3} "
          f"({check['H3_reading']})")
    print(f"[C19] 층화 {len(cells)} 셀 -> H4 {h4}")
    print(f"[C19] 판정: {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
