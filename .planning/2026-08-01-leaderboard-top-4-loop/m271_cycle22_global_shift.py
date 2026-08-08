"""M271 P4 사이클 22 — 사이클 21 실패의 기전과, 층화 없는 전역 이동.

사이클 21 이 크게 실패했다. `median_shift` 는 Total 을 `-0.0636` 깎았는데 층별 이동의
부호 일치율은 **1.000** 이었다. 세 분할에서 완벽히 재현되는 신호를 지웠는데 점수가 무너진
것이므로, 그 신호가 편향이 아니었다는 뜻이다.

가설: **평균회귀(회귀희석)**. 층을 예측대역으로 잡으면 예측이 높은 층에서 실측은 평균
쪽으로 낮게, 낮은 층에서는 높게 나온다. 층내 잔차 중앙값의 체계적 부호는 편향의 증거가
아니라 조건화의 산물이다. 이것을 "보정" 하면 예측을 전역 평균으로 끌어당겨 예보를 무디게
만든다 — 사이클 11 이 이미 무딘 표현의 FICR 손실을 쟀다(-0.0139).

이 노드는 두 가지를 한다.
  A. 그 기전을 **기계검증 가능한 서명**으로 확인해 사이클 21 의 폐기에 근거를 붙인다.
  B. 사이클 19 가 잰 **무조건부** 과대예측(57.8%)을 층화 없이 확인한다. 그룹당 스칼라
     하나면 평균회귀가 개입할 여지가 없다.

① 방법 리서치 (실행 전)
  - 새 방법 리서치 없음. 사이클 21 의 MOS 근거(Glahn & Lowry 1972)를 그대로 쓰되
    **층화를 제거한** 최단순형으로 되돌린다. 층화가 실패 원인이라는 가설을 시험하는 것이
    이 노드의 일이므로, 같은 방법을 층화 유무로만 갈라 비교하는 것이 옳은 설계다.
  - 평균회귀의 진단 서명은 고전적이다: 조건화 변수에 대해 잔차 이동이 **단조 감소**하고,
    보정이 예측 분포를 **압축**한다. 둘 다 직접 잰다.

② 사양 동결

  사전확약(실행 전 동결):
    H1  (기전) 층별 이동이 예측대역 순서에 대해 **단조 감소**한다.
        세 그룹 모두 Spearman <= -0.80.
    H2  (기전) 사이클 21 보정 후 예측 분포의 표준편차가 **감소**한다(압축).
        세 그룹 모두에서 감소.
    H3  (전역) 그룹별 전역 이동의 부호가 세 LOO 분할에서 일치한다.
    H4  (전역) 전역 이동 보정이 `M271_MEDIAN4` 대비 Total 을 개선하고 **동결 게이트를
        통과**한다.

  H1·H2 가 성립하면 사이클 21 의 폐기 전제는 "예측대역 조건화는 평균회귀를 편향으로
  오인한다" 로 확정되고, 이는 기계검증 가능한 술어다.
  H4 가 기각되면 편향보정 계열 전체가 닫힌다 — 층화형도 전역형도 듣지 않는 것이므로.

**게이트를 수정하지 않는다.** 읽기만 한다.

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
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))

from m270_gate import GATE_VERSION, evaluate_gate
from m271_cycle13_ensemble import ENSEMBLES
from m271_cycle17_combiner import combine, stack_members
from m271_cycle21_mos import (
    FOLDS,
    QUARTER_OF_MONTH,
    apply_shifts,
    estimate_shifts,
)
from m271_evaluate_candidate import official
from m271_n0_deficit_init import Y_BAND_EDGES

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
REPORT_MD = REPORTS / "m271_cycle22_global_shift.md"
RECEIPT = REPORTS / "m271_cycle22_global_shift_receipt.json"

NODE_ID = "C1N22_GLOBAL_SHIFT"
LANE = "L1"
PARENT_NODE = "C1N21_MOS_BIAS_CORRECTION"
CLOSES = "C1N21_MOS_BIAS_CORRECTION"
INCUMBENT = "M271_MEDIAN4"
BASE_ENSEMBLE = "E3_FOUR_FAMILY"
COMBINER = "median"

H1_MAX_SPEARMAN = -0.80


def build_base() -> pd.DataFrame:
    members = ENSEMBLES[BASE_ENSEMBLE]
    k = len(members)
    stacked = stack_members(members)
    base = combine(stacked, k, COMBINER)
    base["capacity"] = stacked["capacity"].to_numpy()
    base["fold"] = base["month"].map(QUARTER_OF_MONTH)
    rate = base["prediction_kwh"] / base["capacity"]
    base["pred_band"] = pd.cut(rate, bins=list(Y_BAND_EDGES), right=True).astype(str)
    base["pred_band_ord"] = pd.cut(rate, bins=list(Y_BAND_EDGES), right=True, labels=False)
    assert base["fold"].notna().all(), "fold 매핑에 구멍이 있다"
    return base


def main() -> int:
    base = build_base()
    incumbent_score = official(base)

    # --- A. 사이클 21 실패의 기전
    strata, _ = estimate_shifts(base, "median_shift")
    ord_of = (
        base.drop_duplicates(["group_id", "pred_band"])
        .set_index(["group_id", "pred_band"])["pred_band_ord"]
        .to_dict()
    )
    mechanism = []
    for g in sorted({int(key[0]) for key in strata}):
        pairs = sorted(
            ((ord_of[key], value) for key, value in strata.items() if int(key[0]) == g),
            key=lambda x: x[0],
        )
        if len(pairs) < 3:
            continue
        r, p = spearmanr([x for x, _ in pairs], [y for _, y in pairs])
        mechanism.append(
            {
                "group": g,
                "n_bands": len(pairs),
                "spearman": float(r),
                "p_value": float(p),
                "monotone_decreasing": bool(r <= H1_MAX_SPEARMAN),
                "shifts_by_band_ordinal": [
                    {"band_ord": int(x), "shift": round(float(y), 4)} for x, y in pairs
                ],
            }
        )
    h1 = bool(mechanism) and all(m["monotone_decreasing"] for m in mechanism)

    # 사이클 21 의 보정을 그대로 재현해 압축 여부를 잰다.
    pieces = []
    for held in FOLDS:
        train = base.loc[base["fold"] != held]
        test = base.loc[base["fold"] == held].copy()
        s, fb = estimate_shifts(train, "median_shift")
        test["prediction_kwh"] = apply_shifts(test, s, fb)
        pieces.append(test)
    c21 = pd.concat(pieces, ignore_index=True)

    compression = []
    for g in sorted(base["group_id"].unique()):
        before = float((base.loc[base["group_id"] == g, "prediction_kwh"]
                        / base.loc[base["group_id"] == g, "capacity"]).std())
        after = float((c21.loc[c21["group_id"] == g, "prediction_kwh"]
                       / c21.loc[c21["group_id"] == g, "capacity"]).std())
        compression.append(
            {
                "group": int(g),
                "sd_before": before,
                "sd_after": after,
                "ratio": after / before,
                "compressed": bool(after < before),
            }
        )
    h2 = all(c["compressed"] for c in compression)

    # --- B. 층화 없는 전역 이동 (그룹당 스칼라 하나)
    global_shifts: dict[int, list[float]] = {}
    pieces = []
    for held in FOLDS:
        train = base.loc[base["fold"] != held]
        test = base.loc[base["fold"] == held].copy()
        shift = {
            int(g): float(np.median((cell["actual_kwh"] - cell["prediction_kwh"])
                                    / cell["capacity"]))
            for g, cell in train.groupby("group_id", observed=True)
        }
        for g, v in shift.items():
            global_shifts.setdefault(g, []).append(v)
        test["prediction_kwh"] = test["prediction_kwh"] + test["group_id"].map(
            shift
        ).astype(float) * test["capacity"]
        pieces.append(test)
    corrected = pd.concat(pieces, ignore_index=True)
    assert len(corrected) == len(base), "LOO 이어붙이기에서 행 수가 바뀌었다"

    h3 = all(
        len({np.sign(x) for x in v}) == 1 and np.sign(v[0]) != 0
        for v in global_shifts.values()
    )
    g_score = official(corrected)
    g_gate = evaluate_gate(corrected, base)
    g_stats = g_gate.evidence
    improves = bool(g_score["total"] > incumbent_score["total"])
    h4 = bool(improves and g_gate.passed)

    verdict = (
        "GLOBAL_SHIFT_PROMOTED" if h4
        else ("BIAS_CORRECTION_FAMILY_CLOSED" if h1 and h2
              else "GLOBAL_SHIFT_REJECTED_MECHANISM_UNCONFIRMED")
    )
    promoted_total = g_score["total"] if h4 else incumbent_score["total"]

    check = {
        "H1_expectation": "층별 이동이 예측대역에 대해 단조 감소 "
                          f"(세 그룹 Spearman <= {H1_MAX_SPEARMAN})",
        "H1_held": h1,
        "H2_expectation": "사이클 21 보정이 예측 분포를 압축한다 (세 그룹 SD 감소)",
        "H2_held": h2,
        "H3_expectation": "그룹별 전역 이동의 부호가 세 분할에서 일치",
        "H3_held": h3,
        "H4_expectation": "전역 이동이 Total 개선 + 동결 게이트 통과",
        "H4_held": h4,
        "verdict": verdict,
    }

    payload = {
        "node": NODE_ID,
        "parent_node": PARENT_NODE,
        "closes_with_premise": CLOSES,
        "gate_version": GATE_VERSION,
        "gate_modified": False,
        "incumbent": INCUMBENT,
        "incumbent_score": incumbent_score,
        "mechanism_regression_to_mean": mechanism,
        "distribution_compression": compression,
        "global_shift": {
            "per_group_by_split": {
                str(g): [round(x, 5) for x in v] for g, v in sorted(global_shifts.items())
            },
            **g_score,
            "delta_vs_incumbent": g_score["total"] - incumbent_score["total"],
            "gate": {
                "passed": bool(g_gate.passed),
                "flags": {la.split()[0]: bool(ok) for la, ok in g_gate.conditions.items()},
                "positive_months": int(g_stats["positive_months"]),
                "months_scored": int(g_stats["months_scored"]),
                "sign_test_p": float(g_stats["sign_test_p_greater"]),
                "bootstrap_q05": float(g_stats["block_bootstrap_q05"]),
            },
        },
        "predeclared_check": check,
        "promoted_total": promoted_total,
        "gap_to_target": 0.66 - promoted_total,
    }

    lines = [
        "# M271 P4 사이클 22 — 사이클 21 실패의 기전과, 층화 없는 전역 이동",
        "",
        f"- 판정일: {datetime.now(UTC).strftime('%Y-%m-%d')} (UTC)",
        f"- 노드: `{NODE_ID}` / 레인 {LANE} / 부모 `{PARENT_NODE}`",
        f"- 기존 승격후보 `{INCUMBENT}` Total **{incumbent_score['total']:.6f}**",
        f"- 게이트: `{GATE_VERSION}` (읽기만 함)",
        "",
        "## 1. 사이클 21 은 왜 실패했는가 — 평균회귀 서명 (H1)",
        "",
        "예측대역으로 층화하면 잔차 이동이 대역 순서에 대해 **단조 감소**해야 한다.",
        "높은 예측 층에서는 실측이 아래로, 낮은 층에서는 위로 — 편향이 아니라 조건화의 산물이다.",
        "",
        "| 그룹 | 대역수 | Spearman | p | 단조감소 |",
        "|---:|---:|---:|---:|:---:|",
    ]
    for m in mechanism:
        lines.append(
            f"| {m['group']} | {m['n_bands']} | **{m['spearman']:+.4f}** | {m['p_value']:.4g} | "
            f"{'O' if m['monotone_decreasing'] else 'X'} |"
        )
    lines += ["", "대역 순서별 이동(용량 단위):", ""]
    for m in mechanism:
        lines.append(
            f"- 그룹 {m['group']}: "
            + " / ".join(f"`{d['shift']:+.4f}`" for d in m["shifts_by_band_ordinal"])
        )

    lines += [
        "",
        "## 2. 보정이 예보를 무디게 하는가 (H2)",
        "",
        "| 그룹 | 보정 전 SD | 보정 후 SD | 비 | 압축 |",
        "|---:|---:|---:|---:|:---:|",
    ]
    for c in compression:
        lines.append(
            f"| {c['group']} | {c['sd_before']:.4f} | {c['sd_after']:.4f} | "
            f"**{c['ratio']:.4f}** | {'O' if c['compressed'] else 'X'} |"
        )

    g = payload["global_shift"]
    flags = "".join("O" if g["gate"]["flags"].get(x) else "-" for x in ("G1", "G2", "G3", "G4"))
    lines += [
        "",
        "## 3. 층화 없는 전역 이동 (H3 · H4)",
        "",
        "그룹당 스칼라 하나. 평균회귀가 개입할 여지가 없는 최단순형이다.",
        "",
        "| 그룹 | 세 분할의 이동 (용량 단위) | 부호일치 |",
        "|---:|---|:---:|",
    ]
    for gid, v in sorted(global_shifts.items()):
        same = len({np.sign(x) for x in v}) == 1 and np.sign(v[0]) != 0
        lines.append(
            f"| {gid} | " + " / ".join(f"`{x:+.5f}`" for x in v) + f" | {'O' if same else 'X'} |"
        )
    lines += [
        "",
        f"보정 후 Total **{g['total']:.6f}** (기존대비 {g['delta_vs_incumbent']:+.6f}), "
        f"1-NMAE {g['one_minus_nmae']:.6f}, FICR {g['ficr']:.6f}.",
        f"게이트 `{flags}` {g['gate']['positive_months']}/{g['gate']['months_scored']}월 "
        f"p={g['gate']['sign_test_p']:.4f} q05={g['gate']['bootstrap_q05']:+.6f} -> "
        f"**{'통과' if g['gate']['passed'] else '기각'}**",
        "",
        "## 4. 사전확약 대조",
        "",
        f"- H1 `{check['H1_expectation']}` -> **{h1}**",
        f"- H2 `{check['H2_expectation']}` -> **{h2}**",
        f"- H3 `{check['H3_expectation']}` -> **{h3}**",
        f"- H4 `{check['H4_expectation']}` -> **{h4}**",
        "",
        f"판정: **{verdict}**",
        "",
        f"승격 Total **{promoted_total:.6f}**, 목표 0.66 까지 **{0.66 - promoted_total:+.6f}**.",
        "",
        "## 5. 폐기 전제 (기계검증 가능)",
        "",
        "```",
        "C1N21_MOS_BIAS_CORRECTION:",
        "  premise: 예측대역 층화는 평균회귀를 편향으로 오인한다",
        "  holds_if: 층별 이동이 대역 순서에 단조 감소 (측정 Spearman "
        + ", ".join(f"g{m['group']}={m['spearman']:+.3f}" for m in mechanism)
        + ")",
        "    그리고 보정이 예측 분포를 압축 (측정 비 "
        + ", ".join(f"g{c['group']}={c['ratio']:.3f}" for c in compression)
        + ")",
        "  flips_if: 예보시점 가용하면서 예측값과 독립인 층화 변수가 발견되면",
        "```",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    receipt: dict[str, Any] = {
        "schema_version": 1,
        "stage": "M271_P4_CYCLE22_GLOBAL_SHIFT",
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

    print("[C22] 평균회귀 Spearman  "
          + "  ".join(f"g{m['group']}={m['spearman']:+.4f}" for m in mechanism)
          + f"  -> H1 {h1}")
    print("[C22] 분포 압축 비  "
          + "  ".join(f"g{c['group']}={c['ratio']:.4f}" for c in compression)
          + f"  -> H2 {h2}")
    print(f"[C22] 전역이동 부호일치 -> H3 {h3}")
    print(f"[C22] 전역이동 Total {g['total']:.6f} "
          f"(기존대비 {g['delta_vs_incumbent']:+.6f}) "
          f"게이트 {'통과' if g['gate']['passed'] else '기각'} -> H4 {h4}")
    print(f"[C22] 판정: {verdict}  ->  Total {promoted_total:.6f} "
          f"(목표까지 {0.66 - promoted_total:+.6f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
