"""M271 P4 사이클 18 — 멤버 불일치가 정산 성패를 예보하는가 (C9 전제 재검).

사이클 17 이 뜻밖의 것을 쟀다. 멤버 스프레드 중앙값 `0.0833` 은 앙상블 절대오차 중앙값
`0.0978` 의 **85%** 다. 유효행의 64.9% 가 6% 밴드보다 넓게 불일치한다. 즉 **예보시점에
라벨 없이 얻을 수 있는 큰 신호**가 하나 있다.

이것이 닫힌 축 하나의 전제를 건드린다. 사이클 8 은 "일반화 가능한 조건화로는 결정층
오라클의 16.5% 가 한계" 라며 정책 조건화를 닫았다. 그때 후보 조건화 변수에 **멤버
불일치는 없었다** — 앙상블 축은 사이클 13 에서야 열렸기 때문이다. 없던 변수가 생겼으므로
C9(전제 뒤집힘) 조건이다. 다만 **전제가 실제로 뒤집혔는지 재는 것**이 이 노드의 일이고,
정책을 바꾸는 것은 이 노드의 일이 아니다.

① 방법 리서치 (실행 전)
  - Whitaker & Loughe (1998), *Mon. Wea. Rev.* — spread-skill 관계. 완벽한 앙상블에서도
    스프레드와 **단일 실현** 절대오차의 원상관은 1 보다 크게 낮다. 원상관으로 판정하면
    실재하는 관계를 놓친다.
  - ECMWF, *Verifying the Relationship between Ensemble Forecast Spread and Skill* —
    표준 진단은 **층화 비닝**: 스프레드 구간별로 오차의 조건부 분포가 바뀌는지 본다.
  - Pinson & Kariniotakis, *Skill forecasting from ensemble predictions of wind power*,
    Applied Energy — 풍력에 앙상블 스프레드 기반 prediction risk index 를 직접 적용한
    선례. 스프레드를 예보 위험도로 쓰는 것이 이 도메인에서 확립된 용법임을 보인다.
  - 채택: 층화 비닝 + 정산단위 히트율. 기각: 원상관 판정(위 이유), 스프레드로 가중치를
    적합하는 것(평가 fold 선택 편향).

② 사양 동결 — 재는 것은 넷

  통제변수는 **예측** 대역이지 실측 y 대역이 아니다. 결정 정책은 예보시점에 실측을 모른다.
  실측으로 통제하면 사이클 1 이 빠졌던 충돌부 함정에 그대로 다시 빠진다.

  사전확약(실행 전 동결):
    H1  최저 스프레드 십분위와 최고 십분위의 **FICR 히트율(|err| <= 6%) 차이 >= 15%p**.
    H2  십분위별 히트율이 **단조**다 (Spearman |rho| >= 0.8). 단조가 아니면 정책
        조건화에 쓸 수 없다.
    H3  관계의 **부호가 세 그룹 모두 같다**. 그룹별로 뒤집히면 일반화되지 않는다.
    H4  (핵심) 스프레드가 **예측대역 x 그룹으로 설명되지 않는 정보**를 갖는다. 각 셀
        안에서 스프레드 상/하 절반의 히트율 차이가 행가중 평균 **>= 10%p**.
  H4 가 요점이다. 스프레드가 단지 "발전량 크면 오차도 크다" 의 대리면 새 정보가 아니고
  사이클 8 의 전제는 그대로 유지된다. 넷 다 성립해야 전제가 뒤집힌 것으로 본다.

**게이트를 수정하지 않는다. 정책도 바꾸지 않는다.** 이 노드는 측정만 한다.

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

from m271_cycle13_ensemble import ENSEMBLES
from m271_cycle17_combiner import combine, stack_members
from m271_n0_deficit_init import Y_BAND_EDGES

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
REPORT_MD = REPORTS / "m271_cycle18_spread_skill.md"
RECEIPT = REPORTS / "m271_cycle18_spread_skill_receipt.json"

NODE_ID = "C1N18_SPREAD_SKILL"
LANE = "L4"  # 검증전략 — 조건부 검증
PARENT_NODE = "C1N17_METRIC_ALIGNED_COMBINER"
REOPENS = "C1N8_POLICY_SELECTION_CONDITIONING"
BASE_ENSEMBLE = "E3_FOUR_FAMILY"

BAND_HIT = 0.06  # 정산단위 4
BAND_PARTIAL = 0.08  # 정산단위 3
N_DECILES = 10

H1_MIN_HITRATE_GAP = 0.15
H2_MIN_SPEARMAN = 0.80
H4_MIN_CONTROLLED_GAP = 0.10

METHOD_SOURCES = (
    {
        "id": "whitaker_loughe_1998",
        "cite": "Whitaker & Loughe (1998), Mon. Wea. Rev. — spread-skill relationship",
        "claim": "완벽한 앙상블에서도 스프레드와 단일 실현 오차의 원상관은 1 보다 크게 낮다",
        "applicability": "directly_supported",
        "use": "원상관 판정을 금지하고 층화 비닝을 채택하는 근거",
    },
    {
        "id": "ecmwf_spread_skill_verification",
        "cite": "ECMWF — Verifying the Relationship between Ensemble Forecast Spread and Skill",
        "claim": "표준 진단은 스프레드 구간별 오차 조건부 분포의 층화 비닝",
        "applicability": "directly_supported",
        "use": "십분위 층화 + 히트율 곡선 설계",
    },
    {
        "id": "pinson_kariniotakis_skill_forecasting",
        "cite": "Pinson & Kariniotakis, Skill forecasting from ensemble predictions of "
                "wind power, Applied Energy",
        "claim": "앙상블 스프레드로 prediction risk index 를 만들어 풍력 예보에 적용",
        "applicability": "directly_supported",
        "use": "이 도메인에서 스프레드를 예보 위험도로 쓰는 것이 확립된 용법임을 보임",
    },
)


def band_of(values: np.ndarray) -> pd.Series:
    return pd.cut(pd.Series(values), bins=list(Y_BAND_EDGES), right=True).astype(str)


def main() -> int:
    members = ENSEMBLES[BASE_ENSEMBLE]
    k = len(members)
    stacked = stack_members(members)

    cols = [f"m{i}" for i in range(k)]
    arr = stacked.loc[:, cols].to_numpy(dtype="float64")
    cap = stacked["capacity"].to_numpy(dtype="float64")
    actual = stacked["actual_kwh"].to_numpy(dtype="float64")

    # 사이클 17 이 고른 강건 결합자를 쓴다. 스프레드는 예보시점 가용 — 라벨 미사용.
    pred = combine(stacked, k, "median")["prediction_kwh"].to_numpy(dtype="float64")

    frame = pd.DataFrame(
        {
            "group_id": stacked["group_id"].to_numpy(),
            "month": stacked["month"].to_numpy(),
            "spread": (arr.max(axis=1) - arr.min(axis=1)) / cap,
            "spread_sd": arr.std(axis=1, ddof=1) / cap,
            "abs_err": np.abs(pred - actual) / cap,
            "y": actual / cap,
            "pred_rate": pred / cap,
        }
    )
    frame = frame.loc[frame["y"] >= 0.10].reset_index(drop=True)  # 채점 대상 유효행
    frame["hit6"] = (frame["abs_err"] <= BAND_HIT).astype(float)
    frame["hit8"] = (frame["abs_err"] <= BAND_PARTIAL).astype(float)
    frame["unit"] = np.where(frame["hit6"] > 0, 4.0, np.where(frame["hit8"] > 0, 3.0, 0.0))
    # 예측대역 — 예보시점 가용. 실측 y 대역을 쓰면 충돌부 조건화가 된다.
    frame["pred_band"] = band_of(frame["pred_rate"].to_numpy())

    # --- H1 · H2 십분위 층화
    frame["decile"] = pd.qcut(frame["spread"], N_DECILES, labels=False, duplicates="drop")
    grouped = frame.groupby("decile", observed=True)
    deciles = []
    for d, g in grouped:
        deciles.append(
            {
                "decile": int(d),
                "rows": len(g),
                "spread_lo": float(g["spread"].min()),
                "spread_hi": float(g["spread"].max()),
                "median_abs_err": float(g["abs_err"].median()),
                "hit_rate_6pct": float(g["hit6"].mean()),
                "hit_rate_8pct": float(g["hit8"].mean()),
                "mean_unit": float(g["unit"].mean()),
            }
        )
    hit_curve = [d["hit_rate_6pct"] for d in deciles]
    hitrate_gap = hit_curve[0] - hit_curve[-1]
    rho, rho_p = spearmanr([d["decile"] for d in deciles], hit_curve)
    h1 = bool(hitrate_gap >= H1_MIN_HITRATE_GAP)
    h2 = bool(abs(rho) >= H2_MIN_SPEARMAN)

    # --- H3 그룹별 부호 일치
    per_group = []
    for g_id, g in frame.groupby("group_id", observed=True):
        d = pd.qcut(g["spread"], N_DECILES, labels=False, duplicates="drop")
        curve = g.groupby(d, observed=True)["hit6"].mean()
        gap = float(curve.iloc[0] - curve.iloc[-1])
        r, _ = spearmanr(curve.index.to_numpy(dtype=float), curve.to_numpy())
        per_group.append(
            {
                "group": int(g_id),
                "rows": len(g),
                "hit_rate_gap": gap,
                "spearman": float(r),
                "sign_negative": bool(r < 0),
            }
        )
    h3 = bool(len({p["sign_negative"] for p in per_group}) == 1)

    # --- H4 예측대역 x 그룹 통제 후에도 남는가
    controlled = []
    total_rows = 0
    weighted_gap = 0.0
    for (g_id, band), cell in frame.groupby(["group_id", "pred_band"], observed=True):
        if len(cell) < 100:  # 표본이 얇으면 히트율 차이가 잡음이다
            continue
        cut = cell["spread"].median()
        lo = cell.loc[cell["spread"] <= cut, "hit6"]
        hi = cell.loc[cell["spread"] > cut, "hit6"]
        if len(lo) < 30 or len(hi) < 30:
            continue
        gap = float(lo.mean() - hi.mean())
        controlled.append(
            {
                "group": int(g_id),
                "pred_band": band,
                "rows": len(cell),
                "hit_low_spread": float(lo.mean()),
                "hit_high_spread": float(hi.mean()),
                "gap": gap,
            }
        )
        total_rows += len(cell)
        weighted_gap += gap * len(cell)
    controlled_gap = weighted_gap / total_rows if total_rows else 0.0
    h4 = bool(controlled_gap >= H4_MIN_CONTROLLED_GAP)

    # 참고: 문헌이 판정 근거로 쓰지 말라 한 원상관도 기록만 한다.
    raw_corr = float(np.corrcoef(frame["spread"], frame["abs_err"])[0, 1])

    all_held = bool(h1 and h2 and h3 and h4)
    check = {
        "H1_expectation": f"최저/최고 십분위 히트율 차이 >= {H1_MIN_HITRATE_GAP:.0%}",
        "H1_held": h1,
        "H1_measured": hitrate_gap,
        "H2_expectation": f"십분위 히트율 단조 (|Spearman| >= {H2_MIN_SPEARMAN})",
        "H2_held": h2,
        "H2_measured": float(rho),
        "H3_expectation": "세 그룹 부호 일치",
        "H3_held": h3,
        "H4_expectation": f"예측대역x그룹 통제 후 잔여 차이 >= {H4_MIN_CONTROLLED_GAP:.0%}",
        "H4_held": h4,
        "H4_measured": controlled_gap,
        "verdict": (
            "PREMISE_FLIPPED_C8_REOPENS" if all_held else "PREMISE_HOLDS_C8_STAYS_CLOSED"
        ),
    }

    payload = {
        "node": NODE_ID,
        "parent_node": PARENT_NODE,
        "reopens_candidate": REOPENS,
        "gate_modified": False,
        "policy_modified": False,
        "combiner": "median",
        "members": list(members),
        "method_sources": list(METHOD_SOURCES),
        "rows_eligible": len(frame),
        "raw_correlation_spread_vs_abserr": raw_corr,
        "deciles": deciles,
        "per_group": per_group,
        "controlled_cells": controlled,
        "controlled_gap_row_weighted": controlled_gap,
        "predeclared_check": check,
    }

    lines = [
        "# M271 P4 사이클 18 — 멤버 불일치가 정산 성패를 예보하는가",
        "",
        f"- 판정일: {datetime.now(UTC).strftime('%Y-%m-%d')} (UTC)",
        f"- 노드: `{NODE_ID}` / 레인 {LANE} / 부모 `{PARENT_NODE}`",
        f"- 재검 대상 폐기축: `{REOPENS}` (사이클 8)",
        f"- 유효행 {len(frame):,} / 결합자 `median` / 게이트·정책 **미변경**",
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
        f"참고로 원상관은 `{raw_corr:+.4f}` 다. 문헌이 이것으로 판정하지 말라고 하므로",
        "기록만 하고 판정에는 쓰지 않는다.",
        "",
        "## 1. 스프레드 십분위 층화 (H1 · H2)",
        "",
        "| 십분위 | 스프레드 범위 | 행 | 절대오차 중앙값 | **히트율 6%** | 히트율 8% "
        "| 평균 정산단위 |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for d in deciles:
        lines.append(
            f"| {d['decile']} | {d['spread_lo']:.4f}~{d['spread_hi']:.4f} | {d['rows']:,} | "
            f"{d['median_abs_err']:.4f} | **{d['hit_rate_6pct']:.4f}** | "
            f"{d['hit_rate_8pct']:.4f} | {d['mean_unit']:.4f} |"
        )
    lines += [
        "",
        f"최저-최고 히트율 차이 **{hitrate_gap:+.4f}**, Spearman `{rho:+.4f}` (p={rho_p:.3g}).",
        "",
        "## 2. 그룹별 부호 (H3)",
        "",
        "| 그룹 | 행 | 히트율 차이 | Spearman | 음의 관계 |",
        "|---:|---:|---:|---:|:---:|",
    ]
    for p in per_group:
        lines.append(
            f"| {p['group']} | {p['rows']:,} | {p['hit_rate_gap']:+.4f} | "
            f"{p['spearman']:+.4f} | {'O' if p['sign_negative'] else 'X'} |"
        )
    lines += [
        "",
        "## 3. 예측대역 x 그룹 통제 (H4)",
        "",
        "스프레드가 단지 발전량 수준의 대리라면 셀 안에서는 차이가 사라진다.",
        "**통제변수는 예측대역이다** — 실측 y 대역으로 통제하면 충돌부 조건화가 된다.",
        "",
        "| 그룹 | 예측대역 | 행 | 저스프레드 히트율 | 고스프레드 히트율 | 차이 |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for c in sorted(controlled, key=lambda x: -x["rows"])[:14]:
        lines.append(
            f"| {c['group']} | {c['pred_band']} | {c['rows']:,} | "
            f"{c['hit_low_spread']:.4f} | {c['hit_high_spread']:.4f} | **{c['gap']:+.4f}** |"
        )
    lines += [
        "",
        f"셀 {len(controlled)} 개, 행가중 평균 잔여 차이 **{controlled_gap:+.4f}**.",
        "",
        "## 4. 사전확약 대조",
        "",
        f"- H1 `{check['H1_expectation']}` -> **{h1}** (실측 {hitrate_gap:+.4f})",
        f"- H2 `{check['H2_expectation']}` -> **{h2}** (실측 {rho:+.4f})",
        f"- H3 `{check['H3_expectation']}` -> **{h3}**",
        f"- H4 `{check['H4_expectation']}` -> **{h4}** (실측 {controlled_gap:+.4f})",
        "",
        f"판정: **{check['verdict']}**",
        "",
        "## 5. 이 노드가 하지 않은 것",
        "",
        "전제가 뒤집혔는지만 쟀다. **정책은 바꾸지 않았다.** 조건부 정책은 별도 노드이며,",
        "사이클 7 이 보인 대로 재배분형 정책은 동결 게이트에서 기각된다. 조건화 변수가",
        "진짜 정보를 가질 때만 그 노드를 열 근거가 된다.",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    receipt: dict[str, Any] = {
        "schema_version": 1,
        "stage": "M271_P4_CYCLE18_SPREAD_SKILL",
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

    print(f"[C18] 유효행 {len(frame):,}  원상관 {raw_corr:+.4f} (판정 미사용)")
    print(f"[C18] 히트율 최저십분위 {hit_curve[0]:.4f} -> 최고십분위 {hit_curve[-1]:.4f}  "
          f"차이 {hitrate_gap:+.4f}  -> H1 {h1}")
    print(f"[C18] Spearman {rho:+.4f} -> H2 {h2}")
    print("[C18] 그룹부호 " + " ".join(f"g{p['group']}={p['spearman']:+.3f}" for p in per_group)
          + f"  -> H3 {h3}")
    print(f"[C18] 통제 후 잔여 차이 {controlled_gap:+.4f} ({len(controlled)} 셀) -> H4 {h4}")
    print(f"[C18] 판정: {check['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
