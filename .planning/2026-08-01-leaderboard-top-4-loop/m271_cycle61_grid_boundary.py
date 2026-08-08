"""M271 P4 사이클 61 — 격자가 물려 있다: 동결 온도격자를 수준별 요구에 맞게 연다.

사이클 60 이 낸 온도표가 스스로 결함을 가리킨다.

    수준 0 (yhat<=0.25)      2.2  2.2  2.2     <- 격자 **상한**
    수준 1 (0.25<yhat<=0.70) 2.2  2.2  2.2     <- 격자 **상한**
    수준 2 (yhat>0.70)       0.3  0.4  0.4     <- 격자 **하한**

세 fold 전부, 모든 수준이 격자 `{0.3 ... 2.2}` 의 **끝점**을 골랐다. 저·중 수준은 최대
평탄화를, 고 수준은 최대 날카로움을 원한다 — 정반대 방향이고 둘 다 벽에 막혀 있다.

그 격자는 C44 가 **전역 T 하나**를 전제로 동결한 것이다. 하나의 T 는 행 수가 많은 저·중
수준에 끌려가 2.2 에 앉으므로 하한이 닿을 일이 없었다. 수준별로 쪼개는 순간 각 수준이
요구하는 범위가 격자 밖으로 나간다. **끝점 최적해는 결과가 아니라 계측 실패다.**

**① 방법 리서치**

  - 격자탐색에서 최적해가 경계에 있으면 격자를 넓히는 것이 표준 절차다. 새 방법이
    아니라 계측 위생이다. 넓힌 뒤에도 경계면 다시 넓힌다 — H3 이 그것을 강제한다.
  - 다만 격자를 넓히면 **선택 자유도가 커진다.** C60 이 이미 9 파라미터가 3 파라미터에
    지는 것을 봤으므로(과적합), 격자 확장의 이득과 선택비용을 분리해서 재야 한다.
  - **채택**: 2x2 요인설계. `{동결격자, 확장격자} x {전역 T, 수준별 T}`. 요인설계는
    두 인자의 주효과와 상호작용을 같은 실행에서 분리하는 표준이고, 여기서는
    "격자를 넓힌 덕"과 "수준별로 쪼갠 덕"을 섞지 않기 위해 필요하다.
  - 앞단은 `m271_decision_surface` 캐시로 고정한다. 네 팔이 **같은 확률행렬**을 쓰므로
    처리효과가 결정층에만 귀속된다(C60 과 동일한 논리).

**② 사양 동결**

  확률면  `m271_decision_surface.load_surface()`. 시드·스레드 고정된 결정적 앞단.
  동결격자 (0.3, 0.4, 0.5, 0.6, 0.75, 1.0, 1.3, 1.7, 2.2)            <- C44 원본
  확장격자 (0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.75, 1.00,
            1.30, 1.70, 2.20, 3.00, 4.00, 6.00)                      <- 양끝 확장
  수준     C60 과 동일. 예비 점추정 yhat0 = T=1.0 Bayes 결정 / 용량, 경계 (0.25, 0.70).
           **이번 사이클에서 수준 경계는 건드리지 않는다** — 한 번에 한 인자.
  선택     fold-외. 다파라미터 팔은 전역 최적에서 출발해 좌표상승 2 회전.

  **타당성 가드 — 둘 다 재현 대조다**
    V1  `global_frozen` == 0.604043 (C44 대조군, ±0.0005)
    V2  `level_frozen`  == 0.613033 (C60 LEVEL, ±0.0005)
        V2 가 핵심이다. 확률면을 캐시로 옮기고 코드를 재구성했으므로, C60 의 값을
        **소수점까지 재현하지 못하면** 리팩터가 계산을 바꾼 것이고 나머지를 버린다.

  사전확약 (V1·V2 통과시에만 판정):
    H1  level_extended > level_frozen.        격자가 물려 있었다면 참.
    H2  level_extended > global_extended.     공정한 격자에서도 수준별 자유도가 값을 한다.
    H3  선택된 T 가 **모든 수준·모든 fold 에서 확장격자의 내부점**이다.
        거짓이면 아직도 벽에 막힌 것이고 **결과를 주장하지 않고 다시 넓힌다.**
    H4  level_extended 가 global_frozen 대비 **동결 게이트 통과**.
    H5  이득이 FICR 쪽에서 우세.
    H6  global_extended ~= global_frozen (차 < 0.001). 전역 T 문제에서는 C44 격자가
        충분했음을 뜻한다. 크게 다르면 C44 의 대조군 자체가 과소격자였다는 뜻이고,
        그건 C56·C58·C60 이 모두 그 대조군에 기대므로 **되짚어야 할 문제**다.

  H3 을 판정 게이트로 두는 것이 요점이다. 경계 최적해를 결과로 보고하면 그 숫자는
  격자의 성질이지 문제의 성질이 아니다.

게이트 미수정. lockbox·외부데이터·2024 행·`scada_ws` 미사용. 제출 없음.
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

from m270_gate import GATE_VERSION, evaluate_gate
from m271_cycle40_band_classifier import bayes_decision
from m271_cycle44_sharpened_decision import sharpen
from m271_cycle60_level_temperature import (
    LEVEL_EDGES,
    N_LEVEL,
    level_of,
    sharpen_by_row,
)
from m271_decision_surface import load_surface
from m271_evaluate_candidate import official

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
REPORT_MD = REPORTS / "m271_cycle61_grid_boundary.md"
RECEIPT = REPORTS / "m271_cycle61_grid_boundary_receipt.json"

NODE_ID = "C1N61_GRID_BOUNDARY"
LANE = "L7"
PARENT_NODE = "C1N60_LEVEL_TEMPERATURE"

C44_CONTROL = 0.604043
C60_LEVEL = 0.613033
TOLERANCE = 0.0005
CHAMPION_LOCAL = 0.630310

FROZEN_GRID = (0.3, 0.4, 0.5, 0.6, 0.75, 1.0, 1.3, 1.7, 2.2)
EXTENDED_GRID = (0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.75, 1.00,
                 1.30, 1.70, 2.20, 3.00, 4.00, 6.00)
SWEEPS = 2
ARMS = (
    ("global_frozen", "global", FROZEN_GRID),
    ("level_frozen", "level", FROZEN_GRID),
    ("global_extended", "global", EXTENDED_GRID),
    ("level_extended", "level", EXTENDED_GRID),
)


def main() -> int:
    store, info = load_surface()
    for fold, entry in store.items():
        preliminary = bayes_decision(sharpen(entry["probability"], 1.0))
        entry["level"] = level_of(preliminary)
        entry["meta"] = entry["meta"].copy()

    def scored(fold: str, temperature: np.ndarray) -> pd.DataFrame:
        e = store[fold]
        out = e["meta"].copy()
        out["prediction_kwh"] = (
            bayes_decision(sharpen_by_row(e["probability"], temperature)) * e["capacity"]
        )
        out["group_id"] = e["group"]
        out["month"] = out["forecast_kst_dtm"].dt.to_period("M").astype(str)
        return out

    def expand(fold: str, kind: str, table: Any) -> np.ndarray:
        e = store[fold]
        if kind == "global":
            return np.full(len(e["level"]), float(table))
        return np.asarray([table[int(v)] for v in e["level"]], dtype="float64")

    def pooled(folds: list[str], kind: str, table: Any) -> float:
        frame = pd.concat(
            [scored(f, expand(f, kind, table)) for f in folds], ignore_index=True
        )
        return float(official(frame)["total"])

    def select(folds: list[str], kind: str, grid: tuple[float, ...]) -> Any:
        best_t, best_score = grid[0], -np.inf
        for temperature in grid:
            score = pooled(folds, "global", temperature)
            if score > best_score:
                best_t, best_score = temperature, score
        if kind == "global":
            return float(best_t)
        table = {level: float(best_t) for level in range(N_LEVEL)}
        for _ in range(SWEEPS):
            for level in range(N_LEVEL):
                current, incumbent = table[level], best_score
                for temperature in grid:
                    table[level] = float(temperature)
                    score = pooled(folds, "level", table)
                    if score > incumbent:
                        current, incumbent = float(temperature), score
                table[level], best_score = current, incumbent
        return table

    chosen: dict[str, dict[str, Any]] = {}
    pieces: dict[str, list[pd.DataFrame]] = {name: [] for name, _, _ in ARMS}
    for name, kind, grid in ARMS:
        chosen[name] = {}
        for held in sorted(store):
            others = [f for f in sorted(store) if f != held]
            table = select(others, kind, grid)
            chosen[name][held] = (
                float(table) if kind == "global" else {str(k): v for k, v in table.items()}
            )
            pieces[name].append(scored(held, expand(held, kind, table)))

    frames = {name: pd.concat(parts, ignore_index=True) for name, parts in pieces.items()}
    results = {name: official(frames[name]) for name in frames}

    v1_gap = abs(results["global_frozen"]["total"] - C44_CONTROL)
    v2_gap = abs(results["level_frozen"]["total"] - C60_LEVEL)
    v1 = bool(v1_gap <= TOLERANCE)
    v2 = bool(v2_gap <= TOLERANCE)

    h1 = bool(results["level_extended"]["total"] > results["level_frozen"]["total"])
    h2 = bool(results["level_extended"]["total"] > results["global_extended"]["total"])

    interior: list[bool] = []
    boundary_hits: list[str] = []
    lo, hi = EXTENDED_GRID[0], EXTENDED_GRID[-1]
    for held, table in chosen["level_extended"].items():
        for key, value in table.items():
            ok = lo < float(value) < hi
            interior.append(ok)
            if not ok:
                boundary_hits.append(f"{held}/L{key}={value}")
    h3 = bool(all(interior))

    gate = evaluate_gate(frames["level_extended"], frames["global_frozen"])
    gd = gate.evidence
    h4 = bool(gate.passed)
    flags = {label.split()[0]: bool(ok) for label, ok in gate.conditions.items()}
    signature = "[" + "".join("O" if flags[g] else "-" for g in sorted(flags)) + "]"

    ficr_contrib = 0.5 * (
        results["level_extended"]["ficr"] - results["global_frozen"]["ficr"]
    )
    nmae_contrib = 0.5 * (
        results["level_extended"]["one_minus_nmae"]
        - results["global_frozen"]["one_minus_nmae"]
    )
    h5 = bool(ficr_contrib > nmae_contrib)
    global_grid_effect = (
        results["global_extended"]["total"] - results["global_frozen"]["total"]
    )
    h6 = bool(abs(global_grid_effect) < 0.001)

    if not v1 or not v2:
        verdict = "REPRODUCTION_CONTROL_FAILED_RESULT_VOID"
    elif not h3:
        verdict = "GRID_STILL_BINDING_EXTEND_AGAIN"
    elif not h1:
        verdict = "FROZEN_GRID_WAS_ADEQUATE_FOR_LEVELS"
    elif results["level_extended"]["total"] > CHAMPION_LOCAL:
        verdict = "EXTENDED_LEVEL_TEMPERATURE_BEATS_CHAMPION"
    elif h4:
        verdict = "EXTENDED_LEVEL_TEMPERATURE_GATE_PASSED"
    else:
        verdict = "EXTENDED_LEVEL_GAIN_NOT_MONTHLY_CONSISTENT"

    payload: dict[str, Any] = {
        "node_id": NODE_ID,
        "lane": LANE,
        "parent": PARENT_NODE,
        "gate_version": GATE_VERSION,
        "surface": info,
        "frozen_grid": list(FROZEN_GRID),
        "extended_grid": list(EXTENDED_GRID),
        "level_edges": list(LEVEL_EDGES),
        "arms": results,
        "chosen": chosen,
        "checks": {
            "V1_global_frozen_reproduces_c44": v1, "V1_gap": float(v1_gap),
            "V2_level_frozen_reproduces_c60": v2, "V2_gap": float(v2_gap),
        },
        "effects": {
            "grid_on_level": float(
                results["level_extended"]["total"] - results["level_frozen"]["total"]
            ),
            "grid_on_global": float(global_grid_effect),
            "level_on_extended": float(
                results["level_extended"]["total"] - results["global_extended"]["total"]
            ),
            "level_on_frozen": float(
                results["level_frozen"]["total"] - results["global_frozen"]["total"]
            ),
        },
        "hypotheses": {
            "H1_extended_beats_frozen_levels": h1,
            "H2_level_beats_global_on_extended": h2,
            "H3_all_choices_interior": h3,
            "H4_gate_passed": h4,
            "H5_ficr_dominant": h5,
            "H6_global_unaffected_by_grid": h6,
        },
        "boundary_hits": boundary_hits,
        "contributions": {"ficr": float(ficr_contrib), "nmae": float(nmae_contrib)},
        "gate": {
            "signature": signature, "flags": flags,
            "positive_months": int(gd["positive_months"]),
            "months_scored": int(gd["months_scored"]),
            "sign_test_p": float(gd["sign_test_p_greater"]),
            "median_delta": float(gd["median_total_delta"]),
            "bootstrap_q05": float(gd["block_bootstrap_q05"]),
            "min_delta": float(gd["min_total_delta"]),
        },
        "champion_local": CHAMPION_LOCAL,
        "verdict": verdict,
        "dacon_upload": False,
        "external_actions": [],
        "lockbox_used": False,
        "generated_at": datetime.now(UTC).isoformat(),
    }
    payload["digest"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]
    RECEIPT.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    lines = [
        "# M271 P4 사이클 61 — 온도격자 경계",
        "",
        f"노드 `{NODE_ID}` / 레인 {LANE} / 부모 `{PARENT_NODE}`",
        "",
        f"확률면 spec `{info['spec_digest']}` / prob `{info['probability_digest']}` "
        f"/ 캐시 {info['from_cache']} — **네 팔이 같은 행렬을 쓴다**",
        "",
        "## 1. 2x2 요인",
        "",
        "| 팔 | 격자 | 자유도 | Total | 1-NMAE | FICR |",
        "|---|---|---|---:|---:|---:|",
    ]
    for name, kind, grid in ARMS:
        r = results[name]
        label = "동결" if grid is FROZEN_GRID else "확장"
        free = "전역 1" if kind == "global" else f"수준 {N_LEVEL}"
        lines.append(
            f"| {name} | {label} | {free} | {r['total']:.6f} | "
            f"{r['one_minus_nmae']:.6f} | {r['ficr']:.6f} |"
        )
    e = payload["effects"]
    lines += [
        "",
        "**주효과 분해**",
        "",
        f"- 격자 확장이 수준별 팔에 준 것: **{e['grid_on_level']:+.6f}**",
        f"- 격자 확장이 전역 팔에 준 것: **{e['grid_on_global']:+.6f}**",
        f"- 수준 자유도가 확장격자에서 준 것: **{e['level_on_extended']:+.6f}**",
        f"- 수준 자유도가 동결격자에서 준 것: **{e['level_on_frozen']:+.6f}** (C60 재현)",
        "",
        "## 2. 재현 대조",
        "",
        f"- V1 `global_frozen` {results['global_frozen']['total']:.6f} vs C44 {C44_CONTROL} "
        f"(차 {v1_gap:.6f}) -> **{v1}**",
        f"- V2 `level_frozen` {results['level_frozen']['total']:.6f} vs C60 {C60_LEVEL} "
        f"(차 {v2_gap:.6f}) -> **{v2}**",
        "",
        "## 3. 사전확약",
        "",
        f"- H1 확장 > 동결 (수준별) -> **{h1}**",
        f"- H2 수준별 > 전역 (확장격자) -> **{h2}**",
        f"- H3 선택이 전부 격자 **내부점** -> **{h3}**"
        + (f"  경계 접촉: `{', '.join(boundary_hits)}`" if boundary_hits else ""),
        f"- H4 게이트 통과 -> **{h4}** {signature} "
        f"({gd['positive_months']}/{gd['months_scored']} 월, "
        f"p={gd['sign_test_p_greater']:.4f}, q05={gd['block_bootstrap_q05']:+.6f})",
        f"- H5 FICR 우세 (FICR {ficr_contrib:+.6f} / 1-NMAE {nmae_contrib:+.6f}) -> **{h5}**",
        f"- H6 전역 팔이 격자에 무감 ({global_grid_effect:+.6f}) -> **{h6}**",
        "",
        "## 4. 선택된 온도 (fold-외)",
        "",
        "```",
        json.dumps(chosen["level_extended"], indent=1, ensure_ascii=False),
        "```",
        "",
        "## 5. 판정",
        "",
        f"**{verdict}**",
        "",
        f"챔피언 로컬 {CHAMPION_LOCAL} / level_extended "
        f"{results['level_extended']['total']:.6f}",
        "",
        f"digest `{payload['digest']}`",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("=== 완료 ===")
    for name, _kind, _grid in ARMS:
        r = results[name]
        print(f"[C61] {name:16s} {r['total']:.6f} "
              f"(1-NMAE {r['one_minus_nmae']:.6f} / FICR {r['ficr']:.6f})")
    print(f"[C61] V1 {v1} (차 {v1_gap:.6f}) / V2 {v2} (차 {v2_gap:.6f})")
    print(f"[C61] 격자효과 수준별 {e['grid_on_level']:+.6f} / 전역 {e['grid_on_global']:+.6f}")
    print(f"[C61] H1 {h1} / H2 {h2} / H3 내부점 {h3} / H4 게이트 {h4} {signature} "
          f"/ H5 {h5} / H6 {h6}")
    if boundary_hits:
        print(f"[C61] 경계 접촉: {', '.join(boundary_hits)}")
    print(f"[C61] 판정: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
