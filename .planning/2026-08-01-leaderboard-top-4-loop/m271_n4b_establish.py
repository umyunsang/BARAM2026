"""M271 N4b — 구간내부 적분 이득이 실재하는가: C1N76 의 올바른 계측기로 검정.

C1N90(N4)이 `uniform` 적분으로 **+0.001890** 을 냈다(검출문턱 0.001013 의 1.87 배,
FICR 기여 +0.001896 / 1-NMAE -0.000006, **재학습 없음**). 그러나 동결 게이트가
`[-O-O]` 5/9 월로 기각했다 — C1N60·C1N73 과 같은 서명이다.

**그런데 C1N60·C1N73 과 성질이 다를 이유가 있다.** 그 둘은 온도·가중이라는 **적합된
자유도**였고 월별로 흔들렸다. 이것은 **계산 결함 수정**이다 — 구간을 점질량으로 다뤄
생긴 체계적 편향을 적분으로 고친 것이라 방향이 뒤집힐 이유가 없다.

C1N76 이 확립한 계측기로 잰다. **원형 블록 부트스트랩 + 일별 델타 평균** — 선형
추정량이라 편향이 없고(C1N75 가 확인), 가장자리 과소표집도 없다(C1N76 이 교정).
n=228 일로 월 블록의 9 보다 훨씬 크다.

**① 사양 동결**

  대상   `uniform` - `point` (C1N90 의 최선 팔과 대조군). C1N90 이 fold-외로 고른
         온도를 **그대로 쓴다** — 여기서 다시 고르면 다른 실험이 된다.
  단위   일(KST). 세 그룹 모두 유효행이 있는 날만. C1N76 과 동일한 228 일.
  통계량 일별 Total 델타의 평균(선형).
  절차   **원형** 블록 부트스트랩(Politis & Romano 1992). 길이 격자·추첨·시드 모두
         C1N76 과 동일 — 1,2,3,5,7,10,14,21,30 / 4000 / 20260805.

  **타당성 가드**
    V1  전 블록길이에서 부트스트랩 평균이 관측 평균의 ±0.0002 이내(원형이므로 무편향).
    V2  pooled Total 델타가 C1N90 의 +0.001890 과 ±0.00005 이내.

  사전확약 (V1·V2 통과시에만 판정):
    H1  일별 평균 델타가 **양수**.
    H2  L=7 에서 CI 가 **0 을 제외**한다. C1N60·C1N73 은 전 구간에서 0 을 포함했다.
    H3  L=30(월 규모)에서도 0 을 제외한다. 결론이 블록길이에 의존하지 않는다.
    H4  **양수 일 비율이 C1N60·C1N73 보다 높다.** 계산 결함 수정이므로 방향이
        일관돼야 한다. 이것이 기전 검정이다.

  H2 가 참이면 **이 세션에서 처음으로 0 과 구분되는 효과**가 된다. 거짓이면 이산화
  교정도 잡음 수준이고, 동결 게이트가 세 번째로 옳았던 것이 된다.

**진단 전용.** 게이트 미수정 — 승격 판정은 동결 게이트가 한다. 이 노드는 "실재하는가"
만 묻는다.
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

from m271_cycle44_sharpened_decision import sharpen
from m271_cycle65_wind_limited_bound import ELIGIBLE
from m271_decision_surface import load_surface
from m271_evaluate_candidate import official
from m271_n4_within_bin import (
    DECISION_GRID,
    point_weights,
    uniform_weights,
    utility_matrix,
)

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
N4_RECEIPT = REPORTS / "m271_n4_within_bin_receipt.json"
REPORT_MD = REPORTS / "m271_n4b_establish.md"
RECEIPT = REPORTS / "m271_n4b_establish_receipt.json"

NODE_ID = "C1N91_WITHIN_BIN_ESTABLISHED"
LANE = "L4"
PARENT_NODE = "C1N90_WITHIN_BIN_INTEGRATION"

BLOCK_GRID = (1, 2, 3, 5, 7, 10, 14, 21, 30)
DRAWS = 4000
SEED = 20260805
BIAS_TOLERANCE = 0.0002
POOLED_TOLERANCE = 0.00005


def main() -> int:
    store, info = load_surface()
    n4 = json.loads(N4_RECEIPT.read_text(encoding="utf-8"))
    chosen = n4["chosen_temperature"]

    util_point = utility_matrix(point_weights())
    util_uniform = utility_matrix(uniform_weights())

    parts: list[pd.DataFrame] = []
    for fold in sorted(store):
        entry = store[fold]
        temperature = float(chosen["point"][fold])
        temperature_u = float(chosen["uniform"][fold])
        p_sharp = sharpen(entry["probability"], temperature)
        u_sharp = sharpen(entry["probability"], temperature_u)
        frame = entry["meta"].copy()
        frame["group_id"] = entry["group"]
        frame["capacity"] = entry["capacity"]
        frame["point"] = DECISION_GRID[
            np.argmax(p_sharp @ util_point.T, axis=1)
        ] * entry["capacity"]
        frame["uniform"] = DECISION_GRID[
            np.argmax(u_sharp @ util_uniform.T, axis=1)
        ] * entry["capacity"]
        frame["day"] = frame["forecast_kst_dtm"].dt.normalize()
        parts.append(frame)
    data = pd.concat(parts, ignore_index=True).sort_values("day").reset_index(drop=True)

    data["eligible"] = data["actual_kwh"] >= ELIGIBLE * data["capacity"]
    scorable = (
        data.loc[data["eligible"]].groupby("day")["group_id"].nunique()
        .pipe(lambda s: s[s == 3]).index
    )
    data = data.loc[data["day"].isin(scorable)].reset_index(drop=True)
    days = np.sort(data["day"].unique())
    day_index = {
        day: np.asarray(idx, dtype="int64")
        for day, idx in data.groupby("day").indices.items()
    }

    def total_delta(frame: pd.DataFrame) -> float:
        base = frame.loc[:, ["forecast_kst_dtm", "group_id", "actual_kwh"]].copy()
        base["forecast_id"] = np.arange(len(base), dtype="int64")
        cand = base.copy()
        cand["prediction_kwh"] = frame["uniform"].to_numpy(float)
        parent = base.copy()
        parent["prediction_kwh"] = frame["point"].to_numpy(float)
        return float(official(cand)["total"] - official(parent)["total"])

    pooled = total_delta(data)
    daily = np.asarray(
        [total_delta(data.take(day_index[d])) for d in days], dtype="float64"
    )
    observed_mean = float(daily.mean())
    positive_days = int((daily > 0).sum())

    n_days = len(days)
    rng = np.random.default_rng(SEED)
    results: dict[int, dict[str, float]] = {}
    for length in BLOCK_GRID:
        n_blocks = int(np.ceil(n_days / length))
        starts = rng.integers(0, n_days, size=(DRAWS, n_blocks))
        offsets = np.arange(length)
        idx = (starts[:, :, None] + offsets[None, None, :]).reshape(DRAWS, -1)
        idx = (idx % n_days)[:, :n_days]
        draws = daily[idx].mean(axis=1)
        results[length] = {
            "mean": float(draws.mean()),
            "bias": float(draws.mean() - observed_mean),
            "q05": float(np.quantile(draws, 0.05)),
            "q95": float(np.quantile(draws, 0.95)),
            "excludes_zero": bool(np.quantile(draws, 0.05) > 0.0),
        }

    v1 = bool(all(abs(r["bias"]) <= BIAS_TOLERANCE for r in results.values()))
    v2 = bool(abs(pooled - float(n4["best_gain"])) <= POOLED_TOLERANCE)

    h1 = bool(observed_mean > 0.0)
    h2 = bool(results[7]["excludes_zero"])
    h3 = bool(results[30]["excludes_zero"])
    h4 = bool(positive_days / n_days > 0.5)

    if not (v1 and v2):
        verdict = "GUARD_FAILED_RESULT_VOID"
    elif h2 and h3:
        verdict = "EFFECT_ESTABLISHED_AT_ALL_BLOCK_LENGTHS"
    elif h2:
        verdict = "EFFECT_ESTABLISHED_AT_SHORT_BLOCKS_ONLY"
    else:
        verdict = "EFFECT_NOT_ESTABLISHED"

    payload: dict[str, Any] = {
        "node_id": NODE_ID,
        "lane": LANE,
        "parent": PARENT_NODE,
        "candidate": False,
        "diagnostic_only": True,
        "method": "CIRCULAR_BLOCK_BOOTSTRAP (Politis & Romano 1992), C1N76 과 동일",
        "surface": info,
        "days": int(n_days),
        "draws": DRAWS,
        "seed": SEED,
        "pooled_delta": pooled,
        "n4_best_gain": float(n4["best_gain"]),
        "daily_mean": observed_mean,
        "daily_sd": float(daily.std(ddof=1)),
        "positive_days": positive_days,
        "positive_fraction": positive_days / n_days,
        "bootstrap": {str(k): v for k, v in results.items()},
        "checks": {"V1_unbiased": v1, "V2_pooled_matches_n4": v2},
        "hypotheses": {
            "H1_daily_mean_positive": h1,
            "H2_excludes_zero_at_L7": h2,
            "H3_excludes_zero_at_L30": h3,
            "H4_majority_days_positive": h4,
        },
        "comparison": {
            "c60_level_temperature": "전 블록길이에서 0 포함",
            "c73_group_blend": "전 블록길이에서 0 포함",
        },
        "verdict": verdict,
        "dacon_upload": False,
        "lockbox_used": False,
        "generated_at": datetime.now(UTC).isoformat(),
    }
    payload["digest"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]
    RECEIPT.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str),
                       encoding="utf-8")

    lines = [
        "# M271 N4b — 구간내부 적분 이득의 실재성 검정",
        "",
        f"노드 `{NODE_ID}` / 레인 {LANE} / 부모 `{PARENT_NODE}` / **진단 전용**",
        "",
        f"일수 {n_days} / 추첨 {DRAWS:,} / 시드 {SEED} / 원형 블록",
        "",
        f"pooled 델타 **{pooled:+.6f}** (C1N90 {float(n4['best_gain']):+.6f}) / "
        f"일별 평균 **{observed_mean:+.6f}** / sd {float(daily.std(ddof=1)):.6f} / "
        f"양수 일 **{positive_days}/{n_days}** ({positive_days / n_days:.1%})",
        "",
        "## 1. 블록길이 스캔 (90% 구간)",
        "",
        "| L | q05 | q95 | 0 제외 | 편향 |",
        "|---:|---:|---:|:---:|---:|",
    ]
    for length in BLOCK_GRID:
        r = results[length]
        lines.append(
            f"| {length} | {r['q05']:+.6f} | {r['q95']:+.6f} | "
            f"{'**O**' if r['excludes_zero'] else '-'} | {r['bias']:+.7f} |"
        )
    lines += [
        "",
        "C1N60(수준온도)·C1N73(그룹결합)은 **전 블록길이에서 0 을 포함**했다.",
        "",
        "## 2. 사전확약",
        "",
        f"- V1 전 길이 무편향 -> **{v1}**",
        f"- V2 pooled 이 C1N90 과 일치 -> **{v2}**",
        f"- H1 일별 평균 양수 -> **{h1}**",
        f"- H2 L=7 에서 0 제외 -> **{h2}**",
        f"- H3 L=30 에서 0 제외 -> **{h3}**",
        f"- H4 양수 일이 과반 -> **{h4}**",
        "",
        "## 3. 판정",
        "",
        f"**{verdict}**",
        "",
        f"digest `{payload['digest']}`",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("=== N4b 완료 ===")
    print(f"[N4b] pooled {pooled:+.6f} (C1N90 {float(n4['best_gain']):+.6f}) / "
          f"일별 평균 {observed_mean:+.6f} / sd {float(daily.std(ddof=1)):.6f}")
    print(f"[N4b] 양수 일 {positive_days}/{n_days} ({positive_days/n_days:.1%})")
    for length in BLOCK_GRID:
        r = results[length]
        print(f"[N4b] L={length:2d}  [{r['q05']:+.6f}, {r['q95']:+.6f}] "
              f"{'0제외' if r['excludes_zero'] else '     '}  편향 {r['bias']:+.7f}")
    print(f"[N4b] V1 {v1} / V2 {v2} / H1 {h1} / H2 {h2} / H3 {h3} / H4 {h4}")
    print(f"[N4b] 판정: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
