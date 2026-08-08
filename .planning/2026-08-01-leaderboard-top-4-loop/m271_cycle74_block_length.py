"""M271 P4 사이클 74 — 검증면을 넓힌다: 월 블록이 자료가 요구하는 길이인가.

사이클 73 이 검정력 한계를 정량화했다. 월 델타 표준편차 0.012~0.016 에 9 개월이므로
표준오차가 0.004~0.005 이고, 지금 테이블에 오르는 +0.005 규모 효과는 세울 수 없다.

**개월 수로는 못 푼다.** 세 그룹 라벨이 모두 있는 구간은 2023-01~2025-01 이고, Q1-2023 은
학습이 2022 뿐인데 g3 라벨이 2022 에 없어 fold 로 쓸 수 없다. 2024 는 소진된 lockbox 다.
무료로 늘릴 달이 **0 개**이고, 9 -> 24 로 가도 표준오차는 39% 만 준다.

그러면 남는 것은 **블로킹**이다. 표준오차 0.004 는 "한 달 = 독립 관측 1 개" 로 세었을
때 나온다. 자료의 실제 상관 스케일이 달이 아니라 일 단위라면 유효 표본이 훨씬 크고
표준오차는 훨씬 작다. 월 블록이 필요 이상으로 길면 **자료에 있는 검정력을 버리고 있는
것**이다.

동결 게이트는 바꾸지 않는다. 그것은 보수적 승격 기준으로 그대로 두고, **"효과가
실재하는가" 를 묻는 계측기를 하나 더 놓는다.**

**① 방법 리서치**

  - 종속 자료의 블록 부트스트랩에서 블록 길이 선택은 확립된 문제다.
    Kunsch(1989) 이동블록, Politis & Romano(1994) 정상부트스트랩, 그리고
    **Politis & White(2004)** 의 자동 블록길이(Patton, Politis & White 2009 정정)가
    표준이다.
  - 자동 선택 알고리즘은 스펙트럼 추정에 의존해 구현 세부가 미묘하다. 여기서는
    **블록 길이를 훑어 신뢰구간이 어디서 평탄해지는지 보는** 더 투명한 절차를 쓴다.
    상관 스케일보다 블록이 길어지면 CI 폭이 더 늘지 않고 평탄해진다 — 그 지점이
    자료가 요구하는 길이다. 알고리즘을 잘못 구현해 숨기는 것보다 낫다.
  - 보조로 일별 델타 계열의 자기상관과 **적분 자기상관 시간**
    `tau = 1 + 2 * sum_k rho_k` 를 보고한다. 유효표본 `n_eff = n / tau`.
  - **채택**: 일 단위 이동블록 부트스트랩 + 블록길이 스캔. 적합 없음(캐시).

**② 사양 동결**

  대상   두 쌍. `C73` GROUP_ALPHA - MODEL, `C60` LEVEL - GLOBAL.
         둘 다 pooled Total 에서 컸으나 월 블록에서 0 과 구분되지 않았다.
  단위   **일**(KST 달력일). 세 그룹이 모두 있는 날만.
  절차   이동블록 부트스트랩 — 길이 `L` 일의 연속 블록을 복원추출로 이어 붙여
         원래 일수만큼 만들고, 그 날들의 행을 모아 **공식 산식을 다시 계산**한다.
         후보와 부모에 **같은 재표집**을 적용해 짝지은 델타를 얻는다.
  격자   `L` = 1, 2, 3, 5, 7, 10, 14, 21, 30 일 — 실행 전 동결.
  추첨   2000 회, 시드 20260805.

  **타당성 가드**
    V1  `L = 30`(월 규모)의 q05 가 동결 게이트의 월별 부트스트랩 q05 와 **같은 부호**.
        월 규모에서 두 계측기가 어긋나면 비교가 성립하지 않는다.
    V2  전 블록길이에서 부트스트랩 평균이 관측 pooled 델타의 ±0.001 이내.
        재표집이 편향되지 않았음을 확인한다.

  사전확약 (V1·V2 통과시에만 판정):
    H1  일별 델타 계열의 적분 자기상관 시간이 **10 일 미만**. 참이면 30 일 블록은
        필요 길이의 3 배 이상이다.
    H2  CI 폭이 `L` 에 대해 **평탄해진다** — L=14 와 L=30 의 폭 차이가 L=1 과 L=14 의
        차이보다 작다. 평탄화가 없으면 상관이 길어 월 블록이 정당하다.
    H3  평탄 구간의 블록길이에서 **C73 의 CI 가 0 을 제외**한다.
    H4  같은 조건에서 **C60 의 CI 도 0 을 제외**한다.

  H1·H2 가 참이고 H3·H4 가 거짓이면 결론은 명확하다 — 블로킹이 문제가 아니라
  효과가 없다. H3·H4 가 참이면 **효과는 실재하되 동결 게이트의 월 블록이 그것을
  탐지하지 못한다**는 뜻이고, 게이트를 바꾸지 않고 그 긴장을 그대로 보고한다.

  **게이트를 수정하지 않는다.** 이 노드는 승격 판정을 내리지 않는다.

lockbox·외부데이터·2024 행 미사용. 제출 없음.
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

from m271_cycle40_band_classifier import bayes_decision
from m271_cycle60_level_temperature import level_of, sharpen_by_row
from m271_cycle65_wind_limited_bound import MIN_ROWS
from m271_cycle67_exact_curve_propagation import build_curve
from m271_decision_surface import load_surface
from m271_evaluate_candidate import official

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
C57_RECEIPT = REPORTS / "m271_cycle57_ficr_ceiling_receipt.json"
C60_RECEIPT = REPORTS / "m271_cycle60_level_temperature_receipt.json"
C73_RECEIPT = REPORTS / "m271_cycle73_group_blend_gate_receipt.json"
REPORT_MD = REPORTS / "m271_cycle74_block_length.md"
RECEIPT = REPORTS / "m271_cycle74_block_length_receipt.json"

NODE_ID = "C1N74_BLOCK_LENGTH"
LANE = "L4"
PARENT_NODE = "C1N73_GROUP_BLEND_GATE"

BLOCK_GRID = (1, 2, 3, 5, 7, 10, 14, 21, 30)
DRAWS = 2000
SEED = 20260805
MEAN_TOLERANCE = 0.001
TAU_CEILING = 10.0
GROUPS = (1, 2, 3)


def integrated_time(series: np.ndarray, max_lag: int = 30) -> tuple[float, list[float]]:
    """적분 자기상관 시간. 첫 음수 자기상관에서 절단(Geyer 의 초기양수열)."""
    x = series - series.mean()
    denom = float((x * x).sum())
    acf = []
    total = 1.0
    for lag in range(1, min(max_lag, len(x) - 2) + 1):
        r = float((x[:-lag] * x[lag:]).sum() / denom) if denom > 0 else 0.0
        acf.append(r)
        if r <= 0:
            break
        total += 2.0 * r
    return total, acf


def main() -> int:
    store, info = load_surface()
    c57 = json.loads(C57_RECEIPT.read_text(encoding="utf-8"))["result"]
    c60 = json.loads(C60_RECEIPT.read_text(encoding="utf-8"))
    c73 = json.loads(C73_RECEIPT.read_text(encoding="utf-8"))
    assert info["probability_digest"] == c60["probability_digest"], "확률면 불일치"

    curves = {
        g: build_curve([b for b in c57["per_group"][str(g)]["bins"]
                        if b["rows"] >= MIN_ROWS])
        for g in GROUPS
    }

    parts: list[pd.DataFrame] = []
    for fold in sorted(store):
        entry = store[fold]
        prob = entry["probability"]
        capacity = entry["capacity"]
        g_t = float(c60["chosen"]["global"][fold])
        l_t = {int(k): float(v) for k, v in c60["chosen"]["level"][fold].items()}
        a_t = {int(k): float(v) for k, v in c73["chosen"]["group"][fold].items()}

        global_rate = bayes_decision(
            sharpen_by_row(prob, np.full(len(capacity), g_t))
        )
        level = level_of(global_rate)
        level_rate = bayes_decision(
            sharpen_by_row(prob, np.asarray([l_t[int(v)] for v in level], dtype=float))
        )
        curve_rate = np.zeros(len(capacity), dtype="float64")
        for group, (cv, cp) in curves.items():
            mask = (entry["group"] == group)
            curve_rate[mask] = np.interp(
                entry["sitewind"][mask], cv, cp, left=0.0, right=cp[-1]
            )
        alpha = np.asarray([a_t[int(g)] for g in entry["group"]], dtype="float64")
        blend_rate = alpha * global_rate + (1.0 - alpha) * curve_rate

        frame = entry["meta"].copy()
        frame["group_id"] = entry["group"]
        frame["capacity"] = capacity
        frame["model"] = global_rate * capacity
        frame["level"] = level_rate * capacity
        frame["blend"] = blend_rate * capacity
        frame["day"] = frame["forecast_kst_dtm"].dt.normalize()
        parts.append(frame)
    data = pd.concat(parts, ignore_index=True).sort_values("day").reset_index(drop=True)

    # 공식 산식은 그룹마다 **유효행**(실제 >= 용량 10%)을 요구한다. 하루 단위로는
    # 저풍속 날에 어떤 그룹의 유효행이 0 이 될 수 있으므로, 채점 가능한 날로
    # 모집단을 맞춘다. 관측 델타·일별 계열·부트스트랩이 모두 같은 날들을 쓴다.
    data["eligible"] = data["actual_kwh"] >= 0.10 * data["capacity"]
    scorable = (
        data.loc[data["eligible"]]
        .groupby("day")["group_id"]
        .nunique()
        .pipe(lambda s: s[s == 3])
        .index
    )
    dropped_days = int(data["day"].nunique() - len(scorable))
    data = data.loc[data["day"].isin(scorable)].reset_index(drop=True)
    days = np.sort(data["day"].unique())
    # 재표집 안쪽에서 프레임을 concat 하면 느리다. 날짜별 **위치 인덱스**만 들고
    # 있다가 한 번의 take 로 뽑는다. 결과는 동일하고 비용만 줄인다.
    day_index = {
        day: np.asarray(idx, dtype="int64")
        for day, idx in data.groupby("day").indices.items()
    }
    by_day = {day: data.take(day_index[day]) for day in days}

    def total_delta(frame: pd.DataFrame, candidate: str) -> float:
        """짝지은 Total 델타. 재표집은 키를 중복시키므로 합성 고유 id 를 준다.

        공식 구현의 `_score_group` 은 `actual_kwh` / `prediction_kwh` / `group_id` 만
        쓰고 `forecast_id` 는 **중복 키 가드에만** 등장한다(`official.py:28`). 따라서
        id 를 갈아끼워도 산식은 바뀌지 않으며, 부트스트랩이 요구하는 복원추출을
        무결성 가드가 막지 않게 할 뿐이다.
        """
        base = frame.loc[:, ["forecast_kst_dtm", "group_id", "actual_kwh"]].copy()
        base["forecast_id"] = np.arange(len(base), dtype="int64")
        cand = base.copy()
        cand["prediction_kwh"] = frame[candidate].to_numpy(float)
        parent = base.copy()
        parent["prediction_kwh"] = frame["model"].to_numpy(float)
        return float(official(cand)["total"] - official(parent)["total"])

    observed = {
        "c73_blend": total_delta(data, "blend"),
        "c60_level": total_delta(data, "level"),
    }

    daily = {name: [] for name in observed}
    for day in days:
        block = by_day[day]
        for name, column in (("c73_blend", "blend"), ("c60_level", "level")):
            daily[name].append(total_delta(block, column))
    daily_series = {name: np.asarray(v, dtype="float64") for name, v in daily.items()}

    taus, acfs = {}, {}
    for name, series in daily_series.items():
        tau, acf = integrated_time(series)
        taus[name] = tau
        acfs[name] = acf
    h1 = bool(all(t < TAU_CEILING for t in taus.values()))

    n_days = len(days)
    results: dict[str, dict[int, dict[str, float]]] = {n: {} for n in observed}
    rng = np.random.default_rng(SEED)
    for length in BLOCK_GRID:
        n_blocks = int(np.ceil(n_days / length))
        draws = {name: [] for name in observed}
        for _ in range(DRAWS):
            starts = rng.integers(0, max(n_days - length + 1, 1), size=n_blocks)
            picked = np.concatenate(
                [np.arange(s, min(s + length, n_days)) for s in starts]
            )[:n_days]
            sample = data.take(
                np.concatenate([day_index[days[i]] for i in picked])
            )
            for name, column in (("c73_blend", "blend"), ("c60_level", "level")):
                draws[name].append(total_delta(sample, column))
        for name in observed:
            arr = np.asarray(draws[name], dtype="float64")
            results[name][length] = {
                "mean": float(arr.mean()),
                "q05": float(np.quantile(arr, 0.05)),
                "q50": float(np.quantile(arr, 0.50)),
                "q95": float(np.quantile(arr, 0.95)),
                "width": float(np.quantile(arr, 0.95) - np.quantile(arr, 0.05)),
                "excludes_zero": bool(np.quantile(arr, 0.05) > 0.0),
            }

    v2 = bool(all(
        abs(results[name][length]["mean"] - observed[name]) <= MEAN_TOLERANCE
        for name in observed for length in BLOCK_GRID
    ))
    gate_q05 = float(c73["gate"]["bootstrap_q05"])
    v1 = bool(np.sign(results["c73_blend"][30]["q05"]) == np.sign(gate_q05))

    def width(name: str, length: int) -> float:
        return results[name][length]["width"]

    h2 = bool(all(
        abs(width(n, 30) - width(n, 14)) < abs(width(n, 14) - width(n, 1))
        for n in observed
    ))
    plateau = 14
    h3 = bool(results["c73_blend"][plateau]["excludes_zero"])
    h4 = bool(results["c60_level"][plateau]["excludes_zero"])

    if not v1 or not v2:
        verdict = "BOOTSTRAP_GUARD_FAILED_RESULT_VOID"
    elif h1 and h2 and (h3 or h4):
        verdict = "MONTHLY_BLOCKS_OVERLONG_EFFECT_ESTABLISHED_AT_DATA_DRIVEN_LENGTH"
    elif h1 and h2:
        verdict = "BLOCKS_OVERLONG_BUT_EFFECT_STILL_NOT_ESTABLISHED"
    else:
        verdict = "MONTHLY_BLOCKING_IS_JUSTIFIED"

    payload: dict[str, Any] = {
        "node_id": NODE_ID,
        "lane": LANE,
        "parent": PARENT_NODE,
        "candidate": False,
        "diagnostic_only": True,
        "method": "MOVING_BLOCK_BOOTSTRAP with length scan (Kunsch 1989; Politis & White 2004)",
        "surface": info,
        "days": int(n_days),
        "days_dropped_not_scorable": dropped_days,
        "draws": DRAWS,
        "seed": SEED,
        "block_grid": list(BLOCK_GRID),
        "observed_delta": observed,
        "integrated_time": taus,
        "acf": {k: v[:12] for k, v in acfs.items()},
        "daily_sd": {k: float(v.std(ddof=1)) for k, v in daily_series.items()},
        "bootstrap": {n: {str(k): v for k, v in results[n].items()} for n in results},
        "gate_monthly_q05": gate_q05,
        "checks": {"V1_sign_agrees_at_month_scale": v1, "V2_bootstrap_unbiased": v2},
        "plateau_length": plateau,
        "hypotheses": {
            "H1_tau_below_ten_days": h1,
            "H2_width_plateaus": h2,
            "H3_c73_excludes_zero_at_plateau": h3,
            "H4_c60_excludes_zero_at_plateau": h4,
        },
        "note": (
            "동결 게이트를 수정하지 않는다. 이 노드는 승격 판정을 내리지 않으며 "
            "'효과가 실재하는가' 만 묻는다."
        ),
        "verdict": verdict,
        "no_training": True,
        "dacon_upload": False,
        "lockbox_used": False,
        "generated_at": datetime.now(UTC).isoformat(),
    }
    payload["digest"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]
    RECEIPT.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    lines = [
        "# M271 P4 사이클 74 — 블록 길이와 검증면의 실효 검정력",
        "",
        f"노드 `{NODE_ID}` / 레인 {LANE} / 부모 `{PARENT_NODE}` / **진단 전용**",
        "",
        f"일수 {n_days} (채점 불가로 제외 {dropped_days}) / 추첨 {DRAWS:,} / 시드 {SEED}",
        "",
        f"관측 pooled 델타 — C73 결합 **{observed['c73_blend']:+.6f}** / "
        f"C60 수준온도 **{observed['c60_level']:+.6f}**",
        "",
        "## 1. 일별 델타의 상관 구조",
        "",
        "| 대상 | 일별 표준편차 | 적분 자기상관 시간 | 자기상관 (lag 1~5) |",
        "|---|---:|---:|---|",
    ]
    for name in observed:
        acf = ", ".join(f"{v:+.3f}" for v in acfs[name][:5])
        lines.append(
            f"| {name} | {float(daily_series[name].std(ddof=1)):.6f} | "
            f"**{taus[name]:.2f}** | {acf} |"
        )
    lines += [
        "",
        "## 2. 블록길이 스캔 (90% 구간)",
        "",
        "| 블록(일) | C73 q05 | C73 q95 | 폭 | 0 제외 | C60 q05 | C60 q95 | 폭 | 0 제외 |",
        "|---:|---:|---:|---:|:---:|---:|---:|---:|:---:|",
    ]
    for length in BLOCK_GRID:
        a = results["c73_blend"][length]
        b = results["c60_level"][length]
        lines.append(
            f"| {length} | {a['q05']:+.6f} | {a['q95']:+.6f} | {a['width']:.6f} | "
            f"{'O' if a['excludes_zero'] else '-'} | {b['q05']:+.6f} | "
            f"{b['q95']:+.6f} | {b['width']:.6f} | "
            f"{'O' if b['excludes_zero'] else '-'} |"
        )
    lines += [
        "",
        f"동결 게이트의 월별 부트스트랩 q05 (C73): {gate_q05:+.6f}",
        "",
        "## 3. 사전확약",
        "",
        f"- V1 L=30 q05 부호가 게이트 월별 q05 와 일치 -> **{v1}**",
        f"- V2 부트스트랩 평균이 관측 델타의 ±{MEAN_TOLERANCE} 이내 -> **{v2}**",
        f"- H1 적분 자기상관 시간 < {TAU_CEILING} 일 -> **{h1}**",
        f"- H2 CI 폭이 평탄해진다 -> **{h2}**",
        f"- H3 L={plateau} 에서 C73 이 0 제외 -> **{h3}**",
        f"- H4 L={plateau} 에서 C60 이 0 제외 -> **{h4}**",
        "",
        "## 4. 판정",
        "",
        f"**{verdict}**",
        "",
        payload["note"],
        "",
        f"digest `{payload['digest']}`",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("=== 완료 ===")
    print(f"[C74] 일수 {n_days} (제외 {dropped_days}) / 관측 델타 C73 {observed['c73_blend']:+.6f} / "
          f"C60 {observed['c60_level']:+.6f}")
    for name in observed:
        print(f"[C74] {name}  일별 sd {float(daily_series[name].std(ddof=1)):.6f}  "
              f"적분시간 {taus[name]:.2f}일  acf1 {acfs[name][0]:+.3f}")
    for length in BLOCK_GRID:
        a = results["c73_blend"][length]
        b = results["c60_level"][length]
        print(f"[C74] L={length:2d}  C73 [{a['q05']:+.6f}, {a['q95']:+.6f}] "
              f"{'0제외' if a['excludes_zero'] else '     '}  |  "
              f"C60 [{b['q05']:+.6f}, {b['q95']:+.6f}] "
              f"{'0제외' if b['excludes_zero'] else ''}")
    print(f"[C74] V1 {v1} / V2 {v2} / H1 {h1} / H2 {h2} / H3 {h3} / H4 {h4}")
    print(f"[C74] 판정: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
