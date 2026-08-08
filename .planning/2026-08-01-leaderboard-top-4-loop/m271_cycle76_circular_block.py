"""M271 P4 사이클 76 — 원형 블록으로 가장자리 과소표집을 없앤다.

사이클 74·75 가 연달아 V1 으로 무효였다. 부트스트랩 평균이 관측값에 앉지 않고
그 편향이 블록길이를 따라 커졌다(L=30 에서 -0.0022 ~ -0.0047).

**C74 의 진단(추정량의 비선형성)은 틀렸다.** C75 가 추정량을 선형(일별 델타의 평균)으로
바꿨는데도 같은 편향이 남았다. 비선형성이 원인이었다면 사라졌어야 한다.

원인은 재표집 구조다. 이동블록 부트스트랩에서 날짜 `i` 를 덮는 블록 시작점의 수는
`i` 가 계열의 **가장자리**에 가까울수록 적다. 따라서 양 끝 날이 과소표집되고, 그 정도가
블록길이에 비례해 커진다. 그리고 이 개발면은 2023-04 에서 시작해 2023-12 에서 끝나는데
**가장 큰 양수 델타를 내는 두 달이 정확히 양 끝**이다(C1N62·C1N73 이 이미 그 둘을
지목했다). 그래서 편향이 음수이고 L 을 따라 커진다.

표준 해법은 **원형 블록 부트스트랩**이다(Politis & Romano 1992) — 계열을 원으로 이어
붙여 모든 관측의 선택 확률을 같게 만든다. 그러면 편향이 구조적으로 0 이 된다.

**① 방법 리서치**

  - Politis & Romano(1992) 의 circular block bootstrap 이 정확히 이 가장자리 문제를
    없애려고 제안됐다. Lahiri(2003) 가 이동블록 대비 편향 우위를 정리했다.
  - 구현 차이는 한 줄이다 — 시작점을 `[0, n)` 전체에서 뽑고 인덱스를 `% n` 으로 감는다.
  - 추정량은 C75 와 동일하게 **일별 델타의 평균**(선형)을 쓴다. 동결 게이트가
    월별 델타의 평균을 쓰는 것과 같은 계열이고, 단위만 고르다.
  - **채택**: 원형 블록 부트스트랩 + 블록길이 스캔. 적합 없음.

**② 사양 동결**

  대상·단위·통계량·격자·시드   C75 와 **완전히 동일**. 바뀌는 것은 재표집이 원형이라는
                              것 하나뿐이고, 그래야 C75 와의 차이가 그 하나에 귀속된다.
  추첨                        4000 회.

  **타당성 가드**
    V1  **모든** 블록길이에서 `|부트스트랩 평균 - 관측 평균| <= 0.0002`.
        원형이면 선택확률이 균일하므로 성립해야 한다. C74·C75 가 실패한 그 가드이고,
        이번에 통과하면 진단(가장자리 과소표집)이 옳다는 뜻이다.
    V2  L=1 결과가 C75 의 L=1 과 ±0.0005 이내. L=1 에서는 원형/이동이 같으므로
        두 구현이 그 지점에서 일치해야 한다.

  사전확약 (V1·V2 통과시에만 판정):
    H1  CI 폭이 `L` 을 따라 평탄해진다 — L=21 과 L=30 의 폭 차이가 L=1 과 L=7 의
        차이보다 작다.
    H2  L=7 에서 **C60 의 CI 가 0 을 제외**한다.
    H3  L=7 에서 **C73 의 CI 가 0 을 제외**한다.
    H4  L=30 에서도 H2·H3 의 결론이 유지된다.

  **부호를 예단하지 않는다.** 다만 C75 의 L=1(가장자리 편향이 0 인 지점)에서 이미
  두 대상 모두 0 을 포함했으므로, 편향을 고쳐도 H2·H3 이 참이 될 근거는 약하다.
  그렇다면 결론은 **블로킹이 아니라 효과가 없다**는 쪽이고, 그 자체가 답이다.

  **동결 게이트를 수정하지 않는다.** 승격 판정을 내리지 않는다.

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
C74_RECEIPT = REPORTS / "m271_cycle74_block_length_receipt.json"
REPORT_MD = REPORTS / "m271_cycle76_circular_block.md"
C75_RECEIPT = REPORTS / "m271_cycle75_daily_gate_estimand_receipt.json"
RECEIPT = REPORTS / "m271_cycle76_circular_block_receipt.json"

NODE_ID = "C1N76_CIRCULAR_BLOCK"
LANE = "L4"
PARENT_NODE = "C1N75_DAILY_GATE_ESTIMAND"

BLOCK_GRID = (1, 2, 3, 5, 7, 10, 14, 21, 30)
DRAWS = 4000
SEED = 20260805
BIAS_TOLERANCE = 0.0002
PLATEAU_CHECK = 7
GROUPS = (1, 2, 3)
TARGETS = (("c73_blend", "blend"), ("c60_level", "level"))


def main() -> int:
    store, info = load_surface()
    c57 = json.loads(C57_RECEIPT.read_text(encoding="utf-8"))["result"]
    c60 = json.loads(C60_RECEIPT.read_text(encoding="utf-8"))
    c73 = json.loads(C73_RECEIPT.read_text(encoding="utf-8"))
    c74 = json.loads(C74_RECEIPT.read_text(encoding="utf-8"))
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

        global_rate = bayes_decision(sharpen_by_row(prob, np.full(len(capacity), g_t)))
        level = level_of(global_rate)
        level_rate = bayes_decision(
            sharpen_by_row(prob, np.asarray([l_t[int(v)] for v in level], dtype=float))
        )
        curve_rate = np.zeros(len(capacity), dtype="float64")
        for group, (cv, cp) in curves.items():
            mask = entry["group"] == group
            curve_rate[mask] = np.interp(
                entry["sitewind"][mask], cv, cp, left=0.0, right=cp[-1]
            )
        alpha = np.asarray([a_t[int(g)] for g in entry["group"]], dtype="float64")

        frame = entry["meta"].copy()
        frame["group_id"] = entry["group"]
        frame["capacity"] = capacity
        frame["model"] = global_rate * capacity
        frame["level"] = level_rate * capacity
        frame["blend"] = (alpha * global_rate + (1.0 - alpha) * curve_rate) * capacity
        frame["day"] = frame["forecast_kst_dtm"].dt.normalize()
        parts.append(frame)
    data = pd.concat(parts, ignore_index=True).sort_values("day").reset_index(drop=True)

    data["eligible"] = data["actual_kwh"] >= 0.10 * data["capacity"]
    scorable = (
        data.loc[data["eligible"]].groupby("day")["group_id"].nunique()
        .pipe(lambda s: s[s == 3]).index
    )
    data = data.loc[data["day"].isin(scorable)].reset_index(drop=True)
    days = np.sort(data["day"].unique())
    by_day = {day: block for day, block in data.groupby("day")}

    def total_delta(frame: pd.DataFrame, candidate: str) -> float:
        base = frame.loc[:, ["forecast_kst_dtm", "group_id", "actual_kwh"]].copy()
        base["forecast_id"] = np.arange(len(base), dtype="int64")
        cand = base.copy()
        cand["prediction_kwh"] = frame[candidate].to_numpy(float)
        parent = base.copy()
        parent["prediction_kwh"] = frame["model"].to_numpy(float)
        return float(official(cand)["total"] - official(parent)["total"])

    # 일별 델타를 **한 번만** 계산한다. 이후 부트스트랩은 이 벡터의 평균만 다시 잡는다
    # — 선형이므로 편향이 없고, 재표집마다 공식 산식을 다시 돌 필요도 없다.
    daily = {name: np.asarray(
        [total_delta(by_day[d], column) for d in days], dtype="float64"
    ) for name, column in TARGETS}
    observed_mean = {name: float(v.mean()) for name, v in daily.items()}
    pooled = {name: total_delta(data, column) for name, column in TARGETS}
    c75 = json.loads(C75_RECEIPT.read_text(encoding="utf-8"))
    sign_ok = all(
        np.sign(observed_mean[name]) == np.sign(pooled[name]) for name, _ in TARGETS
    )

    n_days = len(days)
    rng = np.random.default_rng(SEED)
    results: dict[str, dict[int, dict[str, float]]] = {n: {} for n, _ in TARGETS}
    for length in BLOCK_GRID:
        n_blocks = int(np.ceil(n_days / length))
        # 원형 블록: 시작점을 [0, n) 전체에서 뽑고 인덱스를 감는다. 모든 날이
        # 정확히 `length` 개의 시작점에 덮이므로 선택확률이 균일하고 편향이 0 이다.
        starts = rng.integers(0, n_days, size=(DRAWS, n_blocks))
        offsets = np.arange(length)
        idx = (starts[:, :, None] + offsets[None, None, :]).reshape(DRAWS, -1)
        idx = (idx % n_days)[:, :n_days]
        for name, _ in TARGETS:
            draws = daily[name][idx].mean(axis=1)
            results[name][length] = {
                "mean": float(draws.mean()),
                "bias": float(draws.mean() - observed_mean[name]),
                "q05": float(np.quantile(draws, 0.05)),
                "q50": float(np.quantile(draws, 0.50)),
                "q95": float(np.quantile(draws, 0.95)),
                "width": float(np.quantile(draws, 0.95) - np.quantile(draws, 0.05)),
                "excludes_zero": bool(np.quantile(draws, 0.05) > 0.0),
            }

    v1 = bool(all(
        abs(results[name][length]["bias"]) <= BIAS_TOLERANCE
        for name, _ in TARGETS for length in BLOCK_GRID
    ))
    # V2 — L=1 에서는 원형과 이동이 같은 절차이므로 C75 와 일치해야 한다.
    v2 = bool(sign_ok and all(
        abs(results[name][1]["q05"] - float(c75["bootstrap"][name]["1"]["q05"])) <= 0.0005
        and abs(results[name][1]["q95"] - float(c75["bootstrap"][name]["1"]["q95"])) <= 0.0005
        for name, _ in TARGETS
    ))

    def width(name: str, length: int) -> float:
        return results[name][length]["width"]

    h1 = bool(all(
        abs(width(n, 30) - width(n, 21)) < abs(width(n, PLATEAU_CHECK) - width(n, 1))
        for n, _ in TARGETS
    ))
    h2 = bool(results["c60_level"][PLATEAU_CHECK]["excludes_zero"])
    h3 = bool(results["c73_blend"][PLATEAU_CHECK]["excludes_zero"])
    h4 = bool(
        results["c60_level"][30]["excludes_zero"] == h2
        and results["c73_blend"][30]["excludes_zero"] == h3
    )

    if not v1 or not v2:
        verdict = "ESTIMAND_GUARD_FAILED_RESULT_VOID"
    elif not h1:
        verdict = "CORRELATION_EXCEEDS_MONTH_SCALE_NO_SAFE_BLOCK_LENGTH"
    elif h2 and h3:
        verdict = "BOTH_EFFECTS_ESTABLISHED_ON_DAILY_ESTIMAND"
    elif h2 or h3:
        verdict = "ONE_EFFECT_ESTABLISHED_ON_DAILY_ESTIMAND"
    else:
        verdict = "NEITHER_EFFECT_ESTABLISHED_EVEN_AT_DAILY_GRANULARITY"

    payload: dict[str, Any] = {
        "node_id": NODE_ID,
        "lane": LANE,
        "parent": PARENT_NODE,
        "candidate": False,
        "diagnostic_only": True,
        "method": "MOVING_BLOCK_BOOTSTRAP of a LINEAR estimand (Kunsch 1989)",
        "estimand": "mean of daily Total deltas (게이트의 월별 평균과 같은 계열)",
        "surface": info,
        "days": int(n_days),
        "draws": DRAWS,
        "seed": SEED,
        "observed_daily_mean": observed_mean,
        "observed_pooled": pooled,
        "daily_sd": {n: float(daily[n].std(ddof=1)) for n, _ in TARGETS},
        "integrated_time_c74": c74["integrated_time"],
        "bootstrap": {n: {str(k): v for k, v in results[n].items()} for n in results},
        "gate_monthly_q05": {
            "c73_blend": float(c73["gate"]["bootstrap_q05"]),
        },
        "checks": {"V1_unbiased_all_lengths": v1, "V2_matches_c75_at_L1": v2,
                   "sign_matches_pooled": bool(sign_ok)},
        "hypotheses": {
            "H1_width_plateaus": h1,
            "H2_c60_excludes_zero": h2,
            "H3_c73_excludes_zero": h3,
            "H4_stable_at_month_scale": h4,
        },
        "note": (
            "동결 게이트를 수정하지 않는다. 게이트는 월별 델타의 **평균**(선형)을 "
            "부트스트랩하므로 편향되지 않았다 — C1N74 의 편향은 pooled Total(비선형)을 "
            "재계산한 내 구성에서 나온 것이다."
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
        "# M271 P4 사이클 75 — 일별 델타 평균(게이트와 같은 계열의 선형 추정량)",
        "",
        f"노드 `{NODE_ID}` / 레인 {LANE} / 부모 `{PARENT_NODE}` / **진단 전용**",
        "",
        f"일수 {n_days} / 추첨 {DRAWS:,} / 시드 {SEED}",
        "",
        "| 대상 | 일별 평균 델타 | 일별 sd | pooled Total 델타 | 적분시간(C74) |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, _ in TARGETS:
        lines.append(
            f"| {name} | **{observed_mean[name]:+.6f}** | "
            f"{float(daily[name].std(ddof=1)):.6f} | {pooled[name]:+.6f} | "
            f"{c74['integrated_time'][name]:.2f}일 |"
        )
    lines += [
        "",
        "## 1. 블록길이 스캔 (90% 구간)",
        "",
        "| L | C60 q05 | C60 q95 | 폭 | 0제외 | C73 q05 | C73 q95 | 폭 | 0제외 |",
        "|---:|---:|---:|---:|:---:|---:|---:|---:|:---:|",
    ]
    for length in BLOCK_GRID:
        a = results["c60_level"][length]
        b = results["c73_blend"][length]
        lines.append(
            f"| {length} | {a['q05']:+.6f} | {a['q95']:+.6f} | {a['width']:.6f} | "
            f"{'O' if a['excludes_zero'] else '-'} | {b['q05']:+.6f} | "
            f"{b['q95']:+.6f} | {b['width']:.6f} | "
            f"{'O' if b['excludes_zero'] else '-'} |"
        )
    lines += [
        "",
        "## 2. 편향 (선형 추정량이므로 0 이어야 한다)",
        "",
        "| L | C60 편향 | C73 편향 |",
        "|---:|---:|---:|",
    ]
    for length in BLOCK_GRID:
        lines.append(
            f"| {length} | {results['c60_level'][length]['bias']:+.7f} | "
            f"{results['c73_blend'][length]['bias']:+.7f} |"
        )
    lines += [
        "",
        "C1N74·C1N75 는 같은 자리에서 -0.0025 ~ -0.0047 이었다. 추정량을 선형으로 바꾼 것이 "
        "차이의 전부다.",
        "",
        "## 3. 사전확약",
        "",
        f"- V1 전 블록길이에서 편향 <= {BIAS_TOLERANCE} -> **{v1}**",
        f"- V2 L=1 이 C75 와 일치 + 부호 정합 -> **{v2}**",
        f"- H1 폭이 평탄해진다 -> **{h1}**",
        f"- H2 L={PLATEAU_CHECK} 에서 C60 이 0 제외 -> **{h2}**",
        f"- H3 L={PLATEAU_CHECK} 에서 C73 이 0 제외 -> **{h3}**",
        f"- H4 L=30 에서도 동일 -> **{h4}**",
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
    print(f"[C76] 일수 {n_days}")
    for name, _ in TARGETS:
        print(f"[C76] {name}  일별 평균 {observed_mean[name]:+.6f} / sd "
              f"{float(daily[name].std(ddof=1)):.6f} / pooled {pooled[name]:+.6f}")
    for length in BLOCK_GRID:
        a = results["c60_level"][length]
        b = results["c73_blend"][length]
        print(f"[C76] L={length:2d}  C60 [{a['q05']:+.6f}, {a['q95']:+.6f}] "
              f"{'0제외' if a['excludes_zero'] else '     '} 편향 {a['bias']:+.7f}  |  "
              f"C73 [{b['q05']:+.6f}, {b['q95']:+.6f}] "
              f"{'0제외' if b['excludes_zero'] else '     '} 편향 {b['bias']:+.7f}")
    print(f"[C76] V1 {v1} / V2 {v2} / H1 {h1} / H2 {h2} / H3 {h3} / H4 {h4}")
    print(f"[C76] 판정: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
