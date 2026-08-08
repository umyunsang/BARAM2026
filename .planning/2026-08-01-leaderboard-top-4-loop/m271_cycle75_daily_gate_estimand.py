"""M271 P4 사이클 75 — 게이트와 같은 추정량을 더 고운 단위로: 일별 델타의 평균.

사이클 74 는 V2 로 무효였다. 부트스트랩 평균이 관측값에 앉지 않았고(L=30 에서
-0.0025/-0.0047) 그래서 긴 블록의 분위수를 읽을 수 없었다.

원인을 잘못 일반화했었다. **동결 게이트는 편향되지 않았다.** 게이트가 부트스트랩하는
것은 `m270_monthly_validation.py:118` 에서 보듯 **월별 델타의 평균** — 9 개 수의 평균,
즉 **선형** 통계량이다. 선형이면 `E*[mean*] = mean` 이 정확히 성립한다.

편향은 내 C74 구성에서 나왔다. 나는 **날짜를 재표집해 pooled Total 을 다시 계산**했고,
Total 은 비선형이다(NMAE 는 합의 비, FICR 은 발전량 가중). 그래서 재표집 구성이
흔들릴수록 기대값이 관측값에서 멀어졌다.

그러면 고칠 방법이 BCa 가 아니다. **게이트와 같은 추정량을 쓰되 단위만 고르게 한다.**

    게이트   월별 Total 델타의 평균,  n = 9
    이 노드  **일별** Total 델타의 평균, n = 228,  블록 재표집

선형이므로 편향이 0 이고, 게이트의 구성과 같은 계열이라 비교가 성립한다.

**① 방법 리서치**

  - 표본 단위를 고르게 하면 검정력이 오르지만 **의존성**이 들어온다. 그래서 단순
    복원추출이 아니라 이동블록 부트스트랩을 쓴다(Kunsch 1989).
  - 블록 길이는 C74 가 측정한 적분 자기상관 시간(C60 2.35 일, C73 1.00 일)을 근거로
    고르되, **격자를 훑어 폭이 평탄해지는지** 함께 본다. 알고리즘 하나에 걸지 않는다.
  - 추정량이 선형이므로 편향 보정(BCa 등)이 불필요하다 — V1 이 그것을 검정한다.
  - **채택**: 일별 델타 평균 + 이동블록 부트스트랩 + 블록길이 스캔. 적합 없음.

**② 사양 동결**

  대상   `C73` GROUP_ALPHA - MODEL, `C60` LEVEL - GLOBAL. C74 와 동일.
  단위   일(KST). 세 그룹 모두 **유효행**이 있는 날만 — C74 와 같은 228 일.
  통계량 `mean_d (Total_후보(d) - Total_부모(d))`. 각 날을 **동등 가중**한다.
         게이트가 각 달을 동등 가중하는 것과 같은 규약이고, 발전량이 큰 소수의
         날에 끌려가지 않는다.
  절차   길이 `L` 일의 연속 블록을 복원추출로 이어 붙여 228 일을 만들고 **그 날들의
         일별 델타를 평균**한다. pooled Total 을 다시 계산하지 않는다 — 그것이
         C74 의 편향원이었다.
  격자   `L` = 1, 2, 3, 5, 7, 10, 14, 21, 30 — C74 와 동일, 실행 전 동결.
  추첨   4000 회, 시드 20260805.

  **타당성 가드**
    V1  **모든** 블록길이에서 `|부트스트랩 평균 - 관측 평균| <= 0.0002`.
        선형 추정량이므로 성립해야 한다. C74 에서 실패한 바로 그 가드이고,
        이번에 통과하면 진단(추정량의 비선형성)이 옳았다는 뜻이다.
    V2  일별 델타의 평균 부호가 pooled Total 델타의 부호와 일치(두 대상 모두).
        다르면 두 추정량이 다른 것을 재는 것이므로 게이트와의 비교가 성립하지 않는다.

  사전확약 (V1·V2 통과시에만 판정):
    H1  CI 폭이 `L` 에 대해 단조 증가하다 **평탄해진다** — L=21 과 L=30 의 폭 차이가
        L=1 과 L=7 의 차이보다 작다.
    H2  L=7(두 대상의 적분시간 1.00·2.35 일의 3 배 이상)에서 **C60 의 CI 가 0 을 제외**.
    H3  같은 조건에서 **C73 의 CI 가 0 을 제외**.
    H4  L=30(월 규모)에서도 H2·H3 이 유지된다. 유지되면 결론이 블록길이 선택에
        의존하지 않는다.

  **부호를 예단하지 않는다.** H2·H3 은 검정이지 예측이 아니다. 다만 H1 이 거짓이면
  (폭이 계속 커지면) 상관 스케일이 월을 넘는다는 뜻이고 그때는 **어떤 블록길이도
  안전하지 않으므로** H2~H4 를 판정 근거로 쓰지 않는다.

  **동결 게이트를 수정하지 않는다.** 이 노드는 승격 판정을 내리지 않으며, 게이트가
  보수적인지 여부를 자료로 말할 뿐이다.

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
REPORT_MD = REPORTS / "m271_cycle75_daily_gate_estimand.md"
RECEIPT = REPORTS / "m271_cycle75_daily_gate_estimand_receipt.json"

NODE_ID = "C1N75_DAILY_GATE_ESTIMAND"
LANE = "L4"
PARENT_NODE = "C1N74_BLOCK_LENGTH"

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
    v2 = bool(all(
        np.sign(observed_mean[name]) == np.sign(pooled[name]) for name, _ in TARGETS
    ))

    n_days = len(days)
    rng = np.random.default_rng(SEED)
    results: dict[str, dict[int, dict[str, float]]] = {n: {} for n, _ in TARGETS}
    for length in BLOCK_GRID:
        n_blocks = int(np.ceil(n_days / length))
        starts = rng.integers(
            0, max(n_days - length + 1, 1), size=(DRAWS, n_blocks)
        )
        offsets = np.arange(length)
        idx = (starts[:, :, None] + offsets[None, None, :]).reshape(DRAWS, -1)
        idx = np.clip(idx, 0, n_days - 1)[:, :n_days]
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
        "checks": {"V1_unbiased_all_lengths": v1, "V2_sign_matches_pooled": v2},
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
        "C1N74 는 같은 자리에서 -0.0025 ~ -0.0047 이었다. 추정량을 선형으로 바꾼 것이 "
        "차이의 전부다.",
        "",
        "## 3. 사전확약",
        "",
        f"- V1 전 블록길이에서 편향 <= {BIAS_TOLERANCE} -> **{v1}**",
        f"- V2 일별 평균 부호가 pooled 와 일치 -> **{v2}**",
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
    print(f"[C75] 일수 {n_days}")
    for name, _ in TARGETS:
        print(f"[C75] {name}  일별 평균 {observed_mean[name]:+.6f} / sd "
              f"{float(daily[name].std(ddof=1)):.6f} / pooled {pooled[name]:+.6f}")
    for length in BLOCK_GRID:
        a = results["c60_level"][length]
        b = results["c73_blend"][length]
        print(f"[C75] L={length:2d}  C60 [{a['q05']:+.6f}, {a['q95']:+.6f}] "
              f"{'0제외' if a['excludes_zero'] else '     '} 편향 {a['bias']:+.7f}  |  "
              f"C73 [{b['q05']:+.6f}, {b['q95']:+.6f}] "
              f"{'0제외' if b['excludes_zero'] else '     '} 편향 {b['bias']:+.7f}")
    print(f"[C75] V1 {v1} / V2 {v2} / H1 {h1} / H2 {h2} / H3 {h3} / H4 {h4}")
    print(f"[C75] 판정: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
