"""M271 N11 — 검색 기반 후보를 **재현 가능하게** 새로 만든다. 조합 축의 유일한 남은 경로.

**C1N100·C1N101 이 표적을 숫자로 좁혔다.**

    M115 대비 오차 상관    GBM 재모수화 0.9767 / 순수 물리 사상 0.9097 / **아날로그 0.8436**

  두 노드가 같은 결론으로 수렴했다 — **공통 원인은 방법론이 아니라 NWP** 다. 풍속 예보
  오차가 모든 경로의 지배적 오차원이라(C1N68 의 27 배 비율) 하류에서 무엇을 바꾸든 오차가
  함께 움직인다. GBM 을 전혀 안 거치는 물리 사상조차 0.91 이다.

  예외는 아날로그 M244 뿐이고, 이유는 NWP 를 **회귀 입력이 아니라 검색 키**로 쓰기
  때문이다. 오차 전파 경로가 구조적으로 다르다.

**왜 강화가 아니라 재구축인가.** M244 는 receipt 가 **빈 키 0 개**라(C1N98) 재생성이
불가능하다. 손댈 수 없으므로 같은 계보를 **재현 계약을 지키는 형태로 새로 만든다**.
성공하면 조합 축이 열리고 M244 의 재현성 결함도 함께 해소된다.

**① 방법 리서치**

  아날로그 예보는 검증된 표준이다 — Lorenz(1969)의 아날로그 개념, 풍력에서는
  Analog Ensemble(AnEn; Delle Monache et al. 2013, *Mon. Wea. Rev.* 141)이 확립된
  방법이다. 핵심은 **예보 시점의 NWP 상태와 유사한 과거 예보를 찾아 그때의 실측 분포를
  그대로 쓰는 것**이며, 회귀와 달리 함수 형태를 가정하지 않는다.

  AnEn 의 표준 설계 두 가지를 따른다.
  - **저차원 검색공간**: 고차원에서는 거리가 무의미해진다(차원의 저주). 원 논문도 소수
    예측자만 쓴다. 여기서는 teacher 풍속 + 시간 위상으로 제한한다.
  - **이웃의 실측 분포를 그대로**: 평균이 아니라 **분포**를 취해야 AnEn 의 이점이 산다.
    그래야 결정층이 쓰는 46 구간 확률면이 나오고 다른 후보와 같은 결정규칙을 쓸 수 있다.

**② 사양 동결**

  하네스   C1N60 과 동일(`m271_harness_cache`). 학습행 = fold 시작 이전 전량.
  검색공간 그룹별로 분리. z 표준화 후 고정 가중:
             `sitewind__mean` (가중 1.0)  — teacher 풍속, 유일한 물리 예측자
             `hour_sin, hour_cos`  (각 0.3) — 일주기 위상
             `doy_sin, doy_cos`    (각 0.3) — 연주기 위상
           **가중은 실행 전 동결하고 결과를 보고 바꾸지 않는다.**
  이웃수   k = 200. 46 구간 히스토그램에 충분한 최소 규모로 고정.
  분포     이웃 rate 의 46 구간 히스토그램 + Laplace 평활(alpha=1). 거리가중 없음
           (거리가중은 자유도이고 이 노드는 최소 사양을 먼저 검정한다).
  결정층   `bayes_decision` + fold-외 온도. **12 후보·모든 선행 노드와 동일.**

  **타당성 가드**
    V1  검색이 학습행만 본다 — 이웃 인덱스의 `forecast_kst_dtm` 이 fold 시작 **이전**.
        위반 0 건. (누출 방지)
    V2  확률행렬 정상(행합 1, NaN 없음), 46 구간.
    V3  12 후보와 조인 시 행 수가 C1N99 의 19,782 와 일치.

  사전확약 (V1~V3 통과시에만 판정):
    H1  M115 대비 오차 상관 **<= 0.85** — C1N100·C1N101 이 세운 필요조건.
    H2  상관이 M244 의 **0.8436 보다 낮다** — 더 강한 탈상관.
    H3  단독 Total 이 M244 의 **0.605760 보다 높다** — 같은 저상관에 더 나은 품질.
    H4  이 후보를 포함한 최선 조합이 C1N99 최선 **0.639389** 를 **검출문턱 0.001013
        이상** 넘는다. **이것만이 조합 축을 여는 조건이다.**

  **부호 예단 없음.** H1 이 거짓이면 검색 기반조차 0.85 를 못 뚫는다는 뜻이고, 그러면
  같은 NWP 위의 조합 축은 **완전히** 닫히며 남는 경로는 외부 NWP 뿐이다.
  H1~H3 이 참이고 H4 가 거짓이면 탈상관만으로는 부족하다는 것이 확정된다.

게이트 미수정. lockbox·외부데이터·`scada_ws` 미사용. 제출 없음.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from m271_cycle13_ensemble import ALL_MODELS, load_model
from m271_cycle37_band_loss import KEYS
from m271_cycle40_band_classifier import CLASS_WIDTH, N_CLASS, bayes_decision
from m271_cycle44_sharpened_decision import TEMPERATURES, sharpen
from m271_evaluate_candidate import official
from m271_harness_cache import fold_frames

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
REPORT_MD = REPORTS / "m271_n11_retrieval_member.md"
RECEIPT = REPORTS / "m271_n11_retrieval_member_receipt.json"

NODE_ID = "C1N102_RETRIEVAL_MEMBER"
LANE = "L7"
PARENT_NODE = "C1N101_PHYSICAL_MEMBER_PROBE"

K_NEIGHBOURS = 200
LAPLACE = 1.0
WEIGHTS = {"sitewind__mean": 1.0, "hour_sin": 0.3, "hour_cos": 0.3,
           "doy_sin": 0.3, "doy_cos": 0.3}
M115_BASELINE = 0.638410
M244_TOTAL = 0.605760
M244_CORR = 0.8436
C1N99_BEST = 0.639389
CORR_TARGET = 0.85
DETECTION_THRESHOLD = 0.001013


def _space(frame: pd.DataFrame) -> np.ndarray:
    dt = pd.to_datetime(frame["forecast_kst_dtm"])
    hour = dt.dt.hour.to_numpy(dtype=float)
    doy = dt.dt.dayofyear.to_numpy(dtype=float)
    cols = {
        "sitewind__mean": frame["sitewind__mean"].to_numpy(dtype=float),
        "hour_sin": np.sin(2 * np.pi * hour / 24.0),
        "hour_cos": np.cos(2 * np.pi * hour / 24.0),
        "doy_sin": np.sin(2 * np.pi * doy / 365.25),
        "doy_cos": np.cos(2 * np.pi * doy / 365.25),
    }
    return cols


def main() -> int:
    store, harness_digest, cache_hit = fold_frames()
    folds = sorted(store)
    leak = 0
    parts: list[pd.DataFrame] = []
    probs: dict[str, np.ndarray] = {}
    metas: dict[str, pd.DataFrame] = {}
    caps: dict[str, np.ndarray] = {}

    for fold in folds:
        train, test = store[fold]["train"], store[fold]["test"]
        cut = pd.to_datetime(test["forecast_kst_dtm"]).min()
        leak += int((pd.to_datetime(train["forecast_kst_dtm"]) >= cut).sum())
        tr_cols, te_cols = _space(train), _space(test)
        prob = np.zeros((len(test), N_CLASS), dtype="float64")
        tr_rate = np.clip(train["rate"].to_numpy(dtype="float64"), 0.0, None)
        tr_bin = np.clip((tr_rate / CLASS_WIDTH).astype(int), 0, N_CLASS - 1)
        tr_g = train["group_id"].to_numpy()
        te_g = test["group_id"].to_numpy()

        for gid in np.unique(te_g):
            tm, em = tr_g == gid, te_g == gid
            if tm.sum() < K_NEIGHBOURS:
                prob[em] = 1.0 / N_CLASS
                continue
            # z 표준화는 **학습행 통계**로만 한다(누출 방지).
            X = np.column_stack([
                w * ((tr_cols[c][tm] - tr_cols[c][tm].mean())
                     / max(float(tr_cols[c][tm].std()), 1e-9))
                for c, w in WEIGHTS.items()
            ])
            Q = np.column_stack([
                w * ((te_cols[c][em] - tr_cols[c][tm].mean())
                     / max(float(tr_cols[c][tm].std()), 1e-9))
                for c, w in WEIGHTS.items()
            ])
            bins_g = tr_bin[tm]
            out = np.zeros((Q.shape[0], N_CLASS))
            for start in range(0, Q.shape[0], 512):
                q = Q[start:start + 512]
                d = ((q[:, None, :] - X[None, :, :]) ** 2).sum(axis=2)
                idx = np.argpartition(d, K_NEIGHBOURS - 1, axis=1)[:, :K_NEIGHBOURS]
                for r in range(q.shape[0]):
                    out[start + r] = np.bincount(bins_g[idx[r]], minlength=N_CLASS)
            out += LAPLACE
            prob[em] = out / out.sum(axis=1, keepdims=True)

        probs[fold] = prob
        metas[fold] = test.loc[:, [*KEYS, "actual_kwh"]].copy()
        caps[fold] = test["capacity"].to_numpy(dtype="float64")

    v1 = leak == 0
    v2 = bool(all(np.isfinite(p).all() and abs(p.sum(1) - 1).max() < 1e-9
                  for p in probs.values()))

    def scored(fold: str, t: float) -> pd.DataFrame:
        out = metas[fold].copy()
        out["prediction_kwh"] = bayes_decision(sharpen(probs[fold], t)) * caps[fold]
        out["month"] = pd.to_datetime(out["forecast_kst_dtm"]).dt.to_period(
            "M").astype(str)
        return out

    picks = {}
    for held in folds:
        others = [f for f in folds if f != held]
        best, bs = TEMPERATURES[0], -np.inf
        for t in TEMPERATURES:
            s = official(pd.concat([scored(f, t) for f in others],
                                   ignore_index=True))["total"]
            if s > bs:
                best, bs = t, s
        picks[held] = float(best)
        parts.append(scored(held, float(best)))
    analog = pd.concat(parts, ignore_index=True)
    solo = official(analog)

    fr = {m: load_model(m) for m in ALL_MODELS}
    fr["ANALOG_V2"] = analog
    j = None
    for m in sorted(fr):
        t = fr[m][[*KEYS, "actual_kwh", "prediction_kwh"]].rename(
            columns={"prediction_kwh": m})
        j = t if j is None else j.merge(t.drop(columns=["actual_kwh"]),
                                        on=KEYS, how="inner")
    j["month"] = pd.to_datetime(j["forecast_kst_dtm"]).dt.to_period("M").astype(str)
    v3 = len(j) == 19782

    err = j[["M115_XGBOOST", "ANALOG_V2"]].to_numpy(float) - j[["actual_kwh"]].to_numpy(float)
    corr = float(np.corrcoef(err.T)[0, 1])

    def sc(cols, agg="med"):
        t = j[[*KEYS, "actual_kwh", "month"]].copy()
        v = j[list(cols)].to_numpy(float)
        t["prediction_kwh"] = np.median(v, axis=1) if agg == "med" else v.mean(axis=1)
        return float(official(t)["total"])

    names = sorted(fr)
    combos = []
    for k in (3, 4, 5):
        for c in itertools.combinations(names, k):
            if "ANALOG_V2" not in c:
                continue
            for agg in ("med", "mean"):
                combos.append((sc(c, agg), agg, list(c)))
    combos.sort(reverse=True)
    best_combo = combos[0]

    valid = v1 and v2 and v3
    if valid:
        h1: bool | None = bool(corr <= CORR_TARGET)
        h2: bool | None = bool(corr < M244_CORR)
        h3: bool | None = bool(solo["total"] > M244_TOTAL)
        h4: bool | None = bool(best_combo[0] - C1N99_BEST >= DETECTION_THRESHOLD)
        if h4:
            verdict = "ENSEMBLE_AXIS_REOPENED_BY_RETRIEVAL_MEMBER"
        elif h1:
            verdict = "RETRIEVAL_DECORRELATES_BUT_COMBINATION_GAIN_SUBTHRESHOLD"
        else:
            verdict = "RETRIEVAL_FAILS_CORRELATION_TARGET_SAME_NWP_AXIS_CLOSED"
    else:
        h1 = h2 = h3 = h4 = None
        verdict = "GUARD_FAILED_RESULT_VOID"

    check = {
        "V1_expectation": "검색이 학습행만 본다(누출 0)", "V1_held": v1, "V1_measured": leak,
        "V2_expectation": "확률행렬 정상", "V2_held": v2,
        "V3_expectation": "조인 19,782 행", "V3_held": v3, "V3_measured": len(j),
        "H1_expectation": f"M115 상관 <= {CORR_TARGET}", "H1_held": h1, "H1_measured": corr,
        "H2_expectation": f"상관 < M244 {M244_CORR}", "H2_held": h2,
        "H3_expectation": f"단독 > M244 {M244_TOTAL}", "H3_held": h3,
        "H3_measured": solo["total"],
        "H4_expectation": f"최선 조합 - {C1N99_BEST} >= {DETECTION_THRESHOLD}",
        "H4_held": h4, "H4_measured": best_combo[0],
        "judged": valid, "verdict": verdict,
    }
    receipt: dict[str, Any] = {
        "node_id": NODE_ID, "lane": LANE, "parent": PARENT_NODE,
        "judged_at": datetime.now(UTC).isoformat(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "harness_digest": harness_digest, "harness_cache_hit": cache_hit,
        "k": K_NEIGHBOURS, "weights": WEIGHTS, "laplace": LAPLACE,
        "temperatures": picks,
        "solo": dict(solo), "correlation_vs_m115": corr,
        "best_combo": {"total": best_combo[0], "agg": best_combo[1],
                       "members": best_combo[2]},
        "top_combos": [{"total": v, "agg": a, "members": c} for v, a, c in combos[:6]],
        "references": {"M115": M115_BASELINE, "M244_total": M244_TOTAL,
                       "M244_corr": M244_CORR, "C1N99_best": C1N99_BEST},
        "precommitment": check,
    }
    RECEIPT.write_text(json.dumps(receipt, indent=2, sort_keys=True, default=str),
                       encoding="utf-8")
    REPORT_MD.write_text(
        f"""# M271 N11 — 검색 기반 후보(AnEn)로 조합 축을 여는 시도

- 판정일: {receipt['judged_at']}
- 노드: `{NODE_ID}` / 레인 {LANE} / 부모 `{PARENT_NODE}`
- 하네스 `{harness_digest}` (캐시 {cache_hit}) / k={K_NEIGHBOURS} / 가중 {WEIGHTS}

## 1. 신규 후보

| | 값 |
|---|---:|
| 단독 Total | **{solo['total']:.6f}** (1-NMAE {solo['one_minus_nmae']:.6f} / FICR {solo['ficr']:.6f}) |
| M115 대비 오차 상관 | **{corr:.4f}** |
| 참조 — M244 아날로그 | {M244_TOTAL:.6f} / 상관 {M244_CORR} |
| 참조 — M115 단독 | {M115_BASELINE:.6f} |

## 2. 최선 조합

`{best_combo[1]}` {best_combo[2]} = **{best_combo[0]:.6f}**
(C1N99 최선 {C1N99_BEST:.6f} 대비 **{best_combo[0]-C1N99_BEST:+.6f}**)

## 3. 사전확약 대조

- V1 `누출 0` -> **{v1}** ({leak})
- V2 `확률행렬 정상` -> **{v2}**
- V3 `조인 19,782` -> **{v3}** ({len(j)})
- H1 `상관 <= {CORR_TARGET}` -> **{h1}** ({corr:.4f})
- H2 `상관 < {M244_CORR}` -> **{h2}**
- H3 `단독 > {M244_TOTAL}` -> **{h3}** ({solo['total']:.6f})
- H4 `조합 이득 >= {DETECTION_THRESHOLD}` -> **{h4}** ({best_combo[0]-C1N99_BEST:+.6f})

판정: **{verdict}**
""", encoding="utf-8")
    print(json.dumps({
        "verdict": verdict, "solo": solo["total"], "corr_vs_M115": corr,
        "best_combo": best_combo[0], "delta_vs_C1N99": best_combo[0] - C1N99_BEST,
        "guards": {"V1": v1, "V2": v2, "V3": v3},
        "H": {"H1": h1, "H2": h2, "H3": h3, "H4": h4},
        "top": [(round(v, 6), a, c) for v, a, c in combos[:3]],
    }, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
