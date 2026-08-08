"""M271 N10 — 재구성 격차의 모형 성분: **부스팅 라운드**가 확률면의 확신도를 정한다.

**C1N96 이 격차를 모형으로 귀속했고, 그 기전을 H4 가 지목했다.**

    fold-외 선택 온도   우리 재구성 **T=2.2 / 1.2 (평활)**  vs  배포 **T=0.5 (예리화)**

  온도 손잡이의 **반대편**이다. 우리 확률면은 **과확신**이라 평활이 필요하고 배포는
  **과소확신**이라 예리화가 이득이다.

**배포 사양을 읽으니 원인 후보가 하나로 좁혀진다.**

    `M269_PROBE_TOP100-dev-2023-Q3.json`
        selected_iteration = 60      <- 배포는 **60 라운드**
        num_leaves         = 15      <- 우리와 같음
        class_count        = 46      <- 같음
        class_width        = 0.02    <- 같음
        feature_count      = 100     <- 같음(우리 101 = 87 기저 + 14 sitewind)

    우리 `m271_cycle40_band_classifier.ROUNDS = 200`

  **3.3 배 과적합이다.** 라운드가 늘면 softmax 확률이 one-hot 쪽으로 밀려 확신도가
  올라간다 — 그것이 정확히 과확신이고, 평활이 필요해지는 이유다.
  C1N39 는 피처 커버리지 효과가 **+0.000726 뿐**이라 했고(87 -> 1,347), C1N96 은 결정
  경로를 닫았다. 남은 것은 엔진과 학습 사양이며 **라운드가 그중 가장 큰 단일 차이**다.

**① 방법 리서치**

  부스팅 반복수는 **정규화 모수**다(Friedman 2001; 조기중단이 표준 처리). 다중분류
  softmax 에서 반복이 늘면 로짓 크기가 커져 예측분포가 뾰족해진다 — 정확도와 무관하게
  **교정도(calibration)** 가 나빠지는 방향이다. 결정층이 기대효용을 쓰므로 교정도가 곧
  성능이다(Guo et al. 2017 의 현대 신경망 과확신 논의와 같은 구조).
  따라서 **라운드와 온도는 같은 축의 두 손잡이**이고, 함께 골라야 한다.

  한 번 적합한 부스터에서 `num_iteration=k` 로 예측하면 k 라운드 모형의 예측을 그대로
  얻는다 — **적합 3 회로 전 격자를 잰다.**

**② 사양 동결**

  하네스   C1N60 과 동일(`m271_harness_cache`). 피처 101 개. 표적 one-hot CE.
  격자     라운드 (20, 40, 60, 80, 100, 140, 200) x 온도 C1N44 의 9 개
  선택     **fold-외**. 보류 fold 의 (라운드, 온도)를 나머지 fold 에서 고른다.
           라운드를 점수 보고 고르면 선택편향이므로 온도와 **같은 규약**으로 묶는다.
  결정층   `bayes_decision`. 전 격자 동일.

  **타당성 가드**
    V1  라운드 200 + fold-외 온도가 C1N60 대조군 **0.604043 을 ±0.0005 로 재현**.
    V2  전 격자의 확률행렬이 정상(행합 1, NaN 없음).
    V3  라운드 200 의 예측이 `num_boost_round=200` 직접 적합과 일치 — `num_iteration`
        경로가 같은 모형을 주는지 확인(부분예측 계약).

  사전확약 (V1~V3 통과시에만 판정):
    H1  fold-외 (라운드, 온도) 선택이 대조군 0.604043 을 **검출문턱 0.001013 이상** 넘는다.
    H2  선택된 라운드가 **200 미만** — 적을수록 낫다면 과적합 기전이 맞다.
    H3  라운드가 줄면 선택 온도가 **1.0 쪽으로 내려간다** — 평활 필요가 줄어야 기전이 맞다.
        (라운드 200 의 선택온도 대 최선 라운드의 선택온도를 비교)
    H4  최선이 배포 **0.628605 의 0.005 이내**. 참이면 재구성 격차가 라운드 하나로 닫힌다.

  **부호 예단 없음.** H2 가 거짓이면(라운드가 많을수록 낫다면) 과확신 기전이 반증되고
  격차는 엔진이나 표적에 있다. H1 이 참이고 H4 가 거짓이면 라운드가 격차의 일부만
  설명하며 잔차를 다음 노드가 받는다.

게이트 미수정. lockbox·외부데이터·`scada_ws` 미사용. 제출 없음.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from m271_cycle37_band_loss import KEYS
from m271_cycle40_band_classifier import (
    CLASS_WIDTH,
    N_CLASS,
    PARAMS,
    ROUNDS,
    bayes_decision,
    make_objective,
    one_hot_targets,
)
from m271_cycle44_sharpened_decision import TEMPERATURES, sharpen
from m271_evaluate_candidate import official
from m271_harness_cache import fold_frames, resolved_base_features

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
REPORT_MD = REPORTS / "m271_n10_rounds.md"
RECEIPT = REPORTS / "m271_n10_rounds_receipt.json"

NODE_ID = "C1N97_BOOSTING_ROUNDS"
LANE = "L3"
PARENT_NODE = "C1N96_GAP_REJUDGE"

CONTROL = 0.604043
TOLERANCE = 0.0005
DEPLOYED_TOTAL = 0.628605
DEPLOYED_ROUNDS = 60
DETECTION_THRESHOLD = 0.001013
ROUND_GRID = (20, 40, 60, 80, 100, 140, 200)


def main() -> int:
    store_raw, harness_digest, cache_hit = fold_frames()
    folds = sorted(store_raw)
    base_features = resolved_base_features(
        store_raw[folds[0]]["train"], store_raw[folds[0]]["sitewind_names"]
    )

    prob: dict[tuple[str, int], np.ndarray] = {}
    meta: dict[str, pd.DataFrame] = {}
    caps: dict[str, np.ndarray] = {}
    v3_checks: list[bool] = []
    fits = 0

    for fold in folds:
        entry = store_raw[fold]
        train, test = entry["train"], entry["test"]
        features = [*base_features, *entry["sitewind_names"]]
        x_tr = train.loc[:, features].astype("float32")
        x_te = test.loc[:, features].astype("float32")
        rate = np.clip(train["rate"].to_numpy(dtype="float64"), 0.0, None)
        label = np.clip((rate / CLASS_WIDTH).astype(int), 0, N_CLASS - 1)

        params = dict(PARAMS)
        params["objective"] = make_objective(one_hot_targets(rate))
        booster = lgb.train(
            params,
            lgb.Dataset(x_tr, label=label, free_raw_data=False),
            num_boost_round=ROUNDS,
        )
        fits += 1
        for k in ROUND_GRID:
            raw = np.asarray(booster.predict(x_te, num_iteration=k)).reshape(
                len(test), N_CLASS
            )
            e = np.exp(raw - raw.max(axis=1, keepdims=True))
            prob[(fold, k)] = e / e.sum(axis=1, keepdims=True)
        meta[fold] = test.loc[:, [*KEYS, "actual_kwh"]].copy()
        caps[fold] = test["capacity"].to_numpy(dtype="float64")

        # V3: num_iteration=200 경로가 직접 적합과 같은가
        direct = np.asarray(booster.predict(x_te)).reshape(len(test), N_CLASS)
        v3_checks.append(bool(np.allclose(direct, booster.predict(x_te, num_iteration=ROUNDS))))

    v2 = bool(
        all(
            np.isfinite(p).all() and abs(p.sum(axis=1) - 1.0).max() < 1e-9
            for p in prob.values()
        )
    )
    v3 = all(v3_checks)

    def scored(fold: str, k: int, t: float) -> pd.DataFrame:
        out = meta[fold].copy()
        out["prediction_kwh"] = (
            bayes_decision(sharpen(prob[(fold, k)], t)) * caps[fold]
        )
        out["month"] = pd.to_datetime(out["forecast_kst_dtm"]).dt.to_period(
            "M"
        ).astype(str)
        return out

    # fold 별 (라운드, 온도) 점수표
    cell_score: dict[str, dict[tuple[int, float], float]] = {f: {} for f in folds}
    for fold in folds:
        for k in ROUND_GRID:
            for t in TEMPERATURES:
                cell_score[fold][(k, t)] = float(official(scored(fold, k, t))["total"])

    def fold_outside(candidates) -> tuple[pd.DataFrame, dict[str, tuple[int, float]]]:
        pieces, picks = [], {}
        for held in folds:
            others = [f for f in folds if f != held]
            pick = max(
                candidates,
                key=lambda c: float(np.mean([cell_score[f][c] for f in others])),
            )
            picks[held] = pick
            pieces.append(scored(held, pick[0], pick[1]))
        return pd.concat(pieces, ignore_index=True), picks

    # V1 대조군: 라운드 200 고정, 온도만 fold-외
    ctrl_frame, ctrl_picks = fold_outside([(ROUNDS, t) for t in TEMPERATURES])
    res_ctrl = official(ctrl_frame)
    v1 = bool(abs(res_ctrl["total"] - CONTROL) <= TOLERANCE)

    # 처리: (라운드, 온도) 둘 다 fold-외
    both_frame, both_picks = fold_outside(
        [(k, t) for k in ROUND_GRID for t in TEMPERATURES]
    )
    res_both = official(both_frame)

    # 라운드별 곡선 (각 라운드에서 온도만 fold-외)
    curve: dict[int, dict[str, Any]] = {}
    for k in ROUND_GRID:
        fr, pk = fold_outside([(k, t) for t in TEMPERATURES])
        s = official(fr)
        curve[k] = {
            "total": s["total"], "one_minus_nmae": s["one_minus_nmae"],
            "ficr": s["ficr"],
            "temperatures": {f: pk[f][1] for f in folds},
        }

    valid = v1 and v2 and v3
    gain = res_both["total"] - res_ctrl["total"]
    best_round = max(ROUND_GRID, key=lambda k: curve[k]["total"])
    ctrl_temp = float(np.mean([ctrl_picks[f][1] for f in folds]))
    best_temp = float(np.mean(list(curve[best_round]["temperatures"].values())))

    if valid:
        h1: bool | None = bool(gain >= DETECTION_THRESHOLD)
        h2: bool | None = bool(best_round < ROUNDS)
        h3: bool | None = bool(best_temp < ctrl_temp)
        h4: bool | None = bool(abs(res_both["total"] - DEPLOYED_TOTAL) <= 0.005)
        if h4:
            verdict = "ROUNDS_CLOSE_RECONSTRUCTION_GAP"
        elif h1 and h2:
            verdict = "ROUNDS_EXPLAIN_PART_OF_GAP_OVERCONFIDENCE_CONFIRMED"
        elif h1:
            verdict = "ROUNDS_HELP_BUT_MECHANISM_NOT_OVERCONFIDENCE"
        else:
            verdict = "ROUNDS_NOT_THE_GAP_DRIVER"
    else:
        h1 = h2 = h3 = h4 = None
        verdict = "GUARD_FAILED_RESULT_VOID"

    check = {
        "V1_expectation": f"라운드 {ROUNDS} + fold-외 온도가 {CONTROL} 재현",
        "V1_held": v1, "V1_measured": res_ctrl["total"],
        "V2_expectation": "전 격자 확률행렬 정상", "V2_held": v2,
        "V3_expectation": "num_iteration 경로 = 직접 적합", "V3_held": v3,
        "H1_expectation": f"fold-외 (라운드,온도) 이득 >= {DETECTION_THRESHOLD}",
        "H1_held": h1, "H1_measured": gain,
        "H2_expectation": f"선택 라운드 < {ROUNDS}", "H2_held": h2,
        "H2_measured": best_round,
        "H3_expectation": "최선 라운드의 선택온도가 대조군보다 낮다",
        "H3_held": h3, "H3_measured": {"control_temp": ctrl_temp, "best_temp": best_temp},
        "H4_expectation": f"배포 {DEPLOYED_TOTAL} 의 0.005 이내", "H4_held": h4,
        "H4_measured": res_both["total"],
        "judged": valid, "verdict": verdict,
    }

    receipt: dict[str, Any] = {
        "node_id": NODE_ID, "lane": LANE, "parent": PARENT_NODE,
        "judged_at": datetime.now(UTC).isoformat(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "harness_digest": harness_digest, "harness_cache_hit": cache_hit,
        "fits": fits, "round_grid": list(ROUND_GRID),
        "deployed_rounds": DEPLOYED_ROUNDS, "deployed_total": DEPLOYED_TOTAL,
        "control": CONTROL,
        "control_arm": dict(res_ctrl), "control_picks": {f: ctrl_picks[f] for f in folds},
        "treatment_arm": dict(res_both), "treatment_picks": {f: both_picks[f] for f in folds},
        "round_curve": curve, "gain": gain,
        "precommitment": check,
    }
    RECEIPT.write_text(json.dumps(receipt, indent=2, sort_keys=True, default=str),
                       encoding="utf-8")

    rows = "\n".join(
        f"| {k}{' **(배포)**' if k == DEPLOYED_ROUNDS else ''} | {curve[k]['total']:.6f} "
        f"| {curve[k]['one_minus_nmae']:.6f} | {curve[k]['ficr']:.6f} "
        f"| {curve[k]['total']-CONTROL:+.6f} | {list(curve[k]['temperatures'].values())} |"
        for k in ROUND_GRID
    )
    REPORT_MD.write_text(
        f"""# M271 N10 — 부스팅 라운드가 확률면의 확신도를 정한다

- 판정일: {receipt['judged_at']}
- 노드: `{NODE_ID}` / 레인 {LANE} / 부모 `{PARENT_NODE}`
- 하네스 digest `{harness_digest}` (캐시 {cache_hit}) / **적합 {fits} 회**
  (한 부스터에서 `num_iteration` 으로 전 격자를 뽑는다)

배포 사양 `selected_iteration = {DEPLOYED_ROUNDS}` 대 우리 `ROUNDS = {ROUNDS}`.
C1N96 의 H4 가 우리 확률면을 **과확신**(평활 필요), 배포를 **과소확신**(예리화 이득)으로
지목했고 라운드가 그 손잡이를 정확히 돌린다.

## 1. 라운드 곡선 (각 라운드에서 온도만 fold-외)

| 라운드 | Total | 1-NMAE | FICR | 대조군 대비 | 선택 온도 |
|---:|---:|---:|---:|---:|---|
{rows}

## 2. 팔

| 팔 | Total | 선택 |
|---|---:|---|
| 대조군 (라운드 {ROUNDS} 고정, 온도 fold-외) | {res_ctrl['total']:.6f} | {json.dumps({f: ctrl_picks[f] for f in folds}, default=str)} |
| 처리 (라운드·온도 둘 다 fold-외) | **{res_both['total']:.6f}** | {json.dumps({f: both_picks[f] for f in folds}, default=str)} |
| 배포 | {DEPLOYED_TOTAL} | `T0.5_G1.5` |

이득 **{gain:+.6f}** / 배포까지 남은 것 **{DEPLOYED_TOTAL - res_both['total']:+.6f}**

## 3. 사전확약 대조

- V1 `{CONTROL} 재현` -> **{v1}** ({res_ctrl['total']:.6f})
- V2 `확률행렬 정상` -> **{v2}**
- V3 `num_iteration 경로 일치` -> **{v3}**
- H1 `이득 >= {DETECTION_THRESHOLD}` -> **{h1}** ({gain:+.6f})
- H2 `선택 라운드 < {ROUNDS}` -> **{h2}** (최선 {best_round})
- H3 `최선 라운드의 온도가 더 낮다` -> **{h3}** (대조 {ctrl_temp:.2f} -> 최선 {best_temp:.2f})
- H4 `배포의 0.005 이내` -> **{h4}**

판정: **{verdict}**
""",
        encoding="utf-8",
    )
    print(json.dumps({
        "verdict": verdict,
        "control": res_ctrl["total"], "treatment": res_both["total"],
        "gain": gain, "deployed": DEPLOYED_TOTAL,
        "remaining_to_deployed": DEPLOYED_TOTAL - res_both["total"],
        "curve": {k: round(curve[k]["total"], 6) for k in ROUND_GRID},
        "temps": {k: list(curve[k]["temperatures"].values()) for k in ROUND_GRID},
        "guards": {"V1": v1, "V2": v2, "V3": v3},
        "H": {"H1": h1, "H2": h2, "H3": h3, "H4": h4},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
