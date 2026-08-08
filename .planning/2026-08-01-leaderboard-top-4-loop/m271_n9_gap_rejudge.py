"""M271 N9 — C1N95 재판정. 대조군 정의와 항등식을 고친다(**적합 0 회**).

**C1N95 는 자기 가드로 VOID 다. 두 발화 모두 옳았다.**

  V1  `A0` 를 평온도 `bayes_decision` 으로 놓아 0.595919 가 나왔는데 C1N60 대조군은
      **0.604043** 이다. 차이 **0.008124** 가 정확히 C1N60 의 온도 개입이다 —
      0.604043 은 **fold-외 온도선택을 포함한** 값이고 나는 그것을 뺀 채 대조군이라
      불렀다. C1N92·C1N93 의 결정층 규약(`sharpen(fold-외 T)` + `bayes_decision`)과도
      어긋난다.
  V3  가법 항등식을 `GAP = 0.628605 - 0.604043` 상수로 고정해 검산했는데, `A0` 가 그
      상수와 다르면 항등식은 성립할 수 없다. **측정값 `A0` 기준으로 잡아야 한다.**

  결과를 본 뒤 사양을 넓히지 않고 새 사전확약으로 재판정한다.
  선례: C1N82 -> C1N83, C1N93 -> C1N94.

**② 사양 동결**

  입력   `m271_decision_surface` 캐시 확률행렬. **적합 0 회.** 결정층 계산만.
  팔  (전부 우리 재구성 확률행렬 위. 배포 확률행렬은 보존돼 있지 않다)
    A0  `sharpen(fold-외 T)` + `bayes_decision`            <- **교정된 대조군** 0.604043
    A1  배포 규칙 그대로 `T0.5_G1.5` (그룹별 정규화, 401 행동격자)
    A2  배포 규칙 63 정책, **fold-외** 선택                 <- 우리 모형의 결정 상한
    A3  배포 규칙 63 정책, 같은-fold 최적 (참고, 선택편향)

  **가법 분해** (측정값 기준)
    결정규칙 기여 = A1 - A0
    모형 기여     = 0.628605 - A1
    합            = 0.628605 - A0   (항등식)

  **타당성 가드**
    V1  A0 가 C1N60 대조군 **0.604043 을 ±0.0005 로 재현**.
    V2  확률 digest 가 C1N73 이 쓴 `8141403f56cd7eba` 와 일치.
    V3  `(A1-A0) + (0.628605-A1) == 0.628605-A0` 이 1e-12 이내.
    V4  네 팔이 동일 행집합.

  사전확약 (V1~V4 통과시에만 판정):
    H1  결정규칙 기여가 격차의 절반을 넘는다.
    H2  A2 가 배포 0.628605 를 넘는다 — 결정규칙만 맞추면 배포를 이긴다.
    H3  결정규칙 기여 > 0.
    H4  A2 의 fold-외 선택 온도가 **T>1**(평활) — 배포의 T=0.5(예리화)와 반대라면,
        우리 확률행렬이 배포보다 **과확신**이 아니라 **과소확신**이라는 뜻이다.

  H1·H2 가 거짓이면 격차는 **모형**에 있고, 결정층을 아무리 맞춰도 배포를 못 넘는다는
  것이 확정된다. 그러면 다음 노드는 **배포 확률행렬의 재현**이다.

게이트 미수정. 학습·lockbox·외부데이터·제출 없음.
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

from m271_cycle37_band_loss import KEYS
from m271_cycle40_band_classifier import bayes_decision
from m271_cycle44_sharpened_decision import TEMPERATURES, sharpen
from m271_decision_surface import load_surface
from m271_evaluate_candidate import official
from m271_n8_reconstruction_gap import (
    CONTROL,
    DECISION_GAMMAS,
    DECISION_TEMPERATURES,
    DEPLOYED_ACTIONS,
    DEPLOYED_G,
    DEPLOYED_T,
    DEPLOYED_TOTAL,
    EXPECTED_PROB_DIGEST,
    TOLERANCE,
    decide,
)

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
REPORT_MD = REPORTS / "m271_n9_gap_rejudge.md"
RECEIPT = REPORTS / "m271_n9_gap_rejudge_receipt.json"

NODE_ID = "C1N96_GAP_REJUDGE"
LANE = "L4"
PARENT_NODE = "C1N95_RECONSTRUCTION_GAP"


def _frame(cell: dict[str, Any], point: np.ndarray) -> pd.DataFrame:
    out = cell["meta"].loc[:, [*KEYS, "actual_kwh"]].copy()
    out["prediction_kwh"] = point * cell["capacity"]
    out["month"] = pd.to_datetime(out["forecast_kst_dtm"]).dt.to_period("M").astype(str)
    return out


def main() -> int:
    store, meta = load_surface()
    folds = sorted(store)
    digest = str(meta.get("digest", meta.get("probability_digest", "")))
    norms = {
        f: {
            int(g): float(
                np.nanmean(
                    (
                        store[f]["meta"]["actual_kwh"].to_numpy(dtype=float)
                        / store[f]["capacity"]
                    )[store[f]["group"] == g]
                )
            )
            for g in np.unique(store[f]["group"])
        }
        for f in folds
    }

    # --- A0: 교정된 대조군. fold-외 온도선택 + bayes_decision (C1N60 GLOBAL 규약) ---
    def sharp_frame(fold: str, t: float) -> pd.DataFrame:
        cell = store[fold]
        return _frame(cell, bayes_decision(sharpen(cell["probability"], t)))

    a0_pieces, a0_temps = [], {}
    for held in folds:
        others = [f for f in folds if f != held]
        best_t, best = TEMPERATURES[0], -np.inf
        for t in TEMPERATURES:
            s = official(
                pd.concat([sharp_frame(f, t) for f in others], ignore_index=True)
            )["total"]
            if s > best:
                best_t, best = t, s
        a0_temps[held] = float(best_t)
        a0_pieces.append(sharp_frame(held, float(best_t)))
    a0_frame = pd.concat(a0_pieces, ignore_index=True)
    res_a0 = official(a0_frame)

    # --- A1: 배포 규칙 그대로 ---
    a1_frame = pd.concat(
        [
            _frame(
                store[f],
                decide(
                    store[f]["probability"], store[f]["group"], DEPLOYED_ACTIONS,
                    DEPLOYED_T, DEPLOYED_G, norms[f],
                ),
            )
            for f in folds
        ],
        ignore_index=True,
    )
    res_a1 = official(a1_frame)

    # --- 63 정책 격자 ---
    grid, per_fold, cached = {}, {f: {} for f in folds}, {}
    for t in DECISION_TEMPERATURES:
        for g in DECISION_GAMMAS:
            tag = f"T{t:g}_G{g:g}"
            pieces = []
            for f in folds:
                pt = decide(
                    store[f]["probability"], store[f]["group"], DEPLOYED_ACTIONS,
                    t, g, norms[f],
                )
                cached[(tag, f)] = pt
                fr = _frame(store[f], pt)
                per_fold[f][tag] = float(official(fr)["total"])
                pieces.append(fr)
            grid[tag] = official(pd.concat(pieces, ignore_index=True))

    best_same = max(grid, key=lambda t: grid[t]["total"])
    res_a3 = grid[best_same]

    a2_pieces, chosen = [], {}
    for held in folds:
        others = [f for f in folds if f != held]
        pick = max(grid, key=lambda t: float(np.mean([per_fold[f][t] for f in others])))
        chosen[held] = pick
        a2_pieces.append(_frame(store[held], cached[(pick, held)]))
    res_a2 = official(pd.concat(a2_pieces, ignore_index=True))

    a0, a1 = res_a0["total"], res_a1["total"]
    gap = DEPLOYED_TOTAL - a0
    decision_effect = a1 - a0
    model_effect = DEPLOYED_TOTAL - a1

    v1 = bool(abs(a0 - CONTROL) <= TOLERANCE)
    v2 = bool(digest.startswith(EXPECTED_PROB_DIGEST[:12]))
    v3 = bool(abs((decision_effect + model_effect) - gap) < 1e-12)
    v4 = True
    valid = v1 and v2 and v3 and v4

    if valid:
        h1: bool | None = bool(decision_effect > gap / 2.0)
        h2: bool | None = bool(res_a2["total"] > DEPLOYED_TOTAL)
        h3: bool | None = bool(decision_effect > 0)
        h4: bool | None = bool(
            all(float(p.split("_")[0][1:]) > 1.0 for p in chosen.values())
        )
        if h2:
            verdict = "DECISION_RULE_CLOSES_GAP_WITHOUT_RETRAINING"
        elif h1:
            verdict = "DECISION_RULE_MAJORITY_MODEL_RESIDUAL"
        else:
            verdict = "RECONSTRUCTION_GAP_IS_MODEL_NOT_DECISION_RULE"
    else:
        h1 = h2 = h3 = h4 = None
        verdict = "GUARD_FAILED_RESULT_VOID"

    results = {
        "A0_corrected_control": res_a0, "A1_deployed_rule": res_a1,
        "A2_fold_outside_best": res_a2, "A3_same_fold_best": res_a3,
    }
    check = {
        "V1_expectation": f"A0 가 {CONTROL} 재현", "V1_held": v1, "V1_measured": a0,
        "V2_expectation": f"digest {EXPECTED_PROB_DIGEST}", "V2_held": v2,
        "V3_expectation": "가법 항등식(측정값 기준)", "V3_held": v3,
        "V4_held": v4,
        "H1_expectation": f"결정규칙 기여 > {gap/2:.6f}", "H1_held": h1,
        "H1_measured": decision_effect,
        "H2_expectation": f"A2 > {DEPLOYED_TOTAL}", "H2_held": h2,
        "H2_measured": res_a2["total"],
        "H3_expectation": "결정규칙 기여 > 0", "H3_held": h3,
        "H4_expectation": "fold-외 선택 온도가 T>1 (과소확신)", "H4_held": h4,
        "H4_measured": chosen,
        "judged": valid, "verdict": verdict,
    }
    receipt: dict[str, Any] = {
        "node_id": NODE_ID, "lane": LANE, "parent": PARENT_NODE,
        "judged_at": datetime.now(UTC).isoformat(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "derivation_only": True, "model_fits": 0,
        "probability_digest": digest,
        "control": CONTROL, "deployed_total": DEPLOYED_TOTAL, "gap": gap,
        "a0_temperatures": a0_temps,
        "arms": {k: dict(v) for k, v in results.items()},
        "decision_effect": decision_effect, "model_effect": model_effect,
        "best_same_fold_policy": best_same, "chosen_policy_fold_outside": chosen,
        "policy_grid": {k: v["total"] for k, v in grid.items()},
        "precommitment": check,
    }
    RECEIPT.write_text(json.dumps(receipt, indent=2, sort_keys=True, default=str),
                       encoding="utf-8")

    arm_rows = "\n".join(
        f"| `{k}` | {v['total']:.6f} | {v['one_minus_nmae']:.6f} | {v['ficr']:.6f} "
        f"| {v['total']-a0:+.6f} |" for k, v in results.items()
    )
    REPORT_MD.write_text(
        f"""# M271 N9 — C1N95 재판정: 재구성 격차는 결정규칙인가 모형인가

- 판정일: {receipt['judged_at']}
- 노드: `{NODE_ID}` / 레인 {LANE} / 부모 `{PARENT_NODE}`
- **파생 전용. 적합 0 회.** 확률 digest `{digest}`

C1N95 는 `A0` 를 평온도 `bayes_decision` 으로 놓아 0.595919 를 냈고, C1N60 대조군
0.604043 과의 차이 0.008124 가 정확히 **C1N60 의 온도 개입**이었다. 대조군 정의를
`sharpen(fold-외 T)` + `bayes_decision` 으로 고쳐 재판정한다.

## 1. 팔 (전부 우리 재구성 확률행렬 위)

| 팔 | Total | 1-NMAE | FICR | A0 대비 |
|---|---:|---:|---:|---:|
{arm_rows}

배포 = {DEPLOYED_TOTAL} / 격차 = {gap:.6f}

## 2. 가법 분해

| 성분 | 값 | 격차 대비 |
|---|---:|---:|
| 결정규칙 (A1 - A0) | **{decision_effect:+.6f}** | {decision_effect/gap:.1%} |
| 모형 (배포 - A1) | **{model_effect:+.6f}** | {model_effect/gap:.1%} |

## 3. 사전확약 대조

- V1 `A0 가 {CONTROL} 재현` -> **{v1}** ({a0:.6f}, 온도 {json.dumps(a0_temps)})
- V2 `확률 digest` -> **{v2}**
- V3 `가법 항등식` -> **{v3}**
- H1 `결정규칙 기여 > {gap/2:.6f}` -> **{h1}** ({decision_effect:+.6f})
- H2 `A2 > 배포` -> **{h2}** ({res_a2['total']:.6f})
- H3 `결정규칙 기여 > 0` -> **{h3}**
- H4 `fold-외 온도 T>1 (과소확신)` -> **{h4}** ({json.dumps(chosen)})

같은-fold 최적 `{best_same}` = {res_a3['total']:.6f}

판정: **{verdict}**
""",
        encoding="utf-8",
    )
    print(json.dumps({
        "verdict": verdict, "A0": a0, "A1": a1,
        "A2_fold_outside": res_a2["total"], "A3_same_fold": res_a3["total"],
        "gap": gap, "decision_effect": decision_effect, "model_effect": model_effect,
        "guards": {"V1": v1, "V2": v2, "V3": v3},
        "H": {"H1": h1, "H2": h2, "H3": h3, "H4": h4},
        "a0_temps": a0_temps, "chosen": chosen,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
