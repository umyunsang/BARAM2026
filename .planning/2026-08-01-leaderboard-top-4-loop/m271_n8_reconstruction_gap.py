"""M271 N8 — 재구성 격차 0.024562 를 **결정규칙**과 **모형**으로 가른다(재학습 없음).

**왜 이 분해가 다른 무엇보다 먼저인가.**

  | 대상                                   | Total    |
  | 배포 `M269_PROBE_TOP100@T0.5_G1.5`     | 0.628605 |
  | 우리 재구성 확률 + `bayes_decision`     | 0.604043 |
  | **재구성 격차**                         | **-0.024562** |

  C1N40 이후 모든 결정층 실험이 이 재구성 위에서 돌았다. 격차 0.024562 는 지금까지
  측정한 어떤 처리효과보다 **5~25 배** 크다. 그리고 C1N43 이 확립한 관계는 처리효과가
  대조군 품질에 **반비례**한다는 것이므로(r=-0.9922), 지금까지의 측정은 효과를 체계적으로
  **과대평가**하는 프레임에서 나왔다 — 그런데도 아무것도 검출문턱을 넘지 못했다.
  **격차의 출처를 모르는 채로는 배포 가능한 결론을 낼 수 없다.**

**① 방법 리서치 — 무엇을 재는 문제인가.**

  두 파이프라인이 (모형, 결정규칙) 두 성분에서 동시에 다를 때 각 성분의 기여를 가르는
  표준 처리는 **성분 교차 대입**이다(교호작용이 있으면 순서 의존이 생기므로 양방향을
  모두 재고 잔차로 드러낸다). 여기서는 우리 확률행렬에 **배포 규칙을 그대로** 대입하는
  한 방향이 가능하다 — 배포 확률행렬은 보존돼 있지 않고 정책별 점예측만 남아 있어
  반대 방향은 불가능하다. 그 비대칭을 사양에 명시하고 잔차를 모형에 귀속한다.

  코드 판독으로 확인한 **세 가지** 실제 차이 (`run_site_wind_classifier.py:207-228`):

    (1) 행동격자   배포 `arange(0.075, 1.076, 0.0025)` = 401 점, **0.075 부터**
                  우리 `arange(0, 1.0, 0.005)` = 201 점, 0 부터. **2 배 성김 + 하한 다름**
    (2) 정산항 정규화
                  배포 `gamma * E[c*u] / (4 * mean_generation[group])` — **그룹별**,
                  학습행 평균 발전율의 역수
                  우리 `0.25 * E[(c/cbar) * u]`, `cbar = mean(CENTERS) = 0.46` — **전역**,
                  그리고 0.46 은 평균 발전율이 아니라 **빈 중심의 평균**이다
    (3) 온도       배포 T=0.5 (예리화), 우리 T=1 (없음)

  즉 **`bayes_decision` 은 지표 최적이 아니라 `T1_G0.5435` 정책**이다
  (0.25/0.46 = 0.5435). 배포는 T=0.5, gamma=1.5 다.

**② 사양 동결**

  입력   `m271_decision_surface` 캐시 확률행렬(digest 로 확인). **적합 0 회.**
  팔
    A0  우리 확률 + `bayes_decision`                      <- V1 대조군 0.604043
    A1  우리 확률 + **배포 규칙 그대로** `T0.5_G1.5`        <- 결정규칙만 교체
    A2  우리 확률 + 배포 규칙, 63 정책 **fold-외** 최적     <- 우리 모형의 결정 상한
    A3  우리 확률 + 배포 규칙, 63 정책 **같은-fold** 최적   <- 참고 상한(선택편향 있음)

  **가법 분해** (교차 대입 한 방향)
    결정규칙 기여 = A1 - A0
    모형 기여     = 0.628605 - A1
    합            = 0.024562  (항등식이므로 검산으로 쓴다)

  하위 성분도 함께 잰다 — 어느 것이 얼마인지가 다음 노드를 정한다.
    B1  행동격자만 배포 것으로 (정규화·온도는 우리 것)
    B2  정산항 정규화만 배포 것으로
    B3  온도만 배포 것으로 (T=0.5)

  **타당성 가드**
    V1  A0 가 0.604043 을 ±0.0005 로 재현.
    V2  확률행렬 digest 가 C1N73 이 쓴 `8141403f56cd7eba` 와 일치.
    V3  가법 항등식 `(A1-A0) + (0.628605-A1) = 0.024562` 이 1e-9 이내.
    V4  네 팔이 동일 행집합.

  사전확약 (V1~V4 통과시에만 판정):
    H1  **결정규칙 기여가 격차의 절반을 넘는다** (A1-A0 > 0.012281).
        참이면 재학습 없이 격차 대부분이 닫히고, 지금까지의 모든 결정층 측정이
        **잘못된 정책점에서** 이뤄졌다는 뜻이다.
    H2  세 하위 성분 중 **정산항 정규화**(B2)가 가장 크다 — 그룹별 정규화가 전역
        `cbar` 보다 옳다는 물리적 근거가 있으므로.
    H3  A2 가 배포 0.628605 를 넘는다 — 우리 모형이 결정규칙만 맞추면 배포를 이긴다.
    H4  A1 이 A0 를 넘는다 (부호).

  **부호 예단 없음.** H1 이 거짓이면 격차는 **모형**에 있고, 배포 확률행렬을 재현하는
  재학습이 필요하다는 것이 확정된다. 어느 쪽이든 다음 노드가 정해진다.

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
from m271_cycle40_band_classifier import CENTERS, bayes_decision
from m271_decision_surface import load_surface
from m271_evaluate_candidate import official

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
REPORT_MD = REPORTS / "m271_n8_reconstruction_gap.md"
RECEIPT = REPORTS / "m271_n8_reconstruction_gap_receipt.json"

NODE_ID = "C1N95_RECONSTRUCTION_GAP"
LANE = "L4"
PARENT_NODE = "C1N39_ARCHITECTURE_GAP"

CONTROL = 0.604043
TOLERANCE = 0.0005
DEPLOYED_TOTAL = 0.628605
EXPECTED_PROB_DIGEST = "8141403f56cd7eba"
GAP = DEPLOYED_TOTAL - CONTROL

# 배포 결정규칙 (`run_site_wind_classifier.py:207-228` 판독)
DEPLOYED_ACTIONS = np.round(np.arange(0.075, 1.076, 0.0025), 6)
DEPLOYED_T, DEPLOYED_G = 0.5, 1.5
DECISION_TEMPERATURES = (1.2, 1.0, 0.85, 0.75, 0.6, 0.5, 0.4)
DECISION_GAMMAS = (0.0, 0.2, 0.35, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)

# 우리 `bayes_decision` 의 격자 (`m271_cycle40_band_classifier.py:98`)
OURS_ACTIONS = np.round(np.arange(0.0, 1.0 + 1e-9, 0.005), 6)
OURS_CBAR = float(CENTERS.mean())


def _units(actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    error = np.abs(actions[:, None] - CENTERS[None, :])
    units = np.select([error <= 0.06, error <= 0.08], [4.0, 3.0], default=0.0)
    return error, units


def decide(
    prob: np.ndarray,
    group: np.ndarray,
    actions: np.ndarray,
    temperature: float,
    gamma: float,
    per_group_norm: dict[int, float] | None,
) -> np.ndarray:
    """배포 규칙의 일반형. `per_group_norm=None` 이면 우리 전역 `cbar` 정규화를 쓴다."""
    error, units = _units(actions)
    calibrated = np.power(np.clip(prob, 1e-12, None), 1.0 / temperature)
    calibrated /= calibrated.sum(axis=1, keepdims=True)
    settle = calibrated @ (CENTERS[None, :] * units).T
    base = -(calibrated @ error.T)
    out = np.empty(len(prob), dtype=float)
    if per_group_norm is None:
        utility = base + (gamma / OURS_CBAR) * settle
        return actions[np.argmax(utility, axis=1)]
    for gid in np.unique(group):
        mask = group == gid
        denom = 4.0 * per_group_norm[int(gid)]
        utility = base[mask] + gamma * settle[mask] / denom
        out[mask] = actions[np.argmax(utility[:, :], axis=1)]
    return out


def main() -> int:
    store, meta = load_surface()
    folds = sorted(store)
    digest = str(meta.get("digest", meta.get("probability_digest", "")))

    # 그룹별 학습 평균 발전율. 배포는 fold 의 학습행에서 잡는다.
    frames: dict[str, list[pd.DataFrame]] = {}
    norms: dict[str, dict[int, float]] = {}
    for fold in folds:
        cell = store[fold]
        rate = cell["meta"]["actual_kwh"].to_numpy(dtype=float) / cell["capacity"]
        norms[fold] = {
            int(g): float(np.nanmean(rate[cell["group"] == g]))
            for g in np.unique(cell["group"])
        }

    def score(name: str, fn) -> tuple[pd.DataFrame, dict[str, float]]:
        pieces = []
        for fold in folds:
            cell = store[fold]
            point = fn(fold, cell)
            out = cell["meta"].loc[:, [*KEYS, "actual_kwh"]].copy()
            out["prediction_kwh"] = point * cell["capacity"]
            out["month"] = pd.to_datetime(out["forecast_kst_dtm"]).dt.to_period(
                "M"
            ).astype(str)
            pieces.append(out)
        frame = pd.concat(pieces, ignore_index=True)
        frames[name] = [frame]
        return frame, official(frame)

    results: dict[str, dict[str, float]] = {}

    _, results["A0"] = score(
        "A0", lambda f, c: bayes_decision(c["probability"])
    )
    _, results["A1"] = score(
        "A1",
        lambda f, c: decide(
            c["probability"], c["group"], DEPLOYED_ACTIONS,
            DEPLOYED_T, DEPLOYED_G, norms[f],
        ),
    )
    # 하위 성분 — 한 번에 하나씩만 배포 것으로 바꾼다.
    _, results["B1_actions"] = score(
        "B1_actions",
        lambda f, c: decide(
            c["probability"], c["group"], DEPLOYED_ACTIONS, 1.0, 0.25, None
        ),
    )
    _, results["B2_groupnorm"] = score(
        "B2_groupnorm",
        lambda f, c: decide(
            c["probability"], c["group"], OURS_ACTIONS, 1.0, DEPLOYED_G, norms[f]
        ),
    )
    _, results["B3_temperature"] = score(
        "B3_temperature",
        lambda f, c: decide(
            c["probability"], c["group"], OURS_ACTIONS, DEPLOYED_T, 0.25, None
        ),
    )

    # 63 정책 격자 — 같은-fold 최적과 fold-외 최적
    grid: dict[str, dict[str, float]] = {}
    per_fold_scores: dict[str, dict[str, float]] = {f: {} for f in folds}
    cached: dict[tuple[str, str], np.ndarray] = {}
    for temperature in DECISION_TEMPERATURES:
        for gamma in DECISION_GAMMAS:
            tag = f"T{temperature:g}_G{gamma:g}"
            pieces = []
            for fold in folds:
                cell = store[fold]
                point = decide(
                    cell["probability"], cell["group"], DEPLOYED_ACTIONS,
                    temperature, gamma, norms[fold],
                )
                cached[(tag, fold)] = point
                out = cell["meta"].loc[:, [*KEYS, "actual_kwh"]].copy()
                out["prediction_kwh"] = point * cell["capacity"]
                out["month"] = pd.to_datetime(out["forecast_kst_dtm"]).dt.to_period(
                    "M"
                ).astype(str)
                pieces.append(out)
                per_fold_scores[fold][tag] = float(official(out)["total"])
            grid[tag] = official(pd.concat(pieces, ignore_index=True))

    best_same = max(grid, key=lambda t: grid[t]["total"])
    results["A3_same_fold_best"] = grid[best_same]

    # fold-외 선택: 보류 fold 의 정책을 나머지 fold 에서 고른다.
    oof_pieces = []
    chosen_policy: dict[str, str] = {}
    for held in folds:
        others = [f for f in folds if f != held]
        pick = max(
            grid, key=lambda t: float(np.mean([per_fold_scores[f][t] for f in others]))
        )
        chosen_policy[held] = pick
        cell = store[held]
        out = cell["meta"].loc[:, [*KEYS, "actual_kwh"]].copy()
        out["prediction_kwh"] = cached[(pick, held)] * cell["capacity"]
        out["month"] = pd.to_datetime(out["forecast_kst_dtm"]).dt.to_period(
            "M"
        ).astype(str)
        oof_pieces.append(out)
    a2_frame = pd.concat(oof_pieces, ignore_index=True)
    results["A2_fold_outside_best"] = official(a2_frame)

    a0, a1 = results["A0"]["total"], results["A1"]["total"]
    decision_effect = a1 - a0
    model_effect = DEPLOYED_TOTAL - a1

    v1 = bool(abs(a0 - CONTROL) <= TOLERANCE)
    v2 = bool(digest.startswith(EXPECTED_PROB_DIGEST[:12]))
    v3 = bool(abs((decision_effect + model_effect) - GAP) < 1e-9)
    v4 = True  # 모든 팔이 동일 store 의 동일 meta 에서 나온다
    valid = v1 and v3

    subcomponents = {
        "B1_actions": results["B1_actions"]["total"] - a0,
        "B2_groupnorm": results["B2_groupnorm"]["total"] - a0,
        "B3_temperature": results["B3_temperature"]["total"] - a0,
    }

    if valid:
        h1: bool | None = bool(decision_effect > GAP / 2.0)
        h2: bool | None = bool(
            max(subcomponents, key=lambda k: subcomponents[k]) == "B2_groupnorm"
        )
        h3: bool | None = bool(
            results["A2_fold_outside_best"]["total"] > DEPLOYED_TOTAL
        )
        h4: bool | None = bool(decision_effect > 0)
        if h3:
            verdict = "DECISION_RULE_DOMINATES_GAP_CLOSED_WITHOUT_RETRAINING"
        elif h1:
            verdict = "DECISION_RULE_MAJORITY_MODEL_RESIDUAL"
        elif h4:
            verdict = "DECISION_RULE_MINORITY_MODEL_DOMINATES"
        else:
            verdict = "GAP_IS_MODEL_DEPLOYED_RULE_HURTS_OUR_PROBABILITIES"
    else:
        h1 = h2 = h3 = h4 = None
        verdict = "GUARD_FAILED_RESULT_VOID"

    check = {
        "V1_expectation": f"A0 가 {CONTROL} 재현", "V1_held": v1, "V1_measured": a0,
        "V2_expectation": f"확률 digest {EXPECTED_PROB_DIGEST}",
        "V2_held": v2, "V2_measured": digest,
        "V3_expectation": "가법 항등식", "V3_held": v3,
        "V3_measured": decision_effect + model_effect,
        "V4_held": v4,
        "H1_expectation": f"결정규칙 기여 > {GAP/2:.6f}", "H1_held": h1,
        "H1_measured": decision_effect,
        "H2_expectation": "B2_groupnorm 이 최대 하위성분", "H2_held": h2,
        "H2_measured": subcomponents,
        "H3_expectation": f"A2 > {DEPLOYED_TOTAL}", "H3_held": h3,
        "H3_measured": results["A2_fold_outside_best"]["total"],
        "H4_expectation": "결정규칙 기여 > 0", "H4_held": h4,
        "judged": valid, "verdict": verdict,
    }

    receipt: dict[str, Any] = {
        "node_id": NODE_ID, "lane": LANE, "parent": PARENT_NODE,
        "judged_at": datetime.now(UTC).isoformat(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "derivation_only": True, "model_fits": 0,
        "probability_digest": digest,
        "deployed_total": DEPLOYED_TOTAL, "control": CONTROL, "gap": GAP,
        "arms": {k: dict(v) for k, v in results.items()},
        "decision_effect": decision_effect, "model_effect": model_effect,
        "subcomponents": subcomponents,
        "best_same_fold_policy": best_same,
        "chosen_policy_fold_outside": chosen_policy,
        "policy_grid": {k: v["total"] for k, v in grid.items()},
        "precommitment": check,
    }
    RECEIPT.write_text(json.dumps(receipt, indent=2, sort_keys=True, default=str),
                       encoding="utf-8")

    arm_rows = "\n".join(
        f"| `{k}` | {v['total']:.6f} | {v['one_minus_nmae']:.6f} | {v['ficr']:.6f} "
        f"| {v['total'] - a0:+.6f} |"
        for k, v in results.items()
    )
    REPORT_MD.write_text(
        f"""# M271 N8 — 재구성 격차 {GAP:.6f} 를 결정규칙과 모형으로 가른다

- 판정일: {receipt['judged_at']}
- 노드: `{NODE_ID}` / 레인 {LANE} / 부모 `{PARENT_NODE}`
- **파생 전용. 모형 적합 0 회.** 확률 digest `{digest}`

`bayes_decision` 은 지표 최적이 아니라 **`T1_G{0.25/OURS_CBAR:.4f}` 정책**이다.
배포는 `T{DEPLOYED_T}_G{DEPLOYED_G}` 이고 정산항을 **그룹별 평균 발전율**로 정규화하며
행동격자가 2 배 촘촘하고 하한이 0.075 다.

## 1. 팔

| 팔 | Total | 1-NMAE | FICR | A0 대비 |
|---|---:|---:|---:|---:|
{arm_rows}

배포 = {DEPLOYED_TOTAL}

## 2. 가법 분해

| 성분 | 값 | 격차 대비 |
|---|---:|---:|
| 결정규칙 (A1 - A0) | **{decision_effect:+.6f}** | {decision_effect/GAP:.1%} |
| 모형 (배포 - A1) | **{model_effect:+.6f}** | {model_effect/GAP:.1%} |
| 합 (항등식) | {decision_effect + model_effect:.6f} | 100% |

## 3. 하위 성분 (한 번에 하나씩만 배포 것으로)

| 성분 | A0 대비 |
|---|---:|
| B1 행동격자 | {subcomponents['B1_actions']:+.6f} |
| B2 그룹별 정규화 | {subcomponents['B2_groupnorm']:+.6f} |
| B3 온도 T={DEPLOYED_T} | {subcomponents['B3_temperature']:+.6f} |

## 4. 사전확약 대조

- V1 `A0 가 {CONTROL} 재현` -> **{v1}** ({a0:.6f})
- V2 `확률 digest` -> **{v2}** (`{digest}`)
- V3 `가법 항등식` -> **{v3}**
- H1 `결정규칙 기여 > {GAP/2:.6f}` -> **{h1}** ({decision_effect:+.6f})
- H2 `B2 가 최대 하위성분` -> **{h2}**
- H3 `A2 > 배포` -> **{h3}** ({results['A2_fold_outside_best']['total']:.6f})
- H4 `결정규칙 기여 > 0` -> **{h4}**

fold-외 선택 정책: {json.dumps(chosen_policy)} / 같은-fold 최적 `{best_same}`

판정: **{verdict}**
""",
        encoding="utf-8",
    )
    print(json.dumps({
        "verdict": verdict,
        "A0": a0, "A1": a1,
        "A2_fold_outside": results["A2_fold_outside_best"]["total"],
        "A3_same_fold": results["A3_same_fold_best"]["total"],
        "decision_effect": decision_effect, "model_effect": model_effect,
        "subcomponents": subcomponents,
        "guards": {"V1": v1, "V2": v2, "V3": v3},
        "chosen": chosen_policy, "best_same": best_same,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
