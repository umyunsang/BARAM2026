"""M271 N7 — C1N93 재판정. 계측기를 고치고 같은 산출물을 다시 읽는다(**학습 없음**).

**C1N93 은 자기 가드로 VOID 다. 그 발화가 옳았고, 발화 이유는 결함이 아니었다.**

  C1N93 의 V2 사양은 `재배열 후 분위 교차 0 건` 인데 코드는 재배열이 일어나기 **전**의
  원시 행렬을 쟀다 — `quantiles_to_bins` 안에서 `np.maximum.accumulate` 가 재배열한다.
  원시 교차 63,404 건은 19,785 행 x 18 인접쌍 = 356,130 쌍의 **17.8%** 이고, 독립 적합
  분위수회귀에서 **정상**이다. 애초에 그것 때문에 Chernozhukov, Fernandez-Val &
  Galichon (2010) 의 재배열이 존재한다. 정상 현상에 가드를 잘못 건 것이다.

  **결과를 본 뒤 사양을 재해석하지 않는다.** C1N93 은 VOID 로 두고, 이 노드가 **새
  사전확약**으로 재판정한다. 선례: C1N82 -> C1N83, C1N74/75 -> C1N76, C1N70 -> C1N71.

**왜 재계산하지 않는가.** 고칠 것은 **데이터 측정**이 아니라 **코드 성질**이다. 재배열 후
단조성은 `np.maximum.accumulate` 의 정의상 성질이므로 입력에 의존하지 않는다. 따라서
무작위·적대적 입력에 대한 단위검정으로 확인하는 것이 재실행보다 **강한** 검증이다.
팔 점수는 재배열이 이미 적용된 뒤의 값이므로 C1N93 의 산출물 그대로 유효하다.

**① 방법 리서치**

  이 노드의 과업은 '이미 실행된 산출물을 고친 계측기로 재판정' 이다. 표준 처리는
  **재계산 없는 재판독**(re-adjudication)이며 이 프로젝트에 확립된 절차가 있다 —
  C1N83 이 C1N82 의 산출물을 재계산 없이 재판정했고, C1N91 이 C1N90 의 산출물을
  C1N76 계측기로 재검정했다. 그 절차를 따른다.

**② 사양 동결**

  입력   `reports/m271_n6_ordinal_representation_receipt.json` (C1N93 산출물)
  계산   없음. 파생과 단위검정만.

  **타당성 가드**
    V1  C1N93 의 V1/V3/V4/V5 가 전부 참이었다(receipt 판독). 하나라도 거짓이면 판정 불가.
    V2' **교정된 계측기** — `quantiles_to_bins` 가 CDF 보간에 넘기는 앵커열이
        무작위 500 행 x 19 분위, 완전역전, 동률, 극단꼬리 입력 전부에서 **비감소**.
        위반 0 건이어야 한다. 이것이 C1N93 이 재려던 바로 그 성질이다.
    V3  원시 교차율을 **진단값으로만** 기록한다. 가드가 아니다 — 독립 적합 분위수회귀에서
        교차는 예상되는 현상이고, 그것이 재배열의 존재 이유다.

  사전확약 (V1·V2' 통과시에만 판정):
    H1  `mq19` > `onehot`.
    H2  최선 팔(`mq19`/`blend`)의 이득이 검출문턱 0.001013 이상.
    H3  차이가 **FICR 쪽**.
    H4  `blend` > max(`onehot`, `mq19`).
    H5  세 fold **전부**에서 `mq19` < `onehot` — 부호가 일관하면 pooled 우연이 아니다.

  **부호 예단 없음.** C1N93 의 실측은 이미 receipt 에 있으므로 이 노드는 그것을 재해석
  하는 것이 아니라, 계측기가 고쳐진 상태에서 **판정 자격**을 부여하는 것이다.
  H1 이 거짓이면 CDF 추정 축이 근거를 갖고 닫힌다.

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

sys.path.insert(0, str(Path(__file__).resolve().parent))

from m271_n6_ordinal_representation import (
    C1N44_BAND_EFFECT,
    CONTROL,
    DETECTION_THRESHOLD,
    LEVELS,
    quantiles_to_bins,
)

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
SOURCE = REPORTS / "m271_n6_ordinal_representation_receipt.json"
REPORT_MD = REPORTS / "m271_n7_cdf_rejudge.md"
RECEIPT = REPORTS / "m271_n7_cdf_rejudge_receipt.json"

NODE_ID = "C1N94_CDF_REJUDGE"
LANE = "L3"
PARENT_NODE = "C1N93_ORDINAL_REPRESENTATION"

RAW_PAIRS = 19785 * (len(LEVELS) - 1)


def anchor_monotone_violations(qmat: np.ndarray) -> int:
    """`quantiles_to_bins` 가 CDF 보간에 넘기는 앵커열의 단조 위반 수.

    함수 본체와 **같은 식**을 쓴다. 여기서 0 이면 보간 입력이 유효한 CDF 지지점이고,
    C1N93 의 V2 가 재려던 성질이 성립한다.
    """
    q = np.maximum.accumulate(np.clip(qmat, 0.0, 1.0), axis=1)
    lv = LEVELS.astype("float64")
    w_lo = lv[0] * (q[:, 1] - q[:, 0]) / (lv[1] - lv[0])
    w_hi = (1.0 - lv[-1]) * (q[:, -1] - q[:, -2]) / (lv[-1] - lv[-2])
    lo = np.clip(q[:, 0] - w_lo, 0.0, None)
    hi = np.clip(q[:, -1] + w_hi, None, 1.0)
    xs = np.maximum.accumulate(
        np.concatenate([lo[:, None], q, hi[:, None]], axis=1), axis=1
    )
    return int((np.diff(xs, axis=1) < -1e-12).sum())


def adversarial_inputs() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(20260806)
    k = len(LEVELS)
    return {
        "random_unsorted": rng.normal(0.4, 0.15, (500, k)),
        "fully_reversed": np.tile(np.linspace(0.9, 0.1, k), (50, 1)),
        "all_tied": np.full((50, k), 0.37),
        "extreme_low": rng.normal(0.005, 0.002, (50, k)),
        "extreme_high": rng.normal(0.995, 0.002, (50, k)),
        "out_of_range": rng.normal(0.5, 3.0, (200, k)),
    }


def main() -> int:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    pre = source["precommitment"]
    arms = source["arms"]
    gains = source["gains_vs_onehot"]
    per_fold = source["per_fold"]

    v1 = bool(
        pre["V1_held"] and pre["V3_held"] and pre["V4_held"] and pre["V5_held"]
    )

    cases = adversarial_inputs()
    violations = {name: anchor_monotone_violations(q) for name, q in cases.items()}
    probs = {name: quantiles_to_bins(q, LEVELS) for name, q in cases.items()}
    prob_ok = bool(
        all(
            np.isfinite(p).all() and abs(p.sum(axis=1) - 1.0).max() < 1e-9
            for p in probs.values()
        )
    )
    v2 = bool(sum(violations.values()) == 0 and prob_ok)

    raw_crossings = int(pre["V2_measured"])
    raw_rate = raw_crossings / RAW_PAIRS

    valid = v1 and v2
    best_arm = max(("mq19", "blend"), key=lambda a: gains[a])
    best_gain = gains[best_arm]
    ficr_contrib = 0.5 * (arms[best_arm]["ficr"] - arms["onehot"]["ficr"])
    nmae_contrib = 0.5 * (
        arms[best_arm]["one_minus_nmae"] - arms["onehot"]["one_minus_nmae"]
    )
    fold_signs = {
        f: per_fold[f]["total"]["mq19"] - per_fold[f]["total"]["onehot"]
        for f in sorted(per_fold)
    }

    if valid:
        h1: bool | None = bool(gains["mq19"] > 0.0)
        h2: bool | None = bool(best_gain >= DETECTION_THRESHOLD)
        h3: bool | None = bool(abs(ficr_contrib) > abs(nmae_contrib))
        h4: bool | None = bool(
            arms["blend"]["total"] > max(arms["onehot"]["total"], arms["mq19"]["total"])
        )
        h5: bool | None = bool(all(d < 0 for d in fold_signs.values()))
        if h2 and (h1 or h4):
            verdict = "CDF_ESTIMATOR_AXIS_OPEN"
        elif h1 or h4:
            verdict = "CDF_ESTIMATOR_SUBTHRESHOLD"
        else:
            verdict = "SOFTMAX_BEATS_QUANTILE_CDF_ESTIMATOR_AXIS_CLOSES"
    else:
        h1 = h2 = h3 = h4 = h5 = None
        verdict = "REJUDGE_GUARD_FAILED"

    check = {
        "V1_expectation": "C1N93 의 V1/V3/V4/V5 가 전부 참",
        "V1_held": v1,
        "V1_source": {k: pre[k] for k in ("V1_held", "V3_held", "V4_held", "V5_held")},
        "V2_expectation": "교정된 계측기 — 재배열 후 앵커열 단조 위반 0 건 (적대적 입력 포함)",
        "V2_held": v2,
        "V2_violations": violations,
        "V2_prob_normal": prob_ok,
        "V3_diagnostic_raw_crossing_rate": raw_rate,
        "V3_note": "원시 교차는 독립 적합 분위수회귀의 정상 현상이며 재배열의 존재 이유다. 가드가 아니다",
        "H1_expectation": "mq19 > onehot",
        "H1_held": h1,
        "H1_measured": gains["mq19"],
        "H2_expectation": f"최선 팔 이득 >= {DETECTION_THRESHOLD}",
        "H2_held": h2,
        "H2_measured": best_gain,
        "H3_expectation": "차이가 FICR 쪽",
        "H3_held": h3,
        "H3_ficr_contrib": ficr_contrib,
        "H3_nmae_contrib": nmae_contrib,
        "H4_expectation": "blend > max(onehot, mq19)",
        "H4_held": h4,
        "H5_expectation": "세 fold 전부에서 mq19 < onehot",
        "H5_held": h5,
        "H5_measured": fold_signs,
        "judged": valid,
        "verdict": verdict,
    }

    receipt: dict[str, Any] = {
        "node_id": NODE_ID,
        "lane": LANE,
        "parent": PARENT_NODE,
        "judged_at": datetime.now(UTC).isoformat(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_receipt": str(SOURCE.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "derivation_only": True,
        "model_fits": 0,
        "control": CONTROL,
        "c1n44_band_effect": C1N44_BAND_EFFECT,
        "arms": arms,
        "gains_vs_onehot": gains,
        "per_fold": per_fold,
        "best_arm": best_arm,
        "precommitment": check,
    }
    RECEIPT.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )

    arm_rows = "\n".join(
        f"| `{a}` | {arms[a]['total']:.6f} | {arms[a]['one_minus_nmae']:.6f} "
        f"| {arms[a]['ficr']:.6f} | {gains[a]:+.6f} |"
        for a in ("onehot", "band", "mq19", "blend")
    )
    viol_rows = "\n".join(
        f"| `{name}` | {q.shape[0]} | {violations[name]} |"
        for name, q in cases.items()
    )
    fold_rows_md = "\n".join(
        f"| `{f}` | {per_fold[f]['total']['onehot']:.6f} "
        f"| {per_fold[f]['total']['mq19']:.6f} | {fold_signs[f]:+.6f} |"
        for f in sorted(per_fold)
    )

    REPORT_MD.write_text(
        f"""# M271 N7 — C1N93 재판정: 계측기를 고치고 같은 산출물을 다시 읽는다

- 판정일: {receipt['judged_at']}
- 노드: `{NODE_ID}` / 레인 {LANE} / 부모 `{PARENT_NODE}`
- **파생 전용. 모형 적합 0 회.** 출처 `{receipt['source_receipt']}`
  sha256 `{receipt['source_sha256'][:16]}`

C1N93 의 V2 사양은 `재배열 후 분위 교차 0 건` 인데 코드는 재배열 **전**의 원시 행렬을
쟀다. 원시 교차 {raw_crossings:,} 건은 {RAW_PAIRS:,} 인접쌍의 **{raw_rate:.1%}** 로
독립 적합 분위수회귀에서 정상이며, 그것이 재배열의 존재 이유다. 정상 현상에 가드를
잘못 걸었다. **결과를 본 뒤 사양을 재해석하지 않고** 새 사전확약으로 재판정한다.

## 1. 산출물 (C1N93, 재계산 없음)

| 팔 | Total | 1-NMAE | FICR | onehot 대비 |
|---|---:|---:|---:|---:|
{arm_rows}

대조군 {CONTROL} / `band` 는 C1N44 의 {C1N44_BAND_EFFECT} 재현 대조군이다.

## 2. 교정된 계측기 (V2')

`quantiles_to_bins` 가 CDF 보간에 넘기는 앵커열의 단조 위반. 적대적 입력 포함.

| 입력 | 행 | 위반 |
|---|---:|---:|
{viol_rows}

확률행렬 정상: {prob_ok}

## 3. fold 별 부호

| fold | onehot | mq19 | 차이 |
|---|---:|---:|---:|
{fold_rows_md}

## 4. 사전확약 대조

- V1 `C1N93 의 V1/V3/V4/V5 전부 참` -> **{v1}**
- V2' `교정 계측기 위반 0 건` -> **{v2}**
- H1 `mq19 > onehot` -> **{h1}** (실측 {gains['mq19']:+.6f})
- H2 `최선 팔 이득 >= {DETECTION_THRESHOLD}` -> **{h2}** (실측 {best_gain:+.6f}, 팔 `{best_arm}`)
- H3 `차이가 FICR 쪽` -> **{h3}** (FICR {ficr_contrib:+.6f} / 1-NMAE {nmae_contrib:+.6f})
- H4 `blend > max(onehot, mq19)` -> **{h4}**
- H5 `세 fold 전부 mq19 < onehot` -> **{h5}**

판정: **{verdict}**
""",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "verdict": verdict,
                "V1": v1,
                "V2_corrected": v2,
                "violations": violations,
                "raw_crossing_rate": round(raw_rate, 4),
                "H": {"H1": h1, "H2": h2, "H3": h3, "H4": h4, "H5": h5},
                "fold_signs": {k: round(v, 6) for k, v in fold_signs.items()},
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
