"""M271 C7c — `deep` 을 시간 분할에서 재확인한다: 누출 없는 면에서도 남는가.

C1N83(C7b)이 `deep`(255 잎) 팔에서 sigma_v **2.83% 감소**를 냈고 C16 문턱(2.72%)을
통과했다. 그러나 판정문에 `PENDING_CHRONOLOGICAL_CONFIRMATION` 을 남겼다 — **무작위
KFold 면에서 잰 값**이기 때문이다.

    C1N58 학습행 shuffle KFold OOF   1.0923
    C1N58 학습행 blocked             1.6772
    C1N71 평가 fold **test 행**       g1 1.4957 / g2 1.5947 / g3 1.6521 (allweather)

무작위 KFold 는 이웃 시각이 학습에 들어가므로 잔차가 작다. C1N54 가 잰 누출분이
17.8~21.3%p 다. 따라서 2.83% 가 누출된 면의 성질일 수 있다.

**어느 면에서 재야 하는가.** C1N69 의 반응곡선은 `sigma_v` 를 Total 로 환산하는데, 그
보정이 선 면은 **평가 fold 의 test 행**이다(C1N66·C1N71 이 같은 면). 그리고 모형이 실제로
쓰는 것도 test 행 예측이다. 그러므로 학습행 OOF 가 아니라 **test 행에서 재는 것**이 옳다.

이 노드는 학습을 **엄격히 시간순**으로 한다 — fold 시작 이전 행으로만 적합하고 fold
test 행에 예측한다. 내부 KFold 가 없으므로 누출 경로 자체가 없다.

**① 방법 리서치**

  새 방법 없음. C1N42 의 `teach()` 가 보류 fold 에 쓰는 것과 **같은 절차**(전량 적합 후
  test 예측)이고, 학습 구간만 시간순으로 자른다. 용량 override 만 C1N82 에서 가져온다.

**② 사양 동결**

  분할   fold 시작 시각 이전 행으로 학습, fold test 행에 예측. 그룹별.
         **내부 KFold 없음** — 이것이 C1N82 와의 유일한 차이다.
  열     `all_weather_columns` (C1N82 `base`/`deep` 과 동일).
  팔     base(400/63) / deep(400/**255**). C1N82 의 최선 팔만 가져온다 —
         `long`(2.07%)·`seq`(-0.29%)·`seq_deep`(2.06%)은 `deep` 에 뒤졌으므로 제외.
  지표   `std(scada_ws - 예측)`, 그룹별과 pooled.

  **타당성 가드**
    V1  base 의 **그룹별** sigma 가 C1N71 의 allweather test 행 값
        (g1 1.4957 / g2 1.5947 / g3 1.6521)의 ±0.05 이내.
        **같은 열·같은 표적·같은 test 행**이므로 일치해야 한다. C1N82 에서 기준을
        잘못 골라 V1 이 발화했으므로 이번엔 like-for-like 를 명시한다.
    V2  두 팔이 같은 행·같은 학습구간을 쓴다.

  사전확약 (V1·V2 통과시에만 판정):
    H1  `deep` < `base` (누출 없는 면에서도 용량이 값을 한다).
    H2  감소율이 C16 문턱 **2.72% 이상**. 이것이 승격 가능성의 판정이다.
    H3  세 그룹 **모두** 개선.
    H4  시간 분할 감소율이 무작위 KFold 의 **2.83% 보다 작다**.
        누출이 개선을 부풀렸다면 참이어야 한다. **부호 예단 없음** — 반대면 무작위
        KFold 가 오히려 용량 이득을 감췄다는 뜻이고 그것도 정보다.

  H1·H2 가 참이면 `deep` 은 **실제 평가면에서 C16 을 통과한 첫 후보**가 되고, 다음은
  전체 파이프라인 검정과 동결 게이트다. H2 가 거짓이면 teacher 용량 축도 문턱 미달이고,
  그러면 C1N69 가 요구한 27.3% 에 닿을 방향이 남지 않는다.

게이트 미수정. lockbox·외부데이터 미사용. `scada_ws` 는 teacher 표적으로만(C1N39).
제출 없음.
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
from lightgbm import LGBMRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))

from m271_cycle37_band_loss import fold_rows
from m271_cycle42_teacher_restored import TEACHER_PARAMS, all_weather_columns
from run_sequence_classifier import _surface

from baram.constants import CAPACITIES_KWH

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
C7B_RECEIPT = REPORTS / "m271_c7b_rejudge_receipt.json"
REPORT_MD = REPORTS / "m271_c7c_chronological.md"
RECEIPT = REPORTS / "m271_c7c_chronological_receipt.json"

NODE_ID = "C1N84_TEACHER_CHRONOLOGICAL"
LANE = "L6"
PARENT_NODE = "C1N83_TEACHER_SCALEUP_REJUDGED"

# C1N71 이 캐시(평가 fold test 행)에서 잰 allweather teacher 잔차. like-for-like 기준.
C71_ALLWEATHER = {1: 1.4957, 2: 1.5947, 3: 1.6521}
V1_TOLERANCE = 0.05
SHUFFLE_REDUCTION = 0.0283  # C1N83 이 무작위 KFold 면에서 잰 값
RESPONSE_SLOPE = 0.164
MAGNITUDE_FLOOR = 0.15
REMAINING_GAP = 0.66 - 0.630310
REQUIRED_REDUCTION = MAGNITUDE_FLOOR * REMAINING_GAP / RESPONSE_SLOPE

ARMS: dict[str, dict[str, Any]] = {"base": {}, "deep": {"num_leaves": 255}}


def main() -> int:
    surface, _base, _aux = _surface()
    surface["forecast_kst_dtm"] = pd.to_datetime(surface["forecast_kst_dtm"])
    surface["capacity"] = surface["group_id"].map(CAPACITIES_KWH).astype(float)
    surface = surface.loc[surface["actual_kwh"].notna()].reset_index(drop=True)
    columns = all_weather_columns(surface)

    folds = fold_rows()
    residuals: dict[str, dict[int, list[np.ndarray]]] = {a: {} for a in ARMS}
    rows_used = 0
    fits = 0

    for _probe_fold, meta in folds.items():
        train = surface.loc[surface["forecast_kst_dtm"] < meta["start"]]
        test = surface.loc[
            np.array([
                (fid, gid) in meta["keys"]
                for fid, gid in zip(surface["forecast_id"], surface["group_id"],
                                    strict=True)
            ])
        ]
        for group in (1, 2, 3):
            tr = train.loc[
                (train["group_id"] == group) & train["scada_ws"].notna()
            ]
            te = test.loc[
                (test["group_id"] == group) & test["scada_ws"].notna()
            ]
            if len(tr) < 200 or len(te) < 50:
                continue
            rows_used += len(te)
            x_tr = tr.loc[:, columns].astype("float32")
            y_tr = tr["scada_ws"].to_numpy(dtype="float64")
            x_te = te.loc[:, columns].astype("float32")
            y_te = te["scada_ws"].to_numpy(dtype="float64")
            for arm, overrides in ARMS.items():
                model = LGBMRegressor(**{**TEACHER_PARAMS, **overrides})
                model.fit(x_tr, y_tr)
                fits += 1
                residuals[arm].setdefault(group, []).append(y_te - model.predict(x_te))

    results: dict[str, dict[str, float]] = {}
    for arm, per_group in residuals.items():
        entry: dict[str, float] = {}
        pooled = []
        for group, parts in per_group.items():
            joined = np.concatenate(parts)
            entry[f"g{group}"] = float(np.std(joined, ddof=1))
            pooled.append(joined)
        entry["overall"] = float(np.std(np.concatenate(pooled), ddof=1))
        results[arm] = entry

    v1 = bool(all(
        abs(results["base"].get(f"g{g}", 1e9) - C71_ALLWEATHER[g]) <= V1_TOLERANCE
        for g in (1, 2, 3)
    ))
    v2 = True  # 두 팔이 같은 루프 안에서 같은 tr/te 를 쓴다.

    reduction = 1.0 - results["deep"]["overall"] / results["base"]["overall"]
    implied_total = reduction * RESPONSE_SLOPE

    h1 = bool(results["deep"]["overall"] < results["base"]["overall"])
    h2 = bool(reduction >= REQUIRED_REDUCTION)
    h3 = bool(all(
        results["deep"].get(f"g{g}", 1e9) < results["base"].get(f"g{g}", 0.0)
        for g in (1, 2, 3)
    ))
    h4 = bool(reduction < SHUFFLE_REDUCTION)

    if not v1:
        verdict = "BASELINE_MISMATCH_RESULT_VOID"
    elif h1 and h2:
        verdict = "DEEP_CONFIRMED_ON_CHRONOLOGICAL_SURFACE"
    elif h1:
        verdict = "DEEP_HELPS_BUT_BELOW_MAGNITUDE_GATE"
    else:
        verdict = "CAPACITY_GAIN_WAS_LEAKAGE_ARTEFACT"

    c7b = json.loads(C7B_RECEIPT.read_text(encoding="utf-8"))

    payload: dict[str, Any] = {
        "node_id": NODE_ID,
        "lane": LANE,
        "parent": PARENT_NODE,
        "surface": "평가 fold test 행 (C1N66·C1N71 과 같은 면)",
        "split": "fold 시작 이전 행으로만 학습. 내부 KFold 없음 — 누출 경로 없음",
        "model_fits": fits,
        "test_rows": rows_used,
        "arms": results,
        "reduction": reduction,
        "implied_total_gain": implied_total,
        "shuffle_reduction_c7b": SHUFFLE_REDUCTION,
        "required_reduction_for_c16": REQUIRED_REDUCTION,
        "c16_floor_total": MAGNITUDE_FLOOR * REMAINING_GAP,
        "reference_c71_allweather": C71_ALLWEATHER,
        "checks": {"V1_base_matches_c71_allweather": v1, "V2_same_rows": v2},
        "hypotheses": {
            "H1_deep_beats_base": h1,
            "H2_clears_magnitude_gate": h2,
            "H3_helps_all_groups": h3,
            "H4_smaller_than_shuffle_reduction": h4,
        },
        "c7b_verdict": c7b.get("verdict"),
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
        "# M271 C7c — `deep` 의 시간 분할 재확인",
        "",
        f"노드 `{NODE_ID}` / 레인 {LANE} / 부모 `{PARENT_NODE}`",
        "",
        "**평가 fold test 행**에서 잰다 — C1N69 반응곡선이 보정된 면이고 모형이 실제로 "
        "쓰는 면이다. 학습은 fold 시작 이전 행으로만 하며 **내부 KFold 가 없어 누출 경로 "
        "자체가 없다**.",
        "",
        f"적합 {fits} 회 / test 행 {rows_used:,}",
        "",
        "## 1. sigma_v (test 행)",
        "",
        "| 팔 | 전체 | g1 | g2 | g3 |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        r = results[arm]
        lines.append(
            f"| {arm} | **{r['overall']:.4f}** | {r.get('g1', float('nan')):.4f} | "
            f"{r.get('g2', float('nan')):.4f} | {r.get('g3', float('nan')):.4f} |"
        )
    lines += [
        "",
        f"C1N71 allweather 기준 g1 {C71_ALLWEATHER[1]} / g2 {C71_ALLWEATHER[2]} / "
        f"g3 {C71_ALLWEATHER[3]}",
        "",
        f"**감소율 {reduction:+.2%}** -> 환산 Total **{implied_total:+.6f}** "
        f"/ C16 문턱 {REQUIRED_REDUCTION:.2%} ({MAGNITUDE_FLOOR * REMAINING_GAP:.6f})",
        "",
        f"무작위 KFold 면(C1N83)에서는 {SHUFFLE_REDUCTION:.2%} 였다.",
        "",
        "## 2. 사전확약",
        "",
        f"- V1 base 가 C1N71 allweather 와 ±{V1_TOLERANCE} 이내 -> **{v1}**",
        f"- H1 deep < base -> **{h1}**",
        f"- H2 C16 문턱 통과 -> **{h2}**",
        f"- H3 세 그룹 모두 개선 -> **{h3}**",
        f"- H4 시간 분할 감소가 무작위 KFold 보다 작다 -> **{h4}**",
        "",
        "## 3. 판정",
        "",
        f"**{verdict}**",
        "",
        f"digest `{payload['digest']}`",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("=== C7c 완료 ===")
    print(f"[C7c] 적합 {fits} / test 행 {rows_used:,}")
    for arm in ARMS:
        r = results[arm]
        print(f"[C7c] {arm:5s} sigma {r['overall']:.4f} "
              f"(g1 {r.get('g1', float('nan')):.4f} / g2 {r.get('g2', float('nan')):.4f} / "
              f"g3 {r.get('g3', float('nan')):.4f})")
    print(f"[C7c] 감소 {reduction:+.2%} -> Total {implied_total:+.6f} "
          f"(문턱 {REQUIRED_REDUCTION:.2%}) / 무작위KFold 였을 때 {SHUFFLE_REDUCTION:.2%}")
    print(f"[C7c] V1 {v1} / H1 {h1} / H2 {h2} / H3 {h3} / H4 {h4}")
    print(f"[C7c] 판정: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
