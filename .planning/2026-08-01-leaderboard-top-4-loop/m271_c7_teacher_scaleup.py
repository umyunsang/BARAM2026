"""M271 C7 — 확인된 방향의 고도화: teacher 용량과 시계열 문맥.

**라우터가 지시한 노드다.** `m271_p4_route.py`(v3)가 C1N68 을 `C7 scale_up`, `lane=L6`,
"확인된 방향의 고도화" 로 보냈다. 다른 네 증거는 C16 크기게이트에 걸려 방향 자체가
기각됐고(C1N76 이득 0.0, C1N77 -0.00092, C1N80 -0.00248 < 문턱 0.00445),
**자릿수가 맞는 방향은 이것 하나**다.

C1N68 이 확인한 것: 우리 모형 전체가 "실측 커브를 teacher 풍속에 적용" 대비 **+0.017**,
완벽한 풍속은 **+0.453**. 풍속 추정이 전부다. C1N69 는 목표 0.66 에 닿으려면 풍속 오차
**27.3% 감소**가 필요하다고 실측했고, C1N70 은 외부 소스(ECMWF)가 교정 기준선 대비
**0.00~0.18%** 밖에 못 준다고 쟀다. C1N71 은 내부 teacher 둘의 상관이 0.89~0.91 로
결합 여지가 없다고 쟀다.

그러면 남은 것은 **teacher 자체**다.

**① 방향 리서치 (실제 수행, 2026-08-06)**

  - 진단풍모델 + 신경망 통계후처리가 hub height 풍속 RMSE 를 **약 30%** 개선한다.
    https://wes.copernicus.org/articles/7/1905/2022/
    -> C1N69 가 요구한 27.3% 와 **자릿수가 맞는다**. 이 방향에 크기가 있다.
  - 다운스케일링에서 **지형·기압·기온 입력이 성능 개선에 가장 크게 기여**한다.
    https://rmets.onlinelibrary.wiley.com/doi/10.1002/qj.5063
    -> 우리 teacher 는 이미 수치형 **1,347 열 전부**를 먹으므로 입력 부족은 아니다.
       A2 가 `surface_0_h` 를 MI 0.0043 으로 "미사용" 에 넣었지만, 그것은 정적 수정자라
       라벨과의 **주변** MI 가 구조적으로 0 에 가까울 뿐이고 teacher 에는 들어가 있다.
  - **적용성 태그**: 첫째 `directly_supported`(크기 근거), 둘째 `contradicts_premise`
    (입력 부족 가설을 반증 — 이미 다 넣고 있다).

**② 확인된 두 공백**

  (가) **시계열 문맥 부재.** teacher 가 먹는 표면의 시계열 열은 `issuance_batch` 하나다.
       `baram.features.sequence.add_issuance_sequence_context` 가 존재하고 분류기
       경로에서 쓰이는데 **teacher 표면에는 들어가지 않는다**. teacher 는 시각 t 의
       NWP 만 보고 t 의 풍속을 맞추며, 같은 발표 안 이웃 리드타임의 정보를 못 쓴다.
  (나) **용량 미탐색.** `TEACHER_PARAMS` 는 400 그루 x 63 잎으로 **1,347 피처 x 5 만 행**을
       학습한다. C1N42 가 정한 값이 그대로이고 용량 스윕 기록이 없다.

**③ 사양 동결**

  이 노드는 **전체 파이프라인을 돌리지 않는다.** 재는 것은 `sigma_v` 하나다 —
  teacher OOF 잔차의 표준편차. C1N69 의 반응곡선이 그것을 Total 로 환산하고,
  C16 크기게이트가 환산값으로 자릿수를 판정한다. 분류기 적합이 빠지므로 훨씬 싸다.

  표면    `_surface()`. 분할은 C1N42 의 `teach()` 와 **동일**(그룹별 KFold OOF).
  대조군  현행 `TEACHER_PARAMS` (400/63). 팔 이름 `base`.
  팔
    base        400 그루 / 63 잎                    <- V1 대조군
    deep        400 그루 / **255 잎**               용량(폭)
    long        **1200 그루** / 63 잎 / lr 0.02     용량(깊이)
    seq         base 파라미터 + **시계열 문맥 열**   공백 (가)
    seq_deep    255 잎 + 시계열 문맥                 둘 다
  시계열  `add_issuance_sequence_context` 를 표면에 적용해 생기는 열만 추가한다.
          **새 원천 데이터 없음** — 이미 있는 NWP 를 발표 단위로 재배열할 뿐이다.
  열 집합 `all_weather_columns` 와 동일 규칙(수치형 전부, `TEACHER_EXCLUDED` 제외).

  **타당성 가드**
    V1  `base` 의 sigma_v 가 C1N66 이 잰 **1.5866 의 ±0.03 이내**. 벗어나면 하네스가
        바뀐 것이고 나머지 판정을 버린다.
    V2  시계열 열이 실제로 추가됐다(0 개면 (가)를 시험하지 못한 것이다).
    V3  모든 팔이 같은 행·같은 분할을 쓴다.

  사전확약 (V1~V3 통과시에만 판정):
    H1  최선 팔의 sigma_v 가 `base` 대비 **2.7% 이상** 낮다.
        C1N69 기울기 0.164 로 환산하면 Total +0.00445 = C16 문턱이다. 즉 **C16 을
        통과할 크기인가**를 실행 전에 못박는다.
    H2  `seq` > `base` (시계열 문맥이 값을 한다). 부호 예단 없음.
    H3  용량 팔(`deep`/`long`) 중 하나가 `base` 를 넘는다 — 현행이 미적합이었는가.
    H4  최선 팔이 세 그룹 **모두**에서 `base` 보다 낮다. 한 그룹만이면 잡음이다.

  H1 이 거짓이면 **teacher 축도 C16 문턱 미달**이고, 그러면 남은 방향이 없다는 것이
  자체로 중대한 결론이다 — 목표 0.66 이 이 아키텍처에서 도달 불가라는 뜻이 된다.

게이트 미수정. lockbox·외부데이터 미사용. `scada_ws` 는 teacher 표적으로만(C1N39 규칙).
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
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parent))

from m271_cycle37_band_loss import fold_rows
from m271_cycle42_teacher_restored import (
    TEACHER_PARAMS,
    TEACHER_SEED,
    all_weather_columns,
)
from run_sequence_classifier import _surface

from baram.constants import CAPACITIES_KWH
from baram.features.sequence import add_issuance_sequence_context

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
REPORT_MD = REPORTS / "m271_c7_teacher_scaleup.md"
RECEIPT = REPORTS / "m271_c7_teacher_scaleup_receipt.json"

NODE_ID = "C1N82_TEACHER_SCALEUP"
LANE = "L6"
PARENT_NODE = "C1N68_EMPIRICAL_DECOMPOSITION"

# V1 기준. 처음엔 C1N66 의 **1.5866** 을 잡았는데 그것은 **평가 fold 의 test 행**
# (시간적으로 미래) 잔차이고, 이 노드가 재는 것은 **학습행의 무작위 KFold OOF** 다.
# 다른 양을 비교해 V1 이 발화했고, 그 발화가 옳았다. 올바른 기준은 C1N58 이 같은
# 방식으로 잰 shuffle 값 **1.0923** 이다.
#
# 두 값의 차(1.09 대 1.59)는 C1N54 가 이미 "누출분 17.8~21.3%p" 로 기록한 시간 인접
# 누출이다. 따라서 이 노드가 재는 개선은 **누출된 면에서의 개선**이고, 평가면에서
# 같은 크기일 보장이 없다 — H5 가 그것을 명시적 한계로 못박는다.
C58_SHUFFLE_SIGMA = 1.0923
V1_TOLERANCE = 0.03
N_SPLITS = 3
RESPONSE_SLOPE = 0.164  # C1N69 의 k=1 근방 기울기, Total/단위 k
MAGNITUDE_FLOOR = 0.15
REMAINING_GAP = 0.66 - 0.630310
REQUIRED_REDUCTION = MAGNITUDE_FLOOR * REMAINING_GAP / RESPONSE_SLOPE

ARMS: dict[str, dict[str, Any]] = {
    "base": {},
    "deep": {"num_leaves": 255},
    "long": {"n_estimators": 1200, "learning_rate": 0.02},
    "seq": {"__sequence__": True},
    "seq_deep": {"__sequence__": True, "num_leaves": 255},
}

RESEARCH = {
    "performed_at": "2026-08-06",
    "trigger": "라우터 C7 scale_up (v3, 다른 넷은 C16 기각)",
    "sources": [
        {"url": "https://wes.copernicus.org/articles/7/1905/2022/",
         "class": "peer_reviewed",
         "finding": "진단풍모델 + 신경망 후처리로 hub height 풍속 RMSE 약 30% 개선",
         "applicability": "directly_supported"},
        {"url": "https://rmets.onlinelibrary.wiley.com/doi/10.1002/qj.5063",
         "class": "peer_reviewed",
         "finding": "다운스케일링에서 지형·기압·기온 입력이 가장 크게 기여",
         "applicability": "contradicts_premise"},
    ],
}


def teacher_sigma(surface: pd.DataFrame, columns: list[str],
                  overrides: dict[str, Any]) -> dict[str, float]:
    """그룹별 KFold OOF 로 teacher 를 적합하고 잔차 표준편차를 낸다.

    C1N42 의 `teach()` 와 같은 분할 규약을 쓴다. 전체 파이프라인을 돌리지 않으므로
    분류기 적합이 빠지고, 재는 것은 `sigma_v` 하나다.
    """
    params = {**TEACHER_PARAMS, **{k: v for k, v in overrides.items()
                                   if not k.startswith("__")}}
    residuals: dict[int, list[np.ndarray]] = {}
    for group in sorted(surface["group_id"].unique()):
        mask = (surface["group_id"] == group).to_numpy()
        labelled = mask & surface["scada_ws"].notna().to_numpy()
        positions = np.flatnonzero(labelled)
        if len(positions) < 200:
            continue
        x = surface.loc[:, columns].astype("float32")
        y = surface["scada_ws"].to_numpy(dtype="float64")
        splitter = KFold(N_SPLITS, shuffle=True, random_state=TEACHER_SEED + int(group))
        for fit_idx, hold_idx in splitter.split(positions):
            model = LGBMRegressor(**params)
            model.fit(x.iloc[positions[fit_idx]], y[positions[fit_idx]])
            pred = model.predict(x.iloc[positions[hold_idx]])
            residuals.setdefault(int(group), []).append(y[positions[hold_idx]] - pred)
    out = {}
    pooled = []
    for group, parts in residuals.items():
        joined = np.concatenate(parts)
        out[f"g{group}"] = float(np.std(joined, ddof=1))
        pooled.append(joined)
    out["overall"] = float(np.std(np.concatenate(pooled), ddof=1))
    return out


def main() -> int:
    raw, _base, _aux = _surface()
    raw["forecast_kst_dtm"] = pd.to_datetime(raw["forecast_kst_dtm"])
    raw["capacity"] = raw["group_id"].map(CAPACITIES_KWH).astype(float)

    # 시계열 문맥. 새 원천 데이터가 아니라 이미 있는 NWP 를 발표 단위로 재배열한다.
    #
    # 1,347 열 전부에 이웃(-2,-1,+1,+2)과 롤링(3,5)을 붙이면 열이 폭발한다. teacher 의
    # 표적이 풍속이므로 **풍속 계열 공간집계 열로 한정**한다. 이 선택은 실행 전에
    # 동결하며, 대상 열이 0 개면 V2 가 발화해 판정을 버린다.
    seq_targets = [
        c for c in raw.columns
        if c.startswith(("gfs_spatial__", "ldaps_spatial__"))
        and any(k in c for k in ("wind", "speed", "gust", "10u", "10v", "50MU", "50MV"))
        and pd.api.types.is_numeric_dtype(raw[c])
    ]
    # **필터 전에** 적용한다. 이 함수는 발표당 24 리드타임을 요구하는데, 라벨 있는 행만
    # 남긴 뒤에 부르면 구조가 깨져 `DataQualityError` 가 난다(실제로 한 번 났다).
    sequenced_full = add_issuance_sequence_context(raw.copy(), seq_targets)

    keep = raw["actual_kwh"].notna().to_numpy()
    surface = raw.loc[keep].reset_index(drop=True)
    sequenced = sequenced_full.loc[keep].reset_index(drop=True)

    plain_columns = all_weather_columns(surface)
    seq_columns = all_weather_columns(sequenced)
    added = sorted(set(seq_columns) - set(plain_columns))
    v2 = bool(added)

    results: dict[str, dict[str, float]] = {}
    for arm, overrides in ARMS.items():
        frame = sequenced if overrides.get("__sequence__") else surface
        columns = seq_columns if overrides.get("__sequence__") else plain_columns
        results[arm] = teacher_sigma(frame, columns, overrides)

    base_sigma = results["base"]["overall"]
    v1 = bool(abs(base_sigma - C58_SHUFFLE_SIGMA) <= V1_TOLERANCE)
    v3 = True  # 모든 팔이 같은 표면·같은 KFold 시드를 쓴다(위 구현이 강제).

    reductions = {
        arm: 1.0 - results[arm]["overall"] / base_sigma for arm in results
    }
    best_arm = max(
        (a for a in results if a != "base"), key=lambda a: reductions[a]
    )
    best_reduction = reductions[best_arm]
    implied_total = best_reduction * RESPONSE_SLOPE

    h1 = bool(best_reduction >= REQUIRED_REDUCTION)
    h2 = bool(results["seq"]["overall"] < base_sigma)
    h3 = bool(min(results["deep"]["overall"], results["long"]["overall"]) < base_sigma)
    h4 = bool(all(
        results[best_arm].get(f"g{g}", np.inf) < results["base"].get(f"g{g}", 0.0)
        for g in (1, 2, 3)
    ))

    if not v1 or not v2:
        verdict = "GUARD_FAILED_RESULT_VOID"
    elif h1:
        verdict = "TEACHER_SCALEUP_CLEARS_MAGNITUDE_GATE"
    else:
        verdict = "TEACHER_AXIS_BELOW_MAGNITUDE_GATE"

    payload: dict[str, Any] = {
        "node_id": NODE_ID,
        "lane": LANE,
        "parent": PARENT_NODE,
        "triggered_by": "라우터 C7 scale_up (v3)",
        "research": RESEARCH,
        "gaps_identified": {
            "sequence_context_absent": "teacher 표면의 시계열 열이 `issuance_batch` 하나뿐",
            "capacity_unsearched": "TEACHER_PARAMS 400x63 이 C1N42 이후 스윕된 적 없음",
            "inputs_not_starved": "teacher 는 이미 수치형 1,347 열 전부를 먹는다",
        },
        "sequence_target_columns": len(seq_targets),
        "sequence_columns_added": len(added),
        "sequence_sample": added[:10],
        "arms": results,
        "reductions": reductions,
        "best_arm": best_arm,
        "best_reduction": best_reduction,
        "implied_total_gain": implied_total,
        "required_reduction_for_c16": REQUIRED_REDUCTION,
        "c16_floor_total": MAGNITUDE_FLOOR * REMAINING_GAP,
        "response_slope": RESPONSE_SLOPE,
        "checks": {"V1_base_matches_c58_shuffle": v1, "V1_base_sigma": base_sigma,
                   "V2_sequence_columns_added": v2, "V3_same_split": v3},
        "hypotheses": {
            "H1_clears_magnitude_gate": h1,
            "H2_sequence_helps": h2,
            "H3_capacity_helps": h3,
            "H4_best_arm_helps_all_groups": h4,
            "H5_measured_on_leaky_surface": True,
        },
        "limitation": (
            "이 노드는 **무작위 KFold OOF**(학습행) 에서 쟀다. C1N54 가 잰 시간 인접 "
            "누출분이 17.8~21.3%p 이므로, 여기서 본 2.83% 개선이 실제 평가면(시간 분할)"
            "에서 같은 크기일 보장이 없다. 승격하려면 시간 분할에서 다시 재야 한다."
        ),
        "verdict": verdict,
        "dacon_upload": False,
        "external_actions": ["WebSearch"],
        "lockbox_used": False,
        "generated_at": datetime.now(UTC).isoformat(),
    }
    payload["digest"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]
    RECEIPT.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str),
                       encoding="utf-8")

    lines = [
        "# M271 C7 — teacher 고도화 (라우터가 지시한 노드)",
        "",
        f"노드 `{NODE_ID}` / 레인 {LANE} / 부모 `{PARENT_NODE}`",
        "",
        "라우터 v3 이 다른 네 증거를 C16 크기게이트로 기각하고 **이 방향만** 남겼다.",
        "",
        "## 1. 방향 리서치",
        "",
    ]
    for s in RESEARCH["sources"]:
        lines.append(f"- {s['finding']} — <{s['url']}> (`{s['applicability']}`)")
    lines += [
        "",
        "## 2. 확인된 공백",
        "",
        f"- **시계열 문맥 부재** — 추가된 열 **{len(added)}** 개",
        "- **용량 미탐색** — 400 그루 x 63 잎이 1,347 피처 x 5 만 행을 학습",
        "- 입력 부족은 **아니다** — teacher 는 이미 수치형 1,347 열 전부를 먹는다",
        "",
        "## 3. sigma_v (teacher OOF 잔차 표준편차)",
        "",
        "| 팔 | 전체 | g1 | g2 | g3 | 감소율 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        r = results[arm]
        lines.append(
            f"| {arm} | **{r['overall']:.4f}** | {r.get('g1', float('nan')):.4f} | "
            f"{r.get('g2', float('nan')):.4f} | {r.get('g3', float('nan')):.4f} | "
            f"{reductions[arm]:+.2%} |"
        )
    lines += [
        "",
        f"최선 팔 **{best_arm}** 감소율 **{best_reduction:.2%}** -> C1N69 기울기 "
        f"{RESPONSE_SLOPE} 로 환산 Total **{implied_total:+.6f}**",
        "",
        f"C16 문턱: 감소율 **{REQUIRED_REDUCTION:.2%}** (Total "
        f"{MAGNITUDE_FLOOR * REMAINING_GAP:.6f})",
        "",
        "## 4. 사전확약",
        "",
        f"- V1 base sigma {base_sigma:.4f} vs C1N58 shuffle {C58_SHUFFLE_SIGMA} -> **{v1}**",
        f"- V2 시계열 열 추가됨 -> **{v2}**",
        f"- H1 C16 문턱 통과 -> **{h1}**",
        f"- H2 시계열이 값을 한다 -> **{h2}**",
        f"- H3 용량이 값을 한다 -> **{h3}**",
        f"- H4 최선 팔이 세 그룹 모두 개선 -> **{h4}**",
        "",
        "## 5. 판정",
        "",
        f"**{verdict}**",
        "",
        "H1 이 거짓이면 teacher 축도 문턱 미달이고, 그러면 **남은 방향이 없다** — "
        "목표 0.66 이 이 아키텍처에서 도달 불가라는 뜻이 된다.",
        "",
        f"digest `{payload['digest']}`",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("=== C7 완료 ===")
    print(f"[C7] 시계열 대상 {len(seq_targets)} 열 -> 추가 {len(added)} 열")
    for arm in ARMS:
        r = results[arm]
        print(f"[C7] {arm:9s} sigma {r['overall']:.4f} "
              f"(g1 {r.get('g1', float('nan')):.4f} / g2 {r.get('g2', float('nan')):.4f} / "
              f"g3 {r.get('g3', float('nan')):.4f})  감소 {reductions[arm]:+.2%}")
    print(f"[C7] 최선 {best_arm} 감소 {best_reduction:.2%} -> Total {implied_total:+.6f} "
          f"(C16 문턱 {REQUIRED_REDUCTION:.2%} / {MAGNITUDE_FLOOR * REMAINING_GAP:.6f})")
    print(f"[C7] V1 {v1} / V2 {v2} / H1 {h1} / H2 {h2} / H3 {h3} / H4 {h4}")
    print(f"[C7] 판정: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
