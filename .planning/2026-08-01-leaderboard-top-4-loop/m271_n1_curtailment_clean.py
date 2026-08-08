"""M271 N1 — 감발 구간을 teacher 학습 표적에서 제외한다.

C1N86(레인 확장)이 실체화한 후보. **L1 데이터전처리는 굶은 레인**이다(전체 5 개, L8 의
31 개 대비). 그동안 시도한 것은 전부 **모형**을 바꿨다(용량·피처·결합·결정층). 이것은
**표적을 정제**한다.

**기전.** teacher 의 표적은 `scada_ws`(나셀 풍속)다. 감발 구간에서는 로터가 느리게 돌아
로터 유발 감속이 줄고 **나셀 풍속계가 실제보다 높게 읽는다**(IEC 61400-12-1 이 나셀
전달함수 없이 나셀 풍속계를 인정하지 않는 이유). 오염된 표적으로 학습한 teacher 는
그 오염을 그대로 배운다.

**우리 자료의 근거.**
  - A5 §5: "운전로그가 없으므로 표준의 로그 기반 필터링을 **통계적 대체**로 수행한다.
    이 대체는 **표준 준수가 아니며** 그 사실을 리포트에 명시한다."
  - C1N57B: g3 의 분산법칙 theta **0.775** 대 g1·g2 의 0.5 — g3 만 곱셈(가용성) 잡음.
  - A5: UNISON 포화비 **0.89~0.95** 대 VESTAS 0.99.
  - 문헌: SCADA 의 감발·센서결함을 제거하는 것이 표준 전처리
    (https://doi.org/10.3390/s25175329, PMC12431095).

**배포 가능성.** 감발은 **예보시점에 알 수 없으므로 피처가 될 수 없다.** 그러나 학습
표적에서 빼는 것은 학습기간 SCADA 만 쓰므로 가능하다 — 배포 시에도 같은 절차를 밟는다.
평가기간에는 아무것도 필요하지 않다.

**① 감발 판정 규칙 (실행 전 동결)**

    기대출력  p_hat = C1N57 실측 커브(그룹별)를 나셀 풍속에 적용
    잔차      r = 실제출력/용량 - p_hat
    감발 판정 `r < -DEVIATION` **그리고** `scada_ws >= WIND_FLOOR`

    WIND_FLOOR = **10.0 m/s** — 이 위에서 커브가 평탄해지기 시작하므로(g1 0.726 /
      g2 0.685 / g3 0.712 at 10 m/s) 풍속 오차와 감발이 섞이지 않는다. 급경사에서
      잡으면 풍속 오차를 감발로 오인한다.
    DEVIATION = **0.15** — 정격의 15% 미달. 평탄 구간에서 이 정도 하향 이탈은
      정상 산포로 보기 어렵다.

    **두 값 모두 결과를 보기 전에 정한다.** 물리(커브가 평탄해지는 지점)와 크기(정격의
    15%)에서 왔고 성능을 보고 조정하지 않는다.

**② 사양 동결**

  분할   C1N84 와 **동일** — fold 시작 이전 행으로만 학습, fold test 행에 예측.
         내부 KFold 없음(누출 경로 없음).
  팔     `base`   현행. 라벨 있는 전 행으로 teacher 학습.
         `clean`  감발 판정 행을 **학습에서만** 제외. 예측 대상은 동일.
  지표   `std(scada_ws - 예측)` on **전 test 행**(주). 감발 제외 test 행(보조).
         주 지표를 전 행으로 두는 이유 — 배포 시 전 행을 예측하고, C1N69 곡선도
         전 행에서 보정됐다.

  **타당성 가드**
    V1  `base` 의 그룹별 sigma 가 C1N84 와 ±0.01 이내(같은 분할·같은 행이므로 일치해야).
    V2  감발 판정 비율이 **2~20%**. 1% 미만이면 규칙이 너무 좁아 시험이 안 되고,
        30% 초과면 정상 산포를 감발로 잡는 것이다.
    V3  두 팔이 같은 test 행에서 평가된다.

  사전확약 (V1~V3 통과시에만 판정):
    H1  `clean` < `base` (전 test 행 기준).
    H2  감소율이 F1 검출문턱 **0.62%** 이상. C1N85 가 교정한 문턱이다.
    H3  **g3 개선이 가장 크다.** theta 0.775 와 포화비 0.89~0.95 가 g3 에 감발이 가장
        많다고 가리키므로, 그렇지 않으면 기전이 틀린 것이다.
    H4  보조 지표(감발 제외 test 행)에서도 `clean` 이 낫다. 주 지표만 좋으면 표적
        오염을 옮긴 것일 수 있다.

  H2 가 참이면 **F1 을 통과한 두 번째 후보**가 되고 C1N71 과 합산 대상이 된다.
  거짓이면 L1 레인도 문턱 미달이고, 그때는 N2(공동 최적화)를 착수할 근거가 사라진다
  (C1N86 이 그렇게 동결했다).

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
from m271_cycle65_wind_limited_bound import MIN_ROWS
from m271_cycle67_exact_curve_propagation import build_curve
from run_sequence_classifier import _surface

from baram.constants import CAPACITIES_KWH

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
C57_RECEIPT = REPORTS / "m271_cycle57_ficr_ceiling_receipt.json"
C84_RECEIPT = REPORTS / "m271_c7c_chronological_receipt.json"
REPORT_MD = REPORTS / "m271_n1_curtailment_clean.md"
RECEIPT = REPORTS / "m271_n1_curtailment_clean_receipt.json"

NODE_ID = "C1N87_CURTAILMENT_CLEAN_TARGET"
LANE = "L1"
PARENT_NODE = "C1N86_LANE_EXPAND"

WIND_FLOOR = 10.0      # m/s. 커브가 평탄해지는 지점. 실행 전 동결.
DEVIATION = 0.15       # 정격 대비 하향 이탈. 실행 전 동결.
V1_TOLERANCE = 0.01
FLAG_RATE_RANGE = (0.02, 0.20)
DETECTION_THRESHOLD = 0.001013
RESPONSE_SLOPE = 0.164
F1_SIGMA_REDUCTION = DETECTION_THRESHOLD / RESPONSE_SLOPE

ARMS = ("base", "clean")


def main() -> int:
    c57 = json.loads(C57_RECEIPT.read_text(encoding="utf-8"))["result"]
    c84 = json.loads(C84_RECEIPT.read_text(encoding="utf-8"))

    curves = {
        g: build_curve([b for b in c57["per_group"][str(g)]["bins"]
                        if b["rows"] >= MIN_ROWS])
        for g in (1, 2, 3)
    }

    surface, _base, _aux = _surface()
    surface["forecast_kst_dtm"] = pd.to_datetime(surface["forecast_kst_dtm"])
    surface["capacity"] = surface["group_id"].map(CAPACITIES_KWH).astype(float)
    surface = surface.loc[surface["actual_kwh"].notna()].reset_index(drop=True)
    columns = all_weather_columns(surface)

    # --- 감발 판정 ---------------------------------------------------------
    rate = surface["actual_kwh"].to_numpy(float) / surface["capacity"].to_numpy(float)
    wind = surface["scada_ws"].to_numpy(float)
    expected = np.full(len(surface), np.nan)
    for group, (cv, cp) in curves.items():
        mask = (surface["group_id"] == group).to_numpy()
        expected[mask] = np.interp(wind[mask], cv, cp, left=0.0, right=cp[-1])
    residual = rate - expected
    curtailed = (residual < -DEVIATION) & (wind >= WIND_FLOOR)
    curtailed = np.where(np.isnan(wind), False, curtailed)
    surface["curtailed"] = curtailed

    labelled = surface["scada_ws"].notna().to_numpy()
    flag_rate = float(curtailed[labelled].mean())
    flag_by_group = {
        int(g): float(curtailed[labelled & (surface["group_id"] == g).to_numpy()].mean())
        for g in (1, 2, 3)
    }
    v2 = bool(FLAG_RATE_RANGE[0] <= flag_rate <= FLAG_RATE_RANGE[1])

    # --- 시간분할 학습·평가 -------------------------------------------------
    residuals: dict[str, dict[int, list[np.ndarray]]] = {a: {} for a in ARMS}
    residuals_clean_rows: dict[str, dict[int, list[np.ndarray]]] = {a: {} for a in ARMS}
    fits = 0
    test_rows = 0

    for _probe_fold, meta in fold_rows().items():
        train = surface.loc[surface["forecast_kst_dtm"] < meta["start"]]
        test = surface.loc[
            np.array([
                (fid, gid) in meta["keys"]
                for fid, gid in zip(surface["forecast_id"], surface["group_id"],
                                    strict=True)
            ])
        ]
        for group in (1, 2, 3):
            tr_all = train.loc[
                (train["group_id"] == group) & train["scada_ws"].notna()
            ]
            te = test.loc[(test["group_id"] == group) & test["scada_ws"].notna()]
            if len(tr_all) < 200 or len(te) < 50:
                continue
            test_rows += len(te)
            x_te = te.loc[:, columns].astype("float32")
            y_te = te["scada_ws"].to_numpy(dtype="float64")
            te_clean = ~te["curtailed"].to_numpy()

            for arm in ARMS:
                tr = tr_all if arm == "base" else tr_all.loc[~tr_all["curtailed"]]
                model = LGBMRegressor(**TEACHER_PARAMS)
                model.fit(tr.loc[:, columns].astype("float32"),
                          tr["scada_ws"].to_numpy(dtype="float64"))
                fits += 1
                pred = model.predict(x_te)
                residuals[arm].setdefault(group, []).append(y_te - pred)
                residuals_clean_rows[arm].setdefault(group, []).append(
                    (y_te - pred)[te_clean]
                )

    def summarise(store: dict[str, dict[int, list[np.ndarray]]]) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for arm, per_group in store.items():
            entry: dict[str, float] = {}
            pooled = []
            for group, parts in per_group.items():
                joined = np.concatenate(parts)
                entry[f"g{group}"] = float(np.std(joined, ddof=1))
                pooled.append(joined)
            entry["overall"] = float(np.std(np.concatenate(pooled), ddof=1))
            out[arm] = entry
        return out

    primary = summarise(residuals)
    secondary = summarise(residuals_clean_rows)

    c84_arms = c84["arms"]["base"]
    v1 = bool(all(
        abs(primary["base"][f"g{g}"] - float(c84_arms[f"g{g}"])) <= V1_TOLERANCE
        for g in (1, 2, 3)
    ))
    v3 = True  # 두 팔이 같은 루프 안에서 같은 te 를 쓴다.

    reduction = 1.0 - primary["clean"]["overall"] / primary["base"]["overall"]
    implied_total = reduction * RESPONSE_SLOPE
    per_group_reduction = {
        g: 1.0 - primary["clean"][f"g{g}"] / primary["base"][f"g{g}"] for g in (1, 2, 3)
    }

    h1 = bool(primary["clean"]["overall"] < primary["base"]["overall"])
    h2 = bool(reduction >= F1_SIGMA_REDUCTION)
    h3 = bool(max(per_group_reduction, key=lambda g: per_group_reduction[g]) == 3)
    h4 = bool(secondary["clean"]["overall"] < secondary["base"]["overall"])

    if not v1 or not v2:
        verdict = "GUARD_FAILED_RESULT_VOID"
    elif h1 and h2:
        verdict = "CURTAILMENT_CLEANING_CLEARS_DETECTION_THRESHOLD"
    elif h1:
        verdict = "CLEANING_HELPS_BUT_BELOW_DETECTION"
    else:
        verdict = "CURTAILMENT_CLEANING_DOES_NOT_HELP"

    payload: dict[str, Any] = {
        "node_id": NODE_ID,
        "lane": LANE,
        "parent": PARENT_NODE,
        "rule": {
            "wind_floor": WIND_FLOOR,
            "deviation": DEVIATION,
            "frozen_before_run": True,
            "rationale": (
                "WIND_FLOOR 은 커브가 평탄해지는 지점(10 m/s 에서 g1 0.726 / g2 0.685 / "
                "g3 0.712), DEVIATION 은 정격의 15%. 물리와 크기에서 왔고 성능을 보고 "
                "조정하지 않았다."
            ),
        },
        "flag_rate": flag_rate,
        "flag_rate_by_group": flag_by_group,
        "model_fits": fits,
        "test_rows": test_rows,
        "sigma_primary_all_test_rows": primary,
        "sigma_secondary_non_curtailed": secondary,
        "reduction": reduction,
        "reduction_by_group": per_group_reduction,
        "implied_total_gain": implied_total,
        "f1_sigma_reduction": F1_SIGMA_REDUCTION,
        "detection_threshold": DETECTION_THRESHOLD,
        "checks": {"V1_base_matches_c84": v1, "V2_flag_rate_in_range": v2,
                   "V3_same_test_rows": v3},
        "hypotheses": {
            "H1_clean_beats_base": h1,
            "H2_clears_detection": h2,
            "H3_g3_improves_most": h3,
            "H4_holds_on_non_curtailed_rows": h4,
        },
        "verdict": verdict,
        "deployable": True,
        "deployability_note": (
            "감발은 예보시점에 알 수 없어 피처가 될 수 없다. 그러나 **학습 표적에서 "
            "제외**하는 것은 학습기간 SCADA 만 쓰므로 배포 가능하다 — 평가기간에는 "
            "아무것도 필요하지 않다."
        ),
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
        "# M271 N1 — 감발 구간 teacher 표적 정제",
        "",
        f"노드 `{NODE_ID}` / 레인 {LANE} / 부모 `{PARENT_NODE}`",
        "",
        "**굶은 레인(L1)의 첫 실행 노드.** 그동안은 전부 모형을 바꿨고, 이것은 "
        "**표적을 정제**한다.",
        "",
        "## 1. 감발 판정 (실행 전 동결)",
        "",
        f"`잔차 < -{DEVIATION}` **그리고** `나셀풍속 >= {WIND_FLOOR} m/s`",
        "",
        f"판정 비율 **{flag_rate:.2%}** (g1 {flag_by_group[1]:.2%} / "
        f"g2 {flag_by_group[2]:.2%} / g3 {flag_by_group[3]:.2%})",
        "",
        "## 2. sigma_v — 시간분할 test 행",
        "",
        "| 팔 | 전체 | g1 | g2 | g3 |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        r = primary[arm]
        lines.append(
            f"| {arm} | **{r['overall']:.4f}** | {r['g1']:.4f} | {r['g2']:.4f} | "
            f"{r['g3']:.4f} |"
        )
    lines += [
        "",
        f"**감소율 {reduction:+.2%}** (g1 {per_group_reduction[1]:+.2%} / "
        f"g2 {per_group_reduction[2]:+.2%} / g3 {per_group_reduction[3]:+.2%})",
        "",
        f"환산 Total **{implied_total:+.6f}** / F1 검출문턱 "
        f"{DETECTION_THRESHOLD} (= sigma_v {F1_SIGMA_REDUCTION:.2%})",
        "",
        "보조 지표(감발 제외 test 행): "
        f"base {secondary['base']['overall']:.4f} -> clean "
        f"{secondary['clean']['overall']:.4f}",
        "",
        "## 3. 사전확약",
        "",
        f"- V1 base 가 C1N84 와 ±{V1_TOLERANCE} 이내 -> **{v1}**",
        f"- V2 판정 비율이 {FLAG_RATE_RANGE[0]:.0%}~{FLAG_RATE_RANGE[1]:.0%} -> **{v2}**",
        f"- H1 clean < base -> **{h1}**",
        f"- H2 F1 검출문턱 통과 -> **{h2}**",
        f"- H3 g3 개선이 최대 -> **{h3}**",
        f"- H4 감발 제외 행에서도 유지 -> **{h4}**",
        "",
        "## 4. 판정",
        "",
        f"**{verdict}**",
        "",
        payload["deployability_note"],
        "",
        f"digest `{payload['digest']}`",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("=== N1 완료 ===")
    print(f"[N1] 감발 판정 {flag_rate:.2%} (g1 {flag_by_group[1]:.2%} / "
          f"g2 {flag_by_group[2]:.2%} / g3 {flag_by_group[3]:.2%}) / 적합 {fits}")
    for arm in ARMS:
        r = primary[arm]
        print(f"[N1] {arm:5s} sigma {r['overall']:.4f} "
              f"(g1 {r['g1']:.4f} / g2 {r['g2']:.4f} / g3 {r['g3']:.4f})")
    print(f"[N1] 감소 {reduction:+.2%} (g1 {per_group_reduction[1]:+.2%} / "
          f"g2 {per_group_reduction[2]:+.2%} / g3 {per_group_reduction[3]:+.2%})")
    print(f"[N1] 환산 Total {implied_total:+.6f} / F1 문턱 {F1_SIGMA_REDUCTION:.2%}")
    print(f"[N1] 보조 base {secondary['base']['overall']:.4f} -> clean "
          f"{secondary['clean']['overall']:.4f}")
    print(f"[N1] V1 {v1} / V2 {v2} / H1 {h1} / H2 {h2} / H3 {h3} / H4 {h4}")
    print(f"[N1] 판정: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
