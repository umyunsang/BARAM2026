"""M271 P4 사이클 59 — 완전예보 상한(Perfect Prog): 남은 격차가 풍속예보 오차인가.

사이클 53~57 이 병목 후보를 하나씩 닫았다.

    NWP → 풍속        아님 (C54: teacher 가 IDW 대비 24.7~29.8% 감소, 요구 13.3%)
    풍속 → 출력 산포   아님 (C57: 천장 0.7488 > 요구 0.4459)
    기저함수          아님 (C56: -0.000039)

그런데 이 셋은 모두 **상대적 여유**를 쟀지 우리 오차가 어디서 오는지를 재지 않았다.
남은 질문은 회계다.

    err = ŷ(NWP) - actual
        = [ŷ(NWP) - ŷ(관측풍속)]   ← 풍속예보 오차
        + [ŷ(관측풍속) - E(actual|풍속)] ← 사상 오차
        + [E(actual|풍속) - actual]     ← 불가피 산포  (C57 이 0.7488 로 잼)

첫 항의 크기를 재는 표준 방법이 **Perfect Prog** 다. Klein·Lewis·Enger(1959)가 도입하고
Glahn & Lowry(1972)가 MOS 와 대비해 정식화했다 — 통계 모형을 **관측 예측인자**로 학습·평가해
NWP 가 완벽할 때 도달 가능한 스킬의 상한을 얻는다. 이 프로젝트는 Glahn & Lowry 를 이미
MOS 근거로 인용했다. 새 방법이 아니라 같은 논문의 반대쪽 팔이다.

**① 방법 리서치 — 이 진단의 알려진 오염원과 그 처리**

  (가) 나셀 풍속계는 로터 **뒤**에 있다. A5 가 이미 기록했다("실제 유입풍속이 아니고").
       IEC 61400-12-1 이 나셀 풍속계 단독 사용을 인정하지 않고 나셀전달함수(NTF)를
       요구하는 이유가 이것이다. 로터 유발 유동왜곡 때문에 나셀 신호는 **운전상태를
       역으로 실어 나른다.** actual_kwh 와 상관 0.9266 은 순수 풍속 상관으로 보기에 높다.
       ⇒ ORACLE 은 "완벽한 풍속 지식"의 상한을 **위에서** 다시 한 번 넘는 값이다.

  (나) scada_ws 결측 자체가 정보다. 결측은 SCADA 부재·정지와 상관될 수 있고, 그러면
       결측표시만으로도 가용성이 새어 들어온다.
       ⇒ MASKONLY 팔로 그 경로를 **분리 측정**한다. 풍속 지식의 값 = ORACLE - MASKONLY.

  이 두 오염이 같은 방향(상향)이므로 판정을 **비대칭**으로 설계한다.

      이득 <  격차  →  풍속 축만으로는 격차를 못 닫는다.  **강한 결론** (상한이 부족하므로)
      이득 >= 격차  →  방향만 확인. 크기는 미확정.       **약한 결론**

  상한이 미달하는 쪽에서만 결론이 강하다. 이것이 이 설계가 답할 수 있는 질문의 전부다.

**② 사양 동결**

  하네스   사이클 56 과 동일 (teacher 복원, generic 기저, leaves 15, lr 0.1, 200 rounds,
           fold-외 온도선택, Bayes 결정). C56 의 generic 팔이 C44 대조군을 0.000000 로
           재현했으므로 이 경로는 검증돼 있다.
  팔 셋    CONTROL   기존 101 피처
           MASKONLY  + `oracle__ws_missing` (결측표시만, 풍속값 없음)
           ORACLE    + 결측표시 + scada_ws 및 그 변환(제곱·세제곱·generic 파워커브)
  격차     0.66 - 0.630310 = **0.029690**  (목표 하한 - M115@T0.6_G0.2 로컬)

  **타당성 가드**
    V1  CONTROL 이 C44 대조군 0.604043 의 ±0.0005 이내. (C56 이 0.000000 를 냈으므로
        C56 보다 10배 조인다. 벗어나면 하네스가 바뀐 것이고 나머지 판정을 버린다.)
    V2  MASKONLY - CONTROL < 0.005. 결측 경로가 작아야 ORACLE 해석이 성립한다.
        초과하면 ORACLE 은 풍속이 아니라 가용성 누출을 재고 있는 것이다.

  사전확약 (V1·V2 통과시에만 판정):
    H1  ORACLE > MASKONLY.                    부호 예상: 양, 큼
    H2  (ORACLE - MASKONLY) >= 0.029690.      ← 판정의 핵심
    H3  이득이 FICR 쪽에서 우세하게 나온다.
    H4  이득이 그룹3 에서 가장 크다. (C57 이 g3 고출력대 unit/4 0.364 로 최악을 쟀다.
        g3 열세가 풍속예보 탓이면 오라클이 g3 을 가장 많이 올려야 한다.)

  **표면간 가법성 경고**: 이득은 C56 표면(대조군 0.604)에서 재고, 격차는 챔피언
  표면(0.630310)에서 정의된다. H2 는 두 표면 사이 **가법성을 가정**한다. C33·C45 에서
  기준선 혼동으로 두 번 틀렸으므로 이 가정을 판정문에 명시하고, H2 가 아슬아슬하면
  (|이득 - 격차| < 0.005) 결론을 유보한다.

**진단 전용.** scada_ws 는 2025 평가기간에 없다(C39 가 확정, 그 팔은 철회됨). 이 노드는
후보가 될 수 없고 게이트 승격 대상이 아니다. 제출·lockbox·외부데이터 미사용.
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

from m271_cycle37_band_loss import KEYS, PROBE, fold_rows
from m271_cycle40_band_classifier import (
    CLASS_WIDTH,
    N_CLASS,
    PARAMS,
    ROUNDS,
    bayes_decision,
    make_objective,
    one_hot_targets,
)
from m271_cycle42_teacher_restored import all_weather_columns, teach
from m271_cycle44_sharpened_decision import TEMPERATURES, sharpen
from m271_cycle56_measured_powercurve import (
    add_sitewind_with_basis,
    generic_curve,
    measured_curves,
)
from m271_evaluate_candidate import official
from run_sequence_classifier import _surface

from baram.constants import CAPACITIES_KWH

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
REPORT_MD = REPORTS / "m271_cycle59_perfect_prog_bound.md"
RECEIPT = REPORTS / "m271_cycle59_perfect_prog_bound_receipt.json"

NODE_ID = "C1N59_PERFECT_PROG_BOUND"
LANE = "L6"
PARENT_NODE = "C1N57_FICR_CEILING"
DIAGNOSTIC_ONLY = True

C44_CONTROL = 0.604043
V1_TOLERANCE = 0.0005
V2_LIMIT = 0.005
CHAMPION_LOCAL = 0.630310
TARGET_FLOOR = 0.66
GAP = TARGET_FLOOR - CHAMPION_LOCAL
UNDECIDED_BAND = 0.005
ARMS = ("control", "maskonly", "oracle")


def add_oracle(matrix: pd.DataFrame, arm: str) -> list[str]:
    """관측 나셀풍속을 팔에 맞게 주입한다. CONTROL 은 아무것도 넣지 않는다."""
    if arm == "control":
        return []
    speed = matrix["scada_ws"].to_numpy(dtype="float64")
    missing = np.isnan(speed)
    matrix["oracle__ws_missing"] = missing.astype("float64")
    if arm == "maskonly":
        return ["oracle__ws_missing"]
    matrix["oracle__ws"] = speed
    matrix["oracle__ws2"] = speed**2
    matrix["oracle__ws3"] = speed**3
    matrix["oracle__ws_powercurve"] = np.where(missing, np.nan, generic_curve(speed))
    return [name for name in matrix if name.startswith("oracle__")]


def main() -> int:
    curves = measured_curves()

    surface, _base, auxiliary = _surface()
    surface["forecast_kst_dtm"] = pd.to_datetime(surface["forecast_kst_dtm"])
    surface["capacity"] = surface["group_id"].map(CAPACITIES_KWH).astype(float)
    surface["rate"] = surface["actual_kwh"] / surface["capacity"]
    surface = surface.loc[surface["rate"].notna()].reset_index(drop=True)

    assert "scada_ws" in surface.columns, "scada_ws 가 표면에 없다 — 진단 불가"
    coverage = float(surface["scada_ws"].notna().mean())

    wanted = json.loads(
        (PROBE / "M115_XGBOOST-dev-2023-Q3.json").read_text(encoding="utf-8")
    )["selected_feature_names"]
    base_features = [c for c in wanted if c in surface.columns and c != "scada_ws"]
    aux_cols = [c for c in auxiliary if c in surface.columns and c != "scada_ws"]
    aw_cols = all_weather_columns(surface)

    store: dict[str, dict[str, Any]] = {}
    fits = 0
    for probe_fold, meta in fold_rows().items():
        train = surface.loc[surface["forecast_kst_dtm"] < meta["start"]].copy()
        test = surface.loc[
            np.array(
                [
                    (fid, gid) in meta["keys"]
                    for fid, gid in zip(surface["forecast_id"], surface["group_id"],
                                        strict=True)
                ]
            )
        ].copy()
        legacy_tr, legacy_te = teach(train, test, aux_cols)
        aw_tr, aw_te = teach(train, test, aw_cols)

        rate = np.clip(train["rate"].to_numpy(dtype="float64"), 0.0, None)
        label = np.clip((rate / CLASS_WIDTH).astype(int), 0, N_CLASS - 1)
        target = one_hot_targets(rate)
        entry: dict[str, Any] = {
            "meta": test.loc[:, [*KEYS, "actual_kwh"]].copy(),
            "capacity": test["capacity"].to_numpy(dtype="float64"),
            "group": test["group_id"].to_numpy(),
        }
        for arm in ARMS:
            tr = train.copy()
            te = test.copy()
            names = add_sitewind_with_basis(tr, legacy_tr, aw_tr, "generic", curves)
            add_sitewind_with_basis(te, legacy_te, aw_te, "generic", curves)
            oracle_names = add_oracle(tr, arm)
            add_oracle(te, arm)
            features = [*base_features, *names, *oracle_names]
            dataset = lgb.Dataset(
                tr.loc[:, features].astype("float32"), label=label, free_raw_data=False
            )
            params = dict(PARAMS)
            params["objective"] = make_objective(target)
            booster = lgb.train(params, dataset, num_boost_round=ROUNDS)
            fits += 1
            raw = np.asarray(
                booster.predict(te.loc[:, features].astype("float32"))
            ).reshape(len(te), N_CLASS)
            exp = np.exp(raw - raw.max(axis=1, keepdims=True))
            entry[arm] = exp / exp.sum(axis=1, keepdims=True)
        store[probe_fold] = entry

    def scored(fold: str, arm: str, temperature: float) -> pd.DataFrame:
        e = store[fold]
        out = e["meta"].copy()
        out["prediction_kwh"] = (
            bayes_decision(sharpen(e[arm], temperature)) * e["capacity"]
        )
        out["group_id"] = e["group"]
        out["month"] = out["forecast_kst_dtm"].dt.to_period("M").astype(str)
        return out

    chosen_t: dict[str, dict[str, float]] = {}
    pieces: dict[str, list[pd.DataFrame]] = {arm: [] for arm in ARMS}
    for arm in ARMS:
        chosen_t[arm] = {}
        for held in store:
            others = [f for f in store if f != held]
            best_t, best_score = None, -np.inf
            for temperature in TEMPERATURES:
                frame = pd.concat(
                    [scored(f, arm, temperature) for f in others], ignore_index=True
                )
                total = official(frame)["total"]
                if total > best_score:
                    best_t, best_score = temperature, total
            chosen_t[arm][held] = float(best_t)
            pieces[arm].append(scored(held, arm, float(best_t)))

    frames = {arm: pd.concat(parts, ignore_index=True) for arm, parts in pieces.items()}
    results = {arm: official(frames[arm]) for arm in ARMS}

    v1_gap = abs(results["control"]["total"] - C44_CONTROL)
    v1 = bool(v1_gap <= V1_TOLERANCE)
    mask_channel = results["maskonly"]["total"] - results["control"]["total"]
    v2 = bool(mask_channel < V2_LIMIT)

    wind_value = results["oracle"]["total"] - results["maskonly"]["total"]
    h1 = bool(wind_value > 0.0)
    h2 = bool(wind_value >= GAP)
    ficr_contrib = 0.5 * (results["oracle"]["ficr"] - results["maskonly"]["ficr"])
    nmae_contrib = 0.5 * (
        results["oracle"]["one_minus_nmae"] - results["maskonly"]["one_minus_nmae"]
    )
    h3 = bool(ficr_contrib > nmae_contrib)

    # 그룹별 Total 은 공식 산식의 그룹 성분에서 직접 만든다. `official()` 은 세 그룹이
    # 모두 있어야 하므로 부분집합을 다시 채점할 수 없다(공식 구현이 그렇게 막아 뒀다).
    per_group: dict[str, dict[str, float]] = {}
    for gid in (1, 2, 3):
        row = {}
        for arm in ("maskonly", "oracle"):
            r = results[arm]
            row[arm] = 0.5 * (1.0 - r["group_nmae"][gid]) + 0.5 * r["group_ficr"][gid]
        row["delta"] = row["oracle"] - row["maskonly"]
        per_group[str(gid)] = row
    best_group = max(per_group, key=lambda g: per_group[g]["delta"])
    h4 = bool(best_group == "3")

    undecided = bool(abs(wind_value - GAP) < UNDECIDED_BAND)
    if not v1:
        verdict = "HARNESS_DRIFT_RESULT_VOID"
    elif not v2:
        verdict = "MISSINGNESS_CHANNEL_CONTAMINATES_BOUND"
    elif undecided:
        verdict = "BOUND_WITHIN_ADDITIVITY_NOISE_UNDECIDED"
    elif not h2:
        verdict = "WIND_AXIS_INSUFFICIENT_TO_CLOSE_GAP"
    else:
        verdict = "WIND_AXIS_DIRECTION_CONFIRMED_MAGNITUDE_OPEN"

    payload: dict[str, Any] = {
        "node_id": NODE_ID,
        "lane": LANE,
        "parent": PARENT_NODE,
        "diagnostic_only": DIAGNOSTIC_ONLY,
        "gate_version": None,
        "method": "PERFECT_PROG (Klein/Lewis/Enger 1959; Glahn & Lowry 1972)",
        "scada_coverage": coverage,
        "model_fits": fits,
        "chosen_temperature": chosen_t,
        "arms": {arm: results[arm] for arm in ARMS},
        "checks": {
            "V1_control_reproduces_c44": v1,
            "V1_gap": float(v1_gap),
            "V2_missingness_channel_small": v2,
            "V2_measured": float(mask_channel),
        },
        "wind_value": float(wind_value),
        "gap_to_target": float(GAP),
        "hypotheses": {
            "H1_oracle_beats_maskonly": h1,
            "H2_wind_value_spans_gap": h2,
            "H3_ficr_dominant": h3,
            "H4_group3_largest": h4,
        },
        "contributions": {"ficr": float(ficr_contrib), "nmae": float(nmae_contrib)},
        "per_group": per_group,
        "best_group": best_group,
        "additivity_assumed": True,
        "undecided_band": undecided,
        "verdict": verdict,
        "dacon_upload": False,
        "external_actions": [],
        "lockbox_used": False,
        "generated_at": datetime.now(UTC).isoformat(),
    }
    payload["digest"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]
    RECEIPT.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    lines = [
        f"# M271 P4 사이클 59 — 완전예보 상한 (Perfect Prog)",
        "",
        f"노드 `{NODE_ID}` / 레인 {LANE} / 부모 `{PARENT_NODE}` / **진단 전용**",
        "",
        "## 1. 팔별 점수",
        "",
        "| 팔 | Total | 1-NMAE | FICR |",
        "|---|---:|---:|---:|",
    ]
    for arm in ARMS:
        r = results[arm]
        lines.append(
            f"| {arm} | {r['total']:.6f} | {r['one_minus_nmae']:.6f} | {r['ficr']:.6f} |"
        )
    lines += [
        "",
        f"SCADA 커버리지 {coverage:.3f} / 적합 {fits} 회",
        "",
        "## 2. 타당성 가드",
        "",
        f"- V1 CONTROL {results['control']['total']:.6f} vs C44 {C44_CONTROL} "
        f"(차 {v1_gap:.6f}, 허용 {V1_TOLERANCE}) -> **{v1}**",
        f"- V2 결측 경로 {mask_channel:+.6f} (한계 {V2_LIMIT}) -> **{v2}**",
        "",
        "## 3. 사전확약",
        "",
        f"- H1 ORACLE > MASKONLY -> **{h1}**",
        f"- H2 풍속 지식의 값 {wind_value:+.6f} >= 격차 {GAP:.6f} -> **{h2}**",
        f"- H3 FICR 우세 (FICR {ficr_contrib:+.6f} / 1-NMAE {nmae_contrib:+.6f}) -> **{h3}**",
        f"- H4 그룹3 최대 (최대 그룹 {best_group}) -> **{h4}**",
        "",
        "## 4. 그룹별",
        "",
        "| 그룹 | MASKONLY | ORACLE | 차 |",
        "|---:|---:|---:|---:|",
    ]
    for gid, row in per_group.items():
        lines.append(
            f"| {gid} | {row['maskonly']:.6f} | {row['oracle']:.6f} | {row['delta']:+.6f} |"
        )
    lines += [
        "",
        "## 5. 판정",
        "",
        f"**{verdict}**",
        "",
        "이 상한은 두 경로로 **위쪽으로 편향**돼 있다. (가) 나셀 풍속계는 로터 뒤에 있어",
        "운전상태를 역으로 싣는다(IEC 61400-12-1 이 NTF 없이 인정하지 않는 이유). (나) 결측",
        "표시가 가용성을 실어 나를 수 있다 — V2 가 그 크기를 잰다. 따라서 **미달일 때만**",
        "결론이 강하다. 초과는 방향 확인에 그친다.",
        "",
        f"표면간 가법성 가정: 이득은 C56 표면(대조군 {C44_CONTROL})에서, 격차는 챔피언 "
        f"표면({CHAMPION_LOCAL})에서 정의된다. 미결정대 ±{UNDECIDED_BAND}.",
        "",
        "`scada_ws` 는 2025 평가기간에 없다(C39). 이 노드는 후보가 될 수 없다.",
        "",
        f"digest `{payload['digest']}`",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("=== 완료 ===")
    for arm in ARMS:
        print(f"[C59] {arm:9s} {results[arm]['total']:.6f} "
              f"(1-NMAE {results[arm]['one_minus_nmae']:.6f} / FICR {results[arm]['ficr']:.6f})")
    print(f"[C59] V1 {v1} (차 {v1_gap:.6f}) / V2 {v2} (결측 경로 {mask_channel:+.6f})")
    print(f"[C59] 풍속 지식의 값 {wind_value:+.6f} vs 격차 {GAP:.6f} -> H2 {h2}")
    print(f"[C59] 그룹별 차 " + " / ".join(
        f"g{g}={row['delta']:+.6f}" for g, row in per_group.items()))
    print(f"[C59] 판정: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
