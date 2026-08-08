"""M271 P4 사이클 56 — 파워커브 기저함수를 손으로 쓴 램프에서 A5 실측 커브로 교체.

`sitewind__{source}_powercurve` 는 지금 `clip((v-3)/9, 0, 1)**3` 이다. 손으로 쓴 일반형
3 차 램프이고, A5 가 SCADA 208 만 행으로 잰 이 사이트의 실측 커브와 크게 어긋난다.

    v (m/s)   실측(정격비)   일반형 램프   차이
      5.25       0.122        0.016      +0.106
      6.75       0.273        0.072      +0.200
      8.25       0.491        0.198      +0.292
      9.75       0.731        0.422      +0.309
     12.75       0.930        1.000      -0.070

    전체 RMS 0.1453 / 최대 0.3158 / **급경사 구간(정격비 0.2~0.8) RMS 0.2679**

`sitewind__mean_powercurve` 는 M102 선택 피처 목록의 **2 번**이다. 모델이 가장 많이 기대는
기저함수 중 하나가 정격의 27% 만큼 어긋나 있다.

물리적으로 하나 더 고친다. A5 커브는 **밀도정규화 풍속** `v_n = v * (rho/1.225)^(1/3)`
기준인데 일반형 램프는 밀도를 아예 무시한다. 실측 커브를 쓰려면 정규화를 함께 적용하는 것이
옳은 적용이며, 그 자체가 IEC 61400-12-1 의 표준 절차다.

① 방법 리서치 (실행 전)
  - IEC 61400-12-1 — 파워커브 bin 화와 공기밀도 정규화. A5 가 이미 그 절차로 커브를 냈다.
  - 새 학습 방법 없음. 사이클 44 의 설정을 **기저함수만** 바꿔 다시 돈다. 한 변수만
    움직여야 귀속이 된다(사이클 28->30->31, 42->44 에서 쓴 원칙).
  - **집계 커브를 쓴다.** 그룹은 5~6 기를 합치므로 개별 터빈 커브가 아니라 그룹 평균
    커브가 맞는 대상이다. 사이클 55 가 잰 대로 터빈간 상관이 0.27~0.30 이라 집계 커브는
    개별 커브보다 완만하다.

② 사양 동결

  기저함수
    GENERIC   `clip((v - 3) / 9, 0, 1)**3`                       (현행)
    MEASURED  그룹별 A5 실측 커브의 단조 보간, 입력은 `v_n = v * (rho/1.225)^(1/3)`
  그 외는 사이클 44 와 **동일**: teacher 복원, 46 class one-hot 목표, leaves 15 / lr 0.1 /
  rounds 200, Bayes 결정 + **fold-외 온도 선택**, 87 + sitewind 피처, `scada_ws` 제외.

  **타당성 가드**
    V1  GENERIC 팔이 사이클 44 의 CONTROL(0.604043)을 `±0.005` 이내로 재현한다.
        벗어나면 파이프라인이 달라진 것이므로 기저함수 효과를 귀속할 수 없다.

  사전확약(V1 통과시에만 판정):
    H1  MEASURED > GENERIC (pooled Total).
    H2  MEASURED 가 GENERIC 을 부모로 한 **동결 게이트를 통과**한다.
    H3  MEASURED > `M115@T0.6_G0.2` (0.630310).
    H4  이득이 **FICR 쪽**에서 나온다.

  H1 이 기각되면 기저함수 축이 닫힌다 — GBM 이 단조 변환을 스스로 학습하므로 기저함수
  품질이 중요하지 않다는 뜻이 되고, 그 자체가 유용한 정보다.

**게이트를 수정하지 않는다. `actual_kwh` 미사용(학습 표적은 라벨이지만 2024 행 미사용).**
lockbox·외부데이터 미사용.
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

from m270_gate import GATE_VERSION, evaluate_gate
from m270_monthly_validation import load_predictions
from m271_cycle37_band_loss import KEYS, PROBE, fold_rows
from m271_cycle40_band_classifier import (
    CLASS_WIDTH,
    DEPLOYED,
    FOLDS,
    M115_FIXED_TOTAL,
    N_CLASS,
    PARAMS,
    ROUNDS,
    bayes_decision,
    by_fold_total,
    make_objective,
    one_hot_targets,
)
from m271_cycle42_teacher_restored import all_weather_columns, teach
from m271_cycle44_sharpened_decision import TEMPERATURES, sharpen
from m271_evaluate_candidate import official
from run_sequence_classifier import _surface

from baram.constants import CAPACITIES_KWH

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
A5_RECEIPT = REPORTS / "m271_n0_scada_receipt.json"
REPORT_MD = REPORTS / "m271_cycle56_measured_powercurve.md"
RECEIPT = REPORTS / "m271_cycle56_measured_powercurve_receipt.json"

NODE_ID = "C1N56_MEASURED_POWERCURVE"
LANE = "L2"
PARENT_NODE = "C1N44_SHARPENED_DECISION"
C44_CONTROL = 0.604043
V1_TOLERANCE = 0.005
RHO_REF = 1.225
GROUP_OF_TURBINE = {
    **{f"vestas_wtg{n:02d}": (1 if n <= 6 else 2) for n in range(1, 13)},
    **{f"unison_wtg{n:02d}": 3 for n in range(1, 6)},
}


def measured_curves() -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """그룹별 집계 파워커브(정격비). A5 실측을 터빈 평균으로 모은다."""
    curves = json.loads(A5_RECEIPT.read_text(encoding="utf-8"))["result"]["curves"]
    pooled: dict[int, dict[float, list[float]]] = {}
    for turbine in curves:
        group = GROUP_OF_TURBINE.get(str(turbine["scada_key"]))
        if group is None:
            continue
        rated = turbine["rated_kwh_per_10min"]
        for point in turbine["curve"]:
            centre, mean = point.get("bin_center"), point.get("mean_norm")
            if centre is None or mean is None:
                continue
            pooled.setdefault(group, {}).setdefault(round(float(centre), 2), []).append(
                float(mean) / rated
            )
    out = {}
    for group, slots in pooled.items():
        v = np.array(sorted(slots), dtype="float64")
        p = np.array([float(np.mean(slots[x])) for x in v], dtype="float64")
        p = np.clip(np.maximum.accumulate(np.clip(p, 0.0, 1.0)), 0.0, 1.0)  # 단조화
        out[group] = (v, p)
    return out


def generic_curve(speed: np.ndarray) -> np.ndarray:
    return np.clip((speed - 3.0) / 9.0, 0.0, 1.0) ** 3


def add_sitewind_with_basis(
    matrix: pd.DataFrame,
    legacy: np.ndarray,
    allweather: np.ndarray,
    basis: str,
    curves: dict[int, tuple[np.ndarray, np.ndarray]],
) -> list[str]:
    """사이클 42 의 파생을 그대로 만들되 `_powercurve` 기저함수만 교체한다."""
    matrix["sitewind__legacy"] = legacy
    matrix["sitewind__allweather"] = allweather
    matrix["sitewind__mean"] = (legacy + allweather) / 2.0
    matrix["sitewind__delta"] = allweather - legacy
    matrix["sitewind__disagreement"] = np.abs(allweather - legacy)
    density = matrix.get("phys__air_density")
    scale = (
        np.clip(density.to_numpy(dtype="float64") / RHO_REF, 0.5, 1.5) ** (1.0 / 3.0)
        if density is not None else np.ones(len(matrix))
    )
    groups = matrix["group_id"].to_numpy()
    for source in ("legacy", "allweather", "mean"):
        value = matrix[f"sitewind__{source}"]
        matrix[f"sitewind__{source}2"] = value**2
        matrix[f"sitewind__{source}3"] = value**3
        speed = value.to_numpy(dtype="float64")
        if basis == "generic":
            curve = generic_curve(speed)
        else:
            normalized = speed * scale
            curve = np.zeros(len(speed))
            for group, (v, p) in curves.items():
                mask = groups == group
                if mask.any():
                    curve[mask] = np.interp(normalized[mask], v, p, left=0.0, right=p[-1])
        matrix[f"sitewind__{source}_powercurve"] = curve
    return [name for name in matrix if name.startswith("sitewind__")]


def main() -> int:
    curves = measured_curves()
    assert set(curves) == {1, 2, 3}, f"그룹 커브 누락: {sorted(curves)}"

    surface, _base, auxiliary = _surface()
    surface["forecast_kst_dtm"] = pd.to_datetime(surface["forecast_kst_dtm"])
    surface["capacity"] = surface["group_id"].map(CAPACITIES_KWH).astype(float)
    surface["rate"] = surface["actual_kwh"] / surface["capacity"]
    surface = surface.loc[surface["rate"].notna()].reset_index(drop=True)

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
        }
        for basis in ("generic", "measured"):
            tr = train.copy()
            te = test.copy()
            names = add_sitewind_with_basis(tr, legacy_tr, aw_tr, basis, curves)
            add_sitewind_with_basis(te, legacy_te, aw_te, basis, curves)
            features = [*base_features, *names]
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
            entry[basis] = exp / exp.sum(axis=1, keepdims=True)
        store[probe_fold] = entry

    def scored(fold: str, arm: str, temperature: float) -> pd.DataFrame:
        e = store[fold]
        out = e["meta"].copy()
        out["prediction_kwh"] = (
            bayes_decision(sharpen(e[arm], temperature)) * e["capacity"]
        )
        out["month"] = out["forecast_kst_dtm"].dt.to_period("M").astype(str)
        return out

    chosen_t: dict[str, dict[str, float]] = {}
    pieces: dict[str, list[pd.DataFrame]] = {"generic": [], "measured": []}
    for arm in ("generic", "measured"):
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

    frames, results = {}, {}
    for arm, parts in pieces.items():
        frame = pd.concat(parts, ignore_index=True)
        frames[arm] = frame
        results[arm] = official(frame)

    parent = load_predictions(DEPLOYED)
    parent_score = official(parent)
    v1_gap = abs(results["generic"]["total"] - C44_CONTROL)
    v1 = bool(v1_gap <= V1_TOLERANCE)

    delta = results["measured"]["total"] - results["generic"]["total"]
    ficr_contrib = 0.5 * (results["measured"]["ficr"] - results["generic"]["ficr"])
    nmae_contrib = 0.5 * (
        results["measured"]["one_minus_nmae"] - results["generic"]["one_minus_nmae"]
    )
    gate = evaluate_gate(frames["measured"], frames["generic"])
    gd = gate.evidence

    if v1:
        h1: bool | None = bool(delta > 0)
        h2: bool | None = bool(gate.passed)
        h3: bool | None = bool(results["measured"]["total"] > M115_FIXED_TOTAL)
        h4: bool | None = bool(ficr_contrib > nmae_contrib)
        verdict = (
            "MEASURED_CURVE_PROMOTED" if (h1 and h2)
            else ("MEASURED_CURVE_HELPS_BUT_GATE_REJECTED" if h1
                  else "BASIS_FUNCTION_AXIS_CLOSED")
        )
    else:
        h1 = h2 = h3 = h4 = None
        verdict = "PIPELINE_DRIFT_NOT_JUDGED"

    by_fold = {arm: by_fold_total(frame) for arm, frame in frames.items()}
    check = {
        "V1_expectation": f"GENERIC 이 사이클 44 CONTROL({C44_CONTROL})을 "
                          f"±{V1_TOLERANCE} 이내로 재현",
        "V1_held": v1, "V1_gap": v1_gap,
        "H1_expectation": "MEASURED > GENERIC", "H1_held": h1, "H1_measured": delta,
        "H2_expectation": "MEASURED 가 GENERIC 부모 동결 게이트 통과", "H2_held": h2,
        "H3_expectation": f"MEASURED > {M115_FIXED_TOTAL}", "H3_held": h3,
        "H4_expectation": "이득이 FICR 쪽", "H4_held": h4,
        "judged": v1, "verdict": verdict,
    }

    payload = {
        "node": NODE_ID, "parent_node": PARENT_NODE,
        "gate_version": GATE_VERSION, "gate_modified": False,
        "lockbox_used": False, "external_data_used": False,
        "changed_vs_cycle44": "`sitewind__*_powercurve` 기저함수만. GENERIC=손으로 쓴 램프 / "
                              "MEASURED=A5 실측 그룹 집계 커브 + IEC 밀도정규화",
        "curve_mismatch_measured_in_advance": {
            "rms_all": 0.1453, "max": 0.3158, "rms_steep": 0.2679,
        },
        "curves": {
            str(g): {"v": v.tolist(), "p": p.tolist()} for g, (v, p) in curves.items()
        },
        "features_used": len(base_features) + 14,
        "classifier_fits": fits,
        "chosen_temperature_out_of_fold": chosen_t,
        "scores": {**results, "deployed": parent_score},
        "by_fold": by_fold,
        "delta_measured_minus_generic": delta,
        "contribution": {"ficr": ficr_contrib, "one_minus_nmae": nmae_contrib},
        "gate_measured_vs_generic": {
            "passed": bool(gate.passed),
            "flags": {la.split()[0]: bool(ok) for la, ok in gate.conditions.items()},
            "positive_months": int(gd["positive_months"]),
            "months_scored": int(gd["months_scored"]),
            "sign_test_p": float(gd["sign_test_p_greater"]),
            "bootstrap_q05": float(gd["block_bootstrap_q05"]),
            "min_delta": float(gd["min_total_delta"]),
        },
        "predeclared_check": check,
    }

    g = payload["gate_measured_vs_generic"]
    flags = "".join("O" if g["flags"].get(x) else "-" for x in ("G1", "G2", "G3", "G4"))
    lines = [
        "# M271 P4 사이클 56 — 실측 파워커브로 기저함수 교체",
        "",
        f"- 판정일: {datetime.now(UTC).strftime('%Y-%m-%d')} (UTC)",
        f"- 노드: `{NODE_ID}` / 레인 {LANE} / 부모 `{PARENT_NODE}`",
        f"- 바뀐 것: **{payload['changed_vs_cycle44']}**",
        f"- 게이트: `{GATE_VERSION}` (읽기만 함) / lockbox·외부데이터 미사용",
        "",
        "## 1. 왜 바꾸는가",
        "",
        "`clip((v-3)/9,0,1)**3` 은 손으로 쓴 일반형이고 A5 실측과 어긋난다: "
        f"전체 RMS **{payload['curve_mismatch_measured_in_advance']['rms_all']}**, "
        f"최대 {payload['curve_mismatch_measured_in_advance']['max']}, "
        f"**급경사 RMS {payload['curve_mismatch_measured_in_advance']['rms_steep']}**.",
        "",
        "`sitewind__mean_powercurve` 는 M102 선택 피처의 2 번이다.",
        "",
        "## 2. 가드",
        "",
        f"V1 GENERIC {results['generic']['total']:.6f} vs 사이클 44 CONTROL "
        f"{C44_CONTROL} -> 차이 **{v1_gap:.6f}** (허용 {V1_TOLERANCE}) -> **{v1}**",
        "",
        "## 3. 결과",
        "",
        "| 팔 | Total | 1-NMAE | FICR |",
        "|---|---:|---:|---:|",
        f"| 배포 `M269@{DEPLOYED}` | {parent_score['total']:.6f} | "
        f"{parent_score['one_minus_nmae']:.6f} | {parent_score['ficr']:.6f} |",
        f"| `GENERIC` (현행 램프) | {results['generic']['total']:.6f} | "
        f"{results['generic']['one_minus_nmae']:.6f} | {results['generic']['ficr']:.6f} |",
        f"| **`MEASURED`** (A5 실측) | **{results['measured']['total']:.6f}** | "
        f"{results['measured']['one_minus_nmae']:.6f} | "
        f"{results['measured']['ficr']:.6f} |",
        "",
        f"MEASURED - GENERIC = **{delta:+.6f}** "
        f"(FICR {ficr_contrib:+.6f} / 1-NMAE {nmae_contrib:+.6f})",
        "",
        "| fold | GENERIC | MEASURED |",
        "|---|---:|---:|",
    ]
    for fold in FOLDS:
        lines.append(
            f"| {fold} | {by_fold['generic'].get(fold, float('nan')):.6f} | "
            f"{by_fold['measured'].get(fold, float('nan')):.6f} |"
        )
    lines += [
        "",
        f"동결 게이트 (부모 GENERIC): `{flags}` "
        f"{g['positive_months']}/{g['months_scored']}월 "
        f"p={g['sign_test_p']:.4f} q05={g['bootstrap_q05']:+.6f} -> "
        f"**{'통과' if g['passed'] else '기각'}**",
        "",
        "## 4. 사전확약 대조",
        "",
        f"- V1 -> **{v1}** ({v1_gap:.6f})",
        f"- H1 -> **{h1 if h1 is not None else '판정안함'}** ({delta:+.6f})",
        f"- H2 -> **{h2 if h2 is not None else '판정안함'}**",
        f"- H3 -> **{h3 if h3 is not None else '판정안함'}**",
        f"- H4 -> **{h4 if h4 is not None else '판정안함'}**",
        "",
        f"판정: **{verdict}**",
        "",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    receipt: dict[str, Any] = {
        "schema_version": 1,
        "stage": "M271_P4_CYCLE56_MEASURED_POWERCURVE",
        "node": NODE_ID, "lane": LANE,
        "decided_utc": datetime.now(UTC).isoformat(),
        "result": payload,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "report_sha256": hashlib.sha256(REPORT_MD.read_bytes()).hexdigest(),
        "dacon_upload": False, "external_actions": [], "model_fits": fits,
        "lockbox_reopened": False, "new_2024_evaluation": False,
    }
    RECEIPT.write_text(
        json.dumps(receipt, indent=1, ensure_ascii=False, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

    print(f"[C56] 적합 {fits} / 피처 {payload['features_used']} / 온도 {chosen_t}")
    print(f"[C56] GENERIC  {results['generic']['total']:.6f} "
          f"(사이클44 {C44_CONTROL}, 차이 {v1_gap:.6f}) -> V1 {v1}")
    print(f"[C56] MEASURED {results['measured']['total']:.6f} "
          f"(GENERIC 대비 {delta:+.6f})")
    print(f"[C56] 기여 FICR {ficr_contrib:+.6f} / 1-NMAE {nmae_contrib:+.6f}")
    print(f"[C56] 게이트 [{flags}] {g['positive_months']}/{g['months_scored']}월 -> "
          f"{'통과' if g['passed'] else '기각'}")
    print(f"[C56] 판정: {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
