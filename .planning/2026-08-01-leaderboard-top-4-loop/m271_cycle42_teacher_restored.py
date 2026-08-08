"""M271 P4 사이클 42 — teacher 피처 복원 후 밴드 인지 학습 재판정.

사이클 41 에서 증량 예산을 소진했고 V2 가 기각됐다 — 용량을 올리자 대조군이 **더** 나빠졌다
(0.584468 -> 0.567875). 격차의 원인은 저용량이 아니라 과적합이었다. 사이클 40·41 이 실행
전에 명시한 나머지 원인은 하나다: **결측된 13 개 `sitewind__*`**, M102 피처 목록의 최상위
6 개를 포함하는 teacher 산출물.

이 노드는 그 결손을 제거한다. **용량 튜닝이 아니라 사전에 특정된 결손 원인의 제거**이므로
사이클 41 이 선언한 증량 예산 제한에 걸리지 않는다.

teacher 구조 (`build_full_history_strict_temporal_champion._add_sitewind_features` 와
`run_site_wind_teacher` 에서 복원)
  - NWP -> 관측 나셀풍속(`scada_ws`) 회귀기를 **두 프로파일**로 적합
      legacy      = `_surface()` 가 반환하는 auxiliary_columns
      allweather  = surface 의 전 수치 컬럼에서 키·라벨·`scada_ws` 제외
  - 파생 14 개: legacy, allweather, mean, delta, disagreement,
                그리고 (legacy, allweather, mean) 각각 제곱·세제곱·파워커브
                `powercurve = clip((v-3)/9, 0, 1)^3`

**누출 규율.** `scada_ws` 는 teacher 의 **표적**이지 피처가 아니다. 분류기에는 teacher 의
**예측**만 들어간다(추론시점 가용). 분류기 학습행에는 **KFold OOF 예측**을 써서 teacher 가
자기가 본 행을 되먹이지 않게 한다 — 원본 구조 그대로다.

① 방법 리서치 (실행 전)
  - 새 방법 없음. 원본 teacher 구조를 복원하고 사이클 40 의 대조를 그대로 다시 돈다.
  - 바뀌는 것은 **피처 집합뿐**: 87 -> 87 + sitewind 파생. 용량은 사이클 40 값으로
    되돌린다(41 이 증량은 해롭다고 보였으므로).

② 사양 동결

  용량   사이클 40 과 동일 (leaves 15, lr 0.1, rounds 200). 41 의 증량은 폐기.
  피처   M115 의 100 개 중 캐시 가용 87 개 + **복원된 sitewind 파생**
  그 외  사이클 40 과 동일: 46 class, CONTROL=one-hot / BAND=정산모양 soft target,
         같은 Bayes 결정규칙, fold 별 chronology-safe, `scada_ws` 피처 제외

  **타당성 가드**
    V1  CONTROL 이 배포(0.628605)의 `-0.03` 이내. (사이클 40·41 과 동일 문턱)
    V3  teacher 복원이 CONTROL 을 실제로 개선한다 (사이클 40 CONTROL 0.584468 초과).
        기각되면 결손 원인 진단이 틀린 것이므로 그 사실을 기록한다.

  사전확약(V1 통과시에만 판정):
    H1  BAND > CONTROL.
    H2  BAND 가 배포 대비 개선 + **동결 게이트 통과**.
    H3  BAND > `M115@T0.6_G0.2` (0.630310).
    H4  이득이 **FICR 쪽**에서 나온다.
    H5  처리효과 부호가 사이클 40·41 과 동일 (세 번째 재현).

**게이트를 수정하지 않는다.** 2024 행·lockbox 미사용. `scada_ws` 는 teacher 표적으로만 사용.
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
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold

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
    V1_TOLERANCE,
    bayes_decision,
    by_fold_total,
    make_objective,
    one_hot_targets,
    soft_targets,
)
from m271_evaluate_candidate import official
from run_sequence_classifier import _surface

from baram.constants import CAPACITIES_KWH

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
REPORT_MD = REPORTS / "m271_cycle42_teacher_restored.md"
RECEIPT = REPORTS / "m271_cycle42_teacher_restored_receipt.json"

NODE_ID = "C1N42_TEACHER_RESTORED"
LANE = "L3"
PARENT_NODE = "C1N41_BAND_CLASSIFIER_CAPACITY"
DEPLOYED_TOTAL = 0.628605
C40_CONTROL = 0.584468
PRIOR_DELTAS = (0.009772, 0.021490)
TEACHER_SEED = 20260802

TEACHER_PARAMS = {
    "n_estimators": 400,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 40,
    "max_bin": 255,
    "subsample": 0.9,
    "subsample_freq": 1,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.1,
    "reg_lambda": 3.0,
    "random_state": TEACHER_SEED,
    "n_jobs": 1,
    "deterministic": True,
    "force_col_wise": True,
    "verbosity": -1,
}
TEACHER_EXCLUDED = {
    "forecast_id", "forecast_kst_dtm", "data_available_kst_dtm", "issuance_batch",
    "actual_kwh", "scada_ws", "capacity", "rate",
}


def all_weather_columns(surface: pd.DataFrame) -> list[str]:
    return [
        name for name in surface
        if name not in TEACHER_EXCLUDED and pd.api.types.is_numeric_dtype(surface[name])
    ]


def teach(
    train: pd.DataFrame, test: pd.DataFrame, columns: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """그룹별 teacher. 학습행은 KFold OOF, 보류행은 전량 적합 모형."""
    train_pred = np.full(len(train), np.nan, dtype="float64")
    test_pred = np.full(len(test), np.nan, dtype="float64")
    for group in sorted(train["group_id"].unique()):
        tr_mask = (train["group_id"] == group).to_numpy()
        te_mask = (test["group_id"] == group).to_numpy()
        labelled = tr_mask & train["scada_ws"].notna().to_numpy()
        positions = np.flatnonzero(labelled)
        if len(positions) < 200:
            continue
        x = train.loc[:, columns].astype("float32")
        y = train["scada_ws"].to_numpy(dtype="float64")
        splitter = KFold(3, shuffle=True, random_state=TEACHER_SEED + int(group))
        for fit_idx, hold_idx in splitter.split(positions):
            model = LGBMRegressor(**TEACHER_PARAMS)
            model.fit(x.iloc[positions[fit_idx]], y[positions[fit_idx]])
            train_pred[positions[hold_idx]] = model.predict(x.iloc[positions[hold_idx]])
        final = LGBMRegressor(**TEACHER_PARAMS)
        final.fit(x.iloc[positions], y[positions])
        # 라벨이 없는 학습행에도 최종 모형으로 채운다(그 행은 teacher 가 본 적 없다).
        unlabelled = np.flatnonzero(tr_mask & ~labelled)
        if len(unlabelled):
            train_pred[unlabelled] = final.predict(x.iloc[unlabelled])
        if te_mask.any():
            test_pred[te_mask] = final.predict(
                test.loc[te_mask, columns].astype("float32")
            )
    return train_pred, test_pred


def add_sitewind(matrix: pd.DataFrame, legacy: np.ndarray, allweather: np.ndarray) -> list[str]:
    matrix["sitewind__legacy"] = legacy
    matrix["sitewind__allweather"] = allweather
    matrix["sitewind__mean"] = (legacy + allweather) / 2.0
    matrix["sitewind__delta"] = allweather - legacy
    matrix["sitewind__disagreement"] = np.abs(allweather - legacy)
    for source in ("legacy", "allweather", "mean"):
        value = matrix[f"sitewind__{source}"]
        matrix[f"sitewind__{source}2"] = value**2
        matrix[f"sitewind__{source}3"] = value**3
        normalized = np.clip((value - 3.0) / 9.0, 0.0, 1.0)
        matrix[f"sitewind__{source}_powercurve"] = normalized**3
    return [name for name in matrix if name.startswith("sitewind__")]


def main() -> int:
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
    assert "scada_ws" not in aux_cols and "scada_ws" not in aw_cols

    pieces: dict[str, list[pd.DataFrame]] = {"BAND": [], "CONTROL": []}
    fits = 0
    teacher_fits = 0
    sitewind_names: list[str] = []
    for _fold, meta in fold_rows().items():
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
        teacher_fits += 2 * 4 * train["group_id"].nunique()
        sitewind_names = add_sitewind(train, legacy_tr, aw_tr)
        add_sitewind(test, legacy_te, aw_te)

        features = [*base_features, *sitewind_names]
        x_tr = train.loc[:, features].astype("float32")
        rate = np.clip(train["rate"].to_numpy(dtype="float64"), 0.0, None)
        label = np.clip((rate / CLASS_WIDTH).astype(int), 0, N_CLASS - 1)
        for name, target in (
            ("BAND", soft_targets(rate)),
            ("CONTROL", one_hot_targets(rate)),
        ):
            dataset = lgb.Dataset(x_tr, label=label, free_raw_data=False)
            params = dict(PARAMS)
            params["objective"] = make_objective(target)
            booster = lgb.train(params, dataset, num_boost_round=ROUNDS)
            fits += 1
            raw = np.asarray(
                booster.predict(test.loc[:, features].astype("float32"))
            ).reshape(len(test), N_CLASS)
            exp = np.exp(raw - raw.max(axis=1, keepdims=True))
            prob = exp / exp.sum(axis=1, keepdims=True)
            out = test.loc[:, [*KEYS, "actual_kwh"]].copy()
            out["prediction_kwh"] = bayes_decision(prob) * test["capacity"].to_numpy()
            pieces[name].append(out)

    frames, results = {}, {}
    for name, parts in pieces.items():
        frame = pd.concat(parts, ignore_index=True)
        frame["month"] = frame["forecast_kst_dtm"].dt.to_period("M").astype(str)
        frames[name] = frame
        results[name] = official(frame)

    parent = load_predictions(DEPLOYED)
    parent_score = official(parent)
    v1_gap = results["CONTROL"]["total"] - parent_score["total"]
    v1 = bool(v1_gap >= -V1_TOLERANCE)
    v3 = bool(results["CONTROL"]["total"] > C40_CONTROL)

    delta = results["BAND"]["total"] - results["CONTROL"]["total"]
    ficr_contrib = 0.5 * (results["BAND"]["ficr"] - results["CONTROL"]["ficr"])
    nmae_contrib = 0.5 * (
        results["BAND"]["one_minus_nmae"] - results["CONTROL"]["one_minus_nmae"]
    )
    gate = evaluate_gate(frames["BAND"], parent)
    gd = gate.evidence
    h5 = bool(all(np.sign(delta) == np.sign(d) for d in PRIOR_DELTAS))

    if v1:
        h1: bool | None = bool(delta > 0)
        h2: bool | None = bool(
            results["BAND"]["total"] > parent_score["total"] and gate.passed
        )
        h3: bool | None = bool(results["BAND"]["total"] > M115_FIXED_TOTAL)
        h4: bool | None = bool(ficr_contrib > nmae_contrib)
        verdict = (
            "BAND_CLASSIFIER_PROMOTED" if h2
            else ("BAND_CLASSIFIER_HELPS_INTERNALLY_ONLY" if h1
                  else "BAND_AWARE_TRAINING_CLOSED")
        )
    else:
        h1 = h2 = h3 = h4 = None
        verdict = "TEACHER_RESTORED_STILL_UNJUDGED"

    by_fold = {name: by_fold_total(frame) for name, frame in frames.items()}
    check = {
        "V1_expectation": f"CONTROL 이 배포의 -{V1_TOLERANCE} 이내",
        "V1_held": v1, "V1_measured_gap": v1_gap,
        "V3_expectation": f"teacher 복원이 CONTROL 을 개선 (> {C40_CONTROL})",
        "V3_held": v3, "V3_measured": results["CONTROL"]["total"] - C40_CONTROL,
        "H1_expectation": "BAND > CONTROL", "H1_held": h1, "H1_measured": delta,
        "H2_expectation": "BAND 가 배포 대비 개선 + 게이트 통과", "H2_held": h2,
        "H3_expectation": f"BAND > {M115_FIXED_TOTAL}", "H3_held": h3,
        "H4_expectation": "이득이 FICR 쪽", "H4_held": h4,
        "H5_expectation": f"처리효과 부호가 사이클 40·41{PRIOR_DELTAS} 과 동일",
        "H5_held": h5,
        "judged": v1, "verdict": verdict,
    }

    payload = {
        "node": NODE_ID, "parent_node": PARENT_NODE,
        "gate_version": GATE_VERSION, "gate_modified": False, "lockbox_used": False,
        "changed_vs_cycle40": "피처에 복원된 sitewind 파생 추가. 용량은 사이클 40 값으로 복귀",
        "leakage_discipline": "scada_ws 는 teacher 표적으로만 사용. 분류기에는 teacher "
                              "예측만 들어가며 학습행은 KFold OOF",
        "teacher": {
            "profiles": {"legacy": len(aux_cols), "allweather": len(aw_cols)},
            "params": TEACHER_PARAMS, "fits": teacher_fits,
            "derived_features": sitewind_names,
        },
        "features_used": len(base_features) + len(sitewind_names),
        "params": PARAMS, "rounds": ROUNDS, "classifier_fits": fits,
        "scores": {**results, "deployed": parent_score},
        "by_fold": by_fold,
        "delta_band_minus_control": delta,
        "contribution": {"ficr": ficr_contrib, "one_minus_nmae": nmae_contrib},
        "prior_deltas": list(PRIOR_DELTAS),
        "gate_vs_deployed": {
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

    g = payload["gate_vs_deployed"]
    flags = "".join("O" if g["flags"].get(x) else "-" for x in ("G1", "G2", "G3", "G4"))
    lines = [
        "# M271 P4 사이클 42 — teacher 피처 복원 후 재판정",
        "",
        f"- 판정일: {datetime.now(UTC).strftime('%Y-%m-%d')} (UTC)",
        f"- 노드: `{NODE_ID}` / 레인 {LANE} / 부모 `{PARENT_NODE}`",
        f"- 사이클 40 대비 바뀐 것: **{payload['changed_vs_cycle40']}**",
        f"- 누출 규율: {payload['leakage_discipline']}",
        f"- 게이트: `{GATE_VERSION}` (읽기만 함) / lockbox 미사용",
        "",
        "## 1. teacher",
        "",
        f"- 프로파일 `legacy` {len(aux_cols)} 피처 / `allweather` {len(aw_cols)} 피처",
        f"- teacher 적합 {teacher_fits} 회, 파생 {len(sitewind_names)} 개",
        f"- 분류기 피처 총 **{payload['features_used']}** 개",
        "",
        "## 2. 가드",
        "",
        f"- V1 CONTROL {results['CONTROL']['total']:.6f} vs 배포 "
        f"{parent_score['total']:.6f} -> **{v1_gap:+.6f}** -> **{v1}**",
        f"- V3 teacher 복원 효과 **{results['CONTROL']['total'] - C40_CONTROL:+.6f}** "
        f"(사이클 40 CONTROL {C40_CONTROL:.6f}) -> **{v3}**",
        "",
        "## 3. 결과",
        "",
        "| 모델 | Total | 1-NMAE | FICR |",
        "|---|---:|---:|---:|",
        f"| 배포 | {parent_score['total']:.6f} | "
        f"{parent_score['one_minus_nmae']:.6f} | {parent_score['ficr']:.6f} |",
        f"| `CONTROL` | {results['CONTROL']['total']:.6f} | "
        f"{results['CONTROL']['one_minus_nmae']:.6f} | {results['CONTROL']['ficr']:.6f} |",
        f"| **`BAND`** | **{results['BAND']['total']:.6f}** | "
        f"{results['BAND']['one_minus_nmae']:.6f} | {results['BAND']['ficr']:.6f} |",
        "",
        f"BAND - CONTROL = **{delta:+.6f}** "
        f"(FICR {ficr_contrib:+.6f} / 1-NMAE {nmae_contrib:+.6f})",
        f"— 사이클 40 {PRIOR_DELTAS[0]:+.6f}, 41 {PRIOR_DELTAS[1]:+.6f}",
        "",
        "| fold | CONTROL | BAND |",
        "|---|---:|---:|",
    ]
    for fold in FOLDS:
        lines.append(
            f"| {fold} | {by_fold['CONTROL'].get(fold, float('nan')):.6f} | "
            f"{by_fold['BAND'].get(fold, float('nan')):.6f} |"
        )
    lines += [
        "",
        f"BAND 대 배포 게이트: `{flags}` {g['positive_months']}/{g['months_scored']}월 "
        f"p={g['sign_test_p']:.4f} q05={g['bootstrap_q05']:+.6f} -> "
        f"**{'통과' if g['passed'] else '기각'}**",
        "",
        "## 4. 사전확약 대조",
        "",
        f"- V1 -> **{v1}** ({v1_gap:+.6f})",
        f"- V3 `{check['V3_expectation']}` -> **{v3}**",
        f"- H1 -> **{h1 if h1 is not None else '판정안함'}** ({delta:+.6f})",
        f"- H2 -> **{h2 if h2 is not None else '판정안함'}**",
        f"- H3 -> **{h3 if h3 is not None else '판정안함'}**",
        f"- H4 -> **{h4 if h4 is not None else '판정안함'}**",
        f"- H5 `{check['H5_expectation']}` -> **{h5}**",
        "",
        f"판정: **{verdict}**",
        "",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    receipt: dict[str, Any] = {
        "schema_version": 1,
        "stage": "M271_P4_CYCLE42_TEACHER_RESTORED",
        "node": NODE_ID, "lane": LANE,
        "decided_utc": datetime.now(UTC).isoformat(),
        "result": payload,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "report_sha256": hashlib.sha256(REPORT_MD.read_bytes()).hexdigest(),
        "dacon_upload": False, "external_actions": [],
        "model_fits": fits + teacher_fits,
        "lockbox_reopened": False, "new_2024_evaluation": False,
    }
    RECEIPT.write_text(
        json.dumps(receipt, indent=1, ensure_ascii=False, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

    print(f"[C42] teacher 적합 {teacher_fits} / 분류기 적합 {fits} / "
          f"피처 {payload['features_used']} (sitewind {len(sitewind_names)})")
    print(f"[C42] CONTROL {results['CONTROL']['total']:.6f} "
          f"(배포 대비 {v1_gap:+.6f}, 사이클40 대비 "
          f"{results['CONTROL']['total'] - C40_CONTROL:+.6f}) -> V1 {v1} V3 {v3}")
    print(f"[C42] BAND    {results['BAND']['total']:.6f} (CONTROL 대비 {delta:+.6f})")
    print(f"[C42] 기여 FICR {ficr_contrib:+.6f} / 1-NMAE {nmae_contrib:+.6f} | H5 {h5}")
    print(f"[C42] 게이트 [{flags}] {g['positive_months']}/{g['months_scored']}월 -> "
          f"{'통과' if g['passed'] else '기각'}")
    print(f"[C42] 판정: {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
