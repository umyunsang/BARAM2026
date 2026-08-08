"""M271 P4 사이클 58 — teacher OOF 를 시간 블록 분할로. 학습/테스트 피처 분포 정합.

사이클 53 이 `teach()` 의 `KFold(3, shuffle=True)` 시간 누출을 드러냈고, 54 가 정직한
값을 쟀다.

    무작위 KFold OOF   오차 감소 46~49%  ->  sigma 약 1.00~1.12
    시간 분할          오차 감소 24.7~29.8% ->  sigma 약 1.37~1.50

여기서 **생산 파이프라인의 결함**이 따라 나온다. teacher 는

  - 분류기 **학습행** 에 무작위 KFold OOF 를 준다 -> 누출이 섞인 sigma 약 1.0
  - 분류기 **테스트행** 에 최종 모형 예측을 준다 -> 정직한 sigma 약 1.37

즉 **학습 때 보는 `sitewind__*` 가 테스트 때보다 약 30% 정확하다.** 분류기는 그 피처를
실제보다 더 신뢰하도록 학습되고 테스트에서 배신당한다. 어떤 잔차 프로브에도 안 잡히는
종류의 손실이다 — 피처는 있고 풍속은 정확하고 산포도 여유가 있는데 **분포가 어긋나 있다**.

사이클 56 이 기저함수 축을 닫았으므로(`GBM 이 단조 변환을 스스로 학습`), `풍속->출력`
층에서 남은 후보는 이것이다.

① 방법 리서치 (실행 전)
  - **train/test feature distribution shift** 는 스태킹·타깃인코딩에서 잘 알려진 실패
    모드다. 상위 모형이 하위 모형 산출을 피처로 받을 때, 학습행 산출이 누출로 과대정확하면
    상위 모형이 그것을 과신한다. 표준 처방은 **누출 없는 분할**로 OOF 를 만드는 것이다.
  - 시계열이므로 무작위가 아니라 **시간 블록** 분할이어야 한다. A1 이 잰 lag-1 자기상관
    0.951~0.962 가 그 이유다.
  - 새 학습 방법은 없다. 사이클 44 설정에서 **teacher 의 분할 방식만** 바꾼다.

② 사양 동결

  SHUFFLE  현행 `KFold(3, shuffle=True, random_state=SEED+group)` (사이클 42·44 와 동일)
  BLOCKED  같은 3 분할이지만 **시간 순서대로 연속 블록** (셔플 없음)
  그 외는 사이클 44 와 동일: 46 class one-hot, leaves 15 / lr 0.1 / rounds 200,
  Bayes 결정 + fold-외 온도, 87 + sitewind 피처, `scada_ws` 는 teacher 표적으로만.

  **타당성 가드**
    V1  SHUFFLE 팔이 사이클 44 CONTROL(0.604043)을 `±0.005` 이내로 재현한다.

  사전확약(V1 통과시에만 판정):
    H1  BLOCKED > SHUFFLE (pooled Total). 학습/테스트 피처 정합이 이득을 준다.
    H2  BLOCKED 가 SHUFFLE 을 부모로 한 **동결 게이트를 통과**한다.
    H3  BLOCKED > `M115@T0.6_G0.2` (0.630310).
    H4  이득이 **FICR 쪽**에서 나온다.

  H1 이 기각되면 분포 정합 축이 닫히고, `풍속->출력` 층에서 내가 지목할 수 있는 후보가
  소진된다. 그 경우 병목을 다시 특정해야 한다.

**게이트를 수정하지 않는다.** lockbox·외부데이터 미사용.
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
from m271_cycle42_teacher_restored import (
    TEACHER_PARAMS,
    TEACHER_SEED,
    add_sitewind,
    all_weather_columns,
)
from m271_cycle44_sharpened_decision import TEMPERATURES, sharpen
from m271_evaluate_candidate import official
from run_sequence_classifier import _surface

from baram.constants import CAPACITIES_KWH

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
REPORT_MD = REPORTS / "m271_cycle58_teacher_oof_split.md"
RECEIPT = REPORTS / "m271_cycle58_teacher_oof_split_receipt.json"

NODE_ID = "C1N58_TEACHER_OOF_SPLIT"
LANE = "L3"
PARENT_NODE = "C1N56_MEASURED_POWERCURVE"
C44_CONTROL = 0.604043
V1_TOLERANCE = 0.005
N_SPLITS = 3


def teach_split(
    train: pd.DataFrame, test: pd.DataFrame, columns: list[str], scheme: str
) -> tuple[np.ndarray, np.ndarray]:
    """사이클 42 의 teach() 와 동일하되 **분할 방식만** 매개변수화한다."""
    train_pred = np.full(len(train), np.nan, dtype="float64")
    test_pred = np.full(len(test), np.nan, dtype="float64")
    order = np.argsort(train["forecast_kst_dtm"].to_numpy())
    rank = np.empty(len(train), dtype=int)
    rank[order] = np.arange(len(train))

    for group in sorted(train["group_id"].unique()):
        tr_mask = (train["group_id"] == group).to_numpy()
        te_mask = (test["group_id"] == group).to_numpy()
        labelled = tr_mask & train["scada_ws"].notna().to_numpy()
        positions = np.flatnonzero(labelled)
        if len(positions) < 200:
            continue
        x = train.loc[:, columns].astype("float32")
        y = train["scada_ws"].to_numpy(dtype="float64")

        if scheme == "shuffle":
            from sklearn.model_selection import KFold

            splitter = KFold(N_SPLITS, shuffle=True, random_state=TEACHER_SEED + int(group))
            folds = list(splitter.split(positions))
        else:
            # 시간 순서대로 연속 블록. 셔플 없음.
            local = np.argsort(rank[positions])
            chunks = np.array_split(local, N_SPLITS)
            folds = [
                (np.concatenate([c for j, c in enumerate(chunks) if j != i]), chunks[i])
                for i in range(N_SPLITS)
            ]

        for fit_idx, hold_idx in folds:
            model = LGBMRegressor(**TEACHER_PARAMS)
            model.fit(x.iloc[positions[fit_idx]], y[positions[fit_idx]])
            train_pred[positions[hold_idx]] = model.predict(x.iloc[positions[hold_idx]])
        final = LGBMRegressor(**TEACHER_PARAMS)
        final.fit(x.iloc[positions], y[positions])
        unlabelled = np.flatnonzero(tr_mask & ~labelled)
        if len(unlabelled):
            train_pred[unlabelled] = final.predict(x.iloc[unlabelled])
        if te_mask.any():
            test_pred[te_mask] = final.predict(test.loc[te_mask, columns].astype("float32"))
    return train_pred, test_pred


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

    store: dict[str, dict[str, Any]] = {}
    fits = 0
    teacher_sigma: dict[str, list[float]] = {"shuffle": [], "blocked": []}
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
        rate = np.clip(train["rate"].to_numpy(dtype="float64"), 0.0, None)
        label = np.clip((rate / CLASS_WIDTH).astype(int), 0, N_CLASS - 1)
        target = one_hot_targets(rate)
        entry: dict[str, Any] = {
            "meta": test.loc[:, [*KEYS, "actual_kwh"]].copy(),
            "capacity": test["capacity"].to_numpy(dtype="float64"),
        }
        for scheme in ("shuffle", "blocked"):
            legacy_tr, legacy_te = teach_split(train, test, aux_cols, scheme)
            aw_tr, aw_te = teach_split(train, test, aw_cols, scheme)
            # 학습행 OOF 의 정직도: scada_ws 가 있는 행에서만 잰다
            ok = train["scada_ws"].notna().to_numpy() & np.isfinite(aw_tr)
            teacher_sigma[scheme].append(
                float(np.std(aw_tr[ok] - train["scada_ws"].to_numpy()[ok], ddof=1))
            )
            tr, te = train.copy(), test.copy()
            names = add_sitewind(tr, legacy_tr, aw_tr)
            add_sitewind(te, legacy_te, aw_te)
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
            entry[scheme] = exp / exp.sum(axis=1, keepdims=True)
        store[probe_fold] = entry

    def scored(fold: str, arm: str, temperature: float) -> pd.DataFrame:
        e = store[fold]
        out = e["meta"].copy()
        out["prediction_kwh"] = bayes_decision(sharpen(e[arm], temperature)) * e["capacity"]
        out["month"] = out["forecast_kst_dtm"].dt.to_period("M").astype(str)
        return out

    chosen_t: dict[str, dict[str, float]] = {}
    pieces: dict[str, list[pd.DataFrame]] = {"shuffle": [], "blocked": []}
    for arm in ("shuffle", "blocked"):
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

    parent_score = official(load_predictions(DEPLOYED))
    v1_gap = abs(results["shuffle"]["total"] - C44_CONTROL)
    v1 = bool(v1_gap <= V1_TOLERANCE)

    delta = results["blocked"]["total"] - results["shuffle"]["total"]
    ficr_contrib = 0.5 * (results["blocked"]["ficr"] - results["shuffle"]["ficr"])
    nmae_contrib = 0.5 * (
        results["blocked"]["one_minus_nmae"] - results["shuffle"]["one_minus_nmae"]
    )
    gate = evaluate_gate(frames["blocked"], frames["shuffle"])
    gd = gate.evidence

    if v1:
        h1: bool | None = bool(delta > 0)
        h2: bool | None = bool(gate.passed)
        h3: bool | None = bool(results["blocked"]["total"] > M115_FIXED_TOTAL)
        h4: bool | None = bool(ficr_contrib > nmae_contrib)
        verdict = (
            "BLOCKED_OOF_PROMOTED" if (h1 and h2)
            else ("BLOCKED_OOF_HELPS_BUT_GATE_REJECTED" if h1
                  else "FEATURE_DISTRIBUTION_SHIFT_AXIS_CLOSED")
        )
    else:
        h1 = h2 = h3 = h4 = None
        verdict = "PIPELINE_DRIFT_NOT_JUDGED"

    by_fold = {arm: by_fold_total(frame) for arm, frame in frames.items()}
    sigma_summary = {
        arm: {"mean": float(np.mean(v)), "per_fold": v}
        for arm, v in teacher_sigma.items()
    }
    check = {
        "V1_expectation": f"SHUFFLE 이 사이클 44 CONTROL({C44_CONTROL}) ±{V1_TOLERANCE} 재현",
        "V1_held": v1, "V1_gap": v1_gap,
        "H1_expectation": "BLOCKED > SHUFFLE", "H1_held": h1, "H1_measured": delta,
        "H2_expectation": "BLOCKED 가 SHUFFLE 부모 동결 게이트 통과", "H2_held": h2,
        "H3_expectation": f"BLOCKED > {M115_FIXED_TOTAL}", "H3_held": h3,
        "H4_expectation": "이득이 FICR 쪽", "H4_held": h4,
        "judged": v1, "verdict": verdict,
    }

    payload = {
        "node": NODE_ID, "parent_node": PARENT_NODE,
        "gate_version": GATE_VERSION, "gate_modified": False,
        "lockbox_used": False, "external_data_used": False,
        "defect": "teacher 가 학습행에 무작위 KFold OOF(누출)를, 테스트행에 최종 모형 "
                  "예측(정직)을 준다. 학습 때 보는 sitewind 피처가 테스트보다 정확하다",
        "changed_vs_cycle44": "teacher 의 OOF 분할 방식만 (shuffle -> 시간 블록)",
        "teacher_train_oof_sigma": sigma_summary,
        "features_used": len(base_features) + 14,
        "classifier_fits": fits,
        "chosen_temperature_out_of_fold": chosen_t,
        "scores": {**results, "deployed": parent_score},
        "by_fold": by_fold,
        "delta_blocked_minus_shuffle": delta,
        "contribution": {"ficr": ficr_contrib, "one_minus_nmae": nmae_contrib},
        "gate_blocked_vs_shuffle": {
            "passed": bool(gate.passed),
            "flags": {la.split()[0]: bool(ok) for la, ok in gate.conditions.items()},
            "positive_months": int(gd["positive_months"]),
            "months_scored": int(gd["months_scored"]),
            "sign_test_p": float(gd["sign_test_p_greater"]),
            "bootstrap_q05": float(gd["block_bootstrap_q05"]),
        },
        "predeclared_check": check,
    }

    g = payload["gate_blocked_vs_shuffle"]
    flags = "".join("O" if g["flags"].get(x) else "-" for x in ("G1", "G2", "G3", "G4"))
    lines = [
        "# M271 P4 사이클 58 — teacher OOF 를 시간 블록 분할로",
        "",
        f"- 판정일: {datetime.now(UTC).strftime('%Y-%m-%d')} (UTC)",
        f"- 노드: `{NODE_ID}` / 레인 {LANE} / 부모 `{PARENT_NODE}`",
        f"- 바뀐 것: **{payload['changed_vs_cycle44']}**",
        f"- 게이트: `{GATE_VERSION}` (읽기만 함) / lockbox·외부데이터 미사용",
        "",
        "## 1. 결함",
        "",
        payload["defect"] + ".",
        "",
        "| 분할 | 학습행 OOF sigma (fold 평균) |",
        "|---|---:|",
        f"| `shuffle` (현행) | **{sigma_summary['shuffle']['mean']:.4f}** |",
        f"| `blocked` (시간) | **{sigma_summary['blocked']['mean']:.4f}** |",
        "",
        "shuffle 쪽이 낮으면 그만큼 **학습행 피처가 과대정확**했다는 뜻이다.",
        "",
        "## 2. 가드",
        "",
        f"V1 SHUFFLE {results['shuffle']['total']:.6f} vs 사이클 44 {C44_CONTROL} "
        f"-> 차이 **{v1_gap:.6f}** -> **{v1}**",
        "",
        "## 3. 결과",
        "",
        "| 팔 | Total | 1-NMAE | FICR |",
        "|---|---:|---:|---:|",
        f"| 배포 | {parent_score['total']:.6f} | "
        f"{parent_score['one_minus_nmae']:.6f} | {parent_score['ficr']:.6f} |",
        f"| `SHUFFLE` (현행) | {results['shuffle']['total']:.6f} | "
        f"{results['shuffle']['one_minus_nmae']:.6f} | {results['shuffle']['ficr']:.6f} |",
        f"| **`BLOCKED`** | **{results['blocked']['total']:.6f}** | "
        f"{results['blocked']['one_minus_nmae']:.6f} | {results['blocked']['ficr']:.6f} |",
        "",
        f"BLOCKED - SHUFFLE = **{delta:+.6f}** "
        f"(FICR {ficr_contrib:+.6f} / 1-NMAE {nmae_contrib:+.6f})",
        "",
        "| fold | SHUFFLE | BLOCKED |",
        "|---|---:|---:|",
    ]
    for fold in FOLDS:
        lines.append(
            f"| {fold} | {by_fold['shuffle'].get(fold, float('nan')):.6f} | "
            f"{by_fold['blocked'].get(fold, float('nan')):.6f} |"
        )
    lines += [
        "",
        f"동결 게이트 (부모 SHUFFLE): `{flags}` "
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
        "stage": "M271_P4_CYCLE58_TEACHER_OOF_SPLIT",
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

    print(f"[C58] teacher 학습행 OOF sigma  shuffle "
          f"{sigma_summary['shuffle']['mean']:.4f}  blocked "
          f"{sigma_summary['blocked']['mean']:.4f}")
    print(f"[C58] SHUFFLE {results['shuffle']['total']:.6f} "
          f"(사이클44 {C44_CONTROL}, 차이 {v1_gap:.6f}) -> V1 {v1}")
    print(f"[C58] BLOCKED {results['blocked']['total']:.6f} "
          f"(SHUFFLE 대비 {delta:+.6f})")
    print(f"[C58] 기여 FICR {ficr_contrib:+.6f} / 1-NMAE {nmae_contrib:+.6f}")
    print(f"[C58] 게이트 [{flags}] {g['positive_months']}/{g['months_scored']}월 -> "
          f"{'통과' if g['passed'] else '기각'}")
    print(f"[C58] 판정: {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
