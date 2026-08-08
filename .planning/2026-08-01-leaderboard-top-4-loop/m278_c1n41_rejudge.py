
"""M278 — C1N41 재판정: 밴드타깃 이득은 진짜인가, 캘리브레이션 아티팩트인가.

## 배경
C1N40/41 은 두 팔(CONTROL=one-hot / BAND=정산모양 soft target)에 **같은 평탄 결정규칙**
(`bayes_decision`, m271_n8 판독상 T1_G0.5435)을 적용해 처리효과 +0.009772 / +0.021490 을 얻었다.

M276 판정: 결정층 온도는 캘리브레이션 수리다 (T_cal=1.731, T*=2.00).
BAND 의 soft target 은 `q_i ∝ u(|c_i-y|)` 로 ±0.08 폭(약 9 구간) 평탄분포를 목표로 학습한다.
즉 **BAND 는 학습단계에서 이미 무르게** 되고, CONTROL(one-hot, 1 구간 스파이크)은 뾰족하다.
평탄 결정규칙을 공유하면 BAND 만 공짜로 캘리브레이션 이득을 가져간다.

## 사전확약 (실행 전 동결)
- V1  재현: 아카이브 by_fold 점수를 1e-9 이내로 재현한다
      CONTROL Q2 0.5807629486268431 / BAND Q2 0.5965692367188797
- H1  팔별 최적정책(fold-외 확장격자) 하에서 처리효과가 **축소**된다 (< +0.021490)
- H2  축소폭이 원효과의 50% 초과 (새 처리효과 < +0.010745)
- H3  CONTROL 최적 T > BAND 최적 T (BAND 가 이미 무르므로)

**반증**: H1 이 거짓이면 밴드타깃 이득은 결정정책과 무관한 진짜 학습이득이며,
`C1N41` 은 teacher 피처 복원 후 정식 판정 대상으로 승격한다.

락박스 미접근 / 모델 적합 6회(원본과 동일) / 제출물 없음 / 외부행위 없음.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / "src"))

from m271_cycle37_band_loss import KEYS, PROBE, fold_rows          # noqa: E402
from m271_cycle40_band_classifier import (                          # noqa: E402
    CENTERS, CLASS_WIDTH, LEAKY_COLUMNS, N_CLASS, PARAMS, ROUNDS,
    bayes_decision, make_objective, one_hot_targets, soft_targets,
)
from m271_evaluate_candidate import official                        # noqa: E402
from run_sequence_classifier import _surface                        # noqa: E402
from baram.constants import CAPACITIES_KWH                          # noqa: E402

ARCHIVED = {"BAND": {"Q2": 0.5965692367188797, "Q3": 0.5857480019682426, "Q4": 0.5954467020779709},
            "CONTROL": {"Q2": 0.5807629486268431}}
ORIG_EFFECT = 0.021490
ACTIONS = np.round(np.arange(0.075, 1.076, 0.0025), 6)
TEMPS = (0.5, 0.75, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0)
GAMMAS = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_ERR = np.abs(ACTIONS[:, None] - CENTERS[None, :])
_UNITS = np.select([_ERR <= 0.06, _ERR <= 0.08], [4.0, 3.0], default=0.0)
_SETTLE_M = (CENTERS[None, :] * _UNITS).T


def decide(prob, group, T, G, norms):
    cal = np.power(np.clip(prob, 1e-12, None), 1.0 / T); cal /= cal.sum(1, keepdims=True)
    settle, base = cal @ _SETTLE_M, -(cal @ _ERR.T)
    out = np.empty(len(cal))
    for gid in np.unique(group):
        m = group == gid
        out[m] = ACTIONS[np.argmax(base[m] + G * settle[m] / (4.0 * norms[int(gid)]), axis=1)]
    return out


def main() -> int:
    surface, _, _ = _surface()
    surface["forecast_kst_dtm"] = pd.to_datetime(surface["forecast_kst_dtm"])
    wanted = json.loads((PROBE / "M115_XGBOOST-dev-2023-Q3.json").read_text(encoding="utf-8"))["selected_feature_names"]
    features = [c for c in wanted if c in surface.columns and c not in LEAKY_COLUMNS]
    surface["capacity"] = surface["group_id"].map(CAPACITIES_KWH).astype(float)
    surface["rate"] = surface["actual_kwh"] / surface["capacity"]
    surface = surface.loc[surface["rate"].notna()].reset_index(drop=True)

    cache, fits = {}, 0
    for probe_fold, meta in fold_rows().items():
        train = surface.loc[surface["forecast_kst_dtm"] < meta["start"]]
        tmask = np.array([(f, g) in meta["keys"] for f, g in zip(surface["forecast_id"], surface["group_id"], strict=True)])
        test = surface.loc[tmask]
        x_tr = train.loc[:, features].astype("float32")
        rate = np.clip(train["rate"].to_numpy(float), 0.0, None)
        label = np.clip((rate / CLASS_WIDTH).astype(int), 0, N_CLASS - 1)
        for name, target in (("BAND", soft_targets(rate)), ("CONTROL", one_hot_targets(rate))):
            ds = lgb.Dataset(x_tr, label=label, free_raw_data=False)
            p = dict(PARAMS); p["objective"] = make_objective(target)
            raw = np.asarray(lgb.train(p, ds, num_boost_round=ROUNDS)
                             .predict(test.loc[:, features].astype("float32"))).reshape(len(test), N_CLASS)
            fits += 1
            e = np.exp(raw - raw.max(1, keepdims=True)); prob = e / e.sum(1, keepdims=True)
            cache.setdefault(name, {})[probe_fold] = dict(
                prob=prob, group=test["group_id"].to_numpy(int), cap=test["capacity"].to_numpy(float),
                rate=test["rate"].to_numpy(float), meta=test.loc[:, [*KEYS, "actual_kwh"]].copy())
        print(f"  fold {probe_fold} 적합 완료 (누적 {fits})")

    # --- V1 재현 ---
    repro, v1 = {}, True
    for arm in ("BAND", "CONTROL"):
        for f, c in cache[arm].items():
            fr = c["meta"].copy(); fr["prediction_kwh"] = bayes_decision(c["prob"]) * c["cap"]
            repro.setdefault(arm, {})[f] = float(official(fr)["total"])
    for arm, exp in ARCHIVED.items():
        for f, v in exp.items():
            got = repro[arm].get(f)
            ok = got is not None and abs(got - v) < 1e-9
            v1 &= ok
            print(f"  V1 {arm} {f}: {got!r} vs {v} -> {ok}")
    print(f"V1 재현 -> {v1}")

    def pooled(arm, policy_by_fold):
        parts = []
        for f, c in cache[arm].items():
            T, G = policy_by_fold[f]
            norms = {int(g): float(np.mean(c["rate"][c["group"] == g])) for g in np.unique(c["group"])}
            fr = c["meta"].copy(); fr["prediction_kwh"] = decide(c["prob"], c["group"], T, G, norms) * c["cap"]
            parts.append(fr)
        return official(pd.concat(parts, ignore_index=True))

    def fold_score(arm, f, T, G):
        c = cache[arm][f]
        norms = {int(g): float(np.mean(c["rate"][c["group"] == g])) for g in np.unique(c["group"])}
        fr = c["meta"].copy(); fr["prediction_kwh"] = decide(c["prob"], c["group"], T, G, norms) * c["cap"]
        return float(official(fr)["total"])

    folds = list(cache["BAND"])
    chosen, res = {}, {}
    for arm in ("BAND", "CONTROL"):
        pol = {}
        for f in folds:
            others = [o for o in folds if o != f]
            best, bp = -np.inf, None
            for T in TEMPS:
                for G in GAMMAS:
                    v = float(np.mean([fold_score(arm, o, T, G) for o in others]))
                    if v > best: best, bp = v, (T, G)
            pol[f] = bp
        chosen[arm] = pol
        sc = pooled(arm, pol)
        res[arm] = dict(total=float(sc["total"]), one_minus_nmae=float(sc["one_minus_nmae"]), ficr=float(sc["ficr"]))
        print(f"  {arm} fold-외 최적정책 {pol} -> pooled {sc['total']:.6f}")

    flat = {arm: float(official(pd.concat(
        [c["meta"].assign(prediction_kwh=bayes_decision(c["prob"]) * c["cap"]) for c in cache[arm].values()],
        ignore_index=True))["total"]) for arm in ("BAND", "CONTROL")}
    eff_flat = flat["BAND"] - flat["CONTROL"]
    eff_opt = res["BAND"]["total"] - res["CONTROL"]["total"]
    tb = float(np.mean([chosen["BAND"][f][0] for f in folds]))
    tc = float(np.mean([chosen["CONTROL"][f][0] for f in folds]))

    print(f"\n평탄규칙 처리효과   BAND-CONTROL = {eff_flat:+.6f}  (아카이브 사이클40 +0.009772)")
    print(f"팔별최적 처리효과   BAND-CONTROL = {eff_opt:+.6f}")
    print(f"H1 처리효과 축소            -> {eff_opt < ORIG_EFFECT}  ({eff_opt:+.6f} < {ORIG_EFFECT})")
    print(f"H2 축소폭 > 50%             -> {eff_opt < ORIG_EFFECT / 2}  (기준 {ORIG_EFFECT/2:+.6f})")
    print(f"H3 T_CONTROL > T_BAND       -> {tc > tb}  (CONTROL {tc:.2f} vs BAND {tb:.2f})")
    verdict = ("BAND_GAIN_IS_CALIBRATION_ARTIFACT" if eff_opt < ORIG_EFFECT / 2
               else "BAND_GAIN_SURVIVES_POLICY_OPTIMISATION")
    print(f"\n판정: {verdict}")

    (ROOT / "reports/m278_c1n41_rejudge_receipt.json").write_text(json.dumps(dict(
        node="C1N41_REJUDGE_M278", model_fits=fits, v1_reproduction=v1, repro=repro,
        flat_rule=flat, flat_effect=eff_flat, optimised=res, optimised_effect=eff_opt,
        policies={a: {f: list(p) for f, p in chosen[a].items()} for a in chosen},
        mean_T={"BAND": tb, "CONTROL": tc}, verdict=verdict,
        predeclared=dict(V1="archive reproduce 1e-9", H1="effect shrinks", H2="shrink>50%", H3="T_CONTROL>T_BAND"),
        lockbox_reopened=False, dacon_upload=False, external_actions=[]), indent=1, ensure_ascii=False))
    print("영수증 -> reports/m278_c1n41_rejudge_receipt.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
