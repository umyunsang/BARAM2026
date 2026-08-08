
"""M286 — 온라인 오프셋 모형(가법 vs 승법)에 따른 최적 정책 재선택 (적합 0회).

## 근거
M261 앵커: local (1-NMAE 0.856805, FICR 0.403142) -> online (0.857885, 0.415169)
  - 1-NMAE 오프셋 +0.001080 (거의 무변)
  - FICR   오프셋 +0.012027  (비율 1.029833)

오프셋이 **가법 상수**면 모든 정책에 동일하게 더해지므로 argmax 가 안 바뀐다.
그러나 **승법**(FICR 이 비례 증폭)이면 로컬 FICR 이 높은 정책이 더 큰 절대이득을 받아
**최적 정책이 이동한다.** 프로젝트는 재채점만 했고 이 재선택을 한 적이 없다.

## 사전확약 (실행 전 동결)
- V1  배포 T0.5_G1.5 가 0.628337 재현
- H1  승법 모형의 최적 정책이 로컬 최적 정책과 **다르다**
- H2  승법 최적 정책의 예측 온라인 Total 이 배포의 예측 온라인 Total 을 초과
- H3  그 정책이 fold-외 선택에서도 배포를 초과 (선택편향 배제)
- H4  그 정책이 동결 월별 게이트를 통과

**반증**: H1 이 거짓이면 오프셋 모형은 정책 선택을 바꾸지 않고 축을 닫는다.
**주의**: 앵커 n=1 이므로 승법 가정 자체가 미검증이다. H1~H2 는 가정 하 결과로만 읽는다.

락박스 미접근 / 적합 0회 / 제출 없음.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / "src"))
from m270_gate import GATE_VERSION, evaluate_gate  # noqa: E402
from baram.evaluation.official import evaluate_official  # noqa: E402
from baram.constants import CAPACITIES_KWH  # noqa: E402

SRC = ROOT / "artifacts/backtests/m269-probe"
FOLDS = ("dev-2023-Q2", "dev-2023-Q3", "dev-2023-Q4")
ACTIONS = np.round(np.arange(0.075, 1.076, 0.0025), 6)
DEP = (0.5, 1.5); DEP_TOTAL = 0.628337
NMAE_ADD = 0.001080
FICR_ADD, FICR_MUL = 0.012027, 0.415169 / 0.403142
TEMPS = (0.4, 0.5, 0.6, 0.75, 0.85, 1.0, 1.2, 1.6, 2.0)
GAMMAS = (0.0, 0.2, 0.35, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0)


def load():
    st = {}
    for f in FOLDS:
        z = np.load(SRC / f"M269_PROBE-{f}-probability.npz", allow_pickle=True)
        cap = np.array([CAPACITIES_KWH[int(g)] for g in z["group_id"]], float)
        st[f] = dict(prob=z["probability"], centers=z["centers"], group=z["group_id"].astype(int),
                     cap=cap, actual=z["actual_kwh"].astype(float), dtm=pd.to_datetime(z["forecast_kst_dtm"]))
    return st


def frame_for(c, T, G):
    C = c["centers"]
    err = np.abs(ACTIONS[:, None] - C[None, :])
    units = np.select([err <= 0.06, err <= 0.08], [4.0, 3.0], default=0.0)
    rate = c["actual"] / c["cap"]
    norms = {int(g): float(np.mean(rate[c["group"] == g])) for g in np.unique(c["group"])}
    cal = np.power(np.clip(c["prob"], 1e-12, None), 1.0 / T); cal /= cal.sum(1, keepdims=True)
    base, settle = -(cal @ err.T), cal @ (C[None, :] * units).T
    yhat = np.empty(len(cal))
    for gid in np.unique(c["group"]):
        m = c["group"] == gid
        yhat[m] = ACTIONS[np.argmax(base[m] + G * settle[m] / (4.0 * norms[int(gid)]), axis=1)]
    d = pd.DataFrame(dict(forecast_id=[f"{f_id}-{i}" for f_id, i in zip([id(c)] * len(cal), range(len(cal)))],
                          forecast_kst_dtm=c["dtm"], group_id=c["group"],
                          actual_kwh=c["actual"], prediction_kwh=yhat * c["cap"]))
    return d


def comp(st, T, G, folds=FOLDS):
    s = evaluate_official(pd.concat([frame_for(st[f], T, G) for f in folds], ignore_index=True), CAPACITIES_KWH)
    return float(s.total), float(s.one_minus_nmae), float(s.ficr)


def main() -> int:
    st = load()
    d = comp(st, *DEP)
    v1 = abs(d[0] - DEP_TOTAL) < 1e-5
    print(f"V1 배포 재현 -> {d[0]:.6f} : {v1}   (1-NMAE {d[1]:.6f} / FICR {d[2]:.6f})")

    grid = {}
    for T in TEMPS:
        for G in GAMMAS:
            grid[(T, G)] = comp(st, T, G)
    def add_pred(v): return 0.5 * (v[1] + NMAE_ADD) + 0.5 * (v[2] + FICR_ADD)
    def mul_pred(v): return 0.5 * (v[1] + NMAE_ADD) + 0.5 * (v[2] * FICR_MUL)

    best_local = max(grid, key=lambda k: grid[k][0])
    best_add = max(grid, key=lambda k: add_pred(grid[k]))
    best_mul = max(grid, key=lambda k: mul_pred(grid[k]))
    print(f"\n로컬 Total 최적    T{best_local[0]:g}_G{best_local[1]:g}  local {grid[best_local][0]:.6f}")
    print(f"가법 예측온라인 최적 T{best_add[0]:g}_G{best_add[1]:g}  pred {add_pred(grid[best_add]):.6f}")
    print(f"승법 예측온라인 최적 T{best_mul[0]:g}_G{best_mul[1]:g}  pred {mul_pred(grid[best_mul]):.6f}")
    print(f"배포 T0.5_G1.5 예측온라인: 가법 {add_pred(d):.6f} / 승법 {mul_pred(d):.6f}")

    h1 = best_mul != best_local
    h2 = mul_pred(grid[best_mul]) > mul_pred(d)
    print(f"\nH1 승법최적 != 로컬최적 -> {h1}  ({best_mul} vs {best_local})")
    print(f"H2 승법최적 > 배포      -> {h2}  ({mul_pred(grid[best_mul])-mul_pred(d):+.6f})")

    h3 = h4 = False; fo_tot = None; gate_cond = {}
    if h1 and h2:
        pol, ft = {}, {}
        for f in FOLDS:
            others = [o for o in FOLDS if o != f]
            pol[f] = max(grid, key=lambda k: mul_pred(comp(st, k[0], k[1], folds=others)))
            ft[f] = comp(st, *pol[f], folds=[f])[0]
        cand = pd.concat([frame_for(st[f], *pol[f]) for f in FOLDS], ignore_index=True)
        parent = pd.concat([frame_for(st[f], *DEP) for f in FOLDS], ignore_index=True)
        for fr in (cand, parent): fr["month"] = fr["forecast_kst_dtm"].dt.to_period("M").astype(str)
        fo = evaluate_official(cand.drop(columns=["month"]), CAPACITIES_KWH)
        fo_tot = float(fo.total)
        h3 = fo_tot > DEP_TOTAL
        print(f"\nfold-외 승법 선택 { {f[-2:]: pol[f] for f in FOLDS} } -> local pooled {fo_tot:.6f}")
        print(f"H3 fold-외 > 배포 -> {h3}  ({fo_tot-DEP_TOTAL:+.6f})")
        g = evaluate_gate(cand, parent); h4 = bool(g.passed); gate_cond = {k: bool(v) for k, v in g.conditions.items()}
        for k, ok in g.conditions.items(): print(f"  [{'PASS' if ok else 'FAIL'}] {k}")
        print(f"H4 월별 게이트 -> {h4}")

    verdict = "OFFSET_MODEL_CHANGES_POLICY_AND_PASSES" if (h1 and h2 and h3 and h4) else \
              ("OFFSET_MODEL_CHANGES_POLICY_BUT_GATE_REJECTS" if (h1 and h2) else "OFFSET_MODEL_DOES_NOT_CHANGE_POLICY")
    print(f"\n판정: {verdict}")
    (ROOT / "reports/m286_offset_model_policy_receipt.json").write_text(json.dumps(dict(
        node="M286_OFFSET_MODEL_POLICY", v1_reproduction=bool(v1), gate_version=GATE_VERSION,
        deployed=d, best_local=[*best_local, *grid[best_local]], best_additive=[*best_add, *grid[best_add]],
        best_multiplicative=[*best_mul, *grid[best_mul]], ficr_multiplier=FICR_MUL,
        fold_outside_pooled=fo_tot, gate=gate_cond,
        hypotheses=dict(H1=bool(h1), H2=bool(h2), H3=bool(h3), H4=bool(h4)), verdict=verdict,
        caveat="anchor n=1; multiplicative assumption unverified",
        model_fits=0, lockbox_reopened=False, dacon_upload=False, external_actions=[]),
        indent=1, ensure_ascii=False))
    print("영수증 -> reports/m286_offset_model_policy_receipt.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
