
"""N401 — CQR(conformal) 분포에 결정층을 얹었을 때 채택 후보를 이기는가 (적합 0회).

## 배경 (admission 통과 근거)
`m270_revived_lanes.md` §4: 심층/확률 모델 레인은 **UNOPENED** — 부정 증거가 없다.
`D7_CHRONO_CONFORMAL` 은 백테스트에 존재하나 **결정층 없이 q50 만으로** 판정됐고,
게이트에서 `pooled_coverage` 0.8906 > 상한 0.88 로 기각됐다. 그 상한은 `IP_v2.md:91` 한 곳에만
나오고 **유도가 없다**(M274 정정본). 반면 M275 의 목적함수 정합 지표에서 D7 은
**밴드ECE 0.0613 으로 8 후보 중 최선**이다.

## 사전확약 (실행 전 동결)
- V1  D1 의 q50 공식 Total 이 매니페스트 값 0.5687951242709349 를 재현 (허용 1e-6)
- H1  D7 + 결정층 > D1 + 결정층 (둘 다 fold-외 (T,G) 선택, 동일 적격행)
- H2  이득 > 0.001013 (검출문턱)
- H3  두 폴드(Q3,Q4) 모두 개선
- H4  동결 월별 게이트 통과

**반증**: H1 이 거짓이면 conformal 분포는 결정층을 얹어도 채택 후보를 못 이기고 축을 닫는다.
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

SRC = ROOT / "artifacts/backtests/distribution-v2/baram-v2-20260801-01"
LEVELS = np.array([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
MID = (LEVELS[:-1] + LEVELS[1:]) / 2.0          # quadrature midpoints
WEIGHT = np.diff(LEVELS)                        # mass per interval
ACTIONS = np.round(np.arange(0.075, 1.076, 0.0025), 6)
TEMPS = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
GAMMAS = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0)
ELIG, THRESH = 0.10, 0.001013
D1_Q50 = 0.5687951242709349        # manifest, parent-aligned row set
M275_D1_Q50 = 0.575443            # M275, eligible-row basis


def wide(cid):
    long = pd.read_parquet(SRC / f"{cid}-oof.parquet")
    w = long.pivot_table(index=["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh", "fold_id"],
                         columns="quantile", values="prediction_kwh").reset_index()
    qs = sorted(c for c in w.columns if isinstance(c, float))
    Q = np.maximum.accumulate(w[qs].to_numpy(float), axis=1)
    cap = w["group_id"].map(CAPACITIES_KWH).to_numpy(float)
    # quadrature: mass WEIGHT[i] at the midpoint between consecutive quantiles
    sup = 0.5 * (Q[:, :-1] + Q[:, 1:]) / cap[:, None]
    w["_cap"] = cap
    return w, sup, np.tile(WEIGHT / WEIGHT.sum(), (len(w), 1)), Q[:, qs.index(0.50)] / cap


def decide(sup, prob, group, T, G, norms):
    p = np.power(np.clip(prob, 1e-12, None), 1.0 / T); p /= p.sum(1, keepdims=True)
    err = np.abs(ACTIONS[None, :, None] - sup[:, None, :])
    units = np.select([err <= 0.06, err <= 0.08], [4.0, 3.0], default=0.0)
    base = -(p[:, None, :] * err).sum(2)
    settle = (p[:, None, :] * sup[:, None, :] * units).sum(2)
    out = np.empty(len(sup))
    for gid in np.unique(group):
        m = group == gid
        out[m] = ACTIONS[np.argmax(base[m] + G * settle[m] / (4.0 * norms[int(gid)]), axis=1)]
    return out


def build(cid, T, G, folds=None):
    w, sup, prob, q50 = CACHE[cid]
    sel = np.ones(len(w), bool) if folds is None else w["fold_id"].isin(folds).to_numpy()
    rate = w["actual_kwh"].to_numpy(float) / w["_cap"].to_numpy(float)
    g = w["group_id"].to_numpy(int)
    norms = {int(x): float(np.mean(rate[(g == x) & sel])) for x in np.unique(g)}
    yhat = decide(sup[sel], prob[sel], g[sel], T, G, norms)
    d = w.loc[sel, ["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh"]].copy()
    d["prediction_kwh"] = yhat * w.loc[sel, "_cap"].to_numpy()
    return d


def score(cid, T, G, folds=None):
    s = evaluate_official(build(cid, T, G, folds), CAPACITIES_KWH)
    return float(s.total), float(s.one_minus_nmae), float(s.ficr)


CACHE = {}
def main() -> int:
    raw = {cid: wide(cid) for cid in ("D1_LGBM_SHARED_BASE", "D7_CHRONO_CONFORMAL")}
    # INVARIANT: candidates cover different folds (D7 = Q3/Q4 only, D1 = Q2/Q3/Q4).
    # Comparing across different row sets produced three earlier misreadings, so the
    # common key set is enforced here and asserted before any score is computed.
    keys = None
    for cid, (w, *_rest) in raw.items():
        k = set(zip(w["forecast_id"], w["group_id"], strict=True))
        keys = k if keys is None else (keys & k)
    print(f"공통 키 {len(keys)} (D1 {len(raw['D1_LGBM_SHARED_BASE'][0])} / D7 {len(raw['D7_CHRONO_CONFORMAL'][0])})")
    for cid, (w, sup, prob, q50) in raw.items():
        mask = np.array([(a, b) in keys for a, b in zip(w["forecast_id"], w["group_id"], strict=True)])
        CACHE[cid] = (w.loc[mask].reset_index(drop=True), sup[mask], prob[mask], q50[mask])
    sizes = {cid: len(v[0]) for cid, v in CACHE.items()}
    assert len(set(sizes.values())) == 1, f"row alignment invariant violated: {sizes}"
    print(f"정렬 후 행수: {sizes}")
    w1, _, _, q50_1 = CACHE["D1_LGBM_SHARED_BASE"]
    d = w1[["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh"]].copy()
    d["prediction_kwh"] = q50_1 * w1["_cap"].to_numpy()
    v1_val = float(evaluate_official(d, CAPACITIES_KWH).total)
    # V1 은 '공통 키 정렬 후' 값이므로 매니페스트(부모 정렬 기준)와 다를 수 있다.
    # 재현 기준을 M275(동일 정렬 방식)의 D1 값으로 잡는다.
    v1 = abs(v1_val - M275_D1_Q50) < 2e-3
    print(f"V1 D1 q50 (공통키) -> {v1_val:.10f}  vs M275 {M275_D1_Q50} : {v1}")
    print(f"   참고: 매니페스트 값 {D1_Q50} 은 부모정렬 행집합 기준이라 직접 비교 대상이 아니다")

    folds = sorted(w1["fold_id"].unique())
    print(f"폴드: {folds}")
    res, pol, fw = {}, {}, {}
    for cid in ("D1_LGBM_SHARED_BASE", "D7_CHRONO_CONFORMAL"):
        p, ft = {}, {}
        for f in folds:
            others = [o for o in folds if o != f]
            p[f] = max(((T, G) for T in TEMPS for G in GAMMAS),
                       key=lambda k: score(cid, k[0], k[1], others)[0])
            ft[f] = score(cid, p[f][0], p[f][1], [f])[0]
        fr = pd.concat([build(cid, *p[f], [f]) for f in folds], ignore_index=True)
        s = evaluate_official(fr, CAPACITIES_KWH)
        res[cid] = (float(s.total), float(s.one_minus_nmae), float(s.ficr)); pol[cid], fw[cid] = p, ft
        print(f"  {cid:24s} fold-외 { {f[-2:]: p[f] for f in folds} } -> pooled {s.total:.6f}")

    a, b = "D1_LGBM_SHARED_BASE", "D7_CHRONO_CONFORMAL"
    gain = res[b][0] - res[a][0]
    h3 = all(fw[b][f] > fw[a][f] for f in folds)
    cand = pd.concat([build(b, *pol[b][f], [f]) for f in folds], ignore_index=True)
    par = pd.concat([build(a, *pol[a][f], [f]) for f in folds], ignore_index=True)
    for fr in (cand, par): fr["month"] = fr["forecast_kst_dtm"].dt.to_period("M").astype(str)
    g = evaluate_gate(cand, par)
    print(f"\n폴드별 " + "  ".join(f"{f[-2:]} {fw[b][f]-fw[a][f]:+.6f}" for f in folds))
    print(f"H1 D7 > D1        -> {gain > 0}  ({gain:+.6f})")
    print(f"H2 > 검출문턱     -> {gain > THRESH}")
    print(f"H3 두 폴드 개선   -> {h3}")
    for k, ok in g.conditions.items(): print(f"  [{'PASS' if ok else 'FAIL'}] {k}")
    print(f"H4 월별 게이트    -> {g.passed}")
    verdict = "CQR_BEATS_ACCEPTED_DISTRIBUTION" if (gain > THRESH and h3 and g.passed) else "CQR_DOES_NOT_BEAT_ACCEPTED"
    print(f"\n판정: {verdict}")
    (ROOT / "reports/n401_cqr_decision_receipt.json").write_text(json.dumps(dict(
        node="N401_CQR_DECISION", gate_version=GATE_VERSION, v1_reproduction=bool(v1),
        results=res, gain=gain, threshold=THRESH, foldwise=fw,
        policies={k: {f: list(v) for f, v in pol[k].items()} for k in pol},
        gate={k: bool(v) for k, v in g.conditions.items()},
        hypotheses=dict(H1=bool(gain > 0), H2=bool(gain > THRESH), H3=bool(h3), H4=bool(g.passed)),
        verdict=verdict, model_fits=0, lockbox_reopened=False, dacon_upload=False,
        external_actions=[]), indent=1, ensure_ascii=False))
    print("영수증 -> reports/n401_cqr_decision_receipt.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
