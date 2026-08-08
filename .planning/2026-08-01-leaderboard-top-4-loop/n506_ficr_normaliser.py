
"""N506 — FICR 정규화 상수: 전체평균(배포) vs 적격행평균(정의) (적합 0회).

## 발견
배포 결정규칙 `run_site_wind_classifier.py` 는 정산항을 `4 * mean_generation[g]` 로 나누고
`mean_generation` 은 **학습행 전체 평균**이다(적격 필터 없음).
그러나 FICR 분모는 `sum(actual*4)` = `4 * N_g * mean(actual | actual >= 0.10C)` 이므로
정의상 맞는 상수는 **적격행 평균**이다. `src/baram/decisions/expected_utility.py` 는 그렇게 한다.

전체평균은 적격행평균보다 작고(적격비율 약 60%), 그 비율이 **그룹마다 다르다**:
라벨 실측 기준 적격/전체 평균비 g1 1.59 / g2 1.60 / g3 1.79.
전역 gamma 는 평균적 축척은 흡수할 수 있으나 **그룹간 상대 오차는 흡수하지 못한다.**

## 사전확약 (실행 전 동결)
- V1  팔 A(전체평균) 배포정책 T0.5_G1.5 가 0.628337 재현
- V2  그룹별 적격/전체 평균비가 서로 5% 이상 다르다 (그룹간 왜곡이 실재)
- H1  팔 B(적격행평균) > 팔 A, 둘 다 fold-외 (T,G) 선택
- H2  이득 > 0.001013
- H3  3/3 폴드 개선
- H4  동결 월별 게이트 통과

**반증**: H1 이 거짓이면 정규화 상수 차이는 전역 gamma 에 흡수되며 축을 닫는다.
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
DEP, DEP_TOTAL, THRESH, ELIG = (0.5, 1.5), 0.628337, 0.001013, 0.10
TEMPS = (0.3, 0.4, 0.5, 0.6, 0.75, 0.85, 1.0, 1.2, 1.5)
GAMMAS = (0.0, 0.2, 0.35, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0)


def load():
    st = {}
    for f in FOLDS:
        z = np.load(SRC / f"M269_PROBE-{f}-probability.npz", allow_pickle=True)
        cap = np.array([CAPACITIES_KWH[int(g)] for g in z["group_id"]], float)
        st[f] = dict(prob=z["probability"], centers=z["centers"], group=z["group_id"].astype(int),
                     cap=cap, actual=z["actual_kwh"].astype(float), dtm=pd.to_datetime(z["forecast_kst_dtm"]))
    return st


def norms_for(c, eligible_only):
    rate = c["actual"] / c["cap"]
    out = {}
    for g in np.unique(c["group"]):
        m = c["group"] == g
        r = rate[m]
        if eligible_only:
            r = r[r >= ELIG]
        out[int(g)] = float(np.mean(r))
    return out


def frame_for(c, T, G, eligible_only, tag):
    C = c["centers"]
    err = np.abs(ACTIONS[:, None] - C[None, :])
    units = np.select([err <= 0.06, err <= 0.08], [4.0, 3.0], default=0.0)
    nm = norms_for(c, eligible_only)
    cal = np.power(np.clip(c["prob"], 1e-12, None), 1.0 / T); cal /= cal.sum(1, keepdims=True)
    base, settle = -(cal @ err.T), cal @ (C[None, :] * units).T
    yh = np.empty(len(cal))
    for gid in np.unique(c["group"]):
        m = c["group"] == gid
        yh[m] = ACTIONS[np.argmax(base[m] + G * settle[m] / (4.0 * nm[int(gid)]), axis=1)]
    return pd.DataFrame(dict(forecast_id=[f"{tag}-{i}" for i in range(len(cal))],
                             forecast_kst_dtm=c["dtm"], group_id=c["group"],
                             actual_kwh=c["actual"], prediction_kwh=yh * c["cap"]))


def score(st, T, G, eo, folds=FOLDS):
    fr = pd.concat([frame_for(st[f], T, G, eo, f) for f in folds], ignore_index=True)
    s = evaluate_official(fr, CAPACITIES_KWH)
    return float(s.total), float(s.one_minus_nmae), float(s.ficr)


def main() -> int:
    st = load()
    a = score(st, *DEP, False)
    v1 = abs(a[0] - DEP_TOTAL) < 1e-5
    print(f"V1 팔A 배포정책 -> {a[0]:.6f} (기대 {DEP_TOTAL}) : {v1}")

    ratios = {}
    for f in FOLDS:
        nall, nel = norms_for(st[f], False), norms_for(st[f], True)
        ratios[f] = {g: nel[g] / nall[g] for g in nall}
    print("\n그룹별 적격/전체 평균비:")
    for f in FOLDS:
        print(f"  {f[-2:]}: " + "  ".join(f"g{g} {v:.4f}" for g, v in sorted(ratios[f].items())))
    spread = max(max(r.values()) - min(r.values()) for r in ratios.values())
    v2 = spread >= 0.05
    print(f"V2 그룹간 비율 최대차 {spread:.4f} >= 0.05 -> {v2}")
    print(f"   같은 규칙 + 적격행평균 -> {score(st, *DEP, True)[0]:.6f}")

    res, pol, fw = {}, {}, {}
    for tag, eo in (("A_all", False), ("B_eligible", True)):
        p, ft = {}, {}
        for f in FOLDS:
            others = [o for o in FOLDS if o != f]
            p[f] = max(((T, G) for T in TEMPS for G in GAMMAS),
                       key=lambda k: score(st, k[0], k[1], eo, others)[0])
            ft[f] = score(st, *p[f], eo, [f])[0]
        fr = pd.concat([frame_for(st[f], *p[f], eo, f) for f in FOLDS], ignore_index=True)
        s = evaluate_official(fr, CAPACITIES_KWH)
        res[tag] = (float(s.total), float(s.one_minus_nmae), float(s.ficr)); pol[tag], fw[tag] = p, ft
        print(f"  {tag:11s} fold-외 { {f[-2:]: p[f] for f in FOLDS} } -> pooled {s.total:.6f}")

    gain = res["B_eligible"][0] - res["A_all"][0]
    h3 = all(fw["B_eligible"][f] > fw["A_all"][f] for f in FOLDS)
    cand = pd.concat([frame_for(st[f], *pol["B_eligible"][f], True, f) for f in FOLDS], ignore_index=True)
    par = pd.concat([frame_for(st[f], *pol["A_all"][f], False, f) for f in FOLDS], ignore_index=True)
    for fr in (cand, par): fr["month"] = fr["forecast_kst_dtm"].dt.to_period("M").astype(str)
    g = evaluate_gate(cand, par)
    print(f"\n폴드별 " + "  ".join(f"{f[-2:]} {fw['B_eligible'][f]-fw['A_all'][f]:+.6f}" for f in FOLDS))
    print(f"H1 B > A       -> {gain > 0}  ({gain:+.6f})")
    print(f"H2 > 검출문턱  -> {gain > THRESH}")
    print(f"H3 3/3 폴드    -> {h3}")
    for k, ok in g.conditions.items(): print(f"  [{'PASS' if ok else 'FAIL'}] {k}")
    print(f"H4 월별 게이트 -> {g.passed}")
    print(f"배포 대비: A {res['A_all'][0]-DEP_TOTAL:+.6f} / B {res['B_eligible'][0]-DEP_TOTAL:+.6f}")
    verdict = "ELIGIBLE_NORMALISER_HELPS" if (gain > THRESH and h3 and g.passed) else "ELIGIBLE_NORMALISER_ABSORBED_BY_GAMMA"
    print(f"\n판정: {verdict}")
    (ROOT / "reports/n506_ficr_normaliser_receipt.json").write_text(json.dumps(dict(
        node="N506_FICR_NORMALISER", gate_version=GATE_VERSION, v1_reproduction=bool(v1),
        v2_group_spread=float(spread), ratios={k: {str(g2): float(v) for g2, v in r.items()} for k, r in ratios.items()},
        results=res, gain=gain, threshold=THRESH, foldwise=fw,
        policies={k: {f: list(v) for f, v in pol[k].items()} for k in pol},
        gate={k: bool(v) for k, v in g.conditions.items()},
        hypotheses=dict(H1=bool(gain > 0), H2=bool(gain > THRESH), H3=bool(h3), H4=bool(g.passed)),
        verdict=verdict, model_fits=0, lockbox_reopened=False, dacon_upload=False,
        external_actions=[]), indent=1, ensure_ascii=False))
    print("영수증 -> reports/n506_ficr_normaliser_receipt.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
