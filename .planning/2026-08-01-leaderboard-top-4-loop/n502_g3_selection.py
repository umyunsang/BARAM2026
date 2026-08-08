
"""N502 — 선택 목적함수에서 group_3 를 제외하면 fold-외 선택이 나아지는가 (적합 0회).

## admission 근거
`C1N12_G3_HISTORY` 는 **인과 귀속**("이력 길이가 g3 열세를 설명하는가")을 물었고 기각됐다.
본 노드는 **선택 규칙**을 묻는다 — 다른 질문이다. 근거:
- 배포 fit 의 g3 학습행 17,537 대 로컬 fold 의 2,159 / 4,343 / 6,551 (2.7~8.1 배 차)
- 즉 선택에 쓰이는 g3 신호는 배포와 **전혀 다른 데이터 체제**에서 측정된다
- C1N12 리포트 자체가 "폴드 3 개뿐이라 단조성은 약한 증거이고 계절과 이력이 교락" 이라고 단서를 단다

## 사전확약 (실행 전 동결)
- V1  배포 T0.5_G1.5 가 0.628337 재현
- H1  팔 B(g1+g2 로 선택) > 팔 A(전체로 선택), 둘 다 전체 3 그룹으로 채점
- H2  이득 > 0.001013
- H3  3/3 폴드 개선
- H4  동결 월별 게이트 통과

**반증**: H1 이 거짓이면 g3 를 선택에서 빼는 것은 도움이 안 되고 축을 닫는다.
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
from baram.evaluation.official import evaluate_official, evaluate_group_component  # noqa: E402
from baram.constants import CAPACITIES_KWH  # noqa: E402

SRC = ROOT / "artifacts/backtests/m269-probe"
FOLDS = ("dev-2023-Q2", "dev-2023-Q3", "dev-2023-Q4")
ACTIONS = np.round(np.arange(0.075, 1.076, 0.0025), 6)
DEP, DEP_TOTAL, THRESH = (0.5, 1.5), 0.628337, 0.001013
TEMPS = (0.4, 0.5, 0.6, 0.75, 0.85, 1.0, 1.2)
GAMMAS = (0.0, 0.2, 0.35, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)


def load():
    st = {}
    for f in FOLDS:
        z = np.load(SRC / f"M269_PROBE-{f}-probability.npz", allow_pickle=True)
        cap = np.array([CAPACITIES_KWH[int(g)] for g in z["group_id"]], float)
        st[f] = dict(prob=z["probability"], centers=z["centers"], group=z["group_id"].astype(int),
                     cap=cap, actual=z["actual_kwh"].astype(float), dtm=pd.to_datetime(z["forecast_kst_dtm"]))
    return st


def frame_for(c, T, G, tag):
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
    return pd.DataFrame(dict(forecast_id=[f"{tag}-{i}" for i in range(len(cal))],
                             forecast_kst_dtm=c["dtm"], group_id=c["group"],
                             actual_kwh=c["actual"], prediction_kwh=yhat * c["cap"]))


def full_total(st, T, G, folds):
    fr = pd.concat([frame_for(st[f], T, G, f) for f in folds], ignore_index=True)
    return float(evaluate_official(fr, CAPACITIES_KWH).total)


def partial_total(st, T, G, folds, groups):
    """선택 전용 목적: 지정 그룹의 component total 평균."""
    fr = pd.concat([frame_for(st[f], T, G, f) for f in folds], ignore_index=True)
    vals = []
    for g in groups:
        part = fr.loc[fr["group_id"] == g]
        gs = evaluate_group_component(part, g, CAPACITIES_KWH[g])
        # component total in the official algebra: 0.5*(1-nmae) + 0.5*ficr
        vals.append(0.5 * (1.0 - float(gs.nmae)) + 0.5 * float(gs.ficr))
    return float(np.mean(vals))


def main() -> int:
    st = load()
    dep = full_total(st, *DEP, FOLDS)
    v1 = abs(dep - DEP_TOTAL) < 1e-5
    print(f"V1 배포 재현 -> {dep:.6f} : {v1}")

    res, pol, fw = {}, {}, {}
    for tag, sel_groups in (("A_all3", (1, 2, 3)), ("B_g12", (1, 2))):
        p, ft = {}, {}
        for f in FOLDS:
            others = [o for o in FOLDS if o != f]
            p[f] = max(((T, G) for T in TEMPS for G in GAMMAS),
                       key=lambda k: partial_total(st, k[0], k[1], others, sel_groups))
            ft[f] = full_total(st, *p[f], [f])
        fr = pd.concat([frame_for(st[f], *p[f], f) for f in FOLDS], ignore_index=True)
        s = evaluate_official(fr, CAPACITIES_KWH)
        res[tag] = (float(s.total), float(s.one_minus_nmae), float(s.ficr)); pol[tag], fw[tag] = p, ft
        print(f"  {tag:7s} 선택그룹 {sel_groups}  fold-외 { {f[-2:]: p[f] for f in FOLDS} } -> pooled {s.total:.6f}")

    gain = res["B_g12"][0] - res["A_all3"][0]
    h3 = all(fw["B_g12"][f] > fw["A_all3"][f] for f in FOLDS)
    cand = pd.concat([frame_for(st[f], *pol["B_g12"][f], f) for f in FOLDS], ignore_index=True)
    par = pd.concat([frame_for(st[f], *pol["A_all3"][f], f) for f in FOLDS], ignore_index=True)
    for fr in (cand, par): fr["month"] = fr["forecast_kst_dtm"].dt.to_period("M").astype(str)
    g = evaluate_gate(cand, par)
    print(f"\n폴드별 " + "  ".join(f"{f[-2:]} {fw['B_g12'][f]-fw['A_all3'][f]:+.6f}" for f in FOLDS))
    print(f"H1 B > A        -> {gain > 0}  ({gain:+.6f})")
    print(f"H2 > 검출문턱   -> {gain > THRESH}")
    print(f"H3 3/3 폴드     -> {h3}")
    for k, ok in g.conditions.items(): print(f"  [{'PASS' if ok else 'FAIL'}] {k}")
    print(f"H4 월별 게이트  -> {g.passed}")
    print(f"배포 대비: A {res['A_all3'][0]-DEP_TOTAL:+.6f} / B {res['B_g12'][0]-DEP_TOTAL:+.6f}")
    verdict = "G3_EXCLUSION_FROM_SELECTION_HELPS" if (gain > THRESH and h3 and g.passed) else "G3_EXCLUSION_DOES_NOT_HELP"
    print(f"\n판정: {verdict}")
    (ROOT / "reports/n502_g3_selection_receipt.json").write_text(json.dumps(dict(
        node="N502_G3_SELECTION", gate_version=GATE_VERSION, v1_reproduction=bool(v1),
        deployed_total=DEP_TOTAL, results=res, gain=gain, threshold=THRESH, foldwise=fw,
        policies={k: {f: list(v) for f, v in pol[k].items()} for k in pol},
        gate={k: bool(v) for k, v in g.conditions.items()},
        hypotheses=dict(H1=bool(gain > 0), H2=bool(gain > THRESH), H3=bool(h3), H4=bool(g.passed)),
        verdict=verdict, model_fits=0, lockbox_reopened=False, dacon_upload=False,
        external_actions=[]), indent=1, ensure_ascii=False))
    print("영수증 -> reports/n502_g3_selection_receipt.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
