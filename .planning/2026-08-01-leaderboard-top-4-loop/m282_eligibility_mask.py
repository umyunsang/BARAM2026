
"""M282 — 결정층 기대효용의 적격성 마스크 누락 (적합 0회, 파생 전용).

## 발견
배포 결정규칙의 효용은
    utility(x) = E_p[ -|x-c| ] + (gamma/norm) * E_p[ c * u(|x-c|) ]
로 **46 구간 전체**에 대해 계산된다. 그러나 공식 지표는 `actual >= 0.10 * capacity` 행만 채점한다.
중심 0.01/0.03/0.05/0.07/0.09 의 5 개 구간은 **어떤 점수에도 기여하지 않는데** 기댓값에 들어가
행동을 아래로 끌어당긴다. 올바른 목적은
    utility(x) = E_p[ 1(c >= 0.10) * ( -|x-c| + (gamma/norm) * c * u(|x-c|) ) ]
지시함수는 c(결과)에만 의존하고 x(행동)에는 의존하지 않으므로 argmax 가 바뀐다.

발굴그래프 136 노드 중 이 축을 다룬 노드는 없다 (`C1N23_ELIGIBLE_MOS` 는 MOS 편향, 본 건과 무관).

## 사전확약 (실행 전 동결)
- V1  배포 T0.5_G1.5 무마스크가 0.628337 재현 (M279 확인값, 허용 1e-4)
- H1  마스크 팔 pooled > 무마스크 팔 pooled (둘 다 fold-외 (T,G) 선택)
- H2  이득 > 0.001013 (프로젝트 검출문턱)
- H3  3/3 폴드에서 개선
- H4  마스크 팔이 배포(T0.5_G1.5) 대비 개선

**반증**: H1 이 거짓이면 적격성 마스킹은 실제 분포에서 이득이 없고 축을 닫는다.

표면: `artifacts/backtests/m269-probe/` (배포 계보). 락박스 미접근 / 적합 0회 / 제출 없음.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src"))
from baram.evaluation.official import evaluate_official  # noqa: E402
from baram.constants import CAPACITIES_KWH  # noqa: E402

SRC = ROOT / "artifacts/backtests/m269-probe"
FOLDS = ("dev-2023-Q2", "dev-2023-Q3", "dev-2023-Q4")
ACTIONS = np.round(np.arange(0.075, 1.076, 0.0025), 6)
DEP_T, DEP_G, DEP_TOTAL = 0.5, 1.5, 0.628337
TEMPS = (0.4, 0.5, 0.6, 0.75, 0.85, 1.0, 1.2, 1.6, 2.0)
GAMMAS = (0.0, 0.2, 0.35, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0)
ELIGIBLE = 0.10
THRESH = 0.001013


def load():
    st = {}
    for f in FOLDS:
        z = np.load(SRC / f"M269_PROBE-{f}-probability.npz", allow_pickle=True)
        cap = np.array([CAPACITIES_KWH[int(g)] for g in z["group_id"]], float)
        st[f] = dict(prob=z["probability"], centers=z["centers"], group=z["group_id"].astype(int),
                     cap=cap, actual=z["actual_kwh"].astype(float), dtm=pd.to_datetime(z["forecast_kst_dtm"]))
    return st


def frame_for(c, T, G, mask):
    C = c["centers"]
    err = np.abs(ACTIONS[:, None] - C[None, :])
    units = np.select([err <= 0.06, err <= 0.08], [4.0, 3.0], default=0.0)
    elig = (C >= ELIGIBLE).astype(float) if mask else np.ones_like(C)
    rate = c["actual"] / c["cap"]
    norms = {int(g): float(np.mean(rate[c["group"] == g])) for g in np.unique(c["group"])}
    cal = np.power(np.clip(c["prob"], 1e-12, None), 1.0 / T); cal /= cal.sum(1, keepdims=True)
    base = -(cal @ (err * elig[None, :]).T)                       # E[1(elig)*(-|x-c|)]
    settle = cal @ ((C * units) * elig[None, :]).T                # E[1(elig)*c*u]
    yhat = np.empty(len(cal))
    for gid in np.unique(c["group"]):
        m = c["group"] == gid
        yhat[m] = ACTIONS[np.argmax(base[m] + G * settle[m] / (4.0 * norms[int(gid)]), axis=1)]
    return pd.DataFrame(dict(forecast_id=[f"{id(c)}-{i}" for i in range(len(cal))],
                             forecast_kst_dtm=c["dtm"], group_id=c["group"],
                             actual_kwh=c["actual"], prediction_kwh=yhat * c["cap"]))


def score(st, T, G, mask, folds=FOLDS):
    fr = pd.concat([frame_for(st[f], T, G, mask) for f in folds], ignore_index=True)
    s = evaluate_official(fr, CAPACITIES_KWH)
    return float(s.total), float(s.one_minus_nmae), float(s.ficr)


def main() -> int:
    st = load()
    v1 = abs(score(st, DEP_T, DEP_G, False)[0] - DEP_TOTAL) < 1e-4
    print(f"V1 배포 무마스크 재현 -> {score(st, DEP_T, DEP_G, False)[0]:.6f} (기대 {DEP_TOTAL}) : {v1}")

    res, pol, foldwise = {}, {}, {}
    for mask in (False, True):
        tag = "MASKED" if mask else "PLAIN"
        p, ft = {}, {}
        for f in FOLDS:
            others = [o for o in FOLDS if o != f]
            best = max(((T, G) for T in TEMPS for G in GAMMAS),
                       key=lambda k: score(st, k[0], k[1], mask, folds=others)[0])
            p[f] = best
            ft[f] = score(st, best[0], best[1], mask, folds=[f])[0]
        fr = pd.concat([frame_for(st[f], *p[f], mask) for f in FOLDS], ignore_index=True)
        s = evaluate_official(fr, CAPACITIES_KWH)
        res[tag] = (float(s.total), float(s.one_minus_nmae), float(s.ficr))
        pol[tag], foldwise[tag] = p, ft
        print(f"  {tag:7s} fold-외 정책 { {f: p[f] for f in FOLDS} } -> pooled {s.total:.6f}")

    gain = res["MASKED"][0] - res["PLAIN"][0]
    h3 = all(foldwise["MASKED"][f] > foldwise["PLAIN"][f] for f in FOLDS)
    h4 = res["MASKED"][0] > DEP_TOTAL
    print(f"\n폴드별: " + "  ".join(f"{f[-2:]} {foldwise['MASKED'][f]-foldwise['PLAIN'][f]:+.6f}" for f in FOLDS))
    print(f"H1 마스크 > 무마스크   -> {gain > 0}  ({gain:+.6f})")
    print(f"H2 이득 > 0.001013     -> {gain > THRESH}")
    print(f"H3 3/3 폴드 개선       -> {h3}")
    print(f"H4 배포 대비 개선      -> {h4}  ({res['MASKED'][0]-DEP_TOTAL:+.6f})")
    verdict = "ELIGIBILITY_MASK_ADDS" if (gain > THRESH and h3) else "ELIGIBILITY_MASK_INSUFFICIENT"
    print(f"\n판정: {verdict}")
    print(f"MASKED = Total {res['MASKED'][0]:.6f} / 1-NMAE {res['MASKED'][1]:.6f} / FICR {res['MASKED'][2]:.6f}")

    (ROOT / "reports/m282_eligibility_mask_receipt.json").write_text(json.dumps(dict(
        node="M282_ELIGIBILITY_MASK", v1_reproduction=bool(v1), results=res,
        policies={k: {f: list(v) for f, v in pol[k].items()} for k in pol},
        foldwise=foldwise, gain=gain, threshold=THRESH,
        hypotheses=dict(H1=bool(gain > 0), H2=bool(gain > THRESH), H3=bool(h3), H4=bool(h4)),
        verdict=verdict, model_fits=0, lockbox_reopened=False, dacon_upload=False,
        external_actions=[]), indent=1, ensure_ascii=False))
    print("영수증 -> reports/m282_eligibility_mask_receipt.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
