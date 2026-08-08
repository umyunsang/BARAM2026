
"""M284 — 구간 내부 적분(within-bin)을 **배포 계보**에서 재판정 (적합 0회, 파생 전용).

## 배경
`C1N90_WITHIN_BIN_INTEGRATION` 은 재구성 표면에서 uniform 적분 이득 `+0.001890`
(검출문턱 0.001013 초과)을 측정했으나 **월별 게이트가 기각**(5/9 월)했다.
그 실험은 재구성 표면 위였고, M283 이 보였듯 재구성 표면은 구간 지지 자체가 어긋나 있다.
**배포 계보에서 다시 재판정한다.**

배포 구간 폭 ~0.0198, FICR 창 ±0.06 = 편도 약 3 구간. 중심 점질량 취급은 창 경계에서
기대단위를 과대평가해 argmax 를 밀어낸다.

## 사전확약 (실행 전 동결)
- V1  K=1(점질량) 배포정책이 0.628337 재현 (허용 1e-5)
- H1  K=20(구간 내 균등) 팔 > K=1 팔 (둘 다 fold-외 (T,G) 선택)
- H2  이득 > 0.001013
- H3  3/3 폴드 개선
- H4  K=20 팔이 배포(0.628337) 대비 개선

**반증**: H1 이 거짓이면 구간 내부 적분은 배포 계보에서 이득이 없고 축을 닫는다.
락박스 미접근 / 적합 0회 / 제출 없음.
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
DEP_T, DEP_G, DEP_TOTAL, THRESH = 0.5, 1.5, 0.628337, 0.001013
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


def expand(C, K):
    """각 구간을 폭 w 의 균등분포로 보고 K 개 하위점으로 전개."""
    if K == 1:
        return C, np.ones(len(C))
    w = float(np.median(np.diff(C)))
    off = (np.arange(K) + 0.5) / K - 0.5
    sub = (C[:, None] + w * off[None, :]).ravel()
    return sub, np.full(len(C) * K, 1.0 / K)


def mats(C, K):
    sub, wt = expand(C, K)
    err = np.abs(ACTIONS[:, None] - sub[None, :])
    units = np.select([err <= 0.06, err <= 0.08], [4.0, 3.0], default=0.0)
    n = len(C)
    E = (err * wt[None, :]).reshape(len(ACTIONS), n, K).sum(2)          # (A, n)
    S = ((sub[None, :] * units) * wt[None, :]).reshape(len(ACTIONS), n, K).sum(2)
    return E, S


def frame_for(c, T, G, K, cache):
    key = (id(c), K)
    if key not in cache:
        cache[key] = mats(c["centers"], K)
    E, S = cache[key]
    rate = c["actual"] / c["cap"]
    norms = {int(g): float(np.mean(rate[c["group"] == g])) for g in np.unique(c["group"])}
    cal = np.power(np.clip(c["prob"], 1e-12, None), 1.0 / T); cal /= cal.sum(1, keepdims=True)
    base, settle = -(cal @ E.T), cal @ S.T
    yhat = np.empty(len(cal))
    for gid in np.unique(c["group"]):
        m = c["group"] == gid
        yhat[m] = ACTIONS[np.argmax(base[m] + G * settle[m] / (4.0 * norms[int(gid)]), axis=1)]
    return pd.DataFrame(dict(forecast_id=[f"{key[0]}-{i}" for i in range(len(cal))],
                             forecast_kst_dtm=c["dtm"], group_id=c["group"],
                             actual_kwh=c["actual"], prediction_kwh=yhat * c["cap"]))


def main() -> int:
    st, cache = load(), {}
    def sc(T, G, K, folds=FOLDS):
        s = evaluate_official(pd.concat([frame_for(st[f], T, G, K, cache) for f in folds], ignore_index=True),
                              CAPACITIES_KWH)
        return float(s.total), float(s.one_minus_nmae), float(s.ficr)

    d1 = sc(DEP_T, DEP_G, 1)[0]
    v1 = abs(d1 - DEP_TOTAL) < 1e-5
    print(f"V1 K=1 배포정책 -> {d1:.6f} (기대 {DEP_TOTAL}) : {v1}")
    print(f"   K=20 같은 정책 -> {sc(DEP_T, DEP_G, 20)[0]:.6f}")

    res, pol, fw = {}, {}, {}
    for K in (1, 20):
        tag = f"K{K}"
        p, ft = {}, {}
        for f in FOLDS:
            others = [o for o in FOLDS if o != f]
            p[f] = max(((T, G) for T in TEMPS for G in GAMMAS),
                       key=lambda k: sc(k[0], k[1], K, folds=others)[0])
            ft[f] = sc(p[f][0], p[f][1], K, folds=[f])[0]
        s = evaluate_official(pd.concat([frame_for(st[f], *p[f], K, cache) for f in FOLDS], ignore_index=True),
                              CAPACITIES_KWH)
        res[tag] = (float(s.total), float(s.one_minus_nmae), float(s.ficr)); pol[tag], fw[tag] = p, ft
        print(f"  {tag:4s} fold-외 { {f[-2:]: p[f] for f in FOLDS} } -> pooled {s.total:.6f}")

    gain = res["K20"][0] - res["K1"][0]
    h3 = all(fw["K20"][f] > fw["K1"][f] for f in FOLDS)
    h4 = res["K20"][0] > DEP_TOTAL
    print(f"\n폴드별 " + "  ".join(f"{f[-2:]} {fw['K20'][f]-fw['K1'][f]:+.6f}" for f in FOLDS))
    print(f"H1 K20 > K1        -> {gain > 0}  ({gain:+.6f})")
    print(f"H2 > 검출문턱      -> {gain > THRESH}")
    print(f"H3 3/3 폴드        -> {h3}")
    print(f"H4 배포 대비 개선  -> {h4}  ({res['K20'][0]-DEP_TOTAL:+.6f})")
    verdict = "WITHIN_BIN_ADDS_ON_DEPLOYED" if (gain > THRESH and h3 and h4) else "WITHIN_BIN_INSUFFICIENT_ON_DEPLOYED"
    print(f"\n판정: {verdict}")
    print(f"K20 = Total {res['K20'][0]:.6f} / 1-NMAE {res['K20'][1]:.6f} / FICR {res['K20'][2]:.6f}")
    (ROOT / "reports/m284_within_bin_deployed_receipt.json").write_text(json.dumps(dict(
        node="M284_WITHIN_BIN_DEPLOYED", v1_reproduction=bool(v1), results=res, gain=gain,
        threshold=THRESH, foldwise=fw, deployed_total=DEP_TOTAL,
        policies={k: {f: list(v) for f, v in pol[k].items()} for k in pol},
        hypotheses=dict(H1=bool(gain > 0), H2=bool(gain > THRESH), H3=bool(h3), H4=bool(h4)),
        verdict=verdict, model_fits=0, lockbox_reopened=False, dacon_upload=False,
        external_actions=[]), indent=1, ensure_ascii=False))
    print("영수증 -> reports/m284_within_bin_deployed_receipt.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
