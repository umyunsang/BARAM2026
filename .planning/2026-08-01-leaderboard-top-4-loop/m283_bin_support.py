
"""M283 — 재구성 격차의 구성요소: 구간 지지(support) 정의 (적합 0회, 파생 전용).

## 발견 (M282 부산물)
- 배포 계보(M269_PROBE) 구간 중심 `0.1094 ~ 1.0025` — **적격임계 0.10 미만 구간 0 개**
- 재구성 표면(994ae6) 구간 중심 `0.0100 ~ 0.9100` — 적격임계 미만 **5 개**, 평균 확률질량 **56.6%**

C1N95/C1N96 은 재구성 격차 0.024562 를 "결정규칙 -0.008475 / 모형 +0.033037" 로 갈랐다.
그러나 '모형' 성분에는 **구간 지지 정의**라는 표현 선택이 섞여 있을 수 있다.

## 사전확약 (실행 전 동결)
- V1  재구성 표면 배포규칙(T0.5_G1.5) 무마스크가 0.595568 재현 (m271_n9 공표값, 허용 1e-5)
- H1  적격성 마스크 팔 > 무마스크 팔 (둘 다 fold-외 (T,G) 선택)
- H2  이득 > 0.001013 (검출문턱)
- H3  3/3 폴드 개선
- H4  이득이 재구성 격차 0.024562 의 20% 초과 (> 0.004912)

**반증**: H1 이 거짓이면 구간 지지는 격차의 구성요소가 아니다.
락박스 미접근 / 적합 0회 / 제출 없음.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src"))
from baram.evaluation.official import evaluate_official  # noqa: E402
from baram.constants import CAPACITIES_KWH  # noqa: E402

SURF = ROOT / "artifacts/cache/m271_decision_surface/994ae6dff5796332daf21a6f"
FOLDS = ("dev-2023-Q2", "dev-2023-Q3", "dev-2023-Q4")
CENTERS = (np.arange(46) + 0.5) * 0.02
ACTIONS = np.round(np.arange(0.075, 1.076, 0.0025), 6)
A1_TOTAL, GAP, THRESH, ELIG = 0.595568, 0.024562, 0.001013, 0.10
TEMPS = (0.4, 0.5, 0.75, 1.0, 1.2, 1.4, 1.6, 2.0, 2.5, 3.0)
GAMMAS = (0.0, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)

_ERR = np.abs(ACTIONS[:, None] - CENTERS[None, :])
_UNITS = np.select([_ERR <= 0.06, _ERR <= 0.08], [4.0, 3.0], default=0.0)
_CU = CENTERS[None, :] * _UNITS
_ELIG = (CENTERS >= ELIG).astype(float)


def load():
    st = {}
    for f in FOLDS:
        z = np.load(SURF / f"{f}__arrays.npz")
        m = pd.read_parquet(SURF / f"{f}__meta.parquet")
        rate = m["actual_kwh"].to_numpy(float) / z["capacity"]
        ok = np.isfinite(rate)
        st[f] = dict(prob=z["probability"][ok], group=z["group"][ok].astype(int),
                     cap=z["capacity"][ok], meta=m[ok].reset_index(drop=True), rate=rate[ok])
    return st


def frame_for(c, T, G, mask):
    e = _ELIG if mask else np.ones_like(_ELIG)
    norms = {int(g): float(np.mean(c["rate"][c["group"] == g])) for g in np.unique(c["group"])}
    cal = np.power(np.clip(c["prob"], 1e-12, None), 1.0 / T); cal /= cal.sum(1, keepdims=True)
    base = -(cal @ (_ERR * e[None, :]).T)
    settle = cal @ (_CU * e[None, :]).T
    yhat = np.empty(len(cal))
    for gid in np.unique(c["group"]):
        m = c["group"] == gid
        yhat[m] = ACTIONS[np.argmax(base[m] + G * settle[m] / (4.0 * norms[int(gid)]), axis=1)]
    d = c["meta"][["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh"]].copy()
    d["prediction_kwh"] = yhat * c["cap"]
    return d


def score(st, T, G, mask, folds=FOLDS):
    s = evaluate_official(pd.concat([frame_for(st[f], T, G, mask) for f in folds], ignore_index=True), CAPACITIES_KWH)
    return float(s.total), float(s.one_minus_nmae), float(s.ficr)


def main() -> int:
    st = load()
    plain_dep = score(st, 0.5, 1.5, False)[0]
    v1 = abs(plain_dep - A1_TOTAL) < 1e-5
    print(f"V1 재구성 배포규칙 무마스크 -> {plain_dep:.6f} (기대 {A1_TOTAL}) : {v1}")
    print(f"   같은 규칙 + 마스크        -> {score(st, 0.5, 1.5, True)[0]:.6f}")

    res, pol, fw = {}, {}, {}
    for mask in (False, True):
        tag = "MASKED" if mask else "PLAIN"
        p, ft = {}, {}
        for f in FOLDS:
            others = [o for o in FOLDS if o != f]
            p[f] = max(((T, G) for T in TEMPS for G in GAMMAS),
                       key=lambda k: score(st, k[0], k[1], mask, folds=others)[0])
            ft[f] = score(st, p[f][0], p[f][1], mask, folds=[f])[0]
        s = evaluate_official(pd.concat([frame_for(st[f], *p[f], mask) for f in FOLDS], ignore_index=True),
                              CAPACITIES_KWH)
        res[tag] = (float(s.total), float(s.one_minus_nmae), float(s.ficr)); pol[tag], fw[tag] = p, ft
        print(f"  {tag:7s} fold-외 { {f[-2:]: p[f] for f in FOLDS} } -> pooled {s.total:.6f}")

    gain = res["MASKED"][0] - res["PLAIN"][0]
    h3 = all(fw["MASKED"][f] > fw["PLAIN"][f] for f in FOLDS)
    share = gain / GAP
    print(f"\n폴드별 " + "  ".join(f"{f[-2:]} {fw['MASKED'][f]-fw['PLAIN'][f]:+.6f}" for f in FOLDS))
    print(f"H1 마스크 > 무마스크  -> {gain > 0}  ({gain:+.6f})")
    print(f"H2 > 검출문턱          -> {gain > THRESH}")
    print(f"H3 3/3 폴드            -> {h3}")
    print(f"H4 격차의 20% 초과     -> {share > 0.20}  (격차 {GAP} 의 {share:.1%})")
    verdict = "BIN_SUPPORT_IS_A_GAP_COMPONENT" if (gain > THRESH and h3) else "BIN_SUPPORT_NOT_A_GAP_COMPONENT"
    print(f"\n판정: {verdict}")
    (ROOT / "reports/m283_bin_support_receipt.json").write_text(json.dumps(dict(
        node="M283_BIN_SUPPORT", v1_reproduction=bool(v1), results=res, gain=gain,
        gap=GAP, share_of_gap=share, foldwise=fw,
        policies={k: {f: list(v) for f, v in pol[k].items()} for k in pol},
        hypotheses=dict(H1=bool(gain > 0), H2=bool(gain > THRESH), H3=bool(h3), H4=bool(share > 0.20)),
        verdict=verdict, model_fits=0, lockbox_reopened=False, dacon_upload=False,
        external_actions=[]), indent=1, ensure_ascii=False))
    print("영수증 -> reports/m283_bin_support_receipt.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
