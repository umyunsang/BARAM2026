
"""M276 — 결정층 온도가 곧 캘리브레이션인가? (적합 0회, 파생 전용)

## 사전확약 (실행 전 동결)

M272: 재구성 결정표면에서 정책격자를 확장하니 Total 최적이 T*≈2.0 (G=4) 이었다. T>1 은 분포를 무르게 한다.
M275: 분포 후보 전체가 밴드적중을 과대예측한다 (채택 D1 의 bias +0.0710).

두 현상이 **동일 현상**이라면, 46-bin 확률의 밴드적중 bias 를 0 으로 만드는 온도 T_cal 이
Total 을 최대화하는 T* 와 일치해야 한다.

- H1  원시 확률(T=1)은 밴드적중을 과대예측한다            bias(T=1) > 0
- H2  bias(T_cal)=0 인 T_cal 이 존재하고 1 보다 크다       T_cal > 1
- H3  |T_cal - T*| <= 0.5                                 (핵심 가설)
- H4  bias 는 T 에 대해 단조감소                           (기전 정합성)

**반증 조건**: H3 이 거짓이면 "결정온도 = 캘리브레이션" 가설은 기각된다.
그 경우 결정온도는 캘리브레이션과 별개의 레버이며, 캘리브레이션을 고쳐도 T*는 1 로 가지 않는다.

락박스 미접근 / 모델 적합 0회 / 제출물 없음 / 외부행위 없음.
주: 밴드확률은 `bayes_decision` 과 동일하게 구간 중심 점질량으로 계산한다(C1N90 한계 상속).
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src"))
from baram.evaluation.official import evaluate_official  # noqa: E402
from baram.constants import CAPACITIES_KWH  # noqa: E402

N_CLASS, CLASS_WIDTH = 46, 0.02
CENTERS = (np.arange(N_CLASS) + 0.5) * CLASS_WIDTH
ACTIONS = np.round(np.arange(0.075, 1.076, 0.0025), 6)
FOLDS = ("dev-2023-Q2", "dev-2023-Q3", "dev-2023-Q4")
SURFACE = ROOT / "artifacts/cache/m271_decision_surface/994ae6dff5796332daf21a6f"
G_FIXED, BAND, ELIGIBLE = 4.0, 0.06, 0.10
T_STAR = 2.0

_ERR = np.abs(ACTIONS[:, None] - CENTERS[None, :])
_UNITS = np.select([_ERR <= 0.06, _ERR <= 0.08], [4.0, 3.0], default=0.0)
_SETTLE_M = (CENTERS[None, :] * _UNITS).T
_INBAND = (_ERR <= BAND)          # (n_actions, n_class)


def load():
    st = {}
    for f in FOLDS:
        z = np.load(SURFACE / f"{f}__arrays.npz")
        st[f] = dict(prob=z["probability"], group=z["group"], cap=z["capacity"],
                     meta=pd.read_parquet(SURFACE / f"{f}__meta.parquet"))
    return st


def evaluate(store, T, G=G_FIXED):
    frames, ps, hs = [], [], []
    for f in FOLDS:
        c = store[f]
        rate = c["meta"]["actual_kwh"].to_numpy(float) / c["cap"]
        ok = np.isfinite(rate)
        norms = {int(g): float(np.nanmean(rate[(c["group"] == g) & ok])) for g in np.unique(c["group"])}
        cal = np.power(np.clip(c["prob"], 1e-12, None), 1.0 / T)
        cal /= cal.sum(axis=1, keepdims=True)
        settle, base = cal @ _SETTLE_M, -(cal @ _ERR.T)
        idx = np.empty(len(cal), dtype=int)
        for gid in np.unique(c["group"]):
            m = c["group"] == gid
            idx[m] = np.argmax(base[m] + G * settle[m] / (4.0 * norms[int(gid)]), axis=1)
        yhat = ACTIONS[idx]
        p_band = (cal * _INBAND[idx]).sum(axis=1)          # 예측 밴드적중확률
        hit = (np.abs(yhat - rate) <= BAND).astype(float)  # 실제 밴드적중
        elig = ok & (rate >= ELIGIBLE)
        ps.append(p_band[elig]); hs.append(hit[elig])
        d = c["meta"][["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh"]].copy()
        d["prediction_kwh"] = yhat * c["cap"]
        frames.append(d[ok])
    sc = evaluate_official(pd.concat(frames, ignore_index=True), CAPACITIES_KWH)
    p, h = np.concatenate(ps), np.concatenate(hs)
    return dict(total=float(sc.total), one_minus_nmae=float(sc.one_minus_nmae), ficr=float(sc.ficr),
                pred_band=float(p.mean()), actual_band=float(h.mean()), bias=float(p.mean() - h.mean()))


def main() -> int:
    store = load()
    grid = [0.5, 0.75, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.5, 3.0, 4.0]
    rows = {T: evaluate(store, T) for T in grid}
    print(f"{'T':>5} {'Total':>9} {'1-NMAE':>9} {'FICR':>9} {'예측밴드':>9} {'실제밴드':>9} {'bias':>9}")
    for T in grid:
        r = rows[T]
        print(f"{T:5.2f} {r['total']:9.6f} {r['one_minus_nmae']:9.6f} {r['ficr']:9.6f} "
              f"{r['pred_band']:9.6f} {r['actual_band']:9.6f} {r['bias']:+9.6f}")

    ts = np.array(grid); bias = np.array([rows[T]["bias"] for T in grid])
    tot = np.array([rows[T]["total"] for T in grid])
    T_star = float(ts[tot.argmax()])
    mono = bool(np.all(np.diff(bias) <= 1e-9))
    T_cal = float(np.interp(0.0, bias[::-1], ts[::-1])) if bias.min() < 0 < bias.max() else float("nan")

    print(f"\nH1 bias(T=1) > 0                -> {rows[1.0]['bias'] > 0}  ({rows[1.0]['bias']:+.6f})")
    print(f"H2 T_cal 존재 and > 1           -> {(not np.isnan(T_cal)) and T_cal > 1}  (T_cal={T_cal:.3f})")
    print(f"H3 |T_cal - T*| <= 0.5 [핵심]   -> {(not np.isnan(T_cal)) and abs(T_cal - T_star) <= 0.5}"
          f"  (T*={T_star:.2f}, 차이={abs(T_cal - T_star):.3f})")
    print(f"H4 bias 단조감소                 -> {mono}")
    verdict = ("DECISION_TEMPERATURE_IS_CALIBRATION"
               if (not np.isnan(T_cal)) and abs(T_cal - T_star) <= 0.5
               else "DECISION_TEMPERATURE_IS_NOT_CALIBRATION")
    print(f"\n판정: {verdict}")

    out = dict(node="M276_TEMPERATURE_IS_CALIBRATION", gamma=G_FIXED, T_star=T_star, T_cal=T_cal,
               predeclared=dict(H1="bias(T=1)>0", H2="T_cal>1", H3="|T_cal-T*|<=0.5", H4="bias monotone"),
               results={f"T{T:g}": rows[T] for T in grid}, verdict=verdict)
    (ROOT / "reports/m276_temperature_is_calibration_receipt.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False))
    print("영수증 -> reports/m276_temperature_is_calibration_receipt.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
