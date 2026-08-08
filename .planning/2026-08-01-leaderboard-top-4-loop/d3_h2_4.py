
"""D3_H2_4 — 도달가능성 조건부 **가중** 결합 (적합 0회, fold-외 검증 포함).

H2_3: 행별 멤버 **전환**은 +0.003131 (전부이산 대비) 이나 균일결합에 -0.001148 열위.
      원인 = 전환은 한 멤버만 쓰므로 분산 감소가 없다.
N521: 균일 결합 w=0.7 이 예측온라인 0.639170.

종합: 두 멤버가 **항상 기여**하되 가중을 도달가능성에 따라 바꾼다.
      p_band >= tau 이면 w_hi (이산 쪽), 아니면 w_lo (평활 쪽).
자유도 3개이므로 **fold-외 선택**으로 검증한다.
락박스 미접근 / 적합 0회 / 제출 없음.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path("."); sys.path.insert(0, "src")
from baram.evaluation.official import evaluate_official
from baram.constants import CAPACITIES_KWH

SRC = ROOT / "artifacts/backtests/m269-probe"
P = ROOT / "artifacts/backtests/metric-aligned-probe"
FOLDS = ("dev-2023-Q2", "dev-2023-Q3", "dev-2023-Q4")
ACT = np.round(np.arange(0.075, 1.076, 0.0025), 6)
DEP = (0.5, 1.5); OFF_C, OFF_A = 0.006554, 0.021119; THRESH = 0.001013

parts = {}
for f in FOLDS:
    z = np.load(SRC / f"M269_PROBE-{f}-probability.npz", allow_pickle=True)
    C = z["centers"]; cap = np.array([CAPACITIES_KWH[int(g)] for g in z["group_id"]], float)
    rate = z["actual_kwh"].astype(float) / cap
    err = np.abs(ACT[:, None] - C[None, :])
    units = np.select([err <= 0.06, err <= 0.08], [4.0, 3.0], default=0.0)
    inband = err <= 0.06
    norms = {int(g): float(np.mean(rate[z["group_id"] == g])) for g in np.unique(z["group_id"])}
    cal = np.power(np.clip(z["probability"], 1e-12, None), 1.0 / DEP[0]); cal /= cal.sum(1, keepdims=True)
    base, settle = -(cal @ err.T), cal @ (C[None, :] * units).T
    idx = np.empty(len(cal), dtype=int)
    for gid in np.unique(z["group_id"]):
        m = z["group_id"] == gid
        idx[m] = np.argmax(base[m] + DEP[1] * settle[m] / (4.0 * norms[int(gid)]), axis=1)
    d = pd.DataFrame(dict(forecast_kst_dtm=pd.to_datetime(z["forecast_kst_dtm"]),
                          group_id=z["group_id"].astype(int),
                          actual_kwh=z["actual_kwh"].astype(float), cap=cap,
                          disc=ACT[idx] * cap, p_band=(cal * inband[idx]).sum(1)))
    a = pd.read_parquet(P / f"M244_RARE_EVENT_CORRECTED_ANALOG_Q234-{f}.parquet")[
        ["forecast_kst_dtm", "group_id", "prediction_kwh"]].rename(columns={"prediction_kwh": "smooth"})
    dd = d.merge(a, on=["forecast_kst_dtm", "group_id"], how="inner")
    dd["forecast_id"] = [f"{f}-{i}" for i in range(len(dd))]
    parts[f] = dd
print({f: len(parts[f]) for f in FOLDS})

def score(df, wh, wl, tau):
    p = df.p_band.to_numpy()
    w = np.where(p >= tau, wh, wl)
    pred = np.clip(w * df.disc.to_numpy() + (1 - w) * df.smooth.to_numpy(), 0, df.cap.to_numpy())
    d = df[["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh"]].copy()
    d["prediction_kwh"] = pred
    s = evaluate_official(d, CAPACITIES_KWH)
    frac = float(w.mean())
    return float(s.total), float(s.total) + frac * OFF_C + (1 - frac) * OFF_A

GRID = [(wh, wl, tau) for wh in (0.6, 0.7, 0.8, 0.9) for wl in (0.3, 0.4, 0.5, 0.6)
        for tau in (0.25, 0.30, 0.35, 0.40, 0.45) if wh > wl]
allf = pd.concat(parts.values(), ignore_index=True)
ins = max(GRID, key=lambda k: score(allf, *k)[1])
print(f"\n인샘플 최적 wh={ins[0]} wl={ins[1]} tau={ins[2]} -> 예측온라인 {score(allf, *ins)[1]:.6f}")

# fold-외
frames, offs = [], []
sel = {}
for f in FOLDS:
    others = pd.concat([parts[o] for o in FOLDS if o != f], ignore_index=True)
    best = max(GRID, key=lambda k: score(others, *k)[1])
    sel[f] = best
    df = parts[f]; p = df.p_band.to_numpy()
    w = np.where(p >= best[2], best[0], best[1])
    d = df[["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh"]].copy()
    d["prediction_kwh"] = np.clip(w * df.disc.to_numpy() + (1 - w) * df.smooth.to_numpy(), 0, df.cap.to_numpy())
    frames.append(d); offs.append(float(w.mean()))
s_fo = evaluate_official(pd.concat(frames, ignore_index=True), CAPACITIES_KWH)
frac = float(np.mean(offs))
fo_pred = float(s_fo.total) + frac * OFF_C + (1 - frac) * OFF_A
print(f"fold-외 선택 {sel}")
print(f"fold-외 로컬 {float(s_fo.total):.6f}  예측온라인 {fo_pred:.6f}")

uni = score(allf, 0.7, 0.7, 0.0)
print(f"\n균일 w=0.70 (N521)      예측온라인 {uni[1]:.6f}")
print(f"조건부 가중 (인샘플)      {score(allf, *ins)[1]:.6f}")
print(f"조건부 가중 (fold-외)     {fo_pred:.6f}   균일 대비 {fo_pred-uni[1]:+.6f}")
verdict = "CONDITIONAL_WEIGHT_HELPS" if fo_pred > uni[1] + THRESH else "NO_GAIN_OVER_UNIFORM"
print(f"\n판정: {verdict}")
Path("reports/D3_H2_4_conditional_weight.json").write_text(json.dumps(dict(
    node="D3_H2_4", stage="D3", insample=list(ins), insample_pred=score(allf, *ins)[1],
    foldout_selection={k: list(v) for k, v in sel.items()},
    foldout_local=float(s_fo.total), foldout_pred=fo_pred, uniform_pred=uni[1],
    gain_over_uniform=fo_pred - uni[1], verdict=verdict,
    model_fits=0, dacon_upload=False), indent=1, ensure_ascii=False))
print("영수증 -> reports/D3_H2_4_conditional_weight.json")
