
"""D3_H2_3 — 밴드 도달가능성 조건부 멤버 전환 (적합 0회).

착상: 공식 지표는 밴드 적중(4/3/0)과 MAE 를 반반 섞는다. 어떤 행의 예측분포가 넓어
P(밴드적중) 이 낮으면 밴드 조준은 낭비이고, 그 행에서는 MAE 를 줄이는 편이 낫다.
- 이산 멤버(M102@T0.5_G1.5): 밴드 조준에 강함 (FICR 0.405)
- 평활 멤버(M244): 점예측에 강함 (1-NMAE 0.861 대 0.854)

규칙: p_band(행) >= tau 이면 이산 멤버, 아니면 평활 멤버.
균일 결합(N521 최선 w=0.7, 예측온라인 0.639170)과 달리 **행별로 목적을 바꾼다**.
락박스 미접근 / 적합 0회 / 제출 없음.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path("."); sys.path.insert(0, "src")
from baram.evaluation.official import evaluate_official
from baram.constants import CAPACITIES_KWH
sys.path.insert(0, str(ROOT / ".planning/2026-08-01-leaderboard-top-4-loop"))
from m270_gate import evaluate_gate

SRC = ROOT / "artifacts/backtests/m269-probe"
P = ROOT / "artifacts/backtests/metric-aligned-probe"
FOLDS = ("dev-2023-Q2", "dev-2023-Q3", "dev-2023-Q4")
ACT = np.round(np.arange(0.075, 1.076, 0.0025), 6)
DEP = (0.5, 1.5); OFF_C, OFF_A = 0.006554, 0.021119; THRESH = 0.001013

rows = []
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
    rows.append(pd.DataFrame(dict(
        forecast_kst_dtm=pd.to_datetime(z["forecast_kst_dtm"]), group_id=z["group_id"].astype(int),
        actual_kwh=z["actual_kwh"].astype(float), cap=cap,
        disc=ACT[idx] * cap, p_band=(cal * inband[idx]).sum(1))))
disc = pd.concat(rows, ignore_index=True)

ana = pd.concat([pd.read_parquet(P / f"M244_RARE_EVENT_CORRECTED_ANALOG_Q234-{f}.parquet")
                 [["forecast_kst_dtm", "group_id", "prediction_kwh"]] for f in FOLDS], ignore_index=True)
ana = ana.rename(columns={"prediction_kwh": "smooth"})
m = disc.merge(ana, on=["forecast_kst_dtm", "group_id"], how="inner")
assert float(np.mean(np.abs(m.disc - m.smooth) < 1e-9)) < 0.99, "inputs identical"
m["forecast_id"] = [f"r{i}" for i in range(len(m))]
print(f"결합행 {len(m)}   p_band 범위 {m.p_band.min():.3f}~{m.p_band.max():.3f} 중앙값 {m.p_band.median():.3f}")

def score(tau):
    use_disc = m.p_band.to_numpy() >= tau
    pred = np.where(use_disc, m.disc.to_numpy(), m.smooth.to_numpy())
    d = m[["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh"]].copy()
    d["prediction_kwh"] = pred
    s = evaluate_official(d, CAPACITIES_KWH)
    frac = float(use_disc.mean())
    off = frac * OFF_C + (1 - frac) * OFF_A
    return float(s.total), float(s.one_minus_nmae), float(s.ficr), frac, float(s.total) + off

print(f"\n{'tau':>6} {'이산비율':>8} {'로컬':>10} {'1-NMAE':>10} {'FICR':>10} {'예측온라인':>11}")
best = None
for tau in (0.0, 0.10, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 1.01):
    t, n, fi, frac, po = score(tau)
    if best is None or po > best[1]: best = (tau, po, t, n, fi, frac)
    lab = "  (전부 이산)" if tau == 0.0 else ("  (전부 평활)" if tau > 1 else "")
    print(f"{tau:6.2f} {frac:8.3f} {t:10.6f} {n:10.6f} {fi:10.6f} {po:11.6f}{lab}")

print(f"\n최선 tau={best[0]:.2f}  예측온라인 {best[1]:.6f}  (이산비율 {best[5]:.3f})")
print(f"  전부이산 대비 {best[1]-score(0.0)[4]:+.6f}   N521 균일결합 0.639170 대비 {best[1]-0.639170:+.6f}")
print(f"  M266 실측 0.6374709 대비 {best[1]-0.6374709:+.6f}")
verdict = "REACHABILITY_SWITCH_HELPS" if best[1] > max(score(0.0)[4], 0.639170) + THRESH else "NO_GAIN"
print(f"\n판정: {verdict}")
Path("reports/D3_H2_3_reachability_switch.json").write_text(json.dumps(dict(
    node="D3_H2_3", stage="D3", rows=len(m),
    curve={str(t): list(score(t)) for t in (0.0, 0.20, 0.30, 0.40, 1.01)},
    best=dict(tau=best[0], predicted_online=best[1], local=best[2], nmae=best[3], ficr=best[4],
              discrete_fraction=best[5]),
    verdict=verdict, origin="derived from D3_H2_1/H2_2",
    model_fits=0, dacon_upload=False), indent=1, ensure_ascii=False))
print("영수증 -> reports/D3_H2_3_reachability_switch.json")
