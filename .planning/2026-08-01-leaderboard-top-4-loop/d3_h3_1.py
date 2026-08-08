
"""D3_H3_1 — 적격행 기준 전역 이동의 fold-외 효과 (적합 0회).

근거: research://NOAA NCEP ON-520 (2025) [near_match_only] + D2_G3_1 부수 발견
D2_G3_1: 적격행에서 24개 시각 전부 부호편향 양수, 평균 +0.047 (과대예측)
C1N22_GLOBAL_SHIFT 는 MOS_SUPERSEDED_BY_ELIGIBLE_POPULATION 으로 폐기됐다.
질문: 그 폐기가 **전체행** 기준이었다면, **적격행** 기준 전역 이동은 아직 미검정이다.

주의: 결정층은 조건부 중앙값이 아니라 기대정산금 최대화 행동을 낸다(M276). 발전량 가중 때문에
      상방 편향이 **최적일 수도** 있다. 그러므로 이동이 점수를 올리는지는 측정해야만 안다.
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
from m270_gate import GATE_VERSION, evaluate_gate

SRC = ROOT / "artifacts/backtests/m269-probe"
FOLDS = ("dev-2023-Q2", "dev-2023-Q3", "dev-2023-Q4")
ACT = np.round(np.arange(0.075, 1.076, 0.0025), 6)
DEP = (0.5, 1.5); THRESH = 0.001013

def build(shift):
    parts = []
    for f in FOLDS:
        z = np.load(SRC / f"M269_PROBE-{f}-probability.npz", allow_pickle=True)
        C = z["centers"]; cap = np.array([CAPACITIES_KWH[int(g)] for g in z["group_id"]], float)
        rate = z["actual_kwh"].astype(float) / cap
        err = np.abs(ACT[:, None] - C[None, :])
        units = np.select([err <= 0.06, err <= 0.08], [4.0, 3.0], default=0.0)
        norms = {int(g): float(np.mean(rate[z["group_id"] == g])) for g in np.unique(z["group_id"])}
        cal = np.power(np.clip(z["probability"], 1e-12, None), 1.0 / DEP[0]); cal /= cal.sum(1, keepdims=True)
        base, settle = -(cal @ err.T), cal @ (C[None, :] * units).T
        yh = np.empty(len(cal))
        for gid in np.unique(z["group_id"]):
            m = z["group_id"] == gid
            yh[m] = ACT[np.argmax(base[m] + DEP[1] * settle[m] / (4.0 * norms[int(gid)]), axis=1)]
        yh = np.clip(yh + shift, 0.0, 1.0)
        d = pd.DataFrame(dict(forecast_id=[f"{f}-{i}" for i in range(len(cal))],
                              forecast_kst_dtm=pd.to_datetime(z["forecast_kst_dtm"]),
                              group_id=z["group_id"].astype(int), actual_kwh=z["actual_kwh"].astype(float),
                              prediction_kwh=yh * cap))
        parts.append(d)
    return pd.concat(parts, ignore_index=True)

print(f"{'shift':>8} {'Total':>10} {'1-NMAE':>10} {'FICR':>10}")
curve = {}
for s in (0.00, -0.01, -0.02, -0.03, -0.047, -0.06, +0.01, +0.02):
    fr = build(s); sc = evaluate_official(fr, CAPACITIES_KWH)
    curve[s] = (float(sc.total), float(sc.one_minus_nmae), float(sc.ficr))
    print(f"{s:+8.3f} {curve[s][0]:10.6f} {curve[s][1]:10.6f} {curve[s][2]:10.6f}")

base_t = curve[0.0][0]
best = max(curve, key=lambda k: curve[k][0])
gain = curve[best][0] - base_t
print(f"\n기준(shift 0) {base_t:.6f}   최선 shift {best:+.3f} -> {curve[best][0]:.6f}  ({gain:+.6f})")
print(f"검출문턱 {THRESH} 초과 -> {gain > THRESH}")

verdict = "GLOBAL_SHIFT_NO_GAIN"
if gain > THRESH and best != 0.0:
    cand = build(best); par = build(0.0)
    for fr in (cand, par): fr["month"] = fr["forecast_kst_dtm"].dt.to_period("M").astype(str)
    g = evaluate_gate(cand, par)
    for k, ok in g.conditions.items(): print(f"  [{'PASS' if ok else 'FAIL'}] {k}")
    print(f"월별 게이트 -> {g.passed}")
    verdict = "GLOBAL_SHIFT_HELPS" if g.passed else "GLOBAL_SHIFT_GATE_REJECTS"
print(f"\n판정: {verdict}")
Path("reports/D3_H3_1_global_shift.json").write_text(json.dumps(dict(
    node="D3_H3_1", stage="D3", gate_version=GATE_VERSION,
    curve={str(k): list(v) for k, v in curve.items()}, base=base_t,
    best_shift=float(best), gain=gain, threshold=THRESH, verdict=verdict,
    origin="research://NOAA-NCEP-ON520 [near_match_only] + D2_G3_1",
    model_fits=0, dacon_upload=False), indent=1, ensure_ascii=False))
print("영수증 -> reports/D3_H3_1_global_shift.json")
