
"""D4_J1_1 — 동결 월별게이트의 검정력 측정 (적합 0회, 게이트 무수정).

근거: research://Hewamalage 2022 PMC9718476 [directly_supported]
본 세션에서 문턱 근처 후보가 반복 기각됐다 (M284 +0.001207 월별기각 / M251 월별기각 /
D3_H2_4 +0.000835 문턱미달). 게이트가 다중검정 오탐을 막는 것은 확인됐으나(3건 차단),
**진짜 이득을 얼마나 잡아내는지(검정력)** 는 측정된 적이 없다.

방법: 배포 예측에 **알려진 크기의 진짜 개선**을 인위 주입한다.
      pred' = actual + k*(pred - actual)  (k<1 이면 오차가 k배로 줄어든 진짜 개선)
      각 k 에서 게이트를 무수정 적용해 통과 여부를 본다.
락박스 미접근 / 적합 0회 / 게이트 변경 없음 / 제출 없음.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path("."); sys.path.insert(0, "src")
sys.path.insert(0, str(ROOT / ".planning/2026-08-01-leaderboard-top-4-loop"))
from baram.evaluation.official import evaluate_official
from baram.constants import CAPACITIES_KWH
from m270_gate import GATE_VERSION, evaluate_gate

SRC = ROOT / "artifacts/backtests/m269-probe"
FOLDS = ("dev-2023-Q2", "dev-2023-Q3", "dev-2023-Q4")
ACT = np.round(np.arange(0.075, 1.076, 0.0025), 6)
DEP = (0.5, 1.5)

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
    parts.append(pd.DataFrame(dict(forecast_id=[f"{f}-{i}" for i in range(len(cal))],
                                   forecast_kst_dtm=pd.to_datetime(z["forecast_kst_dtm"]),
                                   group_id=z["group_id"].astype(int),
                                   actual_kwh=z["actual_kwh"].astype(float),
                                   prediction_kwh=yh * cap)))
parent = pd.concat(parts, ignore_index=True)
parent["month"] = parent["forecast_kst_dtm"].dt.to_period("M").astype(str)
base_total = float(evaluate_official(parent.drop(columns=["month"]), CAPACITIES_KWH).total)
print(f"부모(배포) Total {base_total:.6f}   게이트 {GATE_VERSION}\n")

print(f"{'k':>6} {'오차감소':>8} {'delta':>10} {'G1':>4} {'G2':>4} {'G3':>4} {'G4':>4} {'통과':>5}")
rows = []
for k in (1.00, 0.995, 0.99, 0.98, 0.97, 0.96, 0.95, 0.93, 0.90):
    cand = parent.copy()
    cand["prediction_kwh"] = cand.actual_kwh + k * (cand.prediction_kwh - cand.actual_kwh)
    t = float(evaluate_official(cand.drop(columns=["month"]), CAPACITIES_KWH).total)
    g = evaluate_gate(cand, parent)
    flags = ["PASS" if v else "FAIL" for v in g.conditions.values()]
    short = ["O" if v else "X" for v in g.conditions.values()]
    rows.append(dict(k=k, delta=t - base_total, passed=bool(g.passed),
                     conditions={kk: bool(v) for kk, v in g.conditions.items()}))
    print(f"{k:6.3f} {1-k:8.1%} {t-base_total:+10.6f} {short[0]:>4} {short[1]:>4} {short[2]:>4} {short[3]:>4} "
          f"{'PASS' if g.passed else 'FAIL':>5}")

passing = [r for r in rows if r["passed"]]
mdd = min((r["delta"] for r in passing), default=None)
print(f"\n게이트가 통과시킨 최소 개선폭(검출한계) = {mdd if mdd is None else f'{mdd:+.6f}'}")
print(f"검출문턱 명목값 0.001013 과 비교 -> {'게이트가 더 엄격' if mdd and mdd > 0.001013 else '명목과 정합'}")
print(f"\n참고: 본 세션 기각 사례")
print(f"  M284    +0.001207  월별게이트 기각")
print(f"  D3_H2_4 +0.000835  문턱 미달")
verdict = ("GATE_POWER_MEASURED_STRICTER_THAN_NOMINAL" if mdd and mdd > 0.001013
           else "GATE_POWER_CONSISTENT_WITH_NOMINAL")
print(f"\n판정: {verdict}")
Path("reports/D4_J1_1_gate_power.json").write_text(json.dumps(dict(
    node="D4_J1_1", stage="D4", gate_version=GATE_VERSION, base_total=base_total,
    injections=rows, minimum_detectable_delta=mdd, nominal_threshold=0.001013,
    verdict=verdict, origin="research://Hewamalage-2022-PMC9718476 [directly_supported]",
    model_fits=0, dacon_upload=False, gate_modified=False), indent=1, ensure_ascii=False))
print("영수증 -> reports/D4_J1_1_gate_power.json")
