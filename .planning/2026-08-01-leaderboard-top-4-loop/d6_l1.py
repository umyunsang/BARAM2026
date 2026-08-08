
"""D6_L1 — 현재 예보품질에서 도달 가능한 (1-NMAE, FICR) 프론티어 (적합 0회).

리더보드 분해 발견:
  우리는 96팀 **전원보다** 1-NMAE 와 FICR 이 모두 낮다 (0/96).
  FICR 만 1위 수준(0.467670)까지 올리면 Total 0.663222 로 **목표 0.66 달성**.
  NMAE 만 1위 수준(0.879640)까지 올려도 Total 0.647903 로 미달.

질문: 현재 확률표면에서 결정정책을 어떻게 바꿔도 FICR 0.4677 에 도달할 수 있는가?
      = 격차가 결정층 문제인가, 예보품질 문제인가?
방법: T(온도) x G(정산이득) 격자를 넓게 훑어 (1-NMAE, FICR) 프론티어를 그린다.
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
FOLDS = ("dev-2023-Q2", "dev-2023-Q3", "dev-2023-Q4")
ACT = np.round(np.arange(0.075, 1.076, 0.0025), 6)
OFF_C = 0.006554

Z = {}
for f in FOLDS:
    z = np.load(SRC / f"M269_PROBE-{f}-probability.npz", allow_pickle=True)
    cap = np.array([CAPACITIES_KWH[int(g)] for g in z["group_id"]], float)
    Z[f] = dict(C=z["centers"], cap=cap, gid=z["group_id"].astype(int),
                act=z["actual_kwh"].astype(float), P=z["probability"],
                dtm=pd.to_datetime(z["forecast_kst_dtm"]))
    Z[f]["rate"] = Z[f]["act"] / cap

def run(T, G):
    frames = []
    for f in FOLDS:
        d = Z[f]; C = d["C"]
        err = np.abs(ACT[:, None] - C[None, :])
        units = np.select([err <= 0.06, err <= 0.08], [4.0, 3.0], default=0.0)
        norms = {int(g): float(np.mean(d["rate"][d["gid"] == g])) for g in np.unique(d["gid"])}
        cal = np.power(np.clip(d["P"], 1e-12, None), 1.0 / T); cal /= cal.sum(1, keepdims=True)
        base, settle = -(cal @ err.T), cal @ (C[None, :] * units).T
        yh = np.empty(len(cal))
        for gid in np.unique(d["gid"]):
            m = d["gid"] == gid
            yh[m] = ACT[np.argmax(base[m] + G * settle[m] / (4.0 * norms[int(gid)]), axis=1)]
        frames.append(pd.DataFrame(dict(forecast_id=[f"{f}-{i}" for i in range(len(cal))],
                                        forecast_kst_dtm=d["dtm"], group_id=d["gid"],
                                        actual_kwh=d["act"], prediction_kwh=yh * d["cap"])))
    s = evaluate_official(pd.concat(frames, ignore_index=True), CAPACITIES_KWH)
    return float(s.one_minus_nmae), float(s.ficr), float(s.total)

TS = (0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0)
GS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 15.0, 30.0, 100.0)
pts = []
for T in TS:
    for G in GS:
        n, fi, t = run(T, G)
        pts.append(dict(T=T, G=G, nmae=n, ficr=fi, total=t))
df = pd.DataFrame(pts)

print(f"격자 {len(df)}점")
print(f"\n로컬 Total 최대   {df.total.max():.6f}  @ T{df.loc[df.total.idxmax(),'T']}_G{df.loc[df.total.idxmax(),'G']}")
print(f"FICR 최대         {df.ficr.max():.6f}  @ T{df.loc[df.ficr.idxmax(),'T']}_G{df.loc[df.ficr.idxmax(),'G']}"
      f"  (그때 1-NMAE {df.loc[df.ficr.idxmax(),'nmae']:.6f}, Total {df.loc[df.ficr.idxmax(),'total']:.6f})")
print(f"1-NMAE 최대       {df.nmae.max():.6f}  @ T{df.loc[df.nmae.idxmax(),'T']}_G{df.loc[df.nmae.idxmax(),'G']}"
      f"  (그때 FICR {df.loc[df.nmae.idxmax(),'ficr']:.6f})")

print(f"\n=== 리더보드 대비 ===")
print(f"  보드 최저 FICR   0.426650   우리 프론티어 최대 FICR {df.ficr.max():.6f}"
      f"  -> {'도달' if df.ficr.max()>=0.426650 else '미달 ' + f'{df.ficr.max()-0.426650:+.6f}'}")
print(f"  1위 FICR         0.467670   -> {'도달' if df.ficr.max()>=0.467670 else '미달 ' + f'{df.ficr.max()-0.467670:+.6f}'}")
print(f"  보드 최저 1-NMAE 0.869660   우리 프론티어 최대 {df.nmae.max():.6f}"
      f"  -> {'도달' if df.nmae.max()>=0.869660 else '미달 ' + f'{df.nmae.max()-0.869660:+.6f}'}")

best_online = df.total.max() + OFF_C
print(f"\n프론티어 최대 Total(로컬) {df.total.max():.6f} + 분류기오프셋 {OFF_C} = 예측온라인 {best_online:.6f}")
print(f"목표 0.66 까지 {0.66-best_online:+.6f}")
verdict = ("FRONTIER_REACHES_TARGET" if best_online >= 0.66 else
           "DECISION_LAYER_EXHAUSTED_GAP_IS_FORECAST_QUALITY")
print(f"\n판정: {verdict}")
Path("reports/D6_L1_policy_frontier.json").write_text(json.dumps(dict(
    node="D6_L1", grid=len(df), max_total=float(df.total.max()), max_ficr=float(df.ficr.max()),
    max_nmae=float(df.nmae.max()), predicted_online_best=best_online,
    board_min_ficr=0.426650, board_max_ficr=0.467670, board_min_nmae=0.869660,
    verdict=verdict, points=df.to_dict("records"),
    model_fits=0, dacon_upload=False), indent=1, ensure_ascii=False))
print("영수증 -> reports/D6_L1_policy_frontier.json")
