
"""D3_H2_2 — 평활 멤버(M244)에 경험적 잔차 결정층 적용 (적합 0회).

D3_H2_1 발견: M244 는 유일한 평활 계열(고유비 0.966)이고 오차상관 0.8436 으로 유일한 탈상관 멤버.
온라인 실측: M252(M244 계보) 1-NMAE 0.865903 (**우리 자산 중 최고**) / FICR 0.387853 (**최저**).
FICR 이 낮은 이유는 밴드를 조준하는 결정층이 없기 때문이다.

개입: M244 점예측 + fold-외 경험적 잔차 분위수 -> 예측분포 -> 기대정산금 최대화 행동.
     (D0_EMPIRICAL_RESIDUAL 이 v2 분포 작업에서 쓴 것과 같은 구성)
락박스 미접근 / 적합 0회 / 제출 없음.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path("."); sys.path.insert(0, "src")
from baram.evaluation.official import evaluate_official
from baram.constants import CAPACITIES_KWH

P = ROOT / "artifacts/backtests/metric-aligned-probe"
FOLDS = ("dev-2023-Q2", "dev-2023-Q3", "dev-2023-Q4")
KEY = ["forecast_id", "forecast_kst_dtm", "group_id"]
ACT = np.round(np.arange(0.075, 1.076, 0.0025), 6)
OFF_A = 0.021119
QS = np.round(np.arange(0.02, 0.99, 0.02), 3)   # 49 분위수

parts = {f: pd.read_parquet(P / f"M244_RARE_EVENT_CORRECTED_ANALOG_Q234-{f}.parquet")[
             KEY + ["actual_kwh", "prediction_kwh"]] for f in FOLDS}
allf = pd.concat(parts.values(), ignore_index=True)
allf["cap"] = allf.group_id.map(CAPACITIES_KWH)
print(f"M244 행 {len(allf)}")

base = evaluate_official(allf[KEY + ["actual_kwh", "prediction_kwh"]], CAPACITIES_KWH)
print(f"기준(점예측): Total {base.total:.6f} / 1-NMAE {base.one_minus_nmae:.6f} / FICR {base.ficr:.6f}")

def decide(pred_rate, resid_q, T, G, norm):
    """pred + 잔차분위수 = 예측분포, 기대정산금 최대화 행동."""
    sup = np.clip(pred_rate[:, None] + resid_q[None, :], 0.0, 1.10)
    w = np.full(len(resid_q), 1.0 / len(resid_q))
    if T != 1.0:
        w = np.power(w, 1.0 / T); w = w / w.sum()
    err = np.abs(ACT[None, :, None] - sup[:, None, :])
    units = np.select([err <= 0.06, err <= 0.08], [4.0, 3.0], default=0.0)
    baseu = -(err * w[None, None, :]).sum(2)
    settle = (sup[:, None, :] * units * w[None, None, :]).sum(2)
    return ACT[np.argmax(baseu + G * settle / (4.0 * norm), axis=1)]

print(f"\n{'T':>5} {'G':>5} {'Total':>10} {'1-NMAE':>10} {'FICR':>10}")
best = None
for T in (1.0,):
    for G in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0):
        frames = []
        for f in FOLDS:
            others = pd.concat([parts[o] for o in FOLDS if o != f], ignore_index=True)
            oc = others.group_id.map(CAPACITIES_KWH)
            rq = ((others.prediction_kwh - others.actual_kwh) / oc).quantile(QS).to_numpy()
            rq = -rq[::-1]                       # actual - pred 방향
            cur = parts[f].copy(); cap = cur.group_id.map(CAPACITIES_KWH).to_numpy(float)
            pr = cur.prediction_kwh.to_numpy(float) / cap
            rate_o = (others.actual_kwh / oc).to_numpy(float)
            yh = np.empty(len(cur))
            for gid in cur.group_id.unique():
                m = (cur.group_id == gid).to_numpy()
                mo = (others.group_id == gid).to_numpy()
                norm = float(np.mean(rate_o[mo]))
                yh[m] = decide(pr[m], rq, T, G, norm)
            d = cur[KEY + ["actual_kwh"]].copy(); d["prediction_kwh"] = yh * cap
            frames.append(d)
        s = evaluate_official(pd.concat(frames, ignore_index=True), CAPACITIES_KWH)
        tot = float(s.total)
        if best is None or tot > best[2]: best = (T, G, tot, float(s.one_minus_nmae), float(s.ficr))
        print(f"{T:5.1f} {G:5.1f} {tot:10.6f} {float(s.one_minus_nmae):10.6f} {float(s.ficr):10.6f}")

print(f"\n기준 점예측        Total {base.total:.6f}  FICR {base.ficr:.6f}")
print(f"결정층 최선 T{best[0]:g}_G{best[1]:g}  Total {best[2]:.6f}  1-NMAE {best[3]:.6f}  FICR {best[4]:.6f}")
print(f"  로컬 이득 {best[2]-float(base.total):+.6f}")
print(f"  아날로그 오프셋 적용 예측 온라인 {best[2]+OFF_A:.6f}   (M266 실측 0.6374709 대비 {best[2]+OFF_A-0.6374709:+.6f})")
verdict = "DECISION_LAYER_ON_SMOOTH_MEMBER_HELPS" if best[2] > float(base.total) + 0.001013 else "NO_GAIN"
print(f"\n판정: {verdict}")
Path("reports/D3_H2_2_analog_decision.json").write_text(json.dumps(dict(
    node="D3_H2_2", stage="D3", base=[float(base.total), float(base.one_minus_nmae), float(base.ficr)],
    best=dict(T=best[0], G=best[1], total=best[2], nmae=best[3], ficr=best[4]),
    local_gain=best[2]-float(base.total), predicted_online=best[2]+OFF_A, verdict=verdict,
    origin="research://Browell-2020-EEM20 [near_match_only] via D3_H2_1",
    model_fits=0, dacon_upload=False), indent=1, ensure_ascii=False))
print("영수증 -> reports/D3_H2_2_analog_decision.json")
