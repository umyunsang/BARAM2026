
"""N517 — 아날로그 x 분류기 결합 곡선을 로컬에서 직접 측정 (적합 0회).

온라인 업로드 전에, 두 계보의 dev-2023 OOF 로 결합 곡선을 잰다.
결합이 로컬에서도 두 끝점을 넘으면 H1 이 온라인 오프셋과 무관하게 지지된다.
넘지 않으면, 온라인 우위 가설은 오직 계급 오프셋 차이에만 의존하게 된다.

락박스 미접근 / 적합 0회 / 제출·업로드 없음.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import pandas as pd, numpy as np
ROOT = Path("."); sys.path.insert(0, "src")
from baram.evaluation.official import evaluate_official
from baram.constants import CAPACITIES_KWH

P = ROOT / "artifacts/backtests/metric-aligned-probe"
FOLDS = ("dev-2023-Q2", "dev-2023-Q3", "dev-2023-Q4")
KEY = ["forecast_id", "forecast_kst_dtm", "group_id"]

def load(prefix):
    fr = []
    for f in FOLDS:
        p = P / f"{prefix}-{f}.parquet"
        if not p.exists(): return None, f"missing {p.name}"
        fr.append(pd.read_parquet(p))
    return pd.concat(fr, ignore_index=True), None

ana, e1 = load("M244_RARE_EVENT_CORRECTED_ANALOG_Q234")
cls, e2 = load("M102_TOP100")
for nm, d, e in (("analog", ana, e1), ("classifier", cls, e2)):
    print(f"{nm}: {'OK rows='+str(len(d)) if d is not None else e}  cols={list(d.columns)[:8] if d is not None else ''}")
if ana is None or cls is None: raise SystemExit(1)

def pick(df):
    pc = [c for c in df.columns if "prediction" in c.lower()]
    if not pc:
        pc = [c for c in df.columns if c.startswith("T") and "_G" in c]
    return pc

print("\nanalog 예측 컬럼:", pick(ana)[:6])
print("classifier 예측 컬럼:", pick(cls)[:6])

acol = "prediction_kwh" if "prediction_kwh" in ana.columns else pick(ana)[0]
ccol = "prediction_kwh" if "prediction_kwh" in cls.columns else "T0.5_G1.5" if "T0.5_G1.5" in cls.columns else pick(cls)[0]
print(f"사용: analog[{acol}]  classifier[{ccol}]")

cls_r = cls[KEY + [ccol]].rename(columns={ccol: "_pred_cls"})
m = ana[KEY + ["actual_kwh", acol]].rename(columns={acol: "_pred_ana"}).merge(cls_r, on=KEY, how="inner")
# INVARIANT: the two prediction columns must be distinct objects, else the blend is a no-op.
assert "_pred_ana" in m.columns and "_pred_cls" in m.columns
_ident = float(np.mean(np.abs(m["_pred_ana"].to_numpy(float) - m["_pred_cls"].to_numpy(float)) < 1e-9))
assert _ident < 0.99, f"blend inputs are identical on {_ident:.1%} of rows — merge/rename bug"
print(f"두 예측 동일행 비율 {_ident:.4f} (0 에 가까워야 정상)")
print(f"공통 키 {len(m)} (analog {len(ana)} / classifier {len(cls)})")
if len(m) == 0: raise SystemExit(1)

caps = m["group_id"].map(CAPACITIES_KWH).to_numpy(float)
A = m["_pred_ana"].to_numpy(float); C = m["_pred_cls"].to_numpy(float)
print(f"상관 {np.corrcoef(A, C)[0,1]:.4f}")

print(f"\n{'w(cls)':>7} {'Total':>10} {'1-NMAE':>10} {'FICR':>10}")
best = None
for w in np.round(np.arange(0.0, 1.01, 0.1), 2):
    d = m[KEY + ["actual_kwh"]].copy()
    d["prediction_kwh"] = np.clip(w * C + (1 - w) * A, 0, caps)
    s = evaluate_official(d, CAPACITIES_KWH)
    t = float(s.total)
    if best is None or t > best[1]: best = (w, t)
    print(f"{w:7.1f} {t:10.6f} {float(s.one_minus_nmae):10.6f} {float(s.ficr):10.6f}")
w0 = [x for x in [0.0]][0]
d0 = m[KEY + ["actual_kwh"]].copy(); d0["prediction_kwh"] = np.clip(A, 0, caps)
d1 = m[KEY + ["actual_kwh"]].copy(); d1["prediction_kwh"] = np.clip(C, 0, caps)
t_a = float(evaluate_official(d0, CAPACITIES_KWH).total)
t_c = float(evaluate_official(d1, CAPACITIES_KWH).total)
print(f"\n끝점: analog {t_a:.6f}  classifier {t_c:.6f}")
print(f"최선 결합: w={best[0]:.1f}  Total {best[1]:.6f}")
print(f"두 끝점 초과 -> {best[1] > max(t_a, t_c)}  (초과분 {best[1]-max(t_a,t_c):+.6f})")
Path("reports/n517_local_blend_curve.json").write_text(json.dumps(dict(
    node="N517_LOCAL_BLEND_CURVE", rows=len(m), analog_col=acol, classifier_col=ccol,
    endpoint_analog=t_a, endpoint_classifier=t_c, best_w=float(best[0]), best_total=float(best[1]),
    exceeds_both=bool(best[1] > max(t_a, t_c)),
    model_fits=0, lockbox_reopened=False, dacon_upload=False), indent=1, ensure_ascii=False))
print("영수증 -> reports/n517_local_blend_curve.json")
