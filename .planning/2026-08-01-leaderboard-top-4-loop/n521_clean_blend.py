
"""N521 — 오염 제거 결합 곡선: 분류기 멤버를 단일 고정 정책으로 (적합 0회).

N517/N518 은 `M102_TOP100-*.parquet` 의 `prediction_kwh` 를 썼는데, 그 컬럼은
폴드마다 다른 정책이었다 (Q2 T0.5_G1.5 / Q3 T0.4_G2 / Q4 T0.6_G1) — 사후 선택편향.
여기서는 `-policies.parquet` 에서 **전 폴드 동일 정책**을 뽑아 다시 잰다.

불변식: 선택한 정책 컬럼이 세 폴드 모두에 존재하고, 폴드 간 정책이 동일함을 assert.
락박스 미접근 / 적합 0회 / 업로드 없음.
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
OFF_A, OFF_C = 0.021119, 0.006554
POLICY = "T0.5_G1.5"   # 배포 M261 정책, 전 폴드 고정

cls_parts = []
for f in FOLDS:
    main = pd.read_parquet(P / f"M102_TOP100-{f}.parquet")
    pol = pd.read_parquet(P / f"M102_TOP100-{f}-policies.parquet")
    assert POLICY in pol.columns, f"{POLICY} missing in {f}"
    assert main["forecast_id"].equals(pol["forecast_id"]), "key mismatch"
    d = main[KEY + ["actual_kwh"]].copy()
    d["C"] = pol[POLICY].to_numpy(float)
    cls_parts.append(d)
cls = pd.concat(cls_parts, ignore_index=True)

ana = pd.concat([pd.read_parquet(P / f"M244_RARE_EVENT_CORRECTED_ANALOG_Q234-{f}.parquet") for f in FOLDS],
                ignore_index=True)[KEY + ["prediction_kwh"]].rename(columns={"prediction_kwh": "A"})

m = cls.merge(ana, on=KEY, how="inner")
assert float(np.mean(np.abs(m.A - m.C) < 1e-9)) < 0.99, "blend inputs identical"
caps = m["group_id"].map(CAPACITIES_KWH).to_numpy(float)
print(f"공통행 {len(m)}  정책 {POLICY} 고정  상관 {np.corrcoef(m.A, m.C)[0,1]:.4f}")

def score(w):
    d = m[KEY + ["actual_kwh"]].copy()
    d["prediction_kwh"] = np.clip(w * m.C.to_numpy(float) + (1 - w) * m.A.to_numpy(float), 0, caps)
    s = evaluate_official(d, CAPACITIES_KWH)
    return float(s.total), float(s.one_minus_nmae), float(s.ficr)

print(f"\n{'w(cls)':>7} {'로컬':>10} {'1-NMAE':>10} {'FICR':>10} {'혼합오프셋':>10} {'예측온라인':>11}")
best = None
curve = {}
for w in np.round(np.arange(0.0, 1.001, 0.1), 1):
    t, n, fi = score(w)
    off = w * OFF_C + (1 - w) * OFF_A
    p = t + off
    curve[float(w)] = dict(local=t, nmae=n, ficr=fi, offset=off, pred=p)
    if best is None or p > best[1]: best = (float(w), p)
    print(f"{w:7.1f} {t:10.6f} {n:10.6f} {fi:10.6f} {off:10.6f} {p:11.6f}")

print(f"\n=== 오프셋 모형 재검증 (오염 제거 후) ===")
print(f"  w=0.0  예측 {curve[0.0]['pred']:.6f}  실측 M252 0.6268784  오차 {curve[0.0]['pred']-0.6268784:+.6f}")
print(f"  w=1.0  예측 {curve[1.0]['pred']:.6f}  실측 M261 0.6365274  오차 {curve[1.0]['pred']-0.6365274:+.6f}")
print(f"\n최선 w={best[0]:.1f}  예측 온라인 {best[1]:.6f}   현재최고 M266 0.6374709 대비 {best[1]-0.6374709:+.6f}")
Path("reports/n521_clean_blend_curve.json").write_text(json.dumps(dict(
    node="N521_CLEAN_BLEND", policy=POLICY, rows=len(m), curve=curve,
    best_w=best[0], best_pred=best[1],
    validation={"w0_err": curve[0.0]["pred"] - 0.6268784, "w1_err": curve[1.0]["pred"] - 0.6365274},
    model_fits=0, dacon_upload=False), indent=1, ensure_ascii=False))
print("영수증 -> reports/n521_clean_blend_curve.json")
