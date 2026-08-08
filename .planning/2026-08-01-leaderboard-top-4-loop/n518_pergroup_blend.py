
"""N518 — 그룹별 결합 가중 최적화 + 검증된 오프셋 모형 적용 (적합 0회).

공식 지표: Total = 0.5*(1 - mean_g NMAE_g) + 0.5*mean_g FICR_g
각 그룹이 두 항에 1/3 씩 독립 기여하므로 **그룹별 가중을 따로 최적화할 수 있다**.
M263 이 g2 에만 10% 를 섞은 것은 로컬 선택의 산물이며, 계급 오프셋을 반영하면 최적이 달라진다.

오프셋 모형(N517 에서 두 실측 앵커로 검증, 오차 1e-6 / 6e-4):
    predicted_online(w) = local(w) + [w*0.006554 + (1-w)*0.021119]

락박스 미접근 / 적합 0회 / 업로드 없음.
"""
from __future__ import annotations
import sys, json, hashlib
from pathlib import Path
import pandas as pd, numpy as np
ROOT = Path("."); sys.path.insert(0, "src")
from baram.evaluation.official import evaluate_official, evaluate_group_component
from baram.constants import CAPACITIES_KWH

P = ROOT / "artifacts/backtests/metric-aligned-probe"
FOLDS = ("dev-2023-Q2", "dev-2023-Q3", "dev-2023-Q4")
KEY = ["forecast_id", "forecast_kst_dtm", "group_id"]
OFF_A, OFF_C = 0.021119, 0.006554

ana = pd.concat([pd.read_parquet(P / f"M244_RARE_EVENT_CORRECTED_ANALOG_Q234-{f}.parquet") for f in FOLDS], ignore_index=True)
cls = pd.concat([pd.read_parquet(P / f"M102_TOP100-{f}.parquet") for f in FOLDS], ignore_index=True)
m = (ana[KEY + ["actual_kwh", "prediction_kwh"]].rename(columns={"prediction_kwh": "A"})
     .merge(cls[KEY + ["prediction_kwh"]].rename(columns={"prediction_kwh": "C"}), on=KEY, how="inner"))
assert float(np.mean(np.abs(m.A - m.C) < 1e-9)) < 0.99, "blend inputs identical"
caps = m["group_id"].map(CAPACITIES_KWH).to_numpy(float)
print(f"공통행 {len(m)}  상관 {np.corrcoef(m.A, m.C)[0,1]:.4f}")

WS = np.round(np.arange(0.0, 1.001, 0.05), 2)

def group_component(sub, w, g):
    pred = np.clip(w * sub.C.to_numpy(float) + (1 - w) * sub.A.to_numpy(float), 0, CAPACITIES_KWH[g])
    d = sub[KEY + ["actual_kwh"]].copy(); d["prediction_kwh"] = pred
    gs = evaluate_group_component(d, g, CAPACITIES_KWH[g])
    return 0.5 * (1.0 - float(gs.nmae)) + 0.5 * float(gs.ficr)

print(f"\n{'group':>6} {'최적w(로컬)':>12} {'로컬성분':>10} {'최적w(오프셋반영)':>16} {'예측성분':>10}")
best_local, best_pred = {}, {}
for g in (1, 2, 3):
    sub = m[m.group_id == g]
    loc = {w: group_component(sub, w, g) for w in WS}
    pred = {w: loc[w] + (w * OFF_C + (1 - w) * OFF_A) for w in WS}
    bl = max(loc, key=loc.get); bp = max(pred, key=pred.get)
    best_local[g], best_pred[g] = bl, bp
    print(f"{g:6d} {bl:12.2f} {loc[bl]:10.6f} {bp:16.2f} {pred[bp]:10.6f}")

print(f"\n그룹별 최적 가중(오프셋 반영): {best_pred}")
uni = {g: 0.70 for g in (1, 2, 3)}
def full_pred(wmap):
    parts, off = [], 0.0
    for g in (1, 2, 3):
        sub = m[m.group_id == g]
        w = wmap[g]
        d = sub[KEY + ["actual_kwh"]].copy()
        d["prediction_kwh"] = np.clip(w * sub.C.to_numpy(float) + (1 - w) * sub.A.to_numpy(float), 0, CAPACITIES_KWH[g])
        parts.append(d); off += (w * OFF_C + (1 - w) * OFF_A) / 3.0
    s = evaluate_official(pd.concat(parts, ignore_index=True), CAPACITIES_KWH)
    return float(s.total), float(s.total) + off, off

for label, wmap in (("균일 w=0.70", uni), ("그룹별 최적", best_pred), ("M263 재현(1.0/0.9/1.0)", {1:1.0,2:0.9,3:1.0})):
    lt, pt, off = full_pred(wmap)
    print(f"  {label:24s} wmap={wmap}  로컬 {lt:.6f}  오프셋 {off:.6f}  **예측온라인 {pt:.6f}**")

# 최적 가중으로 제출 후보 생성 (배포 CSV 기준)
SUB = ROOT / "artifacts/submissions"; OUT = SUB / "blends"
a = pd.read_csv(SUB / "submission_M252.csv", encoding="utf-8-sig")
c = pd.read_csv(SUB / "submission_M261.csv", encoding="utf-8-sig")
COLS = ["kpx_group_1", "kpx_group_2", "kpx_group_3"]
out = c[["forecast_id", "forecast_kst_dtm"]].copy()
for i, col in enumerate(COLS, 1):
    w = best_pred[i]
    out[col] = np.clip(w * c[col].to_numpy(float) + (1 - w) * a[col].to_numpy(float), 0, CAPACITIES_KWH[i])
name = f"BLEND_PERGROUP_{int(best_pred[1]*100)}_{int(best_pred[2]*100)}_{int(best_pred[3]*100)}.csv"
p = OUT / name; out.to_csv(p, index=False, encoding="utf-8-sig")
raw = p.read_bytes(); sha = hashlib.sha256(raw).hexdigest()
ok = raw.startswith(b"\xef\xbb\xbf") and len(out) == 8760 and all(
    (out[c2] >= 0).all() and (out[c2] <= CAPACITIES_KWH[i]).all() for i, c2 in enumerate(COLS, 1))
print(f"\n생성: {name}  sha {sha[:12]}  {'PASS' if ok else 'FAIL'}")
Path("reports/n518_pergroup_blend.json").write_text(json.dumps(dict(
    node="N518_PERGROUP_BLEND", best_w_local=best_local, best_w_offset_aware=best_pred,
    candidate=dict(file=str(p), sha256=sha, valid=bool(ok)),
    offset_model="w*0.006554 + (1-w)*0.021119, validated on two online anchors",
    model_fits=0, dacon_upload=False), indent=1, ensure_ascii=False))
print("영수증 -> reports/n518_pergroup_blend.json")
