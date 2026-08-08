
"""N520 — N519 다중멤버 결합의 fold-외 검증 (적합 0회).

N519 는 같은 데이터에서 21 자유도를 맞췄다. 선택편향 상한이므로 fold-외로 재판정한다.
각 fold 를 뺀 나머지에서 가중을 적합하고 그 fold 에서 평가한다. 통과하지 못하면 N519 는 폐기.

락박스 미접근 / 적합 0회 / 업로드 없음.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import pandas as pd, numpy as np
ROOT = Path("."); sys.path.insert(0, "src")
from baram.evaluation.official import evaluate_group_component, evaluate_official
from baram.constants import CAPACITIES_KWH

P = ROOT / "artifacts/backtests/metric-aligned-probe"
FOLDS = ("dev-2023-Q2", "dev-2023-Q3", "dev-2023-Q4")
KEY = ["forecast_id", "forecast_kst_dtm", "group_id"]
ANALOG = {"M244_RARE_EVENT_CORRECTED_ANALOG_Q234"}
OFF_A, OFF_C = 0.021119, 0.006554
MEMBERS = ["M102_TOP100", "M113_LGBM_DART", "M115_XGBOOST", "M129_GROUP_FINETUNE",
           "M244_RARE_EVENT_CORRECTED_ANALOG_Q234", "M93_POWER_QUANTILE", "M98_ORDINAL_BIN025"]

parts = {}
for f in FOLDS:
    d = None
    for s in MEMBERS:
        x = pd.read_parquet(P / f"{s}-{f}.parquet")[KEY + ["actual_kwh", "prediction_kwh"]].rename(
            columns={"prediction_kwh": s})
        d = x if d is None else d.merge(x[KEY + [s]], on=KEY, how="inner")
    parts[f] = d
print({f: len(parts[f]) for f in FOLDS})

GRID = np.round(np.arange(0.0, 1.001, 0.05), 2)

def comp(df, g, w):
    sub = df[df.group_id == g]
    if len(sub) == 0: return None
    pred = sum(v * sub[s].to_numpy(float) for s, v in w.items() if v)
    d = sub[KEY + ["actual_kwh"]].copy(); d["prediction_kwh"] = np.clip(pred, 0, CAPACITIES_KWH[g])
    gs = evaluate_group_component(d, g, CAPACITIES_KWH[g])
    loc = 0.5 * (1.0 - float(gs.nmae)) + 0.5 * float(gs.ficr)
    off = sum(v * (OFF_A if s in ANALOG else OFF_C) for s, v in w.items())
    return loc, loc + off

def fit(df, g):
    w = {s: (1.0 if s == "M115_XGBOOST" else 0.0) for s in MEMBERS}
    best = comp(df, g, w)[1]
    for _ in range(2):
        for s in MEMBERS:
            for v in GRID:
                t = dict(w); t[s] = float(v); tot = sum(t.values())
                if tot <= 0: continue
                t = {k: x / tot for k, x in t.items()}
                r = comp(df, g, t)
                if r and r[1] > best + 1e-9: best, w = r[1], t
    return w

held = []
print(f"\n{'fold':>14} {'group':>6} {'held 로컬':>10} {'M115 단독':>10} {'델타':>10}")
for f in FOLDS:
    others = pd.concat([parts[o] for o in FOLDS if o != f], ignore_index=True)
    for g in (1, 2, 3):
        w = fit(others, g)
        r_bl = comp(parts[f], g, w)
        r_base = comp(parts[f], g, {s: (1.0 if s == "M115_XGBOOST" else 0.0) for s in MEMBERS})
        held.append((f, g, w, r_bl[0], r_base[0]))
        print(f"{f[-2:]:>14} {g:6d} {r_bl[0]:10.6f} {r_base[0]:10.6f} {r_bl[0]-r_base[0]:+10.6f}")

# fold-외 가중으로 전체 재구성
frames, off_tot = [], 0.0
for f in FOLDS:
    others = pd.concat([parts[o] for o in FOLDS if o != f], ignore_index=True)
    for g in (1, 2, 3):
        w = fit(others, g)
        sub = parts[f][parts[f].group_id == g]
        pred = sum(v * sub[s].to_numpy(float) for s, v in w.items() if v)
        d = sub[KEY + ["actual_kwh"]].copy(); d["prediction_kwh"] = np.clip(pred, 0, CAPACITIES_KWH[g])
        frames.append(d)
        off_tot += sum(v * (OFF_A if s in ANALOG else OFF_C) for s, v in w.items()) / 9.0
s = evaluate_official(pd.concat(frames, ignore_index=True), CAPACITIES_KWH)
base_frames = []
for f in FOLDS:
    d = parts[f][KEY + ["actual_kwh"]].copy(); d["prediction_kwh"] = parts[f]["M115_XGBOOST"]
    base_frames.append(d)
sb = evaluate_official(pd.concat(base_frames, ignore_index=True), CAPACITIES_KWH)
print(f"\nfold-외 결합  로컬 {float(s.total):.6f}  오프셋 {off_tot:.6f}  예측온라인 {float(s.total)+off_tot:.6f}")
print(f"M115 단독     로컬 {float(sb.total):.6f}  오프셋 {OFF_C:.6f}  예측온라인 {float(sb.total)+OFF_C:.6f}")
print(f"인샘플(N519) 예측온라인 0.646821  ->  fold-외 {float(s.total)+off_tot:.6f}  차 {float(s.total)+off_tot-0.646821:+.6f}")
print(f"\n현재 최고 M266 0.6374709  대비 {float(s.total)+off_tot-0.6374709:+.6f}")
Path("reports/n520_foldout_blend.json").write_text(json.dumps(dict(
    node="N520_FOLDOUT_BLEND", foldout_local=float(s.total), offset=off_tot,
    foldout_predicted_online=float(s.total) + off_tot, insample_predicted=0.646821,
    m115_local=float(sb.total), m115_predicted=float(sb.total) + OFF_C,
    model_fits=0, dacon_upload=False), indent=1, ensure_ascii=False))
print("영수증 -> reports/n520_foldout_blend.json")
