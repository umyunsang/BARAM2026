
"""N519 — 오프셋 인지 다중멤버 결합 최적화 (적합 0회).

멤버 14 종의 OOF 를 그룹별로 결합한다. 목적함수는 **로컬 점수가 아니라 예측 온라인 점수**:
    predicted = local_component(w) + sum_i w_i * offset(class_i)
offset: analog/retrieval 0.021119, classifier 0.006554 (두 실측 앵커로 검증됨, N517)

프로젝트가 조합 축을 닫은 근거는 **로컬** 상한이 문턱 미만이라는 것이었다. 오프셋을 목적에
넣으면 아날로그 계열의 가치가 3.2 배로 재평가되므로 결론이 달라질 수 있다.

탐색은 그룹별 독립(지표가 그룹 분해되므로 유효). 좌표하강 2 회, 가중 격자 0.05.
락박스 미접근 / 적합 0회 / 업로드 없음.
"""
from __future__ import annotations
import sys, json, hashlib, itertools
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

frames = {}
for s in MEMBERS:
    d = pd.concat([pd.read_parquet(P / f"{s}-{f}.parquet") for f in FOLDS], ignore_index=True)
    frames[s] = d[KEY + ["actual_kwh", "prediction_kwh"]].rename(columns={"prediction_kwh": s})

base = frames[MEMBERS[0]]
for s in MEMBERS[1:]:
    base = base.merge(frames[s][KEY + [s]], on=KEY, how="inner")
print(f"공통행 {len(base)}  멤버 {len(MEMBERS)}")
for s in MEMBERS:
    assert base[s].notna().all(), f"{s} NaN"
print("상관(M115 대비):", {s: round(float(np.corrcoef(base.M115_XGBOOST, base[s])[0,1]), 4) for s in MEMBERS})

def comp(sub, g, wts):
    pred = np.zeros(len(sub))
    for s, w in wts.items():
        if w: pred += w * sub[s].to_numpy(float)
    pred = np.clip(pred, 0, CAPACITIES_KWH[g])
    d = sub[KEY + ["actual_kwh"]].copy(); d["prediction_kwh"] = pred
    gs = evaluate_group_component(d, g, CAPACITIES_KWH[g])
    loc = 0.5 * (1.0 - float(gs.nmae)) + 0.5 * float(gs.ficr)
    off = sum(w * (OFF_A if s in ANALOG else OFF_C) for s, w in wts.items())
    return loc, loc + off

GRID = np.round(np.arange(0.0, 1.001, 0.05), 2)
best_w = {}
print(f"\n{'group':>6} {'최적 가중':>60} {'로컬':>9} {'예측온라인':>10}")
for g in (1, 2, 3):
    sub = base[base.group_id == g].reset_index(drop=True)
    w = {s: (1.0 if s == "M115_XGBOOST" else 0.0) for s in MEMBERS}
    bestp = comp(sub, g, w)[1]
    for _ in range(2):
        for s in MEMBERS:
            for v in GRID:
                trial = dict(w); trial[s] = float(v)
                tot = sum(trial.values())
                if tot <= 0: continue
                trial = {k: x / tot for k, x in trial.items()}
                _, p = comp(sub, g, trial)
                if p > bestp + 1e-9: bestp, w = p, trial
    best_w[g] = {k: round(v, 3) for k, v in w.items() if v > 0.001}
    l, p = comp(sub, g, w)
    print(f"{g:6d} {str(best_w[g])[:60]:60s} {l:9.6f} {p:10.6f}")

parts, off_tot = [], 0.0
for g in (1, 2, 3):
    sub = base[base.group_id == g].reset_index(drop=True)
    w = {s: best_w[g].get(s, 0.0) for s in MEMBERS}
    pred = sum(v * sub[s].to_numpy(float) for s, v in w.items() if v)
    d = sub[KEY + ["actual_kwh"]].copy(); d["prediction_kwh"] = np.clip(pred, 0, CAPACITIES_KWH[g])
    parts.append(d); off_tot += sum(v * (OFF_A if s in ANALOG else OFF_C) for s, v in w.items()) / 3.0
s_all = evaluate_official(pd.concat(parts, ignore_index=True), CAPACITIES_KWH)
print(f"\n전체 로컬 {float(s_all.total):.6f}  오프셋 {off_tot:.6f}  **예측 온라인 {float(s_all.total)+off_tot:.6f}**")
print(f"현재 최고 M266 0.6374709  /  N518 그룹별 2멤버 예측 0.641016")
Path("reports/n519_multimember_blend.json").write_text(json.dumps(dict(
    node="N519_MULTIMEMBER_BLEND", members=MEMBERS, best_weights=best_w,
    local_total=float(s_all.total), offset=off_tot, predicted_online=float(s_all.total) + off_tot,
    model_fits=0, dacon_upload=False), indent=1, ensure_ascii=False))
print("영수증 -> reports/n519_multimember_blend.json")
