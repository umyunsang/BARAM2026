
"""N522 — 오염 제거 그룹별 결합 최적화 + fold-외 검증 (적합 0회).

N518 은 오염된 M102 prediction_kwh 로 그룹별 가중을 냈다. 여기서는 단일 고정 정책
(T0.5_G1.5) 입력으로 다시 최적화하고, **fold-외로 검증**한 뒤에만 후보를 만든다.
"""
from __future__ import annotations
import sys, json, hashlib
from pathlib import Path
import pandas as pd, numpy as np
ROOT = Path("."); sys.path.insert(0, "src")
from baram.evaluation.official import evaluate_group_component, evaluate_official
from baram.constants import CAPACITIES_KWH

P = ROOT / "artifacts/backtests/metric-aligned-probe"
FOLDS = ("dev-2023-Q2", "dev-2023-Q3", "dev-2023-Q4")
KEY = ["forecast_id", "forecast_kst_dtm", "group_id"]
OFF_A, OFF_C = 0.021119, 0.006554
POLICY = "T0.5_G1.5"
GRID = np.round(np.arange(0.0, 1.001, 0.05), 2)

parts = {}
for f in FOLDS:
    main = pd.read_parquet(P / f"M102_TOP100-{f}.parquet")
    pol = pd.read_parquet(P / f"M102_TOP100-{f}-policies.parquet")
    assert POLICY in pol.columns and main["forecast_id"].equals(pol["forecast_id"])
    ana = pd.read_parquet(P / f"M244_RARE_EVENT_CORRECTED_ANALOG_Q234-{f}.parquet")[KEY + ["prediction_kwh"]]
    d = main[KEY + ["actual_kwh"]].copy(); d["C"] = pol[POLICY].to_numpy(float)
    parts[f] = d.merge(ana.rename(columns={"prediction_kwh": "A"}), on=KEY, how="inner")

def comp(df, g, w):
    sub = df[df.group_id == g]
    if len(sub) == 0: return None
    pred = np.clip(w * sub.C.to_numpy(float) + (1 - w) * sub.A.to_numpy(float), 0, CAPACITIES_KWH[g])
    d = sub[KEY + ["actual_kwh"]].copy(); d["prediction_kwh"] = pred
    gs = evaluate_group_component(d, g, CAPACITIES_KWH[g])
    loc = 0.5 * (1.0 - float(gs.nmae)) + 0.5 * float(gs.ficr)
    return loc, loc + w * OFF_C + (1 - w) * OFF_A

allf = pd.concat(parts.values(), ignore_index=True)
print(f"{'group':>6} {'인샘플 최적w':>12} {'fold-외 w (Q2/Q3/Q4)':>24}")
insample, foldout = {}, {}
for g in (1, 2, 3):
    ins = max(GRID, key=lambda w: comp(allf, g, w)[1]); insample[g] = float(ins)
    fo = {}
    for f in FOLDS:
        others = pd.concat([parts[o] for o in FOLDS if o != f], ignore_index=True)
        fo[f] = float(max(GRID, key=lambda w: comp(others, g, w)[1]))
    foldout[g] = fo
    print(f"{g:6d} {ins:12.2f} {str([fo[f] for f in FOLDS]):>24}")

def assemble(wsel):
    frames, off = [], 0.0
    for f in FOLDS:
        for g in (1, 2, 3):
            w = wsel(f, g)
            sub = parts[f][parts[f].group_id == g]
            d = sub[KEY + ["actual_kwh"]].copy()
            d["prediction_kwh"] = np.clip(w * sub.C.to_numpy(float) + (1 - w) * sub.A.to_numpy(float), 0, CAPACITIES_KWH[g])
            frames.append(d); off += (w * OFF_C + (1 - w) * OFF_A) / 9.0
    s = evaluate_official(pd.concat(frames, ignore_index=True), CAPACITIES_KWH)
    return float(s.total), float(s.total) + off

l_ins, p_ins = assemble(lambda f, g: insample[g])
l_fo, p_fo = assemble(lambda f, g: foldout[g][f])
l_uni, p_uni = assemble(lambda f, g: 0.70)
l_pure, p_pure = assemble(lambda f, g: 1.0)
print(f"\n{'구성':22s} {'로컬':>10} {'예측 온라인':>12}")
for lab, (l, p) in (("인샘플 그룹별", (l_ins, p_ins)), ("fold-외 그룹별", (l_fo, p_fo)),
                    ("균일 w=0.70", (l_uni, p_uni)), ("분류기 단독 w=1.0", (l_pure, p_pure))):
    print(f"{lab:22s} {l:10.6f} {p:12.6f}")
print(f"\n현재 최고 M266 0.6374709")
ok = p_fo > p_uni and p_fo > p_pure
print(f"fold-외 그룹별이 균일·단독을 모두 상회 -> {ok}")

if ok:
    SUB = ROOT / "artifacts/submissions"; OUT = SUB / "blends"
    a = pd.read_csv(SUB / "submission_M252.csv", encoding="utf-8-sig")
    c = pd.read_csv(SUB / "submission_M261.csv", encoding="utf-8-sig")
    COLS = ["kpx_group_1", "kpx_group_2", "kpx_group_3"]
    out = c[["forecast_id", "forecast_kst_dtm"]].copy()
    wg = {g: float(np.mean([foldout[g][f] for f in FOLDS])) for g in (1, 2, 3)}
    for i, col in enumerate(COLS, 1):
        w = wg[i]
        out[col] = np.clip(w * c[col].to_numpy(float) + (1 - w) * a[col].to_numpy(float), 0, CAPACITIES_KWH[i])
    name = f"BLEND_CLEAN_PG_{int(wg[1]*100)}_{int(wg[2]*100)}_{int(wg[3]*100)}.csv"
    p = OUT / name; out.to_csv(p, index=False, encoding="utf-8-sig")
    raw = p.read_bytes(); sha = hashlib.sha256(raw).hexdigest()
    print(f"생성 {name}  sha {sha[:12]}  가중 {wg}")
else:
    wg, name, sha = None, None, None
    print("fold-외 통과 실패 -> 후보 생성 안 함")

Path("reports/n522_clean_pergroup.json").write_text(json.dumps(dict(
    node="N522_CLEAN_PERGROUP", policy=POLICY, insample_w=insample,
    foldout_w={str(g): foldout[g] for g in foldout},
    results=dict(insample=[l_ins, p_ins], foldout=[l_fo, p_fo], uniform=[l_uni, p_uni], pure=[l_pure, p_pure]),
    foldout_beats_baselines=bool(ok), candidate=dict(file=name, sha256=sha, weights=wg),
    model_fits=0, dacon_upload=False), indent=1, ensure_ascii=False, default=str))
print("영수증 -> reports/n522_clean_pergroup.json")
