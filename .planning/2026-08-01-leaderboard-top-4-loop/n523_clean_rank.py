
"""N523 — 오염 제거 계보 재순위: fold-외 단일정책 기준 (적합 0회).

발견: `prediction_kwh` 컬럼은 폴드(및 그룹)마다 다른 정책을 담고 있어 사후 선택편향이다.
프로젝트의 챔피언 선정이 그 값으로 이뤄졌다면 순위가 뒤집힐 수 있다.
여기서는 각 계보에 대해 **fold-외로 단일 정책을 선택**하고 그 정책의 held-out 성능만 집계한다.

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
OFF_C = 0.006554
STEMS = ["M102_TOP100", "M113_LGBM_DART", "M115_XGBOOST", "M96_ORDINAL_CUMULATIVE", "M84_LEAVES031",
         "M64B_ALLWEATHER_SITEWIND_CLASS", "M68_SITEWIND_CLASS_ITER", "M72_BIN020"]

def load(stem):
    out = {}
    for f in FOLDS:
        mp, pp = P / f"{stem}-{f}.parquet", P / f"{stem}-{f}-policies.parquet"
        if not (mp.exists() and pp.exists()): return None
        main, pol = pd.read_parquet(mp), pd.read_parquet(pp)
        if not main["forecast_id"].equals(pol["forecast_id"]): return None
        out[f] = (main[KEY + ["actual_kwh"]].reset_index(drop=True), pol)
    return out

def score_policy(store, folds, policy):
    fr = []
    for f in folds:
        base, pol = store[f]
        if policy not in pol.columns: return None
        d = base.copy(); d["prediction_kwh"] = pol[policy].to_numpy(float)
        fr.append(d)
    return float(evaluate_official(pd.concat(fr, ignore_index=True), CAPACITIES_KWH).total)

rows = []
for stem in STEMS:
    st = load(stem)
    if st is None: print(f"  {stem:34s} SKIP (정책파일 불완전)"); continue
    sets = [set(c for c in st[f][1].columns if c.startswith("T")) for f in FOLDS]
    cols = sorted(set.intersection(*sets))
    if not cols:
        print(f"  {stem:34s} SKIP (정책 컬럼 교집합 0)"); continue
    print(f"  {stem:34s} 정책 {len(cols)}개")
    # (a) 인샘플 최선 단일정책
    ins_pol = max(cols, key=lambda c: score_policy(st, FOLDS, c))
    ins = score_policy(st, FOLDS, ins_pol)
    # (b) fold-외 선택
    held = []
    for f in FOLDS:
        others = [o for o in FOLDS if o != f]
        best = max(cols, key=lambda c: score_policy(st, others, c))
        held.append((f, best))
    fr = []
    for f, pol_name in held:
        base, pol = st[f]
        d = base.copy(); d["prediction_kwh"] = pol[pol_name].to_numpy(float)
        fr.append(d)
    fo = float(evaluate_official(pd.concat(fr, ignore_index=True), CAPACITIES_KWH).total)
    # (c) 오염 컬럼 값 (참조)
    fr2 = [pd.read_parquet(P / f"{stem}-{f}.parquet")[KEY + ["actual_kwh", "prediction_kwh"]] for f in FOLDS]
    contaminated = float(evaluate_official(pd.concat(fr2, ignore_index=True), CAPACITIES_KWH).total)
    rows.append(dict(stem=stem, insample_policy=ins_pol, insample=ins, foldout=fo,
                     contaminated=contaminated, pred_online=fo + OFF_C,
                     foldout_policies={f: p for f, p in held}))

rows.sort(key=lambda r: -r["foldout"])
print(f"\n{'계보':34s} {'오염값':>9} {'인샘플':>9} {'fold-외':>9} {'예측온라인':>11} 정책")
for r in rows:
    print(f"{r['stem']:34s} {r['contaminated']:9.6f} {r['insample']:9.6f} {r['foldout']:9.6f} "
          f"{r['pred_online']:11.6f} {r['insample_policy']}")
best = rows[0]
print(f"\nfold-외 최선 계보: {best['stem']}  {best['foldout']:.6f}  예측온라인 {best['pred_online']:.6f}")
print(f"현재 최고 M266 실측 0.6374709  대비 {best['pred_online']-0.6374709:+.6f}")
print(f"\n오염값 기준 순위와 fold-외 순위 비교:")
by_cont = sorted(rows, key=lambda r: -r["contaminated"])
print(f"  오염 1위 {by_cont[0]['stem']}  /  fold-외 1위 {rows[0]['stem']}  ->  "
      f"{'동일' if by_cont[0]['stem']==rows[0]['stem'] else '**뒤집힘**'}")
Path("reports/n523_clean_lineage_rank.json").write_text(json.dumps(dict(
    node="N523_CLEAN_LINEAGE_RANK", offset=OFF_C, rows=rows,
    model_fits=0, dacon_upload=False), indent=1, ensure_ascii=False))
print("영수증 -> reports/n523_clean_lineage_rank.json")
