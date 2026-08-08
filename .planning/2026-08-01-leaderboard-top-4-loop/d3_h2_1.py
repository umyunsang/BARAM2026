
"""D3_H2_1 — 저장 후보의 이산/평활 특성 감사 (적합 0회).

근거: research://Browell 2020 EEM20 Strathclyde 74066 [near_match_only]
"이산 트리 기반 + 평활(GAM 계열) 두 모델의 분위수를 결합하면 개선"

C1N99 의 조합 축 폐쇄는 `ENSEMBLE_OVER_STORED_ARTIFACTS_CLOSED_NO_DIVERSITY` 이고
전제가 **현 12 후보 집합에만 걸린다**고 명시돼 있다. 평활 계열이 부재하면 전제를 건드린다.

이산성 지표: 예측값의 고유값 수 / 행 수. 트리 기반 46-bin 정책은 행동격자에 붙어 이산적이고,
평활 모델은 연속값을 낸다.
락박스 미접근 / 적합 0회 / 제출 없음.
"""
from __future__ import annotations
import sys, json, re
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path("."); sys.path.insert(0, "src")

P = ROOT / "artifacts/backtests/metric-aligned-probe"
FOLDS = ("dev-2023-Q2", "dev-2023-Q3", "dev-2023-Q4")
stems = {}
for f in P.glob("*-dev-2023-Q*.parquet"):
    m = re.match(r"(.+)-dev-2023-Q(\d)\.parquet$", f.name)
    if m and "policies" not in f.name: stems.setdefault(m.group(1), set()).add(m.group(2))
full = sorted(s for s, q in stems.items() if q >= {"2", "3", "4"})

rows = []
for s in full:
    try:
        d = pd.concat([pd.read_parquet(P / f"{s}-{f}.parquet") for f in FOLDS], ignore_index=True)
    except Exception:
        continue
    if "prediction_kwh" not in d.columns: continue
    v = d["prediction_kwh"].to_numpy(float)
    v = v[np.isfinite(v)]
    if len(v) == 0: continue
    uniq = len(np.unique(np.round(v, 6)))
    ratio = uniq / len(v)
    # 인접 고유값 간격의 규칙성: 격자에 붙으면 간격이 이산적
    su = np.unique(np.round(v, 6))
    gaps = np.diff(su)
    gap_cv = float(np.std(gaps) / np.mean(gaps)) if len(gaps) > 2 and np.mean(gaps) > 0 else float("nan")
    rows.append(dict(model=s, n=len(v), n_uniq=uniq, uniq_ratio=ratio, gap_cv=gap_cv))

t = pd.DataFrame(rows).sort_values("uniq_ratio")
print(f"{'모델':44s} {'행':>7} {'고유값':>7} {'고유비':>8} {'간격CV':>8}  특성")
for _, r in t.iterrows():
    char = "이산(격자)" if r.uniq_ratio < 0.10 else ("준연속" if r.uniq_ratio < 0.6 else "평활(연속)")
    print(f"{r.model[:44]:44s} {int(r.n):7d} {int(r.n_uniq):7d} {r.uniq_ratio:8.4f} {r.gap_cv:8.3f}  {char}")

smooth = t[t.uniq_ratio >= 0.6]
discrete = t[t.uniq_ratio < 0.10]
print(f"\n이산(격자) {len(discrete)}종 / 준연속 {len(t)-len(discrete)-len(smooth)}종 / 평활(연속) {len(smooth)}종")
if len(smooth) == 0:
    verdict = "NO_SMOOTH_MEMBER_PREMISE_TOUCHED"
    print("-> 평활 계열 멤버가 **부재**. C1N99 의 'NO_DIVERSITY' 전제가 걸린 후보집합에")
    print("   Browell 이 말한 smooth 상대가 없다. 전제를 건드릴 수 있는 유일한 경로.")
else:
    verdict = "SMOOTH_MEMBER_EXISTS_PREMISE_HOLDS"
    print(f"-> 평활 계열 존재: {list(smooth.model)}")
print(f"\n판정: {verdict}")
Path("reports/D3_H2_1_smooth_audit.json").write_text(json.dumps(dict(
    node="D3_H2_1", stage="D3", models=t.to_dict("records"),
    discrete=len(discrete), smooth=len(smooth), verdict=verdict,
    origin="research://Browell-2020-EEM20-Strathclyde-74066 [near_match_only]",
    model_fits=0, dacon_upload=False), indent=1, ensure_ascii=False, default=str))
print("영수증 -> reports/D3_H2_1_smooth_audit.json")
