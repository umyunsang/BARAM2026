
"""D1_F1_1 — NWP 풍속의 분위수 편향 진단 (적합 0회, 재적합 게이트용).

근거: research://Spiliotis 2025 S2213138825004308 [near_match_only]
우리 MOS 계열(C1N21/22/23)은 전부 **출력** MOS 였고 NO_STABLE_METRIC_ALIGNED_BIAS 로 폐기됐다.
**풍속 단계** 보정은 기록이 없다. 다만 보정할 편향이 없다면 quantile mapping 은 무의미하므로
재적합 전에 편향 존재 여부를 먼저 잰다.

진리값: scada_ws (학습기간 전용, 진단 목적. 프로젝트가 C1N51/C1N54 에서 쓴 것과 동일 용법)
락박스 미접근 / 적합 0회 / 제출 없음.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path("."); sys.path.insert(0, "src")

SURF = ROOT / "artifacts/cache/m271_decision_surface/195531818b8a"
cands = sorted((ROOT / "artifacts/cache/m271_decision_surface").glob("*"))
SURF = next(c for c in cands if (c / "dev-2023-Q3__arrays.npz").exists()
            and "scada_ws" in np.load(c / "dev-2023-Q3__arrays.npz").files)
FOLDS = ("dev-2023-Q2", "dev-2023-Q3", "dev-2023-Q4")
print(f"표면 {SURF.name}")

rows = []
for f in FOLDS:
    z = np.load(SURF / f"{f}__arrays.npz")
    keys = set(z.files)
    d = {k: z[k] for k in ("sitewind", "scada_ws", "group") if k in keys}
    if "sitewind_allweather" in keys: d["sitewind_allweather"] = z["sitewind_allweather"]
    rows.append(pd.DataFrame(d))
df = pd.concat(rows, ignore_index=True)
df = df[np.isfinite(df.sitewind) & np.isfinite(df.scada_ws)]
print(f"유효행 {len(df)}  변수 {list(df.columns)}")

for col in [c for c in df.columns if c.startswith("sitewind")]:
    e = df[col] - df.scada_ws
    print(f"\n=== {col} vs scada_ws ===")
    print(f"  전체 편향(mean err) {e.mean():+.4f} m/s   sigma {e.std():.4f}")
    print(f"  {'분위수':>8} {'예측':>8} {'실측':>8} {'분위수편향':>10} {'구간내 mean err':>14}")
    qs = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
    qp = df[col].quantile(qs).to_numpy(); qa = df.scada_ws.quantile(qs).to_numpy()
    for q, a, b in zip(qs, qp, qa):
        m = (df[col] >= df[col].quantile(max(0, q - 0.05))) & (df[col] < df[col].quantile(min(1, q + 0.05)))
        me = float((df.loc[m, col] - df.loc[m, "scada_ws"]).mean()) if m.sum() else float("nan")
        print(f"  {q:8.2f} {a:8.3f} {b:8.3f} {a-b:+10.4f} {me:+14.4f}")
    # quantile mapping 적용 후 sigma
    from numpy import interp
    grid = np.linspace(0.001, 0.999, 199)
    src = df[col].quantile(grid).to_numpy(); tgt = df.scada_ws.quantile(grid).to_numpy()
    mapped = np.interp(df[col].to_numpy(), src, tgt)
    e2 = mapped - df.scada_ws.to_numpy()
    print(f"  quantile mapping 후: 편향 {e2.mean():+.4f}  sigma {e2.std():.4f}  "
          f"(감소 {100*(1-e2.std()/e.std()):+.2f}%)")
    rows_out = dict(column=col, bias=float(e.mean()), sigma=float(e.std()),
                    sigma_after_qmap=float(e2.std()),
                    sigma_reduction=float(1 - e2.std() / e.std()))
    Path(f"reports/D1_F1_1_wind_bias_{col}.json").write_text(json.dumps(rows_out, indent=1))

print("\n주: quantile mapping 은 **주변분포**만 맞춘다. 조건부 오차(sigma)가 줄지 않으면")
print("    하류 점수도 오르지 않는다 — 그것이 이 진단의 판정 기준이다.")
