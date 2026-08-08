
"""D2_G3_1 — 시각(hour-of-day) 조건부 예보-실측 정렬 오차 진단 (적합 0회).

근거: research://arXiv:2510.15474v1 [near_match_only] + A4_error(LIVE, '일주기 편향, 240도 43%')
질문: 밴드 적중률이 시각에 따라 체계적으로 다른가? 다르다면 시각 조건부 개입 여지가 있는가?
주의: M277 이 그룹x수준 3분할 조건부 캘리브레이션에서 fold-외 붕괴를 보였다. 24 셀은 더 나쁘다.
      따라서 이 노드는 **진단**이며, 개입은 편향이 충분히 크고 안정적일 때만 제안한다.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path("."); sys.path.insert(0, "src")
from baram.constants import CAPACITIES_KWH

SRC = ROOT / "artifacts/backtests/m269-probe"
FOLDS = ("dev-2023-Q2", "dev-2023-Q3", "dev-2023-Q4")
ACTIONS = np.round(np.arange(0.075, 1.076, 0.0025), 6)
DEP = (0.5, 1.5)

rows = []
for f in FOLDS:
    z = np.load(SRC / f"M269_PROBE-{f}-probability.npz", allow_pickle=True)
    C = z["centers"]; cap = np.array([CAPACITIES_KWH[int(g)] for g in z["group_id"]], float)
    rate = z["actual_kwh"].astype(float) / cap
    err = np.abs(ACTIONS[:, None] - C[None, :])
    units = np.select([err <= 0.06, err <= 0.08], [4.0, 3.0], default=0.0)
    norms = {int(g): float(np.mean(rate[z["group_id"] == g])) for g in np.unique(z["group_id"])}
    cal = np.power(np.clip(z["probability"], 1e-12, None), 1.0 / DEP[0]); cal /= cal.sum(1, keepdims=True)
    base, settle = -(cal @ err.T), cal @ (C[None, :] * units).T
    yh = np.empty(len(cal))
    for gid in np.unique(z["group_id"]):
        m = z["group_id"] == gid
        yh[m] = ACTIONS[np.argmax(base[m] + DEP[1] * settle[m] / (4.0 * norms[int(gid)]), axis=1)]
    dt = pd.to_datetime(z["forecast_kst_dtm"])
    rows.append(pd.DataFrame(dict(fold=f, hour=dt.hour, group=z["group_id"].astype(int),
                                  rate=rate, pred=yh)))
df = pd.concat(rows, ignore_index=True)
df = df[np.isfinite(df.rate) & (df.rate >= 0.10)].copy()
df["err"] = (df.pred - df.rate).abs()
df["hit6"] = (df.err <= 0.06).astype(float)
df["signed"] = df.pred - df.rate
print(f"적격행 {len(df)}")

g = df.groupby("hour").agg(n=("hit6", "size"), hit6=("hit6", "mean"),
                           bias=("signed", "mean"), mae=("err", "mean")).reset_index()
print(f"\n{'시각':>4} {'n':>5} {'6%적중':>8} {'부호편향':>9} {'평균오차':>9}")
for _, r in g.iterrows():
    print(f"{int(r.hour):4d} {int(r.n):5d} {r.hit6:8.3f} {r.bias:+9.4f} {r.mae:9.4f}")

print(f"\n적중률 범위 {g.hit6.min():.3f} ~ {g.hit6.max():.3f}  (폭 {g.hit6.max()-g.hit6.min():.3f})")
print(f"부호편향 범위 {g.bias.min():+.4f} ~ {g.bias.max():+.4f}  (폭 {g.bias.max()-g.bias.min():.4f})")

# 폴드 간 안정성: 시각별 편향의 폴드 간 상관
piv = df.groupby(["fold", "hour"]).signed.mean().unstack(0)
cor = piv.corr()
print(f"\n시각별 부호편향의 폴드 간 상관:")
print(cor.round(3).to_string())
stable = float(cor.values[np.triu_indices(3, 1)].mean())
print(f"  평균 상관 {stable:.3f}")

# 개입 여지 판정
amp = float(g.bias.abs().max())
print(f"\n판정 기준: 편향 진폭이 밴드 반폭 0.06 대비 유의미하고 폴드 간 상관이 높아야 개입 가치")
print(f"  최대 |편향| {amp:.4f} = 밴드 반폭의 {amp/0.06:.1%}")
print(f"  폴드 간 상관 {stable:.3f}")
verdict = ("DIURNAL_BIAS_ACTIONABLE" if amp > 0.02 and stable > 0.5
           else "DIURNAL_BIAS_TOO_SMALL_OR_UNSTABLE")
print(f"\n판정: {verdict}")
Path("reports/D2_G3_1_diurnal.json").write_text(json.dumps(dict(
    node="D2_G3_1", stage="D2", rows=len(df),
    by_hour=g.to_dict("records"), hit_range=[float(g.hit6.min()), float(g.hit6.max())],
    bias_range=[float(g.bias.min()), float(g.bias.max())],
    fold_correlation=stable, max_abs_bias=amp, verdict=verdict,
    origin="research://arXiv:2510.15474v1 [near_match_only] + A4_error",
    model_fits=0, dacon_upload=False), indent=1, ensure_ascii=False))
print("영수증 -> reports/D2_G3_1_diurnal.json")
