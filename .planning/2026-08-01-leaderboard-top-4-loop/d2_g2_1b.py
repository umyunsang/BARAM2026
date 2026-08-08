
"""D2_G2_1b — 후류 기하 피처의 잔차 설명력 (적합 0회).

D2_G2_1 이 풍향에 따라 비후류비율이 크게 변함을 확인했다(g3 변동폭 0.800).
이제 그 피처가 **배포 모델의 잔차를 설명하는지** 본다. 설명하지 못하면 모델이 이미
다른 경로로 포착한 것이고 축은 닫힌다.
락박스 미접근 / 적합 0회 / 제출 없음.
"""
from __future__ import annotations
import sys, json, re, zipfile, io, math
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path("."); sys.path.insert(0, "src")
from baram.constants import CAPACITIES_KWH

# --- 후류 테이블 재구성 (D2_G2_1 과 동일 로직) ---
def dms(s):
    m = re.match(r"""(\d+)°(\d+)'([\d.]+)"([NSEW])""", s.strip())
    v = int(m.group(1)) + int(m.group(2))/60 + float(m.group(3))/3600
    return -v if m.group(4) in "SW" else v
z = zipfile.ZipFile("inputs/competition/open_wind_236727.zip")
raw = pd.read_excel(io.BytesIO(z.read("info.xlsx"))); raw.columns = raw.iloc[2]
t = raw.iloc[3:].reset_index(drop=True); t = t[t["좌표(Google)"].notna()].copy()
t["KPX그룹"] = t["KPX그룹"].ffill().infer_objects(copy=False).astype(int)
c = t["좌표(Google)"].astype(str).str.split(" ", n=1, expand=True)
t["lat"], t["lon"] = c[0].map(dms), c[1].map(dms)
t["rotor"] = t["Rotor Diameter(m)"].astype(float)
lat0 = t.lat.mean()
t["x"] = (t.lon - t.lon.mean()) * 111320 * math.cos(math.radians(lat0))
t["y"] = (t.lat - t.lat.mean()) * 110540

def unwaked(group, th):
    sub = t[t["KPX그룹"] == group]
    xs, ys, rot = sub.x.to_numpy(), sub.y.to_numpy(), sub.rotor.to_numpy()
    wd = math.radians((th + 180) % 360); ux, uy = math.sin(wd), math.cos(wd)
    n = len(sub); w = np.zeros(n, bool)
    for i in range(n):
        for j in range(n):
            if i == j: continue
            dx, dy = xs[j]-xs[i], ys[j]-ys[i]
            down = dx*ux + dy*uy
            if down <= 0: continue
            cross = abs(-dx*uy + dy*ux); dist = math.hypot(dx, dy)
            if math.degrees(math.asin(min(1.0, cross/max(dist,1e-6)))) <= 15.0 and down/rot[i] <= 10.0:
                w[j] = True
    return 1.0 - w.mean()
LUT = {g: np.array([unwaked(g, th) for th in range(0, 360, 5)]) for g in (1, 2, 3)}

# --- 배포 예측 잔차 ---
SRC = ROOT / "artifacts/backtests/m269-probe"
FOLDS = ("dev-2023-Q2", "dev-2023-Q3", "dev-2023-Q4")
ACT = np.round(np.arange(0.075, 1.076, 0.0025), 6)
rows = []
for f in FOLDS:
    zz = np.load(SRC / f"M269_PROBE-{f}-probability.npz", allow_pickle=True)
    C = zz["centers"]; cap = np.array([CAPACITIES_KWH[int(g)] for g in zz["group_id"]], float)
    rate = zz["actual_kwh"].astype(float)/cap
    err = np.abs(ACT[:, None]-C[None, :])
    units = np.select([err <= 0.06, err <= 0.08], [4.0, 3.0], default=0.0)
    norms = {int(g): float(np.mean(rate[zz["group_id"] == g])) for g in np.unique(zz["group_id"])}
    cal = np.power(np.clip(zz["probability"], 1e-12, None), 2.0); cal /= cal.sum(1, keepdims=True)
    base, settle = -(cal @ err.T), cal @ (C[None, :]*units).T
    yh = np.empty(len(cal))
    for gid in np.unique(zz["group_id"]):
        m = zz["group_id"] == gid
        yh[m] = ACT[np.argmax(base[m] + 1.5*settle[m]/(4.0*norms[int(gid)]), axis=1)]
    rows.append(pd.DataFrame(dict(forecast_kst_dtm=pd.to_datetime(zz["forecast_kst_dtm"]),
                                  group_id=zz["group_id"].astype(int), rate=rate, pred=yh)))
res = pd.concat(rows, ignore_index=True)

# --- 풍향 결합 ---
feat = pd.read_parquet(ROOT/"artifacts/cache/920be0c458d820e855bf79dd25723146f52ce1736138aedba5e6bc853f1f720b/train_features.parquet",
                       columns=["forecast_kst_dtm","gfs__heightAboveGround_10_10u__mean","gfs__heightAboveGround_10_10v__mean"])
feat = feat.drop_duplicates("forecast_kst_dtm")
u = feat["gfs__heightAboveGround_10_10u__mean"].to_numpy(float)
v = feat["gfs__heightAboveGround_10_10v__mean"].to_numpy(float)
feat["wdir"] = (np.degrees(np.arctan2(-u, -v)) % 360)
df = res.merge(feat[["forecast_kst_dtm","wdir"]], on="forecast_kst_dtm", how="inner")
df = df[np.isfinite(df.rate) & (df.rate >= 0.10) & np.isfinite(df.wdir)].copy()
df["unwaked"] = [LUT[int(g)][int(w//5) % 72] for g, w in zip(df.group_id, df.wdir)]
df["signed"] = df.pred - df.rate
df["abserr"] = df.signed.abs()
df["hit6"] = (df.abserr <= 0.06).astype(float)
print(f"결합행 {len(df)}  풍향 유효 {df.wdir.notna().sum()}")

from scipy.stats import spearmanr, pearsonr
print(f"\n{'group':>6} {'n':>6} {'unwaked 범위':>16} {'rho(unwaked,|err|)':>20} {'p':>8} {'rho(unwaked,hit6)':>19}")
out = {}
for g in (1, 2, 3):
    s = df[df.group_id == g]
    r1 = spearmanr(s.unwaked, s.abserr); r2 = spearmanr(s.unwaked, s.hit6)
    out[g] = dict(n=len(s), rho_abserr=float(r1.statistic), p_abserr=float(r1.pvalue),
                  rho_hit=float(r2.statistic), p_hit=float(r2.pvalue))
    print(f"{g:6d} {len(s):6d} {s.unwaked.min():.2f}~{s.unwaked.max():.2f}{'':>7} "
          f"{r1.statistic:+20.4f} {r1.pvalue:8.4f} {r2.statistic:+19.4f}")
sig = [g for g in (1,2,3) if out[g]["p_abserr"] < 0.05 and abs(out[g]["rho_abserr"]) > 0.05]
print(f"\n유의(|rho|>0.05 & p<0.05) 그룹: {sig}")
verdict = "WAKE_FEATURE_EXPLAINS_RESIDUAL" if sig else "WAKE_FEATURE_NO_RESIDUAL_EXPLANATION"
print(f"판정: {verdict}")
Path("reports/D2_G2_1b_wake_residual.json").write_text(json.dumps(dict(
    node="D2_G2_1b", stage="D2", rows=len(df), by_group={str(k): v for k, v in out.items()},
    significant_groups=sig, verdict=verdict, model_fits=0, dacon_upload=False),
    indent=1, ensure_ascii=False))
print("영수증 -> reports/D2_G2_1b_wake_residual.json")
