
"""D2_G2_1 — 풍향 조건부 후류 기하 피처 생성 및 잔차 설명력 측정 (적합 0회).

근거: research://SPIE 14179:141790I (2025) GNN+physical wake model; Zang 2025 STET [near_match_only]
info.xlsx: 태백가덕산 17기 (VESTAS V126 x12 / UNISON U136 x5), 허브 117m, 로터 126/136m
KPX 그룹: 1 = V126 1~6, 2 = V126 7~12, 3 = U136 1~5

후류 기하: 풍향 theta 에서 터빈 j 가 터빈 i 의 후류에 들어가려면
  (a) i->j 방향각이 theta 와 정렬 (|각차| <= 반각), (b) 이격거리 D/rotor 가 짧을수록 손실 큼
그룹별 '유효 비후류 터빈 수' 를 풍향의 함수로 만들고, 그것이 잔차를 설명하는지 본다.
락박스 미접근 / 적합 0회 / 제출 없음.
"""
from __future__ import annotations
import sys, json, re, zipfile, io, math
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path("."); sys.path.insert(0, "src")
from baram.constants import CAPACITIES_KWH

def dms(s):
    m = re.match(r"""(\d+)°(\d+)'([\d.]+)"([NSEW])""", s.strip())
    d, mi, se, h = int(m.group(1)), int(m.group(2)), float(m.group(3)), m.group(4)
    v = d + mi / 60 + se / 3600
    return -v if h in "SW" else v

z = zipfile.ZipFile("inputs/competition/open_wind_236727.zip")
raw = pd.read_excel(io.BytesIO(z.read("info.xlsx")))
raw.columns = raw.iloc[2]
t = raw.iloc[3:].reset_index(drop=True)
t = t[t["좌표(Google)"].notna()].copy()
t["KPX그룹"] = t["KPX그룹"].ffill().astype(int)
coords = t["좌표(Google)"].astype(str).str.split(" ", n=1, expand=True)
t["lat"] = coords[0].map(dms); t["lon"] = coords[1].map(dms)
t["rotor"] = t["Rotor Diameter(m)"].astype(float)
print(f"터빈 {len(t)}기, 그룹별 {t.groupby('KPX그룹').size().to_dict()}")

# 국소 평면 좌표 (m)
lat0 = t.lat.mean()
t["x"] = (t.lon - t.lon.mean()) * 111320 * math.cos(math.radians(lat0))
t["y"] = (t.lat - t.lat.mean()) * 110540

HALF_ANGLE = 15.0          # 후류 반각 (도)
MAX_D = 10.0               # 로터직경 배수 상한

def unwaked_fraction(group, theta_deg):
    """풍향 theta(기상학적: 바람이 불어오는 방향)에서 비후류 터빈 비율."""
    sub = t[t["KPX그룹"] == group]
    xs, ys, rot = sub.x.to_numpy(), sub.y.to_numpy(), sub.rotor.to_numpy()
    # 바람 벡터 (불어가는 방향)
    wd = math.radians((theta_deg + 180) % 360)
    ux, uy = math.sin(wd), math.cos(wd)
    n = len(sub); waked = np.zeros(n, bool)
    for i in range(n):
        for j in range(n):
            if i == j: continue
            dx, dy = xs[j] - xs[i], ys[j] - ys[i]
            down = dx * ux + dy * uy                    # 풍하 거리
            if down <= 0: continue
            cross = abs(-dx * uy + dy * ux)             # 횡방향 거리
            dist = math.hypot(dx, dy)
            ang = math.degrees(math.asin(min(1.0, cross / max(dist, 1e-6))))
            if ang <= HALF_ANGLE and down / rot[i] <= MAX_D:
                waked[j] = True
    return 1.0 - waked.mean()

print(f"\n{'풍향':>5} " + " ".join(f"{'g'+str(g):>7}" for g in (1, 2, 3)))
table = {}
for th in range(0, 360, 30):
    vals = [unwaked_fraction(g, th) for g in (1, 2, 3)]
    table[th] = vals
    print(f"{th:5d} " + " ".join(f"{v:7.3f}" for v in vals))

spread = {g: max(table[th][g-1] for th in table) - min(table[th][g-1] for th in table) for g in (1,2,3)}
print(f"\n풍향에 따른 비후류비율 변동폭: {({g: round(v,3) for g,v in spread.items()})}")
maxspread = max(spread.values())
print(f"최대 변동폭 {maxspread:.3f}")
if maxspread < 0.05:
    verdict = "WAKE_GEOMETRY_NEARLY_INVARIANT_NO_SIGNAL"
    print("-> 풍향에 따른 후류 구조 변화가 거의 없다. 피처화해도 상수에 가깝다.")
else:
    verdict = "WAKE_GEOMETRY_VARIES_CANDIDATE_FEATURE"
    print("-> 풍향에 따라 후류 구조가 실질적으로 변한다. 피처 후보로 승격 가능.")
print(f"\n판정: {verdict}")
Path("reports/D2_G2_1_wake_geometry.json").write_text(json.dumps(dict(
    node="D2_G2_1", stage="D2", turbines=len(t),
    group_sizes={str(k): int(v) for k, v in t.groupby("KPX그룹").size().items()},
    half_angle=HALF_ANGLE, max_downstream_D=MAX_D,
    unwaked_fraction_by_direction={str(k): [float(x) for x in v] for k, v in table.items()},
    spread={str(k): float(v) for k, v in spread.items()}, max_spread=float(maxspread),
    verdict=verdict,
    origin="research://SPIE-14179-141790I + Zang-2025-STET [near_match_only]",
    model_fits=0, dacon_upload=False), indent=1, ensure_ascii=False))
print("영수증 -> reports/D2_G2_1_wake_geometry.json")
