"""S13-N6 * (S6 rung) flow-regime block: dimensionless mountain height, cross-ridge pressure
gradient, and directional wake sectors.

Origin: research/lanes/S13_S6_features_deep.md.  That lane grepped all six prior research lanes
and found wake/sector, mountain-wave/Froude, cross-ridge pressure gradient and
redundancy-driven pruning were never explored, then argued the one mathematical gap our own
S13-N2/E caveat left open:

  a STATIC model-elevation deficit dz is absorbed by the group dummies, but the dimensionless
  mountain height Hhat = h0 * N / U_cross turns it into  dHhat(t) = dz * N(t) / U_cross(t),
  a constant times a TIME-VARYING function, which no group dummy can represent.

Evidence the lane read in full text: Solbakken 2026 (Wind Energy Science 11:155) reports
mountain-wave-induced accelerated downslope winds "tend to occur ... when Hhat < 3" with
normalised wind speed increasing as Hhat goes 0 -> 1.5, and 51%/19% generation differences
between clusters on one ridge at fixed direction; Draxl 2021 (WES 6:45) computes Hhat and the
Scorer parameter from a SINGLE deterministic run and attributes 11% of plant output to mountain
waves at 17% occurrence; Kum & Ho 2021 (KMAPP) state their elevation correction is neutral
Jackson-Hunt linear theory and explicitly call for a stability-aware successor -- which is
exactly this node.

Column availability verified before building: LDAPS carries only 2 m temperature, so N is NOT
computable from LDAPS; GFS carries 850/700/500 hPa temperatures and winds, and 850 hPa (~1500 m)
sits just above the 1078 m ridge, so the 2 m -> 850 hPa layer spans the obstacle.  LDAPS carries
prmsl and surface pressure per grid cell, so the horizontal pressure gradient is available.

GATE DISCIPLINE (imposed by the lane, and by S12-N14 which added 168 informative-but-redundant
columns and LOST 0.000728 while taking 6-11% of gain): screen S1 first -- if every new column
has |corr| >= 0.85 with an existing stability/shear column, the block is a repackaging and the
fold-outside gate is not spent on it.
"""
from __future__ import annotations
import sys, re, json, zipfile, io
import numpy as np, pandas as pd, pyarrow.parquet as pq

CACHE = ('/Users/um-yunsang/BARAM2026/artifacts/cache/'
         '920be0c458d820e855bf79dd25723146f52ce1736138aedba5e6bc853f1f720b/')
GRID = CACHE + 'train_grid_pivot.parquet'
COORD = '/Users/um-yunsang/BARAM2026/research/scratch/grid_coords.json'
ZIP = '/Users/um-yunsang/BARAM2026/inputs/competition/open_wind_236727.zip'
G0, KAPPA, RD = 9.80665, 0.286, 287.05
ROTOR = {1: 126.0, 2: 126.0, 3: 136.0}


def turbine_coords():
    z = zipfile.ZipFile(ZIP)
    n = [x for x in z.namelist() if 'info' in x.lower()][0]
    d = pd.read_excel(io.BytesIO(z.read(n)))
    rows = []
    grp = None
    for _, r in d.iterrows():
        c = str(r.iloc[6])
        if not re.match(r"^\d+°", c):
            continue
        g = r.iloc[7]
        if pd.notna(g):
            grp = int(g)
        m = re.findall(r"(\d+)°(\d+)'([\d.]+)\"([NE])", c)
        lat = int(m[0][0]) + int(m[0][1]) / 60 + float(m[0][2]) / 3600
        lon = int(m[1][0]) + int(m[1][1]) / 60 + float(m[1][2]) / 3600
        rows.append(dict(group_id=grp, lat=lat, lon=lon))
    return pd.DataFrame(rows)


def group_geometry():
    T = turbine_coords()
    out = {}
    for g, s in T.groupby('group_id'):
        la, lo = s.lat.mean(), s.lon.mean()
        x = (s.lon - lo) * 111320 * np.cos(np.radians(la))
        y = (s.lat - la) * 110540
        P = np.c_[x, y]
        w, V = np.linalg.eigh(np.cov(P.T))
        ax = V[:, np.argmax(w)]                       # array principal axis
        axis_deg = (np.degrees(np.arctan2(ax[0], ax[1]))) % 180.0
        out[g] = dict(lat=float(la), lon=float(lo), axis_deg=float(axis_deg),
                      normal_deg=float((axis_deg + 90.0) % 180.0), n=len(s),
                      span_m=float(np.sqrt(w.max()) * 4))
    return out, T


def build_flow_regime() -> pd.DataFrame:
    geom, T = group_geometry()
    print('group geometry (array principal axis, deg from north):')
    for g, v in geom.items():
        print(f'  g{g}: n={v["n"]} axis={v["axis_deg"]:.1f} normal={v["normal_deg"]:.1f} '
              f'span={v["span_m"]:.0f} m  centroid ({v["lat"]:.5f},{v["lon"]:.5f})')

    names = pq.ParquetFile(GRID).schema.names

    def cols(src, var):
        pre = 'ldaps__' if src == 'ld' else 'gfs__'
        return sorted([c for c in names if c.startswith(pre) and c.endswith('__' + var)])

    need = {'sp': cols('ld', 'surface_0_sp'), 'prmsl': cols('ld', 'meanSea_0_prmsl'),
            't2': cols('ld', 'heightAboveGround_2_t'), 'orog': cols('ld', 'surface_0_h'),
            'u50': cols('ld', 'heightAboveGround_50_50MUmax'),
            'v50': cols('ld', 'heightAboveGround_50_50MVmax'),
            'u10': cols('ld', 'heightAboveGround_10_10u'), 'v10': cols('ld', 'heightAboveGround_10_10v'),
            't850': cols('gf', 'isobaricInhPa_850_t'), 't700': cols('gf', 'isobaricInhPa_700_t'),
            'u850': cols('gf', 'isobaricInhPa_850_u'), 'v850': cols('gf', 'isobaricInhPa_850_v'),
            'gu100': cols('gf', 'heightAboveGround_100_100u'), 'gv100': cols('gf', 'heightAboveGround_100_100v')}
    want = sorted({c for v in need.values() for c in v}) + ['forecast_kst_dtm']
    G = pd.read_parquet(GRID, columns=sorted(set(want))).set_index('forecast_kst_dtm').sort_index()

    def M(k):
        a = G[need[k]].to_numpy('float64')
        return np.where(np.isfinite(a), a, np.nanmean(a, axis=1, keepdims=True))

    sp, prmsl, t2 = M('sp').mean(1), M('prmsl'), M('t2').mean(1)
    orog = M('orog')
    u50, v50 = M('u50').mean(1), M('v50').mean(1)
    t850, t700 = M('t850').mean(1), M('t700').mean(1)
    u850, v850 = M('u850').mean(1), M('v850').mean(1)
    gu100, gv100 = M('gu100').mean(1), M('gv100').mean(1)

    # --- static stability N from the 2 m -> 850 hPa layer (the layer spanning the ridge) ------
    p_sfc = np.clip(sp, 5e4, 1.1e5)
    th2 = t2 * (1e5 / p_sfc) ** KAPPA
    th850 = t850 * (1e5 / 85000.0) ** KAPPA
    th700 = t700 * (1e5 / 70000.0) ** KAPPA
    tbar = 0.5 * (t2 + t850)
    dz_layer = np.clip((RD * tbar / G0) * np.log(p_sfc / 85000.0), 50.0, 4000.0)
    n2 = np.clip((G0 / np.maximum(th2, 1.0)) * (th850 - th2) / dz_layer, -5e-3, 5e-3)
    Nb = np.sqrt(np.clip(n2, 1e-8, None))
    dz_up = np.clip((RD * 0.5 * (t850 + t700) / G0) * np.log(85000.0 / 70000.0), 50.0, 4000.0)
    n2_up = np.clip((G0 / np.maximum(th850, 1.0)) * (th700 - th850) / dz_up, -5e-3, 5e-3)

    # --- horizontal pressure gradient over the LDAPS box (least squares plane) ----------------
    cc = json.load(open(COORD))['ldaps']
    ids = [int(re.search(r'grid(\d+)', c).group(1)) for c in need['prmsl']]
    la = np.array([cc[str(i)][0] for i in ids]); lo = np.array([cc[str(i)][1] for i in ids])
    x = (lo - lo.mean()) * 111320 * np.cos(np.radians(la.mean()))
    y = (la - la.mean()) * 110540
    A = np.c_[np.ones_like(x), x, y]
    coef = np.linalg.lstsq(A, prmsl.T, rcond=None)[0]           # (3, n_time)
    dpdx, dpdy = coef[1], coef[2]                                # Pa per metre
    orog_coef = np.linalg.lstsq(A, orog.T, rcond=None)[0]
    dhdx, dhdy = orog_coef[1], orog_coef[2]

    out = {'fr__N_low': Nb, 'fr__N2_low': n2, 'fr__N2_up': n2_up,
           'fr__dtheta_layer': th850 - th2, 'fr__dz_layer': dz_layer,
           'fr__dpdx': dpdx * 1e5, 'fr__dpdy': dpdy * 1e5,
           'fr__dp_mag': np.hypot(dpdx, dpdy) * 1e5,
           'fr__orog_slope_x': dhdx * 1e3, 'fr__orog_slope_y': dhdy * 1e3}

    h0_model = float(np.nanmean(orog.max(1) - orog.min(1)))      # model-resolved relief, ~132 m
    H0_REAL = 380.0                                              # ridge above the Taebaek valleys
    for g, v in geom.items():
        nrm = np.radians(v['normal_deg'])
        nx, ny = np.sin(nrm), np.cos(nrm)                        # unit normal, x=east y=north
        for lvl, (uu, vv) in {'50': (u50, v50), 'g100': (gu100, gv100), '850': (u850, v850)}.items():
            ucr = np.abs(uu * nx + vv * ny)
            ual = np.abs(-uu * ny + vv * nx)
            sp_ = np.hypot(uu, vv)
            out[f'fr__g{g}__ucross_{lvl}'] = ucr
            out[f'fr__g{g}__ualong_{lvl}'] = ual
            out[f'fr__g{g}__crossfrac_{lvl}'] = ucr / np.maximum(sp_, 0.1)
            if lvl == '50':
                u_eff = np.maximum(ucr, 0.5)
                out[f'fr__g{g}__hhat_real'] = H0_REAL * Nb / u_eff
                out[f'fr__g{g}__hhat_model'] = h0_model * Nb / u_eff
                out[f'fr__g{g}__dhhat'] = (H0_REAL - h0_model) * Nb / u_eff
                out[f'fr__g{g}__froude'] = u_eff / np.maximum(H0_REAL * Nb, 1e-6)
                out[f'fr__g{g}__scorer'] = n2 / np.maximum(u_eff ** 2, 0.25)
                out[f'fr__g{g}__dividing_streamline'] = H0_REAL - u_eff / np.maximum(Nb, 1e-4)
                out[f'fr__g{g}__blocked'] = (u_eff / np.maximum(Nb, 1e-4) < H0_REAL).astype(float)
                out[f'fr__g{g}__wave_window'] = ((H0_REAL * Nb / u_eff) < 3.0).astype(float)
        # cross-ridge pressure gradient, the direct forcing of downslope acceleration
        out[f'fr__g{g}__dp_cross'] = (dpdx * nx + dpdy * ny) * 1e5

    # --- F2: directional wake shadow between the three groups (Jensen-style) ------------------
    cen = {g: (v['lat'], v['lon']) for g, v in geom.items()}
    wd = (np.degrees(np.arctan2(-u50, -v50))) % 360.0            # meteorological wind direction
    for tgt in (1, 2, 3):
        shade = np.zeros(len(G))
        for src in (1, 2, 3):
            if src == tgt:
                continue
            dy = (cen[tgt][0] - cen[src][0]) * 110540
            dx = (cen[tgt][1] - cen[src][1]) * 111320 * np.cos(np.radians(cen[tgt][0]))
            dist = np.hypot(dx, dy)
            bear = np.degrees(np.arctan2(dx, dy)) % 360.0        # src -> tgt bearing
            upwind = (bear + 180.0) % 360.0                      # wind must come FROM here
            dth = np.abs((wd - upwind + 180.0) % 360.0 - 180.0)
            halfang = np.degrees(np.arctan(0.075 + 2.0 * ROTOR[src] / max(dist, 1.0)))
            w = np.clip(1.0 - dth / np.maximum(halfang, 1.0), 0.0, 1.0)
            shade = np.maximum(shade, w * (ROTOR[src] / max(dist, 1.0)) * 10.0)
        out[f'fr__g{tgt}__wake_shadow'] = shade
    X = pd.DataFrame(out, index=G.index).astype('float32')
    return X, geom


if __name__ == '__main__':
    X, geom = build_flow_regime()
    X.to_parquet('/Users/um-yunsang/BARAM2026/research/scratch/flow_regime.parquet')
    json.dump(geom, open('/Users/um-yunsang/BARAM2026/research/nodes/S13-N6_geometry.json', 'w'), indent=1)
    print(f'\nflow-regime block: {X.shape}')
    hh = X[[c for c in X.columns if c.endswith('hhat_real')]]
    print('\nHhat distribution (should straddle the 1.5 and 3 thresholds to be informative):')
    print(hh.describe(percentiles=[.05, .25, .5, .75, .95]).round(3).to_string())
    print('\nshare of hours with Hhat < 3 (mountain-wave window):')
    print({c: round(float((X[c] < 3).mean()), 3) for c in hh.columns})
    print('share of hours BLOCKED (U/N < h0):')
    print({c: round(float(X[c].mean()), 3) for c in X.columns if c.endswith('__blocked')})
