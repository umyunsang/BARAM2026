
"""S6 · feature-construction blocks.  Each block is a named, self-contained set of columns
so a node can switch exactly one block on and attribute the delta to it.

B1  grid order statistics x minimal time window  (HEFTCom2024 team GEB spec)
B3  power-curve physical prior: pc_prior / pc_slope / pc_band / pc_smear
B10 air-density corrected wind (IEC 61400-12 style)
B2  rotor-equivalent wind speed proxy + veer
"""
from __future__ import annotations
import json
import numpy as np, pandas as pd

CACHE = ('/Users/um-yunsang/BARAM2026/artifacts/cache/'
         '920be0c458d820e855bf79dd25723146f52ce1736138aedba5e6bc853f1f720b/')
S = '/Users/um-yunsang/BARAM2026/research/scratch/'
PC = {int(k): v for k, v in json.load(open(S+'powercurve_params.json')).items()}
WS_SIGMA = {1: 1.52, 2: 1.62, 3: 1.68}          # frozen constants: measured teacher RMSE
_G = {}


def _grid():
    if 'p' in _G: return _G['p']
    cols = ['forecast_kst_dtm']
    cols += [f'ldaps__grid{i:02d}__{f}' for i in range(1,17)
             for f in ('wind10_speed','wind5_speed','wind50max_speed','wind50min_speed','surface_0_h')]
    cols += [f'gfs__grid{i:02d}__{f}' for i in range(1,10)
             for f in ('wind10_speed','wind80_speed','wind100_speed')]
    d = pd.read_parquet(CACHE+'train_grid_pivot.parquet', columns=cols)
    d = d.drop_duplicates('forecast_kst_dtm').set_index('forecast_kst_dtm').sort_index()
    _G['p'] = d
    return d


def _batch():
    if 'b' in _G: return _G['b']
    d = pd.read_parquet(CACHE+'train_features.parquet',
                        columns=['forecast_kst_dtm','data_available_kst_dtm','group_id'])
    d = d[d.group_id==1].drop_duplicates('forecast_kst_dtm').set_index('forecast_kst_dtm').sort_index()
    _G['b'] = d['data_available_kst_dtm']
    return _G['b']


def curve(v, vin, vr, vout, k):
    x = np.clip((v-vin)/np.maximum(vr-vin, 0.1), 0, 1)
    f = x**k
    f = np.where(v >= vr, 1.0, f)
    f = np.where((v < vin) | (v > vout), 0.0, f)
    return f


def block_B1(index) -> pd.DataFrame:
    """Grid order statistics of hub-interpolated wind, at t-1 / t / t+1 inside the issuance."""
    G = _grid(); batch = _batch().reindex(G.index)
    out = {}
    # per-grid hub-height wind proxies
    lmid = np.stack([0.5*(G[f'ldaps__grid{i:02d}__wind50max_speed'].to_numpy()
                          + G[f'ldaps__grid{i:02d}__wind50min_speed'].to_numpy()) for i in range(1,17)], 1)
    l10  = np.stack([G[f'ldaps__grid{i:02d}__wind10_speed'].to_numpy() for i in range(1,17)], 1)
    with np.errstate(all='ignore'):
        alpha = np.log(np.clip(lmid,.05,None)/np.clip(l10,.05,None))/np.log(5.0)
    alpha = np.clip(np.nan_to_num(alpha, nan=0.14), -0.2, 0.6)
    lhub = lmid*(117.0/50.0)**alpha
    g100 = np.stack([G[f'gfs__grid{i:02d}__wind100_speed'].to_numpy() for i in range(1,10)], 1)
    g80  = np.stack([G[f'gfs__grid{i:02d}__wind80_speed'].to_numpy() for i in range(1,10)], 1)
    with np.errstate(all='ignore'):
        ga = np.log(np.clip(g100,.05,None)/np.clip(g80,.05,None))/np.log(1.25)
    ga = np.clip(np.nan_to_num(ga, nan=0.14), -0.2, 0.6)
    ghub = g100*(117.0/100.0)**ga
    fields = {'ldaps': lhub, 'gfs': ghub}
    for src, M in fields.items():
        base = {'max': np.nanmax(M,1), 'mean': np.nanmean(M,1), 'min': np.nanmin(M,1),
                'p25': np.nanquantile(M,0.25,axis=1), 'p75': np.nanquantile(M,0.75,axis=1)}
        base['rng'] = base['max']-base['min']
        base['iqr'] = base['p75']-base['p25']
        base['cv']  = base['rng']/np.maximum(base['mean'],0.1)
        for k,v in base.items():
            s = pd.Series(v, index=G.index)
            out[f'b1__{src}__{k}'] = s
            for L in (-1,1):
                sh = s.shift(L).where(batch.shift(L).eq(batch))
                out[f'b1__{src}__{k}__t{L:+d}'] = sh
                out[f'b1__{src}__{k}__d{L:+d}'] = s - sh
    F = pd.DataFrame(out, index=G.index)
    return F.reindex(index).astype('float32')


def block_B3(index, group_id: int, hub_series: pd.Series) -> pd.DataFrame:
    """Power-curve physical prior and its local geometry."""
    p = PC[group_id]; s = WS_SIGMA[group_id]
    v = hub_series.reindex(index).to_numpy(float)
    out = {}
    out['b3__pc_prior'] = curve(v, *p)
    out['b3__pc_slope'] = (curve(v+0.5, *p) - curve(v-0.5, *p))
    out['b3__pc_lo'] = curve(np.maximum(v-s,0), *p)
    out['b3__pc_hi'] = curve(v+s, *p)
    out['b3__pc_band'] = out['b3__pc_hi'] - out['b3__pc_lo']
    nodes, wts = np.polynomial.hermite_e.hermegauss(9)
    wts = wts/wts.sum()
    out['b3__pc_smear'] = sum(w*curve(np.maximum(v+s*n,0), *p) for n,w in zip(nodes,wts))
    out['b3__pc_smear_gap'] = out['b3__pc_smear'] - out['b3__pc_prior']
    out['b3__above_rated'] = (v >= p[1]).astype('float32')
    out['b3__below_cutin'] = (v <= p[0]).astype('float32')
    return pd.DataFrame(out, index=index).astype('float32')


def block_B10(index, X: pd.DataFrame) -> pd.DataFrame:
    """IEC-style density correction of the hub wind."""
    out = {}
    if 'phys_v2__air_density' in X:
        rho = X['phys_v2__air_density'].to_numpy(float)
    else:
        rho = np.full(len(X), 1.225)
    ratio = np.clip(rho/1.225, 0.8, 1.2)
    for c in ('atm__hub_consensus','atm__hub_from_gfs100','atm__hub_from_ldaps50'):
        if c in X:
            out[f'b10__{c.split("__")[-1]}_rho'] = X[c].to_numpy(float)*ratio**(1/3)
    out['b10__rho_ratio'] = ratio
    return pd.DataFrame(out, index=index).astype('float32')


def block_B2(index, X: pd.DataFrame) -> pd.DataFrame:
    """Rotor-equivalent wind speed proxy + directional veer across the rotor disc."""
    out = {}
    if {'atm__hub_from_ldaps50','atm__hub_from_gfs100','atm__alpha_100_80'} <= set(X.columns):
        v = X['atm__hub_consensus'].to_numpy(float)
        a = np.clip(X['atm__alpha_100_80'].to_numpy(float), -0.2, 0.6)
        R = 63.0   # V126 rotor radius; U136 is 68 m, close enough for a proxy
        zs = np.array([-0.8,-0.4,0.0,0.4,0.8])*R
        w  = np.array([0.15,0.22,0.26,0.22,0.15])
        rews3 = sum(wi*np.maximum(v*((117.0+z)/117.0)**a, 0)**3 for z,wi in zip(zs,w))
        out['b2__rews'] = rews3**(1/3)
        out['b2__rews_deficit'] = out['b2__rews'] - v
    for a,b,nm in [('ldaps_spatial__idw__wind10_dir_sin','ldaps_spatial__idw__wind50max_dir_sin','veer_sin'),
                   ('ldaps_spatial__idw__wind10_dir_cos','ldaps_spatial__idw__wind50max_dir_cos','veer_cos')]:
        if a in X and b in X:
            out[f'b2__{nm}'] = X[b].to_numpy(float) - X[a].to_numpy(float)
    if 'b2__veer_sin' in out:
        out['b2__veer_mag'] = np.hypot(out['b2__veer_sin'], out['b2__veer_cos'])
    return pd.DataFrame(out, index=index).astype('float32')


# ---------------------------------------------------------------------------
# C01 + C02 : replace the permutation-variant 4x4 reshape (which is wrong -- the
# 16 LDAPS cells are a 3/5/5/3 diamond, verified against the raw archive) with
# permutation-INVARIANT order statistics plus a true-geometry flow projection.
# ---------------------------------------------------------------------------
_COORD = None
_G2 = None

def _coords():
    global _COORD
    if _COORD is None:
        _COORD = json.load(open(S+'grid_coords.json'))
    return _COORD


def block_G2(index) -> pd.DataFrame:
    global _G2
    if _G2 is None:
        import pyarrow.parquet as pq
        names = pq.ParquetFile(CACHE+'train_grid_pivot.parquet').schema.names
        want = ['forecast_kst_dtm'] + [c for c in names if
                ('10u' in c or '10v' in c or 'heightAboveGround_80_u' in c
                 or 'heightAboveGround_80_v' in c or '100u' in c or '100v' in c
                 or '50MUmax' in c or '50MVmax' in c or '50MUmin' in c or '50MVmin' in c)]
        p = pd.read_parquet(CACHE+'train_grid_pivot.parquet', columns=want)
        p = p.drop_duplicates('forecast_kst_dtm').set_index('forecast_kst_dtm').sort_index()
        C = _coords()
        out = {}
        specs = [('ldaps','heightAboveGround_10_10u','heightAboveGround_10_10v','l10'),
                 ('ldaps','heightAboveGround_50_50MUmax','heightAboveGround_50_50MVmax','l50x'),
                 ('ldaps','heightAboveGround_50_50MUmin','heightAboveGround_50_50MVmin','l50n'),
                 ('gfs','heightAboveGround_10_10u','heightAboveGround_10_10v','g10'),
                 ('gfs','heightAboveGround_80_u','heightAboveGround_80_v','g80'),
                 ('gfs','heightAboveGround_100_100u','heightAboveGround_100_100v','g100')]
        for src, un, vn, tag in specs:
            us = sorted(c for c in p.columns if c.startswith(src+'__') and c.endswith('__'+un))
            vs = sorted(c for c in p.columns if c.startswith(src+'__') and c.endswith('__'+vn))
            if not us: continue
            U = p[us].to_numpy('float32'); V = p[vs].to_numpy('float32')
            sp = np.hypot(U, V)
            ids = [c.split('__')[1].replace('grid','').lstrip('0') or '0' for c in us]
            lat = np.array([C[src][str(int(i))][0] for i in ids], dtype='float32')
            lon = np.array([C[src][str(int(i))][1] for i in ids], dtype='float32')
            # metres from the patch centroid (local tangent plane)
            y = (lat - lat.mean())*111320.0
            x = (lon - lon.mean())*111320.0*np.cos(np.deg2rad(float(lat.mean())))
            q = {'min':np.nanmin(sp,1), 'q25':np.nanquantile(sp,.25,axis=1),
                 'mean':np.nanmean(sp,1), 'med':np.nanmedian(sp,1),
                 'q75':np.nanquantile(sp,.75,axis=1), 'max':np.nanmax(sp,1),
                 'std':np.nanstd(sp,1)}
            q['rng'] = q['max']-q['min']
            for k,v in q.items(): out[f'g2__{tag}__{k}'] = v
            # true-geometry flow projection
            um = np.nanmean(U,1); vm = np.nanmean(V,1)
            nrm = np.maximum(np.hypot(um,vm), 1e-3)
            ex = (um/nrm)[:,None]; ey = (vm/nrm)[:,None]
            proj  =  x[None,:]*ex + y[None,:]*ey          # along-flow, +downwind
            cross = -x[None,:]*ey + y[None,:]*ex          # cross-flow
            up_i = np.argmin(proj, axis=1); dn_i = np.argmax(proj, axis=1)
            r = np.arange(len(sp))
            out[f'g2__{tag}__upstream']   = sp[r, up_i]
            out[f'g2__{tag}__downstream'] = sp[r, dn_i]
            out[f'g2__{tag}__up_minus_mean'] = sp[r, up_i] - q['mean']
            # least-squares slope of speed along / across the flow (per 1000 m)
            for nm, coord in (('slope_along', proj), ('slope_cross', cross)):
                cm = coord - coord.mean(axis=1, keepdims=True)
                sm = sp - q['mean'][:,None]
                den = np.maximum((cm*cm).sum(axis=1), 1e-3)
                out[f'g2__{tag}__{nm}'] = (cm*sm).sum(axis=1)/den*1000.0
        _G2 = pd.DataFrame(out, index=p.index).astype('float32')
    return _G2.reindex(index)


def build_blocks(group_id: int, X: pd.DataFrame, blocks) -> pd.DataFrame:
    parts = []
    hub = X['atm__hub_consensus'] if 'atm__hub_consensus' in X else X.iloc[:,0]
    if 'B1'  in blocks: parts.append(block_B1(X.index))
    if 'B3'  in blocks: parts.append(block_B3(X.index, group_id, hub))
    if 'B10' in blocks: parts.append(block_B10(X.index, X))
    if 'B2'  in blocks: parts.append(block_B2(X.index, X))
    if 'G2'  in blocks: parts.append(block_G2(X.index))
    if not parts: return pd.DataFrame(index=X.index)
    return pd.concat(parts, axis=1)
