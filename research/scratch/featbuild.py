
"""Compact physics + temporal-window feature builder for the site-wind teacher.

Design decisions (evidence-backed):
  * base       : ~140 hand-picked NWP columns (idw + grid summary + thermodynamics)
  * atm__      : atmospheric-regime derivations that exist in the repo but never reached
                 the teacher (repo-wind-audit finding #1)
  * lag/lead   : within-issuance temporal window, +-1..3 (value+diff) and +-6 (value).
                 The repo contract forbade +-6h and only grouped inside one issuance;
                 both restrictions are lifted here and measured.
"""
from __future__ import annotations
import numpy as np, pandas as pd, pyarrow.parquet as pq

CACHE = ('/Users/um-yunsang/BARAM2026/artifacts/cache/'
         '920be0c458d820e855bf79dd25723146f52ce1736138aedba5e6bc853f1f720b/')

GFS_IDW = ['wind10_speed','wind10_dir_sin','wind10_dir_cos','wind80_speed','wind80_dir_sin',
           'wind80_dir_cos','wind100_speed','wind100_dir_sin','wind100_dir_cos',
           'heightAboveGround_2_2t','heightAboveGround_2_2r','surface_0_sp','surface_0_gust',
           'planetaryBoundaryLayer_0_VRATE','surface_0_prate',
           'heightAboveGround_10_10u','heightAboveGround_10_10v',
           'heightAboveGround_100_100u','heightAboveGround_100_100v']
LDAPS_IDW = ['wind10_speed','wind10_dir_sin','wind10_dir_cos','wind5_speed','wind5_dir_sin',
             'wind5_dir_cos','wind50max_speed','wind50max_dir_sin','wind50max_dir_cos',
             'wind50min_speed','wind50min_dir_sin','wind50min_dir_cos',
             'heightAboveGround_2_t','heightAboveGround_2_r','surface_0_sp','etc_0_blh',
             'heightAboveGround_10_10u','heightAboveGround_10_10v',
             'heightAboveGround_50_50MUmax','heightAboveGround_50_50MVmax']
GFS_SPEEDS = ['wind10_speed','wind80_speed','wind100_speed','pbl_wind_speed',
              'wind850_speed','wind700_speed','wind500_speed']
LDAPS_SPEEDS = ['wind10_speed','wind5_speed','wind50max_speed','wind50min_speed']
STATS = ['mean','std','q10','q50','q90']
GFS_SCALAR = ['surface_0_gust','heightAboveGround_2_2t','heightAboveGround_2_2d',
              'heightAboveGround_2_2r','heightAboveGround_2_2sh','surface_0_sp','meanSea_0_prmsl',
              'atmosphere_0_tcc','surface_0_dswrf','surface_0_tp','isobaricInhPa_850_t',
              'isobaricInhPa_700_t','isobaricInhPa_500_t','isobaricInhPa_500_gh',
              'planetaryBoundaryLayer_0_VRATE','surface_0_prate']
LDAPS_SCALAR = ['etc_0_blh','heightAboveGround_2_t','heightAboveGround_2_dpt','heightAboveGround_2_r',
                'heightAboveGround_2_q','surface_0_sp','meanSea_0_prmsl','surface_0_NDNSW',
                'etc_0_lcc','etc_0_mcc','etc_0_hcc','surface_0_ncpcp','surface_0_h','surface_0_lsm']
CAL = ['cal__hour_sin','cal__hour_cos','cal__doy_sin','cal__doy_cos','hour','month',
       'day_of_year','lead_hour','operating_year']
PHYS = ['phys__hub117_speed','phys__speed_shear_100_80','phys__air_density','phys__rho_v3',
        'phys_v2__shear_alpha_100_80','phys_v2__hub117_speed','phys_v2__air_density',
        'phys_v2__rho_v3','phys_v2__fleet_power_proxy_w']
DISAGREE = ['source_disagreement__wind10_speed_idw','source_disagreement__wind10_speed_idw__abs',
            'source_disagreement__wind10_speed_nearest','source_disagreement__wind10_speed_nearest__abs',
            'source_disagreement__surface_pressure_idw','source_disagreement__surface_pressure_idw__abs']

LAG_KEYS = ['ldaps_spatial__idw__wind10_speed','ldaps_spatial__idw__wind50max_speed',
            'ldaps_spatial__idw__wind50min_speed','ldaps_spatial__idw__wind5_speed',
            'ldaps_spatial__idw__wind50max_dir_sin','ldaps_spatial__idw__wind50max_dir_cos',
            'gfs_spatial__idw__wind10_speed','gfs_spatial__idw__wind80_speed',
            'gfs_spatial__idw__wind100_speed','gfs_spatial__idw__surface_0_gust',
            'gfs_spatial__idw__wind100_dir_sin','gfs_spatial__idw__wind100_dir_cos',
            'phys__hub117_speed','phys_v2__hub117_speed',
            'ldaps_spatial__idw__etc_0_blh','ldaps_spatial__idw__heightAboveGround_2_t',
            'gfs__wind850_speed__mean','ldaps__wind50max_speed__mean']
LAGS_FULL = (-3,-2,-1,1,2,3)
LAGS_WIDE = (-6,6)


def base_columns() -> list[str]:
    have = set(pq.ParquetFile(CACHE+'train_features.parquet').schema.names)
    cols = ['forecast_kst_dtm','data_available_kst_dtm','group_id']
    cols += [f'gfs_spatial__idw__{c}' for c in GFS_IDW]
    cols += [f'ldaps_spatial__idw__{c}' for c in LDAPS_IDW]
    for s in GFS_SPEEDS:
        cols += [f'gfs__{s}__{t}' for t in STATS]
    for s in LDAPS_SPEEDS:
        cols += [f'ldaps__{s}__{t}' for t in STATS]
    cols += [f'gfs__{c}__mean' for c in GFS_SCALAR] + [f'gfs__{c}__std' for c in GFS_SCALAR]
    cols += [f'ldaps__{c}__mean' for c in LDAPS_SCALAR] + [f'ldaps__{c}__std' for c in LDAPS_SCALAR]
    cols += CAL + PHYS + DISAGREE
    missing = [c for c in cols if c not in have]
    if missing:
        raise KeyError(f'missing columns: {missing[:10]} ({len(missing)})')
    return list(dict.fromkeys(cols))


def _ratio(a, b):
    return a / np.where(np.abs(b) < 1e-6, np.nan, b)


def add_atm(df: pd.DataFrame) -> pd.DataFrame:
    f = {}
    g10 = df['gfs__wind10_speed__mean']; g80 = df['gfs__wind80_speed__mean']
    g100 = df['gfs__wind100_speed__mean']; gpbl = df['gfs__pbl_wind_speed__mean']
    g850 = df['gfs__wind850_speed__mean']
    t2 = df['gfs__heightAboveGround_2_2t__mean']; td2 = df['gfs__heightAboveGround_2_2d__mean']
    t850 = df['gfs__isobaricInhPa_850_t__mean']; t700 = df['gfs__isobaricInhPa_700_t__mean']
    t500 = df['gfs__isobaricInhPa_500_t__mean']
    gust = df['gfs__surface_0_gust__mean']
    th850 = t850*(1000/850)**0.286; th700 = t700*(1000/700)**0.286; th500 = t500*(1000/500)**0.286
    shear = g850 - g10
    f['atm__dewpoint_depression'] = t2 - td2
    f['atm__theta850_minus_t2'] = th850 - t2
    f['atm__theta700_minus_theta850'] = th700 - th850
    f['atm__theta500_minus_theta700'] = th500 - th700
    f['atm__gust_excess'] = gust - g10
    f['atm__gust_factor'] = _ratio(gust, g10)
    f['atm__w100_w10_ratio'] = _ratio(g100, g10)
    f['atm__w80_w10_ratio'] = _ratio(g80, g10)
    f['atm__pbl_w10_ratio'] = _ratio(gpbl, g10)
    f['atm__alpha_80_10'] = _ratio(np.log(g80.clip(lower=.05)/g10.clip(lower=.05)), np.log(8.0))
    f['atm__alpha_100_80'] = _ratio(np.log(g100.clip(lower=.05)/g80.clip(lower=.05)), np.log(1.25))
    f['atm__bulk_richardson_proxy'] = _ratio(th850 - t2, shear*shear + 0.25)
    f['atm__vrate_per_wind'] = _ratio(df['gfs__planetaryBoundaryLayer_0_VRATE__mean'], g10)
    l10 = df['ldaps__wind10_speed__mean']; l5 = df['ldaps__wind5_speed__mean']
    lmax = df['ldaps__wind50max_speed__mean']; lmin = df['ldaps__wind50min_speed__mean']
    env = lmax - lmin; mid = 0.5*(lmax+lmin)
    f['atm__w50_envelope'] = env
    f['atm__w50_midpoint'] = mid
    f['atm__w50_asymmetry'] = _ratio(env, mid)
    f['atm__w50max_w10_ratio'] = _ratio(lmax, l10)
    f['atm__w50min_w10_ratio'] = _ratio(lmin, l10)
    f['atm__w10_w5_ratio'] = _ratio(l10, l5)
    f['atm__alpha_50_10'] = _ratio(np.log(mid.clip(lower=.05)/l10.clip(lower=.05)), np.log(5.0))
    f['atm__blh_norm'] = _ratio(df['ldaps__etc_0_blh__mean'], 117.0)
    f['atm__blh_below_hub'] = (df['ldaps__etc_0_blh__mean'] < 117.0).astype('float32')
    f['atm__lapse_2m_850'] = t2 - t850
    f['atm__rh_deficit'] = 100.0 - df['ldaps__heightAboveGround_2_r__mean']
    # hub extrapolation with a stability-aware exponent, from both sources
    a = f['atm__alpha_100_80'].clip(-0.2, 0.6)
    f['atm__hub_from_gfs100'] = g100 * (117.0/100.0)**a
    f['atm__hub_from_ldaps50'] = mid * (117.0/50.0)**a
    f['atm__hub_consensus'] = 0.5*(f['atm__hub_from_gfs100'] + f['atm__hub_from_ldaps50'])
    f['atm__hub_disagree'] = f['atm__hub_from_gfs100'] - f['atm__hub_from_ldaps50']
    # direction x speed interactions (terrain speed-up is direction dependent)
    for src, sp, sn, cs in [('ldaps','ldaps_spatial__idw__wind50max_speed',
                             'ldaps_spatial__idw__wind50max_dir_sin','ldaps_spatial__idw__wind50max_dir_cos'),
                            ('gfs','gfs_spatial__idw__wind100_speed',
                             'gfs_spatial__idw__wind100_dir_sin','gfs_spatial__idw__wind100_dir_cos')]:
        f[f'atm__{src}__speed_x_dsin'] = df[sp]*df[sn]
        f[f'atm__{src}__speed_x_dcos'] = df[sp]*df[cs]
    return pd.DataFrame(f, index=df.index)


def add_window(df: pd.DataFrame) -> pd.DataFrame:
    keys = [c for c in LAG_KEYS if c in df.columns]
    batch = df['data_available_kst_dtm'].to_numpy()
    B = df[keys]
    out = {}
    for L in LAGS_FULL + LAGS_WIDE:
        sh = B.shift(L)
        same = pd.Series(batch, index=df.index).shift(L).eq(df['data_available_kst_dtm']).to_numpy()
        blk = ~same
        vals = sh.to_numpy(dtype='float32', copy=True)
        vals[blk, :] = np.nan
        for j, c in enumerate(keys):
            out[f'{c}__lag{L}'] = vals[:, j]
            if L in LAGS_FULL:
                out[f'{c}__d{L}'] = df[c].to_numpy('float32') - vals[:, j]
    # rolling mean/std of the full 24h issuance for the primary speed keys
    prim = [k for k in keys if k.endswith('_speed') or 'hub117' in k]
    grp = df.groupby('data_available_kst_dtm', sort=False)
    for c in prim:
        out[f'{c}__batch_mean'] = grp[c].transform('mean').to_numpy('float32')
        out[f'{c}__batch_anom'] = df[c].to_numpy('float32') - out[f'{c}__batch_mean']
        out[f'{c}__batch_rank'] = grp[c].rank(pct=True).to_numpy('float32')
    return pd.DataFrame(out, index=df.index)


def build(group_id: int) -> pd.DataFrame:
    cols = base_columns()
    df = pd.read_parquet(CACHE+'train_features.parquet', columns=cols)
    df = df[df.group_id == group_id].sort_values('forecast_kst_dtm').reset_index(drop=True)
    idx = df['forecast_kst_dtm']
    atm = add_atm(df); win = add_window(df)
    num = df.drop(columns=['forecast_kst_dtm','data_available_kst_dtm','group_id'])
    out = pd.concat([num.reset_index(drop=True), atm.reset_index(drop=True),
                     win.reset_index(drop=True)], axis=1).astype('float32')
    out.index = pd.DatetimeIndex(idx)
    out.index.name = 'forecast_kst_dtm'
    return out


# ---------------------------------------------------------------- extra surfaces
_GEOM = None
_PIVOT = None

def geom_frame() -> pd.DataFrame:
    global _GEOM
    if _GEOM is None:
        g = pd.read_parquet(CACHE+'train_geometric.parquet')
        g = g.drop(columns=[c for c in ('data_available_kst_dtm',) if c in g.columns])
        g = g.set_index('forecast_kst_dtm').astype('float32')
        _GEOM = g[~g.index.duplicated()]
    return _GEOM


def grid_frame() -> pd.DataFrame:
    """Per-grid raw wind components + advection/upstream descriptors."""
    global _PIVOT
    if _PIVOT is not None:
        return _PIVOT
    names = pq.ParquetFile(CACHE+'train_grid_pivot.parquet').schema.names
    want = ['forecast_kst_dtm'] + [c for c in names if
            ('10u' in c or '10v' in c or 'heightAboveGround_80_u' in c or 'heightAboveGround_80_v' in c
             or '100u' in c or '100v' in c or '50MUmax' in c or '50MVmax' in c
             or 'etc_0_blh' in c or 'surface_0_h' in c)]
    p = pd.read_parquet(CACHE+'train_grid_pivot.parquet', columns=want)
    p = p.set_index('forecast_kst_dtm')
    p = p[~p.index.duplicated()]
    out = {}
    for src, uname, vname, tag in [('ldaps','heightAboveGround_10_10u','heightAboveGround_10_10v','l10'),
                                   ('ldaps','heightAboveGround_50_50MUmax','heightAboveGround_50_50MVmax','l50'),
                                   ('gfs','heightAboveGround_10_10u','heightAboveGround_10_10v','g10'),
                                   ('gfs','heightAboveGround_80_u','heightAboveGround_80_v','g80'),
                                   ('gfs','heightAboveGround_100_100u','heightAboveGround_100_100v','g100')]:
        us = sorted(c for c in p.columns if c.startswith(src+'__') and c.endswith('__'+uname))
        vs = sorted(c for c in p.columns if c.startswith(src+'__') and c.endswith('__'+vname))
        if not us:
            continue
        U = p[us].to_numpy('float32'); V = p[vs].to_numpy('float32')
        sp = np.hypot(U, V)
        for j in range(sp.shape[1]):
            out[f'grid__{tag}__sp{j:02d}'] = sp[:, j]
        # horizontal structure of the patch
        n = int(round(np.sqrt(sp.shape[1])))
        if n*n == sp.shape[1] and n > 1:
            G = sp.reshape(-1, n, n)
            gx = np.gradient(G, axis=2).reshape(len(G), -1)
            gy = np.gradient(G, axis=1).reshape(len(G), -1)
            ub = U.reshape(-1, n, n); vb = V.reshape(-1, n, n)
            dudx = np.gradient(ub, axis=2); dvdy = np.gradient(vb, axis=1)
            dvdx = np.gradient(vb, axis=2); dudy = np.gradient(ub, axis=1)
            out[f'grid__{tag}__gradx'] = gx.mean(1); out[f'grid__{tag}__grady'] = gy.mean(1)
            out[f'grid__{tag}__gradmag'] = np.hypot(gx, gy).mean(1)
            out[f'grid__{tag}__div'] = (dudx+dvdy).reshape(len(G), -1).mean(1)
            out[f'grid__{tag}__vort'] = (dvdx-dudy).reshape(len(G), -1).mean(1)
            # advection of speed by the patch-mean flow: -(u.grad)S
            um = U.mean(1); vm = V.mean(1)
            out[f'grid__{tag}__adv'] = -(um*gx.mean(1) + vm*gy.mean(1))
            # upstream value: pick the grid cell displaced against the mean flow
            ang = np.arctan2(vm, um)
            ix = np.clip((n-1)/2 - np.cos(ang)*(n-1)/2, 0, n-1).round().astype(int)
            iy = np.clip((n-1)/2 - np.sin(ang)*(n-1)/2, 0, n-1).round().astype(int)
            out[f'grid__{tag}__upstream'] = G[np.arange(len(G)), iy, ix]
            out[f'grid__{tag}__downstream'] = G[np.arange(len(G)),
                                                (n-1-iy), (n-1-ix)]
            out[f'grid__{tag}__up_minus_mean'] = out[f'grid__{tag}__upstream'] - sp.mean(1)
    _PIVOT = pd.DataFrame(out, index=p.index).astype('float32')
    return _PIVOT


def build2(group_id: int, geom: bool = True, grid: bool = True) -> pd.DataFrame:
    X = build(group_id)
    parts = [X]
    if geom:
        parts.append(geom_frame().reindex(X.index))
    if grid:
        parts.append(grid_frame().reindex(X.index))
    out = pd.concat(parts, axis=1).astype('float32')
    return out.loc[:, ~out.columns.duplicated()]
