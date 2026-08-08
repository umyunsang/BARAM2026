"""S12-N13 * grid statistics of the wind COMPONENTS (an evidence-backed feature gap).

Evidence.  research/lanes/S12_ext_dacon_solutions.md sec D-3 records a competing participant's
published EDA (kohwoohyun/wind_power_forecast, full text read) measuring per-group Pearson
correlation against generation:

    heightAboveGround_5_YBLWS_std   0.668 / 0.673 / 0.658
    meanSea_0_prmsl_std             0.572 / 0.581 / 0.558
    heightAboveGround_50_50MUmax_std0.553 / 0.545 / 0.494
    LDAPS 10 m wind speed           0.727 / 0.737 / 0.731
    GFS 100 m wind speed            0.601 / 0.615 / 0.609

and separately that the LDAPS 50 m U component dominates V (0.681/0.671/0.664 vs
0.270/0.292/0.318), i.e. the ridge's prevailing flow is close to east-west.

Our frozen surface has NO grid statistic of any wind COMPONENT.  It carries IDW point values
of the components (`ldaps_spatial__idw__heightAboveGround_50_50MUmax` etc.) and grid statistics
of wind SPEED magnitudes (`ldaps__wind50max_speed__std`), but the spread of a signed component
across the 4x4 box is a different quantity: |V| spread measures directional/rotational
disagreement across the domain, which a speed-magnitude spread cannot see.  XBLWS/YBLWS (the
5 m boundary-layer wind components) are absent from the surface at any aggregation.

Treatment: add per-group grid statistics {mean,std,q10,q50,q90,rng} of the signed components
XBLWS, YBLWS, 10u, 10v, 50MUmax, 50MVmax, 50MUmin, 50MVmin (LDAPS 4x4) and 10u,10v,80u,80v,
100u,100v (GFS 3x3), plus prmsl and orography spread, plus three derived cross-domain
disagreement measures per level (vector-mean speed vs mean-of-speeds deficit, circular spread
of direction, and along/cross-ridge decomposition using the fitted prevailing axis).
This is a pure feature addition to the teacher's column set; nothing else changes.
"""
from __future__ import annotations
import sys, json
import numpy as np, pandas as pd, pyarrow.parquet as pq

CACHE = ('/Users/um-yunsang/BARAM2026/artifacts/cache/'
         '920be0c458d820e855bf79dd25723146f52ce1736138aedba5e6bc853f1f720b/')
GRID = CACHE + 'train_grid_pivot.parquet'

LD_COMP = ['heightAboveGround_5_XBLWS', 'heightAboveGround_5_YBLWS',
           'heightAboveGround_10_10u', 'heightAboveGround_10_10v',
           'heightAboveGround_50_50MUmax', 'heightAboveGround_50_50MVmax',
           'heightAboveGround_50_50MUmin', 'heightAboveGround_50_50MVmin']
LD_SCAL = ['meanSea_0_prmsl', 'surface_0_h', 'etc_0_blh', 'heightAboveGround_2_t']
GF_COMP = ['heightAboveGround_10_10u', 'heightAboveGround_10_10v',
           'heightAboveGround_80_u', 'heightAboveGround_80_v',
           'heightAboveGround_100_100u', 'heightAboveGround_100_100v']
GF_SCAL = ['meanSea_0_prmsl', 'surface_0_gust']

PAIRS = [('ld', 'heightAboveGround_5_XBLWS', 'heightAboveGround_5_YBLWS', 'blws5'),
         ('ld', 'heightAboveGround_10_10u', 'heightAboveGround_10_10v', 'l10'),
         ('ld', 'heightAboveGround_50_50MUmax', 'heightAboveGround_50_50MVmax', 'l50x'),
         ('gf', 'heightAboveGround_10_10u', 'heightAboveGround_10_10v', 'g10'),
         ('gf', 'heightAboveGround_80_u', 'heightAboveGround_80_v', 'g80'),
         ('gf', 'heightAboveGround_100_100u', 'heightAboveGround_100_100v', 'g100')]


def _cols(names, src, var):
    pre = 'ldaps__' if src == 'ld' else 'gfs__'
    return [c for c in names if c.startswith(pre) and c.endswith('__' + var)]


def _stats(M, tag, out):
    out[f'{tag}__mean'] = M.mean(axis=1)
    out[f'{tag}__std'] = M.std(axis=1)
    out[f'{tag}__q10'] = np.quantile(M, 0.10, axis=1)
    out[f'{tag}__q50'] = np.quantile(M, 0.50, axis=1)
    out[f'{tag}__q90'] = np.quantile(M, 0.90, axis=1)
    out[f'{tag}__rng'] = M.max(axis=1) - M.min(axis=1)


def build_component_grid() -> pd.DataFrame:
    names = pq.ParquetFile(GRID).schema.names
    want = ['forecast_kst_dtm']
    spec = {}
    for src, lst in (('ld', LD_COMP + LD_SCAL), ('gf', GF_COMP + GF_SCAL)):
        for v in lst:
            cs = _cols(names, src, v)
            if cs:
                spec[(src, v)] = cs
                want += cs
    G = pd.read_parquet(GRID, columns=sorted(set(want))).set_index('forecast_kst_dtm').sort_index()
    out = {}
    for (src, v), cs in spec.items():
        M = G[cs].to_numpy('float64')
        M = np.where(np.isfinite(M), M, np.nan)
        M = np.where(np.isnan(M), np.nanmean(M, axis=1, keepdims=True), M)
        short = v.replace('heightAboveGround_', 'h').replace('meanSea_0_', '').replace('surface_0_', '')
        _stats(M, f'cg__{src}__{short}', out)

    # vector-coherence / directional-spread block, per (u,v) level
    for src, uv, vv, nm in PAIRS:
        if (src, uv) not in spec or (src, vv) not in spec:
            continue
        U = G[spec[(src, uv)]].to_numpy('float64')
        V = G[spec[(src, vv)]].to_numpy('float64')
        U = np.where(np.isfinite(U), U, np.nanmean(U, axis=1, keepdims=True))
        V = np.where(np.isfinite(V), V, np.nanmean(V, axis=1, keepdims=True))
        sp = np.hypot(U, V)
        vec = np.hypot(U.mean(axis=1), V.mean(axis=1))
        out[f'cg__{nm}__meanspeed'] = sp.mean(axis=1)
        out[f'cg__{nm}__vecspeed'] = vec
        # coherence < 1 means the 4x4 box disagrees on direction
        out[f'cg__{nm}__coherence'] = vec / np.maximum(sp.mean(axis=1), 1e-6)
        out[f'cg__{nm}__speedstd'] = sp.std(axis=1)
        ang = np.arctan2(V, U)
        out[f'cg__{nm}__dirspread'] = 1.0 - np.hypot(np.cos(ang).mean(axis=1), np.sin(ang).mean(axis=1))
        # along/cross prevailing-axis decomposition; the ridge's prevailing flow is
        # close to east-west (D-3 measurement), so the along axis is u and cross is v
        out[f'cg__{nm}__along_abs'] = np.abs(U.mean(axis=1))
        out[f'cg__{nm}__cross_abs'] = np.abs(V.mean(axis=1))
        out[f'cg__{nm}__along_frac'] = np.abs(U.mean(axis=1)) / np.maximum(vec, 1e-6)
    X = pd.DataFrame(out, index=G.index).astype('float32')
    return X


if __name__ == '__main__':
    X = build_component_grid()
    X.to_parquet('/Users/um-yunsang/BARAM2026/research/scratch/component_grid.parquet')
    print('component-grid block:', X.shape)
    print(list(X.columns)[:24])
