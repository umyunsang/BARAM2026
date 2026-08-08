
"""S9-N6 · terrain lane candidate E (rews_geom, research/lanes/S6_ext_B_terrain.md sec B4
candidate E / sec B2's deeper "vertical-combination redesign" theme) -- rotor-equivalent
wind speed, computed separately from each source's own two-point log-law profile
(LDAPS 10m/50m; GFS 80m/100m), area-weighted over each group's ACTUAL rotor swept height
range. g1/g2 (V126, D=126m, hub 117m) sweep z in [54,180]m; g3 (U136, D=136m) sweeps
[49,185]m -- genuinely different geometry per group, so REWS differs by group even from
an identical domain-mean wind profile. This is the (P3) "rotor-geometry interaction" gate
in sec B4.0, not a per-grid-cell feature like S9-N4/N5.

Two-point log-law solve: u(z) = (u*/kappa) * ln(z/z0), kappa=0.4. Given (z1,u1),(z2,u2):
    z0 = exp[(u1*ln(z2) - u2*ln(z1)) / (u1 - u2)]
    u* = kappa * u1 / ln(z1/z0)
Degenerate/near-inversion cases (u1<=u2 for LDAPS 10/50m, or any non-finite result) are
clipped per the source write-up's own warning (z0 in [1e-3, 3.0]m) and flagged.

Domain-mean (LDAPS-16-cell-mean, GFS-9-cell-mean) profile, not per-grid -- the source
write-up's own point is that group differentiation here comes from rotor geometry, not
grid selection, so this deliberately does NOT duplicate S9-N4/N5's per-grid mechanism.
5-point rotor-disc area-weighted cubic mean approximates REWS; kappa/log-law and the
5-point quadrature are fixed a priori, not tuned. 0 fitted degrees of freedom.
"""
import sys, json
import numpy as np, pandas as pd
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/scratch')
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/nodes')
import harness

GRID_PIVOT = ('/Users/um-yunsang/BARAM2026/artifacts/cache/'
              '920be0c458d820e855bf79dd25723146f52ce1736138aedba5e6bc853f1f720b/'
              'train_grid_pivot.parquet')
KAPPA = 0.4
ROTORS = {  # group -> (z_min, z_max, hub) meters
    1: (54.0, 180.0, 117.0), 2: (54.0, 180.0, 117.0), 3: (49.0, 185.0, 117.0),
}
N_QUAD = 5


def two_point_loglaw(u1, z1, u2, z2):
    """Returns (z0, ustar), both clipped/flagged for degenerate cases."""
    with np.errstate(divide='ignore', invalid='ignore'):
        ln_z0 = (u1 * np.log(z2) - u2 * np.log(z1)) / (u1 - u2)
    z0 = np.exp(ln_z0)
    bad = ~np.isfinite(z0) | (u1 <= u2) | (z0 <= 0)
    z0 = np.clip(np.where(bad, 0.03, z0), 1e-3, 3.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        ustar = KAPPA * u1 / np.log(z1 / z0)
    ustar = np.where(np.isfinite(ustar) & (ustar > 0), ustar, KAPPA * u1 / np.log(z1 / 0.03))
    return z0, ustar


def profile_u(ustar, z0, z):
    return (ustar / KAPPA) * np.log(z / z0)


def rews_from_profile(ustar, z0, z_min, z_max, hub):
    R = (z_max - z_min) / 2.0
    z_i = np.linspace(z_min, z_max, N_QUAD)
    w_i = np.sqrt(np.maximum(R * R - (z_i - hub) ** 2, 0.0))
    w_i = w_i / w_i.sum()
    u3 = np.zeros_like(ustar)
    for zi, wi in zip(z_i, w_i):
        u3 = u3 + wi * np.maximum(profile_u(ustar, z0, zi), 0.0) ** 3
    return np.cbrt(u3)


def rews_features():
    import pyarrow.parquet as pq
    names = pq.ParquetFile(GRID_PIVOT).schema.names
    specs = {
        'l10': [c for c in names if c.startswith('ldaps__') and c.endswith('__heightAboveGround_10_10u')],
        'l10v': [c for c in names if c.startswith('ldaps__') and c.endswith('__heightAboveGround_10_10v')],
        'l50': [c for c in names if c.startswith('ldaps__') and c.endswith('__heightAboveGround_50_50MUmax')],
        'l50v': [c for c in names if c.startswith('ldaps__') and c.endswith('__heightAboveGround_50_50MVmax')],
        'g80': [c for c in names if c.startswith('gfs__') and c.endswith('__heightAboveGround_80_u')],
        'g80v': [c for c in names if c.startswith('gfs__') and c.endswith('__heightAboveGround_80_v')],
        'g100': [c for c in names if c.startswith('gfs__') and c.endswith('__heightAboveGround_100_100u')],
        'g100v': [c for c in names if c.startswith('gfs__') and c.endswith('__heightAboveGround_100_100v')],
    }
    cols = ['forecast_kst_dtm'] + [c for v in specs.values() for c in v]
    raw = pd.read_parquet(GRID_PIVOT, columns=cols)
    raw = raw.drop_duplicates('forecast_kst_dtm').set_index('forecast_kst_dtm').sort_index()

    def dom_mean_speed(u_cols, v_cols):
        U = raw[u_cols].to_numpy('float64'); V = raw[v_cols].to_numpy('float64')
        return np.hypot(U, V).mean(axis=1)

    u10 = dom_mean_speed(specs['l10'], specs['l10v'])
    u50 = dom_mean_speed(specs['l50'], specs['l50v'])
    u80 = dom_mean_speed(specs['g80'], specs['g80v'])
    u100 = dom_mean_speed(specs['g100'], specs['g100v'])

    z0_L, ustar_L = two_point_loglaw(u10, 10.0, u50, 50.0)
    z0_G, ustar_G = two_point_loglaw(u80, 80.0, u100, 100.0)

    out = {}
    for g, (zmin, zmax, hub) in ROTORS.items():
        rews_L = rews_from_profile(ustar_L, z0_L, zmin, zmax, hub)
        rews_G = rews_from_profile(ustar_G, z0_G, zmin, zmax, hub)
        hub_L = np.maximum(profile_u(ustar_L, z0_L, hub), 0.0)
        hub_G = np.maximum(profile_u(ustar_G, z0_G, hub), 0.0)
        out[f'rews_L_g{g}'] = rews_L.astype('float32')
        out[f'rews_G_g{g}'] = rews_G.astype('float32')
        out[f'rews_ratio_L_g{g}'] = (rews_L / np.maximum(hub_L, 1e-3)).astype('float32')
        out[f'rews_ratio_G_g{g}'] = (rews_G / np.maximum(hub_G, 1e-3)).astype('float32')
    return pd.DataFrame(out, index=raw.index)


if __name__ == '__main__':
    A0, FR0, COLS0 = harness.surface(())
    feats = rews_features()
    print(f'REWS features: {feats.shape[1]} columns', flush=True)
    print(feats.describe().T[['mean', 'std', 'min', 'max']].to_string(), flush=True)

    FR = {}
    for g in (1, 2, 3):
        X = FR0[g].copy()
        FR[g] = pd.concat([X, feats.reindex(X.index)], axis=1)
    A = pd.concat(FR.values())
    COLS = COLS0 + list(feats.columns)

    key = ('REWS_E_v1',)
    harness._CACHE[key] = (A, FR, COLS)
    out = harness.run('S9-N6', 'terrainE_rews_geom_two_source_loglaw', blocks=key)
    print(json.dumps(out, indent=1, default=str))
