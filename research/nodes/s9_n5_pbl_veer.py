
"""S9-N5 · terrain lane candidate F (pbl_regime, research/lanes/S6_ext_B_terrain.md sec
B4), trimmed to its non-redundant part only.

Audit finding before implementing: candidate F's headline pieces (z/zi = hub-height/PBLH,
above_pbl = PBLH<hub) are ALREADY present in the harness default baseline --
featbuild.py::add_atm() computes atm__blh_norm and atm__blh_below_hub from LDAPS PBLH on
every node, including S9-N0's control. Implementing candidate F as specified would mostly
duplicate existing information. What is genuinely new: wind-DIRECTION shear with height
(veer) and a below/above-jet flag from the 80/100m GFS layers -- nothing in the existing
atm__ features touches direction, only speed ratios/differences.

veer_10_100_k(t): signed angle difference between the 100m and 10m GFS wind direction at
each of the 9 GFS grid cells, wrapped to [-180, 180] -- per-grid-cell time series, passes
the (P2) gate in S6_ext_B_terrain.md sec B4.0.
jet_below_100_k(t): 1{u100 < u80} at each GFS cell -- a below-100m low-level-jet-style
inversion flag, distinct from the existing scalar alpha_100_80 shear ratio (that's a
continuous magnitude; this is a discrete direction-of-shear flag per grid cell).

0 fitted degrees of freedom (no thresholds tuned, angle wrap is a fixed formula).
"""
import sys, json
import numpy as np, pandas as pd
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/scratch')
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/nodes')
import harness

GRID_PIVOT = ('/Users/um-yunsang/BARAM2026/artifacts/cache/'
              '920be0c458d820e855bf79dd25723146f52ce1736138aedba5e6bc853f1f720b/'
              'train_grid_pivot.parquet')


def veer_features():
    import pyarrow.parquet as pq
    names = pq.ParquetFile(GRID_PIVOT).schema.names
    u10 = sorted(c for c in names if c.startswith('gfs__') and c.endswith('__heightAboveGround_10_10u'))
    v10 = sorted(c for c in names if c.startswith('gfs__') and c.endswith('__heightAboveGround_10_10v'))
    u80 = sorted(c for c in names if c.startswith('gfs__') and c.endswith('__heightAboveGround_80_u'))
    v80 = sorted(c for c in names if c.startswith('gfs__') and c.endswith('__heightAboveGround_80_v'))
    u100 = sorted(c for c in names if c.startswith('gfs__') and c.endswith('__heightAboveGround_100_100u'))
    v100 = sorted(c for c in names if c.startswith('gfs__') and c.endswith('__heightAboveGround_100_100v'))
    assert len(u10) == len(v10) == len(u80) == len(v80) == len(u100) == len(v100) == 9

    cols = ['forecast_kst_dtm'] + u10 + v10 + u80 + v80 + u100 + v100
    raw = pd.read_parquet(GRID_PIVOT, columns=cols)
    raw = raw.drop_duplicates('forecast_kst_dtm').set_index('forecast_kst_dtm').sort_index()

    U10 = raw[u10].to_numpy('float64'); V10 = raw[v10].to_numpy('float64')
    U80 = raw[u80].to_numpy('float64'); V80 = raw[v80].to_numpy('float64')
    U100 = raw[u100].to_numpy('float64'); V100 = raw[v100].to_numpy('float64')

    dir10 = np.degrees(np.arctan2(V10, U10))
    dir100 = np.degrees(np.arctan2(V100, U100))
    veer = (dir100 - dir10 + 180.0) % 360.0 - 180.0  # wrapped to [-180, 180]

    spd80 = np.hypot(U80, V80); spd100 = np.hypot(U100, V100)
    jet_below_100 = (spd100 < spd80).astype('float32')

    out = {}
    for k in range(9):
        out[f'veer10_100_k{k:02d}'] = veer[:, k].astype('float32')
        out[f'jet_below_100_k{k:02d}'] = jet_below_100[:, k]
    return pd.DataFrame(out, index=raw.index)


if __name__ == '__main__':
    A0, FR0, COLS0 = harness.surface(())  # default baseline, identical to S9-N0/S9-N1/S9-N3/S9-N4
    feats = veer_features()
    print(f'veer/jet features: {feats.shape[1]} columns', flush=True)

    FR = {}
    for g in (1, 2, 3):
        X = FR0[g].copy()
        FR[g] = pd.concat([X, feats.reindex(X.index)], axis=1)
    A = pd.concat(FR.values())
    COLS = COLS0 + list(feats.columns)

    key = ('PBL_VEER_F_v1',)
    harness._CACHE[key] = (A, FR, COLS)
    out = harness.run('S9-N5', 'terrainF_veer_and_jet_below_100_gfs9', blocks=key)
    print(json.dumps(out, indent=1, default=str))
