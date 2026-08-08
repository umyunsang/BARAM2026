
"""S9-N4 · terrain lane candidate B (upstream_projection), the "reopened representation
axis" per research/lanes/S6_ext_B_terrain.md sec B4 -- explicitly recommended to try
BEFORE candidate A (Sx_grid) because it is far cheaper ("cheap failure first") and needs
no DEM, just turbine-group and grid coordinates plus wind direction.

Idea: for each timestep, project each LDAPS grid cell's position onto the group's current
wind-direction axis (upwind distance) and its perpendicular (cross-track distance), then
take a Gaussian-kernel-weighted average of grid wind speed, weighted toward cells that are
upwind and close to the flow axis. Bandwidths L=3km (along-flow), sigma=2km (cross-flow)
are FIXED a priori (predeclared in the source write-up, not tuned here) -- 0 fitted degrees
of freedom.

Differs from the already-accepted G2 block (research/nodes/s6feats.py::block_G2) in two
ways the source write-up flags as the reason this might carry independent information: (1)
G2's origin is each grid PATCH's own centroid, this uses each TURBINE GROUP's actual
centroid (parsed from the immutable info.xlsx via the canonical src/baram/data/turbines.py
parser -- 3 distinct locations, not 1 shared patch center); (2) G2 selects a single argmin/
argmax extreme grid cell (discrete, noise-sensitive), this uses a smooth Gaussian-kernel
weighted average over all cells (continuous). Per the write-up's own warning, real overlap
risk exists; this is reported as an isolated single-axis test against the same S9-N0
control, same as every other S9 candidate this session.

LDAPS 10m wind only for this first pass (16 grid cells) -- GFS layers are a natural
follow-up if this shows signal, not added here to keep the first test cheap per the
write-up's "cheap failure first" recommendation.
"""
import sys, json, zipfile
import numpy as np, pandas as pd
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/src')
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/scratch')
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/nodes')
import harness
from baram.data.turbines import parse_turbine_workbook, group_static_metadata

GRID_PIVOT = ('/Users/um-yunsang/BARAM2026/artifacts/cache/'
              '920be0c458d820e855bf79dd25723146f52ce1736138aedba5e6bc853f1f720b/'
              'train_grid_pivot.parquet')
ZIP_PATH = '/Users/um-yunsang/BARAM2026/inputs/competition/open_wind_236727.zip'
COORDS_PATH = '/Users/um-yunsang/BARAM2026/research/scratch/grid_coords.json'
L_KM = 3.0   # along-flow bandwidth, fixed a priori
SIGMA_KM = 2.0  # cross-flow bandwidth, fixed a priori
M_PER_DEG_LAT = 111320.0


def group_centroids():
    with zipfile.ZipFile(ZIP_PATH) as z:
        payload = z.read('info.xlsx')
    turbines = parse_turbine_workbook(payload)
    meta = group_static_metadata(turbines)
    return {int(r.group_id): (float(r.latitude_centroid), float(r.longitude_centroid))
            for r in meta.itertuples()}


def local_xy(lat, lon, lat0):
    y = (lat - lat0) * M_PER_DEG_LAT / 1000.0  # km
    x = (lon - np.mean(lon)) * M_PER_DEG_LAT * np.cos(np.deg2rad(lat0)) / 1000.0
    return x, y


def upwind_features():
    coords = json.load(open(COORDS_PATH))
    ldaps_ids = sorted(coords['ldaps'], key=int)
    lat_j = np.array([coords['ldaps'][i][0] for i in ldaps_ids])
    lon_j = np.array([coords['ldaps'][i][1] for i in ldaps_ids])
    lat0 = float(lat_j.mean())
    x_j, y_j = local_xy(lat_j, lon_j, lat0)  # km, shape (16,)

    groups = group_centroids()
    gx = {}; gy = {}
    for g, (glat, glon) in groups.items():
        gy[g] = (glat - lat0) * M_PER_DEG_LAT / 1000.0
        gx[g] = (glon - np.mean(lon_j)) * M_PER_DEG_LAT * np.cos(np.deg2rad(lat0)) / 1000.0

    import pyarrow.parquet as pq
    schema_names = pq.ParquetFile(GRID_PIVOT).schema.names
    us = sorted(c for c in schema_names if c.startswith('ldaps__') and c.endswith('__heightAboveGround_10_10u'))
    vs = sorted(c for c in schema_names if c.startswith('ldaps__') and c.endswith('__heightAboveGround_10_10v'))
    assert len(us) == len(vs) == 16, (len(us), len(vs))
    raw = pd.read_parquet(GRID_PIVOT, columns=['forecast_kst_dtm'] + us + vs)
    raw = raw.drop_duplicates('forecast_kst_dtm').set_index('forecast_kst_dtm').sort_index()
    U = raw[us].to_numpy('float64')  # (n_rows, 16)
    V = raw[vs].to_numpy('float64')
    speed = np.hypot(U, V)
    um = np.nanmean(U, axis=1); vm = np.nanmean(V, axis=1)
    nrm = np.maximum(np.hypot(um, vm), 1e-6)
    ex = um / nrm; ey = vm / nrm  # unit flow-direction vector, per timestep

    out = {}
    for g in sorted(groups):
        dx = x_j[None, :] - gx[g]  # (n_rows-broadcast, 16) -- constant across rows, but keep shape
        dy = y_j[None, :] - gy[g]
        dx = np.broadcast_to(dx, (len(raw), 16))
        dy = np.broadcast_to(dy, (len(raw), 16))
        upwind = -(ex[:, None] * dx + ey[:, None] * dy)          # km, +ve = grid j is upwind
        cross = np.abs(-ex[:, None] * dy + ey[:, None] * dx)     # km, lateral deviation
        w = np.exp(upwind / L_KM) * np.exp(-(cross ** 2) / (2 * SIGMA_KM ** 2))
        wsum = np.maximum(w.sum(axis=1), 1e-12)
        upw_wspd = (w * speed).sum(axis=1) / wsum
        nearest_j = int(np.argmin(np.hypot(x_j - gx[g], y_j - gy[g])))
        wind10_nearest = speed[:, nearest_j]
        out[f'upw_wspd_g{g}'] = upw_wspd.astype('float32')
        out[f'upw_wspd_ratio_g{g}'] = (upw_wspd / np.maximum(wind10_nearest, 1e-3)).astype('float32')
    return pd.DataFrame(out, index=raw.index)


if __name__ == '__main__':
    A0, FR0, COLS0 = harness.surface(())  # default baseline, identical to S9-N0/S9-N1/S9-N3
    feats = upwind_features()
    print(f'upwind features: {feats.shape[1]} columns, groups={sorted(group_centroids())}', flush=True)

    FR = {}
    for g in (1, 2, 3):
        X = FR0[g].copy()
        FR[g] = pd.concat([X, feats.reindex(X.index)], axis=1)
    A = pd.concat(FR.values())
    COLS = COLS0 + list(feats.columns)

    key = ('UPWIND_B_v1',)
    harness._CACHE[key] = (A, FR, COLS)
    out = harness.run('S9-N4', 'terrainB_upwind_projection_groupcentroid_kernel', blocks=key)
    print(json.dumps(out, indent=1, default=str))
