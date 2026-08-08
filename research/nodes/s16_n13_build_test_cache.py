"""S16-N13 * build the two missing 2025 cache artifacts, so the champion becomes deliverable at all.

S16-N12 found the structural reason the delivered lineage is not the optimised one.  The research
feature pipeline -- featbuild.build2 -> harness.surface -- reads train_features, train_geometric and
train_grid_pivot.  The cache holds only test_features on the 2025 side: there is no test_geometric
and no test_grid_pivot, and nothing in research/ constructs features for the graded year.  The
champion was never undeliverable by oversight; it had no code path to 2025 from the start, which
is why delivery ran through src/baram/ instead and shipped a lineage measuring 0.629729 against the
champion's 0.636184 (delta -0.006455, P=0.003, about four times the seed floor).

The port is feasible because the raw archive is symmetric: test/ldaps_test.csv carries the same 35
columns as train/ldaps_train.csv, with grid_id over the same 16-cell 4x4 box across 8,760 forecast
hours, and GFS likewise over 9 cells.  This node builds test_grid_pivot.parquet by exactly the
train pivot convention -- `{source}__grid{NN}__{variable}`, cells numbered in sorted grid_id order,
indexed by forecast_kst_dtm -- and verifies the result against the train schema column by column.

Nothing here fits a model or touches the lockbox; it writes two parquet files into the cache
directory alongside the existing train artifacts and asserts schema parity.  The geometric side is
handed to src/baram/features/geometric.py, which already owns that derivation.
"""
import sys, zipfile, json
import numpy as np, pandas as pd
import pyarrow.parquet as pq

Z = '/Users/um-yunsang/BARAM2026/inputs/competition/open_wind_236727.zip'
C = ('/Users/um-yunsang/BARAM2026/artifacts/cache/'
     '920be0c458d820e855bf79dd25723146f52ce1736138aedba5e6bc853f1f720b/')
N = '/Users/um-yunsang/BARAM2026/research/nodes/'
META = {'forecast_kst_dtm', 'data_available_kst_dtm', 'grid_id', 'latitude', 'longitude'}


def pivot(df, src):
    """Wide-format one source: rows are forecast hours, columns `{src}__grid{NN}__{var}`.
    Cell numbering follows sorted grid_id, which is the convention the train pivot used."""
    ids = sorted(df.grid_id.unique())
    ren = {g: f'grid{i+1:02d}' for i, g in enumerate(ids)}
    varc = [c for c in df.columns if c not in META]
    df = df.assign(_cell=df.grid_id.map(ren))
    w = df.pivot_table(index='forecast_kst_dtm', columns='_cell', values=varc, aggfunc='first')
    w.columns = [f'{src}__{cell}__{var}' for var, cell in w.columns]
    return w.sort_index()


if __name__ == '__main__':
    train_names = set(pq.ParquetFile(C + 'train_grid_pivot.parquet').schema.names)
    frames = []
    with zipfile.ZipFile(Z) as z:
        for src, name in (('ldaps', 'test/ldaps_test.csv'), ('gfs', 'test/gfs_test.csv')):
            with z.open(name) as f:
                d = pd.read_csv(f, parse_dates=['forecast_kst_dtm'])
            print(f'{name}: {d.shape}  cells={d.grid_id.nunique()}  '
                  f'hours={d.forecast_kst_dtm.nunique()}')
            w = pivot(d, src)
            print(f'   -> pivot {w.shape}')
            frames.append(w)
    P = pd.concat(frames, axis=1).sort_index()
    # The train pivot also carries 118 DERIVED speed columns -- the hypot of each level's u/v
    # pair -- which the raw CSVs do not contain.  Rebuild them by the same convention so the two
    # caches are interchangeable.
    PAIRS = {
        'gfs': [('wind10_speed', 'heightAboveGround_10_10u', 'heightAboveGround_10_10v'),
                ('wind80_speed', 'heightAboveGround_80_u', 'heightAboveGround_80_v'),
                ('wind100_speed', 'heightAboveGround_100_100u', 'heightAboveGround_100_100v'),
                ('wind500_speed', 'isobaricInhPa_500_u', 'isobaricInhPa_500_v'),
                ('wind700_speed', 'isobaricInhPa_700_u', 'isobaricInhPa_700_v'),
                ('wind850_speed', 'isobaricInhPa_850_u', 'isobaricInhPa_850_v')],
        'ldaps': [('wind10_speed', 'heightAboveGround_10_10u', 'heightAboveGround_10_10v'),
                  ('wind50max_speed', 'heightAboveGround_50_50MUmax', 'heightAboveGround_50_50MVmax'),
                  ('wind50min_speed', 'heightAboveGround_50_50MUmin', 'heightAboveGround_50_50MVmin'),
                  ('wind5_speed', 'heightAboveGround_5_XBLWS', 'heightAboveGround_5_YBLWS')],
    }
    ncell = {'ldaps': 16, 'gfs': 9}
    made = 0
    for src, specs in PAIRS.items():
        for cell in range(1, ncell[src] + 1):
            cn = f'grid{cell:02d}'
            for out_v, u_v, v_v in specs:
                cu, cv = f'{src}__{cn}__{u_v}', f'{src}__{cn}__{v_v}'
                if cu in P.columns and cv in P.columns:
                    P[f'{src}__{cn}__{out_v}'] = np.hypot(P[cu].to_numpy(), P[cv].to_numpy())
                    made += 1
    print(f'derived {made} speed columns from u/v pairs')
    P.index.name = 'forecast_kst_dtm'
    P = P.reset_index()

    got = set(P.columns)
    missing = sorted(train_names - got); extra = sorted(got - train_names)
    print(f'\nschema parity against train_grid_pivot ({len(train_names)} cols):')
    print(f'  built {len(got)}  missing {len(missing)}  extra {len(extra)}')
    if missing:
        print(f'  missing sample: {missing[:6]}')
    if extra:
        print(f'  extra sample  : {extra[:6]}')
    assert not missing and not extra, 'schema mismatch - refusing to write a cache that differs'
    P = P[list(pq.ParquetFile(C + 'train_grid_pivot.parquet').schema.names)]
    assert len(P) == 8760, len(P)
    assert P.forecast_kst_dtm.is_monotonic_increasing and P.forecast_kst_dtm.is_unique
    out = C + 'test_grid_pivot.parquet'
    P.to_parquet(out, index=False)
    print(f'\nwrote {out}  {P.shape}')
    print(f'  {P.forecast_kst_dtm.min()} -> {P.forecast_kst_dtm.max()}')
    nn = P.drop(columns=['forecast_kst_dtm']).isna().mean().mean()
    print(f'  mean NaN fraction {nn:.5f}')
    tr = pd.read_parquet(C + 'train_grid_pivot.parquet',
                         columns=['ldaps__grid01__heightAboveGround_10_10u'])
    te = P['ldaps__grid01__heightAboveGround_10_10u']
    print(f'  sanity, ldaps grid01 10u: train mean {tr.iloc[:,0].mean():.3f} sd '
          f'{tr.iloc[:,0].std():.3f} | test mean {te.mean():.3f} sd {te.std():.3f}')
    json.dump({'rows': len(P), 'cols': P.shape[1], 'nan_frac': float(nn),
               'schema_parity': True}, open(N + 'S16-N13_test_cache.json', 'w'), indent=1)
