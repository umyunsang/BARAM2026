"""S15-N1 * verify the NWP lane's central claim before building on it.

The lane reports, from descriptive statistics on our own parquet, that the correlation with
MEASURED hub wind orders the spatial reduction operators OPPOSITELY for the two sources:

    LDAPS   max over the 4x4 box  0.8475   >  inverse-distance weighting  0.7933
    GFS     inverse-distance      0.7200   >  max over the 3x3 box        0.5371

and that `ldaps_spatial__idw` ranks 6th of 7 and `ldaps_spatial__nearest` 7th of 7 among candidate
columns -- i.e. the two features this repository built specifically to do site transfer are its
worst, beaten by a plain max().

If true this is a build instruction, not a feature idea: it says one shared reduction rule is
provably wrong for one of the two sources, and it has a physical reading -- our 17 turbines sit on
the RIDGE, LDAPS under-resolves that ridge by 80-140 m, so the most exposed cell in the box is
closer to ridge-top conditions than a distance-weighted average that mixes in valley cells, while
GFS at 0.25 deg has no ridge to resolve at all and its box maximum is just noise.

This node reproduces the claim independently, on the measured hub wind from SCADA, per group and
per source, over every reduction operator we can form, and reports the ranking with sample sizes.
"""
import sys, json
import numpy as np, pandas as pd, pyarrow.parquet as pq
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/scratch')
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/nodes')

C = ('/Users/um-yunsang/BARAM2026/artifacts/cache/'
     '920be0c458d820e855bf79dd25723146f52ce1736138aedba5e6bc853f1f720b/')
S = '/Users/um-yunsang/BARAM2026/research/scratch/'
N = '/Users/um-yunsang/BARAM2026/research/nodes/'


def box(src, var):
    names = pq.ParquetFile(C + 'train_grid_pivot.parquet').schema.names
    pre = 'ldaps__' if src == 'ld' else 'gfs__'
    cols = sorted([c for c in names if c.startswith(pre) and c.endswith('__' + var)])
    G = pd.read_parquet(C + 'train_grid_pivot.parquet', columns=['forecast_kst_dtm'] + cols)
    G = G.set_index('forecast_kst_dtm').sort_index()
    M = G[cols].to_numpy('float64')
    M = np.where(np.isfinite(M), M, np.nanmean(M, 1, keepdims=True))
    return G.index, M


if __name__ == '__main__':
    T = pd.read_parquet(S + 'teacher_targets.parquet')
    idx_l, L = box('ld', 'wind50max_speed')
    idx_g, Gf = box('gf', 'wind100_speed')
    feat = pd.read_parquet(C + 'train_features.parquet',
                           columns=['forecast_kst_dtm', 'group_id',
                                    'ldaps_spatial__idw__wind50max_speed',
                                    'gfs_spatial__idw__wind100_speed'])
    print(f'LDAPS box {L.shape}, GFS box {Gf.shape}')

    ops = {'mean': lambda M: M.mean(1), 'max': lambda M: M.max(1), 'min': lambda M: M.min(1),
           'q90': lambda M: np.quantile(M, .9, axis=1), 'q75': lambda M: np.quantile(M, .75, axis=1),
           'median': lambda M: np.median(M, axis=1), 'q25': lambda M: np.quantile(M, .25, axis=1)}
    rows = []
    for g in (1, 2, 3):
        v = T[f'g{g}_v_mean']
        fi = feat[feat.group_id == g].set_index('forecast_kst_dtm')
        for src, M, idx in (('LDAPS50', L, idx_l), ('GFS100', Gf, idx_g)):
            for nm, fn in ops.items():
                s = pd.Series(fn(M), index=idx)
                j = pd.concat([s.rename('x'), v.rename('y')], axis=1).dropna()
                rows.append(dict(group=g, source=src, op=nm, n=len(j),
                                 corr=float(j.x.corr(j.y))))
            key = ('ldaps_spatial__idw__wind50max_speed' if src == 'LDAPS50'
                   else 'gfs_spatial__idw__wind100_speed')
            j = pd.concat([fi[key].rename('x'), v.rename('y')], axis=1).dropna()
            rows.append(dict(group=g, source=src, op='idw (repo)', n=len(j),
                             corr=float(j.x.corr(j.y))))
    D = pd.DataFrame(rows)
    P = D.pivot_table(index=['source', 'op'], columns='group', values='corr')
    P['mean'] = P.mean(axis=1)
    print('\n=== correlation of each spatial reduction with MEASURED hub wind, per group ===')
    for src in ('LDAPS50', 'GFS100'):
        print(f'\n{src}:')
        print(P.loc[src].sort_values('mean', ascending=False).round(4).to_string())
    best = {src: P.loc[src]['mean'].idxmax() for src in ('LDAPS50', 'GFS100')}
    print(f'\nBEST OPERATOR PER SOURCE: {best}')
    ld = P.loc['LDAPS50', 'mean']; gf = P.loc['GFS100', 'mean']
    print(f"\n  LDAPS: max {ld.get('max', float('nan')):.4f} vs idw(repo) "
          f"{ld.get('idw (repo)', float('nan')):.4f}  -> "
          f"{'CONFIRMED' if ld.get('max',0) > ld.get('idw (repo)',1) else 'NOT confirmed'}")
    print(f"  GFS:   idw {gf.get('idw (repo)', float('nan')):.4f} vs max "
          f"{gf.get('max', float('nan')):.4f}  -> "
          f"{'CONFIRMED' if gf.get('idw (repo)',0) > gf.get('max',1) else 'NOT confirmed'}")
    print('\n  => the ordering INVERTS between sources' if
          (ld.get('max', 0) > ld.get('idw (repo)', 1)) and (gf.get('idw (repo)', 0) > gf.get('max', 1))
          else '\n  => the ordering does NOT invert')
    json.dump({'table': D.to_dict('records'), 'best_per_source': best},
              open(N + 'S15-N1_reduction_verify.json', 'w'), indent=1, default=str)
