"""S12-N19 * CONTROL: run the organiser's own baseline on OUR local protocol.

Why this control was missing and why it is decisive.  The whole S12 diagnosis rests on the
claim that our point accuracy is below the field.  That claim was built by comparing our LOCAL
1-NMAE (0.864617, dev-2023 Q2/Q3/Q4) against ONLINE leaderboard 1-NMAE values (top-100 min
0.86777, median 0.87425, organiser RF baseline 0.86371) measured on the 2024 test period.
Those are different periods and different row sets.  AGENTS.md already records that the
local->online Total offset does not even transfer across method classes (3.2x difference), so
transferring a raw LEVEL across periods is exactly the mistake that file warns about.

The organiser's baseline is the one artifact whose online 1-NMAE is known (0.86371) and whose
recipe is fully public, so re-running it on our local folds measures the period offset directly:

    offset_1mnmae = baseline_ONLINE_1mnmae(0.86371) - baseline_LOCAL_1mnmae(measured here)

Then our local 0.864617 can be put on the online scale and compared with the field honestly.

Recipe reproduced from inputs/notebooks/baseline.ipynb (sha256 verified in AGENTS.md):
  features  = per-variable MEAN over the forecast grid for every ldaps and gfs column
              (= aggregate_weather), plus month/day/hour/dayofweek/is_weekend and the
              hour/month sin-cos pairs (= calendar_features)
  model     = RandomForestRegressor(n_estimators=120, max_depth=14, min_samples_leaf=8,
              max_features='sqrt', random_state=42), ONE MODEL PER GROUP, median imputation
  target    = raw kwh per group, clipped to [0, capacity]
  no decision layer, no policy grid -- the raw regression IS the submission
The only change is the evaluation window: the notebook fits on all train rows and predicts the
2024 test set; here it fits on each fold's expanding training window and predicts that fold.
"""
import sys, json, time
import numpy as np, pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/scratch')
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/nodes')
from lib import FOLDS, CAPS, official_total

N = '/Users/um-yunsang/BARAM2026/research/nodes/'
CACHE = ('/Users/um-yunsang/BARAM2026/artifacts/cache/'
         '920be0c458d820e855bf79dd25723146f52ce1736138aedba5e6bc853f1f720b/')
GRID = CACHE + 'train_grid_pivot.parquet'
BASELINE_ONLINE = {'total': 0.58792, 'one_minus_nmae': 0.86371, 'ficr': 0.31213}


def grid_mean_features():
    """= aggregate_weather(): per-variable mean over grid cells, for ldaps and gfs."""
    names = pq.ParquetFile(GRID).schema.names
    var_of = {}
    for c in names:
        if c == 'forecast_kst_dtm':
            continue
        parts = c.split('__')
        src, var = parts[0], '__'.join(parts[2:])
        var_of.setdefault(f'{src}_{var}_mean', []).append(c)
    G = pd.read_parquet(GRID).set_index('forecast_kst_dtm').sort_index()
    out = {k: G[v].mean(axis=1) for k, v in var_of.items()}
    X = pd.DataFrame(out, index=G.index)
    dt = X.index
    X['month'] = dt.month; X['day'] = dt.day; X['hour'] = dt.hour
    X['dayofweek'] = dt.dayofweek; X['is_weekend'] = dt.dayofweek.isin([5, 6]).astype(int)
    X['hour_sin'] = np.sin(2 * np.pi * X.hour / 24); X['hour_cos'] = np.cos(2 * np.pi * X.hour / 24)
    X['month_sin'] = np.sin(2 * np.pi * X.month / 12); X['month_cos'] = np.cos(2 * np.pi * X.month / 12)
    return X


if __name__ == '__main__':
    X = grid_mean_features()
    LAB = pd.read_parquet('/Users/um-yunsang/BARAM2026/research/scratch/labels.parquet').set_index('kst_dtm')
    print(f'baseline feature frame: {X.shape}')

    rows = []
    for f, (a, b) in FOLDS.items():
        a = pd.Timestamp(a); b = pd.Timestamp(b)
        tr = X.index < a; va = (X.index >= a) & (X.index <= b)
        imp = SimpleImputer(strategy='median')
        Xtr = imp.fit_transform(X[tr]); Xva = imp.transform(X[va])
        t0 = time.time()
        for g in (1, 2, 3):
            y = LAB[f'kpx_group_{g}'].reindex(X.index)
            m = tr & np.isfinite(y.to_numpy())
            mdl = RandomForestRegressor(n_estimators=120, max_depth=14, min_samples_leaf=8,
                                        max_features='sqrt', random_state=42, n_jobs=6)
            mdl.fit(imp.transform(X[m]), y[m])
            p = np.clip(mdl.predict(Xva), 0, CAPS[g])
            yv = y[va].to_numpy()
            keep = np.isfinite(yv)
            rows.append(pd.DataFrame({'fold_id': f, 'group_id': g,
                                      'forecast_kst_dtm': X.index[va][keep],
                                      'actual_kwh': yv[keep], 'prediction_kwh': p[keep]}))
        print(f'  {f} fitted {round(time.time()-t0,1)}s', flush=True)
    B = pd.concat(rows, ignore_index=True)
    s = official_total(B[['group_id', 'actual_kwh', 'prediction_kwh']])
    print(f'\nORGANISER BASELINE, our local 3-fold protocol:')
    print(f'  Total   = {s["total"]:.6f}   (online {BASELINE_ONLINE["total"]:.5f})')
    print(f'  1-NMAE  = {s["one_minus_nmae"]:.6f}   (online {BASELINE_ONLINE["one_minus_nmae"]:.5f})')
    print(f'  FICR    = {s["ficr"]:.6f}   (online {BASELINE_ONLINE["ficr"]:.5f})')
    off = BASELINE_ONLINE['one_minus_nmae'] - s['one_minus_nmae']
    print(f'\n  period offset on 1-NMAE (online - local) = {off:+.6f}')
    ours_local = 0.864617
    print(f'  our best local 1-NMAE {ours_local:.6f} -> on the online scale {ours_local+off:.6f}')
    print(f'  public top-100: min 0.86777, median 0.87425, first 0.87964')
    json.dump({'baseline_local': s, 'baseline_online': BASELINE_ONLINE,
               'offset_1mnmae_online_minus_local': off,
               'our_local_best_1mnmae': ours_local,
               'our_best_on_online_scale': ours_local + off},
              open(N + 'S12-N19_baseline_anchor.json', 'w'), indent=1, default=str)
    B.to_parquet(N + 'S12-N19_baseline_local.parquet', index=False)
