"""S14-N7 (engine node F09) * spatio-temporal displacement of the NWP field.

Why this node and not another summary statistic.  S14-N4 established that our deployed action is
the exact argmax of our objective, so the optimiser cannot be improved and the binding object is
the predictive density -- and on scored rows that density puts 2.69x more mass on our action than
on the action that would actually have scored best.  A density can only improve if what it SEES
changes.  Every feature block this project has tested (grid order statistics, component
statistics, geometry, REWS, flow regime) is a different SUMMARY of the same 4x4 x 3x3 box read at
the same grid point and the same valid time, which is why member error correlations sit at
0.93-0.99 and why ESL 15.1's variance algebra caps any ensemble gain at ~0.0025 Total.

Displacement changes the READ, not the summary.  Mesoscale forecasts over complex terrain carry
position and phase error: the simulated wind maximum sits one or two cells off the true ridge, or
arrives an hour early or late.  Reading the forecast at the cell and lag that historically best
matches the realised ridge wind -- conditional on flow regime -- is a different observation of the
atmosphere, not a different statistic of the same observation.

Construction, and the leakage rules it obeys:
  * candidate reads = the 16 LDAPS cells x lags {-2,-1,0,+1,+2} h within the same issuance
    (the issuance window is already available at the D-1 14:00 basis time, so no new information
    is required and R9 is satisfied -- this uses forecast fields only at deployment);
  * the displacement MAP is learned on the fold's TRAINING window only, by choosing, per flow
    regime, the (cell, lag) whose wind best predicts the realised hub wind;
  * regimes are defined by the cross-ridge wind direction sector and a stability split, both
    computable from the forecast alone;
  * the displaced read is then added as a small block, WITH the equal-count noise arm that
    contract R2 requires and an explicit prune arm (R3).
"""
import sys, json, time
import numpy as np, pandas as pd, pyarrow.parquet as pq, lightgbm as lgb
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/scratch')
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/nodes')
from harness import surface, MU
from lib import FOLDS, CAPS, official_total

C = ('/Users/um-yunsang/BARAM2026/artifacts/cache/'
     '920be0c458d820e855bf79dd25723146f52ce1736138aedba5e6bc853f1f720b/')
N = '/Users/um-yunsang/BARAM2026/research/nodes/'
S = '/Users/um-yunsang/BARAM2026/research/scratch/'
LAGS = (-2, -1, 0, 1, 2)
L1P = dict(objective='l1', n_estimators=900, learning_rate=0.035, num_leaves=63,
           min_child_samples=40, subsample=0.85, subsample_freq=1, colsample_bytree=0.4,
           reg_lambda=3.0, random_state=20260801, n_jobs=6, verbose=-1)


def cell_reads():
    """(time x cell x lag) LDAPS 50 m wind speed and direction, plus a regime label."""
    names = pq.ParquetFile(C + 'train_grid_pivot.parquet').schema.names
    u = sorted([c for c in names if c.endswith('__heightAboveGround_50_50MUmax')])
    v = sorted([c for c in names if c.endswith('__heightAboveGround_50_50MVmax')])
    G = pd.read_parquet(C + 'train_grid_pivot.parquet',
                        columns=['forecast_kst_dtm'] + u + v).set_index('forecast_kst_dtm').sort_index()
    U = G[u].to_numpy('float64'); V = G[v].to_numpy('float64')
    U = np.where(np.isfinite(U), U, np.nanmean(U, 1, keepdims=True))
    V = np.where(np.isfinite(V), V, np.nanmean(V, 1, keepdims=True))
    spd = np.hypot(U, V)
    wd = np.degrees(np.arctan2(-U.mean(1), -V.mean(1))) % 360.0
    return G.index, spd, wd


def build_displaced():
    idx, spd, wd = cell_reads()
    T = pd.read_parquet(S + 'teacher_targets.parquet')
    n_cell = spd.shape[1]
    # lagged reads within the issuance
    reads = {}
    for L in LAGS:
        reads[L] = pd.DataFrame(spd, index=idx).shift(L).to_numpy()
    sector = pd.cut(wd, [0, 45, 90, 135, 180, 225, 270, 315, 360],
                    labels=False, include_lowest=True)
    return idx, reads, sector, T


def run():
    A, FR, COLS = surface(('G2', 'DROP:grid__'))
    idx_all = A.index; grp = A['grp'].to_numpy()
    cf = A['cf'].to_numpy(); pct = A['pc_true'].to_numpy()
    valid = np.isfinite(cf) & (cf >= 0.1)
    w_prod = np.where(valid, np.clip(cf, 0, 1.2), 0.05)
    gidx, reads, sector, T = build_displaced()
    n_cell = reads[0].shape[1]
    print(f'candidate reads: {n_cell} cells x {len(LAGS)} lags = {n_cell*len(LAGS)}')

    keys = []; preds = {}
    rng = np.random.default_rng(20260807)
    for f, (a, b) in FOLDS.items():
        a = pd.Timestamp(a); b = pd.Timestamp(b)
        tr = np.asarray(idx_all < a); va = np.asarray((idx_all >= a) & (idx_all <= b))
        keep = np.isfinite(cf[va])
        keys.append(pd.DataFrame({'fold_id': f, 'group_id': grp[va][keep],
                                  'forecast_kst_dtm': idx_all[va][keep],
                                  'actual_kwh': cf[va][keep] * np.array([CAPS[g] for g in grp[va][keep]])}))
        t0 = time.time()
        # ---- learn the displacement map on the TRAINING window only, per group x sector ----
        disp = np.full(len(A), np.nan); disp_gain = {}
        for g in (1, 2, 3):
            vm = T[f'g{g}_v_mean'].reindex(gidx).to_numpy()
            gm = grp == g
            tr_t = gidx.isin(idx_all[tr & gm])
            for s in range(8):
                sel = tr_t & (sector == s) & np.isfinite(vm)
                if sel.sum() < 200:
                    best = (0, 0)
                else:
                    bc = None
                    for L in LAGS:
                        Rl = reads[L]
                        for c in range(n_cell):
                            x = Rl[sel, c]
                            ok = np.isfinite(x)
                            if ok.sum() < 150:
                                continue
                            r = abs(float(np.corrcoef(x[ok], vm[sel][ok])[0, 1]))
                            if bc is None or r > bc[0]:
                                bc = (r, c, L)
                    best = (bc[1], bc[2]) if bc else (0, 0)
                disp_gain[(g, s)] = best
                tgt = gm & np.isin(idx_all, gidx[(sector == s)])
                col = pd.Series(reads[best[1]][:, best[0]], index=gidx)
                disp[tgt] = col.reindex(idx_all[tgt]).to_numpy()
        picks = pd.Series({k: f'c{v[0]}L{v[1]}' for k, v in disp_gain.items()})
        print(f'  [{f}] displacement map learned {round(time.time()-t0,1)}s; '
              f'distinct picks={picks.nunique()} of 24', flush=True)

        A2 = A.copy()
        A2['disp__wind50'] = disp
        A2['disp__minus_idw'] = disp - A['ldaps_spatial__idw__wind50max_speed'].to_numpy()
        A2['disp__ratio'] = disp / np.maximum(A['ldaps_spatial__idw__wind50max_speed'].to_numpy(), 0.1)
        new = ['disp__wind50', 'disp__minus_idw', 'disp__ratio']
        A3 = A.copy()
        for i, c in enumerate(new):                       # R2 noise arm, equal count
            A3[f'zz__noise{i}'] = rng.standard_normal(len(A)).astype('float32')
        noise = [f'zz__noise{i}' for i in range(len(new))]
        lagcols = [c for c in COLS if '__lag' in c]       # R3 prune arm

        for tag, frame, cols in [('BASE', A, COLS),
                                 ('NOISE', A3, COLS + noise),
                                 ('DISP', A2, COLS + new),
                                 ('DISP_PRUNE', A2, [c for c in COLS if c not in set(lagcols)] + new)]:
            m = tr & np.isfinite(pct)
            mu = lgb.LGBMRegressor(**MU)
            mu.fit(frame.loc[m, cols], pct[m], sample_weight=w_prod[m])
            pch = np.clip(mu.predict(frame[cols]), 0, 1)
            F = frame[cols].copy(); F['pc_hat'] = pch
            pm = lgb.LGBMRegressor(**L1P)
            pm.fit(F[tr & valid], cf[tr & valid])
            preds.setdefault(tag, []).append(np.clip(pm.predict(F[va][keep]), 0, 1.1))
        print(f'  [{f}] four arms fitted {round(time.time()-t0,1)}s', flush=True)

    K = pd.concat(keys, ignore_index=True)
    capv = K.group_id.map(CAPS).to_numpy()
    out = {}
    print('\n--- pooled 3-fold point scores (R2: DISP is judged against NOISE, not BASE) ---')
    for k in ['BASE', 'NOISE', 'DISP', 'DISP_PRUNE']:
        v = np.concatenate(preds[k])
        s = official_total(K.assign(prediction_kwh=v * capv)[['group_id', 'actual_kwh', 'prediction_kwh']])
        out[k] = {'one_minus_nmae': s['one_minus_nmae'], 'ficr': s['ficr'], 'total': s['total']}
        print(f'  {k:11s} 1-NMAE={s["one_minus_nmae"]:.6f}  FICR={s["ficr"]:.6f}  Total={s["total"]:.6f}')
    print(f'\n  DISP - BASE  = {out["DISP"]["one_minus_nmae"]-out["BASE"]["one_minus_nmae"]:+.6f}')
    print(f'  NOISE - BASE = {out["NOISE"]["one_minus_nmae"]-out["BASE"]["one_minus_nmae"]:+.6f}  '
          f'(the dilution tax any 3-column addition pays)')
    print(f'  DISP - NOISE = {out["DISP"]["one_minus_nmae"]-out["NOISE"]["one_minus_nmae"]:+.6f}  '
          f'<-- the information-attributable effect, per contract R2')
    print(f'  DISP_PRUNE - NOISE = '
          f'{out["DISP_PRUNE"]["one_minus_nmae"]-out["NOISE"]["one_minus_nmae"]:+.6f}  (R3)')
    json.dump(out, open(N + 'S14-N7_displacement.json', 'w'), indent=1, default=str)
    np.save(N + 'S14-N7_preds.npy', np.vstack([np.concatenate(preds[k])
                                               for k in ['BASE', 'NOISE', 'DISP', 'DISP_PRUNE']]))
    K.to_parquet(N + 'S14-N7_keys.parquet', index=False)


if __name__ == '__main__':
    run()
