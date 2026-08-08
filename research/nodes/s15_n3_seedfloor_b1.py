"""S15-N3 * the seed-variance floor, and stage B1 re-measured against it.

TWO CORRECTIONS BUILT IN.

(a) MY OWN BUG.  S15-N2 replaced both the speed AND the direction columns of the LDAPS site
    transfer.  The verification in S15-N1 only ever established the claim for SPEED; the direction
    replacement used the meteorological convention arctan2(-u,-v) while the repository's existing
    `*_dir_sin/cos` columns evidently use a different one -- corr(old,new) came out at -0.15 to
    -0.25 and the mean of dir_sin moved 0.074 -> -0.404, a rotation, not an improvement.  This
    node replaces SPEEDS ONLY, which is exactly what was verified.

(b) THE LANE'S CORRECTION.  research/lanes/S15_sota_model.md rejects the winner's-curse reading of
    our feature results using our own arithmetic: sqrt(2 log k) predicts a ratio of 17.8 between
    the 872->620 and 872->893 effects and dilution predicts 12.0, while we MEASURED 0.60.  All the
    effects we have been interpreting -- noise arm -0.000411, physical block -0.000640, prune
    +0.000245, B1 -0.000495 -- sit inside [2.4e-4, 6.4e-4] despite a twelvefold difference in the
    size of the change.  That is the signature of a noise floor, and the lane's instruction is to
    settle it with seed refits rather than more interpretation.

So this node fits the SAME configuration under three different random seeds and reports the
spread.  If the seed spread covers the effects above, then every feature conclusion this project
has drawn at k=1 -- including the two I drew this session -- is unfalsified noise, and the only
admissible protocol is composition with reverse ablation.
"""
import sys, json, time
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/scratch')
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/nodes')
from harness import surface, MU
from lib import FOLDS, CAPS, official_total

S = '/Users/um-yunsang/BARAM2026/research/scratch/'
N = '/Users/um-yunsang/BARAM2026/research/nodes/'
L1P = dict(objective='l1', n_estimators=900, learning_rate=0.035, num_leaves=63,
           min_child_samples=40, subsample=0.85, subsample_freq=1, colsample_bytree=0.4,
           reg_lambda=3.0, n_jobs=6, verbose=-1)
SEEDS = (20260801, 20260802, 20260803)

if __name__ == '__main__':
    A, FR, COLS = surface(('G2', 'DROP:grid__'))
    X = pd.read_parquet(S + 'b1_reduction.parquet')
    speed_cols = [c for c in X.columns if c.endswith('_speed') and c in A.columns]
    print(f'B1 now replaces SPEED columns only: {speed_cols}')
    A2 = A.copy()
    for c in speed_cols:
        A2[c] = X[c].reindex(A.index).to_numpy()

    cf = A['cf'].to_numpy(); grp = A['grp'].to_numpy(); idx = A.index
    pct = A['pc_true'].to_numpy()
    valid = np.isfinite(cf) & (cf >= 0.1)
    w_prod = np.where(valid, np.clip(cf, 0, 1.2), 0.05)
    keys = []; preds = {}
    for f, (a, b) in FOLDS.items():
        a = pd.Timestamp(a); b = pd.Timestamp(b)
        tr = np.asarray(idx < a); va = np.asarray((idx >= a) & (idx <= b)); keep = np.isfinite(cf[va])
        keys.append(pd.DataFrame({'fold_id': f, 'group_id': grp[va][keep],
                                  'forecast_kst_dtm': idx[va][keep],
                                  'actual_kwh': cf[va][keep] * np.array([CAPS[g] for g in grp[va][keep]])}))
        for tag, frame in (('BASE', A), ('B1', A2)):
            for sd in SEEDS:
                t0 = time.time()
                mp = dict(MU); mp['random_state'] = sd
                lp = dict(L1P); lp['random_state'] = sd
                m = tr & np.isfinite(pct)
                mu = lgb.LGBMRegressor(**mp)
                mu.fit(frame.loc[m, COLS], pct[m], sample_weight=w_prod[m])
                pch = np.clip(mu.predict(frame[COLS]), 0, 1)
                F = frame[COLS].copy(); F['pc_hat'] = pch
                pm = lgb.LGBMRegressor(**lp); pm.fit(F[tr & valid], cf[tr & valid])
                preds.setdefault(f'{tag}_s{sd}', []).append(np.clip(pm.predict(F[va][keep]), 0, 1.1))
                print(f'  [{f}] {tag} seed={sd} {round(time.time()-t0,1)}s', flush=True)
    K = pd.concat(keys, ignore_index=True)
    capv = K.group_id.map(CAPS).to_numpy()

    def sc(v):
        return official_total(K.assign(prediction_kwh=v * capv)[
            ['group_id', 'actual_kwh', 'prediction_kwh']])
    res = {k: sc(np.concatenate(v)) for k, v in preds.items()}
    out = {'per_run': {k: {'total': r['total'], 'one_minus_nmae': r['one_minus_nmae'],
                           'ficr': r['ficr']} for k, r in res.items()}}
    print('\n=== SEED-VARIANCE FLOOR ===')
    for tag in ('BASE', 'B1'):
        t = np.array([res[f'{tag}_s{s}']['total'] for s in SEEDS])
        n1 = np.array([res[f'{tag}_s{s}']['one_minus_nmae'] for s in SEEDS])
        print(f'  {tag}: Total {np.round(t,6)}  spread={t.max()-t.min():.6f}  sd={t.std(ddof=1):.6f}')
        print(f'        1-NMAE {np.round(n1,6)}  spread={n1.max()-n1.min():.6f}')
        out[f'{tag}_total_spread'] = float(t.max() - t.min())
        out[f'{tag}_1mnmae_spread'] = float(n1.max() - n1.min())
    tb = np.array([res[f'BASE_s{s}']['total'] for s in SEEDS])
    t1 = np.array([res[f'B1_s{s}']['total'] for s in SEEDS])
    nb = np.array([res[f'BASE_s{s}']['one_minus_nmae'] for s in SEEDS])
    n1 = np.array([res[f'B1_s{s}']['one_minus_nmae'] for s in SEEDS])
    print(f'\n=== STAGE B1, seed-averaged (speeds only) ===')
    print(f'  Total  BASE {tb.mean():.6f} -> B1 {t1.mean():.6f}   delta {t1.mean()-tb.mean():+.6f}')
    print(f'  1-NMAE BASE {nb.mean():.6f} -> B1 {n1.mean():.6f}   delta {n1.mean()-nb.mean():+.6f}')
    print(f'\n  effects this project has been interpreting, against the seed floor '
          f'{max(out["BASE_total_spread"], out["B1_total_spread"]):.6f}:')
    for nm, v in [('noise arm (21 cols)', -0.000411), ('physical block (21 cols)', -0.000640),
                  ('prune 252 cols', +0.000245), ('B1 both-cols (buggy)', -0.000495),
                  ('displacement vs noise', -0.000329)]:
        print(f'    {nm:28s} {v:+.6f}')
    print('\n  3-seed averaging as an ENSEMBLE (the lane\'s Phase 2 lead):')
    for tag in ('BASE', 'B1'):
        avg = np.mean([np.concatenate(preds[f'{tag}_s{s}']) for s in SEEDS], axis=0)
        s = sc(avg)
        solo = np.mean([res[f'{tag}_s{sd}']['total'] for sd in SEEDS])
        print(f'    {tag}: mean-of-seeds Total={s["total"]:.6f} vs mean-of-solos {solo:.6f}  '
              f'gain={s["total"]-solo:+.6f}   1-NMAE={s["one_minus_nmae"]:.6f}')
        out[f'{tag}_seedavg'] = {'total': s['total'], 'one_minus_nmae': s['one_minus_nmae'],
                                 'ficr': s['ficr'], 'gain_vs_solo': s['total'] - solo}
    json.dump(out, open(N + 'S15-N3_seedfloor.json', 'w'), indent=1, default=str)
    np.save(N + 'S15-N3_preds.npy', np.vstack([np.concatenate(preds[k]) for k in sorted(preds)]))
    K.to_parquet(N + 'S15-N3_keys.parquet', index=False)
