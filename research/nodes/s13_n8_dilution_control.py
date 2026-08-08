"""S13-N8 * is our feature-rejection protocol measuring INFORMATION or DILUTION?

Two physically unrelated, independently motivated, screen-passing feature blocks produced
almost the same loss on the same gate:

    S12-N14  component-grid statistics, 168 columns, 6-11% of model gain   ->  -0.000728
    S13-N7   flow-regime block (Hhat/Froude/Scorer/pressure gradient/wake),
             21 columns surviving a |corr|<0.85 novelty screen, 3.7-4.7% gain -> -0.000640

If two blocks with completely different content cost the same, the cost is unlikely to be about
their content.  The surface already carries 872 columns for 15-22k scored training rows and the
learner uses colsample_bytree = 0.4, so every added column reduces the probability that any
given split samples one of the genuinely useful ones.  That is a capacity/dilution effect, and
it would mean this project's long list of rejected feature blocks was partly measuring the
protocol rather than the features.

This node tests the protocol itself with three controls, all on the S12-N12 best point
configuration and the identical folds/rows/loss/seed:

  N   NOISE control       add 21 columns of pure Gaussian noise.  Under the information
                          hypothesis this costs ~0; under the dilution hypothesis it costs
                          about the same as the flow-regime block did.
  C   colsample control   re-run the flow-regime block with colsample_bytree raised from 0.40
                          to 0.55, restoring the per-split probability of sampling a useful
                          column.  If the flow block's loss disappears, dilution is confirmed
                          and the block's information was real.
  P1  prune control       DROP the 301-column geom__ block (the largest single block).
  P2  prune control       DROP the within-issuance lag/lead columns.
If pruning GAINS, the surface is over-wide and "add a feature" was the wrong move all along;
the S6 rung should be run backwards.
"""
import sys, json, time
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/scratch')
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/nodes')
from harness import surface, MU
from lib import FOLDS, CAPS, official_total
from s13_n7_flow_regime_gate import per_group_columns

N = '/Users/um-yunsang/BARAM2026/research/nodes/'
FR = '/Users/um-yunsang/BARAM2026/research/scratch/flow_regime.parquet'
L1P = dict(objective='l1', n_estimators=900, learning_rate=0.035, num_leaves=63,
           min_child_samples=40, subsample=0.85, subsample_freq=1, colsample_bytree=0.4,
           reg_lambda=3.0, random_state=20260801, n_jobs=6, verbose=-1)


def run():
    A, FRM, COLS = surface(('G2', 'DROP:grid__'))
    X = pd.read_parquet(FR)
    grp = A['grp'].to_numpy(); idx = A.index
    cf = A['cf'].to_numpy(); pct = A['pc_true'].to_numpy()
    valid = np.isfinite(cf) & (cf >= 0.1)
    w_prod = np.where(valid, np.clip(cf, 0, 1.2), 0.05)

    passed = json.load(open(N + 'S13-N7_flow_regime_gate.json'))['passed_cols']
    blocks = []
    for g in (1, 2, 3):
        m = grp == g
        Xi = per_group_columns(X, g).reindex(idx[m]); Xi.index = np.where(m)[0]
        blocks.append(Xi)
    NEW = pd.concat(blocks).sort_index(); NEW.index = A.index

    rng = np.random.default_rng(20260807)
    A_noise = A.copy()
    noise_cols = [f'zz__noise{i}' for i in range(len(passed))]
    for c in noise_cols:
        A_noise[c] = rng.standard_normal(len(A)).astype('float32')
    A_flow = A.copy()
    for c in passed:
        A_flow[c] = NEW[c].to_numpy()

    geom_cols = [c for c in COLS if c.startswith('geom__')]
    lag_cols = [c for c in COLS if '__lag' in c or c.endswith(('__d-1', '__d-2', '__d-3',
                                                               '__d1', '__d2', '__d3'))]
    print(f'block sizes: geom {len(geom_cols)}, lag/lead {len(lag_cols)}, '
          f'flow {len(passed)}, noise {len(noise_cols)}')

    CFG = {
        'BASE':  (A,       COLS,                                              0.40),
        'NOISE': (A_noise, COLS + noise_cols,                                 0.40),
        'FLOW':  (A_flow,  COLS + passed,                                     0.40),
        'FLOWC': (A_flow,  COLS + passed,                                     0.55),
        'BASEC': (A,       COLS,                                              0.55),
        'P1':    (A,       [c for c in COLS if c not in set(geom_cols)],      0.40),
        'P2':    (A,       [c for c in COLS if c not in set(lag_cols)],       0.40),
    }
    keys = []; preds = {}
    for f, (a, b) in FOLDS.items():
        a = pd.Timestamp(a); b = pd.Timestamp(b)
        tr = np.asarray(idx < a); va = np.asarray((idx >= a) & (idx <= b)); keep = np.isfinite(cf[va])
        keys.append(pd.DataFrame({'fold_id': f, 'group_id': grp[va][keep],
                                  'forecast_kst_dtm': idx[va][keep],
                                  'actual_kwh': cf[va][keep] * np.array([CAPS[g] for g in grp[va][keep]])}))
        for tag, (frame, cols, cs) in CFG.items():
            t0 = time.time()
            mp = dict(MU); mp['colsample_bytree'] = cs
            lp = dict(L1P); lp['colsample_bytree'] = cs
            m = tr & np.isfinite(pct)
            mu = lgb.LGBMRegressor(**mp)
            mu.fit(frame.loc[m, cols], pct[m], sample_weight=w_prod[m])
            pch = np.clip(mu.predict(frame[cols]), 0, 1)
            F = frame[cols].copy(); F['pc_hat'] = pch
            pm = lgb.LGBMRegressor(**lp)
            pm.fit(F[tr & valid], cf[tr & valid])
            preds.setdefault(tag, []).append(np.clip(pm.predict(F[va][keep]), 0, 1.1))
            print(f'  [{f}] {tag:6s} ncol={len(cols):4d} cs={cs} {round(time.time()-t0,1)}s', flush=True)

    K = pd.concat(keys, ignore_index=True)
    capv = K.group_id.map(CAPS).to_numpy()
    out = {}
    print('\n--- pooled 3-fold 1-NMAE ---')
    for k in CFG:
        v = np.concatenate(preds[k])
        s = official_total(K.assign(prediction_kwh=v * capv)[['group_id', 'actual_kwh', 'prediction_kwh']])
        out[k] = {'one_minus_nmae': s['one_minus_nmae'], 'ficr': s['ficr'], 'total': s['total'],
                  'n_cols': len(CFG[k][1]), 'colsample': CFG[k][2]}
        print(f'  {k:6s} ncol={len(CFG[k][1]):4d} cs={CFG[k][2]}  1-NMAE={s["one_minus_nmae"]:.6f}  '
              f'FICR={s["ficr"]:.6f}  Total={s["total"]:.6f}')
    b = out['BASE']['one_minus_nmae']
    print('\n--- deltas on 1-NMAE vs BASE ---')
    for k in CFG:
        if k != 'BASE':
            print(f'  {k:6s} {out[k]["one_minus_nmae"]-b:+.6f}')
    print(f'\n  DILUTION TEST: noise delta = {out["NOISE"]["one_minus_nmae"]-b:+.6f} '
          f'vs flow delta = {out["FLOW"]["one_minus_nmae"]-b:+.6f}')
    print(f'  colsample repair: FLOWC-BASEC = '
          f'{out["FLOWC"]["one_minus_nmae"]-out["BASEC"]["one_minus_nmae"]:+.6f}')
    json.dump(out, open(N + 'S13-N8_dilution_control.json', 'w'), indent=1, default=str)
    np.save(N + 'S13-N8_preds.npy', np.vstack([np.concatenate(preds[k]) for k in CFG]))
    K.to_parquet(N + 'S13-N8_keys.parquet', index=False)


if __name__ == '__main__':
    run()
