"""S13-N4 * (S5 rung) metric-aligned row weighting and outage-block removal.

Two S5 defects fall straight out of the S13-N1/N2 decomposition.

DEFECT 1 -- GROUP WEIGHT MISMATCH.  Both halves of the metric are computed PER GROUP and then
averaged, so every group carries exactly 1/3 of the score.  Our models are trained pooled with
uniform row weights, and group 3 has no 2022 labels, so its share of the scored training rows is

    fold Q2  g1 45.5%  g2 45.5%  g3  9.0%     (g3 under-weighted 3.7x versus its metric share)
    fold Q3  g1 43.4%  g2 43.3%  g3 13.3%
    fold Q4  g1 42.3%  g2 42.3%  g3 15.4%

Group 3 is simultaneously our worst group (NMAE 0.15142 versus 0.13054 / 0.13245; FICR 0.35283
versus 0.40454 / 0.47415).  Re-weighting rows by 1/n_g makes the training objective match the
scoring objective exactly.  This costs zero fitted degrees of freedom -- the weights are read
off the metric definition, not tuned -- and it is the same 0-dof, read-off-the-scorer move that
the accepted_gain_ledger already credits with +0.002621 for the cf>=0.10 argmax condition.
The external lane independently reports three separate competing teams adopting group-3
specific treatment.

DEFECT 2 -- OUTAGE ROWS TEACH A CORRUPTED MAP.  S13-N2 measured deficit = pc_true - cf with
lag-1 autocorrelation 0.90 and lag-24 0.26-0.34, i.e. availability losses arrive in multi-hour
BLOCKS, not as independent per-row noise.  17.5% of scored rows carry a deficit >= 0.05.  The
point model of S12-N12 trains on every scored row including those, so a sixth of its training
signal says "this atmospheric state produced little power" when the truth is "the fleet was
partly down".  The project's existing availability gate is a per-row threshold applied to the
old harness calibration step, never to the point model, and it is not block-aware.

Variants (feature set, loss, folds, seed and hyper-parameters all identical to the S12-N12 best
point configuration L1P; only the training rows/weights change):
  V0  control, uniform weights on scored rows
  V1  group-balanced weights  w_i = 1 / n_{g(i)}   (exact metric alignment)
  V2  outage-BLOCK removal    drop runs of >= 2 consecutive hours with deficit >= 0.05
  V3  V1 + V2
  V4  softened balancing      w_i proportional to n_{g(i)}^-0.5, in case full balancing
                              over-corrects on the fold where g3 is scarcest
"""
import sys, json, time
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/scratch')
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/nodes')
from harness import surface, MU
from lib import FOLDS, CAPS, official_total

N = '/Users/um-yunsang/BARAM2026/research/nodes/'
L1P = dict(objective='l1', n_estimators=900, learning_rate=0.035, num_leaves=63,
           min_child_samples=40, subsample=0.85, subsample_freq=1, colsample_bytree=0.4,
           reg_lambda=3.0, random_state=20260801, n_jobs=6, verbose=-1)


def outage_blocks(idx, grp, deficit, th=0.05, run=2):
    """Flag rows inside runs of >= `run` consecutive hours with deficit >= th, per group."""
    flag = np.zeros(len(idx), bool)
    raw = deficit >= th
    for g in (1, 2, 3):
        m = np.where(grp == g)[0]
        order = m[np.argsort(idx[m].values)]
        r = raw[order].astype(int)
        if len(r) == 0:
            continue
        # contiguous-run lengths over the (hourly) ordered series
        start = 0
        for i in range(1, len(r) + 1):
            if i == len(r) or r[i] != r[start]:
                if r[start] == 1 and (i - start) >= run:
                    flag[order[start:i]] = True
                start = i
    return flag


def run():
    A, FR, COLS = surface(('G2', 'DROP:grid__'))
    cf = A['cf'].to_numpy(); grp = A['grp'].to_numpy(); idx = A.index
    pct = A['pc_true'].to_numpy()
    valid = np.isfinite(cf) & (cf >= 0.1)
    w_prod = np.where(valid, np.clip(cf, 0, 1.2), 0.05)
    deficit = np.where(np.isfinite(pct) & np.isfinite(cf), pct - cf, 0.0)
    ob = outage_blocks(idx, grp, deficit)
    print(f'outage-block rows: {ob.sum()} / {len(A)} ({ob.mean():.2%}); '
          f'of scored rows {ob[valid].mean():.2%}')

    keys = []; preds = {}
    for f, (a, b) in FOLDS.items():
        a = pd.Timestamp(a); b = pd.Timestamp(b)
        tr = np.asarray(idx < a); va = np.asarray((idx >= a) & (idx <= b)); keep = np.isfinite(cf[va])
        keys.append(pd.DataFrame({'fold_id': f, 'group_id': grp[va][keep],
                                  'forecast_kst_dtm': idx[va][keep],
                                  'actual_kwh': cf[va][keep] * np.array([CAPS[g] for g in grp[va][keep]])}))
        m = tr & np.isfinite(pct)
        mu = lgb.LGBMRegressor(**MU)
        mu.fit(A.loc[m, COLS], pct[m], sample_weight=w_prod[m])
        pch = np.clip(mu.predict(A[COLS]), 0, 1)
        F = A[COLS].copy(); F['pc_hat'] = pch

        base = tr & valid
        ng = {g: max(int((base & (grp == g)).sum()), 1) for g in (1, 2, 3)}
        wbal = np.array([1.0 / ng[g] for g in grp])
        wsoft = np.array([ng[g] ** -0.5 for g in grp])
        variants = {
            'V0': (base, None),
            'V1': (base, wbal),
            'V2': (base & ~ob, None),
            'V3': (base & ~ob, wbal),
            'V4': (base, wsoft),
        }
        for nm, (rows, w) in variants.items():
            t0 = time.time()
            pm = lgb.LGBMRegressor(**L1P)
            pm.fit(F[rows], cf[rows], sample_weight=None if w is None else w[rows])
            preds.setdefault(nm, []).append(np.clip(pm.predict(F[va][keep]), 0, 1.1))
            print(f'  [{f}] {nm} n={int(rows.sum())} {round(time.time()-t0,1)}s', flush=True)

    K = pd.concat(keys, ignore_index=True)
    capv = K.group_id.map(CAPS).to_numpy()
    out = {}
    print('\n--- 1-NMAE of the point forecast, by training-row treatment ---')
    for k in sorted(preds):
        v = np.concatenate(preds[k])
        s = official_total(K.assign(prediction_kwh=v * capv)[['group_id', 'actual_kwh', 'prediction_kwh']])
        out[k] = {'one_minus_nmae': s['one_minus_nmae'], 'ficr': s['ficr'], 'total': s['total'],
                  'group_nmae': s['group_nmae']}
        star = '  <-- control' if k == 'V0' else ''
        print(f'  {k} 1-NMAE={s["one_minus_nmae"]:.6f}  FICR={s["ficr"]:.6f}  Total={s["total"]:.6f}  '
              f'gNMAE={ {g: round(x,5) for g,x in s["group_nmae"].items()} }{star}')
    for k in ['V1', 'V2', 'V3', 'V4']:
        print(f'  delta {k} vs V0 on 1-NMAE: {out[k]["one_minus_nmae"]-out["V0"]["one_minus_nmae"]:+.6f}')
    np.save(N + 'S13-N4_preds.npy', np.vstack([np.concatenate(preds[k]) for k in sorted(preds)]))
    json.dump({'scores': out, 'order': sorted(preds), 'outage_row_share': float(ob[valid].mean())},
              open(N + 'S13-N4_metric_aligned_rows.json', 'w'), indent=1, default=str)
    K.to_parquet(N + 'S13-N4_keys.parquet', index=False)


if __name__ == '__main__':
    run()
