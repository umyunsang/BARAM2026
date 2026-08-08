"""S14-N5 (engine node F02) * fit under the score's OWN penalty instead of L1 then post-processing.

Where this comes from.  The S14 foundation lane's strongest structural point: Boyd EE364a names
our reward.  FICR is the DEAD-ZONE LINEAR penalty phi(u) = max{0, |u| - a} with a = 0.06*capacity,
and slide 6.5 states that the "shape of penalty function affects distribution of residuals" --
dead-zone fitting produces a residual histogram with a spike inside the dead zone and heavy tails,
a shape that L1 or L2 fitting structurally cannot produce.  We have never once fitted under it.
We fit L1/L2 and then apply a two-scalar (T,G) post-hoc policy, which Murphy 5.7 says is a rank-1
approximation to what is actually a functional argmax over p(y|x).

S14-N4 sharpened why that matters.  Ng's optimisation-verification test is degenerate here in a
useful way: our deployed action IS the exact argmax of our objective over a 0.0025-spaced grid,
so the SEARCH cannot be improved and any remaining failure lives in the objective/density.  The
informative number from that node is that on scored rows the predictive density assigns 2.69x
more mass to our deployed action (mean q 0.2690) than to the action that would actually have
maximised the row's score (mean q 0.0999).

Treatment: a LightGBM custom objective that IS the per-row score contribution, smoothed only
enough to have gradients:

    L(r) = 0.5 * a_delta(r)              (smoothed |r|, the NMAE half)
         - 0.5 * (y / ybar_g) * K(r) / 4 (the FICR half, actual-weighted as the metric weights it)
    K(r) = 3*sigmoid((0.08-|r|)/tau) + 1*sigmoid((0.06-|r|)/tau)   -> ~4 inside 0.06, ~3 in the
                                                                      shell, ~0 outside
Trained on scored rows only (the truncated support the metric actually grades).

PRIMARY OBSERVABLE, pre-declared: not Total but the RESIDUAL SHAPE -- P(|r| <= 0.06) on scored
rows.  Boyd's claim is falsifiable there even if Total does not move, and Total is reported
second so a null on shape cannot be rescued by a lucky Total.
"""
import sys, json, time
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/scratch')
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/nodes')
from harness import surface, MU
from lib import FOLDS, CAPS, official_total

N = '/Users/um-yunsang/BARAM2026/research/nodes/'
BASE = dict(n_estimators=900, learning_rate=0.035, num_leaves=63, min_child_samples=40,
            subsample=0.85, subsample_freq=1, colsample_bytree=0.4, reg_lambda=3.0,
            random_state=20260801, n_jobs=6, verbose=-1)
DELTA = 0.004


def make_deadzone(y, wy, tau):
    # LGBMRegressor's custom-objective signature is (y_true, y_pred) -> (grad, hess).  The native
    # Booster signature is (preds, train_data).  Writing the native one against the sklearn API
    # silently makes r = y_true - y = 0, gradient 0, and no learning at all -- which is exactly
    # what the first two attempts produced (all variants identical, MAE 0.240, and the plumbing
    # control degenerating too).  The control is the only reason this was caught.
    def obj(y_true, y_pred):
        r = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
        ar = np.sqrt(r * r + DELTA * DELTA)
        dar = r / ar
        d2ar = DELTA * DELTA / (ar ** 3)
        s1 = 1.0 / (1.0 + np.exp(-(0.06 - ar) / tau))  # ~1 inside the +-0.06 band
        s2 = 1.0 / (1.0 + np.exp(-(0.08 - ar) / tau))
        b = (3.0 * s2 * (1 - s2) + 1.0 * s1 * (1 - s1)) / tau      # -dK/d(ar)
        grad = 0.5 * dar + 0.5 * (wy / 4.0) * b * dar
        # LightGBM's own regression_l1 uses a UNIT hessian; a near-zero analytic hessian makes
        # the leaf value -sum(grad)/sum(hess) explode, which is what produced the degenerate
        # constant fit on the first attempt (all three tau identical, MAE 0.240).
        hess = np.full_like(grad, 1.0)
        return grad, hess
    return obj


def make_l1_control(y):
    """The same plumbing with the FICR term switched off; must reproduce built-in L1."""
    def obj(y_true, y_pred):
        r = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
        ar = np.sqrt(r * r + DELTA * DELTA)
        return 0.5 * (r / ar), np.full_like(r, 1.0)
    return obj


def run():
    A, FR, COLS = surface(('G2', 'DROP:grid__'))
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
        m = tr & np.isfinite(pct)
        mu = lgb.LGBMRegressor(**MU)
        mu.fit(A.loc[m, COLS], pct[m], sample_weight=w_prod[m])
        pch = np.clip(mu.predict(A[COLS]), 0, 1)
        F = A[COLS].copy(); F['pc_hat'] = pch
        rows = tr & valid
        ybar = {g: float(np.nanmean(cf[rows & (grp == g)])) for g in (1, 2, 3)}
        wy = np.array([cf[i] / ybar[grp[i]] for i in np.where(rows)[0]])

        t0 = time.time()
        pm = lgb.LGBMRegressor(objective='l1', **BASE)
        pm.fit(F[rows], cf[rows])
        preds.setdefault('L1', []).append(np.clip(pm.predict(F[va][keep]), 0, 1.1))
        print(f'  [{f}] L1 {round(time.time()-t0,1)}s', flush=True)

        base0 = float(np.median(cf[rows]))
        init = np.full(int(rows.sum()), base0)
        # PLUMBING CONTROL: the identical custom-objective path with the FICR term removed must
        # reproduce the built-in L1 fit. If it does not, any dead-zone result is uninterpretable.
        t0 = time.time()
        cm = lgb.LGBMRegressor(objective=make_l1_control(cf[rows]), **BASE)
        cm.fit(F[rows], cf[rows], init_score=init)
        preds.setdefault('CTRL_L1', []).append(
            np.clip(cm.predict(F[va][keep], raw_score=True) + base0, 0, 1.1))
        print(f'  [{f}] CTRL_L1 (plumbing) {round(time.time()-t0,1)}s', flush=True)

        for tau in (0.010, 0.020, 0.035):
            t0 = time.time()
            dm = lgb.LGBMRegressor(objective=make_deadzone(cf[rows], wy, tau), **BASE)
            dm.fit(F[rows], cf[rows], init_score=init)
            p = dm.predict(F[va][keep], raw_score=True) + base0
            preds.setdefault(f'DZ{tau}', []).append(np.clip(p, 0, 1.1))
            print(f'  [{f}] DZ tau={tau} {round(time.time()-t0,1)}s  '
                  f'pred sd={np.std(p):.4f}', flush=True)

    K = pd.concat(keys, ignore_index=True)
    capv = K.group_id.map(CAPS).to_numpy()
    out = {}
    print('\n--- PRIMARY (pre-declared): residual shape on scored rows ---')
    y = K.actual_kwh.to_numpy() / capv
    sc = y >= 0.10
    for k in sorted(preds):
        v = np.concatenate(preds[k]); r = np.abs(v - y)
        hit6 = float((r[sc] <= 0.06).mean()); hit8 = float((r[sc] <= 0.08).mean())
        s = official_total(K.assign(prediction_kwh=v * capv)[['group_id', 'actual_kwh', 'prediction_kwh']])
        out[k] = {'hit6': hit6, 'hit8': hit8, 'mae': float(r[sc].mean()),
                  'one_minus_nmae': s['one_minus_nmae'], 'ficr': s['ficr'], 'total': s['total']}
        star = '  <-- control' if k == 'L1' else ''
        print(f'  {k:9s} P(|r|<=0.06)={hit6:.4f}  P(|r|<=0.08)={hit8:.4f}  MAE={r[sc].mean():.5f}'
              f'{star}')
    print('\n--- SECONDARY: official decomposition ---')
    for k in sorted(preds):
        o = out[k]
        print(f'  {k:9s} 1-NMAE={o["one_minus_nmae"]:.6f}  FICR={o["ficr"]:.6f}  Total={o["total"]:.6f}')
    b = out['L1']
    print(f'\n  Boyd prediction: dead-zone fitting raises P(|r|<=0.06) at the cost of MAE.')
    for k in sorted(preds):
        if k != 'L1':
            print(f'    {k:9s} d hit6={out[k]["hit6"]-b["hit6"]:+.4f}   '
                  f'd MAE={out[k]["mae"]-b["mae"]:+.5f}   d Total={out[k]["total"]-b["total"]:+.6f}')
    np.save(N + 'S14-N5_preds.npy', np.vstack([np.concatenate(preds[k]) for k in sorted(preds)]))
    json.dump({'scores': out, 'order': sorted(preds)},
              open(N + 'S14-N5_deadzone.json', 'w'), indent=1, default=str)
    K.to_parquet(N + 'S14-N5_keys.parquet', index=False)


if __name__ == '__main__':
    run()
