"""S13-N3 * (S5 rung) recalibrate the TEACHER TARGET, the root of the two-stage architecture.

S13-N2 measured that `pc_true` -- the physics capacity factor obtained by integrating each
group's fitted power curve over the measured 10-minute wind, and the target every downstream
stage is taught on -- is systematically wrong in identifiable wind regimes:

    measured hub wind   median(pc_true - cf)      mean pc_true / mean cf
      8-10 m/s            -0.021                   0.557 / 0.572   (curve too LOW)
     12-14 m/s            +0.054                   0.927 / 0.831   (curve too HIGH)
     14-16 m/s            +0.066                   0.970 / 0.826
     16-18 m/s            +0.064   (g3: +0.311)    0.932 / 0.805

The high-wind bias is the signature of storm control / de-rating that the fitted curve does not
represent; the 8-10 m/s bias is the steep part of the curve being under-fitted.  A monotone
recalibration of the target is therefore an S5 action, and because `pc_true` is the teacher of
the whole architecture it propagates to every stage below.

Treatments (teacher target only; features, folds, losses, weights, seeds all identical):
  T0  pc_true                          control, what the project uses today
  T1  isotonic MEDIAN calibration      monotone map pc_true -> median(cf | pc_true), per group.
                                       The metric's NMAE is an absolute loss, whose optimum is
                                       the conditional median, and outage mass is one-sided, so
                                       the median is the target-consistent choice.
  T2  isotonic MEAN calibration        monotone map pc_true -> E[cf | pc_true], per group.
  T3  clean-row median calibration     T1 but fitted only on rows without a detected outage
                                       (deficit < 0.05), i.e. calibrate the CURVE, not the
                                       curve-plus-outage-process.
Every map is fitted on the fold's TRAINING window only and applied unchanged to the held-out
fold, so no calibration information crosses the fold boundary.

Scored quantity: 1-NMAE of the resulting point forecast (the binding constraint per S12-N11),
using the best point configuration found in S12-N12 (LightGBM L1 on scored rows with the
teacher as an input feature).
"""
import sys, json, time
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/scratch')
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/nodes')
from harness import surface, MU
from lib import FOLDS, CAPS, official_total

N = '/Users/um-yunsang/BARAM2026/research/nodes/'
L1P = dict(objective='l1', n_estimators=900, learning_rate=0.035, num_leaves=63,
           min_child_samples=40, subsample=0.85, subsample_freq=1, colsample_bytree=0.4,
           reg_lambda=3.0, random_state=20260801, n_jobs=6, verbose=-1)
NB = 40


def monotone_map(x, y, kind='median', n_bins=NB):
    """Fit a monotone x->y map through binned conditional medians/means, PAVA-monotonised."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    e = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)))
    idx = np.clip(np.searchsorted(e, x, side='right') - 1, 0, len(e) - 2)
    cx, cy, cw = [], [], []
    for b in range(len(e) - 1):
        m = idx == b
        if m.sum() < 25:
            continue
        cx.append(float(x[m].mean()))
        cy.append(float(np.median(y[m]) if kind == 'median' else y[m].mean()))
        cw.append(int(m.sum()))
    ir = IsotonicRegression(increasing=True, out_of_bounds='clip')
    ir.fit(np.array(cx), np.array(cy), sample_weight=np.array(cw, float))
    return ir


def run():
    A, FR, COLS = surface(('G2', 'DROP:grid__'))
    cf = A['cf'].to_numpy(); grp = A['grp'].to_numpy(); idx = A.index
    pct = A['pc_true'].to_numpy()
    valid = np.isfinite(cf) & (cf >= 0.1)
    w_prod = np.where(valid, np.clip(cf, 0, 1.2), 0.05)
    deficit = pct - cf

    keys = []; preds = {}
    for f, (a, b) in FOLDS.items():
        a = pd.Timestamp(a); b = pd.Timestamp(b)
        tr = np.asarray(idx < a); va = np.asarray((idx >= a) & (idx <= b)); keep = np.isfinite(cf[va])
        keys.append(pd.DataFrame({'fold_id': f, 'group_id': grp[va][keep],
                                  'forecast_kst_dtm': idx[va][keep],
                                  'actual_kwh': cf[va][keep] * np.array([CAPS[g] for g in grp[va][keep]])}))
        # ---- build the four teacher targets on this fold's training window ----------------
        tgt = {'T0': pct.copy()}
        for name, kind, clean in [('T1', 'median', False), ('T2', 'mean', False), ('T3', 'median', True)]:
            v = np.full(len(A), np.nan)
            for g in (1, 2, 3):
                gm = grp == g
                fit = tr & gm & np.isfinite(pct) & np.isfinite(cf)
                if clean:
                    fit = fit & (deficit < 0.05)
                if fit.sum() < 500:
                    v[gm] = pct[gm]; continue
                ir = monotone_map(pct[fit], cf[fit], kind=kind)
                v[gm] = ir.predict(np.clip(np.nan_to_num(pct[gm], nan=0.0), 0, 1.2))
            tgt[name] = np.clip(v, 0, 1.2)

        for name, y in tgt.items():
            t0 = time.time()
            m = tr & np.isfinite(y)
            mu = lgb.LGBMRegressor(**MU)
            mu.fit(A.loc[m, COLS], y[m], sample_weight=w_prod[m])
            pch = np.clip(mu.predict(A[COLS]), 0, 1.2)
            F = A[COLS].copy(); F['pc_hat'] = pch
            pm = lgb.LGBMRegressor(**L1P)
            pm.fit(F[tr & valid], cf[tr & valid])
            preds.setdefault(name, []).append(np.clip(pm.predict(F[va][keep]), 0, 1.1))
            preds.setdefault(name + '_teacher', []).append(pch[va][keep])
            print(f'  [{f}] {name} {round(time.time()-t0,1)}s', flush=True)

    K = pd.concat(keys, ignore_index=True)
    capv = K.group_id.map(CAPS).to_numpy()
    out = {}
    print('\n--- 1-NMAE of the point forecast, by teacher target ---')
    for k in sorted(preds):
        v = np.concatenate(preds[k])
        s = official_total(K.assign(prediction_kwh=np.clip(v, 0, 1.1) * capv)[
            ['group_id', 'actual_kwh', 'prediction_kwh']])
        out[k] = {'one_minus_nmae': s['one_minus_nmae'], 'ficr': s['ficr'], 'total': s['total']}
        star = '  <-- control' if k == 'T0' else ''
        print(f'  {k:12s} 1-NMAE={s["one_minus_nmae"]:.6f}  FICR={s["ficr"]:.6f}  '
              f'Total={s["total"]:.6f}{star}')
    print(f'\n  best point config of S12-N12 (same pipeline, control teacher) = 0.865946')
    for k in ['T1', 'T2', 'T3']:
        print(f'  delta {k} vs T0 on 1-NMAE: {out[k]["one_minus_nmae"]-out["T0"]["one_minus_nmae"]:+.6f}')
    np.save(N + 'S13-N3_preds.npy', np.vstack([np.concatenate(preds[k]) for k in sorted(preds)]))
    json.dump({'scores': out, 'order': sorted(preds)},
              open(N + 'S13-N3_teacher_recalibration.json', 'w'), indent=1)
    K.to_parquet(N + 'S13-N3_keys.parquet', index=False)


if __name__ == '__main__':
    run()
