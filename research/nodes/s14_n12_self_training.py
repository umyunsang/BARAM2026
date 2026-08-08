"""S14-N12 (engine node F14) * self-training on the held-out inputs.

Cost-reduced by design.  The lane's version retrains the DART member; this runs the identical
mechanism on the point model (~30 s per fold instead of ~150 s), because the question is whether
pseudo-labelling the unlabelled manifold changes anything AT ALL, and the point model answers
that at a fifth of the cost.  If the mechanism is inert here it is inert there.

Protocol.  For each fold, the held-out fold's inputs stand in for the graded period (its labels
are of course never touched).  The champion's action on those rows is used as a pseudo-label,
the point model is refitted on training rows PLUS the pseudo-labelled held-out rows at weight w,
and the refitted model is then scored on the held-out fold against the same model without them.

What a null means, stated in advance.  The pseudo-labels ARE the champion's own predictions, so
this cannot inject information; it can only act as a manifold regulariser (the "expansion"
assumption behind self-training theory, which F12 already showed is unverified for this site --
the domain classifier separates adjacent 2023 quarters as easily as it separates train from
test). A null is therefore the expected outcome and is recorded as confirmatory, not as a
surprise.
"""
import sys, json, time
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/scratch')
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/nodes')
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/engine')
from harness import surface, MU
from lib import CAPS, FOLDS, official_total
from loop_lib import canonical_keys, align_prob, load_depavg, utility_frames, fo_policy, KEY

N = '/Users/um-yunsang/BARAM2026/research/nodes/'
L1P = dict(objective='l1', n_estimators=900, learning_rate=0.035, num_leaves=63,
           min_child_samples=40, subsample=0.85, subsample_freq=1, colsample_bytree=0.4,
           reg_lambda=3.0, random_state=20260801, n_jobs=6, verbose=-1)

if __name__ == '__main__':
    A, FRM, COLS = surface(('G2', 'DROP:grid__'))
    R = canonical_keys(); dep = load_depavg()
    Dm, _, _ = fo_policy(utility_frames(align_prob('D', R), R), R)
    J = Dm[KEY + ['actual_kwh', 'prediction_kwh']].rename(columns={'prediction_kwh': 'D'}).merge(dep, on=KEY)
    J['CHAMP'] = 0.30 * J.D + 0.70 * J.DEPAVG
    capJ = J.group_id.map(CAPS).to_numpy()
    pseudo = pd.Series((J.CHAMP / capJ).to_numpy(),
                       index=pd.MultiIndex.from_arrays([J.group_id, pd.to_datetime(J.forecast_kst_dtm)]))

    cf = A['cf'].to_numpy(); grp = A['grp'].to_numpy(); idx = A.index
    pct = A['pc_true'].to_numpy()
    valid = np.isfinite(cf) & (cf >= 0.1)
    w_prod = np.where(valid, np.clip(cf, 0, 1.2), 0.05)
    Aidx = pd.MultiIndex.from_arrays([grp, idx])
    ps = pseudo.reindex(Aidx).to_numpy()

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
        t0 = time.time()
        pm = lgb.LGBMRegressor(**L1P); pm.fit(F[rows], cf[rows])
        preds.setdefault('BASE', []).append(np.clip(pm.predict(F[va][keep]), 0, 1.1))
        # self-training arms: add the held-out inputs with the champion's action as a pseudo-label
        pl = va & np.isfinite(ps)
        for w in (0.25, 0.5, 1.0):
            Xa = pd.concat([F[rows], F[pl]])
            ya = np.r_[cf[rows], ps[pl]]
            wa = np.r_[np.ones(int(rows.sum())), np.full(int(pl.sum()), w)]
            sm = lgb.LGBMRegressor(**L1P); sm.fit(Xa, ya, sample_weight=wa)
            preds.setdefault(f'ST{w}', []).append(np.clip(sm.predict(F[va][keep]), 0, 1.1))
        print(f'  [{f}] base + 3 self-training arms {round(time.time()-t0,1)}s '
              f'(pseudo rows {int(pl.sum())})', flush=True)

    K = pd.concat(keys, ignore_index=True)
    capv = K.group_id.map(CAPS).to_numpy()
    out = {}
    print('\n--- pooled 3-fold point scores ---')
    for k in ['BASE', 'ST0.25', 'ST0.5', 'ST1.0']:
        v = np.concatenate(preds[k])
        s = official_total(K.assign(prediction_kwh=v * capv)[['group_id', 'actual_kwh', 'prediction_kwh']])
        out[k] = {'one_minus_nmae': s['one_minus_nmae'], 'ficr': s['ficr'], 'total': s['total'],
                  'pred_sd': float(np.std(v))}
        print(f'  {k:8s} 1-NMAE={s["one_minus_nmae"]:.6f}  FICR={s["ficr"]:.6f}  '
              f'Total={s["total"]:.6f}  sd(pred)={np.std(v):.4f}')
    b = out['BASE']
    print(f'\n  sharpness check (the lane\'s stated observable): sd(pred) BASE={b["pred_sd"]:.4f}')
    for k in ['ST0.25', 'ST0.5', 'ST1.0']:
        print(f'    {k:8s} d 1-NMAE={out[k]["one_minus_nmae"]-b["one_minus_nmae"]:+.6f}  '
              f'd sd(pred)={out[k]["pred_sd"]-b["pred_sd"]:+.4f}')
    json.dump(out, open(N + 'S14-N12_self_training.json', 'w'), indent=1, default=str)
