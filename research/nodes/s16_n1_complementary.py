"""S16-N1 * a member built to be COMPLEMENTARY, not accurate.

The finding that forces this.  Every member this project has produced was trained toward accuracy,
and they are all 0.92-0.99 error-correlated.  The most accurate member ever built here -- the
composed pipeline of S15 -- turned out to be the LEAST complementary at 0.9643 against the
incumbent D, higher even than D's 0.9217 against DEPAVG, and lost the blend despite better solo
and better point accuracy.  So accuracy is not the axis; complementarity is.

Two constructions, both leaving the champion untouched and both aimed at the same thing --
capacity spent where the incumbent already fails:

  FOCUS  negative-correlation learning in its simplest honest form.  The DART classifier's sample
         weights are multiplied by a factor that rises on rows where the CHAMPION misses the
         settlement band, so the member's capacity is allocated to the incumbent's failures rather
         than to the rows the incumbent already wins.  Weight = 1 on a champion hit within 0.06,
         rising to `boost` on a clean miss, interpolated through the 0.08 shell.

  STACK  the champion's own action enters as a feature.  The member then models
         p(cf | x, a_champion) instead of p(cf | x), which is the standard stacking construction
         and lets it learn WHERE the incumbent is wrong rather than rediscovering what it is right
         about.  Leakage discipline: a_champion on training rows is in-sample for D exactly as
         `pc_hat` already is, so this introduces no new leakage class; the fold-outside policy and
         blend weight are unchanged.

Scored as a 3-seed average per contract R10, against the honest champion baseline 0.634573 +-
0.000849 rather than the deployed 0.636184.  The quantity that decides success is not the blend
score alone but the ERROR CORRELATION with D and DEPAVG: a member that lands near 0.85 while
holding its solo is the thing this project has never had.
"""
import sys, json, time
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/scratch')
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/nodes')
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/engine')
from harness import surface, MU
from lib import FOLDS, CAPS, official_total
from loop_lib import canonical_keys, align_prob, load_depavg, utility_frames, fo_policy, KEY, W
from arbiter import arbitrate

N = '/Users/um-yunsang/BARAM2026/research/nodes/'
NC = 26
SEEDS = (20260803, 20260804, 20260805)
BOOST = 4.0
DART = dict(objective='multiclass', boosting_type='dart', n_estimators=400, learning_rate=0.08,
            num_leaves=31, min_child_samples=60, subsample=0.85, subsample_freq=1,
            colsample_bytree=0.4, reg_lambda=3.0, n_jobs=6, verbose=-1)


def blend(P, R, dep):
    Dm, solo, _ = fo_policy(utility_frames(P, R), R)
    J = Dm[KEY + ['actual_kwh', 'prediction_kwh']].rename(columns={'prediction_kwh': 'M'}).merge(dep, on=KEY)
    rows = []
    for f in FOLDS:
        oth = J[J.fold_id != f]; held = J[J.fold_id == f]; best = None
        for w in np.arange(0, 1.001, 0.05):
            t = official_total(oth.assign(prediction_kwh=w * oth.M + (1 - w) * oth.DEPAVG)[
                ['group_id', 'actual_kwh', 'prediction_kwh']])['total']
            if best is None or t > best[0]:
                best = (t, w)
        rows.append(held.assign(pred=best[1] * held.M + (1 - best[1]) * held.DEPAVG, w=best[1]))
    B = pd.concat(rows, ignore_index=True)
    return B, solo['total'], official_total(B.assign(prediction_kwh=B.pred)[
        ['group_id', 'actual_kwh', 'prediction_kwh']])


if __name__ == '__main__':
    A, FR, COLS = surface(('G2', 'DROP:grid__'))
    R = canonical_keys(); dep = load_depavg()
    Dm, _, _ = fo_policy(utility_frames(align_prob('D', R), R), R)
    J = Dm[KEY + ['actual_kwh', 'prediction_kwh']].rename(columns={'prediction_kwh': 'D'}).merge(dep, on=KEY)
    J['CHAMP'] = 0.30 * J.D + 0.70 * J.DEPAVG
    capJ = J.group_id.map(CAPS).to_numpy()
    ch = pd.Series((J.CHAMP / capJ).to_numpy(),
                   index=pd.MultiIndex.from_arrays([J.group_id, pd.to_datetime(J.forecast_kst_dtm)]))

    grp = A['grp'].to_numpy(); idx = A.index
    cf = A['cf'].to_numpy(); pct = A['pc_true'].to_numpy()
    valid = np.isfinite(cf) & (cf >= 0.1)
    w_prod = np.where(valid, np.clip(cf, 0, 1.2), 0.05)
    w_valid = np.where(valid, 1.0, 0.15)
    gapv = pct - cf
    a_ch = ch.reindex(pd.MultiIndex.from_arrays([grp, idx])).to_numpy()
    err_ch = np.abs(a_ch - cf)
    focus = np.where(np.isfinite(err_ch),
                     1.0 + (BOOST - 1.0) * np.clip((err_ch - 0.06) / 0.02, 0, 1), 1.0)
    print(f'champion action available on {np.isfinite(a_ch).sum()} of {len(A)} rows; '
          f'focus weight mean={np.nanmean(focus):.3f} (1 on a band hit, {BOOST} on a clean miss)')

    probs = {f'{arm}_{s}': [] for arm in ('FOCUS', 'STACK') for s in SEEDS}
    rows = []
    for f, (a, b) in FOLDS.items():
        a = pd.Timestamp(a); b = pd.Timestamp(b)
        tr = np.asarray(idx < a); va = np.asarray((idx >= a) & (idx <= b)); keep = np.isfinite(cf[va])
        rows.append(pd.DataFrame({'fold_id': f, 'group_id': grp[va][keep],
                                  'forecast_kst_dtm': idx[va][keep], 'cf': cf[va][keep],
                                  'mean_gen_g': [float(np.nanmean(cf[tr & (grp == g)])) for g in grp[va][keep]]}))
        cls = np.clip(np.nan_to_num(cf, nan=0.0) / W, 0, NC - 1).astype(int)
        cm = tr & np.isfinite(cf) & (~(gapv >= 0.05))
        for sd in SEEDS:
            t0 = time.time()
            mp = dict(MU); mp['random_state'] = sd
            m = tr & np.isfinite(pct)
            mu = lgb.LGBMRegressor(**mp)
            mu.fit(A.loc[m, COLS], pct[m], sample_weight=w_prod[m])
            pc = np.clip(mu.predict(A[COLS]), 0, 1)
            sel = list(pd.Series(mu.feature_importances_, index=COLS).sort_values(ascending=False).head(150).index)
            B0 = A[sel].copy(); B0['pc_hat'] = pc
            for k in (1, 2, 3):
                B0[f'ig{k}'] = (grp == k).astype('float32')
            for arm in ('FOCUS', 'STACK'):
                Bx = B0.copy()
                w = w_valid.copy()
                if arm == 'FOCUS':
                    w = w * focus
                else:
                    Bx['stk__a_champ'] = np.where(np.isfinite(a_ch), a_ch, pc)
                    Bx['stk__gap'] = Bx['stk__a_champ'] - pc
                dp = dict(DART); dp['random_state'] = sd
                d = lgb.LGBMClassifier(**dp)
                cmm = cm & np.isfinite(w)
                d.fit(Bx[cmm], cls[cmm], sample_weight=w[cmm])
                raw = d.predict_proba(Bx[va])
                P = np.zeros((raw.shape[0], NC)); P[:, np.asarray(d.classes_, int)] = raw
                probs[f'{arm}_{sd}'].append(P[keep])
            print(f'  [{f}] seed {sd} FOCUS+STACK {round(time.time()-t0,1)}s', flush=True)

    Rk = pd.concat(rows, ignore_index=True)
    assert Rk[KEY].equals(R[KEY])
    Ps = {k: np.vstack(v) for k, v in probs.items()}
    Bch, solo_ch, sc_ch = blend(align_prob('D', R), R, dep)
    cap = J.group_id.map(CAPS)
    out = {}
    print('\n=== complementary members, 3-seed, vs honest champion 0.634573 (sd 0.000849) ===')
    for arm in ('FOCUS', 'STACK'):
        tots, solos, cors = [], [], []
        for sd in SEEDS:
            B, solo, sc = blend(Ps[f'{arm}_{sd}'], R, dep)
            tots.append(sc['total']); solos.append(solo)
            e = pd.DataFrame({'m': (B.M if 'M' in B else B.pred), 'D': J.D, 'DEP': J.DEPAVG,
                              'y': J.actual_kwh})
            Pm = Ps[f'{arm}_{sd}']
            C = (np.arange(NC) + 0.5) * W
            mq = (Pm * C[None, :]).sum(1)
            cors.append(float(np.corrcoef((mq * cap - J.actual_kwh) / cap,
                                          (J.D - J.actual_kwh) / cap)[0, 1]))
        t = np.array(tots)
        Pav = np.mean([Ps[f'{arm}_{s}'] for s in SEEDS], axis=0)
        Ba, solo_a, sc_a = blend(Pav, R, dep)
        out[arm] = {'per_seed': t.tolist(), 'mean': float(t.mean()), 'sd': float(t.std(ddof=1)),
                    'seedavg_blend': sc_a['total'], 'mean_solo': float(np.mean(solos)),
                    'corr_with_D': float(np.mean(cors))}
        print(f'  {arm}: blend mean={t.mean():.6f} sd={t.std(ddof=1):.6f}  '
              f'seed-avg={sc_a["total"]:.6f}  solo={np.mean(solos):.6f}  '
              f'corr(err, D)={np.mean(cors):.4f}  [D vs DEPAVG is 0.9217]')
        cmp = Bch[KEY + ['actual_kwh']].copy(); cmp['champ'] = Bch.pred.to_numpy()
        cmp = cmp.merge(Ba[KEY + ['pred']].rename(columns={'pred': 'cand'}), on=KEY)
        took, arb = arbitrate(cmp, 'cand', 'champ', n_comparisons=1)
        print(f'    ARBITER vs deployed: delta={arb["point_delta"]:+.6f} sd={arb["paired_sd"]:.6f} '
              f'P={arb["p_better"]:.3f} | vs honest mean: {t.mean()-0.634573:+.6f} '
              f'({(t.mean()-0.634573)/0.000849:+.1f} seed-sd)')
        out[arm]['arb'] = arb
        np.save(N + f'S16-N1_{arm}_prob.npy', Pav)
    json.dump(out, open(N + 'S16-N1_complementary.json', 'w'), indent=1, default=str)
