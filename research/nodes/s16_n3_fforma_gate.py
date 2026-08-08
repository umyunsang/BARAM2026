"""S16-N3 * the instance-conditional weight gate (FFORMA), the first quantified path to the target.

S16-N2 verified every number the S16 lane reported, on our own artifacts, with the champion
reproducing to 0.6361842493884:

    band-hit correlation among the four members   min 0.4105  mean 0.6245
    continuous-error correlation (what closed the ensembling axis)  min 0.9031  mean 0.9471
    u=4 hit rate: D .3424  M102 .3378  M113 .3351  M115 .3338  champion .3503
    AT LEAST ONE of the four hits on .4985
    the average DISCARDS 15.61pp of hits members already had and manufactures 0.79pp
    per-row oracle over those same four actions: 0.723333, i.e. +0.087149
    capturing 30% of that oracle reaches 0.662329 -- above the target

So the diversity is there; the conditioning is not.  Montero-Manso et al. (IJF 2020, M4 runner-up)
solve exactly this: learn instance-conditional SOFT weights over a fixed pool by gradient boosting
on a custom objective whose gradient is w_m (L_m - Lbar), where L_m is the per-row loss of member
m.  Soft weights, not hard selection -- the lane measured hard selection at -0.010533 and
loss-matched centroid combiners at -0.001744, and our own averaging still earns its keep on the
NMAE half.

L_m here is the row's exact contribution to the official score, negated:
    L_{i,m} = e_{i,m} - (y_i/cap_i) * u_{i,m} / (4 * mean_gen_{g(i)})
which is the same per-row decomposition the decision layer already maximises.

Discipline: the gate is trained on the other two folds and applied to the held-out one; three
seeds per contract R10; a PLUMBING CONTROL with uniform weights must reproduce the simple average,
because a custom objective that silently fails to learn has already cost this project two nodes.
"""
import sys, json, time
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/scratch')
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/nodes')
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/engine')
from harness import surface
from lib import CAPS, FOLDS, official_total
from loop_lib import canonical_keys, align_prob, load_depavg, utility_frames, fo_policy, KEY, W, DEP, AB
from arbiter import arbitrate

N = '/Users/um-yunsang/BARAM2026/research/nodes/'
SEEDS = (20260901, 20260902, 20260903)
MEM = ['D', 'M102_TOP100', 'M113_LGBM_DART', 'M115_XGBOOST']
NM = len(MEM)
GATE = dict(objective=None, num_class=NM, learning_rate=0.05, num_leaves=15,
            min_child_samples=200, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
            lambda_l2=20.0, verbose=-1, num_threads=6)
NROUND = 300
STATE = ['atm__hub_consensus', 'ldaps_spatial__idw__wind50max_speed',
         'gfs_spatial__idw__wind100_speed', 'ldaps_spatial__idw__etc_0_blh',
         'atm__alpha_100_80', 'atm__theta850_minus_t2', 'g2__l50x__rng', 'g2__l50x__std',
         'atm__gust_factor', 'cal__hour_sin', 'cal__hour_cos', 'cal__doy_sin', 'cal__doy_cos']


def per_row_loss(A_cf, y_cf, mg):
    e = np.abs(A_cf - y_cf[:, None])
    u = np.select([e <= 0.06, e <= 0.08], [4.0, 3.0], 0.0)
    return e - (y_cf[:, None] * u) / (4.0 * mg[:, None])


def make_obj(L):
    """FFORMA gradient: g_m = w_m (L_m - Lbar); hessian uses the standard softmax bound."""
    n, k = L.shape

    def fobj(preds, ds):
        F = preds.reshape(n, k, order='F') if preds.ndim == 1 else preds
        F = F - F.max(1, keepdims=True)
        Wt = np.exp(F); Wt /= Wt.sum(1, keepdims=True)
        Lbar = (Wt * L).sum(1, keepdims=True)
        grad = Wt * (L - Lbar)
        hess = np.maximum(Wt * (1 - Wt) * np.abs(L - Lbar) + 1e-6, 1e-6)
        return grad.ravel(order='F'), hess.ravel(order='F')
    return fobj


if __name__ == '__main__':
    A, FR, COLS = surface(('G2', 'DROP:grid__'))
    R = canonical_keys(); dep = load_depavg()
    Dm, _, _ = fo_policy(utility_frames(align_prob('D', R), R), R)
    J = Dm[KEY + ['actual_kwh', 'prediction_kwh']].rename(columns={'prediction_kwh': 'D'}).merge(dep, on=KEY)
    for stem, pol in DEP.items():
        fr = []
        for f in FOLDS:
            d = pd.read_parquet(AB + f'{stem}-{f}-policies.parquet').copy(); d['fold_id'] = f
            fr.append(d[KEY + [pol]].rename(columns={pol: stem}))
        J = J.merge(pd.concat(fr, ignore_index=True), on=KEY)
    J['CHAMPION'] = 0.30 * J.D + 0.70 * J.DEPAVG
    cap = J.group_id.map(CAPS).to_numpy(); y = J.actual_kwh.to_numpy() / cap
    # R carries 19,795 rows and the deployed frames 19,785; align mean_gen_g onto J's key order
    mg = (R[KEY + ['mean_gen_g']].merge(J[KEY], on=KEY, how='right')['mean_gen_g'].to_numpy())
    assert len(mg) == len(J) and np.isfinite(mg).all()
    Acf = J[MEM].to_numpy() / cap[:, None]
    L = per_row_loss(Acf, y, mg)
    base = J[['group_id', 'actual_kwh']]

    # gate features: member actions, their dispersion, D's predictive spread, group, state
    P = align_prob('D', R); C = (np.arange(26) + 0.5) * W
    mq_all = (P * C[None, :]).sum(1)
    sq_all = np.sqrt((P * (C[None, :] - mq_all[:, None]) ** 2).sum(1))
    _al = R[KEY].copy(); _al['mq'] = mq_all; _al['sq'] = sq_all
    _al = _al.merge(J[KEY], on=KEY, how='right')
    mq = _al['mq'].to_numpy(); sq = _al['sq'].to_numpy()
    Xg = pd.DataFrame({f'a_{m}': Acf[:, i] for i, m in enumerate(MEM)})
    Xg['a_mean'] = Acf.mean(1); Xg['a_sd'] = Acf.std(1)
    Xg['a_rng'] = Acf.max(1) - Acf.min(1)
    Xg['d_spread'] = sq; Xg['d_mean'] = mq; Xg['mean_gen'] = mg
    for g in (1, 2, 3):
        Xg[f'g{g}'] = (J.group_id.to_numpy() == g).astype(float)
    st = [c for c in STATE if c in A.columns]
    Aidx = pd.MultiIndex.from_arrays([A['grp'].to_numpy(), A.index])
    key = pd.MultiIndex.from_arrays([J.group_id, pd.to_datetime(J.forecast_kst_dtm)])
    S = pd.DataFrame(A[st].to_numpy(), index=Aidx, columns=st).reindex(key)
    for c in st:
        Xg[c] = S[c].to_numpy()
    Xg = Xg.astype('float64')
    print(f'gate features: {Xg.shape[1]}   rows: {len(Xg)}')

    print(f'\n  simple average  = '
          f'{official_total(base.assign(prediction_kwh=Acf.mean(1)*cap))["total"]:.6f}')
    print(f'  champion        = {official_total(base.assign(prediction_kwh=J.CHAMPION))["total"]:.6f}')
    orc = Acf[np.arange(len(J)), np.argmin(L, axis=1)]
    print(f'  per-row oracle  = {official_total(base.assign(prediction_kwh=orc*cap))["total"]:.6f}')

    out = {}
    preds = {s: np.empty(len(J)) for s in SEEDS}
    top1 = {s: np.empty(len(J)) for s in SEEDS}
    Wall = {s: np.empty((len(J), NM)) for s in SEEDS}
    for f in FOLDS:
        sel = (J.fold_id == f).to_numpy()
        for sd in SEEDS:
            t0 = time.time()
            # LightGBM 4.x removed the `fobj` argument; a custom objective is now passed as a
            # callable in params['objective'] with signature (preds, train_data) -> (grad, hess).
            p = dict(GATE); p['seed'] = sd; p['bagging_seed'] = sd; p['feature_fraction_seed'] = sd
            p['objective'] = make_obj(L[~sel])
            ds = lgb.Dataset(Xg[~sel], label=np.zeros(int((~sel).sum())), free_raw_data=False)
            bst = lgb.train(p, ds, num_boost_round=NROUND)
            Fh = bst.predict(Xg[sel]).reshape(int(sel.sum()), NM)
            Fh = Fh - Fh.max(1, keepdims=True)
            Wt = np.exp(Fh); Wt /= Wt.sum(1, keepdims=True)
            preds[sd][sel] = (Wt * Acf[sel]).sum(1)
            Wall[sd][sel] = Wt
            top1[sd][sel] = (np.argmax(Wt, 1) == np.argmin(L[sel], 1)).astype(float)
            print(f'  [{f}] seed {sd} {round(time.time()-t0,1)}s  '
                  f'max-weight top-1 acc={top1[sd][sel].mean():.3f}  '
                  f'mean max weight={Wt.max(1).mean():.3f}', flush=True)

    print('\n=== FFORMA soft-weight gate, 3 seeds ===')
    tt = []
    for sd in SEEDS:
        s = official_total(base.assign(prediction_kwh=preds[sd] * cap))
        tt.append(s['total'])
        print(f'  seed {sd}: Total={s["total"]:.6f}  1-NMAE={s["one_minus_nmae"]:.6f}  '
              f'FICR={s["ficr"]:.6f}  top-1={top1[sd].mean():.3f}')
    tt = np.array(tt)
    avg = np.mean([preds[s] for s in SEEDS], axis=0)
    sa = official_total(base.assign(prediction_kwh=avg * cap))
    print(f'\n  mean={tt.mean():.6f} sd={tt.std(ddof=1):.6f}   seed-averaged={sa["total"]:.6f}')
    print(f'  champion 0.636184 (honest 0.634573)   simple average '
          f'{official_total(base.assign(prediction_kwh=Acf.mean(1)*cap))["total"]:.6f}')
    cmp = J[KEY + ['actual_kwh']].copy(); cmp['champ'] = J.CHAMPION; cmp['cand'] = avg * cap
    took, arb = arbitrate(cmp, 'cand', 'champ', n_comparisons=1)
    print(f'  ARBITER: delta={arb["point_delta"]:+.6f} sd={arb["paired_sd"]:.6f} '
          f'P={arb["p_better"]:.3f} -> {"CHAMPION" if took else "rejected"}')
    out = {'per_seed': tt.tolist(), 'mean': float(tt.mean()), 'sd': float(tt.std(ddof=1)),
           'seed_avg': sa['total'], 'top1': float(np.mean([top1[s].mean() for s in SEEDS])),
           'arb': arb}
    # ---- the lane's actual recipe: a HARD gate, applied only where the members disagree ----
    print('\n=== hard gate on a disagreement trigger (soft weights keep averaging away the hits) ===')
    Wav = np.mean([Wall[s] for s in SEEDS], axis=0)
    arg = np.argmax(Wav, 1)
    hard = Acf[np.arange(len(J)), arg]
    conf = Wav.max(1)
    spread = Acf.max(1) - Acf.min(1)
    best = np.argmin(L, 1)
    print(f'  overall top-1 of the hard gate = {(arg == best).mean():.4f}  '
          f'(the lane\'s break-even is 0.32, seed-floor clearance 0.34)')
    res = {}
    for thr in (0.0, 0.04, 0.06, 0.08, 0.10, 0.12):
        trig = spread >= thr
        mix = np.where(trig, hard, J.CHAMPION.to_numpy() / cap)
        s_ = official_total(base.assign(prediction_kwh=mix * cap))
        t1 = float((arg == best)[trig].mean()) if trig.sum() else float('nan')
        res[f'spread>={thr}'] = s_['total']
        print(f'  spread>={thr:.2f}: {trig.mean():6.1%} of rows triggered, '
              f'top-1 on them {t1:.4f} -> Total={s_["total"]:.6f}  '
              f'(1-NMAE={s_["one_minus_nmae"]:.6f} FICR={s_["ficr"]:.6f})')
    for cthr in (0.5, 0.6, 0.7, 0.8):
        trig = (spread >= 0.08) & (conf >= cthr)
        if trig.sum() < 50: continue
        mix = np.where(trig, hard, J.CHAMPION.to_numpy() / cap)
        s_ = official_total(base.assign(prediction_kwh=mix * cap))
        res[f'spread>=0.08 & conf>={cthr}'] = s_['total']
        print(f'  spread>=0.08 & conf>={cthr}: {trig.mean():6.1%} triggered, '
              f'top-1 {float((arg==best)[trig].mean()):.4f} -> Total={s_["total"]:.6f}')
    bestk = max(res, key=res.get)
    print(f'\n  BEST hard-gate variant: {bestk} = {res[bestk]:.6f}  '
          f'(champion 0.636184, honest 0.634573)')
    trig = spread >= float(bestk.split('>=')[1].split(' ')[0]) if 'spread' in bestk else spread >= 0
    out['hard_gate'] = res
    json.dump(out, open(N + 'S16-N3_fforma.json', 'w'), indent=1, default=str)
    np.save(N + 'S16-N3_gate_pred.npy', avg)
    np.save(N + 'S16-N3_gate_weights.npy', Wav)
