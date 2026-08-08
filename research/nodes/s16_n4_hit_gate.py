"""S16-N4 * reframe the gate: predict BAND-HIT ATTAINABILITY, not which member is best.

Why the FFORMA gate failed even though it works.  S16-N3 built the instance-conditional
soft-weight gate and it does what the literature says: overall top-1 0.5363, far above the 0.32
break-even the lane derived.  Every application rule still lost:

    soft weights, all rows          0.633259
    hard, spread>=0.08 (26.3%)      0.630868   top-1 on the trigger 0.365
    hard, spread>=0.08 & conf>=0.7  0.635768   top-1 0.538, but only 6.2% of rows fire
    hard, spread>=0.08 & conf>=0.8  0.635832   top-1 0.583, only 3.5% fire
    champion                        0.636184

The structure is unambiguous: tightening confidence raises top-1 (0.365 -> 0.583) but collapses
coverage (26% -> 3.5%), because the rows where the gate is confident are the rows where the
members already agree, and on the high-disagreement rows that actually carry the oracle the gate
is barely better than chance among four options.

So the target was wrong.  The oracle's value is not "pick the best of four" -- it is that AT LEAST
ONE member hits on 0.4985 of scored rows while the champion hits on 0.3503, a recoverable margin
of 14.8pp.  This node measures that recoverable set directly and attacks it with the matching
estimator: four BINARY band-hit models, P(member m lands within +-0.06 | x), which is the event
the metric actually pays for, rather than a four-way argmin of a continuous loss.
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
BIN = dict(objective='binary', learning_rate=0.05, num_leaves=31, min_child_samples=150,
           feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0,
           n_estimators=350, verbose=-1, n_jobs=6)
STATE = ['atm__hub_consensus', 'ldaps_spatial__idw__wind50max_speed',
         'gfs_spatial__idw__wind100_speed', 'ldaps_spatial__idw__etc_0_blh',
         'atm__alpha_100_80', 'atm__theta850_minus_t2', 'g2__l50x__rng', 'g2__l50x__std',
         'atm__gust_factor', 'cal__hour_sin', 'cal__hour_cos', 'cal__doy_sin', 'cal__doy_cos']

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
    Acf = J[MEM].to_numpy() / cap[:, None]
    ch = J.CHAMPION.to_numpy() / cap
    base = J[['group_id', 'actual_kwh']]
    scored = y >= 0.10
    hit = np.abs(Acf - y[:, None]) <= 0.06
    hit_ch = np.abs(ch - y) <= 0.06
    anyh = hit.any(1)

    print('=== the recoverable set ===')
    rec = scored & (~hit_ch) & anyh
    lost = scored & hit_ch & (~anyh)
    print(f'  scored rows                                  {int(scored.sum())}')
    print(f'  champion hits                                {hit_ch[scored].mean():.4f}')
    print(f'  at least one member hits                     {anyh[scored].mean():.4f}')
    print(f'  RECOVERABLE (champion misses, a member hits) {rec[scored].mean():.4f}  '
          f'= {int(rec.sum())} rows')
    print(f'  champion hits where NO member does           {lost[scored].mean():.4f}  '
          f'(the averaging bonus we would forfeit)')
    nh = hit.sum(1)
    print(f'  on the recoverable set, how many members hit: '
          f'{ {int(k): round(float((nh[rec]==k).mean()),3) for k in (1,2,3)} }')
    print(f'  -> a perfect gate on that set alone would raise the u=4 rate from '
          f'{hit_ch[scored].mean():.4f} to {anyh[scored].mean():.4f}')

    # ---- gate features -------------------------------------------------------------
    P = align_prob('D', R); C = (np.arange(26) + 0.5) * W
    mq_all = (P * C[None, :]).sum(1)
    sq_all = np.sqrt((P * (C[None, :] - mq_all[:, None]) ** 2).sum(1))
    _al = R[KEY].copy(); _al['mq'] = mq_all; _al['sq'] = sq_all
    _al = _al.merge(J[KEY], on=KEY, how='right')
    Xg = pd.DataFrame({f'a_{m}': Acf[:, i] for i, m in enumerate(MEM)})
    Xg['a_mean'] = Acf.mean(1); Xg['a_sd'] = Acf.std(1)
    Xg['a_rng'] = Acf.max(1) - Acf.min(1); Xg['a_champ'] = ch
    Xg['d_spread'] = _al['sq'].to_numpy(); Xg['d_mean'] = _al['mq'].to_numpy()
    for g in (1, 2, 3):
        Xg[f'g{g}'] = (J.group_id.to_numpy() == g).astype(float)
    st = [c for c in STATE if c in A.columns]
    S = pd.DataFrame(A[st].to_numpy(),
                     index=pd.MultiIndex.from_arrays([A['grp'].to_numpy(), A.index]),
                     columns=st).reindex(pd.MultiIndex.from_arrays(
                         [J.group_id, pd.to_datetime(J.forecast_kst_dtm)]))
    for c in st:
        Xg[c] = S[c].to_numpy()
    Xg = Xg.astype('float64')

    print('\n=== four BINARY band-hit models, P(member m lands within +-0.06 | x) ===')
    Pm = {s: np.zeros((len(J), len(MEM))) for s in SEEDS}
    for f in FOLDS:
        sel = (J.fold_id == f).to_numpy()
        trn = (~sel) & scored
        for sd in SEEDS:
            t0 = time.time()
            for i, m in enumerate(MEM):
                p = dict(BIN); p['random_state'] = sd
                clf = lgb.LGBMClassifier(**p)
                clf.fit(Xg[trn], hit[trn, i].astype(int))
                Pm[sd][sel, i] = clf.predict_proba(Xg[sel])[:, 1]
            print(f'  [{f}] seed {sd} four binaries {round(time.time()-t0,1)}s', flush=True)
    Pav = np.mean([Pm[s] for s in SEEDS], axis=0)
    k = np.argmax(Pav, 1)
    print(f'\n  argmax-P(hit) picks a member that actually hits on '
          f'{hit[np.arange(len(J)), k][scored].mean():.4f} of scored rows '
          f'(champion {hit_ch[scored].mean():.4f}, oracle {anyh[scored].mean():.4f})')
    print(f'  on the RECOVERABLE set it picks a hitting member on '
          f'{hit[np.arange(len(J)), k][rec].mean():.4f}')

    print('\n=== application rules ===')
    res = {}
    sel_a = Acf[np.arange(len(J)), k]
    conf = Pav.max(1)
    for thr in (0.0, 0.35, 0.45, 0.55, 0.65, 0.75):
        trig = conf >= thr
        mix = np.where(trig, sel_a, ch)
        s_ = official_total(base.assign(prediction_kwh=mix * cap))
        res[f'P(hit)>={thr}'] = s_['total']
        print(f'  switch when max P(hit) >= {thr:.2f}: {trig.mean():6.1%} fire -> '
              f'Total={s_["total"]:.6f}  1-NMAE={s_["one_minus_nmae"]:.6f} FICR={s_["ficr"]:.6f}')
    # switch only when the gate thinks the champion itself will miss
    p_ch = None
    for thr in (0.05, 0.10, 0.15):
        trig = (conf - Pav[np.arange(len(J)), 0]) >= thr
        mix = np.where(trig, sel_a, ch)
        s_ = official_total(base.assign(prediction_kwh=mix * cap))
        res[f'margin>={thr}'] = s_['total']
        print(f'  switch when P(best)-P(D) >= {thr:.2f}: {trig.mean():6.1%} fire -> '
              f'Total={s_["total"]:.6f}')
    b = max(res, key=res.get)
    print(f'\n  BEST: {b} = {res[b]:.6f}   champion 0.636184 (honest 0.634573)')
    best_trig = conf >= float(b.split('>=')[1]) if 'P(hit)' in b else \
        (conf - Pav[np.arange(len(J)), 0]) >= float(b.split('>=')[1])
    cmp = J[KEY + ['actual_kwh']].copy(); cmp['champ'] = J.CHAMPION
    cmp['cand'] = np.where(best_trig, sel_a, ch) * cap
    took, arb = arbitrate(cmp, 'cand', 'champ', n_comparisons=1)
    print(f'  ARBITER: delta={arb["point_delta"]:+.6f} sd={arb["paired_sd"]:.6f} '
          f'P={arb["p_better"]:.3f} -> {"CHAMPION" if took else "rejected"}')
    json.dump({'recoverable_share': float(rec[scored].mean()),
               'champion_hit': float(hit_ch[scored].mean()),
               'any_hit': float(anyh[scored].mean()),
               'gate_hit_rate': float(hit[np.arange(len(J)), k][scored].mean()),
               'gate_hit_on_recoverable': float(hit[np.arange(len(J)), k][rec].mean()),
               'rules': res, 'arb': arb},
              open(N + 'S16-N4_hit_gate.json', 'w'), indent=1, default=str)
