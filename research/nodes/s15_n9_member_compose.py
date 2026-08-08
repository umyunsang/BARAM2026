"""S15-N9 * port the surviving stages into member D and run the full champion pipeline.

Reverse ablation (S15-N8, every arm a 3-seed average) attributed the composed gain:
    B2 supervised hub-wind feature      +0.004214   KEEP
    D1 extra_trees/path_smooth/ffbynode +0.002685   KEEP
    A3 SCADA-only power curve           +0.001912   KEEP
    B1 per-source spatial reduction     +0.000145   KEEP (free: a replacement, not a widening)
    D4 metric-matched group weights     -0.000563   PRUNE
The pruned pipeline is the no_D4 arm at 0.605922 against BASE 0.602299, i.e. +0.003623 on the
POINT pipeline.  The contributions sum to +0.008392 against a composed effect of +0.003060, so the
stages are strongly sub-additive -- B2 and A3 both inject SCADA information and overlap.

The question this node settles: does that gain survive the decision layer and the DEPAVG blend?
Every previous session has watched point-accuracy gains evaporate at exactly this boundary
(S13-N5's DW arms, S12-N15's DV, S14-N12's self-training), because the decision layer feeds on
distribution SHAPE and most accuracy treatments flatten it.  So member D is rebuilt with the four
surviving stages, under three seeds, and pushed through the identical fold-outside (T,G) gate and
1-dof blend.

The comparator is the HONEST champion baseline, not the deployed number: S15-N6 measured the
champion's own configuration family at mean 0.634573 (sd 0.000849) with the deployed 0.636184
sitting +1.90 sd above it and none of six siblings exceeding it.
"""
import sys, json, time
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/scratch')
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/nodes')
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/engine')
from harness import surface, MU
from lib import FOLDS, CAPS, official_total
from loop_lib import canonical_keys, load_depavg, align_prob, utility_frames, fo_policy, KEY, W
from arbiter import arbitrate
from s15_n7_compose import build_pc_scada, D1_EXTRA

S = '/Users/um-yunsang/BARAM2026/research/scratch/'
N = '/Users/um-yunsang/BARAM2026/research/nodes/'
NC = 26
SEEDS = (20260803, 20260804, 20260805)
DART = dict(objective='multiclass', boosting_type='dart', n_estimators=400, learning_rate=0.08,
            num_leaves=31, min_child_samples=60, subsample=0.85, subsample_freq=1,
            colsample_bytree=0.4, reg_lambda=3.0, n_jobs=6, verbose=-1)


def pipeline(P, R, dep):
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
        rows.append(held.assign(pred=best[1] * held.M + (1 - best[1]) * held.DEPAVG))
    B = pd.concat(rows, ignore_index=True)
    return B, solo['total'], official_total(B.assign(prediction_kwh=B.pred)[
        ['group_id', 'actual_kwh', 'prediction_kwh']])


if __name__ == '__main__':
    A, FR, COLS = surface(('G2', 'DROP:grid__'))
    X = pd.read_parquet(S + 'b1_reduction.parquet')
    T = pd.read_parquet(S + 'teacher_targets.parquet')
    grp = A['grp'].to_numpy(); idx = A.index
    cf = A['cf'].to_numpy(); pct = A['pc_true'].to_numpy()
    valid = np.isfinite(cf) & (cf >= 0.1)
    w_prod = np.where(valid, np.clip(cf, 0, 1.2), 0.05)
    w_valid = np.where(valid, 1.0, 0.15)
    gapv = pct - cf
    A2 = A.copy()
    for c in [c for c in X.columns if c.endswith('_speed') and c in A.columns]:
        A2[c] = X[c].reindex(A.index).to_numpy()
    pcs = build_pc_scada()
    pc_scada = np.full(len(A), np.nan)
    for g in (1, 2, 3):
        m = grp == g
        pc_scada[m] = pcs[g].reindex(idx[m]).to_numpy()
    tgt = np.where(np.isfinite(pc_scada), pc_scada, pct)
    hub_obs = np.full(len(A), np.nan)
    for g in (1, 2, 3):
        m = grp == g
        hub_obs[m] = T[f'g{g}_v_mean'].reindex(idx[m]).to_numpy()

    rows = []; probs = {s: [] for s in SEEDS}
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
            mp = dict(MU); mp['random_state'] = sd; mp.update(D1_EXTRA)
            m = tr & np.isfinite(tgt)
            mu = lgb.LGBMRegressor(**mp)
            mu.fit(A2.loc[m, COLS], tgt[m], sample_weight=w_prod[m])
            pc = np.clip(mu.predict(A2[COLS]), 0, 1.2)
            hm = tr & np.isfinite(hub_obs)
            hw = lgb.LGBMRegressor(**mp)
            hw.fit(A2.loc[hm, COLS], hub_obs[hm], sample_weight=w_prod[hm])
            hub = np.clip(hw.predict(A2[COLS]), 0, 40)
            sel = list(pd.Series(mu.feature_importances_, index=COLS).sort_values(ascending=False).head(150).index)
            B = A2[sel].copy(); B['pc_hat'] = pc; B['hub__ws_pred'] = hub
            for k in (1, 2, 3):
                B[f'ig{k}'] = (grp == k).astype('float32')
            dp = dict(DART); dp['random_state'] = sd; dp.update(D1_EXTRA)
            d = lgb.LGBMClassifier(**dp)
            d.fit(B[cm], cls[cm], sample_weight=w_valid[cm])
            raw = d.predict_proba(B[va])
            P = np.zeros((raw.shape[0], NC)); P[:, np.asarray(d.classes_, int)] = raw
            probs[sd].append(P[keep])
            print(f'  [{f}] seed {sd} {round(time.time()-t0,1)}s', flush=True)

    Rk = pd.concat(rows, ignore_index=True)
    R = canonical_keys(); dep = load_depavg()
    assert Rk[KEY].equals(R[KEY])
    Ps = {s: np.vstack(v) for s, v in probs.items()}
    for s, P in Ps.items():
        np.save(N + f'S15-N9_prob_{s}.npy', P)

    print('\n=== composed member through the full champion pipeline (3 seeds) ===')
    tot = []
    for s in SEEDS:
        Bc, solo, sc = pipeline(Ps[s], R, dep)
        tot.append(sc['total'])
        print(f'  seed {s}: solo={solo:.6f}  blend={sc["total"]:.6f} '
              f'(1-NMAE={sc["one_minus_nmae"]:.6f} FICR={sc["ficr"]:.6f})')
    tot = np.array(tot)
    Bavg, solo_a, sc_a = pipeline(np.mean(list(Ps.values()), axis=0), R, dep)
    Bch, solo_c, sc_c = pipeline(align_prob('D', R), R, dep)
    print(f'\n  composed  mean={tot.mean():.6f}  sd={tot.std(ddof=1):.6f}  '
          f'seed-averaged-prob blend={sc_a["total"]:.6f}')
    print(f'  champion  deployed={sc_c["total"]:.6f}   HONEST seed-mean (S15-N6)=0.634573 sd=0.000849')
    print(f'\n  composed mean - honest champion mean = {tot.mean()-0.634573:+.6f}')
    print(f'  composed mean - deployed champion     = {tot.mean()-sc_c["total"]:+.6f}')
    cmp = Bch[KEY + ['actual_kwh']].copy(); cmp['champ'] = Bch.pred.to_numpy()
    cmp = cmp.merge(Bavg[KEY + ['pred']].rename(columns={'pred': 'cand'}), on=KEY)
    took, arb = arbitrate(cmp, 'cand', 'champ', n_comparisons=1)
    print(f'  ARBITER (seed-avg composed vs DEPLOYED champion): delta={arb["point_delta"]:+.6f} '
          f'sd={arb["paired_sd"]:.6f} P={arb["p_better"]:.3f} -> {"CHAMPION" if took else "rejected"}')
    json.dump({'per_seed': tot.tolist(), 'mean': float(tot.mean()), 'sd': float(tot.std(ddof=1)),
               'seedavg_blend': sc_a['total'], 'deployed_champion': sc_c['total'],
               'honest_champion': 0.634573, 'arb': arb},
              open(N + 'S15-N9_member_compose.json', 'w'), indent=1, default=str)
