"""S14-N11 (engine node F13) * transductive regime discovery with shrinkage.

The one legal use of the supplied test inputs that survived F12.  F12 killed importance
weighting: the domain classifier separates ANY two time windows perfectly (adjacent 2023 quarters
give AUC 1.0000 against a null of 0.4990), so p_test/p_train is not estimable and reweighting has
no target.  Unsupervised structure is a different object and ESL 7.10.2 licenses it explicitly:
clustering the union of train and test INPUTS uses no label and leaks nothing, and it lets the
regime definition be informed by the period we will actually be graded on.

Design, with the two lessons this session already paid for built in:
  * the clustering is fitted on train UNION the held-out fold's inputs (the local stand-in for the
    graded period), never on any label;
  * per-regime policies are NOT free.  S14-N9 measured that every per-group free fit this project
    has tried is beaten by COMPLETE shrinkage, so the per-regime action is shrunk toward the
    global action, a = (1-c)*a_regime + c*a_global, with a single shrinkage intensity c chosen
    fold-outside.  c = 1 recovers the champion exactly, so the treatment is nested and the gate
    can reject it to baseline.
"""
import sys, json
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/scratch')
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/nodes')
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/engine')
from harness import surface
from lib import CAPS, FOLDS, official_total
from loop_lib import (canonical_keys, align_prob, load_depavg, utility_frames, fo_policy,
                      KEY, TEMPS, GAMMAS)
from arbiter import arbitrate

N = '/Users/um-yunsang/BARAM2026/research/nodes/'
CORE = ['atm__hub_consensus', 'ldaps_spatial__idw__wind50max_speed',
        'gfs_spatial__idw__wind100_speed', 'ldaps_spatial__idw__wind50max_dir_sin',
        'ldaps_spatial__idw__wind50max_dir_cos', 'ldaps_spatial__idw__etc_0_blh',
        'atm__alpha_100_80', 'atm__theta850_minus_t2', 'g2__l50x__rng', 'g2__g100__mean',
        'ldaps_spatial__idw__heightAboveGround_2_t', 'cal__doy_sin', 'cal__doy_cos',
        'cal__hour_sin', 'cal__hour_cos']
CGRID = np.arange(0.0, 1.001, 0.125)

if __name__ == '__main__':
    A, FR, COLS = surface(('G2', 'DROP:grid__'))
    R = canonical_keys(); dep = load_depavg()
    P = align_prob('D', R)
    frames = utility_frames(P, R)
    Dm, _, gpicks = fo_policy(frames, R)
    J = Dm[KEY + ['actual_kwh', 'prediction_kwh']].rename(columns={'prediction_kwh': 'D'}).merge(dep, on=KEY)
    J['CHAMP'] = 0.30 * J.D + 0.70 * J.DEPAVG
    base = J[['group_id', 'actual_kwh']]
    print(f'champion: {official_total(base.assign(prediction_kwh=J.CHAMP))["total"]:.6f}')

    core = [c for c in CORE if c in A.columns]
    pos = R.reset_index().merge(J[KEY], on=KEY)['index'].to_numpy()
    Aidx = pd.MultiIndex.from_arrays([A['grp'].to_numpy(), A.index])
    key = pd.MultiIndex.from_arrays([J.group_id.to_numpy(), pd.to_datetime(J.forecast_kst_dtm)])
    X = pd.DataFrame(A[core].to_numpy(), index=Aidx).reindex(key).to_numpy()
    X = np.where(np.isfinite(X), X, np.nanmedian(X, axis=0))
    # align every candidate action onto J's row order once, instead of re-indexing in the loop
    AF = {pol: 0.30 * v[pos] + 0.70 * J.DEPAVG.to_numpy() for pol, v in frames.items()}

    out = {}
    for K in (4, 8):
        pred = np.empty(len(J)); picks = {}
        for f in FOLDS:
            sel = (J.fold_id == f).to_numpy()
            # TRANSDUCTIVE: fit the clustering on train inputs UNION the held-out fold's inputs
            mu = X.mean(0); sd = np.maximum(X.std(0), 1e-6)
            Z = (X - mu) / sd
            km = KMeans(n_clusters=K, n_init=10, random_state=20260807).fit(Z)
            lab = km.labels_
            # per-regime policy chosen on the OTHER folds only
            oth = ~sel
            a_glob = np.empty(len(J)); a_reg = np.empty(len(J))
            sc_glob = {pol: official_total(base[oth].assign(prediction_kwh=a[oth]))['total']
                       for pol, a in AF.items()}
            gbest = max(sc_glob, key=sc_glob.get)
            a_glob = AF[gbest]
            for k in range(K):
                m = lab == k
                mm = oth & m
                if mm.sum() < 200:
                    a_reg[m] = a_glob[m]; continue
                sc = {pol: official_total(base[mm].assign(prediction_kwh=a[mm]))['total']
                      for pol, a in AF.items()}
                a_reg[m] = AF[max(sc, key=sc.get)][m]
            # shrinkage intensity c on the other folds
            bc = None
            for c in CGRID:
                a = (1 - c) * a_reg + c * a_glob
                t = official_total(base[oth].assign(prediction_kwh=a[oth]))['total']
                if bc is None or t > bc[0]:
                    bc = (t, float(c))
            picks[f] = {'global_policy': str(gbest), 'c': bc[1]}
            pred[sel] = ((1 - bc[1]) * a_reg + bc[1] * a_glob)[sel]
        s = official_total(base.assign(prediction_kwh=pred))
        print(f'\nK={K} regimes: Total={s["total"]:.6f}  picks={picks}')
        cmp = J[KEY + ['actual_kwh']].copy(); cmp['champ'] = J.CHAMP; cmp['cand'] = pred
        took, arb = arbitrate(cmp, 'cand', 'champ', n_comparisons=1)
        print(f'  ARBITER delta={arb["point_delta"]:+.6f} sd={arb["paired_sd"]:.6f} '
              f'P={arb["p_better"]:.3f} -> {"CHAMPION" if took else "rejected"}')
        out[f'K{K}'] = {'total': s['total'], 'picks': picks, 'arb': arb}
    json.dump(out, open(N + 'S14-N11_transductive_regime.json', 'w'), indent=1, default=str)
