"""S12-N6 * analog/k-NN member carrying the SAME metric-aligned decision layer as D.

Why this member and not another GBDT.  S12-N4 measured the deployed pool's minimum pairwise
error correlation at 0.934 and S9-N14 found no untapped combination among 15 GBDT members;
AGENTS.md records 0.984-0.994 correlation for every classifier-family member against M115.
S12-N5 then measured the one analog stem with all three folds saved (M244) at correlation
0.857 with D and 0.840 with DEPAVG -- by far the most decorrelated family available -- but its
solo score is only 0.605760 because its 1-NMAE (0.860933) is competitive while its FICR
(0.350587) is not.  That asymmetry says the analog *representation* is fine and its
*decision layer* is what is losing; M244 emits a point action, not a predictive distribution,
so it never went through the settlement-optimal argmax that every classifier member uses.

Treatment: rebuild the analog from scratch as a CONDITIONAL DISTRIBUTION.  For each target
row, retrieve its k nearest training analogues in a standardised physical-state space, form
the empirical 26-class histogram of their realised capacity factors (distance-kernel
weighted), and hand that histogram to the identical (T,G) decision layer and identical
fold-outside policy gate used by D.  The result is directly poolable with D and DEPAVG on
the same key set.

Leakage control: analogues are drawn only from rows strictly before the fold's start
(same expanding-window rule as every other member), and additionally any analogue within
+-72 h of the target hour is dropped so that autocorrelation cannot masquerade as state
similarity (same guard as research/nodes/s11_floor.py).
"""
import sys, json, time
import numpy as np, pandas as pd
from sklearn.neighbors import NearestNeighbors
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/scratch')
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/nodes')
from harness import surface, MU
from lib import FOLDS, CAPS
import lightgbm as lgb

N = '/Users/um-yunsang/BARAM2026/research/nodes/'
W = 0.04; NC = 26

STATE = ['atm__hub_consensus', 'atm__hub_from_ldaps50', 'atm__hub_from_gfs100',
         'ldaps_spatial__idw__wind50max_speed', 'gfs_spatial__idw__wind100_speed',
         'ldaps_spatial__idw__wind10_speed', 'gfs_spatial__idw__wind10_speed',
         'ldaps_spatial__idw__wind50max_dir_sin', 'ldaps_spatial__idw__wind50max_dir_cos',
         'gfs_spatial__idw__wind100_dir_sin', 'gfs_spatial__idw__wind100_dir_cos',
         'ldaps_spatial__idw__etc_0_blh', 'atm__alpha_100_80', 'atm__alpha_50_10',
         'atm__theta850_minus_t2', 'atm__gust_factor', 'atm__w50_envelope',
         'g2__l50x__rng', 'g2__l50x__std', 'g2__g100__mean',
         'ldaps_spatial__idw__heightAboveGround_2_t', 'phys_v2__air_density',
         'cal__doy_sin', 'cal__doy_cos', 'cal__hour_sin', 'cal__hour_cos']


def build(tag, K=120, tau=1.0, use_pchat=True, seed=20260808):
    A, FR, COLS = surface(('G2', 'DROP:grid__'))
    cf = A['cf'].to_numpy(); grp = A['grp'].to_numpy(); idx = A.index
    valid = np.isfinite(cf) & (cf >= 0.1)
    w_prod = np.where(valid, np.clip(cf, 0, 1.2), 0.05)
    gapv = A['pc_true'].to_numpy() - cf
    hrs = idx.values.astype('datetime64[h]').astype(np.int64)
    state = [c for c in STATE if c in A.columns]
    print(f'state dims: {len(state)}', flush=True)

    rows = []; probs = []
    for f, (a, b) in FOLDS.items():
        a = pd.Timestamp(a); b = pd.Timestamp(b)
        tr = np.asarray(idx < a); va = np.asarray((idx >= a) & (idx <= b)); keep = np.isfinite(cf[va])
        t0 = time.time()
        S = A[state].to_numpy('float64')
        S = np.where(np.isfinite(S), S, np.nan)
        med = np.nanmedian(S[tr], axis=0)
        S = np.where(np.isfinite(S), S, med)
        mu_, sd_ = S[tr].mean(0), np.maximum(S[tr].std(0), 1e-6)
        Z = (S - mu_) / sd_

        if use_pchat:
            m = tr & np.isfinite(A['pc_true'].to_numpy())
            reg = lgb.LGBMRegressor(**MU)
            reg.fit(A.loc[m, COLS], A.loc[m, 'pc_true'], sample_weight=w_prod[m])
            pc = np.clip(reg.predict(A[COLS]), 0, 1)
            # the teacher's own estimate is the single most informative state axis: give it
            # a weight equal to the whole rest of the state vector so retrieval is anchored
            # on predicted power, then refined by the raw atmospheric state.
            Zp = ((pc - pc[tr].mean()) / max(pc[tr].std(), 1e-6))[:, None] * np.sqrt(len(state))
            Z = np.hstack([Z, Zp])

        P_all = np.zeros((int(va.sum()), NC))
        for g in (1, 2, 3):
            lib = tr & (grp == g) & np.isfinite(cf) & (~(gapv >= 0.05))
            qsel = va & (grp == g)
            if lib.sum() == 0 or qsel.sum() == 0:
                continue
            Zl = Z[lib]; yl = cf[lib]; hl = hrs[lib]
            nn = NearestNeighbors(n_neighbors=min(K + 40, lib.sum())).fit(Zl)
            dist, ind = nn.kneighbors(Z[qsel])
            hq = hrs[qsel]
            cls_l = np.clip(np.nan_to_num(yl, nan=0.0) / W, 0, NC - 1).astype(int)
            Pg = np.zeros((qsel.sum(), NC))
            for i in range(qsel.sum()):
                ok = np.abs(hl[ind[i]] - hq[i]) > 72
                sel = ind[i][ok][:K]; dd = dist[i][ok][:K]
                if len(sel) == 0:
                    Pg[i] = 1.0 / NC; continue
                h = max(np.median(dd), 1e-6)
                wgt = np.exp(-0.5 * (dd / (tau * h)) ** 2)
                np.add.at(Pg[i], cls_l[sel], wgt)
                Pg[i] /= max(Pg[i].sum(), 1e-12)
            P_all[qsel[va]] = Pg
        rows.append(pd.DataFrame({'fold_id': f, 'group_id': grp[va][keep],
                                  'forecast_kst_dtm': idx[va][keep], 'cf': cf[va][keep],
                                  'mean_gen_g': [float(np.nanmean(cf[tr & (grp == g)])) for g in grp[va][keep]]}))
        probs.append(P_all[keep])
        print(f'  [{tag}] {f} {round(time.time()-t0,1)}s', flush=True)
    R = pd.concat(rows, ignore_index=True)
    Pf = np.vstack(probs)
    R.to_parquet(N + f'S7-N8_{tag}_keys.parquet', index=False)
    np.save(N + f'S7-N8_{tag}_prob.npy', Pf)
    return R, Pf


if __name__ == '__main__':
    for tag, kw in [('KNN120', dict(K=120, tau=1.0)),
                    ('KNN300', dict(K=300, tau=1.0)),
                    ('KNN60', dict(K=60, tau=0.75))]:
        build(tag, **kw)
    print('DONE', flush=True)
