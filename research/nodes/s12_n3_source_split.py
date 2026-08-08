"""S12-N3 * source-separated members (HEFTCom2024 winner's stacking principle).

Motivation (research/lanes/S6_ext_A_competitions.md sec 0.2): the HEFTCom2024 winning
solution trains a separate model PER NWP SOURCE and stacks them, rather than concatenating
the sources into one design matrix.  Every one of this project's 15 saved members
concatenates LDAPS + GFS into a single matrix, which is the direct mechanical reason their
pairwise error correlation is 0.98-0.99 and why the S9-N14 full-pool blend search found no
untapped combination.  Two NWP sources with an inter-source error correlation of ~0.78
should, if kept apart until the ensemble step, produce members far more decorrelated than
0.98.

Treatment: the architecture, teacher target, weights, gating, discretisation, classifier
hyper-parameters and decision layer are byte-identical to research/nodes/s7_more.py (the
verified generator of `D`).  The ONLY change is the column set fed to both the teacher and
the classifier.

  DL  LDAPS-only : ldaps__ , ldaps_spatial__ , geom__ldaps__ , g2__l* , LDAPS-derived atm__
  DG  GFS-only   : gfs__   , gfs_spatial__   , geom__gfs__   , g2__g* , GFS-derived atm__ ,
                   phys__/phys_v2__ (physics.py:51 shows phys__hub117_speed = GFS 100 m wind
                   extrapolated, so the whole phys block is GFS-derived)
  both also get the source-agnostic calendar/lead/group block.
  Genuinely cross-source columns (source_disagreement__*, atm__hub_consensus,
  atm__hub_disagree, atm__dewpoint_depression, atm__rh_deficit) are excluded from BOTH so
  neither member can see the other source even indirectly.
"""
import sys, json, time
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/scratch')
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/nodes')
from harness import surface, MU
from lib import FOLDS, CAPS, official_total

N = '/Users/um-yunsang/BARAM2026/research/nodes/'
W = 0.04; NC = 26
DART_CLF = dict(objective='multiclass', boosting_type='dart', n_estimators=400,
                learning_rate=0.08, num_leaves=31, min_child_samples=60, subsample=0.85,
                subsample_freq=1, colsample_bytree=0.4, reg_lambda=3.0,
                random_state=20260803, n_jobs=6, verbose=-1)

ATM_LDAPS = {'atm__w50_envelope', 'atm__w50_midpoint', 'atm__w50_asymmetry',
             'atm__w50max_w10_ratio', 'atm__w50min_w10_ratio', 'atm__w10_w5_ratio',
             'atm__alpha_50_10', 'atm__blh_norm', 'atm__blh_below_hub',
             'atm__hub_from_ldaps50', 'atm__ldaps__speed_x_dsin', 'atm__ldaps__speed_x_dcos'}
ATM_GFS = {'atm__theta850_minus_t2', 'atm__theta700_minus_theta850',
           'atm__theta500_minus_theta700', 'atm__gust_excess', 'atm__gust_factor',
           'atm__w100_w10_ratio', 'atm__w80_w10_ratio', 'atm__pbl_w10_ratio',
           'atm__alpha_80_10', 'atm__alpha_100_80', 'atm__bulk_richardson_proxy',
           'atm__vrate_per_wind', 'atm__lapse_2m_850', 'atm__hub_from_gfs100',
           'atm__gfs__speed_x_dsin', 'atm__gfs__speed_x_dcos'}
COMMON_EXACT = {'hour', 'month', 'day_of_year', 'lead_hour', 'operating_year', 'group_id',
                'is_g1', 'is_g2', 'is_g3'}


def split_columns(COLS):
    common = [c for c in COLS if c.startswith('cal__') or c in COMMON_EXACT]
    ld = [c for c in COLS if c.startswith(('ldaps__', 'ldaps_spatial__', 'geom__ldaps__'))
          or c.startswith('g2__l') or c in ATM_LDAPS]
    gf = [c for c in COLS if c.startswith(('gfs__', 'gfs_spatial__', 'geom__gfs__',
                                           'phys__', 'phys_v2__'))
          or c.startswith('g2__g') or c in ATM_GFS]
    excluded = [c for c in COLS if c not in set(common) | set(ld) | set(gf)]
    return common + ld, common + gf, excluded


def build(tag, cols, A, COLS_ALL):
    cf = A['cf'].to_numpy(); grp = A['grp'].to_numpy(); idx = A.index
    valid = np.isfinite(cf) & (cf >= 0.1)
    w_prod = np.where(valid, np.clip(cf, 0, 1.2), 0.05)
    w_valid = np.where(valid, 1.0, 0.15)
    gapv = A['pc_true'].to_numpy() - cf
    rows = []; probs = []
    for f, (a, b) in FOLDS.items():
        a = pd.Timestamp(a); b = pd.Timestamp(b)
        tr = np.asarray(idx < a); va = np.asarray((idx >= a) & (idx <= b)); keep = np.isfinite(cf[va])
        t0 = time.time()
        m = tr & np.isfinite(A['pc_true'].to_numpy())
        mu = lgb.LGBMRegressor(**MU)
        mu.fit(A.loc[m, cols], A.loc[m, 'pc_true'], sample_weight=w_prod[m])
        pc = np.clip(mu.predict(A[cols]), 0, 1)
        sel = list(pd.Series(mu.feature_importances_, index=cols).sort_values(ascending=False).head(150).index)
        B = A[sel].copy(); B['pc_hat'] = pc
        for k in (1, 2, 3):
            B[f'ig{k}'] = (grp == k).astype('float32')
        cls = np.clip(np.nan_to_num(cf, nan=0.0) / W, 0, NC - 1).astype(int)
        cm = tr & np.isfinite(cf) & (~(gapv >= 0.05))
        d = lgb.LGBMClassifier(**DART_CLF)
        d.fit(B[cm], cls[cm], sample_weight=w_valid[cm])
        raw = d.predict_proba(B[va])
        P = np.zeros((raw.shape[0], NC)); P[:, np.asarray(d.classes_, int)] = raw
        rows.append(pd.DataFrame({'fold_id': f, 'group_id': grp[va][keep],
                                  'forecast_kst_dtm': idx[va][keep], 'cf': cf[va][keep],
                                  'mean_gen_g': [float(np.nanmean(cf[tr & (grp == g)])) for g in grp[va][keep]]}))
        probs.append(P[keep])
        print(f'  [{tag}] {f} {round(time.time()-t0,1)}s  (n_cols={len(cols)})', flush=True)
    R = pd.concat(rows, ignore_index=True)
    Pf = np.vstack(probs)
    R.to_parquet(N + f'S7-N8_{tag}_keys.parquet', index=False)
    np.save(N + f'S7-N8_{tag}_prob.npy', Pf)
    return R, Pf


if __name__ == '__main__':
    A, FR, COLS = surface(('G2', 'DROP:grid__'))
    cld, cgf, excl = split_columns(COLS)
    print(f'LDAPS-only cols {len(cld)} | GFS-only cols {len(cgf)} | excluded-from-both {len(excl)}')
    print('excluded:', excl)
    json.dump({'ldaps_cols': cld, 'gfs_cols': cgf, 'excluded': excl},
              open(N + 'S12-N3_columns.json', 'w'), indent=1)
    build('DL', cld, A, COLS)
    build('DG', cgf, A, COLS)
    print('DONE', flush=True)
