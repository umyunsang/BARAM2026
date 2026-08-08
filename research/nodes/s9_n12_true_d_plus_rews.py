
"""S9-N12 · the CORRECT reconstruction, using the actual D-generating script found:
research/nodes/s7_more.py (found via mtime correlation with S7-N8_D_*.{parquet,npy},
11:42:27 today -- s7_savemembers.py, which every prior reconstruction this session was
modeled on, only builds members P/L/G/Q and was a red herring for D specifically).

D is NOT built through harness.py's regression-teacher/quantile/decision-layer pipeline
at all (every S9-N7/N8/N9/N10/N11 attempt this session used that pipeline and was
therefore reconstructing the wrong architecture). D is:
  1. GBDT regressor (harness.MU, plain gbdt, NOT dart) -> pc_hat, sample-weighted by
     production-proportional w_prod, exactly as in s7_savemembers.py.
  2. Top-150 features by teacher importance + pc_hat + group one-hot (ig1/ig2/ig3) -> B.
  3. cf discretized into NC=26 classes (width W=0.04).
  4. A DART-boosted (boosting_type='dart') LightGBM multiclass CLASSIFIER over those 26
     classes, trained on cm-gated rows (tr & isfinite(cf) & ~(gapv>=0.05)) with
     w_valid sample weights (NOT w_prod -- different weight vector than the regressor).
  5. predict_proba() -> saved probability array, later turned into an action via
     s10_final3.py::member()'s (T,G) grid + fold-outside policy selection.

This script reproduces that exact pipeline (verified against s7_more.py line-by-line)
with ONE change: REWS columns are concatenated into B before the classifier, bypassing
the top-150 selection (so REWS is never at risk of being dropped by unrelated feature
competition) -- the cleanest, least invasive way to test its marginal value on the
TRUE architecture.
"""
import sys, json, time
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/scratch')
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/nodes')
from harness import surface, MU
from lib import FOLDS, CAPS, official_total
from s9_n6_rews_geom import rews_features

N = '/Users/um-yunsang/BARAM2026/research/nodes/'
AB = '/Users/um-yunsang/BARAM2026/artifacts/backtests/metric-aligned-probe/'
KEY = ['fold_id', 'group_id', 'forecast_kst_dtm']
DEP = {'M102_TOP100': 'T0.5_G1.5', 'M113_LGBM_DART': 'T0.5_G0.5', 'M115_XGBOOST': 'T0.6_G0.35'}
W = 0.04; NC = 26
ACT = np.arange(0.02, 1.0801, 0.0025)
SC = {1: 0.985, 2: 0.989, 3: 1.005}
TEMPS = [0.6, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
GAMMAS = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0]

DART_CLF = dict(objective='multiclass', boosting_type='dart', n_estimators=400,
                 learning_rate=0.08, num_leaves=31, min_child_samples=60, subsample=0.85,
                 subsample_freq=1, colsample_bytree=0.4, reg_lambda=3.0,
                 random_state=20260803, n_jobs=6, verbose=-1)


def build_and_save(tag, add_rews):
    A, FR, COLS = surface(('G2', 'DROP:grid__'))
    cf = A['cf'].to_numpy(); grp = A['grp'].to_numpy(); idx = A.index
    valid = np.isfinite(cf) & (cf >= 0.1)
    w_prod = np.where(valid, np.clip(cf, 0, 1.2), 0.05)
    w_valid = np.where(valid, 1.0, 0.15)
    gapv = A['pc_true'].to_numpy() - cf

    rews = rews_features() if add_rews else None
    rows = []; probs = []
    for f, (a, b) in FOLDS.items():
        a = pd.Timestamp(a); b = pd.Timestamp(b)
        tr = np.asarray(idx < a); va = np.asarray((idx >= a) & (idx <= b)); keep = np.isfinite(cf[va])
        t0 = time.time()
        m = tr & np.isfinite(A['pc_true'].to_numpy())
        mu = lgb.LGBMRegressor(**MU)
        mu.fit(A.loc[m, COLS], A.loc[m, 'pc_true'], sample_weight=w_prod[m])
        pc = np.clip(mu.predict(A[COLS]), 0, 1)
        sel = list(pd.Series(mu.feature_importances_, index=COLS).sort_values(ascending=False).head(150).index)
        B = A[sel].copy(); B['pc_hat'] = pc
        for k in (1, 2, 3):
            B[f'ig{k}'] = (grp == k).astype('float32')
        if add_rews:
            for c in rews.columns:
                B[c] = rews.reindex(A.index)[c].to_numpy()
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
        print(f'  [{tag}] fold {f} fitted {round(time.time()-t0,1)}s', flush=True)
    R = pd.concat(rows, ignore_index=True)
    Pfull = np.vstack(probs)
    R.to_parquet(N + f'S7-N8_{tag}_keys.parquet', index=False)
    np.save(N + f'S7-N8_{tag}_prob.npy', Pfull)
    return R, Pfull


def member_from_saved(tag):
    """Exact port of s10_final3.py::member()."""
    R = pd.read_parquet(N + f'S7-N8_{tag}_keys.parquet')
    P = np.load(N + f'S7-N8_{tag}_prob.npy')
    NCk = P.shape[1]; C = (np.arange(NCk) + 0.5) * W
    err = np.abs(ACT[:, None] - C[None, :]); units = np.where(err <= 0.06, 4., np.where(err <= 0.08, 3., 0.))
    g = R.group_id.to_numpy(); mg = R.mean_gen_g.to_numpy()
    capv = np.array([CAPS[x] for x in g]); hi = np.array([SC[x] for x in g]); act = R.cf.to_numpy() * capv
    mask = (C >= 0.10).astype(float); frames = {}
    for tp in TEMPS:
        q = P ** (1.0 / tp); q = q / np.maximum(q.sum(axis=1, keepdims=True), 1e-12)
        q = q * mask[None, :]; q = q / np.maximum(q.sum(axis=1, keepdims=True), 1e-12)
        nm = -(q @ err.T); fic = (q @ ((C[None, :] * units).T))
        for gm in GAMMAS:
            frames[(tp, gm)] = np.minimum(ACT[np.argmax(nm + gm * fic / (4.0 * mg[:, None]), axis=1)], hi) * capv
    Dm = pd.DataFrame({'fold_id': R.fold_id, 'group_id': g, 'forecast_kst_dtm': R.forecast_kst_dtm, 'actual_kwh': act})
    out = np.empty(len(Dm))
    for f in FOLDS:
        sel = (Dm.fold_id == f).to_numpy()
        s2 = {k: official_total(Dm[~sel].assign(prediction_kwh=v[~sel])[['group_id', 'actual_kwh', 'prediction_kwh']])['total']
              for k, v in frames.items()}
        out[sel] = frames[max(s2, key=s2.get)][sel]
    Dm['prediction_kwh'] = out
    return Dm, official_total(Dm[['group_id', 'actual_kwh', 'prediction_kwh']])['total']


def load_depavg():
    fr = []
    for stem, pol in DEP.items():
        parts = []
        for f in FOLDS:
            d = pd.read_parquet(AB + f'{stem}-{f}-policies.parquet')
            d = d.copy(); d['fold_id'] = f
            parts.append(d[KEY + [pol]].rename(columns={pol: stem}))
        fr.append(pd.concat(parts, ignore_index=True))
    J = fr[0]
    for x in fr[1:]:
        J = J.merge(x, on=KEY)
    J['DEPAVG'] = J[list(DEP)].mean(axis=1)
    return J[KEY + ['DEPAVG']]


def fo_blend_1dof(J, col_a, col_b):
    grid = [(w, 1 - w) for w in np.arange(0, 1.001, 0.05)]
    rows = []; picks = {}
    for f in FOLDS:
        oth = J[J.fold_id != f]; held = J[J.fold_id == f]
        best = None
        for wa, wb in grid:
            pred = wa * oth[col_a] + wb * oth[col_b]
            t = official_total(oth.assign(prediction_kwh=pred)[
                ['group_id', 'actual_kwh', 'prediction_kwh']])['total']
            if best is None or t > best[0]:
                best = (t, (wa, wb))
        picks[f] = best[1]
        pred_h = best[1][0] * held[col_a] + best[1][1] * held[col_b]
        rows.append(held.assign(prediction_kwh=pred_h))
    Dd = pd.concat(rows, ignore_index=True)
    return official_total(Dd[['group_id', 'actual_kwh', 'prediction_kwh']]), picks


if __name__ == '__main__':
    build_and_save('D_REPRO', add_rews=False)
    Dm_repro, solo_repro = member_from_saved('D_REPRO')
    print(f'D_REPRO solo (should match real D ~0.625669): {solo_repro:.6f}', flush=True)

    build_and_save('D_REWS', add_rews=True)
    Dm_rews, solo_rews = member_from_saved('D_REWS')
    print(f'D_REWS solo: {solo_rews:.6f}  delta vs D_REPRO: {solo_rews - solo_repro:+.6f}', flush=True)

    dep = load_depavg()
    J_repro = Dm_repro[KEY + ['actual_kwh', 'prediction_kwh']].rename(columns={'prediction_kwh': 'D_REPRO'}).merge(dep, on=KEY)
    blend_repro, _ = fo_blend_1dof(J_repro, 'D_REPRO', 'DEPAVG')
    print(f'DEPAVG + D_REPRO: {blend_repro["total"]:.6f}  (reference current_best=0.636184)', flush=True)

    J_rews = Dm_rews[KEY + ['actual_kwh', 'prediction_kwh']].rename(columns={'prediction_kwh': 'D_REWS'}).merge(dep, on=KEY)
    blend_rews, _ = fo_blend_1dof(J_rews, 'D_REWS', 'DEPAVG')
    print(f'DEPAVG + D_REWS: {blend_rews["total"]:.6f}  delta vs current_best: {blend_rews["total"]-0.636184:+.6f}', flush=True)
    print(f'delta vs DEPAVG+D_REPRO: {blend_rews["total"]-blend_repro["total"]:+.6f}', flush=True)

    json.dump({'solo_repro': solo_repro, 'solo_rews': solo_rews,
               'depavg_plus_repro': blend_repro, 'depavg_plus_rews': blend_rews,
               'delta_vs_current_best': blend_rews['total'] - 0.636184,
               'delta_vs_repro': blend_rews['total'] - blend_repro['total']},
              open(N + 'S9-N12_true_d_plus_rews.json', 'w'), indent=1, default=str)
