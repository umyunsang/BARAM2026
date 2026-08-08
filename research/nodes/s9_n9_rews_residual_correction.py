
"""S9-N9 · residual correction: instead of re-deriving the D member (its generating
script is missing and two reconstruction attempts fell well short, S9-N8 v1/v2), treat
D's ALREADY-SAVED prediction as a black box and test whether REWS explains any of its
residual error. This sidesteps the reconstruction problem entirely.

member() reproduces D's actual fold-outside kwh prediction exactly as
research/nodes/s10_final3.py does (same formula, same constants: W, SC, TEMPS, GAMMAS,
mask, fold-outside policy selection) from the saved S7-N8_D_keys.parquet/D_prob.npy.

Correction model: for each fold, fit a small LightGBM regressor on REWS features ->
(actual_kwh - D_prediction) using only the OTHER two folds (fold-outside, matching the
project's standing dof discipline), predict the held-out fold's residual, add it to D's
prediction (clipped to [0, soft_cap*capacity]), and re-score. If this beats D alone
(and, blended with DEPAVG, beats current_best), REWS carries real marginal information
D's own decision layer isn't capturing.
"""
import sys, json
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/scratch')
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/nodes')
import harness
from lib import official_total, FOLDS, CAPS
from s9_n6_rews_geom import rews_features

N = '/Users/um-yunsang/BARAM2026/research/nodes/'
AB = '/Users/um-yunsang/BARAM2026/artifacts/backtests/metric-aligned-probe/'
KEY = ['fold_id', 'group_id', 'forecast_kst_dtm']
DEP = {'M102_TOP100': 'T0.5_G1.5', 'M113_LGBM_DART': 'T0.5_G0.5', 'M115_XGBOOST': 'T0.6_G0.35'}
W = 0.04
ACT = np.arange(0.02, 1.0801, 0.0025)
SC = {1: 0.985, 2: 0.989, 3: 1.005}
TEMPS = [0.6, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
GAMMAS = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0]

CORR = dict(objective='l2', n_estimators=300, learning_rate=0.05, num_leaves=15,
            min_child_samples=80, subsample=0.85, subsample_freq=1, colsample_bytree=0.8,
            reg_lambda=3.0, random_state=20260801, n_jobs=6, verbose=-1)


def member(name):
    """Exact reproduction of s10_final3.py::member() -- D's real fold-outside prediction."""
    R = pd.read_parquet(N + f'S7-N8_{name}_keys.parquet')
    P = np.load(N + f'S7-N8_{name}_prob.npy')
    NC = P.shape[1]; C = (np.arange(NC) + 0.5) * W
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
    Dm, d_solo = member('D')
    print(f'D solo fold-outside (reproduced): {d_solo:.6f}  (recorded: 0.625651)', flush=True)

    # REWS features aligned to (group_id, forecast_kst_dtm)
    feats = rews_features()
    rews_cols = list(feats.columns)
    Dm = Dm.copy()
    for c in rews_cols:
        Dm[c] = np.nan
    for g in (1, 2, 3):
        sel = Dm.group_id == g
        # rews_features has one row per timestamp shared across groups; reindex by fdt
        Dm.loc[sel, rews_cols] = feats.reindex(Dm.loc[sel, 'forecast_kst_dtm']).to_numpy()

    Dm['residual'] = Dm['actual_kwh'] - Dm['prediction_kwh']
    cap_of = Dm.group_id.map(CAPS)

    corrected = np.zeros(len(Dm))
    for f in FOLDS:
        tr = (Dm.fold_id != f).to_numpy()
        va = (Dm.fold_id == f).to_numpy()
        m = lgb.LGBMRegressor(**CORR)
        m.fit(Dm.loc[tr, rews_cols], Dm.loc[tr, 'residual'])
        corrected[va] = Dm.loc[va, 'prediction_kwh'].to_numpy() + m.predict(Dm.loc[va, rews_cols])
    Dm['prediction_kwh_corrected'] = np.clip(corrected, 0, (cap_of * 1.05).to_numpy())

    corrected_score = official_total(Dm[['group_id', 'actual_kwh', 'prediction_kwh_corrected']]
                                      .rename(columns={'prediction_kwh_corrected': 'prediction_kwh'}))
    print(f'D + REWS residual-correction fold-outside: {corrected_score["total"]:.6f}', flush=True)
    print(f'delta vs D solo: {corrected_score["total"] - d_solo:+.6f}', flush=True)

    dep = load_depavg()
    J_orig = Dm[KEY + ['actual_kwh', 'prediction_kwh']].rename(columns={'prediction_kwh': 'D'}).merge(dep, on=KEY)
    blend_orig, picks_orig = fo_blend_1dof(J_orig, 'D', 'DEPAVG')
    print(f'DEPAVG + D (reproduced) fold-outside: {blend_orig["total"]:.6f}  (reference current_best=0.636184)', flush=True)

    J_corr = Dm[KEY + ['actual_kwh', 'prediction_kwh_corrected']].rename(columns={'prediction_kwh_corrected': 'D_corrected'}).merge(dep, on=KEY)
    blend_corr, picks_corr = fo_blend_1dof(J_corr, 'D_corrected', 'DEPAVG')
    print(f'DEPAVG + D_corrected(REWS) fold-outside: {blend_corr["total"]:.6f}', flush=True)
    print(f'delta vs current_best (0.636184): {blend_corr["total"] - 0.636184:+.6f}', flush=True)
    print(f'delta vs reproduced DEPAVG+D ({blend_orig["total"]:.6f}): {blend_corr["total"] - blend_orig["total"]:+.6f}', flush=True)

    json.dump({'d_solo_reproduced': d_solo, 'd_plus_rews_correction_solo': corrected_score,
               'depavg_plus_d_reproduced': blend_orig, 'depavg_plus_d_corrected': blend_corr,
               'delta_vs_current_best': blend_corr['total'] - 0.636184,
               'delta_vs_reproduced_baseline': blend_corr['total'] - blend_orig['total']},
              open(N + 'S9-N9_residual_correction.json', 'w'), indent=1, default=str)
