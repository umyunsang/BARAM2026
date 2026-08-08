
"""S9-N11 · closes two previously-unexplored gaps between this session's S5
reconstructions (best solo ~0.6104-0.6257) and D's real solo score (0.625669):
  1. D is explicitly described as "DART-boosted" (current_best.composition) -- every
     prior reconstruction this session used harness.MU's default boosting_type='gbdt'.
     DART (Dropouts meet Multiple Additive Regression Trees) is a real LightGBM
     boosting_type, not a hyperparameter tweak -- switching it is architecture, not tuning.
  2. research/nodes/s7_savemembers.py's member-building loop selects only the top-150
     columns by teacher feature_importance before the calibration/decision stage
     (`sel=list(pd.Series(mu.feature_importances_,index=cols)...head(150).index)`) --
     no prior reconstruction this session did this feature-selection step either.

Both are read directly from existing repo code, not guessed. Combined with the S5
treatment (production weight, gapv>=0.05 gating, soft_cap, conditional_sigma=False,
n_quantile=81) and REWS, this is the most faithful reconstruction attempted this
session. If solo score doesn't land close to 0.625669, the remaining gap is genuinely
unexplained by anything visible in this repo, which is itself a useful, honestly-reported
conclusion -- not a reason to keep guessing indefinitely.
"""
import sys, json
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/scratch')
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/nodes')
import harness
from lib import official_total, FOLDS, CAPS, sharpen_weights
from s9_n6_rews_geom import rews_features

N = '/Users/um-yunsang/BARAM2026/research/nodes/'
AB = '/Users/um-yunsang/BARAM2026/artifacts/backtests/metric-aligned-probe/'
KEY = ['fold_id', 'group_id', 'forecast_kst_dtm']
DEP = {'M102_TOP100': 'T0.5_G1.5', 'M113_LGBM_DART': 'T0.5_G0.5', 'M115_XGBOOST': 'T0.6_G0.35'}
SOFT_CAP = {1: 0.985, 2: 0.989, 3: 1.005}

MU_DART = dict(objective='l2', boosting_type='dart', n_estimators=900, learning_rate=0.035,
               num_leaves=63, min_child_samples=40, subsample=0.85, subsample_freq=1,
               colsample_bytree=0.4, reg_lambda=3.0, drop_rate=0.1, skip_drop=0.5,
               random_state=20260801, n_jobs=6, verbose=-1)


def s5_teacher_weight(A):
    cf = A['cf'].to_numpy()
    valid = np.isfinite(cf) & (cf >= 0.1)
    return np.where(valid, np.clip(cf, 0, 1.2), 0.05)


def s5_calib_rows(A):
    gapv = A['pc_true'].to_numpy() - A['cf'].to_numpy()
    return ~(gapv >= 0.05)


def run_fold_member(A0, COLS0, tr_mask, va_mask, fold_id):
    A = A0.copy()
    w = s5_teacher_weight(A)
    m = tr_mask & np.isfinite(A['pc_true'].to_numpy())
    mu_mdl = lgb.LGBMRegressor(**MU_DART)
    mu_mdl.fit(A.loc[m, COLS0], A.loc[m, 'pc_true'], sample_weight=w[m])

    # top-150 feature selection by importance, mirroring s7_savemembers.py
    sel = list(pd.Series(mu_mdl.feature_importances_, index=COLS0)
               .sort_values(ascending=False).head(150).index)
    COLS = sel
    A['pc_hat'] = np.clip(mu_mdl.predict(A[COLS0]), 0, 1)

    cal = tr_mask & np.isfinite(A['cf'].to_numpy()) & np.isfinite(A['pc_hat'].to_numpy()) & s5_calib_rows(A)
    A['resid'] = A['cf'] - A['pc_hat']
    A['sigma'] = 1.0  # conditional_sigma=False, per S5-CLOSE's own tag
    qs = np.linspace(0.01, 0.99, 81)
    ztab = {}
    for g in (1, 2, 3):
        m2 = cal & (A['grp'].to_numpy() == g)
        z = (A.loc[m2, 'resid'] / A.loc[m2, 'sigma']).to_numpy()
        ztab[g] = np.quantile(z[np.isfinite(z)], qs)

    parts = []
    for g in (1, 2, 3):
        sel_va = (A['grp'].to_numpy() == g) & va_mask
        pc = A.loc[sel_va, 'pc_hat'].to_numpy(); sd = A.loc[sel_va, 'sigma'].to_numpy()
        cf = A.loc[sel_va, 'cf'].to_numpy()
        fdt = A.index[sel_va]
        keep = np.isfinite(cf)
        cap_hi = SOFT_CAP[g]
        samples = np.clip(pc[:, None] + sd[:, None] * ztab[g][None, :], 0.0, cap_hi)
        mean_gen = float(np.nanmean(A.loc[(A['grp'].to_numpy() == g) & tr_mask, 'cf']))
        err = np.abs(harness.ACTIONS[None, :, None] - samples[:, None, :])
        units = np.where(err <= 0.06, 4.0, np.where(err <= 0.08, 3.0, 0.0))
        rec = {'fold_id': fold_id, 'group_id': g, 'forecast_kst_dtm': fdt[keep],
               'actual_kwh': cf[keep] * CAPS[g]}
        for tp in harness.TEMPS:
            wq = sharpen_weights(samples, tp)[:, None, :]
            nm = -(err * wq).sum(axis=2)
            fi = ((samples[:, None, :] * units) * wq).sum(axis=2) / (4.0 * mean_gen)
            for gm in harness.GAMMAS:
                a = harness.ACTIONS[np.argmax(nm + gm * fi, axis=1)]
                rec[f'T{tp}_G{gm}'] = np.clip(a, 0, cap_hi)[keep] * CAPS[g]
        parts.append(pd.DataFrame(rec))
    return pd.concat(parts, ignore_index=True)


def build_member(add_rews):
    A0, FR0, COLS0 = harness.surface(('G2', 'DROP:grid__'))
    if add_rews:
        feats = rews_features()
        FR = {}
        for g in (1, 2, 3):
            X = FR0[g].copy()
            FR[g] = pd.concat([X, feats.reindex(X.index)], axis=1)
        A = pd.concat(FR.values())
        COLS = COLS0 + list(feats.columns)
    else:
        A, COLS = A0, COLS0

    per_fold = {}
    for f, (a, b) in FOLDS.items():
        a = pd.Timestamp(a); b = pd.Timestamp(b)
        tr_mask = np.asarray(A.index < a)
        va_mask = np.asarray((A.index >= a) & (A.index <= b))
        per_fold[f] = run_fold_member(A, COLS, tr_mask, va_mask, f)
        print(f'  fold {f} fitted ({"S5+DART+top150+REWS" if add_rews else "S5+DART+top150"}), '
              f'{per_fold[f].shape[0]} val rows', flush=True)

    policy_cols = [c for c in per_fold[next(iter(FOLDS))].columns if c.startswith('T')]
    out_rows = []; picks = {}
    for f in FOLDS:
        others = pd.concat([per_fold[o] for o in FOLDS if o != f], ignore_index=True)
        scores = {c: official_total(others.assign(prediction_kwh=others[c])[
            ['group_id', 'actual_kwh', 'prediction_kwh']])['total'] for c in policy_cols}
        best_c = max(scores, key=scores.get)
        picks[f] = best_c
        held = per_fold[f]
        out_rows.append(held.assign(MEMBER=held[best_c])[KEY + ['actual_kwh', 'MEMBER']])
    member_df = pd.concat(out_rows, ignore_index=True)
    solo = official_total(member_df.rename(columns={'MEMBER': 'prediction_kwh'})[
        ['group_id', 'actual_kwh', 'prediction_kwh']])
    print(f'SOLO fold-outside: {solo["total"]:.6f}  (D reference: 0.625669)', flush=True)
    return member_df, picks, solo


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
    dep = load_depavg()
    results = {}
    for tag, add_rews in [('S5_DART_top150', False), ('S5_DART_top150_REWS', True)]:
        member_df, picks, solo = build_member(add_rews)
        J = member_df.merge(dep, on=KEY)
        blend, blend_picks = fo_blend_1dof(J, 'MEMBER', 'DEPAVG')
        print(f'{tag}: solo={solo["total"]:.6f}  DEPAVG+{tag}={blend["total"]:.6f}  '
              f'delta_vs_current_best={blend["total"]-0.636184:+.6f}', flush=True)
        results[tag] = {'solo': solo, 'depavg_blend': blend,
                         'delta_vs_current_best': blend['total'] - 0.636184}

    json.dump(results, open(N + 'S9-N11_dart_s5_rews.json', 'w'), indent=1, default=str)
    print('\n=== SUMMARY ===')
    for tag, r in results.items():
        print(f'  {tag:22s} solo={r["solo"]["total"]:.6f}  blend={r["depavg_blend"]["total"]:.6f}  '
              f'delta={r["delta_vs_current_best"]:+.6f}')
