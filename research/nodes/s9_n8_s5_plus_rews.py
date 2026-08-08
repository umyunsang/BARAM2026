
"""S9-N8 · the correctly-posed test S9-N7 could not run: REWS added to a member that
DOES have the S5 accepted preprocessing (this project's single largest gain,
foldout_delta +0.011612), not a from-scratch member missing it entirely.

S5 treatment reconstructed from parameters still visible in this repo (D-generating
script itself is missing, but its constants survive elsewhere):
  - P4 production-proportional weighting + P1(0.05) availability gating: exact
    w_prod/gapv formulas from research/nodes/s7_savemembers.py (which shares "the frozen
    S5 treatment" per its own docstring):
        valid = isfinite(cf) & (cf>=0.1); w_prod = where(valid, clip(cf,0,1.2), 0.05)
        gapv = pc_true - cf; availability-deficit rows = gapv >= 0.05
  - P7 soft-cap (measured ceiling, not nameplate): SC={1:0.985,2:0.989,3:1.005} from
    research/nodes/s10_final3.py, matches harness.py's soft_cap hook's cap_hi semantics.
  - S5-CLOSE_full.json's own tag confirms this combination: "P1(0.05)+P7+P4+uncond
    sigma+nq81 -- S5 declared best" (that recorded score, 0.615339, predates the DEF-1
    fix and cannot be reused for a temperature-sensitive comparison; re-derived here).

Four variants run through the identical 3-fold fold-outside protocol + DEPAVG blend
(same procedure as S9-N7 and S10-FINAL3):
  1. S5 only (no G2, no REWS)          -- reconstructs something close to D minus G2
  2. S5 + G2                            -- reconstructs something close to the real D
  3. S5 + REWS (no G2)                  -- this session's actual proposal
  4. S5 + G2 + REWS                     -- everything combined
"""
import sys, json
import numpy as np, pandas as pd
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


def s5_teacher_weight(A):
    cf = A['cf'].to_numpy()
    valid = np.isfinite(cf) & (cf >= 0.1)
    return np.where(valid, np.clip(cf, 0, 1.2), 0.05)


def s5_calib_rows(A):
    gapv = A['pc_true'].to_numpy() - A['cf'].to_numpy()
    return ~(gapv >= 0.05)


def run_fold_member(A0, COLS0, tr_mask, va_mask, fold_id, use_s5, use_softcap,
                     conditional_sigma=False, n_quantile=81):
    A, COLS = A0.copy(), COLS0
    w = s5_teacher_weight(A) if use_s5 else None
    m = tr_mask & np.isfinite(A['pc_true'].to_numpy())
    ww = None if w is None else w[m]
    mu_mdl = harness.lgb.LGBMRegressor(**harness.MU)
    mu_mdl.fit(A.loc[m, COLS], A.loc[m, 'pc_true'], sample_weight=ww)
    A['pc_hat'] = np.clip(mu_mdl.predict(A[COLS]), 0, 1)

    cal = tr_mask & np.isfinite(A['cf'].to_numpy()) & np.isfinite(A['pc_hat'].to_numpy())
    if use_s5:
        cal = cal & s5_calib_rows(A)
    A['resid'] = A['cf'] - A['pc_hat']
    if conditional_sigma:
        A['absres'] = np.abs(A['resid'])
        sg_mdl = harness.lgb.LGBMRegressor(**harness.SG)
        sg_mdl.fit(A.loc[cal, COLS], A.loc[cal, 'absres'])
        A['sigma'] = np.clip(sg_mdl.predict(A[COLS]), 1e-3, 1.0)
    else:
        A['sigma'] = 1.0
    qs = np.linspace(0.01, 0.99, n_quantile)
    ztab = {}
    for g in (1, 2, 3):
        m2 = cal & (A['grp'].to_numpy() == g)
        z = (A.loc[m2, 'resid'] / A.loc[m2, 'sigma']).to_numpy()
        ztab[g] = np.quantile(z[np.isfinite(z)], qs)

    parts = []
    for g in (1, 2, 3):
        sel = (A['grp'].to_numpy() == g) & va_mask
        pc = A.loc[sel, 'pc_hat'].to_numpy(); sd = A.loc[sel, 'sigma'].to_numpy()
        cf = A.loc[sel, 'cf'].to_numpy()
        fdt = A.index[sel]
        keep = np.isfinite(cf)
        cap_hi = SOFT_CAP[g] if use_softcap else 1.05
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


def build_member(tag, add_g2, add_rews):
    blocks = ('G2', 'DROP:grid__') if add_g2 else ()
    A0, FR0, COLS0 = harness.surface(blocks)
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
        per_fold[f] = run_fold_member(A, COLS, tr_mask, va_mask, f, use_s5=True, use_softcap=True)
        print(f'  [{tag}] fold {f} fitted, {per_fold[f].shape[0]} val rows', flush=True)

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
    print(f'{tag} SOLO fold-outside: {solo["total"]:.6f}  picks={picks}', flush=True)
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
    D = pd.concat(rows, ignore_index=True)
    return official_total(D[['group_id', 'actual_kwh', 'prediction_kwh']]), picks


if __name__ == '__main__':
    dep = load_depavg()
    results = {}
    for tag, add_g2, add_rews in [
        ('S5_only', False, False),
        ('S5_G2', True, False),
        ('S5_REWS', False, True),
        ('S5_G2_REWS', True, True),
    ]:
        member_df, picks, solo = build_member(tag, add_g2, add_rews)
        J = member_df.merge(dep, on=KEY)
        blend, blend_picks = fo_blend_1dof(J, 'MEMBER', 'DEPAVG')
        print(f'{tag}: solo={solo["total"]:.6f}  DEPAVG+{tag}={blend["total"]:.6f}  '
              f'delta_vs_current_best={blend["total"]-0.636184:+.6f}', flush=True)
        results[tag] = {'solo': solo, 'policy_picks': picks,
                         'depavg_blend': blend, 'blend_weight_picks': {k: list(v) for k, v in blend_picks.items()},
                         'delta_vs_current_best': blend['total'] - 0.636184}

    json.dump(results, open(N + 'S9-N8_s5_plus_rews_v2.json', 'w'), indent=1, default=str)
    print('\n=== SUMMARY (all vs current_best 0.636184) ===')
    for tag, r in results.items():
        print(f'  {tag:14s} solo={r["solo"]["total"]:.6f}  blend={r["depavg_blend"]["total"]:.6f}  '
              f'delta={r["delta_vs_current_best"]:+.6f}')
