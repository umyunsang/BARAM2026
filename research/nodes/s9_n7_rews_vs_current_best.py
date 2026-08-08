
"""S9-N7 · direct test: does a member built WITHOUT G2 (flagged this session as a net
drag under fold-outside) and WITH REWS (confirmed +0.001937 isolated gain) beat
current_best (0.636184, S10-FINAL3's DEPAVG+D blend) when blended with the SAME DEPAVG
(M102_TOP100/M113_LGBM_DART/M115_XGBOOST average) using the SAME 1-dof fold-outside
blend procedure as research/nodes/s10_final3.py?

This is the direct, decision-relevant test the isolated harness-level comparisons this
session (S9-N6, S9-N6 stack-on-G2) could only gesture at, since the actual D member's
generating script could not be found in this repo to edit directly.

Member construction: harness default surface (NO G2, matching this session's finding
that G2 alone underperforms fold-outside) + REWS features, teacher+calibrator+decision
per fold (expanding window), fold-outside-selected (T,G) policy applied per held-out
fold -- exactly mirroring s9_n6_foldoutside.py's own protocol, but emitting the actual
per-row kWh prediction column (not just the aggregate score) so it can be blended.
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


def run_fold_member(A0, COLS0, tr_mask, va_mask, fold_id):
    """Returns a DataFrame with kwh predictions for EVERY (T,G) policy on this fold's
    validation rows, so the caller can pick the fold-outside policy afterward."""
    A, COLS = A0.copy(), COLS0
    mu_mdl, _ = harness._fit_pooled(A, COLS, tr_mask, harness.MU, 'pc_true')
    A['pc_hat'] = np.clip(mu_mdl.predict(A[COLS]), 0, 1)
    cal = tr_mask & np.isfinite(A['cf'].to_numpy()) & np.isfinite(A['pc_hat'].to_numpy())
    A['resid'] = A['cf'] - A['pc_hat']
    A['absres'] = np.abs(A['resid'])
    sg_mdl, _ = harness._fit_pooled(A, COLS, cal, harness.SG, 'absres')
    A['sigma'] = np.clip(sg_mdl.predict(A[COLS]), 1e-3, 1.0)
    qs = np.linspace(0.01, 0.99, 41)
    ztab = {}
    for g in (1, 2, 3):
        m = cal & (A['grp'].to_numpy() == g)
        z = (A.loc[m, 'resid'] / A.loc[m, 'sigma']).to_numpy()
        ztab[g] = np.quantile(z[np.isfinite(z)], qs)

    parts = []
    for g in (1, 2, 3):
        sel = (A['grp'].to_numpy() == g) & va_mask
        pc = A.loc[sel, 'pc_hat'].to_numpy(); sd = A.loc[sel, 'sigma'].to_numpy()
        cf = A.loc[sel, 'cf'].to_numpy()
        fdt = A.index[sel]
        keep = np.isfinite(cf)
        samples = np.clip(pc[:, None] + sd[:, None] * ztab[g][None, :], 0.0, 1.05)
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
                rec[f'T{tp}_G{gm}'] = np.clip(a, 0, 1.05)[keep] * CAPS[g]
        parts.append(pd.DataFrame(rec))
    return pd.concat(parts, ignore_index=True)


def build_rews_member():
    A0, FR0, COLS0 = harness.surface(())  # NO G2
    feats = rews_features()
    FR = {}
    for g in (1, 2, 3):
        X = FR0[g].copy()
        FR[g] = pd.concat([X, feats.reindex(X.index)], axis=1)
    A = pd.concat(FR.values())
    COLS = COLS0 + list(feats.columns)

    per_fold = {}
    for f, (a, b) in FOLDS.items():
        a = pd.Timestamp(a); b = pd.Timestamp(b)
        tr_mask = np.asarray(A.index < a)
        va_mask = np.asarray((A.index >= a) & (A.index <= b))
        per_fold[f] = run_fold_member(A, COLS, tr_mask, va_mask, f)
        print(f'  member fold {f} fitted, {per_fold[f].shape[0]} val rows', flush=True)

    policy_cols = [c for c in per_fold[next(iter(FOLDS))].columns if c.startswith('T')]
    out_rows = []; picks = {}
    for f in FOLDS:
        others = pd.concat([per_fold[o] for o in FOLDS if o != f], ignore_index=True)
        scores = {c: official_total(others.assign(prediction_kwh=others[c])[
            ['group_id', 'actual_kwh', 'prediction_kwh']])['total'] for c in policy_cols}
        best_c = max(scores, key=scores.get)
        picks[f] = best_c
        held = per_fold[f]
        out_rows.append(held.assign(REWS_MEMBER=held[best_c])[
            KEY + ['actual_kwh', 'REWS_MEMBER']])
    member_df = pd.concat(out_rows, ignore_index=True)
    print(f'REWS member fold-outside picks: {picks}', flush=True)
    solo = official_total(member_df.rename(columns={'REWS_MEMBER': 'prediction_kwh'})[
        ['group_id', 'actual_kwh', 'prediction_kwh']])
    print(f'REWS member SOLO fold-outside total: {solo["total"]:.6f}', flush=True)
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


def fo_blend_1dof(J, col_a, col_b, actual_col='actual_kwh'):
    grid = [(w, 1 - w) for w in np.arange(0, 1.001, 0.05)]
    rows = []; picks = {}
    for f in FOLDS:
        oth = J[J.fold_id != f]; held = J[J.fold_id == f]
        best = None
        for wa, wb in grid:
            pred = wa * oth[col_a] + wb * oth[col_b]
            t = official_total(oth.assign(prediction_kwh=pred)[
                ['group_id', actual_col, 'prediction_kwh']].rename(columns={actual_col: 'actual_kwh'}))['total']
            if best is None or t > best[0]:
                best = (t, (wa, wb))
        picks[f] = best[1]
        pred_h = best[1][0] * held[col_a] + best[1][1] * held[col_b]
        rows.append(held.assign(prediction_kwh=pred_h))
    D = pd.concat(rows, ignore_index=True)
    D = D.rename(columns={actual_col: 'actual_kwh'}) if actual_col != 'actual_kwh' else D
    return official_total(D[['group_id', 'actual_kwh', 'prediction_kwh']]), picks


if __name__ == '__main__':
    member_df, member_picks, solo = build_rews_member()
    dep = load_depavg()
    J = member_df.merge(dep, on=KEY)
    print(f'merged rows: {len(J)}', flush=True)

    blend, blend_picks = fo_blend_1dof(J, 'REWS_MEMBER', 'DEPAVG')
    print(f'DEPAVG + REWS_MEMBER (1 dof) fold-outside: {blend["total"]:.6f} picks={blend_picks}', flush=True)
    print(f'vs current_best (S10-FINAL3, DEPAVG+D): 0.636184', flush=True)
    print(f'DELTA vs current_best: {blend["total"] - 0.636184:+.6f}', flush=True)

    json.dump({'member_solo_foldoutside': solo, 'member_policy_picks': member_picks,
               'depavg_plus_rews_member_1dof': blend, 'blend_weight_picks': {k: list(v) for k, v in blend_picks.items()},
               'current_best_reference': 0.636184,
               'delta_vs_current_best': blend['total'] - 0.636184},
              open(N + 'S9-N7_vs_current_best.json', 'w'), indent=1, default=str)
