
"""S9-N10 · decision-theoretically consistent REWS use: instead of regressing D's
residual (S9-N9, failed -- D's prediction is an action under a step reward, not a
conditional mean, so plain L2 residual regression is the wrong objective), split rows
into REWS-based regimes and pick a SEPARATE fold-outside (T,G) policy per regime, the
same kind of object D's own decision layer already optimizes (a policy, not a row-level
regression target). This has 0 extra fitted parameters beyond the regime boundaries
themselves, which are fixed a priori (quantile-based, not tuned on score).

Regime: rews_ratio_L_g{group} (the REWS/hub-speed ratio for that row's own group, from
research/nodes/s9_n6_rews_geom.py) split into terciles (fixed cut points: 33rd/67th
percentile of the TRAIN portion only, per fold, to avoid leakage) -- 3 regimes x
(TEMPS x GAMMAS) grid, fold-outside policy selection exactly as in
research/nodes/s10_final3.py::fo_blend/member, just with an extra regime split.
"""
import sys, json
import numpy as np, pandas as pd
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/scratch')
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/nodes')
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


def d_action_grid():
    """Reproduce every (T,G) action for D, per row, WITHOUT collapsing to one policy yet."""
    R = pd.read_parquet(N + 'S7-N8_D_keys.parquet')
    P = np.load(N + 'S7-N8_D_prob.npy')
    NC = P.shape[1]; C = (np.arange(NC) + 0.5) * W
    err = np.abs(ACT[:, None] - C[None, :]); units = np.where(err <= 0.06, 4., np.where(err <= 0.08, 3., 0.))
    g = R.group_id.to_numpy(); mg = R.mean_gen_g.to_numpy()
    capv = np.array([CAPS[x] for x in g]); hi = np.array([SC[x] for x in g]); act = R.cf.to_numpy() * capv
    mask = (C >= 0.10).astype(float)
    D = pd.DataFrame({'fold_id': R.fold_id, 'group_id': g, 'forecast_kst_dtm': R.forecast_kst_dtm, 'actual_kwh': act})
    for tp in TEMPS:
        q = P ** (1.0 / tp); q = q / np.maximum(q.sum(axis=1, keepdims=True), 1e-12)
        q = q * mask[None, :]; q = q / np.maximum(q.sum(axis=1, keepdims=True), 1e-12)
        nm = -(q @ err.T); fic = (q @ ((C[None, :] * units).T))
        for gm in GAMMAS:
            D[f'T{tp}_G{gm}'] = np.minimum(ACT[np.argmax(nm + gm * fic / (4.0 * mg[:, None]), axis=1)], hi) * capv
    return D


def attach_regime(D):
    feats = rews_features()
    rr_cols = {1: 'rews_ratio_L_g1', 2: 'rews_ratio_L_g2', 3: 'rews_ratio_L_g3'}
    val = np.empty(len(D))
    for g, c in rr_cols.items():
        sel = (D.group_id == g).to_numpy()
        val[sel] = feats.reindex(D.loc[sel, 'forecast_kst_dtm'])[c].to_numpy()
    D = D.copy(); D['rews_ratio'] = val
    return D


def group_partial_score(df, g):
    """Per-group partial (nm_g, fi_g), matching official_total's own per-group inner
    loop exactly -- calling official_total() on a group-filtered subset is WRONG (the
    other 2 groups end up empty -> NaN -> corrupts the 3-group mean). This is the fix."""
    p = df[df.group_id == g]
    cap = CAPS[g]
    v = p[p.actual_kwh >= 0.1 * cap]
    if len(v) == 0:
        return np.nan, np.nan
    err = np.abs(v.prediction_kwh.to_numpy(float) - v.actual_kwh.to_numpy(float)) / cap
    units = np.select([err <= 0.06, err <= 0.08], [4.0, 3.0], default=0.0)
    a = v.actual_kwh.to_numpy(float)
    nm_g = float(err.mean())
    fi_g = float((a * units).sum() / (a * 4.0).sum())
    return nm_g, fi_g


def group_partial_objective(df, g):
    """0.5*(1-nm_g)+0.5*fi_g for ONE group -- maximizing this per group is equivalent
    to maximizing the true 3-group-averaged Total with respect to that group's own
    prediction column, since the other groups' terms don't depend on it."""
    nm_g, fi_g = group_partial_score(df, g)
    if not np.isfinite(nm_g):
        return -np.inf
    return 0.5 * (1 - nm_g) + 0.5 * fi_g


def _bin_edges_per_group(oth, n_regimes):
    """Quantile cut points computed WITHIN each group separately, so every (regime,
    group) cell is guaranteed non-empty -- avoids the empty-slice NaN bug from a
    group-agnostic global split."""
    edges = {}
    for g in (1, 2, 3):
        v = oth.loc[oth.group_id == g, 'rews_ratio'].to_numpy()
        e = np.quantile(v, np.linspace(0, 1, n_regimes + 1))
        e[0] -= 1e-9; e[-1] += 1e-9
        edges[g] = e
    return edges


def _assign_bin(df, edges, n_regimes):
    out = np.zeros(len(df), dtype=int)
    for g in (1, 2, 3):
        sel = (df.group_id == g).to_numpy()
        out[sel] = np.clip(np.searchsorted(edges[g], df.loc[sel, 'rews_ratio'].to_numpy(), side='right') - 1,
                            0, n_regimes - 1)
    return out


def regime_foldoutside_score(D, policy_cols, n_regimes=3):
    """For each fold, cut TRAIN-only rews_ratio into n_regimes quantile bins PER GROUP
    (fixed boundaries from the other 2 folds), pick the best policy per (group, regime)
    cell from those 2 folds, apply to the held-out fold's matching cells."""
    out_rows = []
    regime_picks = {}
    for f in FOLDS:
        oth = D[D.fold_id != f]; held = D[D.fold_id == f].copy()
        edges = _bin_edges_per_group(oth, n_regimes)
        oth_bin = _assign_bin(oth, edges, n_regimes)
        held_bin = _assign_bin(held, edges, n_regimes)
        pred = np.empty(len(held))
        picks = {}
        for g in (1, 2, 3):
            for r in range(n_regimes):
                sub = oth[(oth.group_id == g) & (oth_bin == r)]
                if len(sub) < 100:
                    sub = oth[oth.group_id == g]  # fallback: pool the whole group
                scores = {c: group_partial_objective(sub.assign(prediction_kwh=sub[c]), g) for c in policy_cols}
                best_c = max(scores, key=scores.get)
                picks[(g, r)] = best_c
                sel = (held.group_id == g).to_numpy() & (held_bin == r)
                pred[sel] = held.loc[sel, best_c].to_numpy()
        held['prediction_kwh'] = pred
        regime_picks[f] = picks
        out_rows.append(held[['group_id', 'actual_kwh', 'prediction_kwh']])
    Dout = pd.concat(out_rows, ignore_index=True)
    return official_total(Dout), regime_picks


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
    D = d_action_grid()
    D = attach_regime(D)
    policy_cols = [c for c in D.columns if c.startswith('T')]

    baseline, _ = regime_foldoutside_score(D, policy_cols, n_regimes=1)
    print(f'n_regimes=1 (== D reproduced): {baseline["total"]:.6f}  (recorded D solo: 0.625651)', flush=True)

    regime3, picks3 = regime_foldoutside_score(D, policy_cols, n_regimes=3)
    print(f'n_regimes=3 (REWS-ratio terciles): {regime3["total"]:.6f}  picks={picks3}', flush=True)
    print(f'delta vs D solo: {regime3["total"] - baseline["total"]:+.6f}', flush=True)

    dep = load_depavg()

    def predictions_for(n_regimes):
        out_rows = []
        for f in FOLDS:
            oth = D[D.fold_id != f]; held = D[D.fold_id == f].copy()
            edges = _bin_edges_per_group(oth, n_regimes)
            oth_bin = _assign_bin(oth, edges, n_regimes)
            held_bin = _assign_bin(held, edges, n_regimes)
            pred = np.empty(len(held))
            for g in (1, 2, 3):
                for r in range(n_regimes):
                    sub = oth[(oth.group_id == g) & (oth_bin == r)]
                    if len(sub) < 100:
                        sub = oth[oth.group_id == g]
                    scores = {c: group_partial_objective(sub.assign(prediction_kwh=sub[c]), g) for c in policy_cols}
                    best_c = max(scores, key=scores.get)
                    sel = (held.group_id == g).to_numpy() & (held_bin == r)
                    pred[sel] = held.loc[sel, best_c].to_numpy()
            held = held.assign(prediction_kwh=pred)
            out_rows.append(held[KEY + ['actual_kwh', 'prediction_kwh']])
        return pd.concat(out_rows, ignore_index=True)

    pred3 = predictions_for(3).rename(columns={'prediction_kwh': 'D_REGIME3'})
    J3 = pred3.merge(dep, on=KEY)
    blend3, blend3_picks = fo_blend_1dof(J3, 'D_REGIME3', 'DEPAVG')
    print(f'DEPAVG + D_REGIME3 fold-outside: {blend3["total"]:.6f}', flush=True)
    print(f'delta vs current_best (0.636184): {blend3["total"] - 0.636184:+.6f}', flush=True)

    regime2, picks2 = regime_foldoutside_score(D, policy_cols, n_regimes=2)
    print(f'n_regimes=2 solo: {regime2["total"]:.6f}', flush=True)
    pred2 = predictions_for(2).rename(columns={'prediction_kwh': 'D_REGIME2'})
    J2 = pred2.merge(dep, on=KEY)
    blend2, blend2_picks = fo_blend_1dof(J2, 'D_REGIME2', 'DEPAVG')
    print(f'DEPAVG + D_REGIME2 fold-outside: {blend2["total"]:.6f}', flush=True)
    print(f'delta vs current_best (0.636184): {blend2["total"] - 0.636184:+.6f}', flush=True)

    picks3_str = {f: {f'g{g}_r{r}': pol for (g, r), pol in p.items()} for f, p in picks3.items()}
    picks2_str = {f: {f'g{g}_r{r}': pol for (g, r), pol in p.items()} for f, p in picks2.items()}
    json.dump({'n_regimes_1_baseline': baseline, 'n_regimes_3': regime3, 'regime_picks_3': picks3_str,
               'depavg_plus_regime3': blend3, 'delta_vs_current_best_regime3': blend3['total'] - 0.636184,
               'n_regimes_2': regime2, 'regime_picks_2': picks2_str,
               'depavg_plus_regime2': blend2, 'delta_vs_current_best_regime2': blend2['total'] - 0.636184},
              open(N + 'S9-N10_regime_policy.json', 'w'), indent=1, default=str)
