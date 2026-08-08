"""S14-N8 (engine nodes F01 + F04) * pool the DENSITIES, then take ONE Bayes action.

The defect this targets.  The champion is 0.30*D + 0.70*DEPAVG, and DEPAVG is itself the mean of
three deployed actions -- so the deployed number is an average of FOUR ACTIONS.  Murphy 5.7 says
the Bayes action is a functional argmax over p(y|x); averaging four argmaxes is not that
functional, and under a step reward it is not even close: the mean of four band-seeking actions
generally sits between bands and hits none of them.  S14-N4 showed our search is exact, so this
is the one remaining structural error in the deployed pipeline.

Why this is not S12-N1.  S12-N1 averaged MY members' densities against each other and blended the
result with DEPAVG at the action level -- it never removed the action-averaging step, because the
deployed members expose actions, not densities.  Here the deployed actions are lifted back into
density space with an explicit kernel and pooled with D's density BEFORE the single argmax:

    q_pool(c) = w * q_D(c) + (1-w) * (1/3) * sum_m K_h(c - a_m)

with K a discrete kernel of width h over the 26-bin grid.  h -> 0 recovers point masses at the
deployed actions; h large flattens them into the pool.  Both w and h are chosen fold-outside, so
this is 2 dof against the champion's 1, and the arbiter's paired bootstrap decides.

F04 (sharpness routing) rides along on the same arrays: partition scored rows by predictive
spread and report where the achievable FICR actually lives.  If band-hitting is concentrated on a
minority of rows, that reframes what any future node should target.
"""
import sys, json
import numpy as np, pandas as pd
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/scratch')
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/nodes')
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/engine')
from lib import CAPS, FOLDS, official_total
from loop_lib import (canonical_keys, align_prob, load_depavg, utility_frames, fo_policy,
                      KEY, W, ACT, SC, TEMPS, GAMMAS, DEP, AB)

N = '/Users/um-yunsang/BARAM2026/research/nodes/'


def kernel_from_actions(a_cf, NC, h):
    """Lift an action (in capacity-factor units) to a discrete density over the 26 bins."""
    C = (np.arange(NC) + 0.5) * W
    d = np.abs(C[None, :] - a_cf[:, None])
    if h <= 0:
        K = (d <= W / 2 + 1e-9).astype(float)
    else:
        K = np.exp(-0.5 * (d / h) ** 2)
    s = K.sum(1, keepdims=True)
    return K / np.maximum(s, 1e-12)


def bayes_action(q, R, tp, gm):
    NC = q.shape[1]; C = (np.arange(NC) + 0.5) * W
    err = np.abs(ACT[:, None] - C[None, :])
    units = np.where(err <= 0.06, 4., np.where(err <= 0.08, 3., 0.))
    mask = (C >= 0.10).astype(float)
    g = R.group_id.to_numpy(); mg = R.mean_gen_g.to_numpy()
    capv = np.array([CAPS[x] for x in g]); hi = np.array([SC[x] for x in g])
    p = q ** (1.0 / tp); p = p / np.maximum(p.sum(1, keepdims=True), 1e-12)
    p = p * mask[None, :]; p = p / np.maximum(p.sum(1, keepdims=True), 1e-12)
    nm = -(p @ err.T); fic = p @ ((C[None, :] * units).T)
    return np.minimum(ACT[np.argmax(nm + gm * fic / (4.0 * mg[:, None]), axis=1)], hi) * capv


if __name__ == '__main__':
    R = canonical_keys(); dep = load_depavg()
    P = align_prob('D', R); NC = P.shape[1]
    g = R.group_id.to_numpy(); capv = np.array([CAPS[x] for x in g])
    y = R.cf.to_numpy()

    # deployed members' actions at their documented policies, aligned to R
    parts = {}
    for stem, pol in DEP.items():
        fr = []
        for f in FOLDS:
            d = pd.read_parquet(AB + f'{stem}-{f}-policies.parquet').copy(); d['fold_id'] = f
            fr.append(d[KEY + [pol]].rename(columns={pol: stem}))
        parts[stem] = pd.concat(fr, ignore_index=True)
    J = R[KEY].copy()
    for stem, d in parts.items():
        J = J.merge(d, on=KEY, how='left')
    A_dep = J[list(DEP)].to_numpy(float) / capv[:, None]
    ok = np.isfinite(A_dep).all(1)
    print(f'rows with all three deployed actions: {ok.sum()} / {len(R)}')

    base = pd.DataFrame({'fold_id': R.fold_id, 'group_id': g, 'actual_kwh': y * capv})
    champ = 0.30 * fo_policy(utility_frames(P, R), R)[0].prediction_kwh.to_numpy() \
        + 0.70 * dep.merge(R[KEY], on=KEY, how='right').DEPAVG.to_numpy()
    s_champ = official_total(base.assign(prediction_kwh=champ))
    print(f'champion reproduced: {s_champ["total"]:.6f}')

    # ---------------- F01: pooled density, single Bayes action --------------------------
    print('\n=== F01: pool densities then ONE argmax (fold-outside over w, h, T, G) ===')
    GRID_W = [0.2, 0.3, 0.4, 0.5, 0.6]
    GRID_H = [0.0, 0.02, 0.04, 0.08]
    cache = {}
    for h in GRID_H:
        Kd = np.zeros((len(R), NC))
        for m in range(3):
            a = np.where(ok, A_dep[:, m], y)
            Kd += kernel_from_actions(np.clip(a, 0, 1.05), NC, h)
        Kd /= 3.0
        for w in GRID_W:
            q = w * P + (1 - w) * Kd
            q = q / np.maximum(q.sum(1, keepdims=True), 1e-12)
            for tp in TEMPS:
                for gm in GAMMAS:
                    cache[(h, w, tp, gm)] = bayes_action(q, R, tp, gm)
    rows = []; picks = {}
    for f in FOLDS:
        sel = (R.fold_id == f).to_numpy()
        sc = {}
        for k, v in cache.items():
            sc[k] = official_total(base[~sel].assign(prediction_kwh=v[~sel]))['total']
        bk = max(sc, key=sc.get); picks[f] = bk
        rows.append(base[sel].assign(prediction_kwh=cache[bk][sel]))
    Dp = pd.concat(rows, ignore_index=True)
    s_pool = official_total(Dp)
    print(f'  fold-outside picks (h,w,T,G): { {k: str(v) for k, v in picks.items()} }')
    print(f'  pooled-Bayes total = {s_pool["total"]:.6f}  '
          f'(1-NMAE={s_pool["one_minus_nmae"]:.6f} FICR={s_pool["ficr"]:.6f})')
    print(f'  delta vs champion  = {s_pool["total"]-s_champ["total"]:+.6f}')

    # arbitrate properly
    from arbiter import arbitrate
    cmp = base.copy()
    cmp['forecast_kst_dtm'] = R.forecast_kst_dtm.to_numpy()
    cmp['champ'] = champ
    cmp['cand'] = pd.concat(rows, ignore_index=True).sort_index().prediction_kwh.to_numpy() \
        if False else np.concatenate([cache[picks[f]][(R.fold_id == f).to_numpy()] for f in FOLDS])
    order = np.concatenate([np.where((R.fold_id == f).to_numpy())[0] for f in FOLDS])
    cmp = cmp.iloc[order].reset_index(drop=True)
    cmp['cand'] = np.concatenate([cache[picks[f]][(R.fold_id == f).to_numpy()] for f in FOLDS])
    took, arb = arbitrate(cmp, 'cand', 'champ', n_comparisons=1)
    print(f'  ARBITER: delta={arb["point_delta"]:+.6f} paired_sd={arb["paired_sd"]:.6f} '
          f'P(better)={arb["p_better"]:.3f} -> {"CHAMPION" if took else "rejected"}')

    # ---------------- F04: where does achievable FICR actually live? --------------------
    print('\n=== F04: sharpness routing diagnostic ===')
    C = (np.arange(NC) + 0.5) * W
    mq = (P * C[None, :]).sum(1); sdq = np.sqrt((P * (C[None, :] - mq[:, None]) ** 2).sum(1))
    sc_rows = y >= 0.10
    D2 = pd.DataFrame({'sd': sdq, 'y': y, 'a': champ / capv, 'g': g})[sc_rows]
    D2['bin'] = pd.qcut(D2.sd, 5, labels=['q1 sharpest', 'q2', 'q3', 'q4', 'q5 most diffuse'])
    D2['hit6'] = (np.abs(D2.a - D2.y) <= 0.06).astype(float)
    D2['unit'] = np.select([np.abs(D2.a - D2.y) <= 0.06, np.abs(D2.a - D2.y) <= 0.08], [4., 3.], 0.)
    t = D2.groupby('bin', observed=True).apply(
        lambda s: pd.Series({'n': len(s), 'mean_sd': s.sd.mean(), 'hit6': s.hit6.mean(),
                             'prod_share': s.y.sum() / D2.y.sum(),
                             'ficr_realised': (s.y * s.unit).sum() / (D2.y * 4).sum(),
                             'ficr_available': (s.y * 4).sum() / (D2.y * 4).sum()}), include_groups=False)
    t['capture'] = t.ficr_realised / t.ficr_available
    print(t.round(4).to_string())
    print(f'\n  FICR realised in the sharpest two quintiles: '
          f'{t.ficr_realised.iloc[:2].sum()/t.ficr_realised.sum():.1%} of all realised FICR')
    json.dump({'pooled_bayes_total': s_pool['total'], 'champion': s_champ['total'],
               'arbitration': arb, 'sharpness': t.reset_index().astype(str).to_dict('records')},
              open(N + 'S14-N8_pooled_bayes.json', 'w'), indent=1, default=str)
