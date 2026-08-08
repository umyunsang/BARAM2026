"""S14-N6 * engine nodes F06 (kill-or-open the truncation family) and F11 (transductive normaliser).

F06 -- BINNED SIGNED-RESIDUAL SURFACE ON THE SCORED SUBPOPULATION.
The metric grades only rows with y >= 0.1C, so the graded subpopulation is selected on the
OUTCOME.  Greene ch.19 says a regression fitted on such a subpopulation carries an inverse-Mills
term, and Imbens' treatment says the empirical version of that term is simply
E[y - yhat | yhat, spread, group] restricted to the selected rows.  If that surface is flat there
is nothing to correct and the whole truncation family (F05 included) dies in five minutes; if it
has structure, the correction is estimable.  Gate: the correction is fitted on two folds and
applied to the held-out one, so a shape that is real must survive the fold boundary.

F11 -- TRANSDUCTIVE ESTIMATION OF THE METRIC NORMALISER.
FICR_g = sum(y u) / (4 sum y), so the decision layer needs the graded period's mean capacity
factor.  It currently uses `mean_gen_g`, the mean over the TRAINING window -- a quantity from the
wrong period.  Vapnik's transduction principle and ESL 7.10.2's explicit licence for unsupervised
use of test inputs say we may estimate the graded period's mean from the supplied test x.  We
cannot score that directly (no test labels), so the principle is validated locally instead:
compare three normalisers under the identical fold-outside protocol --
   TRAIN  mean over the training window            (what we deploy today)
   PRED   mean of OUR OWN predictions on the held-out fold's inputs  (transductive, legal)
   ORACLE mean of the held-out fold's true y       (upper bound, not deployable)
If PRED tracks ORACLE and beats TRAIN, the transductive normaliser is justified for delivery.
"""
import sys, json
import numpy as np, pandas as pd
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/scratch')
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/nodes')
from lib import CAPS, FOLDS, official_total
from loop_lib import (canonical_keys, align_prob, load_depavg, fo_policy, KEY, W, ACT, SC,
                      TEMPS, GAMMAS)

N = '/Users/um-yunsang/BARAM2026/research/nodes/'


def frames_with_mg(P, R, mg_vec, temps=TEMPS, gammas=GAMMAS):
    NC = P.shape[1]; C = (np.arange(NC) + 0.5) * W
    err = np.abs(ACT[:, None] - C[None, :])
    units = np.where(err <= 0.06, 4., np.where(err <= 0.08, 3., 0.))
    g = R.group_id.to_numpy()
    capv = np.array([CAPS[x] for x in g]); hi = np.array([SC[x] for x in g])
    mask = (C >= 0.10).astype(float)
    out = {}
    for tp in temps:
        q = P ** (1.0 / tp); q = q / np.maximum(q.sum(1, keepdims=True), 1e-12)
        q = q * mask[None, :]; q = q / np.maximum(q.sum(1, keepdims=True), 1e-12)
        nm = -(q @ err.T); fic = q @ ((C[None, :] * units).T)
        for gm in gammas:
            out[(tp, gm)] = np.minimum(
                ACT[np.argmax(nm + gm * fic / (4.0 * mg_vec[:, None]), axis=1)], hi) * capv
    return out


if __name__ == '__main__':
    R = canonical_keys(); dep = load_depavg()
    P = align_prob('D', R)
    NC = P.shape[1]; C = (np.arange(NC) + 0.5) * W
    y = R.cf.to_numpy(); g = R.group_id.to_numpy()
    mean_q = (P * C[None, :]).sum(1)
    sd_q = np.sqrt((P * (C[None, :] - mean_q[:, None]) ** 2).sum(1))

    # ---------------- F06 ----------------
    print('=== F06: signed-residual surface on the scored subpopulation ===')
    D = pd.DataFrame({'fold_id': R.fold_id, 'group_id': g, 'y': y,
                      'yhat': mean_q, 'spread': sd_q})
    D['resid'] = D.y - D.yhat
    S = D[D.y >= 0.10].copy()
    S['ybin'] = pd.cut(S.yhat, [0, .15, .25, .35, .45, .55, .65, .75, .85, 1.2])
    S['sbin'] = pd.qcut(S.spread, 3, labels=['sharp', 'mid', 'diffuse'])
    tab = S.pivot_table(index='ybin', columns='sbin', values='resid', aggfunc='mean', observed=True)
    cnt = S.pivot_table(index='ybin', columns='sbin', values='resid', aggfunc='size', observed=True)
    print('mean signed residual (y - yhat), scored rows only:')
    print(tab.round(4).to_string())
    print('counts:'); print(cnt.to_string())
    rng = float(tab.max().max() - tab.min().min())
    print(f'\n  surface range = {rng:.4f} of capacity  '
          f'(flat => truncation family dead; our band half-width is 0.06)')

    print('\n  fold-outside test of the correction (fit on two folds, apply to the third):')
    rows = []
    for f in FOLDS:
        oth = S[S.fold_id != f]; held = S[S.fold_id == f].copy()
        corr = oth.groupby(['group_id', 'ybin', 'sbin'], observed=True).resid.median()
        key = list(zip(held.group_id, held.ybin, held.sbin))
        held['adj'] = [corr.get(k, 0.0) for k in key]
        rows.append(held)
    H = pd.concat(rows, ignore_index=True)
    capv = H.group_id.map(CAPS).to_numpy()
    base = H[['group_id']].copy(); base['actual_kwh'] = H.y * capv
    s0 = official_total(base.assign(prediction_kwh=np.clip(H.yhat, 0, 1.1) * capv))
    s1 = official_total(base.assign(prediction_kwh=np.clip(H.yhat + H.adj, 0, 1.1) * capv))
    print(f'    uncorrected point 1-NMAE={s0["one_minus_nmae"]:.6f} Total={s0["total"]:.6f}')
    print(f'    corrected   point 1-NMAE={s1["one_minus_nmae"]:.6f} Total={s1["total"]:.6f}  '
          f'delta={s1["total"]-s0["total"]:+.6f}')

    # ---------------- F11 ----------------
    print('\n=== F11: transductive metric normaliser ===')
    Dm, solo_train, picks = fo_policy(frames_with_mg(P, R, R.mean_gen_g.to_numpy()), R)
    print(f'  TRAIN  normaliser (deployed today): solo fold-outside = {solo_train["total"]:.6f}')
    for name, getter in [('PRED  (transductive, legal)', 'pred'), ('ORACLE (not deployable)', 'orac')]:
        mg = np.empty(len(R))
        for f in FOLDS:
            sel = (R.fold_id == f).to_numpy()
            for gg in (1, 2, 3):
                m = sel & (g == gg)
                if getter == 'pred':
                    v = float(np.mean(mean_q[m]))
                else:
                    v = float(np.nanmean(y[m]))
                mg[m] = v
        _, s, _ = fo_policy(frames_with_mg(P, R, mg), R)
        print(f'  {name}: solo fold-outside = {s["total"]:.6f}  '
              f'delta vs TRAIN = {s["total"]-solo_train["total"]:+.6f}')
    print('\n  normaliser values per fold/group:')
    for f in FOLDS:
        sel = (R.fold_id == f).to_numpy()
        tr_ = {gg: round(float(R.mean_gen_g.to_numpy()[sel & (g == gg)][0]), 4) for gg in (1, 2, 3)}
        pr_ = {gg: round(float(np.mean(mean_q[sel & (g == gg)])), 4) for gg in (1, 2, 3)}
        or_ = {gg: round(float(np.nanmean(y[sel & (g == gg)])), 4) for gg in (1, 2, 3)}
        print(f'    {f}: train={tr_}  pred={pr_}  oracle={or_}')
    json.dump({'f06_surface_range': rng, 'f06_delta': s1['total'] - s0['total']},
              open(N + 'S14-N6_f06_f11.json', 'w'), indent=1, default=str)
