"""S14-N4 (engine node F03) * Ng's Optimization Verification test.

The question this settles.  Thirty treatments have failed to move the champion.  There are only
two possible explanations and they demand opposite responses:
   SEARCH AT FAULT   -- our objective is right but our optimiser/model class cannot find the
                        action it prefers.  Response: better models, better search.
   OBJECTIVE AT FAULT-- our optimiser is doing its job, but the objective it optimises ranks a
                        worse action above a better one.  Response: change the objective; more
                        model work is wasted.

Ng's test (Machine Learning Yearning ch.44-45) separates them in one measurement.  For each
scored row take the action our system deploys, a_ours, and the action that would actually have
maximised that row's contribution to the official score, a_star (computable because we know y in
the development period).  Then evaluate OUR OWN training/decision objective J at both.

   if J(a_ours) >= J(a_star) often  -> the optimiser found what it was told to find, and what it
                                       was told to find is wrong.  OBJECTIVE at fault.
   if J(a_ours) <  J(a_star) often  -> the objective correctly prefers the better action but our
                                       search failed to reach it.  SEARCH at fault.

J is taken to be exactly the decision layer's own expected-utility surface under the member's
predictive distribution q, at the (T,G) the fold-outside gate selected -- i.e. the objective the
deployed system actually maximises, not a reconstruction.
"""
import sys, json
import numpy as np, pandas as pd
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/scratch')
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/nodes')
from lib import CAPS, FOLDS, official_total
from loop_lib import (canonical_keys, align_prob, utility_frames, fo_policy, load_depavg,
                      KEY, W, ACT, SC)

N = '/Users/um-yunsang/BARAM2026/research/nodes/'

if __name__ == '__main__':
    R = canonical_keys()
    P = align_prob('D', R)
    NC = P.shape[1]; C = (np.arange(NC) + 0.5) * W
    g = R.group_id.to_numpy(); mg = R.mean_gen_g.to_numpy()
    capv = np.array([CAPS[x] for x in g]); hi = np.array([SC[x] for x in g])
    y = R.cf.to_numpy()

    frames = utility_frames(P, R)
    Dm, solo, picks = fo_policy(frames, R)
    print('deployed fold-outside policy picks:', {k: str(v) for k, v in picks.items()})

    err = np.abs(ACT[:, None] - C[None, :])
    units = np.where(err <= 0.06, 4., np.where(err <= 0.08, 3., 0.))
    mask = (C >= 0.10).astype(float)

    # J(a) for every row and every candidate action, under the deployed (T,G) of that row's fold
    J = np.empty((len(R), len(ACT)))
    for f, (tp, gm) in picks.items():
        sel = (R.fold_id == f).to_numpy()
        q = P[sel] ** (1.0 / tp); q = q / np.maximum(q.sum(1, keepdims=True), 1e-12)
        q = q * mask[None, :]; q = q / np.maximum(q.sum(1, keepdims=True), 1e-12)
        nm = -(q @ err.T)
        fic = q @ ((C[None, :] * units).T)
        J[sel] = nm + gm * fic / (4.0 * mg[sel][:, None])

    # a_ours: what the system deploys
    a_ours_kwh = Dm.prediction_kwh.to_numpy()
    a_ours = a_ours_kwh / capv
    # a_star: the action maximising THIS ROW's true contribution to the official score
    e_true = np.abs(ACT[None, :] - y[:, None])
    u_true = np.where(e_true <= 0.06, 4., np.where(e_true <= 0.08, 3., 0.))
    contrib = -e_true + (y[:, None] * u_true) / (4.0 * mg[:, None])
    a_star_idx = np.argmax(contrib, axis=1)
    a_star = np.minimum(ACT[a_star_idx], hi)

    scored = y >= 0.10
    i_ours = np.abs(ACT[None, :] - a_ours[:, None]).argmin(axis=1)
    J_ours = J[np.arange(len(R)), i_ours]
    J_star = J[np.arange(len(R)), a_star_idx]
    prefers_ours = (J_ours >= J_star - 1e-12)

    print(f'\nscored rows: {int(scored.sum())} / {len(R)}')
    frac = float(prefers_ours[scored].mean())
    print(f'\n=== OPTIMIZATION VERIFICATION TEST ===')
    print(f'  our own objective J prefers the DEPLOYED action over the truly best action')
    print(f'  on {frac:.1%} of scored rows')
    print(f'  -> {"OBJECTIVE at fault" if frac > 0.5 else "SEARCH at fault"}')

    gap = (J_star - J_ours)[scored]
    print(f'\n  J(a*) - J(a_ours): mean={gap.mean():+.6f}  median={np.median(gap):+.6f}  '
          f'share>0 = {float((gap>1e-12).mean()):.3f}')
    d = np.abs(a_star - a_ours)[scored]
    print(f'  |a* - a_ours| (capacity fraction): mean={d.mean():.4f}  median={np.median(d):.4f}')

    # what would we score if we could deploy a_star? (an oracle upper bound, for calibration)
    base = pd.DataFrame({'group_id': g, 'actual_kwh': y * capv})
    s_star = official_total(base.assign(prediction_kwh=a_star * capv))
    s_ours = official_total(base.assign(prediction_kwh=a_ours_kwh))
    print(f'\n  Total with a_ours = {s_ours["total"]:.6f}   with oracle a* = {s_star["total"]:.6f}')

    # decomposition: among rows where our objective prefers our own worse action, is it because
    # q is wrong (density) or because the (T,G) surrogate is wrong (objective shape)?
    q_at_star = np.take_along_axis(P, np.clip((a_star / W).astype(int), 0, NC - 1)[:, None], 1).ravel()
    q_at_ours = np.take_along_axis(P, np.clip((a_ours / W).astype(int), 0, NC - 1)[:, None], 1).ravel()
    m = scored & prefers_ours
    print(f'\n  on the {int(m.sum())} rows where the objective prefers our action:')
    print(f'    mean q at a*    = {q_at_star[m].mean():.4f}')
    print(f'    mean q at a_ours= {q_at_ours[m].mean():.4f}')
    print(f'    -> the density assigns {q_at_ours[m].mean()/max(q_at_star[m].mean(),1e-9):.2f}x more '
          f'mass to our action than to the truly better one')
    json.dump({'prefers_ours_share': frac, 'verdict': 'OBJECTIVE' if frac > 0.5 else 'SEARCH',
               'mean_J_gap': float(gap.mean()), 'total_ours': s_ours['total'],
               'total_oracle_action': s_star['total'],
               'q_at_ours': float(q_at_ours[m].mean()), 'q_at_star': float(q_at_star[m].mean())},
              open(N + 'S14-N4_optverify.json', 'w'), indent=1, default=str)
