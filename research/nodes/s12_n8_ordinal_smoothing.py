"""S12-N8 * impose ordinal structure on the 26-class predictive distribution.

The member `D` is a multiclass softmax over 26 capacity-factor bins.  Multiclass cross-entropy
treats those 26 labels as unordered: bin 12 and bin 13 are as unrelated as bin 12 and bin 25.
But capacity factor is an ordered quantity and the decision layer integrates q against a
+-0.06 window, i.e. against roughly three adjacent bins.  Any leakage of probability mass into
the wrong *ordering* directly corrupts that window integral.

S12-N1a already produced a suggestive clue for this: refining the utility integral to sub-bin
resolution (subbin=4, a strictly more accurate quadrature) made the score WORSE, which only
makes sense if the coarse bin-centre quadrature was accidentally acting as a smoother.  This
node replaces that accident with an explicit, tunable one.

Treatment: convolve q along the class axis with a normalised kernel before the decision layer.
Three kernel families, each with a single width parameter chosen fold-outside:
  box(w)      uniform over +-w bins
  tri(w)      triangular over +-w bins
  gauss(s)    discrete Gaussian, sd s bins
w=0 / s=0 reproduces current_best exactly, so the treatment is nested and the fold-outside gate
can reject it back to baseline.
"""
import sys, json
import numpy as np, pandas as pd
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/nodes')
from loop_lib import *

N = '/Users/um-yunsang/BARAM2026/research/nodes/'


def kernel(kind, p, NC=26):
    if p == 0:
        return np.eye(NC)
    idx = np.arange(NC)
    d = idx[:, None] - idx[None, :]
    if kind == 'box':
        K = (np.abs(d) <= p).astype(float)
    elif kind == 'tri':
        K = np.maximum(0.0, 1.0 - np.abs(d) / (p + 1.0))
    elif kind == 'gauss':
        K = np.exp(-0.5 * (d / p) ** 2)
    else:
        raise ValueError(kind)
    return K / K.sum(axis=1, keepdims=True)


def smooth(P, kind, p):
    if p == 0:
        return P
    Q = P @ kernel(kind, p).T
    return Q / np.maximum(Q.sum(axis=1, keepdims=True), 1e-12)


if __name__ == '__main__':
    R = canonical_keys(); dep = load_depavg()
    PD = align_prob('D', R)
    out = {}

    GRID = [('box', 0)] + [(k, p) for k in ('box', 'tri') for p in (1, 2, 3)] \
           + [('gauss', s) for s in (0.5, 0.75, 1.0, 1.5, 2.0)]
    print('--- in-sample sweep (diagnostic; the fold-outside gate below is the claim) ---')
    cache = {}
    for kind, p in GRID:
        Q = smooth(PD, kind, p)
        r, Dm, J = evaluate_prob(Q, R, tag=f'{kind}({p})', dep=dep)
        out[f'{kind}_{p}'] = r
        cache[(kind, p)] = (Dm, J)

    # ---- proper fold-outside gate over the whole (kernel, width) grid -------------------
    # Build every candidate's blended action, then choose the kernel on the other two folds.
    print('\n--- fold-outside kernel selection (1 dof over the kernel grid) ---')
    blends = {}
    for (kind, p), (Dm, J) in cache.items():
        # recompute the blend action per fold-outside weight, keep the per-row action
        rows = []
        for f in FOLDS:
            oth = J[J.fold_id != f]; held = J[J.fold_id == f]; best = None
            for w in np.arange(0, 1.001, 0.05):
                t = official_total(oth.assign(prediction_kwh=w * oth.MINE + (1 - w) * oth.DEPAVG)[
                    ['group_id', 'actual_kwh', 'prediction_kwh']])['total']
                if best is None or t > best[0]:
                    best = (t, w)
            rows.append(held.assign(pred=best[1] * held.MINE + (1 - best[1]) * held.DEPAVG))
        blends[(kind, p)] = pd.concat(rows, ignore_index=True)

    ref = blends[('box', 0)]
    rows = []; picks = {}
    for f in FOLDS:
        scores = {}
        for key, B in blends.items():
            oth = B[B.fold_id != f]
            scores[key] = official_total(oth.assign(prediction_kwh=oth.pred)[
                ['group_id', 'actual_kwh', 'prediction_kwh']])['total']
        bk = max(scores, key=scores.get); picks[f] = bk
        held = blends[bk]
        rows.append(held[held.fold_id == f].assign(prediction_kwh=held[held.fold_id == f].pred))
    G = pd.concat(rows, ignore_index=True)
    s = official_total(G[['group_id', 'actual_kwh', 'prediction_kwh']])
    print(f'  fold-outside kernel picks={ {k: str(v) for k, v in picks.items()} }')
    print(f'  gated total={s["total"]:.6f}  d_vs_current_best={s["total"]-0.6361842493883538:+.6f}  '
          f'(1-NMAE={s["one_minus_nmae"]:.6f} FICR={s["ficr"]:.6f})')
    out['foldoutside_gate'] = {'total': s['total'], 'picks': {k: str(v) for k, v in picks.items()},
                               'one_minus_nmae': s['one_minus_nmae'], 'ficr': s['ficr']}
    json.dump(out, open(N + 'S12-N8_ordinal_smoothing.json', 'w'), indent=1, default=str)
