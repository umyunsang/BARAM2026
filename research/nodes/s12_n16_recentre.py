"""S12-N16 * translate the exploitable distribution onto the accurate centre.

Two facts measured this session pull in opposite directions.

  (i) Point accuracy.  Training on the rows the metric scores lifts the point forecast:
      member DV's distribution mean scores 1-NMAE 0.866061 against D's 0.862828 (+0.003233),
      and the standalone bake-off's best configuration reaches 0.866147 -- all above the
      0.864617 that was the best 1-NMAE obtainable from any pre-existing artifact.

  (ii) Distribution shape.  D's decision layer converts its distribution into +0.011168 of
      Total over its own point forecast (0.625669 vs 0.614501), while DV's converts only
      +0.002710 (0.624517 vs 0.621807).  Conditioning the training set on cf >= 0.1 sharpens
      the centre but destroys the tail structure the band-seeking argmax feeds on.

So the accurate centre and the exploitable shape currently live in different members.  This
node puts them together with a zero-parameter operation: translate D's 26-class distribution
along the capacity-factor axis so that its mean lands on the better point estimate, leaving
its shape untouched, then run the identical decision layer and fold-outside policy gate.

  q'(c) = q(c - lambda * delta),  delta = point_accurate - mean(q),
  realised by linear interpolation across the 0.04-wide bins (mass conserved, no renormalising
  trick that could silently sharpen the distribution).

lambda = 0 reproduces current_best exactly and lambda = 1 is full recentring, so the treatment
is nested and the fold-outside gate can reject it to baseline.  Three centres are tested:
DV's mean, the S12-N12 bake-off ensemble AVGQ, and its L1P member.
"""
import sys, json
import numpy as np, pandas as pd
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/nodes')
from loop_lib import *

N = '/Users/um-yunsang/BARAM2026/research/nodes/'
C26 = (np.arange(26) + 0.5) * W


def translate(P, delta):
    """Shift each row's discrete distribution by delta (in cf units) with linear interpolation."""
    shift = delta / W
    lo = np.floor(shift).astype(int)
    frac = (shift - lo)[:, None]
    n, k = P.shape
    out = np.zeros_like(P)
    idx = np.arange(k)[None, :]
    for part, off in ((1.0 - frac, 0), (frac, 1)):
        tgt = idx + lo[:, None] + off
        np.clip(tgt, 0, k - 1, out=tgt)
        np.add.at(out, (np.repeat(np.arange(n), k), tgt.ravel()), (part * P).ravel())
    return out / np.maximum(out.sum(axis=1, keepdims=True), 1e-12)


def load_bakeoff_point(R):
    K = pd.read_parquet(N + 'S12-N12_keys.parquet')
    A = np.load(N + 'S12-N12_point_preds.npy')
    order = json.load(open(N + 'S12-N12_point_bakeoff.json'))['order']
    K = K.copy()
    for i, nm in enumerate(order):
        K[nm] = A[i]
    ref = R[KEY].copy()
    M = ref.merge(K[KEY + order], on=KEY, how='left')
    return {nm: M[nm].to_numpy() for nm in order}


if __name__ == '__main__':
    R = canonical_keys(); dep = load_depavg()
    PD = align_prob('D', R)
    meanD = (PD * C26[None, :]).sum(axis=1)
    centres = {}
    centres['DV'] = (align_prob('DV', R) * C26[None, :]).sum(axis=1)
    bo = load_bakeoff_point(R)
    for nm in ('AVGQ', 'L1P'):
        if nm in bo:
            centres[nm] = bo[nm]

    out = {'baseline': 0.6361842493883538}
    cands = {}
    print('--- in-sample lambda sweep per centre (diagnostic) ---')
    for cn, cv in centres.items():
        ok = np.isfinite(cv)
        d0 = np.where(ok, cv - meanD, 0.0)
        print(f'  centre {cn}: n_finite={int(ok.sum())}/{len(cv)}  mean|delta|={np.abs(d0).mean():.4f}')
        for lam in (0.0, 0.25, 0.5, 0.75, 1.0):
            Q = PD if lam == 0 else translate(PD, lam * d0)
            r, Dm, J = evaluate_prob(Q, R, tag=f'{cn} lam={lam}', dep=dep)
            out[f'{cn}_lam{lam}'] = r
            cands[(cn, lam)] = J

    # ---- fold-outside gate over the whole (centre, lambda) grid --------------------------
    print('\n--- fold-outside gate over (centre, lambda) ---')
    blends = {}
    for key, J in cands.items():
        rows = []
        for f in FOLDS:
            oth = J[J.fold_id != f]; held = J[J.fold_id == f]; best = None
            for w in np.arange(0, 1.001, 0.05):
                t = official_total(oth.assign(prediction_kwh=w * oth.MINE + (1 - w) * oth.DEPAVG)[
                    ['group_id', 'actual_kwh', 'prediction_kwh']])['total']
                if best is None or t > best[0]:
                    best = (t, w)
            rows.append(held.assign(pred=best[1] * held.MINE + (1 - best[1]) * held.DEPAVG))
        blends[key] = pd.concat(rows, ignore_index=True)

    rows = []; picks = {}
    for f in FOLDS:
        sc = {}
        for key, B in blends.items():
            oth = B[B.fold_id != f]
            sc[key] = official_total(oth.assign(prediction_kwh=oth.pred)[
                ['group_id', 'actual_kwh', 'prediction_kwh']])['total']
        bk = max(sc, key=sc.get); picks[f] = bk
        h = blends[bk]
        rows.append(h[h.fold_id == f].assign(prediction_kwh=h[h.fold_id == f].pred))
    G = pd.concat(rows, ignore_index=True)
    s = official_total(G[['group_id', 'actual_kwh', 'prediction_kwh']])
    print(f'  picks={ {k: str(v) for k, v in picks.items()} }')
    print(f'  gated total={s["total"]:.6f}  d_vs_current_best={s["total"]-0.6361842493883538:+.6f}  '
          f'(1-NMAE={s["one_minus_nmae"]:.6f} FICR={s["ficr"]:.6f})')
    out['foldoutside_gate'] = {'total': s['total'], 'picks': {k: str(v) for k, v in picks.items()},
                               'one_minus_nmae': s['one_minus_nmae'], 'ficr': s['ficr']}
    json.dump(out, open(N + 'S12-N16_recentre.json', 'w'), indent=1, default=str)
