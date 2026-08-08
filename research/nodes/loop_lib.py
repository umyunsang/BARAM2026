
"""S12 loop library: fast, fit-free evaluation over the saved 26-class member probabilities.

Everything here reads only saved artifacts (S7-N8_*_{keys.parquet,prob.npy}) and the
deployed metric-aligned-probe policy frames, so a full candidate evaluation costs seconds
instead of a re-fit. Semantics are a line-for-line port of research/nodes/s10_final3.py
(the script that produced current_best = 0.636184) so any delta measured here is
attributable to the declared treatment only.
"""
from __future__ import annotations
import sys
import numpy as np, pandas as pd

sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/scratch')
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/nodes')
from lib import official_total, FOLDS, CAPS

N = '/Users/um-yunsang/BARAM2026/research/nodes/'
AB = '/Users/um-yunsang/BARAM2026/artifacts/backtests/metric-aligned-probe/'
KEY = ['fold_id', 'group_id', 'forecast_kst_dtm']
W = 0.04
ACT = np.arange(0.02, 1.0801, 0.0025)
SC = {1: 0.985, 2: 0.989, 3: 1.005}
TEMPS = [0.6, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
GAMMAS = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0]
DEP = {'M102_TOP100': 'T0.5_G1.5', 'M113_LGBM_DART': 'T0.5_G0.5', 'M115_XGBOOST': 'T0.6_G0.35'}
ALL_MEMBERS = ['P', 'L', 'G', 'Q', 'M', 'D', 'X', 'M2', 'R', 'R1', 'R2', 'LV', 'S1', 'W2', 'XG']

_pcache = {}


def load_prob(name):
    """Returns (keys_frame, prob_matrix) with the canonical row order for `name`."""
    if name in _pcache:
        return _pcache[name]
    R = pd.read_parquet(N + f'S7-N8_{name}_keys.parquet')
    P = np.load(N + f'S7-N8_{name}_prob.npy')
    _pcache[name] = (R, P)
    return R, P


def canonical_keys():
    R, _ = load_prob('D')
    return R


def align_prob(name, ref=None):
    """Prob matrix reindexed onto the canonical (D) key order."""
    ref = canonical_keys() if ref is None else ref
    R, P = load_prob(name)
    if R[KEY].equals(ref[KEY]):
        return P
    order = (R.reset_index()
               .merge(ref[KEY].reset_index().rename(columns={'index': 'pos'}), on=KEY)
               .sort_values('pos')['index'].to_numpy())
    return P[order]


def utility_frames(P, R, temps=TEMPS, gammas=GAMMAS, subbin=1, floor=0.10):
    """(temp, gamma) -> per-row kwh action, from a 26-class probability matrix.

    subbin=1 reproduces s10_final3 exactly (all class mass at the bin centre).
    subbin>1 spreads each class's mass uniformly across its bin width, which makes the
    expected band-hit integral resolve the +-0.06/+-0.08 step edges instead of quantising
    them to the 0.04 grid. 0 fitted dof either way.
    """
    NC = P.shape[1]
    if subbin <= 1:
        C = (np.arange(NC) + 0.5) * W
        Wt = np.eye(NC)
    else:
        off = (np.arange(subbin) + 0.5) / subbin
        C = ((np.arange(NC)[:, None] + off[None, :]) * W).ravel()
        Wt = np.repeat(np.eye(NC), subbin, axis=1) / subbin
    err = np.abs(ACT[:, None] - C[None, :])
    units = np.where(err <= 0.06, 4., np.where(err <= 0.08, 3., 0.))
    g = R.group_id.to_numpy(); mg = R.mean_gen_g.to_numpy()
    capv = np.array([CAPS[x] for x in g]); hi = np.array([SC[x] for x in g])
    mask = (C >= floor).astype(float)
    frames = {}
    for tp in temps:
        q = P ** (1.0 / tp)
        q = q / np.maximum(q.sum(axis=1, keepdims=True), 1e-12)
        qf = q @ Wt
        qf = qf * mask[None, :]
        qf = qf / np.maximum(qf.sum(axis=1, keepdims=True), 1e-12)
        nm = -(qf @ err.T)
        fic = qf @ ((C[None, :] * units).T)
        for gm in gammas:
            frames[(tp, gm)] = np.minimum(ACT[np.argmax(nm + gm * fic / (4.0 * mg[:, None]), axis=1)], hi) * capv
    return frames


def fo_policy(frames, R):
    """Fold-outside policy selection: pick (T,G) on the other two folds, apply to held-out."""
    D = pd.DataFrame({'fold_id': R.fold_id.to_numpy(), 'group_id': R.group_id.to_numpy(),
                      'forecast_kst_dtm': R.forecast_kst_dtm.to_numpy(),
                      'actual_kwh': R.cf.to_numpy() * R.group_id.map(CAPS).to_numpy()})
    out = np.empty(len(D)); picks = {}
    for f in FOLDS:
        sel = (D.fold_id == f).to_numpy()
        s2 = {k: official_total(D[~sel].assign(prediction_kwh=v[~sel])[
                  ['group_id', 'actual_kwh', 'prediction_kwh']])['total'] for k, v in frames.items()}
        bk = max(s2, key=s2.get); picks[f] = bk
        out[sel] = frames[bk][sel]
    D['prediction_kwh'] = out
    return D, official_total(D[['group_id', 'actual_kwh', 'prediction_kwh']]), picks


def load_depavg():
    fr = []
    for stem, pol in DEP.items():
        parts = []
        for f in FOLDS:
            d = pd.read_parquet(AB + f'{stem}-{f}-policies.parquet').copy()
            d['fold_id'] = f
            parts.append(d[KEY + [pol]].rename(columns={pol: stem}))
        fr.append(pd.concat(parts, ignore_index=True))
    J = fr[0]
    for x in fr[1:]:
        J = J.merge(x, on=KEY)
    J['DEPAVG'] = J[list(DEP)].mean(axis=1)
    return J[KEY + ['DEPAVG']]


def fo_blend_1dof(J, col_a, col_b, step=0.05):
    grid = np.arange(0, 1.0001, step)
    rows = []; picks = {}
    for f in FOLDS:
        oth = J[J.fold_id != f]; held = J[J.fold_id == f]
        best = None
        for wa in grid:
            pred = wa * oth[col_a] + (1 - wa) * oth[col_b]
            t = official_total(oth.assign(prediction_kwh=pred)[
                ['group_id', 'actual_kwh', 'prediction_kwh']])['total']
            if best is None or t > best[0]:
                best = (t, wa)
        picks[f] = float(best[1])
        rows.append(held.assign(prediction_kwh=best[1] * held[col_a] + (1 - best[1]) * held[col_b]))
    D = pd.concat(rows, ignore_index=True)
    return official_total(D[['group_id', 'actual_kwh', 'prediction_kwh']]), picks, D


def evaluate_prob(P, R=None, tag='', subbin=1, temps=TEMPS, gammas=GAMMAS, floor=0.10, dep=None, verbose=True):
    """Full pipeline for one probability matrix: solo fold-outside + DEPAVG 1-dof blend."""
    R = canonical_keys() if R is None else R
    frames = utility_frames(P, R, temps=temps, gammas=gammas, subbin=subbin, floor=floor)
    Dm, solo, picks = fo_policy(frames, R)
    dep = load_depavg() if dep is None else dep
    J = Dm[KEY + ['actual_kwh', 'prediction_kwh']].rename(columns={'prediction_kwh': 'MINE'}).merge(dep, on=KEY)
    bl, wpick, _ = fo_blend_1dof(J, 'MINE', 'DEPAVG')
    res = dict(tag=tag, solo=solo['total'], solo_1mnmae=solo['one_minus_nmae'], solo_ficr=solo['ficr'],
               blend=bl['total'], blend_1mnmae=bl['one_minus_nmae'], blend_ficr=bl['ficr'],
               policy_picks={k: str(v) for k, v in picks.items()}, weight_picks=wpick,
               delta_vs_current_best=bl['total'] - 0.6361842493883538)
    if verbose:
        print(f'{tag:34s} solo={res["solo"]:.6f}  blend={res["blend"]:.6f}  '
              f'd_vs_best={res["delta_vs_current_best"]:+.6f}  (1-NMAE={res["blend_1mnmae"]:.6f} FICR={res["blend_ficr"]:.6f})',
              flush=True)
    return res, Dm, J
