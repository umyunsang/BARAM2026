
"""S9-N6 stacking check: does REWS (accepted, foldout_delta +0.001937 in isolation)
still add value on top of G2 (already-accepted S6 grid encoding, screening_delta
+0.001713), or does it turn out to carry mostly the same information once G2 is present?

This project's own history warns that isolated single-axis gains often partially or
fully overlap once stacked (e.g. S10's documented "2-dof blends fall below 1-dof
blends" collapse). Testing G2-only vs G2+REWS, both through the same fold-outside
protocol, isolates REWS's marginal contribution ON TOP OF an already-accepted treatment,
which is a stronger and more decision-relevant test than REWS vs the unaugmented
control (S9-N6's original comparison).
"""
import sys, json
import numpy as np, pandas as pd
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/scratch')
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/nodes')
import harness
from lib import official_total, FOLDS, CAPS, sharpen_weights
from s9_n6_rews_geom import rews_features


def run_fold(A0, COLS0, tr_mask, va_mask):
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
        keep = np.isfinite(cf)
        samples = np.clip(pc[:, None] + sd[:, None] * ztab[g][None, :], 0.0, 1.05)
        mean_gen = float(np.nanmean(A.loc[(A['grp'].to_numpy() == g) & tr_mask, 'cf']))
        err = np.abs(harness.ACTIONS[None, :, None] - samples[:, None, :])
        units = np.where(err <= 0.06, 4.0, np.where(err <= 0.08, 3.0, 0.0))
        rec = {'group_id': g, 'actual_kwh': cf[keep] * CAPS[g]}
        for tp in harness.TEMPS:
            wq = sharpen_weights(samples, tp)[:, None, :]
            nm = -(err * wq).sum(axis=2)
            fi = ((samples[:, None, :] * units) * wq).sum(axis=2) / (4.0 * mean_gen)
            for gm in harness.GAMMAS:
                a = harness.ACTIONS[np.argmax(nm + gm * fi, axis=1)]
                rec[f'T{tp}_G{gm}'] = np.clip(a, 0, 1.05)[keep] * CAPS[g]
        parts.append(pd.DataFrame(rec))
    return pd.concat(parts, ignore_index=True)


def score_variant(label, A0, COLS0):
    per_fold = {}
    for f, (a, b) in FOLDS.items():
        a = pd.Timestamp(a); b = pd.Timestamp(b)
        tr_mask = np.asarray(A0.index < a)
        va_mask = np.asarray((A0.index >= a) & (A0.index <= b))
        per_fold[f] = run_fold(A0, COLS0, tr_mask, va_mask)
        print(f'  [{label}] fold {f} fitted, {per_fold[f].shape[0]} val rows', flush=True)
    policy_cols = [c for c in per_fold[next(iter(FOLDS))].columns if c.startswith('T')]
    out_rows = []; picks = {}
    for f in FOLDS:
        others = pd.concat([per_fold[o] for o in FOLDS if o != f], ignore_index=True)
        scores = {c: official_total(others.assign(prediction_kwh=others[c])[
            ['group_id', 'actual_kwh', 'prediction_kwh']])['total'] for c in policy_cols}
        best_c = max(scores, key=scores.get)
        picks[f] = best_c
        held = per_fold[f]
        out_rows.append(held.assign(prediction_kwh=held[best_c])[
            ['group_id', 'actual_kwh', 'prediction_kwh']])
    D = pd.concat(out_rows, ignore_index=True)
    return official_total(D), picks


if __name__ == '__main__':
    # G2-only baseline (already-accepted S6 treatment: G2 replaces the defective grid__ reshape)
    A_g2, FR_g2, COLS_g2 = harness.surface(('G2', 'DROP:grid__'))
    g2_only, g2_only_picks = score_variant('G2-only', A_g2, COLS_g2)
    print(f'G2-ONLY fold-outside: {g2_only["total"]:.6f} picks={g2_only_picks}', flush=True)

    # G2 + REWS
    feats = rews_features()
    FR = {}
    for g in (1, 2, 3):
        X = FR_g2[g].copy()
        FR[g] = pd.concat([X, feats.reindex(X.index)], axis=1)
    A_stack = pd.concat(FR.values())
    COLS_stack = COLS_g2 + list(feats.columns)
    g2_rews, g2_rews_picks = score_variant('G2+REWS', A_stack, COLS_stack)
    print(f'G2+REWS fold-outside: {g2_rews["total"]:.6f} picks={g2_rews_picks}', flush=True)

    delta = g2_rews['total'] - g2_only['total']
    print(f'MARGINAL DELTA of REWS on top of G2 (fold-outside): {delta:+.6f}', flush=True)
    json.dump({'g2_only': g2_only, 'g2_only_picks': g2_only_picks,
               'g2_rews': g2_rews, 'g2_rews_picks': g2_rews_picks,
               'marginal_delta_foldoutside': delta},
              open('/Users/um-yunsang/BARAM2026/research/nodes/S9-N6_stack_g2.json', 'w'),
              indent=1, default=str)
