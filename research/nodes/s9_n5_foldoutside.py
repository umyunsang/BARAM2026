
"""S9-N5 fold-outside gate for the veer/jet_below_100 (terrain candidate F, trimmed)
screening result. Mirrors s9_n4_foldoutside.py's structure -- veer_features() is also
purely deterministic (angle arithmetic on raw wind components, nothing fit from data),
so no per-fold refit of the feature itself is needed, only of the teacher/calibrator.
"""
import sys, json
import numpy as np, pandas as pd
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/src')
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/scratch')
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/nodes')
import harness
from lib import official_total, FOLDS, CAPS, sharpen_weights
from s9_n5_pbl_veer import veer_features


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
    A0, FR0, COLS0 = harness.surface(())
    ctrl, ctrl_picks = score_variant('control', A0, COLS0)
    print(f'CONTROL fold-outside: {ctrl["total"]:.6f} picks={ctrl_picks}', flush=True)

    feats = veer_features()
    FR = {}
    for g in (1, 2, 3):
        X = FR0[g].copy()
        FR[g] = pd.concat([X, feats.reindex(X.index)], axis=1)
    A_v = pd.concat(FR.values())
    COLS_v = COLS0 + list(feats.columns)
    v, v_picks = score_variant('veer', A_v, COLS_v)
    print(f'VEER(F) fold-outside: {v["total"]:.6f} picks={v_picks}', flush=True)

    delta = v['total'] - ctrl['total']
    print(f'DELTA (fold-outside): {delta:+.6f}', flush=True)
    json.dump({'control': ctrl, 'control_picks': ctrl_picks,
               'veer_f': v, 'veer_f_picks': v_picks, 'delta_foldoutside': delta},
              open('/Users/um-yunsang/BARAM2026/research/nodes/S9-N5_foldoutside.json', 'w'),
              indent=1, default=str)
