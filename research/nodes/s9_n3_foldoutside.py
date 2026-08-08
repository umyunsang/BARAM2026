
"""S9-N3 fold-outside gate: promote the screening-positive R2 (grid PCA) result to a
genuine 3-fold expanding-window fold-outside test, matching this project's stricter
"full" protocol (registry.json.protocol.full) and s10_final3.py's fo_blend pattern --
policy chosen on the OTHER two folds, applied to the held-out fold.

Runs both a no-PCA control and the PCA variant through an identical per-fold pipeline
(teacher + calibrator + T/G decision grid, all DEF-1-fixed), so the isolated fold-outside
delta can be read directly, the same way S9-N0 vs S9-N3 were compared under screening.
"""
import sys, json
import numpy as np, pandas as pd
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/scratch')
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/nodes')
import harness
from lib import official_total, FOLDS, CAPS, sharpen_weights
from s9_n3_grid_pca import GRID_PIVOT, K


def pca_features_for_cutoff(cutoff_ts):
    raw = pd.read_parquet(GRID_PIVOT).set_index('forecast_kst_dtm')
    raw = raw[~raw.index.duplicated()].sort_index()
    tr = np.asarray(raw.index < cutoff_ts)
    groups = {}
    for c in raw.columns:
        groups.setdefault(c.split('__')[-1], []).append(c)
    out = {}
    for suffix, cols in groups.items():
        if len(cols) <= K:
            continue
        X = raw[cols].to_numpy('float64')
        mu = np.nanmean(X[tr], axis=0)
        X = np.where(np.isnan(X), mu[None, :], X)
        sd = X[tr].std(axis=0)
        sd[sd < 1e-9] = 1.0
        Xs = (X - mu) / sd
        _, _, Vt = np.linalg.svd(Xs[tr], full_matrices=False)
        comps = Vt[:K]
        proj = Xs @ comps.T
        for k in range(K):
            out[f'pca__{suffix}__{k}'] = proj[:, k].astype('float32')
    return pd.DataFrame(out, index=raw.index)


def run_fold(A0, FR0, COLS0, tr_mask, va_mask, use_pca, cutoff_ts):
    if use_pca:
        pca = pca_features_for_cutoff(cutoff_ts)
        FR = {}
        for g in (1, 2, 3):
            X = FR0[g].copy()
            FR[g] = pd.concat([X, pca.reindex(X.index)], axis=1)
        A = pd.concat(FR.values())
        COLS = COLS0 + list(pca.columns)
    else:
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


def score_variant(use_pca, A0, FR0, COLS0):
    per_fold = {}
    for f, (a, b) in FOLDS.items():
        a = pd.Timestamp(a); b = pd.Timestamp(b)
        tr_mask = np.asarray(A0.index < a)
        va_mask = np.asarray((A0.index >= a) & (A0.index <= b))
        per_fold[f] = run_fold(A0, FR0, COLS0, tr_mask, va_mask, use_pca, cutoff_ts=a)
        print(f'  [{"PCA" if use_pca else "control"}] fold {f} fitted, '
              f'{per_fold[f].shape[0]} val rows', flush=True)
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
    ctrl, ctrl_picks = score_variant(False, A0, FR0, COLS0)
    print(f'CONTROL fold-outside: {ctrl["total"]:.6f} picks={ctrl_picks}', flush=True)
    pca_res, pca_picks = score_variant(True, A0, FR0, COLS0)
    print(f'PCA(R2) fold-outside: {pca_res["total"]:.6f} picks={pca_picks}', flush=True)
    delta = pca_res['total'] - ctrl['total']
    print(f'DELTA (fold-outside): {delta:+.6f}', flush=True)
    json.dump({'control': ctrl, 'control_picks': ctrl_picks,
               'pca_r2': pca_res, 'pca_r2_picks': pca_picks,
               'delta_foldoutside': delta},
              open('/Users/um-yunsang/BARAM2026/research/nodes/S9-N3_foldoutside.json', 'w'),
              indent=1, default=str)
