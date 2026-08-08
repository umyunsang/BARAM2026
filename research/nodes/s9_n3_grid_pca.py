
"""S9-N3 · R2: PCA latent representation of the raw 914-column NWP grid, grouped by
variable type (wind components / pressure / temperature / humidity / cloud / radiation /
etc.), K=3 components per group fixed a priori (not tuned on any fold or score), injected
as extra GBDT features alongside the existing default-harness baseline.

Per research/lanes/S6_ext_C_repr.md sec C3 R2 -- lowest-priority, conditional candidate:
raw grid pivot direct injection already failed at -0.4% in prior work; PCA only helps if
that failure was sample-vs-dimension starvation, not information absence, and the write-up
itself expects "low to medium, indirect" gain. K is fixed to 3 up front specifically because
picking it from an explained-variance threshold computed per fold would be a second, hidden
degree of freedom on top of K itself -- a fixed integer, decided before any run, avoids that.

Leakage handling: the PCA basis (mean/std/components) is fit on TRAIN rows only (index <
harness.SPLIT, matching the screening protocol used for S9-N0/S9-N1) and then used to
transform ALL rows, so no validation-period grid statistics leak into the basis.

This is injected directly into harness.py's surface() cache under a new block-key rather
than routed through s6feats.py's build_blocks(), because that shared cache has no notion
of a train/validation split -- retrofitting leakage-safe fitting into it was judged riskier
than fitting here, once, and handing harness.run() an already-augmented frame to run its
otherwise-unmodified teacher/calibrator/decision pipeline over (same methodology as
S9-N0/S9-N1, single declared axis against the frozen default baseline).
"""
import sys, json
import numpy as np, pandas as pd
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/scratch')
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/nodes')
import harness

K = 3  # components per variable group, fixed a priori
GRID_PIVOT = ('/Users/um-yunsang/BARAM2026/artifacts/cache/'
              '920be0c458d820e855bf79dd25723146f52ce1736138aedba5e6bc853f1f720b/'
              'train_grid_pivot.parquet')


def pca_features():
    raw = pd.read_parquet(GRID_PIVOT).set_index('forecast_kst_dtm')
    raw = raw[~raw.index.duplicated()].sort_index()
    tr = np.asarray(raw.index < harness.SPLIT)
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
    print(f'PCA: {len(groups)} variable groups, {sum(1 for c in groups if len(groups[c]) > K)} '
          f'used (>{K} cols), {len(out)} new columns', flush=True)
    return pd.DataFrame(out, index=raw.index)


if __name__ == '__main__':
    A0, FR0, COLS0 = harness.surface(())  # default baseline, identical to S9-N0/S9-N1
    pca = pca_features()

    FR = {}
    for g in (1, 2, 3):
        X = FR0[g].copy()
        add = pca.reindex(X.index)
        FR[g] = pd.concat([X, add], axis=1)
    A = pd.concat(FR.values())
    COLS = COLS0 + list(pca.columns)

    key = ('PCA_R2_v1',)
    harness._CACHE[key] = (A, FR, COLS)
    out = harness.run('S9-N3', 'R2_grid_pca_k3_per_variable_group_train_only_fit', blocks=key)
    print(json.dumps(out, indent=1, default=str))
