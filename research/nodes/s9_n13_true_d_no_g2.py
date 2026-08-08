
"""S9-N13 · re-test the "G2 might be a net drag" lead (found at harness-baseline level:
research/nodes/s9_n6_stack_g2.py, control vs G2-only was -0.000812) against the TRUE,
verified D architecture (research/nodes/s7_more.py, confirmed exact via s9_n12) instead
of the much weaker harness-default baseline. Only change from D_REPRO: surface(())
instead of surface(('G2','DROP:grid__')) -- i.e. keep the ORIGINAL (not G2-fixed) grid__
reshape rather than removing it entirely, since D's COLS baseline without G2 still needs
SOME grid representation; this isolates G2's specific marginal contribution the same way
s9_n6_stack_g2.py did.
"""
import sys, json, time
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/scratch')
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/nodes')
from harness import surface, MU
from lib import FOLDS, CAPS, official_total

N = '/Users/um-yunsang/BARAM2026/research/nodes/'
AB = '/Users/um-yunsang/BARAM2026/artifacts/backtests/metric-aligned-probe/'
KEY = ['fold_id', 'group_id', 'forecast_kst_dtm']
DEP = {'M102_TOP100': 'T0.5_G1.5', 'M113_LGBM_DART': 'T0.5_G0.5', 'M115_XGBOOST': 'T0.6_G0.35'}
W = 0.04; NC = 26
ACT = np.arange(0.02, 1.0801, 0.0025)
SC = {1: 0.985, 2: 0.989, 3: 1.005}
TEMPS = [0.6, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
GAMMAS = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0]
DART_CLF = dict(objective='multiclass', boosting_type='dart', n_estimators=400,
                 learning_rate=0.08, num_leaves=31, min_child_samples=60, subsample=0.85,
                 subsample_freq=1, colsample_bytree=0.4, reg_lambda=3.0,
                 random_state=20260803, n_jobs=6, verbose=-1)


def build_and_save(tag, blocks):
    A, FR, COLS = surface(blocks)
    cf = A['cf'].to_numpy(); grp = A['grp'].to_numpy(); idx = A.index
    valid = np.isfinite(cf) & (cf >= 0.1)
    w_prod = np.where(valid, np.clip(cf, 0, 1.2), 0.05)
    w_valid = np.where(valid, 1.0, 0.15)
    gapv = A['pc_true'].to_numpy() - cf

    rows = []; probs = []
    for f, (a, b) in FOLDS.items():
        a = pd.Timestamp(a); b = pd.Timestamp(b)
        tr = np.asarray(idx < a); va = np.asarray((idx >= a) & (idx <= b)); keep = np.isfinite(cf[va])
        t0 = time.time()
        m = tr & np.isfinite(A['pc_true'].to_numpy())
        mu = lgb.LGBMRegressor(**MU)
        mu.fit(A.loc[m, COLS], A.loc[m, 'pc_true'], sample_weight=w_prod[m])
        pc = np.clip(mu.predict(A[COLS]), 0, 1)
        sel = list(pd.Series(mu.feature_importances_, index=COLS).sort_values(ascending=False).head(150).index)
        B = A[sel].copy(); B['pc_hat'] = pc
        for k in (1, 2, 3):
            B[f'ig{k}'] = (grp == k).astype('float32')
        cls = np.clip(np.nan_to_num(cf, nan=0.0) / W, 0, NC - 1).astype(int)
        cm = tr & np.isfinite(cf) & (~(gapv >= 0.05))
        d = lgb.LGBMClassifier(**DART_CLF)
        d.fit(B[cm], cls[cm], sample_weight=w_valid[cm])
        raw = d.predict_proba(B[va])
        P = np.zeros((raw.shape[0], NC)); P[:, np.asarray(d.classes_, int)] = raw
        rows.append(pd.DataFrame({'fold_id': f, 'group_id': grp[va][keep],
                                   'forecast_kst_dtm': idx[va][keep], 'cf': cf[va][keep],
                                   'mean_gen_g': [float(np.nanmean(cf[tr & (grp == g)])) for g in grp[va][keep]]}))
        probs.append(P[keep])
        print(f'  [{tag}] fold {f} fitted {round(time.time()-t0,1)}s', flush=True)
    R = pd.concat(rows, ignore_index=True)
    Pfull = np.vstack(probs)
    R.to_parquet(N + f'S7-N8_{tag}_keys.parquet', index=False)
    np.save(N + f'S7-N8_{tag}_prob.npy', Pfull)
    return R, Pfull


def member_from_saved(tag):
    R = pd.read_parquet(N + f'S7-N8_{tag}_keys.parquet')
    P = np.load(N + f'S7-N8_{tag}_prob.npy')
    NCk = P.shape[1]; C = (np.arange(NCk) + 0.5) * W
    err = np.abs(ACT[:, None] - C[None, :]); units = np.where(err <= 0.06, 4., np.where(err <= 0.08, 3., 0.))
    g = R.group_id.to_numpy(); mg = R.mean_gen_g.to_numpy()
    capv = np.array([CAPS[x] for x in g]); hi = np.array([SC[x] for x in g]); act = R.cf.to_numpy() * capv
    mask = (C >= 0.10).astype(float); frames = {}
    for tp in TEMPS:
        q = P ** (1.0 / tp); q = q / np.maximum(q.sum(axis=1, keepdims=True), 1e-12)
        q = q * mask[None, :]; q = q / np.maximum(q.sum(axis=1, keepdims=True), 1e-12)
        nm = -(q @ err.T); fic = (q @ ((C[None, :] * units).T))
        for gm in GAMMAS:
            frames[(tp, gm)] = np.minimum(ACT[np.argmax(nm + gm * fic / (4.0 * mg[:, None]), axis=1)], hi) * capv
    Dm = pd.DataFrame({'fold_id': R.fold_id, 'group_id': g, 'forecast_kst_dtm': R.forecast_kst_dtm, 'actual_kwh': act})
    out = np.empty(len(Dm))
    for f in FOLDS:
        sel = (Dm.fold_id == f).to_numpy()
        s2 = {k: official_total(Dm[~sel].assign(prediction_kwh=v[~sel])[['group_id', 'actual_kwh', 'prediction_kwh']])['total']
              for k, v in frames.items()}
        out[sel] = frames[max(s2, key=s2.get)][sel]
    Dm['prediction_kwh'] = out
    return Dm, official_total(Dm[['group_id', 'actual_kwh', 'prediction_kwh']])['total']


def load_depavg():
    fr = []
    for stem, pol in DEP.items():
        parts = []
        for f in FOLDS:
            d = pd.read_parquet(AB + f'{stem}-{f}-policies.parquet')
            d = d.copy(); d['fold_id'] = f
            parts.append(d[KEY + [pol]].rename(columns={pol: stem}))
        fr.append(pd.concat(parts, ignore_index=True))
    J = fr[0]
    for x in fr[1:]:
        J = J.merge(x, on=KEY)
    J['DEPAVG'] = J[list(DEP)].mean(axis=1)
    return J[KEY + ['DEPAVG']]


def fo_blend_1dof(J, col_a, col_b):
    grid = [(w, 1 - w) for w in np.arange(0, 1.001, 0.05)]
    rows = []; picks = {}
    for f in FOLDS:
        oth = J[J.fold_id != f]; held = J[J.fold_id == f]
        best = None
        for wa, wb in grid:
            pred = wa * oth[col_a] + wb * oth[col_b]
            t = official_total(oth.assign(prediction_kwh=pred)[
                ['group_id', 'actual_kwh', 'prediction_kwh']])['total']
            if best is None or t > best[0]:
                best = (t, (wa, wb))
        picks[f] = best[1]
        pred_h = best[1][0] * held[col_a] + best[1][1] * held[col_b]
        rows.append(held.assign(prediction_kwh=pred_h))
    Dd = pd.concat(rows, ignore_index=True)
    return official_total(Dd[['group_id', 'actual_kwh', 'prediction_kwh']]), picks


if __name__ == '__main__':
    dep = load_depavg()
    build_and_save('D_NOG2', blocks=())
    Dm, solo = member_from_saved('D_NOG2')
    print(f'D_NOG2 solo: {solo:.6f}  (D_REPRO with G2 was 0.625669)', flush=True)
    J = Dm[KEY + ['actual_kwh', 'prediction_kwh']].rename(columns={'prediction_kwh': 'D_NOG2'}).merge(dep, on=KEY)
    blend, _ = fo_blend_1dof(J, 'D_NOG2', 'DEPAVG')
    print(f'DEPAVG + D_NOG2: {blend["total"]:.6f}  delta vs current_best: {blend["total"]-0.636184:+.6f}', flush=True)
    print(f'delta vs DEPAVG+D_REPRO(0.636184): {blend["total"]-0.636184:+.6f}', flush=True)
    json.dump({'solo': solo, 'depavg_blend': blend, 'delta_vs_current_best': blend['total'] - 0.636184},
              open(N + 'S9-N13_true_d_no_g2.json', 'w'), indent=1, default=str)
