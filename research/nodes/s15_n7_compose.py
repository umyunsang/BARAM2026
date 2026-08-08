"""S15-N7 * the COMPOSED pipeline: five stage upgrades built together, scored as a 3-seed average.

Why composed and not one at a time.  research/engine/compose.py: our paired sd is 0.00075 and the
seed spread (S15-N3) is 0.001635, while a realistic stage upgrade is +0.002.  At k=1 that is 1.2
seed-sd and unmeasurable -- which is precisely what happened to B1 (-0.000041), to the component
grid, to the flow-regime block and to the displacement read.  At k=5 the same per-stage effect is
+0.010, six seed-sd, and resolvable.  So the unit of evaluation becomes the whole pipeline and
attribution is recovered afterwards by reverse ablation.

THE FIVE STAGES, each with the lane and the reason it is here.

 B1  per-source spatial reduction.  S15-N1 verified on measured hub wind that LDAPS wants the most
     exposed cell (box max 0.8405) while GFS wants inverse distance (0.7075 vs box max 0.5238) --
     the ordering inverts, so one shared rule is wrong for LDAPS.  SPEEDS ONLY: the direction
     replacement in S15-N2 used a different angular convention than the repository and was a
     rotation, not an improvement.

 B2  supervised hub-wind transfer (`hub__ws_pred`).  The NWP lane's first finding: scada_vestas /
     scada_unison carry ws at hub height for 26,304 hourly joins, so the grid-to-site and
     level-to-hub maps can be LEARNED against the observed wind instead of assumed from a log law.
     This project has done that cluster unsupervised throughout.

 A3  SCADA-only power curve.  The target lane's reinterpretation: both failed teacher experiments
     (isotonic recalibration, transfer factors + storm curve) fitted against METERED cf -- and so
     does the current powercurve.py.  The binding constraint may be the METER IN THE OBJECTIVE,
     not the curve's shape.  This fits the curve purely on SCADA (UNISON turbine power is valid;
     VESTAS power is permutation- and sign-corrupted but value-intact, so its curve is recovered
     by quantile mapping f(v) = Q_P(F_V(v))), with no meter anywhere in the objective.

 D4  metric-matched group weights w_g = (1/3)/share_g.  Our loss is micro-averaged and the metric
     is macro-averaged over three groups; group 3 supplies 9.0-15.4% of scored training rows while
     carrying exactly 1/3 of the score.  Tested once at k=1 (S13-N5) and rejected -- a measurement
     that contract R10 now declares inadmissible, since it was far below the seed floor.

 D1  LightGBM 4.7 regularisation the repo never switched on: extra_trees, path_smooth (which IS
     hierarchical shrinkage) and feature_fraction_bynode.
"""
import sys, json, time
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/scratch')
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/nodes')
from harness import surface, MU
from lib import FOLDS, CAPS, official_total

S = '/Users/um-yunsang/BARAM2026/research/scratch/'
N = '/Users/um-yunsang/BARAM2026/research/nodes/'
SEEDS = (20260801, 20260802, 20260803)
GT = {1: ('vestas', range(1, 7)), 2: ('vestas', range(7, 13)), 3: ('unison', range(1, 6))}
L1P = dict(objective='l1', n_estimators=900, learning_rate=0.035, num_leaves=63,
           min_child_samples=40, subsample=0.85, subsample_freq=1, colsample_bytree=0.4,
           reg_lambda=3.0, n_jobs=6, verbose=-1)
D1_EXTRA = dict(extra_trees=True, path_smooth=10.0, feature_fraction_bynode=0.7)


def scada_only_curve():
    """A3: power curve fitted on SCADA alone -- the meter never enters the objective."""
    ve = pd.read_parquet(S + 'scada_vestas.parquet')
    un = pd.read_parquet(S + 'scada_unison.parquet')
    curves = {}
    # UNISON: turbine power is valid -> direct monotone binned curve on (ws, power)
    v = np.concatenate([un[f'unison_wtg{i:02d}_ws'].to_numpy() for i in range(1, 6)])
    p = np.concatenate([un[f'unison_wtg{i:02d}_power_kw10m'].to_numpy() for i in range(1, 6)])
    ok = np.isfinite(v) & np.isfinite(p) & (p >= 0)
    curves['unison'] = binned_curve(v[ok], p[ok], np.nanquantile(p[ok], 0.999))
    # VESTAS: power is permutation/sign corrupted but value-intact -> quantile mapping
    v = np.concatenate([ve[f'vestas_wtg{i:02d}_ws'].to_numpy() for i in range(1, 13)])
    w = np.concatenate([np.abs(ve[f'vestas_wtg{i:02d}_power_kw10m'].to_numpy()) for i in range(1, 13)])
    ok = np.isfinite(v) & np.isfinite(w)
    vs = np.sort(v[ok]); ws = np.sort(w[ok])
    rated = float(np.nanquantile(ws, 0.999))
    # f(v) = Q_P(F_V(v)) : match the wind CDF to the (order-recovered) power CDF
    grid = np.linspace(0, 30, 301)
    Fv = np.searchsorted(vs, grid) / max(len(vs), 1)
    fp = np.quantile(ws, np.clip(Fv, 0, 1))
    curves['vestas'] = (grid, np.clip(fp / max(rated, 1e-9), 0, 1))
    return curves


def binned_curve(v, p, rated, nb=60):
    e = np.linspace(0, 30, nb + 1)
    k = np.clip(np.searchsorted(e, v, side='right') - 1, 0, nb - 1)
    med = np.full(nb, np.nan)
    for b in range(nb):
        m = k == b
        if m.sum() >= 40:
            med[b] = np.median(p[m])
    x = 0.5 * (e[:-1] + e[1:])
    good = np.isfinite(med)
    return (x[good], np.clip(med[good] / max(rated, 1e-9), 0, 1))


def apply_curve(curve, v):
    x, y = curve
    return np.interp(v, x, y, left=0.0, right=float(y[-1]))


def build_pc_scada():
    ve = pd.read_parquet(S + 'scada_vestas.parquet')
    un = pd.read_parquet(S + 'scada_unison.parquet')
    cur = scada_only_curve()
    out = {}
    for g, (src, rng) in GT.items():
        df = ve if src == 'vestas' else un
        he = (df['kst_dtm'] - pd.Timedelta('1s')).dt.ceil('h')
        cols = [f'{src}_wtg{i:02d}_ws' for i in rng]
        V = df[cols].to_numpy('float64')
        F = apply_curve(cur[src], V)
        out[g] = pd.Series(np.nanmean(F, axis=1), index=he.to_numpy()).groupby(level=0).mean()
    return out


if __name__ == '__main__':
    A, FR, COLS = surface(('G2', 'DROP:grid__'))
    X = pd.read_parquet(S + 'b1_reduction.parquet')
    T = pd.read_parquet(S + 'teacher_targets.parquet')
    grp = A['grp'].to_numpy(); idx = A.index
    cf = A['cf'].to_numpy(); pct = A['pc_true'].to_numpy()
    valid = np.isfinite(cf) & (cf >= 0.1)
    w_prod = np.where(valid, np.clip(cf, 0, 1.2), 0.05)

    # --- B1 : speeds only -------------------------------------------------------------
    A2 = A.copy()
    speed_cols = [c for c in X.columns if c.endswith('_speed') and c in A.columns]
    for c in speed_cols:
        A2[c] = X[c].reindex(A.index).to_numpy()
    # --- A3 : SCADA-only teacher target -----------------------------------------------
    pcs = build_pc_scada()
    pc_scada = np.full(len(A), np.nan)
    for g in (1, 2, 3):
        m = grp == g
        pc_scada[m] = pcs[g].reindex(idx[m]).to_numpy()
    ok = np.isfinite(pc_scada) & np.isfinite(pct)
    print(f'A3 SCADA-only teacher built: n={int(ok.sum())}  '
          f'corr with pc_true={np.corrcoef(pc_scada[ok], pct[ok])[0,1]:.4f}  '
          f'MAE vs metered cf on scored rows='
          f'{np.nanmean(np.abs(pc_scada[valid]-cf[valid])):.5f} (pc_true was 0.04804)')
    # --- B2 : supervised hub wind target ----------------------------------------------
    hub_obs = np.full(len(A), np.nan)
    for g in (1, 2, 3):
        m = grp == g
        hub_obs[m] = T[f'g{g}_v_mean'].reindex(idx[m]).to_numpy()

    keys = []; preds = {}
    for f, (a, b) in FOLDS.items():
        a = pd.Timestamp(a); b = pd.Timestamp(b)
        tr = np.asarray(idx < a); va = np.asarray((idx >= a) & (idx <= b)); keep = np.isfinite(cf[va])
        keys.append(pd.DataFrame({'fold_id': f, 'group_id': grp[va][keep],
                                  'forecast_kst_dtm': idx[va][keep],
                                  'actual_kwh': cf[va][keep] * np.array([CAPS[g] for g in grp[va][keep]])}))
        ng = {g: max(int((tr & valid & (grp == g)).sum()), 1) for g in (1, 2, 3)}
        gw = np.array([(1.0 / 3.0) / (ng[g] / sum(ng.values())) for g in grp])
        for sd in SEEDS:
            for tag in ('BASE', 'COMPOSED'):
                t0 = time.time()
                frame = A if tag == 'BASE' else A2
                tgt = pct if tag == 'BASE' else np.where(np.isfinite(pc_scada), pc_scada, pct)
                mp = dict(MU); mp['random_state'] = sd
                lp = dict(L1P); lp['random_state'] = sd
                if tag == 'COMPOSED':
                    mp.update(D1_EXTRA); lp.update(D1_EXTRA)
                m = tr & np.isfinite(tgt)
                wt = w_prod * (gw if tag == 'COMPOSED' else 1.0)
                mu = lgb.LGBMRegressor(**mp)
                mu.fit(frame.loc[m, COLS], tgt[m], sample_weight=wt[m])
                pch = np.clip(mu.predict(frame[COLS]), 0, 1.2)
                F = frame[COLS].copy(); F['pc_hat'] = pch
                if tag == 'COMPOSED':
                    hm = tr & np.isfinite(hub_obs)
                    hw = lgb.LGBMRegressor(**mp)
                    hw.fit(frame.loc[hm, COLS], hub_obs[hm], sample_weight=wt[hm])
                    F['hub__ws_pred'] = np.clip(hw.predict(frame[COLS]), 0, 40)
                rows = tr & valid
                pm = lgb.LGBMRegressor(**lp)
                pm.fit(F[rows], cf[rows], sample_weight=(gw[rows] if tag == 'COMPOSED' else None))
                preds.setdefault(f'{tag}_s{sd}', []).append(np.clip(pm.predict(F[va][keep]), 0, 1.1))
                print(f'  [{f}] {tag} seed={sd} {round(time.time()-t0,1)}s', flush=True)

    K = pd.concat(keys, ignore_index=True)
    capv = K.group_id.map(CAPS).to_numpy()
    sc = lambda v: official_total(K.assign(prediction_kwh=v * capv)[
        ['group_id', 'actual_kwh', 'prediction_kwh']])
    res = {k: sc(np.concatenate(v)) for k, v in preds.items()}
    out = {}
    print('\n=== COMPOSED vs BASE, each as a 3-seed average (contract R10) ===')
    for tag in ('BASE', 'COMPOSED'):
        t = np.array([res[f'{tag}_s{s}']['total'] for s in SEEDS])
        n1 = np.array([res[f'{tag}_s{s}']['one_minus_nmae'] for s in SEEDS])
        fi = np.array([res[f'{tag}_s{s}']['ficr'] for s in SEEDS])
        avg = np.mean([np.concatenate(preds[f'{tag}_s{s}']) for s in SEEDS], axis=0)
        sa = sc(avg)
        out[tag] = {'per_seed_total': t.tolist(), 'mean_total': float(t.mean()),
                    'sd_total': float(t.std(ddof=1)), 'mean_1mnmae': float(n1.mean()),
                    'mean_ficr': float(fi.mean()), 'seedavg_total': sa['total'],
                    'seedavg_1mnmae': sa['one_minus_nmae'], 'seedavg_ficr': sa['ficr']}
        print(f'  {tag:9s} per-seed Total {np.round(t,6)}  mean={t.mean():.6f} sd={t.std(ddof=1):.6f}')
        print(f'            mean 1-NMAE={n1.mean():.6f}  mean FICR={fi.mean():.6f}  '
              f'seed-averaged Total={sa["total"]:.6f}')
    d = out['COMPOSED']['mean_total'] - out['BASE']['mean_total']
    d1 = out['COMPOSED']['mean_1mnmae'] - out['BASE']['mean_1mnmae']
    ds = out['COMPOSED']['seedavg_total'] - out['BASE']['seedavg_total']
    print(f'\n  COMPOSED - BASE : mean Total {d:+.6f}   mean 1-NMAE {d1:+.6f}   '
          f'seed-averaged Total {ds:+.6f}')
    print(f'  seed floor 0.001635; five stages at the lanes\' expectations would be ~+0.010')
    np.save(N + 'S15-N7_preds.npy', np.vstack([np.concatenate(preds[k]) for k in sorted(preds)]))
    json.dump(out, open(N + 'S15-N7_compose.json', 'w'), indent=1, default=str)
    K.to_parquet(N + 'S15-N7_keys.parquet', index=False)
