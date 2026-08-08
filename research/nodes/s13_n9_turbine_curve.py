"""S13-N9 * (S7 rung, reduced M1) per-turbine transfer factors and a storm-control curve.

The S7 lane's top node is turbine-level modelling aggregated up (Gilbert 2020 IEEE TSTE:
CRPS -3.95% to -6.50% at COMPLEX-LAYOUT, COMPLEX-TERRAIN sites versus only -1.24% to -2.39% at
simple sites -- our 17 turbines on a 2.5 km ridge with three differently-oriented arrays are the
complex case), with the explicit warning that nacelle-anemometer wake bias must be corrected by
a per-turbine transfer function (NTF) or it is simply pushed into the per-turbine curves.

Reading the existing code first changed the node.  research/scratch/powercurve.py ALREADY
evaluates the curve per turbine per 10 minutes and averages afterwards, so the Jensen term the
lane worried about is already handled.  What it does NOT do:
  * it shares ONE 4-parameter curve  f(v) = clip((v-vin)/(vr-vin),0,1)^k  across all turbines of
    a group, so every per-turbine anemometer bias and wake deficit is unmodelled;
  * that curve has NO storm-control region -- it is exactly 1.0 from rated wind up to a hard
    cut-out step at vout.
The second point is the measured defect: S13-N2 found median(pc_true - cf) = +0.054 / +0.066 /
+0.064 in the 12-14, 14-16 and 16-18 m/s bins (g3 up to +0.311) and -0.021 at 8-10 m/s, i.e. the
curve claims full output through the whole high-wind range while the fleet de-rates.

Treatment (teacher target only; everything downstream identical):
  PC0  current 4-parameter shared curve                                   (control = pc_true)
  PC1  + per-turbine multiplicative transfer factors a_i (17 parameters)
  PC2  + a storm-control ramp: f decays linearly from v_storm to v_out instead of stepping
  PC3  PC1 + PC2
All parameters are fitted by minimising MAE against the metered hourly capacity factor on the
TRAINING window of each fold only, then applied unchanged to the held-out fold.
"""
import sys, json, time
import numpy as np, pandas as pd
from scipy.optimize import minimize
import lightgbm as lgb
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/scratch')
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/nodes')
from harness import surface, MU
from lib import FOLDS, CAPS, official_total

S = '/Users/um-yunsang/BARAM2026/research/scratch/'
N = '/Users/um-yunsang/BARAM2026/research/nodes/'
CAPF = {1: 21600.0, 2: 21600.0, 3: 21000.0}
GT = {1: ('vestas', range(1, 7)), 2: ('vestas', range(7, 13)), 3: ('unison', range(1, 6))}
L1P = dict(objective='l1', n_estimators=900, learning_rate=0.035, num_leaves=63,
           min_child_samples=40, subsample=0.85, subsample_freq=1, colsample_bytree=0.4,
           reg_lambda=3.0, random_state=20260801, n_jobs=6, verbose=-1)


def load_ws():
    ve = pd.read_parquet(S + 'scada_vestas.parquet')
    un = pd.read_parquet(S + 'scada_unison.parquet')
    out = {}
    for g, (src, rng) in GT.items():
        df = ve if src == 'vestas' else un
        he = (df['kst_dtm'] - pd.Timedelta('1s')).dt.ceil('h')
        cols = [f'{src}_wtg{i:02d}_ws' for i in rng]
        out[g] = (he.to_numpy(), df[cols].to_numpy('float64'))
    return out


def curve(v, vin, vr, vout, k, vstorm=None, storm_floor=1.0):
    x = np.clip((v - vin) / max(vr - vin, 0.1), 0, 1)
    f = np.where(v >= vr, 1.0, x ** k)
    if vstorm is not None and vout > vstorm:
        ramp = 1.0 - (1.0 - storm_floor) * np.clip((v - vstorm) / (vout - vstorm), 0, 1)
        f = np.where(v >= vstorm, ramp, f)
    f = np.where((v < vin) | (v > vout), 0.0, f)
    return f


def hourly_pc(he, V, p, ntf=None):
    v = V if ntf is None else V * ntf[None, :]
    f = curve(v, *p[:4], vstorm=p[4] if len(p) > 4 else None,
              storm_floor=p[5] if len(p) > 5 else 1.0)
    m = np.nanmean(f, axis=1)
    return pd.Series(m, index=he).groupby(level=0).mean()


def fit_curve(g, he, V, y_tr, use_ntf, use_storm, seed_p):
    nt = V.shape[1]

    def unpack(z):
        p = list(seed_p)
        p[0], p[1], p[3] = 1.0 + 4.5 / (1 + np.exp(-z[0])), 6.0 + 10.0 / (1 + np.exp(-z[1])), \
            1.0 + 4.0 / (1 + np.exp(-z[2]))
        p[2] = 18.0 + 12.0 / (1 + np.exp(-z[3]))
        i = 4
        if use_storm:
            p = p[:4] + [10.0 + 12.0 / (1 + np.exp(-z[i])), 1.0 / (1 + np.exp(-z[i + 1]))]
            i += 2
        else:
            p = p[:4]
        ntf = None
        if use_ntf:
            a = 1.0 + 0.25 * np.tanh(z[i:i + nt])
            ntf = a / a.mean()
        return p, ntf

    def obj(z):
        p, ntf = unpack(z)
        s = hourly_pc(he, V, p, ntf)
        j = pd.concat([s.rename('p'), y_tr.rename('y')], axis=1).dropna()
        if len(j) < 500:
            return 1e6
        return float(np.abs(j.p - j.y).mean())

    n = 4 + (2 if use_storm else 0) + (nt if use_ntf else 0)
    z0 = np.zeros(n)
    r = minimize(obj, z0, method='Powell',
                 options=dict(maxiter=4000, xtol=1e-3, ftol=1e-5))
    return unpack(r.x), float(r.fun)


def run():
    WS = load_ws()
    lab = pd.read_parquet(S + 'labels.parquet').set_index('kst_dtm')
    A, FRM, COLS = surface(('G2', 'DROP:grid__'))
    grp = A['grp'].to_numpy(); idx = A.index
    cf = A['cf'].to_numpy()
    valid = np.isfinite(cf) & (cf >= 0.1)
    w_prod = np.where(valid, np.clip(cf, 0, 1.2), 0.05)
    seed = json.load(open(S + 'powercurve_params.json'))

    VAR = {'PC0': (False, False), 'PC1': (True, False), 'PC2': (False, True), 'PC3': (True, True)}
    keys = []; preds = {}
    for f, (a, b) in FOLDS.items():
        a = pd.Timestamp(a); b = pd.Timestamp(b)
        tr = np.asarray(idx < a); va = np.asarray((idx >= a) & (idx <= b)); keep = np.isfinite(cf[va])
        keys.append(pd.DataFrame({'fold_id': f, 'group_id': grp[va][keep],
                                  'forecast_kst_dtm': idx[va][keep],
                                  'actual_kwh': cf[va][keep] * np.array([CAPF[g] for g in grp[va][keep]])}))
        tgt = {}
        for nm, (use_ntf, use_storm) in VAR.items():
            series = {}
            for g in (1, 2, 3):
                he, V = WS[g]
                y = (lab[f'kpx_group_{g}'] / CAPF[g])
                y_tr = y[y.index < a]
                if nm == 'PC0':
                    p, ntf = list(seed[str(g)]), None
                    mae = np.nan
                else:
                    t0 = time.time()
                    (p, ntf), mae = fit_curve(g, he, V, y_tr, use_ntf, use_storm, seed[str(g)])
                    print(f'  [{f}] {nm} g{g} trainMAE={mae:.5f} p={np.round(p,2)} '
                          f'ntf={None if ntf is None else np.round(ntf,3)} {round(time.time()-t0,1)}s',
                          flush=True)
                series[g] = hourly_pc(he, V, p, ntf)
            v = np.full(len(A), np.nan)
            for g in (1, 2, 3):
                m = grp == g
                v[m] = series[g].reindex(idx[m]).to_numpy()
            tgt[nm] = np.clip(v, 0, 1.3)

        for nm, y in tgt.items():
            m = tr & np.isfinite(y)
            mu = lgb.LGBMRegressor(**MU)
            mu.fit(A.loc[m, COLS], y[m], sample_weight=w_prod[m])
            pch = np.clip(mu.predict(A[COLS]), 0, 1.3)
            F = A[COLS].copy(); F['pc_hat'] = pch
            pm = lgb.LGBMRegressor(**L1P)
            pm.fit(F[tr & valid], cf[tr & valid])
            preds.setdefault(nm, []).append(np.clip(pm.predict(F[va][keep]), 0, 1.1))
            preds.setdefault(nm + '_teacherbias', []).append(
                (y[va][keep] - cf[va][keep]))
        print(f'  [{f}] all four teachers done', flush=True)

    K = pd.concat(keys, ignore_index=True)
    capv = K.group_id.map(CAPF).to_numpy()
    out = {}
    print('\n--- teacher bias |pc - cf| on scored rows (the S13-N2 defect) ---')
    sc = K.actual_kwh >= 0.1 * capv
    for nm in VAR:
        bi = np.concatenate(preds[nm + '_teacherbias'])
        out[nm + '_teacher_mae'] = float(np.nanmean(np.abs(bi[sc])))
        print(f'  {nm}: MAE(pc,cf)={np.nanmean(np.abs(bi[sc])):.5f}  '
              f'median bias={np.nanmedian(bi[sc]):+.5f}')
    print('\n--- downstream point accuracy (1-NMAE is the binding constraint) ---')
    for nm in VAR:
        v = np.concatenate(preds[nm])
        s = official_total(K.assign(prediction_kwh=v * capv)[['group_id', 'actual_kwh', 'prediction_kwh']])
        out[nm] = {'one_minus_nmae': s['one_minus_nmae'], 'ficr': s['ficr'], 'total': s['total']}
        star = '  <-- control' if nm == 'PC0' else ''
        print(f'  {nm} 1-NMAE={s["one_minus_nmae"]:.6f}  FICR={s["ficr"]:.6f}  Total={s["total"]:.6f}{star}')
    b = out['PC0']['one_minus_nmae']
    for nm in ['PC1', 'PC2', 'PC3']:
        print(f'  delta {nm} vs PC0 on 1-NMAE: {out[nm]["one_minus_nmae"]-b:+.6f}')
    json.dump(out, open(N + 'S13-N9_turbine_curve.json', 'w'), indent=1, default=str)
    np.save(N + 'S13-N9_preds.npy', np.vstack([np.concatenate(preds[k]) for k in VAR]))
    K.to_parquet(N + 'S13-N9_keys.parquet', index=False)


if __name__ == '__main__':
    run()
