"""S15-N8 * reverse ablation of the composed pipeline (contract R10 protocol, step 3).

S15-N7 measured the five stages together: mean Total 0.602299 -> 0.605359 (+0.003061, about 1.9
seed floors and 5.6 of the composed configuration's own sd 0.000549), with 1-NMAE +0.000778, and
the composition also cut the seed variance fourfold.

Attribution now proceeds by REVERTING one stage at a time from the full pipeline, each variant
scored as a 3-seed average.  This costs five extra configurations but spends NO additional
championship comparison, so the multiplicity budget is untouched.  A stage whose reversion does
not hurt was carried by the others and should be pruned.

The five stages, as built in S15-N7:
  B1  per-source spatial reduction (LDAPS most-exposed cell, speeds only)
  B2  supervised hub-wind feature `hub__ws_pred` learned against measured SCADA hub wind
  A3  SCADA-only power curve as the teacher target (no meter anywhere in the objective)
  D4  metric-matched group weights w_g = (1/3)/share_g
  D1  LightGBM extra_trees + path_smooth + feature_fraction_bynode
"""
import sys, json, time
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/scratch')
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/nodes')
from harness import surface, MU
from lib import FOLDS, CAPS, official_total
from s15_n7_compose import build_pc_scada, L1P, D1_EXTRA, SEEDS

S = '/Users/um-yunsang/BARAM2026/research/scratch/'
N = '/Users/um-yunsang/BARAM2026/research/nodes/'
ARMS = ['FULL', 'no_B1', 'no_B2', 'no_A3', 'no_D4', 'no_D1']

if __name__ == '__main__':
    A, FR, COLS = surface(('G2', 'DROP:grid__'))
    X = pd.read_parquet(S + 'b1_reduction.parquet')
    T = pd.read_parquet(S + 'teacher_targets.parquet')
    grp = A['grp'].to_numpy(); idx = A.index
    cf = A['cf'].to_numpy(); pct = A['pc_true'].to_numpy()
    valid = np.isfinite(cf) & (cf >= 0.1)
    w_prod = np.where(valid, np.clip(cf, 0, 1.2), 0.05)
    A2 = A.copy()
    for c in [c for c in X.columns if c.endswith('_speed') and c in A.columns]:
        A2[c] = X[c].reindex(A.index).to_numpy()
    pcs = build_pc_scada()
    pc_scada = np.full(len(A), np.nan)
    for g in (1, 2, 3):
        m = grp == g
        pc_scada[m] = pcs[g].reindex(idx[m]).to_numpy()
    tgt_scada = np.where(np.isfinite(pc_scada), pc_scada, pct)
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
            for arm in ARMS:
                t0 = time.time()
                frame = A if arm == 'no_B1' else A2
                tgt = pct if arm == 'no_A3' else tgt_scada
                use_hub = arm != 'no_B2'
                use_gw = arm != 'no_D4'
                mp = dict(MU); mp['random_state'] = sd
                lp = dict(L1P); lp['random_state'] = sd
                if arm != 'no_D1':
                    mp.update(D1_EXTRA); lp.update(D1_EXTRA)
                wt = w_prod * (gw if use_gw else 1.0)
                m = tr & np.isfinite(tgt)
                mu = lgb.LGBMRegressor(**mp)
                mu.fit(frame.loc[m, COLS], tgt[m], sample_weight=wt[m])
                F = frame[COLS].copy()
                F['pc_hat'] = np.clip(mu.predict(frame[COLS]), 0, 1.2)
                if use_hub:
                    hm = tr & np.isfinite(hub_obs)
                    hw = lgb.LGBMRegressor(**mp)
                    hw.fit(frame.loc[hm, COLS], hub_obs[hm], sample_weight=wt[hm])
                    F['hub__ws_pred'] = np.clip(hw.predict(frame[COLS]), 0, 40)
                rows = tr & valid
                pm = lgb.LGBMRegressor(**lp)
                pm.fit(F[rows], cf[rows], sample_weight=(gw[rows] if use_gw else None))
                preds.setdefault(f'{arm}_s{sd}', []).append(np.clip(pm.predict(F[va][keep]), 0, 1.1))
            print(f'  [{f}] seed={sd} six arms done {round(time.time()-t0,1)}s each', flush=True)

    K = pd.concat(keys, ignore_index=True)
    capv = K.group_id.map(CAPS).to_numpy()
    sc = lambda v: official_total(K.assign(prediction_kwh=v * capv)[
        ['group_id', 'actual_kwh', 'prediction_kwh']])
    out = {}
    print('\n=== reverse ablation, each arm a 3-seed average ===')
    for arm in ARMS:
        t = np.array([sc(np.concatenate(preds[f'{arm}_s{s}']))['total'] for s in SEEDS])
        n1 = np.array([sc(np.concatenate(preds[f'{arm}_s{s}']))['one_minus_nmae'] for s in SEEDS])
        out[arm] = {'mean_total': float(t.mean()), 'sd': float(t.std(ddof=1)),
                    'mean_1mnmae': float(n1.mean())}
        print(f'  {arm:7s} mean Total={t.mean():.6f} sd={t.std(ddof=1):.6f}  '
              f'mean 1-NMAE={n1.mean():.6f}')
    full = out['FULL']['mean_total']
    print(f'\n=== per-stage contribution (FULL minus the arm without it) ===')
    contrib = []
    for arm in ARMS[1:]:
        c = full - out[arm]['mean_total']
        contrib.append((arm.replace('no_', ''), c))
        print(f'  {arm.replace("no_",""):3s} {c:+.6f}   '
              f'{"KEEP" if c > 0 else "PRUNE (carried by the others)"}')
    print(f'\n  sum of contributions = {sum(c for _, c in contrib):+.6f}   '
          f'FULL - BASE(S15-N7) = {full-0.602299:+.6f}')
    json.dump({'arms': out, 'contributions': dict(contrib)},
              open(N + 'S15-N8_ablate.json', 'w'), indent=1, default=str)
