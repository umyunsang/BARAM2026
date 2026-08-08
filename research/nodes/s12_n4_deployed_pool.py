"""S12-N4 * fold-outside policy selection over the FULL deployed backtest pool.

DEPAVG (the 0.70 half of current_best) is a hard-coded 3-stem average at 3 hard-coded
policies (M102_TOP100 T0.5_G1.5, M113_LGBM_DART T0.5_G0.5, M115_XGBOOST T0.6_G0.35).
Twelve deployed stems actually have all three fold frames saved, and each frame carries the
whole (T,G) policy grid.  Nothing in the ledger ever (a) picked each deployed stem's policy
under the fold-outside gate rather than by hand, or (b) searched the wider 12-stem pool.

Per AGENTS.md's standing rule this node states, for every number it reports:
  (a) policy provenance   -- fold-outside: policy chosen on the other two folds only;
  (b) weight provenance   -- fold-outside: blend weights chosen on the other two folds only;
  (c) row-alignment key   -- ['fold_id','group_id','forecast_kst_dtm'].
It never reads a `prediction_kwh` column (the documented mixed-policy trap).
"""
import sys, json, itertools, os, re
import numpy as np, pandas as pd
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/nodes')
from loop_lib import *

AB = '/Users/um-yunsang/BARAM2026/artifacts/backtests/metric-aligned-probe/'
STEMS = ['M102_TOP100', 'M113_LGBM_DART', 'M115_XGBOOST', 'M269_PROBE_TOP100', 'M271_M269REPRO',
         'M64B_ALLWEATHER_SITEWIND_CLASS', 'M68_SITEWIND_CLASS_ITER', 'M72_BIN020',
         'M84_LEAVES031', 'M93_POWER_QUANTILE', 'M96_ORDINAL_CUMULATIVE', 'M98_ORDINAL_BIN025']


def load_stem(stem):
    parts = []
    for f in FOLDS:
        d = pd.read_parquet(AB + f'{stem}-{f}-policies.parquet').copy()
        d['fold_id'] = f
        parts.append(d)
    D = pd.concat(parts, ignore_index=True)
    skip = {'forecast_id', 'forecast_kst_dtm', 'group_id', 'actual_kwh', 'fold_id',
            'prediction_kwh', 'MEDIAN'}
    pols = [c for c in D.columns if c not in skip and pd.api.types.is_numeric_dtype(D[c])]
    return D, pols


def fo_policy_stem(D, pols):
    base = D[KEY + ['actual_kwh']].copy()
    out = np.empty(len(D)); picks = {}
    for f in FOLDS:
        sel = (D.fold_id == f).to_numpy()
        s = {p: official_total(base[~sel].assign(prediction_kwh=D.loc[~sel, p])[
                 ['group_id', 'actual_kwh', 'prediction_kwh']])['total'] for p in pols}
        bk = max(s, key=s.get); picks[f] = bk
        out[sel] = D.loc[sel, bk].to_numpy()
    base['pred'] = out
    return base, official_total(base.assign(prediction_kwh=base.pred)[
        ['group_id', 'actual_kwh', 'prediction_kwh']]), picks


if __name__ == '__main__':
    res = {}; frames = {}
    for s in STEMS:
        D, pols = load_stem(s)
        b, sc, pk = fo_policy_stem(D, pols)
        res[s] = {'foldoutside_total': sc['total'], 'one_minus_nmae': sc['one_minus_nmae'],
                  'ficr': sc['ficr'], 'policy_picks': pk, 'n_policies': len(pols)}
        frames[s] = b.rename(columns={'pred': s})[KEY + ['actual_kwh', s]]
        print(f'{s:32s} FO={sc["total"]:.6f}  picks={list(pk.values())}', flush=True)

    J = frames[STEMS[0]]
    for s in STEMS[1:]:
        J = J.merge(frames[s].drop(columns=['actual_kwh']), on=KEY)
    print('\njoined rows', len(J))

    # fixed-policy DEPAVG reference (exactly what current_best uses)
    dep = load_depavg()
    J = J.merge(dep, on=KEY)
    tot = lambda c: official_total(J.assign(prediction_kwh=J[c])[
        ['group_id', 'actual_kwh', 'prediction_kwh']])['total']
    print('DEPAVG (fixed policies, reference)', round(tot('DEPAVG'), 6))

    # error-correlation structure of the deployed pool
    cap = J.group_id.map(CAPS)
    E = pd.DataFrame({s: (J[s] - J.actual_kwh) / cap for s in STEMS})
    C = E.corr()
    print('\n--- deployed-pool error correlation (min pair) ---')
    cc = C.where(~np.eye(len(C), dtype=bool)).stack().sort_values()
    print(cc.head(8).round(4).to_string())

    # uniform average of the k least-correlated / best stems, fold-outside subset choice
    print('\n--- uniform averages of stem subsets ---')
    sub_res = {}
    ranked = sorted(STEMS, key=lambda s: -res[s]['foldoutside_total'])
    cands = {'ALL12': STEMS, 'TOP3': ranked[:3], 'TOP4': ranked[:4], 'TOP5': ranked[:5],
             'TOP6': ranked[:6], 'TOP8': ranked[:8],
             'DEP3': ['M102_TOP100', 'M113_LGBM_DART', 'M115_XGBOOST']}
    for nm, sub in cands.items():
        J[f'AVG_{nm}'] = J[sub].mean(axis=1)
        sub_res[nm] = tot(f'AVG_{nm}')
        print(f'  AVG_{nm:8s} ({len(sub)} stems) {sub_res[nm]:.6f}', flush=True)

    # exhaustive 3-subset search, in-sample only (reported as in-sample, NOT a claim)
    best3 = None
    for combo in itertools.combinations(STEMS, 3):
        v = J[list(combo)].mean(axis=1)
        t = official_total(J.assign(prediction_kwh=v)[['group_id', 'actual_kwh', 'prediction_kwh']])['total']
        if best3 is None or t > best3[0]:
            best3 = (t, combo)
    print(f'\nbest 3-stem uniform average (IN-SAMPLE subset choice): {best3[0]:.6f} {best3[1]}')

    # fold-outside subset choice among the named candidates, then blend with member D
    R = canonical_keys()
    fr = utility_frames(align_prob('D', R), R)
    Dm, dsolo, dpicks = fo_policy(fr, R)
    J2 = J.merge(Dm[KEY + ['prediction_kwh']].rename(columns={'prediction_kwh': 'D'}), on=KEY)
    print('\n--- 1-dof fold-outside blend of D with each deployed aggregate ---')
    blends = {}
    for nm in list(cands) + ['DEPAVG']:
        col = f'AVG_{nm}' if nm != 'DEPAVG' else 'DEPAVG'
        bl, wp, _ = fo_blend_1dof(J2, 'D', col)
        blends[nm] = {'total': bl['total'], 'one_minus_nmae': bl['one_minus_nmae'],
                      'ficr': bl['ficr'], 'weights': wp}
        print(f'  D + {col:14s} {bl["total"]:.6f}  d_vs_current_best={bl["total"]-0.6361842493883538:+.6f}  w={list(wp.values())}',
              flush=True)

    json.dump({'stems': res, 'subset_avgs': sub_res, 'best3_insample': [best3[0], list(best3[1])],
               'blends_with_D': blends,
               'min_error_corr_pairs': {f'{a}|{b}': float(v) for (a, b), v in cc.head(10).items()}},
              open('/Users/um-yunsang/BARAM2026/research/nodes/S12-N4_deployed_pool.json', 'w'),
              indent=1, default=str)
    J2.to_parquet('/Users/um-yunsang/BARAM2026/research/nodes/S12-N4_pool.parquet', index=False)
