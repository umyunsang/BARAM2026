"""S16-N5 * does a wider action pool raise the attainable band-hit ceiling, or only the oracle?

S16-N4 left one coordinate open: on the recoverable set (14.82% of scored rows, where the champion
misses and some member hits) 70.7% have exactly ONE hitting member out of four, and our gate finds
it 42.05% of the time against a 25% chance baseline.  That is real skill and still not enough --
the best switching rule scored +0.000063 against the champion.

Two ways the pool could help, and they have opposite consequences:
  GOOD  a new member hits on rows where all four currently miss, RAISING "at least one hits"
        above 0.4985 and shrinking the fraction of recoverable rows that are 1-in-k needles.
  BAD   a new member only adds another option on rows already covered, raising the ORACLE while
        making selection harder -- more needles, same haystack.

Candidates to add, all already built: the composed pipeline member of S15 (best point accuracy
this project has produced, 1-NMAE 0.867079), the analog lineage M244 (error correlation 0.857 with
D, the most decorrelated family available), and the eight remaining deployed stems that carry all
three folds.  For each expansion this node reports the oracle, "at least one hits", the needle
fraction, and the CONDITIONAL ceiling: what a gate with our measured 0.42 recoverable-set accuracy
would actually deliver.
"""
import sys, json
import numpy as np, pandas as pd
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/scratch')
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/nodes')
from lib import CAPS, FOLDS, official_total
from loop_lib import canonical_keys, align_prob, load_depavg, utility_frames, fo_policy, KEY, DEP, AB

N = '/Users/um-yunsang/BARAM2026/research/nodes/'
EXTRA = ['M269_PROBE_TOP100', 'M271_M269REPRO', 'M72_BIN020', 'M68_SITEWIND_CLASS_ITER',
         'M84_LEAVES031', 'M93_POWER_QUANTILE', 'M96_ORDINAL_CUMULATIVE', 'M98_ORDINAL_BIN025']

if __name__ == '__main__':
    R = canonical_keys(); dep = load_depavg()
    Dm, _, _ = fo_policy(utility_frames(align_prob('D', R), R), R)
    J = Dm[KEY + ['actual_kwh', 'prediction_kwh']].rename(columns={'prediction_kwh': 'D'}).merge(dep, on=KEY)
    for stem, pol in DEP.items():
        fr = []
        for f in FOLDS:
            d = pd.read_parquet(AB + f'{stem}-{f}-policies.parquet').copy(); d['fold_id'] = f
            fr.append(d[KEY + [pol]].rename(columns={pol: stem}))
        J = J.merge(pd.concat(fr, ignore_index=True), on=KEY)
    # the composed member from S15
    PC = np.mean([np.load(N + f'S15-N9_prob_{s}.npy') for s in (20260803, 20260804, 20260805)], axis=0)
    Cm, _, _ = fo_policy(utility_frames(PC, R), R)
    J = J.merge(Cm[KEY + ['prediction_kwh']].rename(columns={'prediction_kwh': 'COMPOSED'}), on=KEY)
    # the analog lineage
    parts = []
    for f in FOLDS:
        d = pd.read_parquet(AB + f'M244_RARE_EVENT_CORRECTED_ANALOG_Q234-{f}.parquet').copy()
        d['fold_id'] = f
        parts.append(d[KEY + ['prediction_kwh']].rename(columns={'prediction_kwh': 'ANALOG'}))
    J = J.merge(pd.concat(parts, ignore_index=True), on=KEY, how='left')
    for stem in EXTRA:
        fr = []
        for f in FOLDS:
            d = pd.read_parquet(AB + f'{stem}-{f}-policies.parquet').copy(); d['fold_id'] = f
            pol = [c for c in d.columns if c.startswith('T0.6_G0.35')] or \
                  [c for c in d.columns if c.startswith('T0.5_G0.5')] or \
                  [c for c in d.columns if c not in ('forecast_id', 'forecast_kst_dtm', 'group_id',
                                                     'actual_kwh', 'fold_id', 'MEDIAN')][:1]
            fr.append(d[KEY + [pol[0]]].rename(columns={pol[0]: stem}))
        J = J.merge(pd.concat(fr, ignore_index=True), on=KEY, how='left')
    J = J.dropna().reset_index(drop=True)
    cap = J.group_id.map(CAPS).to_numpy(); y = J.actual_kwh.to_numpy() / cap
    base = J[['group_id', 'actual_kwh']]
    J['CHAMPION'] = 0.30 * J.D + 0.70 * J.DEPAVG
    ch = J.CHAMPION.to_numpy() / cap
    scored = y >= 0.10
    hit_ch = np.abs(ch - y) <= 0.06
    print(f'rows {len(J)}  scored {int(scored.sum())}  champion hit {hit_ch[scored].mean():.4f}')

    POOLS = {
        'base 4': ['D'] + list(DEP),
        '+composed': ['D'] + list(DEP) + ['COMPOSED'],
        '+analog': ['D'] + list(DEP) + ['ANALOG'],
        '+both': ['D'] + list(DEP) + ['COMPOSED', 'ANALOG'],
        'all 14': ['D'] + list(DEP) + ['COMPOSED', 'ANALOG'] + EXTRA,
    }
    print(f'\n{"pool":12s} {"k":>3s} {"anyhit":>8s} {"oracle":>9s} {"recov":>7s} '
          f'{"needle%":>8s} {"gate@0.42":>10s}')
    out = {}
    for nm, cols in POOLS.items():
        Acf = J[cols].to_numpy() / cap[:, None]
        hit = np.abs(Acf - y[:, None]) <= 0.06
        anyh = hit.any(1)
        e = np.abs(Acf - y[:, None])
        u = np.select([e <= 0.06, e <= 0.08], [4.0, 3.0], 0.0)
        mg = J.groupby('group_id').actual_kwh.transform('mean').to_numpy() / cap
        contrib = -e + (y[:, None] * u) / (4.0 * mg[:, None])
        orc = Acf[np.arange(len(J)), np.argmax(contrib, 1)]
        s_or = official_total(base.assign(prediction_kwh=orc * cap))['total']
        rec = scored & (~hit_ch) & anyh
        needle = float((hit[rec].sum(1) == 1).mean()) if rec.sum() else float('nan')
        # what a gate with our measured recoverable-set accuracy would deliver
        gate_hits = hit_ch[scored].mean() + 0.4205 * rec[scored].mean()
        out[nm] = dict(k=len(cols), anyhit=float(anyh[scored].mean()), oracle=s_or,
                       recoverable=float(rec[scored].mean()), needle=needle,
                       gate_at_042=float(gate_hits))
        print(f'  {nm:10s} {len(cols):3d} {anyh[scored].mean():8.4f} {s_or:9.6f} '
              f'{rec[scored].mean():7.4f} {needle:8.3f} {gate_hits:10.4f}')
    print(f'\n  champion u=4 rate {hit_ch[scored].mean():.4f}')
    print('  "gate@0.42" = the u=4 rate a gate with our MEASURED recoverable-set accuracy delivers.')
    print('  Widening the pool raises the ORACLE, but if it also raises the needle fraction the')
    print('  attainable number moves far less than the oracle does.')
    json.dump(out, open(N + 'S16-N5_pool_expand.json', 'w'), indent=1, default=str)
