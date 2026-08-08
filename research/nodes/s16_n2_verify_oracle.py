"""S16-N2 * verify the S16 lane's central claim independently before building on it.

The lane reports that this project has been reading the wrong correlation.  Our members are
0.90-0.99 correlated in CONTINUOUS error -- the number that closed the ensembling axis -- but only
0.41-0.43 correlated in the BAND-HIT INDICATOR, which is what the metric actually pays for.  It
further reports u=4 hit rates of .3424 (D), .3378 (M102), .3351 (M113), .3338 (M115), .3503
(champion), while AT LEAST ONE of the four hits on .4985; that averaging manufactures a hit no
member had on only 0.79% of rows while discarding 15.6pp of hits members already had; and that a
per-row oracle over the four actions we already own scores 0.723333, i.e. +0.087149, about 3.4x
the remaining gap.

If that holds, the diagnosis of the last four sessions inverts: we are not short of diversity, we
are short of CONDITIONING, and the information needed to reach the target is already inside
artifacts we possess.
"""
import sys, json
import numpy as np, pandas as pd
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/scratch')
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/nodes')
from lib import CAPS, FOLDS, official_total
from loop_lib import canonical_keys, align_prob, load_depavg, utility_frames, fo_policy, KEY, DEP, AB

N = '/Users/um-yunsang/BARAM2026/research/nodes/'

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
    J['CHAMPION'] = 0.30 * J.D + 0.70 * J.DEPAVG
    MEM = ['D'] + list(DEP)
    cap = J.group_id.map(CAPS).to_numpy(); y = J.actual_kwh.to_numpy()
    base = J[['group_id', 'actual_kwh']]
    print(f'champion reproduces: '
          f'{official_total(base.assign(prediction_kwh=J.CHAMPION))["total"]:.13f}')

    scored = y >= 0.1 * cap
    hit = {m: (np.abs(J[m].to_numpy() - y) / cap <= 0.06) for m in MEM + ['CHAMPION', 'DEPAVG']}
    print('\n--- u=4 band-hit rate on scored rows ---')
    for m in MEM + ['DEPAVG', 'CHAMPION']:
        print(f'  {m:16s} {hit[m][scored].mean():.4f}')
    anyhit = np.zeros(len(J), bool)
    for m in MEM:
        anyhit |= hit[m]
    print(f'  {"AT LEAST ONE of 4":16s} {anyhit[scored].mean():.4f}')

    print('\n--- correlation: CONTINUOUS error vs BAND-HIT indicator ---')
    Ec = pd.DataFrame({m: (J[m].to_numpy() - y) / cap for m in MEM})[scored]
    Eb = pd.DataFrame({m: hit[m].astype(float) for m in MEM})[scored]
    cc = Ec.corr().where(~np.eye(len(MEM), dtype=bool)).stack()
    cb = Eb.corr().where(~np.eye(len(MEM), dtype=bool)).stack()
    print(f'  continuous error : min {cc.min():.4f}  mean {cc.mean():.4f}  max {cc.max():.4f}')
    print(f'  band-hit indicator: min {cb.min():.4f}  mean {cb.mean():.4f}  max {cb.max():.4f}')
    print('  (the ensembling axis was closed on the first row; the metric pays on the second)')

    print('\n--- what averaging does to hits ---')
    ch = hit['CHAMPION']
    made = (ch & ~anyhit)[scored].mean()
    lost = (anyhit & ~ch)[scored].mean()
    print(f'  hits the average MANUFACTURES that no member had : {made:.4f}')
    print(f'  hits members had that the average DISCARDS        : {lost:.4f}')

    print('\n--- per-row oracle over the four actions we already own ---')
    Aa = J[MEM].to_numpy()
    e = np.abs(Aa - y[:, None]) / cap[:, None]
    u = np.select([e <= 0.06, e <= 0.08], [4.0, 3.0], 0.0)
    contrib = -e + (y[:, None] / cap[:, None]) * u / 4.0
    k = np.argmax(contrib, axis=1)
    orc = Aa[np.arange(len(J)), k]
    s_or = official_total(base.assign(prediction_kwh=orc))
    s_ch = official_total(base.assign(prediction_kwh=J.CHAMPION))
    print(f'  oracle Total = {s_or["total"]:.6f}   (1-NMAE={s_or["one_minus_nmae"]:.6f} '
          f'FICR={s_or["ficr"]:.6f})')
    print(f'  champion     = {s_ch["total"]:.6f}')
    print(f'  headroom     = {s_or["total"]-s_ch["total"]:+.6f}   '
          f'(remaining gap to 0.66 from the honest 0.634573 is 0.025427)')
    print(f'  oracle top-1 share by member: '
          f'{ {MEM[i]: round(float((k==i)[scored].mean()),3) for i in range(len(MEM))} }')

    print('\n--- how much of the oracle does a gate need to capture? ---')
    for cap_frac in (0.05, 0.10, 0.20, 0.30, 0.50):
        print(f'  capture {cap_frac:.0%} of the oracle -> '
              f'{s_ch["total"] + cap_frac*(s_or["total"]-s_ch["total"]):.6f}')
    json.dump({'hit_rates': {m: float(hit[m][scored].mean()) for m in MEM + ['DEPAVG', 'CHAMPION']},
               'any_of_four': float(anyhit[scored].mean()),
               'corr_continuous': [float(cc.min()), float(cc.mean()), float(cc.max())],
               'corr_bandhit': [float(cb.min()), float(cb.mean()), float(cb.max())],
               'oracle_total': s_or['total'], 'champion': s_ch['total'],
               'manufactured': float(made), 'discarded': float(lost)},
              open(N + 'S16-N2_verify_oracle.json', 'w'), indent=1, default=str)
