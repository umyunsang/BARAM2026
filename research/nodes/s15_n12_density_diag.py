"""S15-N12 * why did the composed member's density lose? accuracy or sharpness?

S15-N9 and S15-N10 both put the composed member below the honest champion baseline (0.632027 and
0.631553 against 0.634573), and its solo fell to 0.6225-0.6238 against the original D's 0.625669.
Two mutually exclusive causes demand opposite fixes:

  H-ACC    the composed features/teacher make the CLASSIFIER less accurate, so the density is
           centred worse.  Fix: the composed inputs are wrong for this member and the transplant
           is simply bad.
  H-SHARP  the density is at least as accurate but FLATTER, so the settlement argmax has less
           structure to exploit.  Fix: sharpen -- and everything else in the composition can stay.

The two are separated by measuring, on identical rows: the point accuracy of each density's mean,
its dispersion and entropy, the mass it concentrates in the +-0.06 settlement window around its
own argmax, and the resulting solo score.  Nothing here needs a refit.
"""
import sys, json
import numpy as np, pandas as pd
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/scratch')
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/nodes')
from lib import CAPS, official_total
from loop_lib import canonical_keys, align_prob, utility_frames, fo_policy, W

N = '/Users/um-yunsang/BARAM2026/research/nodes/'
SEEDS = (20260803, 20260804, 20260805)

if __name__ == '__main__':
    R = canonical_keys()
    C = (np.arange(26) + 0.5) * W
    y = R.cf.to_numpy(); g = R.group_id.to_numpy()
    capv = np.array([CAPS[x] for x in g])
    base = pd.DataFrame({'group_id': g, 'actual_kwh': y * capv})
    cand = {'D (champion)': align_prob('D', R)}
    cand['COMPOSED all stages'] = np.mean([np.load(N + f'S15-N9_prob_{s}.npy') for s in SEEDS], axis=0)
    cand['COMPOSED no-D1-density'] = np.mean([np.load(N + f'S15-N10_prob_{s}.npy') for s in SEEDS], axis=0)
    rows = []
    for nm, P in cand.items():
        mq = (P * C[None, :]).sum(1)
        sd = np.sqrt((P * (C[None, :] - mq[:, None]) ** 2).sum(1))
        ent = -(P * np.log(np.clip(P, 1e-12, None))).sum(1)
        # mass inside the +-0.06 settlement window centred on the density's own argmax
        k = P.argmax(1)
        win = np.abs(C[None, :] - C[k][:, None]) <= 0.06
        band = (P * win).sum(1)
        pt = official_total(base.assign(prediction_kwh=np.clip(mq, 0, 1.1) * capv))
        _, solo, _ = fo_policy(utility_frames(P, R), R)
        rows.append(dict(member=nm, point_1mnmae=pt['one_minus_nmae'], point_ficr=pt['ficr'],
                         mean_sd=float(sd.mean()), mean_entropy=float(ent.mean()),
                         band_mass=float(band.mean()), solo_total=solo['total']))
    T = pd.DataFrame(rows)
    print(T.round(6).to_string(index=False))
    d = T.set_index('member')
    b = d.loc['D (champion)']
    print('\n--- against the champion density ---')
    for nm in ('COMPOSED all stages', 'COMPOSED no-D1-density'):
        r = d.loc[nm]
        print(f'  {nm}:')
        print(f'    point 1-NMAE {r.point_1mnmae-b.point_1mnmae:+.6f}   '
              f'sd {r.mean_sd-b.mean_sd:+.5f}   entropy {r.mean_entropy-b.mean_entropy:+.5f}')
        print(f'    band mass    {r.band_mass-b.band_mass:+.5f}   '
              f'solo {r.solo_total-b.solo_total:+.6f}')
    acc = d.loc['COMPOSED all stages'].point_1mnmae - b.point_1mnmae
    shp = d.loc['COMPOSED all stages'].band_mass - b.band_mass
    print(f'\n  VERDICT: point accuracy {"BETTER" if acc>0 else "WORSE"} ({acc:+.6f}), '
          f'concentration {"SHARPER" if shp>0 else "FLATTER"} ({shp:+.5f})')
    print('  -> H-SHARP if accuracy is >= the champion and band mass is lower;')
    print('     H-ACC   if accuracy is also lower, in which case the transplant is simply bad.')
    json.dump(T.to_dict('records'), open(N + 'S15-N12_density_diag.json', 'w'), indent=1, default=str)
