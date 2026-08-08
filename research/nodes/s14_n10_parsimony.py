"""S14-N10 * parsimony selection INSIDE the model confidence set.

Three measurements now point at the same decision and none of them is about finding a better
model.

 (1) S14-N4/MCS.  The 10% Model Confidence Set over 23 candidates, with family-wise error control
     and a stationary block bootstrap, contains EIGHT models -- and plain DEPAVG (Total 0.632592)
     is one of them alongside the champion (0.636184).  The champion's +0.003592 is not
     distinguishable.
 (2) S12-N19.  Moving from our 2023 selection window to the graded period, the organiser's plain
     RandomForest GAINS +0.012925 of 1-NMAE while our pipeline LOSES -0.003091 -- a relative
     transfer disadvantage of 0.016016, larger than the 0.014839 we need.  Complexity that was
     selected on 2023 is what does not survive.
 (3) S14-N9.  Per-group blend weights oscillate across folds ({1:0.0,2:0.4,3:0.5} / {1:0.0,2:0.3,
     3:0.5} / {1:0.5,2:0.3,3:0.3}) and James-Stein shrinkage improves on free estimation
     (0.635483 > 0.634550 > 0.634375) but never beats COMPLETE shrinkage, i.e. the pooled
     estimator.  Every time this project has been allowed to fit something per group, total
     shrinkage has won.

The classical response, and the one the foundation lane cited as E2, is the one-standard-error
rule: among models that are statistically indistinguishable, take the simplest.  This node makes
that concrete by counting, for each member of the confidence set, the number of quantities that
were FITTED ON THE DEVELOPMENT DATA, and reports the parsimony frontier.
"""
import sys, json
import numpy as np, pandas as pd
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/scratch')
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/nodes')
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/engine')
from lib import CAPS, FOLDS, official_total
from loop_lib import canonical_keys, align_prob, load_depavg, utility_frames, fo_policy, KEY
from arbiter import paired_bootstrap

N = '/Users/um-yunsang/BARAM2026/research/nodes/'
MCS_INCLUDED = ['CHAMPION', 'BL_KNN120', 'BL_KNN60', 'BL_DL', 'BL_DV', 'DEPAVG', 'BL_DG', 'BL_KNN300']

# fitted-on-development-data degrees of freedom, counted honestly
DOF = {
    'DEPAVG':   {'members': 3, 'policy_per_member': 0, 'blend_weight': 0, 'member_training': 3,
                 'note': 'three deployed stems at fixed documented policies, uniform average; '
                         'no weight and no policy is re-fitted by us'},
    'CHAMPION': {'members': 4, 'policy_per_member': 1, 'blend_weight': 1, 'member_training': 4,
                 'note': 'DEPAVG plus member D, whose (T,G) policy is chosen fold-outside and '
                         'whose blend weight is chosen fold-outside'},
}
for m in ['BL_KNN120', 'BL_KNN60', 'BL_KNN300', 'BL_DL', 'BL_DV', 'BL_DG']:
    DOF[m] = {'members': 4, 'policy_per_member': 1, 'blend_weight': 1, 'member_training': 4,
              'note': 'same shape as CHAMPION with a different own-member'}

if __name__ == '__main__':
    R = canonical_keys(); dep = load_depavg()
    Dm, _, _ = fo_policy(utility_frames(align_prob('D', R), R), R)
    J = Dm[KEY + ['actual_kwh', 'prediction_kwh']].rename(columns={'prediction_kwh': 'D'}).merge(dep, on=KEY)
    J['CHAMPION'] = 0.30 * J.D + 0.70 * J.DEPAVG
    base = J[['group_id', 'actual_kwh']]

    print('=== the eight models inside the 10% model confidence set ===')
    rows = []
    for m in ['CHAMPION', 'DEPAVG']:
        s = official_total(base.assign(prediction_kwh=J[m]))
        d = DOF[m]
        fitted = d['policy_per_member'] + d['blend_weight']
        rows.append(dict(model=m, total=s['total'], one_minus_nmae=s['one_minus_nmae'],
                         ficr=s['ficr'], fitted_dof=fitted, members=d['members']))
    T = pd.DataFrame(rows)
    print(T.round(6).to_string(index=False))
    for m in ['CHAMPION', 'DEPAVG']:
        print(f'  {m}: {DOF[m]["note"]}')

    print('\n=== paired comparison, champion vs the simplest member of the set ===')
    cmp = J[KEY + ['actual_kwh']].copy()
    cmp['cand'] = J.DEPAVG; cmp['champ'] = J.CHAMPION
    r = paired_bootstrap(cmp, 'cand', 'champ')
    print(f'  DEPAVG - CHAMPION: point {r["point_delta"] if "point_delta" in r else r["point_cand"]-r["point_champ"]:+.6f}  '
          f'paired mean {r["paired_mean"]:+.6f}  sd {r["paired_sd"]:.6f}  '
          f'95% CI [{r["ci95"][0]:+.6f}, {r["ci95"][1]:+.6f}]')
    print(f'  P(DEPAVG better) = {r["p_better"]:.3f}')
    print('  MCS verdict: both are IN the 10% set, so the difference is not resolvable under')
    print('  family-wise error control across the 23 candidates that were compared.')

    print('\n=== what the one-standard-error rule says ===')
    print('  Among statistically indistinguishable models, prefer the one with fewer quantities')
    print('  fitted on the development data:')
    print(f'    DEPAVG   fitted dof = 0  (fixed policies, uniform average)')
    print(f'    CHAMPION fitted dof = 2  (member D policy chosen fold-outside + blend weight)')
    print('  and S12-N19 independently measured that the extra complexity is exactly what fails to')
    print('  transfer: the organiser baseline gains +0.012925 of 1-NMAE across the period change')
    print('  while our lineage loses -0.003091.')

    print('\n=== what this does NOT say ===')
    print('  It does not say DEPAVG scores higher locally -- it does not (0.632592 vs 0.636184).')
    print('  It says the local ranking between them is inside the noise once multiplicity is')
    print('  controlled, and that every independent piece of transfer evidence favours the simpler')
    print('  member. The decision is a risk preference, not a measurement, and it should be taken')
    print('  explicitly rather than inherited from an unadjusted local maximum.')
    json.dump({'table': T.to_dict('records'), 'paired': r, 'dof': DOF},
              open(N + 'S14-N10_parsimony.json', 'w'), indent=1, default=str)
