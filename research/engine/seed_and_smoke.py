"""Seed the excavation graph with the current champion and smoke-test the arbiter end to end.

The smoke test re-runs, through the new engine, a comparison whose answer is already known from
S12-N17 (DV_BLEND versus the champion: paired delta -0.000816, paired sd 0.000545,
P(better) 0.062).  If the engine reproduces those numbers the arbitration path is correct.
"""
import sys, json
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/engine')
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/nodes')
import numpy as np, pandas as pd
from graph import ExcavationGraph
from contract import Node, CHAMPION_SEED, CLOSED_AXES
from arbiter import arbitrate, KEY
from loop import status
from loop_lib import (canonical_keys, load_depavg, align_prob, utility_frames,
                      fo_policy, CAPS)

g = ExcavationGraph.load()
if not g.nodes:
    root = Node(id='C000', stage='S0', title=CHAMPION_SEED['title'],
                principle='incumbent, reproduced exactly by research/nodes/loop_lib.py',
                status='champion', result=CHAMPION_SEED['score'],
                prereg={'treatment': 'incumbent', 'control': 'n/a', 'primary_metric': 'total',
                        'min_effect': 0.0, 'widens_matrix': False, 'noise_arm': None,
                        'prune_arm': 'none', 'axis': 'incumbent',
                        'provenance': CHAMPION_SEED['provenance']})
    g.nodes['C000'] = root; g.G.add_node('C000')
    g.champion = 'C000'; g.champion_record = dict(CHAMPION_SEED)
    print('seeded champion C000')

R = canonical_keys(); dep = load_depavg()
Dm, _, _ = fo_policy(utility_frames(align_prob('D', R), R), R)
J = Dm[KEY + ['actual_kwh', 'prediction_kwh']].rename(columns={'prediction_kwh': 'D'}).merge(dep, on=KEY)
J['champ'] = 0.30 * J.D + 0.70 * J.DEPAVG
DVm, _, _ = fo_policy(utility_frames(align_prob('DV', R), R), R)
J = J.merge(DVm[KEY + ['prediction_kwh']].rename(columns={'prediction_kwh': 'DV'}), on=KEY)
J['cand'] = 0.30 * J.DV + 0.70 * J.DEPAVG

took, arb = arbitrate(J, 'cand', 'champ', n_comparisons=1)
print('\n--- SMOKE TEST: DV_BLEND vs champion (S12-N17 said delta -0.000816, sd 0.000545, P 0.062) ---')
for k in ('point_champ', 'point_cand', 'point_delta', 'paired_mean', 'paired_sd', 'p_better',
          'p_required_adjusted', 'took_champion'):
    print(f'  {k:22s} {arb[k]}')

print('\n--- engine status ---')
print(status(g))
print('\n--- closed-axis register (refused at admission) ---')
for k in CLOSED_AXES:
    print(f'  {k}')
g.save()
print('\ngraph saved ->', '/Users/um-yunsang/BARAM2026/research/engine/graph.json')
print(g.summary())
