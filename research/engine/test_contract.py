"""Contract tests: the engine must REFUSE the exact mistakes this session actually made."""
import sys
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/engine')
from contract import Node, admit
from graph import ExcavationGraph
from select import FamilyBandit, select
import numpy as np

PROV = {'policy': 'fold-outside', 'weights': 'fold-outside 1-dof',
        'row_key': ['fold_id', 'group_id', 'forecast_kst_dtm']}
cases = []

# 1. R8: a closed axis must be refused (this is S12-N4/N18, run 3 times before it was learned)
cases.append(('R8 closed axis (ensemble blending)', Node(
    id='T1', stage='S10', title='blend one more member against DEPAVG',
    prereg={'treatment': 'add member', 'control': 'champion', 'primary_metric': 'total',
            'min_effect': 0.001, 'widens_matrix': False, 'noise_arm': None, 'prune_arm': 'none',
            'axis': 'ensemble_blending', 'provenance': PROV}), False))

# 2. R2: widening the matrix without a noise arm -- exactly the S12-N14 / S13-N7 defect
cases.append(('R2 widens matrix, no noise arm', Node(
    id='T2', stage='S6', title='add 40 new terrain columns',
    prereg={'treatment': 'add 40 cols', 'control': 'base', 'primary_metric': 'one_minus_nmae',
            'min_effect': 0.0008, 'widens_matrix': True, 'prune_arm': 'none',
            'axis': 'features_terrain', 'provenance': PROV}), False))

# 3. R3: widening without declaring a prune arm
cases.append(('R3 widens matrix, no prune arm', Node(
    id='T3', stage='S6', title='add 40 new terrain columns (noise arm ok)',
    prereg={'treatment': 'add 40 cols', 'control': 'base', 'primary_metric': 'one_minus_nmae',
            'min_effect': 0.0008, 'widens_matrix': True, 'noise_arm': '40 gaussian columns',
            'axis': 'features_terrain', 'provenance': PROV}), False))

# 4. R4: missing provenance -- the M129_GROUP_FINETUNE trap
cases.append(('R4 provenance missing', Node(
    id='T4', stage='S7', title='use a saved prediction_kwh column',
    prereg={'treatment': 'x', 'control': 'y', 'primary_metric': 'total', 'min_effect': 0.001,
            'widens_matrix': False, 'noise_arm': None, 'prune_arm': 'none',
            'axis': 'estimator_new', 'provenance': {'policy': 'unknown'}}), False))

# 5. R7 via R8: teacher recalibration, twice-replicated failure
cases.append(('R7/R8 teacher recalibration', Node(
    id='T5', stage='S5', title='isotonic-recalibrate the teacher target',
    prereg={'treatment': 'isotonic', 'control': 'pc_true', 'primary_metric': 'one_minus_nmae',
            'min_effect': 0.0008, 'widens_matrix': False, 'noise_arm': None, 'prune_arm': 'none',
            'axis': 'teacher_recalibration', 'provenance': PROV}), False))

# 6. a well-formed novel node must be ADMITTED
cases.append(('well-formed novel node', Node(
    id='T6', stage='S7', title='transductive use of supplied test-period inputs',
    prereg={'treatment': 'covariate-shift importance weights from test features',
            'control': 'unweighted', 'primary_metric': 'total', 'min_effect': 0.001,
            'widens_matrix': False, 'noise_arm': None, 'prune_arm': 'none',
            'axis': 'transductive', 'provenance': PROV}), True))

print('--- admission gate ---')
ok_all = True
for name, node, expect in cases:
    ok, why = admit(node)
    mark = 'PASS' if ok == expect else 'FAIL'
    ok_all &= (ok == expect)
    print(f'  [{mark}] {name:38s} admitted={ok}  {why[:80]}')

print('\n--- bandit: a family that keeps failing must be sampled down ---')
b = FamilyBandit()
rng = np.random.default_rng(0)
print(f'  fresh family mean p          = {b.mean("fresh"):.3f}')
for _ in range(18):
    b.update('ensemble_blending', False, partial=0.05)
print(f'  after 18 failures            = {b.mean("ensemble_blending"):.3f}')
b.update('transductive', True)
print(f'  after 1 success              = {b.mean("transductive"):.3f}')

nodes = [Node(id='A', stage='S10', title='another blend', cost_min=5, prior_effect=0.002,
              prereg={'axis': 'ensemble_blending'}, status='proposed'),
         Node(id='B', stage='S7', title='transductive', cost_min=30, prior_effect=0.004,
              prereg={'axis': 'transductive'}, status='proposed'),
         Node(id='C', stage='S6', title='unexplored family', cost_min=15, prior_effect=0.003,
              prereg={'axis': 'representation'}, status='proposed')]
wins = {}
for s in range(200):
    pick = select(nodes, b, budget_min=120, seed=s, k=1)
    if pick:
        wins[pick[0][0].id] = wins.get(pick[0][0].id, 0) + 1
print(f'  selection frequency over 200 draws: {wins}')
print(f"\nCONTRACT TESTS {'ALL PASS' if ok_all else 'FAILED'}")
