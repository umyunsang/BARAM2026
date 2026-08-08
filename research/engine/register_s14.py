"""Register the S14 foundation lane's 17 hypotheses as engine nodes and run admission."""
import sys
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/engine')
from graph import ExcavationGraph
from contract import Node

PROV = {'policy': 'fold-outside', 'weights': 'fold-outside',
        'row_key': ['fold_id', 'group_id', 'forecast_kst_dtm']}


def mk(i, stage, title, axis, mech, principle, cost, eff, widens=False, extra=None):
    pre = {'treatment': mech, 'control': 'champion C000', 'primary_metric': 'total',
           'min_effect': 0.0010, 'widens_matrix': widens,
           'noise_arm': ('equal-count gaussian columns' if widens else None),
           'prune_arm': ('declare at scripting time' if widens else 'none'),
           'axis': axis, 'provenance': PROV, 'six_component_bin': extra or '1.5/1.6'}
    return Node(id=f'F{i:02d}', stage=stage, title=title, principle=principle,
                mechanism=mech, parent='C000', prereg=pre, cost_min=cost, prior_effect=eff)


NODES = [
 mk(1, 'S7', 'conditional-density head + explicit Bayes action', 'bayes_action',
    'replace the point head with a 50-bin conditional density and deploy the exact Bayes action '
    'a*=argmax_a[w2(p~*K)(a) - w1 E|a-y|1{y>=c}] per row instead of a grid-searched (T,G) scalar',
    'Murphy 5.7 Bayes action is a functional of p(y|x); Boyd EE364a 6.5 deadzone penalty', 45, 0.004,
    extra='1.3 evaluation criteria + 1.5'),
 mk(2, 'S7', 'train under the dead-zone penalty itself', 'deadzone_loss',
    'custom booster objective 0.5|r|/C - 0.5 w_y K(r) with a smoothed trapezoid K, instead of '
    'fitting L1/L2 and post-processing',
    'Boyd EE364a slide 6.5: penalty shape determines the residual distribution', 25, 0.003,
    extra='1.3 evaluation criteria'),
 mk(3, 'S8', 'Optimization Verification test', 'diagnostic',
    'for each scored row compare the TRAINING objective at our deployed action vs at the realised '
    'Total-maximising action; if the objective prefers ours >50% of the time the objective, not '
    'the search, is at fault', 'Ng MLY ch.44-45', 3, 0.000,
    extra='diagnostic, gates 1.5/1.6'),
 mk(4, 'S7', 'sharpness routing', 'bayes_action',
    'partition rows by predictive spread into FICR-winnable and FICR-hopeless; apply the Bayes '
    'action only where it can pay', '18.657 excess-risk weighting', 15, 0.002),
 mk(5, 'S5', 'truncation-corrected target', 'truncation',
    'retarget at median(y | x, y >= 0.1C) via explicit truncated estimation rather than the '
    'unconditional median', 'Greene ch.19-6 truncated regression', 20, 0.002,
    extra='1.2 assumptions'),
 mk(6, 'S8', 'binned signed-residual curve on the scored subpopulation', 'diagnostic',
    'fit E[y-yhat | yhat, spread, group] restricted to scored rows -- the empirical inverse-Mills '
    'term; a flat curve kills the truncation family in ten minutes', 'Imbens ARE213', 5, 0.001),
 mk(7, 'S5', 'metric-matched row weights', 'metric_weights',
    'train with w_i proportional to lam1*1{y>=c}/(3 n_g) + lam2*y_i/(3 n_g), the weights the score '
    'literally implies including the macro-average over 3 unequal groups',
    'Gelman-Hill 12.16 / the score algebra itself', 20, 0.002, extra='1.3 evaluation criteria'),
 mk(8, 'S6', 'NWP error-state features from recent verification residuals', 'information_set',
    'features describing the error state of the current issuance: LDAPS/GFS verification residuals '
    'against observed generation over D-2..D-8', 'Ng MLY ch.57 information axis', 40, 0.008,
    widens=True, extra='1.1 problem class'),
 mk(9, 'S6', 'spatio-temporal displacement of the NWP field', 'information_set',
    'estimate a field displacement (which cell/lag best matches realised ridge wind, per regime) '
    'and read the forecast from the displaced location -- changes what the member SEES, the only '
    'kind of change that can move rho', 'ESL 15.2 ensemble ambiguity', 60, 0.005, widens=True,
    extra='1.1 problem class'),
 mk(11, 'S4', 'transductive estimation of the metric normalisers', 'transductive',
    'use the supplied test x to estimate E[sum y] and E[N_scored] over the graded period and DERIVE '
    'the lam1:lam2 weight between the two halves instead of grid-searching T/G',
    'Vapnik transduction; ESL 7.10.2 licenses unsupervised use of test x', 15, 0.002,
    extra='1.3 evaluation criteria'),
 mk(13, 'S7', 'transductive regime discovery with shrinkage', 'transductive',
    'cluster train union test inputs into atmospheric regimes and fit per-regime policies with '
    'James-Stein shrinkage across regimes', 'ESL 7.10.2 + James-Stein', 45, 0.003),
 mk(14, 'S7', 'self-training on test inputs', 'transductive',
    'pseudo-label test rows with the champion, retrain with regularisation, check SHARPNESS on the '
    'test manifold rather than accuracy', 'SAIL/CS229M self-training expansion', 40, 0.002),
 mk(17, 'S7', 'James-Stein shrinkage of every per-group parameter', 'shrinkage',
    'any per-group quantity (bias offset, policy width, calibration slope) becomes a shrunk '
    'estimator toward the common value; 1 dof, unlike the already-rejected 3-dof per-group weights',
    'James-Stein; p=3 is exactly where the theorem bites', 15, 0.002),
]

# these three the parent has already executed this session -- register as done, not proposed
DONE = [
 ('F12', 'covariate-shift audit', 'transductive',
  'EXECUTED as S14-N1 + control S14-N3: domain AUC 0.9999 train-vs-test, but the control shows '
  '2022-23 vs 2024 = 0.9999 and adjacent quarters 2023Q2 vs Q3 = 1.0000 while the NULL random '
  'halves give 0.4990. The classifier measures elapsed time in 813 columns, not harmful shift. '
  'Importance-weighting branch CLOSED.'),
 ('F15', 'cross-group replicate noise floor', 'diagnostic',
  'EXECUTED as S14-N2 + control S14-N3: E|y1-y2| = 0.08526 implies an MAE floor of 0.0568-0.0603, '
  'but the two groups share the same NWP so the weather-model error cancels in the difference '
  '(group-specific 0.93 m/s vs common 1.49-1.78 m/s). The floor bounds only the group-specific '
  'component and is a LOWER bound, not the achievable MAE.'),
 ('F16', 'stopping rule + six-component audit', 'process',
  'SUPERSEDED by the Model Confidence Set (arch 8.0.0): the 10% MCS over 23 candidates contains 8 '
  'models INCLUDING plain DEPAVG, so the champion is not distinguishable under family-wise error '
  'control. This is strictly stronger than the eta=0.0015 ladder rule the lane proposed.'),
]

if __name__ == '__main__':
    g = ExcavationGraph.load()
    print(f'graph before: {len(g.nodes)} nodes')
    for n in NODES:
        ok, why = g.propose(n)
        print(f'  {n.id} {n.title[:52]:54s} {"ADMITTED" if ok else "REFUSED"}  {why[:70]}')
    for nid, title, axis, note in DONE:
        nd = Node(id=nid, stage='S14', title=title, parent='C000', status='arbitrated',
                  notes=note, prereg={'axis': axis})
        g.nodes[nid] = nd; g.G.add_node(nid); g.G.add_edge('C000', nid)
        print(f'  {nid} {title[:52]:54s} ALREADY EXECUTED')
    g.save()
    print(f'\ngraph after: {len(g.nodes)} nodes')
    print(g.summary())
