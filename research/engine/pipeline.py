"""Pipeline engineering: the forecast system as an explicit, finely-decomposed stage graph.

WHY THIS REPLACES THE PREVIOUS NODE LOOP.
The S12-S14 loop tested one treatment at a time against a frozen incumbent and arbitrated each
with a paired bootstrap.  That protocol is correctly powered for what it measures and it is the
wrong measurement.  The paired sd is 0.00055-0.00093, while a realistic per-stage improvement in
this problem class is 0.001-0.003; so almost every genuine stage upgrade is INSIDE the noise when
tested alone, and the Model Confidence Set duly returned eight indistinguishable models.  Thirty
consecutive "failures" were thirty draws at a granularity too fine to resolve.

Composed, the arithmetic changes completely: ten stage upgrades at +0.002 each is +0.02, which is
~25 paired sd and decisively detectable.  So the unit of work becomes the STAGE, the unit of
evaluation becomes the COMPOSED PIPELINE, and per-stage attribution is recovered afterwards by
reverse ablation (build the upgraded pipeline, then revert one stage at a time).

EACH STAGE CARRIES, EXPLICITLY:
  current      what this repository does today
  sota         the benchmarked state of the art for that stage
  evidence     the benchmark/paper that measures it, with the effect size THEY report
  migration    how it maps onto THIS hackathon's data structure -- the step that is usually
               skipped and where transfer actually fails
  contribution measured later by reverse ablation, never asserted
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
import json
from pathlib import Path

ROOT = Path('/Users/um-yunsang/BARAM2026/research/engine')
SPEC = ROOT / 'pipeline_spec.json'


@dataclass
class Stage:
    id: str
    group: str
    name: str
    current: str = ""
    sota: str = ""
    evidence: str = ""
    migration: str = ""
    status: str = "unresearched"   # unresearched|researched|built|ablated|frozen
    lane: str = ""
    contribution: float | None = None
    notes: str = ""

    def to_dict(self):
        return asdict(self)


# --- the decomposition -------------------------------------------------------------------
# Grouped so that deep research can be commissioned per cluster rather than per stage, while the
# BUILD and the ABLATION stay at stage granularity.
STAGES = [
    # A. target and label construction
    Stage('A1', 'target', 'label QC and metering reconstruction',
          current='hourly kpx_group_* taken as given; 10-min SCADA used only to build pc_true'),
    Stage('A2', 'target', 'availability / outage / curtailment identification',
          current='per-row gate deficit >= 0.05 drops calibration rows; block structure '
                  '(lag-1 autocorrelation 0.90) is not used'),
    Stage('A3', 'target', 'power-curve estimation and the teacher target',
          current='one 4-parameter curve per group, x^k form, no storm-control region, '
                  'integrated over per-turbine 10-min wind'),
    Stage('A4', 'target', 'target transformation and support',
          current='raw capacity factor in [0,1]; classifier discretises to 26 bins of width 0.04'),

    # B. NWP ingestion and site transfer
    Stage('B1', 'nwp', 'grid-to-site reduction',
          current='inverse-distance weighting on horizontal distance, plus box order statistics'),
    Stage('B2', 'nwp', 'vertical extrapolation to hub height',
          current='fixed power-law and two-point log-law from 10/50 m (LDAPS) and 80/100 m (GFS)'),
    Stage('B3', 'nwp', 'systematic bias correction / MOS of the NWP fields',
          current='none as a preprocessing step; the learner absorbs bias implicitly'),
    Stage('B4', 'nwp', 'terrain and micro-siting adjustment',
          current='geometric block and G2 encoding; no DEM, no exposure index'),
    Stage('B5', 'nwp', 'multi-source combination',
          current='LDAPS and GFS columns concatenated into one matrix'),

    # C. features and representation
    Stage('C1', 'repr', 'atmospheric regime and stability representation',
          current='atm__ block: shear exponents, bulk Richardson proxy, theta gradients, BLH'),
    Stage('C2', 'repr', 'temporal structure within the issuance',
          current='lag/lead +-1,2,3 and +-6 h, batch anomaly and rank'),
    Stage('C3', 'repr', 'dimensionality and feature selection',
          current='872 columns, top-150 by teacher gain, colsample 0.4; no FDR control'),

    # D. estimator and uncertainty
    Stage('D1', 'model', 'point estimator',
          current='LightGBM L2 teacher on pc_true, then a downstream learner'),
    Stage('D2', 'model', 'conditional distribution estimator',
          current='26-class DART multiclass softmax over discretised capacity factor'),
    Stage('D3', 'model', 'calibration of the predictive distribution',
          current='a temperature exponent T inside the decision grid; no separate calibration step'),
    Stage('D4', 'model', 'multi-task / hierarchical structure across the three groups',
          current='pooled fit with group one-hots; no partial pooling'),

    # E. decision and combination
    Stage('E1', 'decide', 'action selection under the settlement metric',
          current='argmax of an expected-utility surface over a 0.0025 action grid, (T,G) chosen '
                  'fold-outside'),
    Stage('E2', 'decide', 'combination across members',
          current='0.30 * D + 0.70 * DEPAVG, an average of four ACTIONS'),
    Stage('E3', 'decide', 'constraints and post-processing',
          current='per-group soft cap {0.985, 0.989, 1.005}; clip to [0, cap]'),

    # F. protocol
    Stage('F1', 'protocol', 'validation design',
          current='dev-2023 Q2/Q3/Q4 expanding window -- CONTAINS NO WINTER, while the graded '
                  'period is all of 2025'),
    Stage('F2', 'protocol', 'model selection and arbitration',
          current='paired moving-block bootstrap plus a Model Confidence Set (arch 8.0.0)'),
    Stage('F3', 'protocol', 'delivery and submission construction',
          current='not yet built for the 2025 graded period'),
]


def load() -> dict[str, Stage]:
    if SPEC.exists():
        d = json.load(open(SPEC))
        return {k: Stage(**v) for k, v in d['stages'].items()}
    return {s.id: s for s in STAGES}


def save(stages: dict[str, Stage]):
    json.dump({'stages': {k: v.to_dict() for k, v in stages.items()}},
              open(SPEC, 'w'), indent=1, ensure_ascii=False)


CLUSTERS = {
    'target': ['A1', 'A2', 'A3', 'A4'],
    'nwp': ['B1', 'B2', 'B3', 'B4', 'B5'],
    'repr': ['C1', 'C2', 'C3'],
    'model': ['D1', 'D2', 'D3', 'D4'],
    'decide': ['E1', 'E2', 'E3'],
    'protocol': ['F1', 'F2', 'F3'],
}

if __name__ == '__main__':
    st = load(); save(st)
    print(f'{len(st)} stages in {len(CLUSTERS)} research clusters\n')
    for c, ids in CLUSTERS.items():
        print(f'[{c}]')
        for i in ids:
            s = st[i]
            print(f'  {s.id}  {s.name:52s} {s.status}')
    print(f'\nspec -> {SPEC}')
