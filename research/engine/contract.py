
"""BARAM2026 excavation engine -- the CONTRACT.

Every rule below was forced by a measurement made in this repository, and the measurement is
cited next to the rule.  Nothing here is stylistic.

R1  PAIRED ARBITRATION.  A candidate never competes on its raw score.  S12-N17 measured the
    pooled 3-fold Total with a 7-day moving-block bootstrap: the MARGINAL sd is 0.009600 but the
    PAIRED candidate-minus-champion sd is 0.00055-0.00093.  Comparing raw scores throws away an
    order of magnitude of precision, and comparing them against a target in marginal units makes
    a 0.0238 gap look like 2.5 sd when in paired units it is ~30.

R2  NOISE CONTROL FOR ANY COLUMN ADDITION.  S13-N8 measured that adding 21 columns of pure
    Gaussian noise to the 872-column surface costs -0.000411 of (1-NMAE), versus -0.000640 for a
    physically motivated block that had passed a |corr|<0.85 novelty screen.  Two thirds of the
    "rejection" was the act of adding columns.  Any node that widens the design matrix MUST ship
    an equal-count noise arm, and its effect is measured against that arm, not against the bare
    control.

R3  ADD AND PRUNE ARE ONE TREATMENT.  Same measurement: dropping 252 lag/lead columns GAINS
    +0.000245 while dropping the 301-column geom block LOSES -0.001229.  Width is a free
    parameter of every feature node, so a node that only adds is under-specified.

R4  PROVENANCE OR INADMISSIBLE.  Every reported number states (a) which policy produced each
    input, (b) whether any weight/policy/threshold was fitted in-sample or fold-outside, and
    (c) the row-alignment key set.  S12-N5 caught M129_GROUP_FINETUNE reporting solo 0.636249,
    apparently beating the champion, while its own receipt exposed per-group in-sample selection
    (`selected_policy`, `selected_fine_iteration`).  AGENTS.md records six earlier misreadings
    from the same class of defect.

R5  PRE-DECLARATION.  Treatment, control, primary metric, minimum effect size and cost are
    written into the node BEFORE it runs.  A node that reports a metric it did not pre-declare is
    marked EXPLORATORY and can never take the championship.

R6  MULTIPLICITY IS TRACKED.  S12+S13 ran ~30 treatments against one champion and every one
    landed at or below it.  The engine counts comparisons and reports the selection-adjusted
    threshold, because the probability that the best of k noise draws beats the champion grows
    with k.

R7  TEACHER REALISM IS NOT A GOAL.  S13-N3 (isotonic recalibration, -0.022 to -0.027) and S13-N9
    (per-turbine transfer factors plus a storm-control curve, which improved the teacher's own
    MAE by 5.7% and still lost downstream) independently established that making the physics
    teacher closer to metered truth injects the unpredictable availability process into a feature
    that was previously a noise-free function of measured wind.  Nodes proposing it are refused
    at admission.

R8  CLOSED AXES ARE REFUSED AT ADMISSION.  The engine holds an explicit closed-axis register with
    the measurement that closed each one, so no cycle can re-propose it.

R10 SEED FLOOR -- and the retraction of R2's evidential basis.  S15-N3 refitted one identical
    configuration under three seeds: the pooled Total spread is 0.001635 (sd 0.000828) and the
    1-NMAE spread is 0.000270.  Every feature effect this project has interpreted --
    S13-N8's noise arm -0.000411 and physical block -0.000640, its prune +0.000245, S14-N7's
    displacement -0.000329, S15-N2's B1 -0.000495 -- is between a quarter and a third of that
    spread.  They were not measurements.  R2 was written on the strength of one of them and its
    evidential basis is hereby RETRACTED: the noise arm is still cheap and still worth shipping,
    but it settles nothing on its own.
    THE RULE: every configuration is scored as the average of at least three seed refits, and no
    effect smaller than the measured seed spread may be claimed, gated on, or written into the
    closed-axis register.  Corollary, measured in the same run: seed averaging is itself worth
    +0.001129 of Total (0.602325 -> 0.603454), because the 0.934 error-correlation that closed the
    ensembling axis was measured across model CLASSES, never across SEEDS.

R9  DEPLOYMENT AVAILABILITY.  Every input a node uses must exist at the graded period's basis
    time, not merely in the development window.  Training labels end 2024-12-31 and the graded
    period is the whole of 2025 submitted at once, so ANY feature that is a function of recent
    observed generation (verification residuals, persistence, recent availability) exists on
    dev-2023 and does NOT exist for most of 2025.  Such a node would inflate the local score and
    fail on delivery -- the exact failure mode AGENTS.md records for the local->online offset.
    A node must declare `needs_recent_observations`, and if it is true it is refused.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any

PRIMARY = "total"
MIN_EFFECT = {"total": 0.0010, "one_minus_nmae": 0.0008}
SEED_FLOOR = {"total": 0.001635, "one_minus_nmae": 0.000303}   # S15-N3, 3 refits, same config
N_SEEDS = 3
P_BETTER = 0.90          # paired-bootstrap posterior threshold to take the championship
N_BOOT = 400
BLOCK_DAYS = 7

CHAMPION_SEED = {
    "id": "C000",
    "title": "DEPAVG + D, 1-dof fold-outside blend",
    "score": {"total": 0.6361842493883538,
              "one_minus_nmae": 0.8618657013604555,
              "ficr": 0.4105027974162521},
    "provenance": {
        "policy": "fold-outside (T,G) chosen on the other two folds, per member",
        "weights": "fold-outside 1-dof, w=0.30 on all three folds",
        "row_key": ["fold_id", "group_id", "forecast_kst_dtm"],
        "script": "research/nodes/loop_lib.py (exact port of s10_final3.py)"},
}

CLOSED_AXES = {
    "external_nwp": "Only GEFS is admissible (00Z lands 03:51-04:04 UTC, before the D-1 basis "
                    "time); its spread-vs-|error| correlation is 0.020-0.140 "
                    "(reports/m270_gefs_probe.md). ECMWF's 2023 archive is 0.4 deg with no 100 m "
                    "wind and lands 07:34, after the basis time.",
    "ensemble_blending": "12 deployed stems + 15 own members; minimum pairwise error correlation "
                         "0.934; S12-N4/N14/N18 found no combination above the champion and the "
                         "fold-outside gate assigns weight 0 to every new member.",
    "decision_layer": "S12-N11: member D's gamma frontier is flat -- 1-NMAE 0.863925->0.864617 "
                      "and FICR 0.377378->0.387189 rise together from gamma 0 to 40. No "
                      "accuracy/band trade remains.",
    "band_tricks": "S12-N8 ordinal smoothing, S12-N10 global affine, S12-N7 dispersion rescaling, "
                   "S12-N16 distribution recentring: all rejected fold-outside.",
    "teacher_recalibration": "R7.",
    "grid_dispersion_features": "S12-N14 component-grid 168 columns -0.000728; an independent "
                                "competitor rejected 54 grid-dispersion features on their board.",
    "lead_time_axis": "Exactly one issuance per target hour makes lead time perfectly collinear "
                      "with hour-of-day (lead = hour + 11).",
    "estimator_swap_alone": "S13-N9 plus the S7 lane's five A-grade head-to-heads: measured "
                            "estimator-class swaps deliver -3.3% to -5.9% relative MAE against "
                            "weaker baselines than ours, versus the -10.71% required.",
}


@dataclass
class Node:
    id: str
    stage: str
    title: str
    principle: str = ""
    hypothesis: str = ""
    mechanism: str = ""
    parent: str | None = None
    status: str = "proposed"     # proposed|selected|scripted|run|arbitrated|rejected|champion|refused
    prereg: dict[str, Any] = field(default_factory=dict)
    script: str | None = None
    result: dict[str, Any] = field(default_factory=dict)
    arbitration: dict[str, Any] = field(default_factory=dict)
    cost_min: float = 10.0
    prior_effect: float = 0.001
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


REQUIRED_PREREG = ("treatment", "control", "primary_metric", "min_effect",
                   "widens_matrix", "provenance")


def admit(node: Node) -> tuple[bool, str]:
    """R4/R5/R7/R8 enforced at admission."""
    missing = [k for k in REQUIRED_PREREG if k not in node.prereg]
    if missing:
        return False, f"R5 pre-declaration incomplete: missing {missing}"
    if node.prereg["primary_metric"] not in MIN_EFFECT:
        return False, f"R5 unknown primary metric {node.prereg['primary_metric']}"
    if node.prereg.get("widens_matrix") and not node.prereg.get("noise_arm"):
        return False, "R2 a node that widens the design matrix must declare a noise arm"
    if node.prereg.get("widens_matrix") and "prune_arm" not in node.prereg:
        return False, "R3 a feature node must declare its prune arm (may be 'none', explicitly)"
    p = node.prereg.get("provenance", {})
    for k in ("policy", "weights", "row_key"):
        if k not in p:
            return False, f"R4 provenance missing '{k}'"
    axis = node.prereg.get("axis")
    if axis in CLOSED_AXES:
        return False, f"R8 axis '{axis}' is closed: {CLOSED_AXES[axis]}"
    if node.prereg.get("needs_recent_observations"):
        return False, ("R9 the node needs recent observed generation at prediction time; training "
                       "labels end 2024-12-31 and the graded period is all of 2025 submitted at "
                       "once, so this input does not exist on delivery")
    return True, "admitted"
