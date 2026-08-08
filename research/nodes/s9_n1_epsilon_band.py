
"""S9-N1 · O1: epsilon-insensitive (band) loss as a custom LightGBM objective for the
physics teacher. Per research/lanes/S6_ext_C_repr.md §C4 O1: the FICR settlement band
(+-6% of cap) is exactly an epsilon-insensitive loss shape -- L1/L2 keep penalising
error inside the band, which pulls near-boundary predictions toward the wrong side.
eps is fixed a priori from the metric's own 6% band, so this is a 0-degree-of-freedom
change (not tuned on any fold).

Isolated test: only mu_params/teacher_weight are overridden here, everything else is
the frozen harness baseline (S5/S6/S7 defaults untouched), so this measures O1's OWN
effect in isolation, per this project's established one-axis-at-a-time convention
(S5_preprocessing_research.md sec 3.2). Verified empirically before this run: LightGBM's
sklearn API honors sample_weight for custom objectives (rows with weight=0 do not
influence the fit) -- see conversation record, not re-derived here.

Expected trade-off per the write-up: point accuracy (NMAE component, via pc_hat) may
get worse while FICR improves, since the epsilon-insensitive loss stops correcting
errors that are already inside the settlement band. Net effect on Total must be
measured, not assumed.
"""
import sys, json
import numpy as np
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/scratch')
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/nodes')
from harness import run, MU

EPS = 0.06  # capacity-factor units; matches the official 6% settlement band exactly


def epsilon_band_objective(y_true, y_pred):
    e = np.asarray(y_pred) - np.asarray(y_true)
    grad = np.where(np.abs(e) > EPS, np.sign(e), 0.0)
    hess = np.ones_like(e)
    return grad, hess


def teacher_weight_lowcf_zero(A):
    """Zero out rows the official metric itself excludes (actual < 0.1*cap), so the
    custom objective and the scoring metric stay aligned (per the O1 write-up's caveat)."""
    cf = A['cf'].to_numpy()
    return np.where(np.isfinite(cf) & (cf >= 0.1), 1.0, 0.0)


if __name__ == '__main__':
    mu_params = dict(MU, objective=epsilon_band_objective)
    out = run('S9-N1', 'O1_epsilon_band_objective_vs_l2_baseline',
              mu_params=mu_params, teacher_weight=teacher_weight_lowcf_zero)
    print(json.dumps(out, indent=1, default=str))
