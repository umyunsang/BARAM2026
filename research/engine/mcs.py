"""Engine upgrade: Model Confidence Set replaces the hand-rolled multiplicity bump (contract R6).

Why.  The engine's arbiter currently raises its posterior threshold by an ad-hoc Sidak-style bump
as the comparison count grows.  That is a guess.  Hansen, Lunde & Nason's Model Confidence Set is
the correct instrument: given a per-row LOSS SERIES for each candidate and a block bootstrap that
respects dependence, it returns the set of models that cannot be distinguished from the best at a
stated level, with family-wise error control across ALL of them simultaneously.  The `arch`
package implements MCS, SPA and StepM and takes exactly the inputs we already produce.

Linearising our metric into a per-row loss.  Total is not a row mean -- FICR is a ratio per group
-- so we linearise with the group constants held at their full-sample values:

    Total = 0.5*(1 - (1/3) sum_g NMAE_g) + 0.5*(1/3) sum_g FICR_g
    NMAE_g = (1/n_g) sum_{i in g} e_i ,  FICR_g = (sum_{i in g} y_i u_i) / (4 sum_{i in g} y_i)

    L_i = (1/6) * [ e_i / n_g(i) ]  -  (1/6) * [ y_i u_i / (4 S_g(i)) ]

so that sum_i L_i = 0.5 - Total exactly, with n_g and S_g fixed.  Minimising mean L is therefore
identical to maximising Total, and MCS can be applied directly.
"""
from __future__ import annotations
import sys
import numpy as np, pandas as pd

sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/scratch')
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/nodes')
from lib import CAPS   # noqa: E402


def loss_series(df: pd.DataFrame, pred_col: str) -> np.ndarray:
    """Per-row loss whose SUM equals 0.5 - Total (group constants fixed at full sample)."""
    cap = df.group_id.map(CAPS).to_numpy(float)
    y = df.actual_kwh.to_numpy(float)
    scored = y >= 0.1 * cap
    e = np.abs(df[pred_col].to_numpy(float) - y) / cap
    u = np.select([e <= 0.06, e <= 0.08], [4.0, 3.0], 0.0)
    L = np.zeros(len(df))
    for g in (1, 2, 3):
        m = scored & (df.group_id.to_numpy() == g)
        n_g = m.sum()
        S_g = y[m].sum()
        L[m] = (1.0 / 6.0) * (e[m] / n_g) - (1.0 / 6.0) * (y[m] * u[m] / (4.0 * S_g))
    return L


def run_mcs(df: pd.DataFrame, pred_cols: list[str], size: float = 0.10,
            reps: int = 2000, block: int = 168, seed: int = 20260807):
    """block is in ROWS; 168 = one week x 24 h for a group-stacked hourly frame."""
    from arch.bootstrap import MCS, StationaryBootstrap  # noqa: F401
    L = pd.DataFrame({c: loss_series(df, c) for c in pred_cols})
    mcs = MCS(L, size=size, reps=reps, block_size=block, method='R', seed=seed)
    mcs.compute()
    return mcs, L
