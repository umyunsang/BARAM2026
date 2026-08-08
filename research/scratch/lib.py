
"""Shared local-evaluation helpers for the BARAM2026 excavation (read-only wrt repo)."""
from __future__ import annotations
import numpy as np, pandas as pd

CAPS = {1: 21600.0, 2: 21600.0, 3: 21000.0}

FOLDS = {
    "dev-2023-Q2": ("2023-04-01 01:00:00", "2023-07-01 00:00:00"),
    "dev-2023-Q3": ("2023-07-01 01:00:00", "2023-10-01 00:00:00"),
    "dev-2023-Q4": ("2023-10-01 01:00:00", "2024-01-01 00:00:00"),
}


def official_total(df: pd.DataFrame) -> dict:
    """df: columns group_id, actual_kwh, prediction_kwh."""
    nm, fi = {}, {}
    for g in (1, 2, 3):
        p = df[df.group_id == g]
        cap = CAPS[g]
        v = p[p.actual_kwh >= 0.1 * cap]
        err = np.abs(v.prediction_kwh.to_numpy(float) - v.actual_kwh.to_numpy(float)) / cap
        units = np.select([err <= 0.06, err <= 0.08], [4.0, 3.0], default=0.0)
        a = v.actual_kwh.to_numpy(float)
        nm[g] = float(err.mean())
        fi[g] = float((a * units).sum() / (a * 4.0).sum())
    one_m = 1.0 - float(np.mean(list(nm.values())))
    ficr = float(np.mean(list(fi.values())))
    return {"total": 0.5 * one_m + 0.5 * ficr, "one_minus_nmae": one_m, "ficr": ficr,
            "group_nmae": nm, "group_ficr": fi}


def sharpen_weights(samples: np.ndarray, temp: float, n_bins: int = 15) -> np.ndarray:
    """Per-row density-based weights over each row's own predictive samples,
    sharpened (temp<1) or flattened (temp>1) toward dense regions of that row's
    sample distribution.

    Fixes DEF-1: a uniform index-weight vector raised to any power and renormalised
    is unchanged regardless of temperature, so `temp` was previously inert. Binning
    each row's own sample *values* makes the weight depend on where the samples
    actually cluster, so temperature has a real effect.

    samples: (n_rows, n_samples) predictive sample values, e.g. quantile draws that
        are evenly spaced in probability but not in value.
    Returns: (n_rows, n_samples) weights, each row summing to 1.
    """
    n_rows, n_samples = samples.shape
    lo = samples.min(axis=1, keepdims=True)
    hi = samples.max(axis=1, keepdims=True)
    span = np.where(hi > lo, hi - lo, 1.0)
    bin_idx = np.clip(((samples - lo) / span * n_bins).astype(int), 0, n_bins - 1)
    counts = np.zeros((n_rows, n_bins))
    np.add.at(counts, (np.repeat(np.arange(n_rows), n_samples), bin_idx.ravel()), 1.0)
    density = np.take_along_axis(counts, bin_idx, axis=1)
    density = density ** (1.0 / temp)
    return density / density.sum(axis=1, keepdims=True)


def fold_mean_total(frames: dict) -> dict:
    """frames: fold_id -> df with group_id/actual_kwh/prediction_kwh."""
    per = {k: official_total(v) for k, v in frames.items()}
    return {"mean_total": float(np.mean([p["total"] for p in per.values()])),
            "mean_one_minus_nmae": float(np.mean([p["one_minus_nmae"] for p in per.values()])),
            "mean_ficr": float(np.mean([p["ficr"] for p in per.values()])),
            "per_fold": per}
