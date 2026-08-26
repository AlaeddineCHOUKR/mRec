"""Confidence intervals and paired significance.

`EVALUATION_PROTOCOL.md` §5: a single-seed number is never reported, and any claim
of improvement carries a paired bootstrap p-value against the named baseline.
"""

from __future__ import annotations

import numpy as np


def bootstrap_ci(
    values: np.ndarray, seed: int, n_boot: int = 10_000, alpha: float = 0.05
) -> tuple[float, float, float]:
    """Mean and percentile bootstrap interval of `values`, resampled with replacement.

    Used two ways: over the five evaluation cohorts for a reported number, and over
    users when a single run needs its own interval.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError(f"expected a non-empty 1-d array, got shape {values.shape}")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[draws].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(values.mean()), float(lo), float(hi)


def paired_bootstrap(
    a: np.ndarray, b: np.ndarray, seed: int, n_boot: int = 10_000
) -> tuple[float, float]:
    """Two-sided p-value for `mean(a) - mean(b)`, pairing on the shared unit (a user).

    The null is centered by subtracting the observed mean difference, so the p-value
    is the share of resamples whose difference is at least as extreme as observed.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"paired inputs must align: {a.shape} vs {b.shape}")
    diff = a - b
    observed = float(diff.mean())
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    centered = (diff - observed)[draws].mean(axis=1)
    p = (np.sum(np.abs(centered) >= abs(observed)) + 1) / (n_boot + 1)
    return observed, float(p)
