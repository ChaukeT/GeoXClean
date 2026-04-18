"""
Value transforms for geostatistical estimation.

  - Normal-score transform / back-transform
  - Log transform
  - Indicator transform
  - Top-cut (grade capping)

Reference: CLAUDE_CODE_PROMPT_FastRBF_Engine.md
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import PchipInterpolator
from scipy.stats import norm

logger = logging.getLogger(__name__)


@dataclass
class NormalScoreTable:
    """
    Stores the forward/inverse transform mapping.

    ``original_sorted`` and ``normal_sorted`` are paired arrays
    that define the monotonic mapping between original and
    normal-score spaces.
    """

    original_sorted: NDArray[np.float64]
    normal_sorted: NDArray[np.float64]
    n: int


def normal_score_transform(
    values: NDArray[np.float64],
    seed: int = 42,
) -> Tuple[NDArray[np.float64], NormalScoreTable]:
    """
    Transform values to standard normal distribution.

    Stores the transform table for back-transformation.
    Handles ties using random jitter.

    Parameters
    ----------
    values : (N,) ndarray
    seed : int
        Random seed for jitter (determinism).

    Returns
    -------
    ns_values : (N,) ndarray
        Transformed values in standard normal space.
    table : NormalScoreTable
        Mapping table for back-transformation.
    """
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    rng = np.random.Generator(np.random.PCG64(seed))

    # Break ties with random jitter scaled to data range.
    # A fixed ±1e-10 is too small for datasets with many identical
    # values (assay detection limits, zeros).  Scaling to data range
    # × 1e-8 ensures all ties are broken without distorting ranks.
    data_range = float(np.max(values) - np.min(values))
    jitter_scale = max(data_range * 1e-8, 1e-10)
    jitter = rng.uniform(-jitter_scale, jitter_scale, size=n)
    jittered = values + jitter

    # Rank → probability → normal quantile
    ranks = np.argsort(np.argsort(jittered)).astype(np.float64)
    # Probability using Hazen formula: p = (rank + 0.5) / N
    probs = (ranks + 0.5) / n
    ns_values = norm.ppf(probs)

    # Clip extreme quantiles
    ns_values = np.clip(ns_values, -6.0, 6.0)

    # Build monotonic lookup table (sorted by original values)
    sort_idx = np.argsort(values)
    table = NormalScoreTable(
        original_sorted=values[sort_idx].copy(),
        normal_sorted=ns_values[sort_idx].copy(),
        n=n,
    )

    return ns_values, table


def normal_score_backtransform(
    ns_values: NDArray[np.float64],
    table: NormalScoreTable,
) -> NDArray[np.float64]:
    """
    Back-transform from normal score space to original units.

    Uses PCHIP monotonic interpolation on the stored transform table.

    Parameters
    ----------
    ns_values : (M,) ndarray
        Values in normal-score space.
    table : NormalScoreTable

    Returns
    -------
    original_values : (M,) ndarray
    """
    ns_values = np.asarray(ns_values, dtype=np.float64)

    # Build PCHIP interpolator: normal → original
    # Ensure strictly increasing for interpolation
    ns_sorted = table.normal_sorted
    orig_sorted = table.original_sorted

    # Remove duplicates in normal scores
    unique_mask = np.diff(ns_sorted, prepend=-np.inf) > 1e-15
    ns_unique = ns_sorted[unique_mask]
    orig_unique = orig_sorted[unique_mask]

    if len(ns_unique) < 2:
        return np.full_like(ns_values, orig_unique[0] if len(orig_unique) > 0 else 0.0)

    interp = PchipInterpolator(ns_unique, orig_unique, extrapolate=True)
    result = interp(ns_values)

    # Clamp to the observed data range.  PCHIP with extrapolate=True
    # uses polynomial extrapolation beyond the table endpoints, which
    # can produce wildly unrealistic values (negative grades, values
    # exceeding any physical maximum).  Clamping ensures the
    # back-transform never exceeds the original data bounds.
    data_min = float(orig_unique[0])
    data_max = float(orig_unique[-1])
    result = np.clip(result, data_min, data_max)

    n_clamped = int(np.sum((result == data_min) | (result == data_max)))
    if n_clamped > 0:
        n_total = len(result)
        logger.debug(
            "Back-transform clamped %d/%d values (%.1f%%) to data range [%.4f, %.4f]",
            n_clamped, n_total, 100.0 * n_clamped / max(n_total, 1),
            data_min, data_max,
        )

    return result


def log_transform(
    values: NDArray[np.float64],
    constant: float = 0.0,
) -> NDArray[np.float64]:
    """
    Natural log transform with optional additive constant for zero handling.

    Parameters
    ----------
    values : (N,) ndarray
    constant : float
        Added before taking log: log(values + constant).

    Returns
    -------
    log_values : (N,) ndarray
    """
    values = np.asarray(values, dtype=np.float64)
    shifted = values + constant
    if np.any(shifted <= 0):
        raise ValueError(
            "log_transform: values + constant must be > 0. "
            f"Min value after shift: {shifted.min():.6f}"
        )
    return np.log(shifted)


def indicator_transform(
    values: NDArray[np.float64],
    threshold: float,
) -> NDArray[np.float64]:
    """
    Binary indicator: 1 if value >= threshold, 0 otherwise.

    Parameters
    ----------
    values : (N,) ndarray
    threshold : float

    Returns
    -------
    indicators : (N,) ndarray of 0.0 and 1.0
    """
    values = np.asarray(values, dtype=np.float64)
    return (values >= threshold).astype(np.float64)


def top_cut(
    values: NDArray[np.float64],
    cut_value: float,
) -> Tuple[NDArray[np.float64], int]:
    """
    Apply top-cut (grade capping).

    JORC requires reporting of cap value and number of capped samples.

    Parameters
    ----------
    values : (N,) ndarray
    cut_value : float
        Values above this are capped.

    Returns
    -------
    capped : (N,) ndarray
    n_capped : int
        Number of values that were capped.
    """
    values = np.asarray(values, dtype=np.float64)
    mask = values > cut_value
    n_capped = int(mask.sum())

    if n_capped > 0:
        logger.info(
            "Top-cut at %.4f: %d values capped (%.1f%%)",
            cut_value,
            n_capped,
            100.0 * n_capped / len(values),
        )

    capped = np.minimum(values, cut_value)
    return capped, n_capped
