"""
Cell declustering (Deutsch & Journel, 1998).

Assigns weights to clustered samples to reduce bias in the
global mean estimate.  Tests multiple cell sizes and random
origin offsets to find the optimal cell size.

Reference: CLAUDE_CODE_PROMPT_FastRBF_Engine.md
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class DeclusteringResult:
    """Result of cell declustering analysis."""

    optimal_weights: NDArray[np.float64]
    optimal_cell_size: float
    mean_at_each_cell_size: NDArray[np.float64]
    cell_sizes_tested: NDArray[np.float64]
    optimal_mean: float


def cell_declustering(
    points: NDArray[np.float64],
    values: NDArray[np.float64],
    cell_sizes: NDArray[np.float64],
    origin_offsets: int = 5,
    seed: int = 42,
) -> DeclusteringResult:
    """
    Deutsch & Journel cell declustering.

    For each cell size:
      1. Assign each sample to a cell
      2. Weight = 1 / (number of samples in cell)
      3. Normalise so weights sum to N
      4. Compute weighted mean

    The cell size producing the minimum weighted mean (for positively
    skewed data) or maximum (for negatively skewed) is optimal.

    Parameters
    ----------
    points : (N, 3) ndarray
        Sample coordinates.
    values : (N,) ndarray
        Sample values.
    cell_sizes : (K,) ndarray
        Array of cell sizes to test.
    origin_offsets : int
        Number of random origin offsets per cell size.
    seed : int
        Random seed for origin offsets.

    Returns
    -------
    DeclusteringResult
    """
    points = np.asarray(points, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    cell_sizes = np.asarray(cell_sizes, dtype=np.float64)

    n = len(values)
    rng = np.random.Generator(np.random.PCG64(seed))

    from scipy.stats import skew

    data_skew = skew(values)

    mean_per_size = np.empty(len(cell_sizes), dtype=np.float64)
    all_weights = {}  # cell_size_idx → best offset weights

    for s_idx, cs in enumerate(cell_sizes):
        if cs <= 0:
            raise ValueError(f"Cell size must be > 0; got {cs}")

        offset_means = np.empty(origin_offsets, dtype=np.float64)
        offset_weights_list = []

        for o in range(origin_offsets):
            # Random origin offset within one cell size
            origin = rng.uniform(0, cs, size=3)

            # Assign samples to cells
            shifted = points - origin[np.newaxis, :]
            cell_ids = np.floor(shifted / cs).astype(np.int64)

            # Unique cell encoding via structured array (overflow-safe).
            # The previous integer hash (a*1e9 + b*1e3 + c) can overflow
            # int64 for large cell ID values from UTM coordinates.
            cell_struct = np.ascontiguousarray(cell_ids).view(
                dtype=[('x', np.int64), ('y', np.int64), ('z', np.int64)]
            ).ravel()

            # Count per cell
            unique_cells, inverse, counts = np.unique(
                cell_struct, return_inverse=True, return_counts=True
            )
            weights_raw = 1.0 / counts[inverse]

            # Normalise so weights sum to N
            weights = weights_raw * (n / weights_raw.sum())

            weighted_mean = float(np.sum(weights * values) / n)
            offset_means[o] = weighted_mean
            offset_weights_list.append(weights)

        # Average mean across offsets for this cell size
        avg_mean = float(offset_means.mean())
        mean_per_size[s_idx] = avg_mean

        # Store best offset's weights for this cell size
        best_offset_idx = int(np.argmin(offset_means) if data_skew > 0 else np.argmax(offset_means))
        all_weights[s_idx] = offset_weights_list[best_offset_idx]

    # Select optimal cell size by plateau detection.
    #
    # The standard Deutsch & Journel approach picks min (positive skew) or
    # max (negative skew) mean.  This fails when clusters are in low-grade
    # areas (positively skewed data but clustered mean is already low).
    # Plateau detection finds the cell size where the weighted mean
    # stabilises — the first-derivative flattens, indicating the
    # declustering has converged.
    if len(cell_sizes) >= 3:
        # Compute absolute change between consecutive means
        diffs = np.abs(np.diff(mean_per_size))
        # Normalise diffs to [0, 1]
        max_diff = diffs.max() if diffs.max() > 0 else 1.0
        norm_diffs = diffs / max_diff

        # Find first cell size where the change is < 10% of max change
        # (plateau region).  Look from the middle outward to avoid
        # edge artefacts from very small cell sizes.
        plateau_threshold = 0.1
        plateau_idx = None
        for i in range(len(norm_diffs)):
            if norm_diffs[i] < plateau_threshold:
                plateau_idx = i + 1  # +1 because diff[i] spans size[i] to size[i+1]
                break

        if plateau_idx is not None:
            best_idx = plateau_idx
        else:
            # No plateau found — fall back to min/max by skewness
            best_idx = int(np.argmin(mean_per_size) if data_skew > 0 else np.argmax(mean_per_size))
    else:
        best_idx = int(np.argmin(mean_per_size) if data_skew > 0 else np.argmax(mean_per_size))

    best_cell_size = float(cell_sizes[best_idx])
    best_mean = float(mean_per_size[best_idx])
    best_weights = all_weights.get(best_idx, np.ones(n, dtype=np.float64))

    logger.info(
        "Declustering: optimal cell=%.2f, mean=%.4f (skew=%.2f, method=%s)",
        best_cell_size,
        best_mean,
        data_skew,
        "plateau" if len(cell_sizes) >= 3 else "extremum",
    )

    return DeclusteringResult(
        optimal_weights=best_weights,
        optimal_cell_size=float(best_cell_size),
        mean_at_each_cell_size=mean_per_size,
        cell_sizes_tested=cell_sizes,
        optimal_mean=best_mean,
    )
