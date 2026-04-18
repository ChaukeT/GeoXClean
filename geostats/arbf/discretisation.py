"""
ARBF Adaptive Block Discretisation.

Generates interior points within blocks for numerical integration.
Blocks in high-gradient areas get more discretisation points (64)
than those in smooth areas (8).  Eq. 6.3.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def build_discretisation_offsets(n: int) -> np.ndarray:
    """Build normalised discretisation offsets for an n^3 grid.

    Offsets are in [-0.5, 0.5] relative to block centre/size.

    Parameters
    ----------
    n : int
        Points per dimension (2, 3, or 4 for 8, 27, 64 total).

    Returns
    -------
    np.ndarray
        (n^3, 3) relative offsets.
    """
    if n < 1:
        return np.zeros((1, 3), dtype=np.float64)
    t = np.linspace(-0.5 + 0.5 / n, 0.5 - 0.5 / n, n)
    gx, gy, gz = np.meshgrid(t, t, t, indexing="ij")
    return np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])


# Pre-compute common discretisation grids
_OFFSETS_8 = build_discretisation_offsets(2)    # 2^3 = 8
_OFFSETS_27 = build_discretisation_offsets(3)   # 3^3 = 27
_OFFSETS_64 = build_discretisation_offsets(4)   # 4^3 = 64


def get_offsets(n_points: int) -> np.ndarray:
    """Get pre-computed or new discretisation offsets.

    Parameters
    ----------
    n_points : int
        Desired number of interior points (8, 27, or 64).

    Returns
    -------
    np.ndarray
        (n_points, 3) offsets.
    """
    if n_points <= 8:
        return _OFFSETS_8
    elif n_points <= 27:
        return _OFFSETS_27
    else:
        return _OFFSETS_64


def compute_block_interior_points(
    centroid: np.ndarray,
    block_size: np.ndarray,
    n_points: int = 27,
) -> np.ndarray:
    """Generate interior discretisation points for a single block.

    Parameters
    ----------
    centroid : np.ndarray
        (3,) block centre.
    block_size : np.ndarray
        (3,) block dimensions (dx, dy, dz).
    n_points : int
        Number of interior points (8, 27, or 64).

    Returns
    -------
    np.ndarray
        (n_points, 3) interior point coordinates.
    """
    offsets = get_offsets(n_points)
    return centroid + offsets * block_size


def estimate_gradient_magnitude(
    centroid: np.ndarray,
    block_size: np.ndarray,
    predict_fn,
) -> float:
    """Estimate grade gradient magnitude at a block centroid.

    Uses central finite differences along each axis.

    Parameters
    ----------
    centroid : np.ndarray
        (3,) block centre.
    block_size : np.ndarray
        (3,) block dimensions.
    predict_fn : callable
        Function accepting (N, 3) coordinates, returning (N,) values.

    Returns
    -------
    float
        Magnitude of the grade gradient.
    """
    h = block_size * 0.5
    points = np.zeros((6, 3), dtype=np.float64)
    for dim in range(3):
        points[2 * dim] = centroid.copy()
        points[2 * dim][dim] += h[dim]
        points[2 * dim + 1] = centroid.copy()
        points[2 * dim + 1][dim] -= h[dim]

    values = predict_fn(points)
    grad = np.array([
        (values[0] - values[1]) / max(2.0 * h[0], 1e-12),
        (values[2] - values[3]) / max(2.0 * h[1], 1e-12),
        (values[4] - values[5]) / max(2.0 * h[2], 1e-12),
    ])
    return float(np.linalg.norm(grad))


def adaptive_discretisation_density(
    centroids: np.ndarray,
    block_sizes: np.ndarray,
    predict_fn,
    threshold_high: float = 0.0,
    threshold_low: float = 0.0,
) -> np.ndarray:
    """Determine discretisation density for each block.

    High gradient -> 64 points, medium -> 27, low -> 8.

    Parameters
    ----------
    centroids : np.ndarray
        (B, 3) block centroids.
    block_sizes : np.ndarray
        (3,) or (B, 3) block sizes.
    predict_fn : callable
        Function accepting (N, 3), returning (N,) estimates.
    threshold_high : float
        Gradient threshold for 64-point discretisation.
        If 0, auto-computed from gradient distribution.
    threshold_low : float
        Gradient threshold for 27-point discretisation.
        If 0, auto-computed from gradient distribution.

    Returns
    -------
    np.ndarray
        (B,) integer array of discretisation densities (8, 27, or 64).
    """
    B = centroids.shape[0]
    if block_sizes.ndim == 1:
        block_sizes = np.tile(block_sizes, (B, 1))

    # Sample gradient at a subset for efficiency
    max_sample = min(B, 2000)
    sample_idx = np.linspace(0, B - 1, max_sample, dtype=int)
    grads = np.zeros(max_sample, dtype=np.float64)

    for i, idx in enumerate(sample_idx):
        grads[i] = estimate_gradient_magnitude(
            centroids[idx], block_sizes[idx], predict_fn,
        )

    # Auto-compute thresholds from gradient distribution
    if threshold_high <= 0 or threshold_low <= 0:
        p75 = np.percentile(grads, 75)
        p50 = np.percentile(grads, 50)
        if threshold_high <= 0:
            threshold_high = p75
        if threshold_low <= 0:
            threshold_low = p50

    # Assign densities
    densities = np.full(B, 8, dtype=np.int32)

    # For sampled blocks, assign based on gradient
    grad_interp = np.interp(
        np.arange(B),
        sample_idx,
        grads,
    )
    densities[grad_interp > threshold_low] = 27
    densities[grad_interp > threshold_high] = 64

    logger.info(
        "Adaptive discretisation: %d blocks @8, %d @27, %d @64",
        np.sum(densities == 8),
        np.sum(densities == 27),
        np.sum(densities == 64),
    )

    return densities
