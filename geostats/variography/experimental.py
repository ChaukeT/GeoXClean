"""
Experimental variogram computation.

Implements the standard Matheron estimator:
    γ(h) = 1/(2N(h)) · Σ [z(xᵢ) - z(xᵢ + h)]²

Supports omnidirectional and directional variograms.
Fully vectorised with NumPy — no Python loops over pairs.

Reference: CLAUDE_CODE_PROMPT_FastRBF_Engine.md
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)


@dataclass
class ExperimentalVariogram:
    """Result of experimental variogram computation."""

    lags: NDArray[np.float64]
    semivariance: NDArray[np.float64]
    pair_counts: NDArray[np.int64]
    lag_tolerance: float
    direction: Optional[Tuple[float, float]]  # (azimuth, dip) or None
    azimuth_tol: float
    dip_tol: float
    is_directional: bool


def compute_experimental_variogram(
    points: NDArray[np.float64],
    values: NDArray[np.float64],
    n_lags: int = 15,
    lag_tolerance: Optional[float] = None,
    azimuth_tol: float = 22.5,
    dip_tol: float = 22.5,
    direction: Optional[Tuple[float, float]] = None,
    max_lag: Optional[float] = None,
) -> ExperimentalVariogram:
    """
    Compute omnidirectional or directional experimental variogram.

    Uses the Matheron estimator:
        γ(h) = 1/(2N(h)) · Σ [z(xᵢ) - z(xᵢ + h)]²

    For large datasets (N > 10000), a spatial index limits pair
    computation to feasible ranges.

    Parameters
    ----------
    points : (N, 3) ndarray
        Sample coordinates.
    values : (N,) ndarray
        Sample values.
    n_lags : int
        Number of lag bins.
    lag_tolerance : float, optional
        Half-width of each lag bin.  Auto-computed if None.
    azimuth_tol : float
        Angular tolerance in degrees for directional variograms.
    dip_tol : float
        Dip tolerance in degrees for directional variograms.
    direction : (azimuth, dip), optional
        Direction for directional variogram.  None = omnidirectional.
    max_lag : float, optional
        Maximum lag distance.  Default = half the data extent.

    Returns
    -------
    ExperimentalVariogram
    """
    points = np.asarray(points, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    n = len(values)

    # Auto-determine max lag and tolerance
    extent = np.max(np.max(points, axis=0) - np.min(points, axis=0))
    if max_lag is None:
        max_lag = extent / 2.0

    lag_spacing = max_lag / n_lags
    if lag_tolerance is None:
        lag_tolerance = lag_spacing / 2.0

    lag_centers = np.arange(1, n_lags + 1) * lag_spacing
    semivariance = np.zeros(n_lags, dtype=np.float64)
    pair_counts = np.zeros(n_lags, dtype=np.int64)

    is_directional = direction is not None

    # Direction vector (unit)
    if is_directional:
        az_rad = np.radians(direction[0])
        dip_rad = np.radians(direction[1])
        dir_vec = np.array(
            [
                np.sin(az_rad) * np.cos(dip_rad),
                np.cos(az_rad) * np.cos(dip_rad),
                np.sin(dip_rad),
            ]
        )
        cos_az_tol = np.cos(np.radians(azimuth_tol))
        cos_dip_tol = np.cos(np.radians(dip_tol))
    else:
        dir_vec = None

    # Use KDTree for efficient pair finding
    if n > 10000:
        tree = cKDTree(points)
        pairs = tree.query_pairs(r=max_lag + lag_tolerance, output_type="ndarray")
    else:
        # For small N, compute all pairs
        idx_i, idx_j = np.triu_indices(n, k=1)
        pairs = np.column_stack([idx_i, idx_j])

    if len(pairs) == 0:
        logger.warning("No point pairs found within max lag %.2f", max_lag)
        return ExperimentalVariogram(
            lags=lag_centers,
            semivariance=semivariance,
            pair_counts=pair_counts,
            lag_tolerance=lag_tolerance,
            direction=direction,
            azimuth_tol=azimuth_tol,
            dip_tol=dip_tol,
            is_directional=is_directional,
        )

    # Compute differences vectorised
    diff_xyz = points[pairs[:, 1]] - points[pairs[:, 0]]
    dists = np.linalg.norm(diff_xyz, axis=1)
    diff_vals = values[pairs[:, 1]] - values[pairs[:, 0]]
    sq_diffs = diff_vals ** 2

    # Directional filter
    if is_directional and dir_vec is not None:
        unit_diff = diff_xyz / np.maximum(dists[:, np.newaxis], 1e-15)

        # ── V-002 FIX: Apply BOTH azimuth AND dip tolerances ──
        # Previously only cos_az_tol was used, ignoring dip_tol entirely.
        # This caused directional variograms to include pairs from all dip
        # angles, biasing semivariance in steeply-dipping deposits.
        #
        # Standard approach: decompose the 3D lag vector into horizontal
        # azimuth angle and vertical dip angle, then filter each separately.

        # Horizontal projection of lag vectors and direction
        horiz_len = np.sqrt(unit_diff[:, 0] ** 2 + unit_diff[:, 1] ** 2)
        dir_horiz_len = np.sqrt(dir_vec[0] ** 2 + dir_vec[1] ** 2)

        if dir_horiz_len > 1e-12:
            # Azimuth angle: angle between horizontal projections
            # Normalise horizontal components
            horiz_unit_x = unit_diff[:, 0] / np.maximum(horiz_len, 1e-15)
            horiz_unit_y = unit_diff[:, 1] / np.maximum(horiz_len, 1e-15)
            dir_horiz_x = dir_vec[0] / dir_horiz_len
            dir_horiz_y = dir_vec[1] / dir_horiz_len
            cos_az = np.abs(horiz_unit_x * dir_horiz_x + horiz_unit_y * dir_horiz_y)
            cos_az = np.clip(cos_az, 0.0, 1.0)
            az_mask = cos_az >= cos_az_tol
        else:
            # Direction is purely vertical — all azimuths pass
            az_mask = np.ones(len(unit_diff), dtype=bool)

        # Dip angle: vertical angle of lag vector vs direction dip
        # Dip of lag = arcsin(|dz|), Dip of direction = arcsin(dir_vec[2])
        lag_dip = np.abs(unit_diff[:, 2])          # sin(dip) of lag vector
        dir_dip = np.abs(dir_vec[2])                # sin(dip) of direction
        # Angular difference in dip
        cos_dip_diff = np.abs(lag_dip * dir_dip +
                              np.sqrt(np.maximum(1 - lag_dip**2, 0)) *
                              np.sqrt(np.maximum(1 - dir_dip**2, 0)))
        cos_dip_diff = np.clip(cos_dip_diff, 0.0, 1.0)
        dip_mask = cos_dip_diff >= cos_dip_tol

        angle_mask = az_mask & dip_mask
        n_total_before = len(dists)
        dists = dists[angle_mask]
        sq_diffs = sq_diffs[angle_mask]
        # ── ISS-011: Log directional pair counts for QA ──
        logger.info(
            "Directional filter: %d/%d pairs pass (az_tol=%.1f°, dip_tol=%.1f°)",
            len(dists), n_total_before, azimuth_tol, dip_tol,
        )

    # Bin pairs into lag bins
    for k in range(n_lags):
        lo = lag_centers[k] - lag_tolerance
        hi = lag_centers[k] + lag_tolerance
        mask = (dists >= lo) & (dists < hi)
        count = mask.sum()
        if count > 0:
            pair_counts[k] = count
            semivariance[k] = np.sum(sq_diffs[mask]) / (2.0 * count)

    logger.info(
        "Experimental variogram: %d lags, %d total pairs, dir=%s",
        n_lags,
        len(pairs),
        direction,
    )

    return ExperimentalVariogram(
        lags=lag_centers,
        semivariance=semivariance,
        pair_counts=pair_counts,
        lag_tolerance=lag_tolerance,
        direction=direction,
        azimuth_tol=azimuth_tol,
        dip_tol=dip_tol,
        is_directional=is_directional,
    )
