"""
Anisotropic distance computation with 3-D rotation matrices.

Rotation order follows the Leapfrog Geo convention:
  1. Azimuth around Z (clockwise from north)
  2. Dip around rotated X-axis
  3. Pitch (plunge/rake) around rotated Y-axis

All operations are fully vectorised with NumPy — no Python loops
over point pairs.

Reference: CLAUDE_CODE_PROMPT_FastRBF_Engine.md
"""

from __future__ import annotations

import logging
from typing import Optional

import functools

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=32)
def _rotation_matrix(
    azimuth: float, dip: float, pitch: float
) -> NDArray[np.float64]:
    """
    Build the combined 3×3 rotation matrix (Leapfrog convention).

    Parameters
    ----------
    azimuth : float
        Clockwise rotation from north around Z-axis (degrees).
    dip : float
        Rotation around the rotated X-axis (degrees).
    pitch : float
        Rotation around the rotated Y-axis (degrees).

    Returns
    -------
    R : (3, 3) ndarray
    """
    az = np.radians(azimuth)
    dp = np.radians(dip)
    pt = np.radians(pitch)

    cos_az, sin_az = np.cos(az), np.sin(az)
    cos_dp, sin_dp = np.cos(dp), np.sin(dp)
    cos_pt, sin_pt = np.cos(pt), np.sin(pt)

    # Rz — azimuth around Z (clockwise → negate angle for standard maths)
    Rz = np.array(
        [
            [cos_az, sin_az, 0.0],
            [-sin_az, cos_az, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    # Rx — dip around X
    Rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cos_dp, -sin_dp],
            [0.0, sin_dp, cos_dp],
        ]
    )

    # Ry — pitch around Y
    Ry = np.array(
        [
            [cos_pt, 0.0, sin_pt],
            [0.0, 1.0, 0.0],
            [-sin_pt, 0.0, cos_pt],
        ]
    )

    return Ry @ Rx @ Rz


def anisotropic_distance(
    p1: NDArray[np.float64],
    p2: NDArray[np.float64],
    azimuth: float = 0.0,
    dip: float = 0.0,
    pitch: float = 0.0,
    ratio_major: float = 1.0,
    ratio_semi: float = 1.0,
    ratio_minor: float = 1.0,
) -> NDArray[np.float64]:
    """
    Compute anisotropic Euclidean distance between two point sets.

    The coordinate differences are rotated by the variogram ellipsoid
    angles then scaled by the ellipsoid axis ratios before computing
    the Euclidean norm.

    Parameters
    ----------
    p1 : (N, 3) or (3,) ndarray
    p2 : (M, 3) or (3,) ndarray
        Point arrays.  Broadcastable shapes are accepted.
    azimuth, dip, pitch : float
        Rotation angles in degrees (Leapfrog convention).
    ratio_major, ratio_semi, ratio_minor : float
        Ellipsoid axis ratios (>0).

    Returns
    -------
    distances : ndarray
        Element-wise anisotropic distances.  Shape is broadcast of
        ``p1`` and ``p2`` leading dimensions.
    """
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)

    delta = p1 - p2  # (..., 3)

    T = _transform_matrix(azimuth, dip, pitch, ratio_major, ratio_semi, ratio_minor)
    scaled = delta @ T.T  # (..., 3)

    return np.sqrt(np.sum(scaled ** 2, axis=-1))


def isotropic_distance(
    p1: NDArray[np.float64],
    p2: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Standard Euclidean distance (vectorised).

    Parameters
    ----------
    p1, p2 : ndarray
        Point arrays with last dimension = 3. Broadcastable.

    Returns
    -------
    distances : ndarray
    """
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    return np.sqrt(np.sum((p1 - p2) ** 2, axis=-1))


@functools.lru_cache(maxsize=32)
def _transform_matrix(
    azimuth: float, dip: float, pitch: float,
    ratio_major: float, ratio_semi: float, ratio_minor: float,
) -> NDArray[np.float64]:
    """Pre-multiply rotation by inverse-ratio scaling (cached)."""
    R = _rotation_matrix(azimuth, dip, pitch)
    scale = np.array([1.0 / ratio_major, 1.0 / ratio_semi, 1.0 / ratio_minor])
    return R * scale[np.newaxis, :]  # (3,3) — rows scaled


def pairwise_anisotropic_distance(
    points: NDArray[np.float64],
    azimuth: float = 0.0,
    dip: float = 0.0,
    pitch: float = 0.0,
    ratio_major: float = 1.0,
    ratio_semi: float = 1.0,
    ratio_minor: float = 1.0,
) -> NDArray[np.float64]:
    """
    Compute the full N×N pairwise anisotropic distance matrix.

    Parameters
    ----------
    points : (N, 3) ndarray
    azimuth, dip, pitch : float
    ratio_major, ratio_semi, ratio_minor : float

    Returns
    -------
    D : (N, N) ndarray
        Symmetric distance matrix with zero diagonal.
    """
    points = np.asarray(points, dtype=np.float64)
    T = _transform_matrix(azimuth, dip, pitch, ratio_major, ratio_semi, ratio_minor)

    scaled = points @ T.T

    # Pairwise using broadcasting: (N,1,3) - (1,N,3) → (N,N,3)
    diff = scaled[:, np.newaxis, :] - scaled[np.newaxis, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=-1))


def batch_anisotropic_distance(
    queries: NDArray[np.float64],
    data_points: NDArray[np.float64],
    azimuth: float = 0.0,
    dip: float = 0.0,
    pitch: float = 0.0,
    ratio_major: float = 1.0,
    ratio_semi: float = 1.0,
    ratio_minor: float = 1.0,
) -> NDArray[np.float64]:
    """
    Anisotropic distances from B query points to N data points → (B, N).

    Uses the SAME transform as pairwise_anisotropic_distance to
    guarantee identical distance metric in fit() and predict().

    Parameters
    ----------
    queries : (B, 3) ndarray
    data_points : (N, 3) ndarray

    Returns
    -------
    D : (B, N) ndarray
    """
    queries = np.asarray(queries, dtype=np.float64)
    data_points = np.asarray(data_points, dtype=np.float64)

    T = _transform_matrix(azimuth, dip, pitch, ratio_major, ratio_semi, ratio_minor)

    q_transformed = queries @ T.T       # (B, 3)
    d_transformed = data_points @ T.T   # (N, 3)

    # (B,1,3) - (1,N,3) → (B,N,3)
    diff = q_transformed[:, np.newaxis, :] - d_transformed[np.newaxis, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=-1))


def point_to_points_anisotropic(
    query: NDArray[np.float64],
    points: NDArray[np.float64],
    azimuth: float = 0.0,
    dip: float = 0.0,
    pitch: float = 0.0,
    ratio_major: float = 1.0,
    ratio_semi: float = 1.0,
    ratio_minor: float = 1.0,
) -> NDArray[np.float64]:
    """
    Distances from one query point to N data points.

    Parameters
    ----------
    query : (3,) ndarray
    points : (N, 3) ndarray

    Returns
    -------
    distances : (N,) ndarray
    """
    query = np.asarray(query, dtype=np.float64).ravel()
    points = np.asarray(points, dtype=np.float64)

    delta = points - query[np.newaxis, :]

    T = _transform_matrix(azimuth, dip, pitch, ratio_major, ratio_semi, ratio_minor)
    scaled = delta @ T.T

    return np.sqrt(np.sum(scaled ** 2, axis=-1))
