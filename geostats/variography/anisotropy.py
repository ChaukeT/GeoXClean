"""
Variogram ellipsoid rotation matrices.

Provides rotation matrix construction and anisotropy application
for variogram analysis.

Reference: CLAUDE_CODE_PROMPT_FastRBF_Engine.md
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def rotation_matrix_3d(
    azimuth: float, dip: float, pitch: float
) -> NDArray[np.float64]:
    """
    Build a 3×3 rotation matrix (Leapfrog convention).

    Rotation order:
      1. Azimuth around Z (clockwise from north)
      2. Dip around rotated X-axis
      3. Pitch around rotated Y-axis

    Parameters
    ----------
    azimuth, dip, pitch : float
        Angles in degrees.

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

    Rz = np.array([[cos_az, sin_az, 0], [-sin_az, cos_az, 0], [0, 0, 1.0]])
    Rx = np.array([[1.0, 0, 0], [0, cos_dp, -sin_dp], [0, sin_dp, cos_dp]])
    Ry = np.array([[cos_pt, 0, sin_pt], [0, 1.0, 0], [-sin_pt, 0, cos_pt]])

    return Ry @ Rx @ Rz


def apply_anisotropy(
    points: NDArray[np.float64],
    azimuth: float = 0.0,
    dip: float = 0.0,
    pitch: float = 0.0,
    ratio_major: float = 1.0,
    ratio_semi: float = 1.0,
    ratio_minor: float = 1.0,
) -> NDArray[np.float64]:
    """
    Transform points into anisotropic space.

    Rotate by variogram ellipsoid angles then scale by inverse
    axis ratios.

    Parameters
    ----------
    points : (N, 3) ndarray
    azimuth, dip, pitch : float
        Rotation angles (degrees).
    ratio_major, ratio_semi, ratio_minor : float
        Ellipsoid axis ratios (>0).

    Returns
    -------
    transformed : (N, 3) ndarray
    """
    points = np.asarray(points, dtype=np.float64)
    R = rotation_matrix_3d(azimuth, dip, pitch)
    scale = np.array([1.0 / ratio_major, 1.0 / ratio_semi, 1.0 / ratio_minor])
    return (points @ R.T) * scale
