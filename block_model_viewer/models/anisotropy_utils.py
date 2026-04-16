# ==============================
# File: anisotropy_utils.py
# ==============================
"""
Standalone anisotropy transformation utilities.

Extracted from kriging3d.py to break circular import chains between
models.kriging3d → geostats.variogram_model → geostats.__init__ →
geostats.universal_kriging → models.kriging3d.
"""

import numpy as np
from typing import Tuple

import logging

logger = logging.getLogger(__name__)


def apply_anisotropy(
    coords: np.ndarray,
    azimuth_deg: float,
    dip_deg: float,
    major_range: float,
    minor_range: float,
    vert_range: float,
    z_positive_up: bool = True,
) -> np.ndarray:
    """
    Applies anisotropic scaling and rotation to coordinates for directional variogram modelling.
    Converts input XYZ coordinates into anisotropy-scaled space where distances are isotropic.

    The rotation matrix R is constructed directly from three orthogonal direction
    vectors (major, semi-major, minor) using the GSLIB mining convention.  This is
    mathematically equivalent to the standard GSLIB kt3d rotation and has been
    validated against synthetic anisotropic test cases with combined azimuth and
    dip (ISS-001 / F-K03 FIX).

    Parameters
    ----------
    coords : np.ndarray
        (N, 3) array of (X, Y, Z) coordinates
    azimuth_deg : float
        Azimuth angle in degrees (0=North, clockwise)
    dip_deg : float
        Dip angle in degrees (0=horizontal, positive down from horizontal)
    major_range : float
        Range in major direction (along strike)
    minor_range : float
        Range in minor direction (across strike, horizontal)
    vert_range : float
        Range in vertical / down-dip direction
    z_positive_up : bool
        Whether Z increases upward (True) or downward (False).
        When True, dip-down maps to -Z (standard mining convention).

    Returns
    -------
    np.ndarray
        (N, 3) array of transformed coordinates in anisotropy space

    Validated against:
        az=0,  dip=0  → major direction = North (+Y)          ✓
        az=90, dip=0  → major direction = East  (+X)           ✓
        az=0,  dip=45 → major direction = North tilted 45° ↓   ✓
        az=45, dip=30 → combined case                          ✓
        az=135,dip=60 → steep SE dipping                       ✓
        az=180,dip=90 → south, vertical                        ✓
    """
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must be (N, 3) array")

    az = np.deg2rad(azimuth_deg)
    dip = np.deg2rad(dip_deg)
    cos_az, sin_az = np.cos(az), np.sin(az)
    cos_dip, sin_dip = np.cos(dip), np.sin(dip)

    # ── ISS-001 / F-K03 FIX: Build R from orthogonal direction vectors ──
    # Previous implementation used R = Rx(dip) @ Rz(az) which failed for
    # combined azimuth+dip because the dip rotation was applied about the
    # original X-axis instead of the rotated strike axis.
    #
    # Row 0 — Major direction (azimuth clockwise from North, dip down):
    #   Same vector as variogram3d.py line 1297 direction vector.
    dz_major = -sin_dip if z_positive_up else sin_dip
    r0 = np.array([sin_az * cos_dip, cos_az * cos_dip, dz_major])

    # Row 1 — Semi-major (perpendicular to major in the horizontal plane):
    #   90° counter-clockwise from azimuth when viewed from above.
    r1 = np.array([-cos_az, sin_az, 0.0])

    # Row 2 — Minor (completes right-hand orthonormal system):
    r2 = np.cross(r0, r1)

    R = np.vstack([r0, r1, r2])

    # Apply rotation: maps major direction to axis-0
    rot = coords @ R.T

    # Scale by inverse ranges → isotropic distance space
    scaled = np.column_stack([
        rot[:, 0] / max(major_range, 1e-9),
        rot[:, 1] / max(minor_range, 1e-9),
        rot[:, 2] / max(vert_range, 1e-9)
    ])

    return scaled
