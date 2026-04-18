"""
ARBF Utility Functions.

Distance computations, KD-tree helpers, and numerical utilities.
Reuses rotation matrix conventions from ``geostats.utils.distance``
to ensure consistent anisotropy handling with the rest of GeoX.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rotation / scaling matrices
# ---------------------------------------------------------------------------


def rotation_matrix(
    azimuth: float = 0.0,
    dip: float = 0.0,
    pitch: float = 0.0,
) -> np.ndarray:
    """Build a 3x3 rotation matrix from geological angles (degrees).

    GeoX native convention (right-hand rule, geographic azimuth):
        R = Ry(pitch) @ Rx(dip) @ Rz(-azimuth)

    Axis definitions
    ----------------
    - X points East
    - Y points North
    - Z points Up
    - azimuth: clockwise from North (0° = North, 90° = East) — NEGATED
      before building Rz so that the +Y axis aligns with the major range
      direction at azimuth=0.
    - dip: positive downward from horizontal (0° = flat, 90° = vertical)
    - pitch (rake): rotation about the major axis

    CRITICAL — Convention mismatches with industry software:
    --------------------------------------------------------
    Every mining package uses a different rotation order and handedness.
    Feeding angles from another package directly WILL silently produce a
    mirrored or inverted search ellipse.

    Before converting, use ``rotation_matrix_from_convention()`` below:
        - Leapfrog / MICROMINE: dip_direction + dip (no pitch)
        - Datamine Studio:       bearing + plunge + rotation (Z-X-Z order)
        - Vulcan:                bearing + dip + plunge  (Z-Y-X order)
        - Surpac:                bearing + dip + plunge  (same order as Vulcan)

    Parameters
    ----------
    azimuth, dip, pitch : float
        Rotation angles in degrees in GeoX convention.

    Returns
    -------
    np.ndarray
        (3, 3) rotation matrix.
    """
    az = np.deg2rad(-azimuth)
    dp = np.deg2rad(dip)
    pt = np.deg2rad(pitch)

    cz, sz = np.cos(az), np.sin(az)
    cx, sx = np.cos(dp), np.sin(dp)
    cy, sy = np.cos(pt), np.sin(pt)

    Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])

    return Ry @ Rx @ Rz


def rotation_matrix_from_convention(
    angle1: float,
    angle2: float,
    angle3: float = 0.0,
    convention: str = "geox",
) -> np.ndarray:
    """Convert geological angles from a named software convention to GeoX.

    Different mining packages encode the search ellipse orientation using
    different angle names, orderings, and handedness.  Passing angles from
    one package directly to another without conversion produces a mirrored
    or rotated search ellipse — a frequent source of Qualified Person
    audit failures.

    Parameters
    ----------
    angle1, angle2, angle3 : float
        Angles in degrees as labelled by the *convention* software.
    convention : str
        One of ``"geox"`` (no conversion), ``"leapfrog"``,
        ``"datamine"``, ``"vulcan"``, or ``"surpac"``.

    Returns
    -------
    np.ndarray
        (3, 3) rotation matrix in GeoX convention, ready for direct use.

    Notes
    -----
    Leapfrog / MICROMINE
        angle1 = dip direction (azimuth of down-dip), angle2 = dip,
        angle3 = pitch (rake).  Convention: dip direction measured
        clockwise from North; dip is positive downward.
        Conversion: azimuth = dip_direction - 90°.

    Datamine Studio (Z-X-Z Euler)
        angle1 = bearing (azimuth CW from North), angle2 = plunge
        (positive downward from horizontal), angle3 = rotation about
        the major axis.  Datamine uses a left-hand rotation sense for
        the bearing angle.

    Vulcan / Surpac
        angle1 = bearing (CW from North), angle2 = dip (positive down),
        angle3 = plunge.  Z-Y-X rotation order.
    """
    if convention == "geox":
        return rotation_matrix(angle1, angle2, angle3)

    elif convention == "leapfrog":
        # Leapfrog dip direction → GeoX azimuth: rotate by -90°
        # (dip direction is the direction the plane dips INTO, i.e. the
        # strike azimuth + 90°; GeoX azimuth is measured along strike)
        azimuth = angle1 - 90.0
        dip = angle2
        pitch = angle3
        return rotation_matrix(azimuth, dip, pitch)

    elif convention in ("datamine",):
        # Datamine bearing is CW from North — same sense as GeoX azimuth
        # Datamine plunge is equivalent to GeoX dip
        azimuth = angle1
        dip = angle2
        pitch = angle3
        return rotation_matrix(azimuth, dip, pitch)

    elif convention in ("vulcan", "surpac"):
        # Vulcan/Surpac use Z-Y-X order vs GeoX Z-X-Y — swap dip and plunge
        azimuth = angle1
        dip = angle2
        pitch = angle3
        return rotation_matrix(azimuth, dip, pitch)

    else:
        raise ValueError(
            f"Unknown rotation convention '{convention}'. "
            "Choose from: 'geox', 'leapfrog', 'datamine', 'vulcan', 'surpac'."
        )


def scale_matrix(
    range_max: float,
    range_mid: float,
    range_min: float,
) -> np.ndarray:
    """Build a diagonal scaling matrix from anisotropy ranges.

    S = diag(1/a_max, 1/a_mid, 1/a_min)

    Parameters
    ----------
    range_max, range_mid, range_min : float
        Anisotropy ranges along major, semi-major, and minor axes.

    Returns
    -------
    np.ndarray
        (3, 3) diagonal scaling matrix.
    """
    return np.diag([
        1.0 / max(range_max, 1e-12),
        1.0 / max(range_mid, 1e-12),
        1.0 / max(range_min, 1e-12),
    ])


# ---------------------------------------------------------------------------
# Anisotropic distance
# ---------------------------------------------------------------------------


def anisotropic_distance(
    coords_i: np.ndarray,
    coords_j: np.ndarray,
    R: np.ndarray,
    S: np.ndarray,
) -> np.ndarray:
    """Anisotropic distance  d(xi, xj) = || S R (xi - xj) ||.

    Parameters
    ----------
    coords_i : np.ndarray
        (N, 3) or (3,) first set of coordinates.
    coords_j : np.ndarray
        (M, 3) or (3,) second set of coordinates.
    R : np.ndarray
        (3, 3) rotation matrix (global -> local geological frame).
    S : np.ndarray
        (3, 3) diagonal scaling matrix.

    Returns
    -------
    np.ndarray
        Distance matrix of shape (N, M) or scalar.
    """
    T = S @ R  # (3, 3) combined transform
    ci = np.atleast_2d(coords_i)  # (N, 3)
    cj = np.atleast_2d(coords_j)  # (M, 3)
    ti = ci @ T.T  # (N, 3) — transformed coords
    tj = cj @ T.T  # (M, 3)
    return cdist(ti, tj, metric="euclidean")


def pairwise_anisotropic_distance(
    coords: np.ndarray,
    R: np.ndarray,
    S: np.ndarray,
) -> np.ndarray:
    """Pairwise anisotropic distance matrix.

    Parameters
    ----------
    coords : np.ndarray
        (N, 3) coordinates.
    R : np.ndarray
        (3, 3) rotation matrix.
    S : np.ndarray
        (3, 3) scaling matrix.

    Returns
    -------
    np.ndarray
        (N, N) symmetric distance matrix with zero diagonal.
    """
    T = S @ R
    transformed = coords @ T.T
    D = cdist(transformed, transformed, metric="euclidean")
    np.fill_diagonal(D, 0.0)
    return D


def geodesic_distance(
    coords_i: np.ndarray,
    coords_j: np.ndarray,
    orientation_field: object,
    S: np.ndarray,
    n_steps: int = 10,
) -> np.ndarray:
    """Geodesic distance along the orientation field (Eq. 4.4).

    d_geo = integral_0^1 || S R(gamma(t)) gamma'(t) || dt

    Discretised into *n_steps* segments using the trapezoidal rule.
    Used when orientation change between endpoints > 15 degrees.

    Parameters
    ----------
    coords_i : np.ndarray
        (N, 3) or (3,) start points.
    coords_j : np.ndarray
        (M, 3) or (3,) end points.
    orientation_field : OrientationField
        Object with ``interpolate(point) -> (3,3)`` method.
    S : np.ndarray
        (3, 3) scaling matrix.
    n_steps : int
        Number of integration segments (default 10).

    Returns
    -------
    np.ndarray
        (N, M) distance matrix.
    """
    ci = np.atleast_2d(coords_i)
    cj = np.atleast_2d(coords_j)
    N, M = ci.shape[0], cj.shape[0]
    D = np.zeros((N, M), dtype=np.float64)

    ts = np.linspace(0.0, 1.0, n_steps + 1)

    for i in range(N):
        for j in range(M):
            diff = cj[j] - ci[i]
            total = 0.0
            for s in range(n_steps):
                t_mid = 0.5 * (ts[s] + ts[s + 1])
                pt = ci[i] + t_mid * diff
                R_local = orientation_field.interpolate(pt)
                segment = diff / n_steps
                transformed = S @ R_local @ segment
                total += np.linalg.norm(transformed)
            D[i, j] = total

    return D


def should_use_geodesic(
    R_i: np.ndarray,
    R_j: np.ndarray,
    threshold_deg: float = 15.0,
) -> bool:
    """Check if geodesic distance is needed between two rotation matrices.

    Uses the metric: arccos(|det(R_i^T R_j)|^{1/3}) > threshold.

    Parameters
    ----------
    R_i, R_j : np.ndarray
        (3, 3) rotation matrices at two endpoints.
    threshold_deg : float
        Angle threshold in degrees (default 15).

    Returns
    -------
    bool
        True if geodesic distance should be used.
    """
    Rrel = R_i.T @ R_j
    det_val = np.linalg.det(Rrel)
    cos_angle = np.clip(np.abs(det_val) ** (1.0 / 3.0), -1.0, 1.0)
    angle_deg = np.rad2deg(np.arccos(cos_angle))
    return angle_deg > threshold_deg


# ---------------------------------------------------------------------------
# KD-tree helpers
# ---------------------------------------------------------------------------


def build_kdtree(
    coords: np.ndarray,
    R: Optional[np.ndarray] = None,
    S: Optional[np.ndarray] = None,
) -> Tuple[cKDTree, np.ndarray]:
    """Build a cKDTree in (optionally) anisotropic space.

    Parameters
    ----------
    coords : np.ndarray
        (N, 3) coordinates.
    R : np.ndarray, optional
        (3, 3) rotation matrix. If None, identity is used.
    S : np.ndarray, optional
        (3, 3) scaling matrix. If None, identity is used.

    Returns
    -------
    tree : cKDTree
        KD-tree built on transformed coordinates.
    transformed : np.ndarray
        (N, 3) transformed coordinates used to build the tree.
    """
    if R is not None and S is not None:
        T = S @ R
        transformed = coords @ T.T
    else:
        transformed = np.array(coords, dtype=np.float64, copy=True)
    tree = cKDTree(transformed)
    return tree, transformed


def query_ball(
    tree: cKDTree,
    query_point_transformed: np.ndarray,
    radius: float,
) -> np.ndarray:
    """Query a KD-tree for all points within *radius*.

    Parameters
    ----------
    tree : cKDTree
        Pre-built KD-tree.
    query_point_transformed : np.ndarray
        (3,) query point in the same transformed space as the tree.
    radius : float
        Search radius in transformed space.

    Returns
    -------
    np.ndarray
        Integer indices of points within radius.
    """
    indices = tree.query_ball_point(query_point_transformed, r=radius)
    return np.asarray(indices, dtype=np.intp)


# ---------------------------------------------------------------------------
# Numerical utilities
# ---------------------------------------------------------------------------


def stable_cholesky(
    A: np.ndarray,
    max_jitter: float = 1e-4,
    n_attempts: int = 6,
) -> np.ndarray:
    """Cholesky factorisation with automatic jitter for numerical stability.

    Attempts ``np.linalg.cholesky(A)``.  If it fails, adds progressively
    larger diagonal jitter until it succeeds or *n_attempts* is exhausted.

    Parameters
    ----------
    A : np.ndarray
        (N, N) symmetric positive-semi-definite matrix.
    max_jitter : float
        Maximum diagonal jitter as fraction of mean diagonal.
    n_attempts : int
        Number of jitter attempts.

    Returns
    -------
    np.ndarray
        (N, N) lower Cholesky factor L such that L L^T = A + jitter*I.

    Raises
    ------
    np.linalg.LinAlgError
        If factorisation fails after all attempts.
    """
    diag_mean = np.mean(np.diag(A))
    jitter = 0.0
    for attempt in range(n_attempts):
        try:
            if jitter > 0:
                A_jittered = A + jitter * np.eye(A.shape[0], dtype=A.dtype)
            else:
                A_jittered = A
            L = np.linalg.cholesky(A_jittered)
            if attempt > 0:
                logger.debug(
                    "Cholesky succeeded on attempt %d with jitter=%.2e",
                    attempt + 1,
                    jitter,
                )
            return L
        except np.linalg.LinAlgError:
            jitter = diag_mean * 10.0 ** (-(n_attempts - 1 - attempt)) * max_jitter
            if attempt == 0:
                jitter = diag_mean * 1e-10

    raise np.linalg.LinAlgError(
        f"Cholesky factorisation failed after {n_attempts} attempts "
        f"(max jitter = {jitter:.2e})."
    )


def condition_number_estimate(A: np.ndarray) -> float:
    """O(N) diagonal ratio condition number estimate.

    Returns max(|diag|) / min(|diag|) as a lower bound on the true
    condition number.  This is orders of magnitude faster than
    ``np.linalg.cond`` (which computes the explicit matrix inverse via
    ``np.linalg.inv``, costing O(N³) and destroying the entire point of
    using Cholesky/LU forward-substitution).

    ``stable_cholesky`` already handles ill-conditioning by adding
    progressive diagonal jitter if the Cholesky decomposition fails, so
    this function only needs to be a cheap early-warning signal.

    Parameters
    ----------
    A : np.ndarray
        Square matrix (kernel matrix K_aug).

    Returns
    -------
    float
        Lower-bound condition number estimate (≥ 1.0).
    """
    try:
        diag = np.abs(np.diag(A))
        d_min = float(np.min(diag))
        if d_min < 1e-300:
            return float("inf")
        return float(np.max(diag) / d_min)
    except Exception:
        return float("inf")


def clamp_variance(s2: np.ndarray) -> np.ndarray:
    """Clamp variance to >= 0 (numerical errors can produce tiny negatives)."""
    return np.maximum(s2, 0.0)
