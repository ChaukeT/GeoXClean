"""
ARBF Orientation Field for Locally Varying Anisotropy (LVA).

Constructs a 3D grid of rotation matrices R(x) that map global
coordinates to the local geological frame (Eq. 4.1, 4.2).

Sources (in preference order):
1. Wireframe surface normals (most reliable)
2. Structural measurements (oriented core, dip/azimuth logs)
3. Data-driven inference (moment-of-inertia method, Boisvert 2009)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

try:
    from numba import njit, prange
except ImportError:
    def njit(*args, **kwargs):
        def wrapper(fn):
            return fn
        if args and callable(args[0]):
            return args[0]
        return wrapper
    prange = range

from .utils import rotation_matrix

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Numba-accelerated trilinear interpolation of rotation fields
# ---------------------------------------------------------------------------


@njit(cache=True, parallel=True)
def _batch_trilinear_interp(
    points: np.ndarray,        # (B, 3)
    grid_origin: np.ndarray,   # (3,)
    grid_spacing: np.ndarray,  # (3,)
    nx: int, ny: int, nz: int,
    rotations: np.ndarray,     # (nx, ny, nz, 3, 3)
) -> np.ndarray:
    """Trilinear interpolation of rotation matrices at B points.

    Returns (B, 3, 3) interpolated rotation matrices (before SVD).
    """
    B = points.shape[0]
    out = np.zeros((B, 3, 3), dtype=np.float64)

    for b in prange(B):
        # Continuous grid indices
        cx = (points[b, 0] - grid_origin[0]) / max(grid_spacing[0], 1e-12)
        cy = (points[b, 1] - grid_origin[1]) / max(grid_spacing[1], 1e-12)
        cz = (points[b, 2] - grid_origin[2]) / max(grid_spacing[2], 1e-12)

        # Clamp to grid bounds
        cx = min(max(cx, 0.0), nx - 1.001)
        cy = min(max(cy, 0.0), ny - 1.001)
        cz = min(max(cz, 0.0), nz - 1.001)

        # Integer corners
        i0 = int(cx)
        j0 = int(cy)
        k0 = int(cz)
        i1 = min(i0 + 1, nx - 1)
        j1 = min(j0 + 1, ny - 1)
        k1 = min(k0 + 1, nz - 1)

        # Fractional parts
        fx = cx - i0
        fy = cy - j0
        fz = cz - k0

        # 8-corner trilinear weights
        w000 = (1.0 - fx) * (1.0 - fy) * (1.0 - fz)
        w100 = fx * (1.0 - fy) * (1.0 - fz)
        w010 = (1.0 - fx) * fy * (1.0 - fz)
        w110 = fx * fy * (1.0 - fz)
        w001 = (1.0 - fx) * (1.0 - fy) * fz
        w101 = fx * (1.0 - fy) * fz
        w011 = (1.0 - fx) * fy * fz
        w111 = fx * fy * fz

        for r in range(3):
            for c in range(3):
                out[b, r, c] = (
                    w000 * rotations[i0, j0, k0, r, c]
                    + w100 * rotations[i1, j0, k0, r, c]
                    + w010 * rotations[i0, j1, k0, r, c]
                    + w110 * rotations[i1, j1, k0, r, c]
                    + w001 * rotations[i0, j0, k1, r, c]
                    + w101 * rotations[i1, j0, k1, r, c]
                    + w011 * rotations[i0, j1, k1, r, c]
                    + w111 * rotations[i1, j1, k1, r, c]
                )

    return out


@njit(cache=True, parallel=True)
def _batch_compute_T(
    R_interp: np.ndarray,  # (B, 3, 3) rotation matrices
    S: np.ndarray,          # (3, 3) scale matrix
) -> np.ndarray:
    """Compute T = S @ R for each of B rotation matrices.

    Returns (B, 3, 3) combined transform matrices.
    """
    B = R_interp.shape[0]
    out = np.zeros((B, 3, 3), dtype=np.float64)
    for b in prange(B):
        for i in range(3):
            for j in range(3):
                val = 0.0
                for k in range(3):
                    val += S[i, k] * R_interp[b, k, j]
                out[b, i, j] = val
    return out


def _batch_rotation_matrices(
    azimuth: np.ndarray,
    dip: np.ndarray,
    pitch: np.ndarray,
) -> np.ndarray:
    """Vectorised batch rotation matrix builder (replaces Python loop).

    Equivalent to calling ``rotation_matrix(az, dp, pt)`` for each
    triple but uses numpy broadcasting — zero Python iterations.

    Parameters
    ----------
    azimuth, dip, pitch : np.ndarray
        (N,) angles in degrees (GeoX convention).

    Returns
    -------
    np.ndarray
        (N, 3, 3) rotation matrices.
    """
    az = np.deg2rad(-np.asarray(azimuth, dtype=np.float64))
    dp = np.deg2rad(np.asarray(dip,     dtype=np.float64))
    pt = np.deg2rad(np.asarray(pitch,   dtype=np.float64))

    cz, sz = np.cos(az), np.sin(az)
    cx, sx = np.cos(dp), np.sin(dp)
    cy, sy = np.cos(pt), np.sin(pt)

    n = az.shape[0]
    zeros = np.zeros(n, dtype=np.float64)
    ones  = np.ones(n,  dtype=np.float64)

    # Rz[i] = [[ cz, -sz, 0], [ sz,  cz, 0], [0, 0, 1]]
    Rz = np.stack([
        np.stack([ cz, -sz, zeros], axis=1),
        np.stack([ sz,  cz, zeros], axis=1),
        np.stack([zeros, zeros, ones],  axis=1),
    ], axis=1)  # (n, 3, 3)

    # Rx[i] = [[1, 0,   0 ], [0, cx, -sx], [0, sx,  cx]]
    Rx = np.stack([
        np.stack([ones,  zeros, zeros], axis=1),
        np.stack([zeros,  cx,   -sx  ], axis=1),
        np.stack([zeros,  sx,    cx  ], axis=1),
    ], axis=1)  # (n, 3, 3)

    # Ry[i] = [[ cy, 0, sy], [0, 1, 0], [-sy, 0,  cy]]
    Ry = np.stack([
        np.stack([ cy,  zeros,  sy], axis=1),
        np.stack([zeros,  ones, zeros], axis=1),
        np.stack([-sy,  zeros,  cy], axis=1),
    ], axis=1)  # (n, 3, 3)

    # R = Ry @ Rx @ Rz  (batched matmul — no Python loop)
    return np.matmul(Ry, np.matmul(Rx, Rz))  # (n, 3, 3)


@dataclass
class OrientationField:
    """3D grid of rotation matrices for locally varying anisotropy.

    Attributes
    ----------
    grid_origin : np.ndarray
        (3,) origin of the orientation grid.
    grid_spacing : np.ndarray
        (3,) spacing in each dimension.
    grid_dims : tuple
        (nx, ny, nz) grid dimensions.
    rotations : np.ndarray
        (nx, ny, nz, 3, 3) rotation matrices.
    """

    grid_origin: np.ndarray
    grid_spacing: np.ndarray
    grid_dims: Tuple[int, int, int]
    rotations: np.ndarray

    def interpolate(self, point: np.ndarray) -> np.ndarray:
        """Trilinear interpolation of rotation matrix at arbitrary point.

        Parameters
        ----------
        point : np.ndarray
            (3,) coordinate.

        Returns
        -------
        np.ndarray
            (3, 3) interpolated rotation matrix.
        """
        pts = np.ascontiguousarray(point.reshape(1, 3), dtype=np.float64)
        R_batch = self.batch_interpolate(pts)
        return R_batch[0]

    def batch_interpolate(self, points: np.ndarray) -> np.ndarray:
        """Vectorized trilinear interpolation at B points (no Python loops).

        Parameters
        ----------
        points : np.ndarray
            (B, 3) coordinates.

        Returns
        -------
        np.ndarray
            (B, 3, 3) interpolated rotation matrices (re-orthogonalised).
        """
        points = np.ascontiguousarray(points, dtype=np.float64)
        nx, ny, nz = self.grid_dims

        # Numba-accelerated trilinear interpolation
        R_raw = _batch_trilinear_interp(
            points, self.grid_origin, self.grid_spacing,
            nx, ny, nz, self.rotations,
        )

        # Batch SVD re-orthogonalisation: np.linalg.svd handles (..., 3, 3)
        U, _, Vt = np.linalg.svd(R_raw)
        # U @ Vt for each matrix in the batch
        return np.einsum('bij,bjk->bik', U, Vt)

    def batch_interpolate_T(
        self, points: np.ndarray, S: np.ndarray,
    ) -> np.ndarray:
        """Interpolate rotations and compute T = S @ R for each point.

        Combines batch_interpolate + matrix multiply in one call,
        avoiding intermediate Python allocations.

        Parameters
        ----------
        points : np.ndarray
            (B, 3) coordinates.
        S : np.ndarray
            (3, 3) diagonal scaling matrix.

        Returns
        -------
        np.ndarray
            (B, 3, 3) combined transform matrices T = S @ R.
        """
        R_batch = self.batch_interpolate(points)
        S = np.ascontiguousarray(S, dtype=np.float64)
        return _batch_compute_T(R_batch, S)

    @classmethod
    def from_structural_data(
        cls,
        locations: np.ndarray,
        dip: np.ndarray,
        azimuth: np.ndarray,
        plunge: np.ndarray,
        grid_origin: np.ndarray,
        grid_spacing: np.ndarray,
        grid_dims: Tuple[int, int, int],
        k_neighbours: int = 8,
    ) -> "OrientationField":
        """Construct from drillhole structural measurements.

        Uses inverse-distance-weighted interpolation with circular
        angle handling (wrapping at 360 degrees).

        Parameters
        ----------
        locations : np.ndarray
            (M, 3) measurement locations.
        dip : np.ndarray
            (M,) dip angles in degrees.
        azimuth : np.ndarray
            (M,) azimuth angles in degrees.
        plunge : np.ndarray
            (M,) plunge angles in degrees.
        grid_origin : np.ndarray
            (3,) grid origin.
        grid_spacing : np.ndarray
            (3,) grid spacing.
        grid_dims : tuple
            (nx, ny, nz) grid dimensions.
        k_neighbours : int
            Number of nearest neighbours for interpolation.

        Returns
        -------
        OrientationField
        """
        nx, ny, nz = grid_dims
        grid_origin = np.asarray(grid_origin, dtype=np.float64)
        grid_spacing = np.asarray(grid_spacing, dtype=np.float64)

        # Build all grid points at once: (nx*ny*nz, 3)
        ii, jj, kk = np.meshgrid(
            np.arange(nx), np.arange(ny), np.arange(nz), indexing='ij',
        )
        grid_pts = (
            grid_origin
            + np.column_stack([ii.ravel(), jj.ravel(), kk.ravel()]) * grid_spacing
        )
        n_pts = grid_pts.shape[0]

        tree = cKDTree(locations)
        K = min(k_neighbours, len(locations))
        dists_all, idx_all = tree.query(grid_pts, k=K)  # (n_pts, K)
        if K == 1:
            dists_all = dists_all.reshape(-1, 1)
            idx_all = idx_all.reshape(-1, 1)

        # Inverse-distance weights: (n_pts, K)
        w = 1.0 / np.maximum(dists_all, 1e-10)
        w /= w.sum(axis=1, keepdims=True)

        # Circular interpolation for azimuth
        az_rad = np.deg2rad(azimuth[idx_all])  # (n_pts, K)
        az_interp = np.rad2deg(np.arctan2(
            np.sum(w * np.sin(az_rad), axis=1),
            np.sum(w * np.cos(az_rad), axis=1),
        )) % 360.0  # (n_pts,)

        dip_interp = np.sum(w * dip[idx_all], axis=1)      # (n_pts,)
        plunge_interp = np.sum(w * plunge[idx_all], axis=1)  # (n_pts,)

        # Build rotation matrices — fully vectorised, zero Python loop
        rotations = _batch_rotation_matrices(az_interp, dip_interp, plunge_interp)

        return cls(
            grid_origin=grid_origin,
            grid_spacing=grid_spacing,
            grid_dims=grid_dims,
            rotations=rotations.reshape(nx, ny, nz, 3, 3),
        )

    @classmethod
    def from_grade_data(
        cls,
        composite_coords: np.ndarray,
        composite_grades: np.ndarray,
        grid_origin: np.ndarray,
        grid_spacing: np.ndarray,
        grid_dims: Tuple[int, int, int],
        k_neighbours: int = 40,
    ) -> "OrientationField":
        """Data-driven orientation inference (Boisvert 2009, Eq. 4.2).

        At each grid node:
        1. Find K nearest samples.
        2. Compute grade-weighted covariance matrix:
           M = SUM_i |z_i - z_bar| * (x_i - x_bar)(x_i - x_bar)^T
        3. Eigendecomposition: M = V Lambda V^T
        4. R = [v1 | v2 | v3] (eigenvectors as columns,
           sorted by decreasing eigenvalue).

        Parameters
        ----------
        composite_coords : np.ndarray
            (N, 3) sample coordinates.
        composite_grades : np.ndarray
            (N,) sample values.
        grid_origin : np.ndarray
            (3,) grid origin.
        grid_spacing : np.ndarray
            (3,) grid spacing.
        grid_dims : tuple
            (nx, ny, nz) grid dimensions.
        k_neighbours : int
            Number of nearest neighbours (default 40).

        Returns
        -------
        OrientationField
        """
        nx, ny, nz = grid_dims
        grid_origin = np.asarray(grid_origin, dtype=np.float64)
        grid_spacing = np.asarray(grid_spacing, dtype=np.float64)

        # Build all grid points at once: (n_pts, 3)
        ii, jj, kk = np.meshgrid(
            np.arange(nx), np.arange(ny), np.arange(nz), indexing='ij',
        )
        grid_pts = (
            grid_origin
            + np.column_stack([ii.ravel(), jj.ravel(), kk.ravel()]) * grid_spacing
        )
        n_pts = grid_pts.shape[0]

        tree = cKDTree(composite_coords)
        K = min(k_neighbours, len(composite_coords))
        _, idx_all = tree.query(grid_pts, k=K)  # (n_pts, K)
        if K == 1:
            idx_all = idx_all.reshape(-1, 1)

        # Vectorised: compute all rotation matrices with batched numpy ops —
        # zero Python loop regardless of grid size.

        # (n_pts, K, 3) and (n_pts, K)
        local_coords_all = composite_coords[idx_all]
        local_grades_all = composite_grades[idx_all]

        x_bar_all = local_coords_all.mean(axis=1, keepdims=True)   # (n_pts, 1, 3)
        z_bar_all = local_grades_all.mean(axis=1, keepdims=True)   # (n_pts, 1)

        dx_all = local_coords_all - x_bar_all                      # (n_pts, K, 3)
        grade_weights_all = np.abs(local_grades_all - z_bar_all)   # (n_pts, K)

        # M[p] = sum_k w_k * outer(dx_k, dx_k)
        M_all = np.einsum('pk,pki,pkj->pij', grade_weights_all, dx_all, dx_all)

        # Batched eigendecomposition — numpy handles (..., 3, 3) natively
        _, eigenvectors_all = np.linalg.eigh(M_all)   # evecs: (n_pts, 3, 3)

        # eigh returns eigenvalues in ascending order; we want descending, so
        # simply reverse the column order (axis=2) for each grid point.
        V_all = eigenvectors_all[:, :, ::-1].copy()   # (n_pts, 3, 3)

        # Ensure right-hand orientation: flip last column where det < 0
        dets = np.linalg.det(V_all)                   # (n_pts,)
        V_all[dets < 0, :, 2] *= -1

        rotations = V_all

        return cls(
            grid_origin=grid_origin,
            grid_spacing=grid_spacing,
            grid_dims=grid_dims,
            rotations=rotations.reshape(nx, ny, nz, 3, 3),
        )

    @classmethod
    def identity(
        cls,
        grid_origin: np.ndarray,
        grid_spacing: np.ndarray,
        grid_dims: Tuple[int, int, int],
    ) -> "OrientationField":
        """Create an identity orientation field (no LVA).

        Every grid node has R = I (identity matrix).
        """
        nx, ny, nz = grid_dims
        rotations = np.broadcast_to(
            np.eye(3, dtype=np.float64),
            (nx, ny, nz, 3, 3),
        ).copy()  # copy so it's writable
        return cls(
            grid_origin=np.asarray(grid_origin, dtype=np.float64),
            grid_spacing=np.asarray(grid_spacing, dtype=np.float64),
            grid_dims=grid_dims,
            rotations=rotations,
        )
