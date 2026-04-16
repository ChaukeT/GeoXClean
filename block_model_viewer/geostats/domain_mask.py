"""
Domain masking utilities for estimation and simulation engines.

Provides two approaches:

1. **Pre-filter** (preferred): ``prefilter_centroids()`` computes the mask
   and returns only the inside centroids + indices.  The engine estimates
   only those blocks.  ``scatter_results()`` maps the reduced results back
   into a full-size NaN grid.  This avoids wasting computation on blocks
   that have no data influence.

2. **Post-filter** (legacy): ``apply_mask_to_results()`` sets outside
   blocks to NaN after the engine has already estimated everything.
   Kept for backward-compatibility but should be phased out.

No UI imports. No DataRegistry imports. Pure NumPy / SciPy only.
"""

from __future__ import annotations

import copy
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_distance_mask(
    block_centroids: np.ndarray,
    data_coords: np.ndarray,
    search_radii: tuple,
    azimuth_deg: float = 0.0,
    dip_deg: float = 0.0,
    min_neighbours: int = 1,
) -> np.ndarray:
    """Return boolean mask — True for blocks inside the informing volume.

    Parameters
    ----------
    block_centroids : (N_blocks, 3) X, Y, Z
    data_coords     : (N_data, 3) X, Y, Z of conditioning samples
    search_radii    : (major_m, minor_m, vert_m)
    azimuth_deg     : azimuth of major axis (0=North, 90=East)
    dip_deg         : dip of major axis
    min_neighbours  : block must have >= this many data within ellipsoid
    """
    n_blocks = block_centroids.shape[0] if block_centroids.ndim == 2 else 0
    if n_blocks == 0:
        return np.zeros(0, dtype=bool)

    # Degenerate: no data → all outside
    if data_coords is None or data_coords.shape[0] == 0:
        return np.zeros(n_blocks, dtype=bool)

    min_neighbours = max(1, min_neighbours)

    major, minor, vert = [max(abs(float(r)), 1e-6) for r in search_radii]

    # Build rotation + scale matrix to convert ellipsoid → unit sphere
    R = _rotation_matrix(azimuth_deg, dip_deg)  # 3x3
    S = np.diag([1.0 / major, 1.0 / minor, 1.0 / vert])
    T = S @ R  # 3x3 transform

    # Transform data points into anisotropic space
    data_transformed = (T @ data_coords.T).T  # (N_data, 3)

    # Use KDTree in transformed space for efficient neighbour counting
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(data_transformed)

        # Transform block centroids
        blocks_transformed = (T @ block_centroids.T).T  # (N_blocks, 3)

        # Query: how many data points within unit sphere (radius=1.0)?
        counts = tree.query_ball_point(blocks_transformed, r=1.0, return_length=True)
        mask = np.asarray(counts) >= min_neighbours
    except ImportError:
        # Fallback: brute-force vectorised (slower but no scipy dependency)
        blocks_transformed = (T @ block_centroids.T).T
        mask = np.zeros(n_blocks, dtype=bool)
        for i in range(n_blocks):
            diffs = data_transformed - blocks_transformed[i]
            dists_sq = np.sum(diffs ** 2, axis=1)
            if np.sum(dists_sq <= 1.0) >= min_neighbours:
                mask[i] = True

    logger.info(
        f"Domain mask: {int(mask.sum())}/{n_blocks} blocks inside "
        f"({100.0 * mask.sum() / n_blocks:.1f}%), "
        f"radii=({major:.1f}, {minor:.1f}, {vert:.1f}), "
        f"az={azimuth_deg:.1f}, dip={dip_deg:.1f}, min_n={min_neighbours}"
    )
    return mask


def apply_mask_to_results(
    results: dict,
    mask: np.ndarray,
    grade_keys: list[str],
    variance_keys: list[str],
    flag_key: str = "domain_mask",
) -> dict:
    """Apply domain mask to estimation/simulation results.

    Sets values where mask == False to NaN for grade and variance arrays.
    Adds a flag column (uint8: 1=inside, 0=outside).
    Returns a copy — does not modify in-place.
    """
    out = copy.copy(results)  # shallow copy of dict

    outside = ~mask

    for key in grade_keys + variance_keys:
        if key not in out:
            continue
        arr = out[key]
        if isinstance(arr, np.ndarray) and arr.shape[0] == mask.shape[0]:
            arr = arr.astype(float, copy=True)
            arr[outside] = np.nan
            out[key] = arr

    # Also mask realization arrays (key pattern: real_0, real_1, ...)
    for key in list(out.keys()):
        if key.startswith("real_"):
            arr = out[key]
            if isinstance(arr, np.ndarray) and arr.shape[0] == mask.shape[0]:
                arr = arr.astype(float, copy=True)
                arr[outside] = np.nan
                out[key] = arr

    # Add flag column
    out[flag_key] = mask.astype(np.uint8)

    return out


def prefilter_centroids(
    block_centroids: np.ndarray,
    data_coords: np.ndarray,
    search_radii: tuple,
    azimuth_deg: float = 0.0,
    dip_deg: float = 0.0,
    min_neighbours: int = 1,
    method: str = "distance",
    buffer_m: float = 0.0,
) -> tuple:
    """Pre-filter block centroids — return only those inside the mask.

    This is the **correct** way to apply domain masking: compute the mask
    BEFORE estimation so the engine only processes informed blocks.  Blocks
    outside the data support are never sent to the solver, saving time
    proportional to the fraction of the grid outside the drillhole footprint
    (typically 70-90%).

    Parameters
    ----------
    block_centroids : (N, 3)
    data_coords     : (M, 3)
    search_radii    : (major, minor, vert)
    azimuth_deg, dip_deg : search ellipsoid orientation
    min_neighbours  : minimum composites within ellipsoid
    method          : "distance" (ellipsoid) or "convex_hull"
    buffer_m        : hull buffer (only for method="convex_hull")

    Returns
    -------
    inside_centroids : (K, 3) — only the blocks inside the mask
    inside_idx       : (K,) int — indices into the original array
    mask             : (N,) bool — full mask for reference / visualization
    """
    if method == "convex_hull":
        mask = compute_convex_hull_mask(block_centroids, data_coords,
                                        buffer_m=buffer_m)
    else:
        mask = compute_distance_mask(block_centroids, data_coords,
                                     search_radii=search_radii,
                                     azimuth_deg=azimuth_deg,
                                     dip_deg=dip_deg,
                                     min_neighbours=min_neighbours)

    inside_idx = np.where(mask)[0]
    inside_centroids = block_centroids[inside_idx]

    logger.info(
        "Pre-filter: %d / %d blocks inside mask (%.1f%%) — "
        "%d blocks skipped (not sent to estimator)",
        len(inside_idx), len(mask),
        100.0 * len(inside_idx) / max(len(mask), 1),
        len(mask) - len(inside_idx),
    )
    return inside_centroids, inside_idx, mask


def scatter_results(
    inside_results: dict,
    inside_idx: np.ndarray,
    n_total: int,
    grade_keys: list[str],
    variance_keys: list[str],
    flag_key: str = "domain_mask",
    mask: Optional[np.ndarray] = None,
) -> dict:
    """Scatter reduced estimation results back into full-size arrays.

    Blocks that were not estimated (outside the mask) stay as NaN for
    grade/variance arrays and 0 for flag/integer arrays.

    Parameters
    ----------
    inside_results : dict from the engine (arrays have length K = len(inside_idx))
    inside_idx     : (K,) int indices into the full grid
    n_total        : total number of blocks in the full grid
    grade_keys     : keys for grade arrays (NaN fill)
    variance_keys  : keys for variance arrays (NaN fill)
    flag_key       : key name for the domain mask flag
    mask           : optional (N,) bool mask to store as flag
    """
    out = {}
    k = len(inside_idx)

    all_nan_keys = set(grade_keys) | set(variance_keys)
    # Also include realization arrays
    real_keys = [key for key in inside_results if key.startswith("real_")]
    all_nan_keys.update(real_keys)

    for key, val in inside_results.items():
        if not isinstance(val, np.ndarray):
            out[key] = val
            continue
        if val.ndim == 0 or val.shape[0] != k:
            out[key] = val
            continue

        # Create full-size array with appropriate fill
        if key in all_nan_keys:
            full = np.full(n_total, np.nan, dtype=float)
        else:
            full = np.zeros(n_total, dtype=val.dtype)
        full[inside_idx] = val
        out[key] = full

    # Add domain mask flag
    if mask is not None:
        out[flag_key] = mask.astype(np.uint8)
    else:
        flag = np.zeros(n_total, dtype=np.uint8)
        flag[inside_idx] = 1
        out[flag_key] = flag

    logger.info(
        "Scatter: %d inside results → %d total blocks "
        "(%d keys, %d NaN-filled)",
        k, n_total, len(out), len(all_nan_keys),
    )
    return out


def resample_irbf_mask_to_grid(
    irbf_domain: dict,
    target_origin: tuple,
    target_spacing: tuple,
    target_dims: tuple,
) -> Optional[np.ndarray]:
    """Resample an IRBF domain mask onto a target grid via nearest-neighbour.

    The IRBF panel builds its mask on its own internal auto-resolution grid
    (with 1-D cell-centre arrays ``x``, ``y``, ``z``). Downstream estimation
    engines run on their own target grid whose extent and spacing may be
    completely different. This helper classifies each target-grid block
    centre as inside/outside by nearest-neighbour lookup against the IRBF
    grid's ``inside_mask_shared`` (or thresholded ``probability_field``).

    Parameters
    ----------
    irbf_domain : dict
        Registry payload from ``get_indicator_rbf_domain()``. Must contain
        either ``inside_mask_shared`` OR (``probability_field`` + ``iso_value``),
        plus ``x``, ``y``, ``z`` cell-centre 1-D arrays.
    target_origin : (x0, y0, z0)
        Minimum corner of the first target block.
    target_spacing : (dx, dy, dz)
        Target block size.
    target_dims : (nx, ny, nz)
        Target block counts.

    Returns
    -------
    np.ndarray (shape nx*ny*nz, dtype bool) or None
        1-D bool mask in X-fastest-then-Y-then-Z order (C-ravel of
        (nz, ny, nx)), matching the convention used by
        ``create_block_model()`` / ``create_pyvista_grid()``. Returns None
        if the IRBF domain can't be interpreted.
    """
    if not isinstance(irbf_domain, dict):
        return None

    x_ax = irbf_domain.get("x")
    y_ax = irbf_domain.get("y")
    z_ax = irbf_domain.get("z")
    if x_ax is None or y_ax is None or z_ax is None:
        logger.debug("resample_irbf_mask: missing x/y/z axes in domain")
        return None

    x_ax = np.asarray(x_ax, dtype=float).ravel()
    y_ax = np.asarray(y_ax, dtype=float).ravel()
    z_ax = np.asarray(z_ax, dtype=float).ravel()

    # Prefer the pre-computed inside mask; else threshold the probability field
    inside = irbf_domain.get("inside_mask_shared")
    if inside is None:
        inside = irbf_domain.get("inside_mask")
    if inside is None:
        prob = irbf_domain.get("probability_field")
        iso = float(irbf_domain.get("iso_value", 0.5))
        if prob is None:
            return None
        inside = np.asarray(prob, dtype=float) >= iso
    inside = np.asarray(inside, dtype=bool)

    # The IRBF mask may be stored in (nz, ny, nx) OR (nx, ny, nz) order — be
    # permissive and reshape to match the axis sizes explicitly.
    nx_s, ny_s, nz_s = len(x_ax), len(y_ax), len(z_ax)
    expected = nx_s * ny_s * nz_s
    if inside.size != expected:
        logger.warning(
            "resample_irbf_mask: mask size %d != nx*ny*nz %d; cannot resample",
            inside.size, expected,
        )
        return None
    if inside.shape == (nz_s, ny_s, nx_s):
        mask_zyx = inside
    elif inside.shape == (nx_s, ny_s, nz_s):
        mask_zyx = inside.transpose(2, 1, 0)
    else:
        # Flat or unknown — assume C-order over (nz, ny, nx)
        mask_zyx = inside.reshape((nz_s, ny_s, nx_s))

    x0, y0, z0 = target_origin
    dx, dy, dz = target_spacing
    nx_t, ny_t, nz_t = target_dims

    # Target block-centre coordinates
    tx = x0 + (np.arange(nx_t) + 0.5) * dx
    ty = y0 + (np.arange(ny_t) + 0.5) * dy
    tz = z0 + (np.arange(nz_t) + 0.5) * dz

    # Nearest-neighbour index into each IRBF axis. Use searchsorted + clip.
    def _nn_idx(ax: np.ndarray, tgt: np.ndarray) -> np.ndarray:
        if ax.size == 1:
            return np.zeros(tgt.size, dtype=int)
        # searchsorted finds insertion point; pick nearest neighbour
        right = np.searchsorted(ax, tgt)
        left = np.clip(right - 1, 0, ax.size - 1)
        right = np.clip(right, 0, ax.size - 1)
        choose_right = np.abs(ax[right] - tgt) < np.abs(ax[left] - tgt)
        return np.where(choose_right, right, left)

    ix = _nn_idx(x_ax, tx)  # (nx_t,)
    iy = _nn_idx(y_ax, ty)  # (ny_t,)
    iz = _nn_idx(z_ax, tz)  # (nz_t,)

    # Out-of-IRBF-bounds → mark outside (don't extrapolate an iso surface)
    x_min, x_max = float(x_ax.min()), float(x_ax.max())
    y_min, y_max = float(y_ax.min()), float(y_ax.max())
    z_min, z_max = float(z_ax.min()), float(z_ax.max())
    # Generous tolerance: half the coarsest IRBF cell
    tol_x = 0.5 * (float(np.median(np.diff(x_ax))) if x_ax.size > 1 else dx)
    tol_y = 0.5 * (float(np.median(np.diff(y_ax))) if y_ax.size > 1 else dy)
    tol_z = 0.5 * (float(np.median(np.diff(z_ax))) if z_ax.size > 1 else dz)
    x_oob = (tx < x_min - tol_x) | (tx > x_max + tol_x)
    y_oob = (ty < y_min - tol_y) | (ty > y_max + tol_y)
    z_oob = (tz < z_min - tol_z) | (tz > z_max + tol_z)

    # Build target mask in (nz_t, ny_t, nx_t) order then C-ravel
    out = np.zeros((nz_t, ny_t, nx_t), dtype=bool)
    # Vectorised via broadcasting. The IRBF mask index triple is (iz, iy, ix).
    IZ = iz[:, None, None]
    IY = iy[None, :, None]
    IX = ix[None, None, :]
    out[:, :, :] = mask_zyx[IZ, IY, IX]

    # Zero out OOB in each axis
    if z_oob.any():
        out[z_oob, :, :] = False
    if y_oob.any():
        out[:, y_oob, :] = False
    if x_oob.any():
        out[:, :, x_oob] = False

    n_inside = int(out.sum())
    n_total = out.size
    logger.info(
        "resample_irbf_mask: %d/%d target blocks inside (%.1f%%) — "
        "source grid %dx%dx%d → target grid %dx%dx%d",
        n_inside, n_total, 100.0 * n_inside / max(n_total, 1),
        nx_s, ny_s, nz_s, nx_t, ny_t, nz_t,
    )
    return out.ravel(order="C")


def resample_irbf_mask_to_points(
    irbf_domain: dict,
    target_points: np.ndarray,
) -> Optional[np.ndarray]:
    """Classify arbitrary (N, 3) target points as inside/outside an IRBF domain.

    For target grids that aren't axis-aligned regular (e.g. PyVista
    StructuredGrid built from 3D mesh arrays, rotated block models), we
    can't use per-axis searchsorted. This helper runs KDTree nearest-
    neighbour against the IRBF cell centroids.

    Parameters
    ----------
    irbf_domain : dict
        Same payload as for ``resample_irbf_mask_to_grid``.
    target_points : np.ndarray, shape (N, 3)
        Target block centroids (or any query points).

    Returns
    -------
    np.ndarray (N,) of bool, or None if the domain can't be interpreted.
    """
    if not isinstance(irbf_domain, dict):
        return None
    target_points = np.asarray(target_points, dtype=float)
    if target_points.ndim != 2 or target_points.shape[1] != 3:
        logger.debug("resample_irbf_mask_to_points: target shape %s invalid", target_points.shape)
        return None
    if target_points.shape[0] == 0:
        return np.zeros(0, dtype=bool)

    x_ax = irbf_domain.get("x")
    y_ax = irbf_domain.get("y")
    z_ax = irbf_domain.get("z")
    if x_ax is None or y_ax is None or z_ax is None:
        return None
    x_ax = np.asarray(x_ax, dtype=float).ravel()
    y_ax = np.asarray(y_ax, dtype=float).ravel()
    z_ax = np.asarray(z_ax, dtype=float).ravel()

    inside = irbf_domain.get("inside_mask_shared")
    if inside is None:
        inside = irbf_domain.get("inside_mask")
    if inside is None:
        prob = irbf_domain.get("probability_field")
        iso = float(irbf_domain.get("iso_value", 0.5))
        if prob is None:
            return None
        inside = np.asarray(prob, dtype=float) >= iso
    inside = np.asarray(inside, dtype=bool)

    nx_s, ny_s, nz_s = len(x_ax), len(y_ax), len(z_ax)
    expected = nx_s * ny_s * nz_s
    if inside.size != expected:
        logger.warning(
            "resample_irbf_mask_to_points: mask size %d != nx*ny*nz %d",
            inside.size, expected,
        )
        return None
    if inside.shape == (nz_s, ny_s, nx_s):
        mask_flat = inside.ravel(order="C")   # (nz, ny, nx) C-order → Z fastest
        # For KDTree lookup we need a parallel (N_cells, 3) of IRBF centroids
        ZZ, YY, XX = np.meshgrid(z_ax, y_ax, x_ax, indexing="ij")
    elif inside.shape == (nx_s, ny_s, nz_s):
        mask_flat = inside.transpose(2, 1, 0).ravel(order="C")
        ZZ, YY, XX = np.meshgrid(z_ax, y_ax, x_ax, indexing="ij")
    else:
        mask_flat = inside.reshape((nz_s, ny_s, nx_s)).ravel(order="C")
        ZZ, YY, XX = np.meshgrid(z_ax, y_ax, x_ax, indexing="ij")
    centroids = np.column_stack([XX.ravel(), YY.ravel(), ZZ.ravel()])

    # KDTree nearest-neighbour lookup
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(centroids)
        _, nn_idx = tree.query(target_points, k=1)
    except Exception as exc:
        logger.debug("resample_irbf_mask_to_points: KDTree failed (%s), brute force", exc)
        # Brute-force fallback
        nn_idx = np.empty(len(target_points), dtype=int)
        for i, p in enumerate(target_points):
            d2 = np.sum((centroids - p) ** 2, axis=1)
            nn_idx[i] = int(np.argmin(d2))

    result = mask_flat[nn_idx]

    # Mark points far outside the IRBF bounding box as outside (no extrapolation)
    x_min, x_max = float(x_ax.min()), float(x_ax.max())
    y_min, y_max = float(y_ax.min()), float(y_ax.max())
    z_min, z_max = float(z_ax.min()), float(z_ax.max())
    tol_x = 0.5 * (float(np.median(np.diff(x_ax))) if x_ax.size > 1 else 0.0)
    tol_y = 0.5 * (float(np.median(np.diff(y_ax))) if y_ax.size > 1 else 0.0)
    tol_z = 0.5 * (float(np.median(np.diff(z_ax))) if z_ax.size > 1 else 0.0)
    oob = (
        (target_points[:, 0] < x_min - tol_x) | (target_points[:, 0] > x_max + tol_x) |
        (target_points[:, 1] < y_min - tol_y) | (target_points[:, 1] > y_max + tol_y) |
        (target_points[:, 2] < z_min - tol_z) | (target_points[:, 2] > z_max + tol_z)
    )
    if oob.any():
        result[oob] = False

    n_inside = int(result.sum())
    logger.info(
        "resample_irbf_mask_to_points: %d/%d target points inside (%.1f%%) — "
        "IRBF grid %dx%dx%d",
        n_inside, result.size,
        100.0 * n_inside / max(result.size, 1),
        nx_s, ny_s, nz_s,
    )
    return result.astype(bool)


def compute_convex_hull_mask(
    block_centroids: np.ndarray,
    data_coords: np.ndarray,
    buffer_m: float = 0.0,
) -> np.ndarray:
    """Return boolean mask using convex hull of conditioning data.

    Parameters
    ----------
    block_centroids : (N_blocks, 3)
    data_coords     : (N_data, 3)
    buffer_m        : expand hull outward by this many metres
    """
    n_blocks = block_centroids.shape[0] if block_centroids.ndim == 2 else 0
    if n_blocks == 0:
        return np.zeros(0, dtype=bool)

    if data_coords is None or data_coords.shape[0] == 0:
        return np.zeros(n_blocks, dtype=bool)

    try:
        from scipy.spatial import ConvexHull
        if data_coords.shape[0] < 4:
            raise ValueError("Too few points for 3D hull")

        hull = ConvexHull(data_coords)
        # Halfspace equations: A x + b <= 0
        # hull.equations: each row is [A_x, A_y, A_z, b]
        A = hull.equations[:, :3]
        b = hull.equations[:, 3]

        # Apply buffer: expand outward by buffer_m
        # Since normals point outward and constraint is Ax + b <= 0,
        # subtracting buffer from b expands the hull
        b_buffered = b - buffer_m

        # Test all blocks: Ax + b <= 0 for all halfspaces
        vals = block_centroids @ A.T + b_buffered  # (N_blocks, N_faces)
        mask = np.all(vals <= 0, axis=1)

    except Exception:
        # Fallback: bounding box
        logger.warning("ConvexHull failed, falling back to bounding box mask")
        mask = _bounding_box_mask(block_centroids, data_coords, buffer_m)

    logger.info(
        f"Convex hull mask: {int(mask.sum())}/{n_blocks} blocks inside "
        f"({100.0 * mask.sum() / n_blocks:.1f}%), buffer={buffer_m:.1f}m"
    )
    return mask


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _rotation_matrix(azimuth_deg: float, dip_deg: float) -> np.ndarray:
    """Build 3x3 rotation matrix for mining convention.

    Transforms world (X=East, Y=North, Z=Up) into ellipsoid principal axes
    where axis-0 = major, axis-1 = minor, axis-2 = vertical.

    azimuth: clockwise from North (Y-axis) in horizontal plane.
             0° = major axis along North (Y), 90° = major axis along East (X).
    dip: downward tilt of major axis from horizontal.
    """
    az = np.radians(azimuth_deg)
    dp = np.radians(dip_deg)

    cos_az, sin_az = np.cos(az), np.sin(az)
    cos_dp, sin_dp = np.cos(dp), np.sin(dp)

    # Row 0 = major axis direction (azimuth from North, dipped)
    # At az=0: points along Y (North). At az=90: points along X (East).
    major_dir = np.array([
        sin_az * cos_dp,   # X component
        cos_az * cos_dp,   # Y component
        -sin_dp,           # Z component (dip down)
    ])

    # Row 1 = minor axis (perpendicular to major in horizontal plane)
    # At az=0: points along X (East). At az=90: points along -Y (South → but sign doesn't matter for distance).
    minor_dir = np.array([
        cos_az,
        -sin_az,
        0.0,
    ])

    # Row 2 = vertical axis (cross product to complete right-hand system)
    vert_dir = np.cross(major_dir, minor_dir)
    norm = np.linalg.norm(vert_dir)
    if norm > 1e-12:
        vert_dir /= norm
    else:
        vert_dir = np.array([0.0, 0.0, 1.0])

    return np.array([major_dir, minor_dir, vert_dir])


def _bounding_box_mask(
    block_centroids: np.ndarray,
    data_coords: np.ndarray,
    buffer_m: float = 0.0,
) -> np.ndarray:
    """Axis-aligned bounding box fallback mask."""
    dmin = data_coords.min(axis=0) - buffer_m
    dmax = data_coords.max(axis=0) + buffer_m
    inside = np.all(
        (block_centroids >= dmin) & (block_centroids <= dmax), axis=1
    )
    return inside
