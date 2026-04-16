"""
Point Cloud Selection Engine
==============================

Pure numpy algorithms for selecting subsets of a point cloud.
All functions return boolean (N,) masks — True = selected.

No Qt or VTK dependency. Designed for use with the interactive editor.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Box (axis-aligned) selection
# ---------------------------------------------------------------------------

def box_select(
    xyz: np.ndarray,
    min_corner: np.ndarray,
    max_corner: np.ndarray,
) -> np.ndarray:
    """
    Select points inside an axis-aligned bounding box.

    Parameters
    ----------
    xyz : (N, 3)
    min_corner : (3,) lower corner
    max_corner : (3,) upper corner

    Returns
    -------
    mask : (N,) bool
    """
    return (
        (xyz[:, 0] >= min_corner[0]) & (xyz[:, 0] <= max_corner[0]) &
        (xyz[:, 1] >= min_corner[1]) & (xyz[:, 1] <= max_corner[1]) &
        (xyz[:, 2] >= min_corner[2]) & (xyz[:, 2] <= max_corner[2])
    )


# ---------------------------------------------------------------------------
# Frustum selection (screen-space rectangle → 3D)
# ---------------------------------------------------------------------------

def frustum_select(
    xyz: np.ndarray,
    view_matrix: np.ndarray,
    proj_matrix: np.ndarray,
    viewport: Tuple[int, int, int, int],
    screen_rect: Tuple[int, int, int, int],
) -> np.ndarray:
    """
    Select points that fall inside a screen-space rectangle (rubber-band box).

    Parameters
    ----------
    xyz : (N, 3) world-space points
    view_matrix : (4, 4) camera view matrix
    proj_matrix : (4, 4) camera projection matrix
    viewport : (x, y, width, height) in pixels
    screen_rect : (x0, y0, x1, y1) rubber-band rectangle in pixels

    Returns
    -------
    mask : (N,) bool
    """
    # Project all points to normalised device coordinates
    n = len(xyz)
    homogeneous = np.column_stack([xyz, np.ones(n)])  # (N, 4)

    # Combined MVP matrix
    mvp = proj_matrix @ view_matrix
    clip = (mvp @ homogeneous.T).T  # (N, 4)

    # Perspective divide
    w = clip[:, 3]
    valid = w != 0
    ndc = np.zeros((n, 3))
    ndc[valid, 0] = clip[valid, 0] / w[valid]
    ndc[valid, 1] = clip[valid, 1] / w[valid]
    ndc[valid, 2] = clip[valid, 2] / w[valid]

    # NDC to screen pixels
    vx, vy, vw, vh = viewport
    screen_x = (ndc[:, 0] * 0.5 + 0.5) * vw + vx
    screen_y = (ndc[:, 1] * 0.5 + 0.5) * vh + vy

    # Check against rubber-band rectangle
    x0, y0, x1, y1 = screen_rect
    sx0, sx1 = min(x0, x1), max(x0, x1)
    sy0, sy1 = min(y0, y1), max(y0, y1)

    mask = (
        valid &
        (screen_x >= sx0) & (screen_x <= sx1) &
        (screen_y >= sy0) & (screen_y <= sy1) &
        (ndc[:, 2] >= -1) & (ndc[:, 2] <= 1)  # Within clip volume
    )
    return mask


# ---------------------------------------------------------------------------
# Lasso (polygon) selection
# ---------------------------------------------------------------------------

def lasso_select(
    xyz: np.ndarray,
    polygon_screen: np.ndarray,
    view_matrix: np.ndarray,
    proj_matrix: np.ndarray,
    viewport: Tuple[int, int, int, int],
) -> np.ndarray:
    """
    Select points inside a 2D polygon drawn on screen.

    Parameters
    ----------
    xyz : (N, 3) world-space points
    polygon_screen : (M, 2) screen-space polygon vertices
    view_matrix, proj_matrix : (4, 4) camera matrices
    viewport : (x, y, width, height)

    Returns
    -------
    mask : (N,) bool
    """
    n = len(xyz)
    homogeneous = np.column_stack([xyz, np.ones(n)])
    mvp = proj_matrix @ view_matrix
    clip = (mvp @ homogeneous.T).T

    w = clip[:, 3]
    valid = w != 0
    ndc = np.zeros((n, 2))
    ndc[valid, 0] = clip[valid, 0] / w[valid]
    ndc[valid, 1] = clip[valid, 1] / w[valid]

    vx, vy, vw, vh = viewport
    screen_x = (ndc[:, 0] * 0.5 + 0.5) * vw + vx
    screen_y = (ndc[:, 1] * 0.5 + 0.5) * vh + vy

    # Point-in-polygon test (ray casting)
    mask = np.zeros(n, dtype=bool)
    poly = polygon_screen
    m = len(poly)

    for i in range(n):
        if not valid[i]:
            continue
        px, py = screen_x[i], screen_y[i]
        inside = False
        j = m - 1
        for k in range(m):
            yi, yj = poly[k, 1], poly[j, 1]
            xi, xj = poly[k, 0], poly[j, 0]
            if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi + 1e-12) + xi):
                inside = not inside
            j = k
        mask[i] = inside

    return mask


def lasso_select_fast(
    xyz: np.ndarray,
    polygon_screen: np.ndarray,
    view_matrix: np.ndarray,
    proj_matrix: np.ndarray,
    viewport: Tuple[int, int, int, int],
) -> np.ndarray:
    """
    Vectorised lasso select using matplotlib path (much faster for large clouds).
    Falls back to lasso_select() if matplotlib unavailable.
    """
    try:
        from matplotlib.path import Path as MplPath
    except ImportError:
        return lasso_select(xyz, polygon_screen, view_matrix, proj_matrix, viewport)

    n = len(xyz)
    homogeneous = np.column_stack([xyz, np.ones(n)])
    mvp = proj_matrix @ view_matrix
    clip = (mvp @ homogeneous.T).T

    w = clip[:, 3]
    valid = w != 0
    ndc_x = np.zeros(n)
    ndc_y = np.zeros(n)
    ndc_x[valid] = clip[valid, 0] / w[valid]
    ndc_y[valid] = clip[valid, 1] / w[valid]

    vx, vy, vw, vh = viewport
    screen_x = (ndc_x * 0.5 + 0.5) * vw + vx
    screen_y = (ndc_y * 0.5 + 0.5) * vh + vy

    screen_pts = np.column_stack([screen_x, screen_y])
    path = MplPath(polygon_screen)
    mask = valid & path.contains_points(screen_pts)
    return mask


# ---------------------------------------------------------------------------
# Brush (sphere) selection
# ---------------------------------------------------------------------------

def brush_select(
    xyz: np.ndarray,
    center_screen: Tuple[float, float],
    radius_screen: float,
    view_matrix: np.ndarray,
    proj_matrix: np.ndarray,
    viewport: Tuple[int, int, int, int],
) -> np.ndarray:
    """
    Select points within a screen-space circle (brush tool).

    Parameters
    ----------
    center_screen : (cx, cy) screen pixels
    radius_screen : radius in pixels

    Returns
    -------
    mask : (N,) bool
    """
    n = len(xyz)
    homogeneous = np.column_stack([xyz, np.ones(n)])
    mvp = proj_matrix @ view_matrix
    clip = (mvp @ homogeneous.T).T

    w = clip[:, 3]
    valid = w != 0
    ndc_x = np.zeros(n)
    ndc_y = np.zeros(n)
    ndc_x[valid] = clip[valid, 0] / w[valid]
    ndc_y[valid] = clip[valid, 1] / w[valid]

    vx, vy, vw, vh = viewport
    screen_x = (ndc_x * 0.5 + 0.5) * vw + vx
    screen_y = (ndc_y * 0.5 + 0.5) * vh + vy

    cx, cy = center_screen
    dist_sq = (screen_x - cx) ** 2 + (screen_y - cy) ** 2
    mask = valid & (dist_sq <= radius_screen ** 2)
    return mask


# ---------------------------------------------------------------------------
# 3D sphere selection (for world-space proximity)
# ---------------------------------------------------------------------------

def sphere_select(
    xyz: np.ndarray,
    center: np.ndarray,
    radius: float,
) -> np.ndarray:
    """Select points within a 3D sphere."""
    dist_sq = np.sum((xyz - center) ** 2, axis=1)
    return dist_sq <= radius ** 2


# ---------------------------------------------------------------------------
# Elevation slice
# ---------------------------------------------------------------------------

def elevation_select(
    xyz: np.ndarray,
    z_min: float,
    z_max: float,
) -> np.ndarray:
    """Select points within an elevation range."""
    return (xyz[:, 2] >= z_min) & (xyz[:, 2] <= z_max)


# ---------------------------------------------------------------------------
# Flood select (connected region)
# ---------------------------------------------------------------------------

def flood_select(
    xyz: np.ndarray,
    seed_index: int,
    distance_threshold: float = 0.1,
    max_points: int = 50000,
    normals: Optional[np.ndarray] = None,
    normal_threshold_deg: float = 45.0,
) -> np.ndarray:
    """
    Flood-fill selection from a seed point. Grows to connected neighbours
    within distance and (optionally) normal similarity threshold.

    Returns
    -------
    mask : (N,) bool
    """
    from scipy.spatial import cKDTree

    n = len(xyz)
    mask = np.zeros(n, dtype=bool)
    tree = cKDTree(xyz)

    cos_thresh = np.cos(np.radians(normal_threshold_deg)) if normals is not None else None

    queue = [seed_index]
    mask[seed_index] = True
    count = 1

    while queue and count < max_points:
        current = queue.pop(0)
        nbrs = tree.query_ball_point(xyz[current], r=distance_threshold)

        for ni in nbrs:
            if mask[ni]:
                continue

            # Normal similarity check
            if normals is not None and cos_thresh is not None:
                dot = np.dot(normals[current], normals[ni])
                if dot < cos_thresh:
                    continue

            mask[ni] = True
            queue.append(ni)
            count += 1

    return mask


# ---------------------------------------------------------------------------
# Selection utilities
# ---------------------------------------------------------------------------

def invert_selection(mask: np.ndarray) -> np.ndarray:
    """Invert a selection mask."""
    return ~mask


def grow_selection(
    xyz: np.ndarray,
    mask: np.ndarray,
    distance: float,
) -> np.ndarray:
    """Grow selection by a distance (dilate)."""
    from scipy.spatial import cKDTree

    selected = xyz[mask]
    if len(selected) == 0:
        return mask

    tree = cKDTree(selected)
    dists, _ = tree.query(xyz, k=1)
    return mask | (dists <= distance)


def shrink_selection(
    xyz: np.ndarray,
    mask: np.ndarray,
    distance: float,
) -> np.ndarray:
    """Shrink selection by a distance (erode)."""
    from scipy.spatial import cKDTree

    unselected = xyz[~mask]
    if len(unselected) == 0:
        return mask

    tree = cKDTree(unselected)
    dists, _ = tree.query(xyz, k=1)
    # Remove selected points that are within `distance` of unselected
    return mask & (dists > distance)


def selection_stats(
    xyz: np.ndarray,
    mask: np.ndarray,
) -> dict:
    """Compute statistics about the selection."""
    selected = xyz[mask]
    n_sel = int(mask.sum())
    n_total = len(mask)

    if n_sel == 0:
        return {
            "count": 0, "total": n_total, "fraction": 0.0,
            "bbox_min": None, "bbox_max": None,
            "centroid": None, "mean_z": None,
        }

    return {
        "count": n_sel,
        "total": n_total,
        "fraction": n_sel / max(n_total, 1),
        "bbox_min": selected.min(axis=0).tolist(),
        "bbox_max": selected.max(axis=0).tolist(),
        "centroid": selected.mean(axis=0).tolist(),
        "mean_z": float(selected[:, 2].mean()),
    }
