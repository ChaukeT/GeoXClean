"""
Size Extraction Engine
=======================

Pure geometry computations for fragment size metrics, Feret diameters,
oriented bounding boxes, FSD construction, and Rosin-Rammler / Swebrec
distribution fitting.

No Qt imports. Pure computation module.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import ConvexHull

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Oriented Bounding Box
# ---------------------------------------------------------------------------

def compute_obb(points: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Compute PCA-aligned oriented bounding box.

    Returns
    -------
    dict with: center (3,), axes (3,3), half_extents (3,)
    """
    centered = points - points.mean(axis=0)
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)

    # Sort by eigenvalue descending
    order = np.argsort(eigvals)[::-1]
    axes = eigvecs[:, order].T  # rows = principal axes

    # Project to get extents
    projected = centered @ axes.T
    half_extents = (projected.max(axis=0) - projected.min(axis=0)) / 2.0
    center = points.mean(axis=0)

    return {"center": center, "axes": axes, "half_extents": half_extents}


# ---------------------------------------------------------------------------
# Feret Diameters
# ---------------------------------------------------------------------------

def compute_feret_diameters(points: np.ndarray) -> Tuple[float, float]:
    """
    Compute maximum and minimum Feret diameters from 2D convex hull
    of XY projection (top-down view).

    Returns
    -------
    feret_max, feret_min : float (metres)
    """
    if len(points) < 3:
        span = points.max(axis=0) - points.min(axis=0)
        return float(span.max()), float(span.min())

    xy = points[:, :2]
    try:
        hull = ConvexHull(xy)
        hull_pts = xy[hull.vertices]
    except Exception:
        span = xy.max(axis=0) - xy.min(axis=0)
        return float(span.max()), float(max(span.min(), 1e-9))

    n = len(hull_pts)
    if n < 2:
        return 0.0, 0.0

    # Rotating calipers for max Feret
    max_dist = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(hull_pts[i] - hull_pts[j])
            if d > max_dist:
                max_dist = d

    # Min Feret: minimum width across all edge directions
    min_width = float("inf")
    for i in range(n):
        edge = hull_pts[(i + 1) % n] - hull_pts[i]
        edge_len = np.linalg.norm(edge)
        if edge_len < 1e-12:
            continue
        normal = np.array([-edge[1], edge[0]]) / edge_len
        projections = hull_pts @ normal
        width = projections.max() - projections.min()
        if width < min_width:
            min_width = width

    if min_width == float("inf"):
        min_width = max_dist

    return float(max_dist), float(min_width)


# ---------------------------------------------------------------------------
# Single Fragment Metrics
# ---------------------------------------------------------------------------

def compute_single_fragment(
    points: np.ndarray,
    fragment_id: int,
) -> Dict[str, float]:
    """
    Compute all metrics for a single fragment.

    Returns dict with: equiv_diameter, feret_max, feret_min, volume,
    surface_area, projected_area, aspect_ratio, sphericity, elongation, obb.
    """
    n = len(points)
    centroid = points.mean(axis=0)
    result: Dict[str, any] = {
        "fragment_id": fragment_id,
        "point_count": n,
        "centroid": centroid,
    }

    # Volume from convex hull
    volume = 0.0
    surface_area = 0.0
    if n >= 4:
        try:
            hull = ConvexHull(points)
            volume = float(hull.volume)
            surface_area = float(hull.area)
        except Exception:
            pass

    result["volume"] = volume
    result["surface_area"] = surface_area

    # Equivalent diameter
    if volume > 0:
        result["equiv_diameter"] = float((6 * volume / np.pi) ** (1.0 / 3.0))
    else:
        # Fallback: use bounding box diagonal
        span = points.max(axis=0) - points.min(axis=0)
        result["equiv_diameter"] = float(np.linalg.norm(span) * 0.5)

    # Feret diameters
    feret_max, feret_min = compute_feret_diameters(points)
    result["feret_max"] = feret_max
    result["feret_min"] = feret_min

    # Projected area (2D convex hull of XY)
    if n >= 3:
        try:
            hull2d = ConvexHull(points[:, :2])
            result["projected_area"] = float(hull2d.volume)  # 2D hull "volume" = area
        except Exception:
            result["projected_area"] = 0.0
    else:
        result["projected_area"] = 0.0

    # Aspect ratio
    result["aspect_ratio"] = feret_max / max(feret_min, 1e-9)

    # Sphericity (Wadell): ratio of surface area of equivalent sphere to actual
    if volume > 0 and surface_area > 0:
        r_eq = (3 * volume / (4 * np.pi)) ** (1.0 / 3.0)
        sphere_sa = 4 * np.pi * r_eq**2
        result["sphericity"] = float(np.clip(sphere_sa / surface_area, 0, 1))
    else:
        result["sphericity"] = 0.0

    # Elongation from PCA
    if n >= 3:
        centered = points - centroid
        cov = np.cov(centered.T)
        eigvals = np.maximum(np.linalg.eigvalsh(cov), 0)
        dims = np.sqrt(eigvals * 12)
        dims = np.sort(dims)[::-1]
        result["elongation"] = float(dims[0] / max(dims[-1], 1e-9))
    else:
        result["elongation"] = 1.0

    # OBB
    if n >= 3:
        result["obb"] = compute_obb(points)
    else:
        result["obb"] = None

    return result


# ---------------------------------------------------------------------------
# Batch Fragment Metrics
# ---------------------------------------------------------------------------

def compute_all_fragment_metrics(
    cloud: np.ndarray,
    labels: np.ndarray,
    progress_callback: Optional[Callable] = None,
) -> List[Dict[str, any]]:
    """
    Compute metrics for all fragments.

    Parameters
    ----------
    cloud : (N, 3+) point cloud (at least XYZ)
    labels : (N,) per-point labels, -1 = noise

    Returns
    -------
    List of metric dicts, one per fragment
    """
    xyz = cloud[:, :3]
    unique_ids = np.unique(labels)
    fragment_ids = unique_ids[unique_ids >= 0]

    results = []
    for i, fid in enumerate(fragment_ids):
        mask = labels == fid
        pts = xyz[mask]

        if progress_callback and i % 50 == 0:
            pct = int(90 * (i + 1) / max(len(fragment_ids), 1))
            progress_callback(pct, f"Computing metrics: fragment {i+1}/{len(fragment_ids)}")

        try:
            metrics = compute_single_fragment(pts, int(fid))
            metrics["point_indices"] = np.where(mask)[0]
            results.append(metrics)
        except Exception as e:
            logger.warning("Failed metrics for fragment %d: %s", fid, e)

    if progress_callback:
        progress_callback(100, f"Computed metrics for {len(results)} fragments")

    return results


# ---------------------------------------------------------------------------
# FSD Construction
# ---------------------------------------------------------------------------

def build_fsd(
    equiv_diameters: np.ndarray,
    percentiles: Tuple[float, ...] = (10, 50, 80, 90),
) -> Dict[str, any]:
    """
    Build cumulative passing curve from equivalent diameters.

    Returns
    -------
    dict with: diameters_sorted, cumulative_passing, d10, d50, d80, d90
    """
    valid = equiv_diameters[equiv_diameters > 0]
    if len(valid) == 0:
        return {
            "diameters_sorted": np.array([]),
            "cumulative_passing": np.array([]),
            "d10": 0.0, "d50": 0.0, "d80": 0.0, "d90": 0.0,
        }

    sorted_d = np.sort(valid)
    cum_pass = np.arange(1, len(sorted_d) + 1) / len(sorted_d)

    result = {
        "diameters_sorted": sorted_d,
        "cumulative_passing": cum_pass,
    }
    for p in percentiles:
        key = f"d{int(p)}"
        result[key] = float(np.percentile(valid, p))

    return result


# ---------------------------------------------------------------------------
# Rosin-Rammler Fit
# ---------------------------------------------------------------------------

def fit_rosin_rammler(
    diameters_sorted: np.ndarray,
    cumulative_passing: np.ndarray,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Fit Rosin-Rammler distribution: P(x) = 1 - exp(-(x/x_c)^n)

    Returns
    -------
    (n, x_c) or (None, None) if fit fails
    """
    if len(diameters_sorted) < 5:
        return None, None

    try:
        from scipy.optimize import curve_fit

        # Filter valid range (0 < P < 1)
        mask = (cumulative_passing > 0.01) & (cumulative_passing < 0.99) & (diameters_sorted > 0)
        x = diameters_sorted[mask]
        y = cumulative_passing[mask]

        if len(x) < 5:
            return None, None

        def rr_func(x, n, xc):
            return 1.0 - np.exp(-((x / xc) ** n))

        # Initial guess
        x_c0 = float(np.median(x))
        popt, _ = curve_fit(rr_func, x, y, p0=[1.5, x_c0], bounds=([0.1, 1e-6], [10, x.max() * 10]))

        return float(popt[0]), float(popt[1])
    except Exception as e:
        logger.warning("Rosin-Rammler fit failed: %s", e)
        return None, None


# ---------------------------------------------------------------------------
# Swebrec (Ouchterlony) Fit
# ---------------------------------------------------------------------------

def fit_swebrec(
    diameters_sorted: np.ndarray,
    cumulative_passing: np.ndarray,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Fit Swebrec (Ouchterlony) distribution:
        P(x) = 1 / (1 + (ln(x_max/x) / ln(x_max/x_50))^b)

    Returns
    -------
    (x_max, x_50, b) or (None, None, None) if fit fails
    """
    if len(diameters_sorted) < 5:
        return None, None, None

    try:
        from scipy.optimize import curve_fit

        mask = (cumulative_passing > 0.01) & (cumulative_passing < 0.99) & (diameters_sorted > 0)
        x = diameters_sorted[mask]
        y = cumulative_passing[mask]

        if len(x) < 5:
            return None, None, None

        x_max_est = float(x.max() * 1.2)
        x_50_est = float(np.median(x))

        def swebrec_func(x, x_max, x_50, b):
            ratio = np.log(np.clip(x_max / x, 1.001, None)) / np.log(np.clip(x_max / x_50, 1.001, None))
            return 1.0 / (1.0 + ratio**b)

        popt, _ = curve_fit(
            swebrec_func, x, y,
            p0=[x_max_est, x_50_est, 2.0],
            bounds=([x.max() * 0.9, x.min() * 0.5, 0.5], [x.max() * 5, x.max(), 10]),
            maxfev=5000,
        )

        return float(popt[0]), float(popt[1]), float(popt[2])
    except Exception as e:
        logger.warning("Swebrec fit failed: %s", e)
        return None, None, None
