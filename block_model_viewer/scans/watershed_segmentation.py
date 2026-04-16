"""
Watershed Segmentation Engine
==============================

Classical image-based segmentation pipeline operating on orthorectified RGB
imagery or on a 2D projection of the coloured point cloud.

Pipeline: bilateral filter -> Canny edge detection -> distance transform ->
marker-controlled watershed -> optional GrabCut refinement.

No Qt imports. Pure computation module.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def detect_edges(
    image: np.ndarray,
    low_thresh: int = 50,
    high_thresh: int = 150,
    bilateral_d: int = 9,
    bilateral_sigma_color: float = 75.0,
    bilateral_sigma_space: float = 75.0,
) -> np.ndarray:
    """
    Bilateral filter + Canny edge detection.

    Parameters
    ----------
    image : (H, W, 3) RGB uint8

    Returns
    -------
    edge_mask : (H, W) binary edge mask
    """
    import cv2

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    smoothed = cv2.bilateralFilter(gray, bilateral_d, bilateral_sigma_color, bilateral_sigma_space)
    edges = cv2.Canny(smoothed, low_thresh, high_thresh)
    return edges


def generate_markers(
    edge_mask: np.ndarray,
    dist_thresh_ratio: float = 0.5,
    kernel_size: int = 3,
) -> np.ndarray:
    """
    Distance transform + thresholding + connected components -> markers.

    Parameters
    ----------
    edge_mask : (H, W) binary edge image
    dist_thresh_ratio : fraction of max distance for foreground threshold

    Returns
    -------
    markers : (H, W) int32 marker map for watershed
    """
    import cv2

    # Invert edges to get regions
    inv = cv2.bitwise_not(edge_mask)

    # Morphological opening to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(inv, cv2.MORPH_OPEN, kernel, iterations=2)

    # Distance transform
    dist = cv2.distanceTransform(opened, cv2.DIST_L2, 5)

    # Threshold to get sure foreground
    thresh_val = dist_thresh_ratio * dist.max()
    _, sure_fg = cv2.threshold(dist, thresh_val, 255, cv2.THRESH_BINARY)
    sure_fg = sure_fg.astype(np.uint8)

    # Sure background (dilated opening)
    sure_bg = cv2.dilate(opened, kernel, iterations=3)

    # Unknown region
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Connected components for markers
    n_labels, markers = cv2.connectedComponents(sure_fg)

    # Add 1 so background is 1, not 0
    markers = markers + 1

    # Mark unknown region as 0
    markers[unknown == 255] = 0

    return markers.astype(np.int32)


def run_watershed(
    image: np.ndarray,
    markers: np.ndarray,
) -> np.ndarray:
    """
    Marker-controlled watershed.

    Parameters
    ----------
    image : (H, W, 3) RGB uint8
    markers : (H, W) int32 marker map

    Returns
    -------
    label_map : (H, W) int32, -1 = boundary, positive = region ID
    """
    import cv2

    # OpenCV watershed requires BGR
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    result = markers.copy()
    cv2.watershed(bgr, result)

    # Relabel: watershed marks boundaries as -1, background as 1
    # Shift labels so background becomes -1 and regions start at 0
    output = result.copy()
    output[result == -1] = -1  # boundaries
    output[result == 1] = -1  # background
    output[result > 1] = result[result > 1] - 2  # regions start at 0

    return output


def refine_grabcut(
    image: np.ndarray,
    label_map: np.ndarray,
    fragment_id: int,
    iterations: int = 5,
) -> np.ndarray:
    """
    GrabCut refinement for a single fragment.

    Returns
    -------
    refined_mask : (H, W) binary mask for the refined fragment
    """
    import cv2

    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[label_map == fragment_id] = cv2.GC_PR_FGD
    mask[label_map != fragment_id] = cv2.GC_PR_BGD

    # Definite foreground: eroded region
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    eroded = cv2.erode((label_map == fragment_id).astype(np.uint8), kernel, iterations=2)
    mask[eroded == 1] = cv2.GC_FGD

    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)

    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    try:
        cv2.grabCut(bgr, mask, None, bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        logger.warning("GrabCut failed for fragment %d; returning original mask", fragment_id)
        return (label_map == fragment_id).astype(np.uint8)

    refined = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
    return refined


# ---------------------------------------------------------------------------
# Point Cloud to Image Projection
# ---------------------------------------------------------------------------

def _project_cloud_to_image(
    xyz: np.ndarray,
    rgb: Optional[np.ndarray],
    resolution: float = 0.02,
) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float], float]:
    """
    Project a 3D point cloud to a 2D top-down image for watershed.

    Returns
    -------
    image : (H, W, 3) RGB uint8
    point_to_pixel : (N, 2) int array mapping each point to (row, col)
    origin : (x_min, y_max) world coordinates of image top-left
    pixel_size : metres per pixel
    """
    x, y = xyz[:, 0], xyz[:, 1]
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    w = max(1, int(np.ceil((x_max - x_min) / resolution)))
    h = max(1, int(np.ceil((y_max - y_min) / resolution)))

    # Limit image size
    if w * h > 25_000_000:  # 25 megapixels max
        scale = np.sqrt(25_000_000 / (w * h))
        resolution = resolution / scale
        w = max(1, int(np.ceil((x_max - x_min) / resolution)))
        h = max(1, int(np.ceil((y_max - y_min) / resolution)))

    col = np.clip(((x - x_min) / resolution).astype(int), 0, w - 1)
    row = np.clip(((y_max - y) / resolution).astype(int), 0, h - 1)

    image = np.zeros((h, w, 3), dtype=np.uint8)
    counts = np.zeros((h, w), dtype=np.int32)

    if rgb is not None and len(rgb) == len(xyz):
        rgb_uint8 = rgb if rgb.max() > 1 else (rgb * 255).astype(np.uint8)
        # Accumulate
        for i in range(len(xyz)):
            r, c = row[i], col[i]
            image[r, c] = image[r, c].astype(np.int32) + rgb_uint8[i]
            counts[r, c] += 1
        # Average
        valid = counts > 0
        for ch in range(3):
            image[:, :, ch][valid] = (image[:, :, ch][valid].astype(np.float32) / counts[valid]).astype(np.uint8)
    else:
        # Use elevation as grayscale
        z_norm = ((xyz[:, 2] - xyz[:, 2].min()) / max(xyz[:, 2].ptp(), 1e-6) * 255).astype(np.uint8)
        for i in range(len(xyz)):
            image[row[i], col[i]] = z_norm[i]
            counts[row[i], col[i]] += 1

    point_to_pixel = np.column_stack([row, col])
    return image, point_to_pixel, (x_min, y_max), resolution


# ---------------------------------------------------------------------------
# Pipeline Orchestrator
# ---------------------------------------------------------------------------

def run_watershed_segmentation(
    cloud: np.ndarray,
    canny_low: int = 50,
    canny_high: int = 150,
    dist_thresh_ratio: float = 0.5,
    min_fragment_size: int = 50,
    resolution: float = 0.02,
    progress_callback: Optional[Callable] = None,
) -> np.ndarray:
    """
    Full watershed segmentation pipeline operating on a 2D projection of
    the point cloud.

    Parameters
    ----------
    cloud : (N, 3+) fused cloud. Columns 6:9 = RGB if present.

    Returns
    -------
    labels : (N,) per-point labels, -1 = unassigned
    """
    import cv2

    xyz = cloud[:, :3]
    n = len(xyz)

    # Extract RGB if available (columns 6:9 after normals)
    rgb = None
    if cloud.shape[1] >= 9:
        rgb = cloud[:, 6:9]

    if progress_callback:
        progress_callback(10, "Projecting to 2D image...")

    image, pt_to_px, origin, px_size = _project_cloud_to_image(xyz, rgb, resolution)

    if progress_callback:
        progress_callback(25, "Detecting edges...")

    edges = detect_edges(image, canny_low, canny_high)

    if progress_callback:
        progress_callback(40, "Generating markers...")

    markers = generate_markers(edges, dist_thresh_ratio)

    if progress_callback:
        progress_callback(60, "Running watershed...")

    label_map = run_watershed(image, markers)

    if progress_callback:
        progress_callback(80, "Mapping labels to 3D points...")

    # Map 2D labels back to 3D points
    labels_3d = np.full(n, -1, dtype=np.int32)
    for i in range(n):
        r, c = pt_to_px[i]
        lbl = label_map[r, c]
        if lbl >= 0:
            labels_3d[i] = lbl

    # Filter small fragments
    unique_labels = np.unique(labels_3d)
    for lid in unique_labels:
        if lid < 0:
            continue
        if np.sum(labels_3d == lid) < min_fragment_size:
            labels_3d[labels_3d == lid] = -1

    # Relabel consecutively
    final = np.full_like(labels_3d, -1)
    for new_id, old_id in enumerate(np.unique(labels_3d[labels_3d >= 0])):
        final[labels_3d == old_id] = new_id

    n_frags = len(np.unique(final[final >= 0]))
    if progress_callback:
        progress_callback(100, f"Watershed segmentation: {n_frags} fragments")

    logger.info("Watershed segmentation: %d fragments from %d points", n_frags, n)
    return final
