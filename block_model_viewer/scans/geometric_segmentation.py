"""
Geometric Segmentation Engine
==============================

Point cloud-based segmentation using curvature-seeded region growing,
iterative RANSAC plane decomposition, DBSCAN fallback for residuals,
and post-process merge of over-segmented regions.

No Qt imports. Pure computation module.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Region Growing (curvature-seeded)
# ---------------------------------------------------------------------------

def region_growing(
    xyz: np.ndarray,
    normals: np.ndarray,
    curvature: np.ndarray,
    angle_thresh_deg: float = 30.0,
    curv_thresh: float = 0.01,
    min_size: int = 100,
    max_size: int = 1_000_000,
    k_neighbors: int = 15,
    progress_callback: Optional[Callable] = None,
) -> np.ndarray:
    """
    Region growing from curvature-minima seeds.

    Parameters
    ----------
    xyz : (N, 3) point coordinates
    normals : (N, 3) unit normals
    curvature : (N,) curvature values
    angle_thresh_deg : max angle difference between normals to merge
    curv_thresh : curvature threshold for seed selection (low curvature = seeds)
    min_size : minimum points per region
    max_size : maximum points per region

    Returns
    -------
    labels : (N,) integer labels, -1 = unassigned
    """
    n = len(xyz)
    labels = np.full(n, -1, dtype=np.int32)
    cos_thresh = np.cos(np.radians(angle_thresh_deg))

    # Build KDTree
    tree = cKDTree(xyz)

    # Sort by curvature ascending (low curvature = flat = good seeds)
    sorted_idx = np.argsort(curvature)

    label_id = 0
    visited = np.zeros(n, dtype=bool)

    for seed_pos, seed_i in enumerate(sorted_idx):
        if visited[seed_i]:
            continue
        if curvature[seed_i] > curv_thresh and label_id > 0:
            # Only use low-curvature points as seeds once we have at least one region
            continue

        # BFS region growing
        queue = [seed_i]
        region = []
        visited[seed_i] = True

        while queue and len(region) < max_size:
            current = queue.pop(0)
            region.append(current)

            # Query neighbours
            _, nbr_idx = tree.query(xyz[current], k=k_neighbors)

            for ni in nbr_idx:
                if ni >= n or visited[ni]:
                    continue
                # Normal similarity check
                dot = np.dot(normals[current], normals[ni])
                if dot >= cos_thresh:
                    visited[ni] = True
                    queue.append(ni)

        # Accept region if large enough
        if len(region) >= min_size:
            labels[region] = label_id
            label_id += 1

        if progress_callback and seed_pos % 1000 == 0:
            pct = min(80, int(80 * seed_pos / n))
            progress_callback(pct, f"Region growing: {label_id} regions found")

    logger.info("Region growing: %d regions from %d points", label_id, n)
    return labels


# ---------------------------------------------------------------------------
# RANSAC Plane Decomposition
# ---------------------------------------------------------------------------

def ransac_plane_decomposition(
    xyz: np.ndarray,
    dist_thresh: float = 0.05,
    min_pts: int = 100,
    max_planes: int = 50,
) -> Tuple[np.ndarray, list]:
    """
    Iterative RANSAC plane fitting.

    Returns
    -------
    labels : (N,) plane labels, -1 = residual
    plane_models : list of (normal, d) tuples
    """
    try:
        import open3d as o3d
    except ImportError:
        logger.warning("open3d not available; skipping RANSAC plane decomposition")
        return np.full(len(xyz), -1, dtype=np.int32), []

    remaining_idx = np.arange(len(xyz))
    labels = np.full(len(xyz), -1, dtype=np.int32)
    plane_models = []
    label_id = 0

    for _ in range(max_planes):
        if len(remaining_idx) < min_pts:
            break

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz[remaining_idx].astype(np.float64))

        try:
            plane_model, inliers = pcd.segment_plane(
                distance_threshold=dist_thresh,
                ransac_n=3,
                num_iterations=1000,
            )
        except Exception:
            break

        if len(inliers) < min_pts:
            break

        # Map local inlier indices back to global
        global_inliers = remaining_idx[inliers]
        labels[global_inliers] = label_id
        plane_models.append(plane_model)
        label_id += 1

        # Remove inliers from remaining
        mask = np.ones(len(remaining_idx), dtype=bool)
        mask[inliers] = False
        remaining_idx = remaining_idx[mask]

    logger.info("RANSAC: %d planes extracted", label_id)
    return labels, plane_models


# ---------------------------------------------------------------------------
# DBSCAN Fallback
# ---------------------------------------------------------------------------

def dbscan_clustering(
    xyz: np.ndarray,
    eps: float = 0.05,
    min_samples: int = 20,
) -> np.ndarray:
    """
    DBSCAN as fallback for residual points after RANSAC.

    Returns
    -------
    labels : (N,) cluster labels, -1 = noise
    """
    try:
        from sklearn.cluster import DBSCAN
        db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        labels = db.fit_predict(xyz)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        logger.info("DBSCAN: %d clusters from %d points", n_clusters, len(xyz))
        return labels
    except ImportError:
        logger.warning("sklearn not available; DBSCAN fallback disabled")
        return np.full(len(xyz), -1, dtype=np.int32)


# ---------------------------------------------------------------------------
# Post-process: Merge Over-segmented Regions
# ---------------------------------------------------------------------------

def merge_oversegmented(
    labels: np.ndarray,
    xyz: np.ndarray,
    normals: np.ndarray,
    merge_thresh: float = 0.8,
) -> np.ndarray:
    """
    Merge adjacent regions with similar normals.

    Parameters
    ----------
    labels : (N,) current labels
    merge_thresh : cosine similarity threshold for merging (0..1)

    Returns
    -------
    labels : (N,) merged labels
    """
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels >= 0]

    if len(unique_labels) <= 1:
        return labels

    # Compute mean normal per region
    mean_normals = {}
    centroids = {}
    for lid in unique_labels:
        mask = labels == lid
        mean_n = normals[mask].mean(axis=0)
        norm = np.linalg.norm(mean_n)
        if norm > 0:
            mean_normals[lid] = mean_n / norm
        else:
            mean_normals[lid] = np.array([0, 0, 1.0])
        centroids[lid] = xyz[mask].mean(axis=0)

    # Build adjacency by checking which regions have nearby points
    tree = cKDTree(xyz)
    merged = labels.copy()
    merge_map = {lid: lid for lid in unique_labels}

    for lid in unique_labels:
        mask = labels == lid
        boundary_pts = xyz[mask]
        # Sample boundary points for efficiency
        if len(boundary_pts) > 200:
            sample_idx = np.random.choice(len(boundary_pts), 200, replace=False)
            boundary_pts = boundary_pts[sample_idx]

        # Find nearby points from other regions
        for pt in boundary_pts[:50]:  # Check up to 50 boundary points
            nbrs = tree.query_ball_point(pt, r=0.1)
            for ni in nbrs:
                other_lid = labels[ni]
                if other_lid < 0 or other_lid == lid:
                    continue
                # Check normal similarity
                if lid in mean_normals and other_lid in mean_normals:
                    cos_sim = abs(np.dot(mean_normals[lid], mean_normals[other_lid]))
                    if cos_sim >= merge_thresh:
                        # Merge: remap the smaller to the larger
                        src = min(lid, other_lid)
                        dst = max(lid, other_lid)
                        # Follow merge chain
                        while merge_map[src] != src:
                            src = merge_map[src]
                        while merge_map[dst] != dst:
                            dst = merge_map[dst]
                        if src != dst:
                            merge_map[src] = dst

    # Apply merge map
    for lid in unique_labels:
        root = lid
        while merge_map[root] != root:
            root = merge_map[root]
        if root != lid:
            merged[labels == lid] = root

    # Relabel consecutively
    final_labels = np.full_like(merged, -1)
    for new_id, old_id in enumerate(np.unique(merged[merged >= 0])):
        final_labels[merged == old_id] = new_id

    n_before = len(unique_labels)
    n_after = len(np.unique(final_labels[final_labels >= 0]))
    if n_before != n_after:
        logger.info("Merged %d -> %d regions", n_before, n_after)

    return final_labels


# ---------------------------------------------------------------------------
# Pipeline Orchestrator
# ---------------------------------------------------------------------------

def run_geometric_segmentation(
    cloud: np.ndarray,
    normal_threshold_deg: float = 30.0,
    curvature_threshold: float = 0.01,
    min_region_size: int = 100,
    max_region_size: int = 1_000_000,
    k_neighbors: int = 15,
    ransac_dist_thresh: float = 0.05,
    enable_ransac: bool = False,
    enable_dbscan_fallback: bool = True,
    dbscan_eps: float = 0.05,
    dbscan_min_samples: int = 20,
    merge_threshold: float = 0.8,
    progress_callback: Optional[Callable] = None,
) -> np.ndarray:
    """
    Full geometric segmentation pipeline.

    Parameters
    ----------
    cloud : (N, 3+) fused cloud (at least XYZ; columns 3:6 = normals if present)

    Returns
    -------
    labels : (N,) per-point labels, -1 = unassigned/noise
    """
    xyz = cloud[:, :3]
    n = len(xyz)

    # Extract or compute normals
    if cloud.shape[1] >= 6:
        normals = cloud[:, 3:6]
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        normals = normals / norms
    else:
        from .preprocessing import estimate_normals
        normals = estimate_normals(cloud, k=k_neighbors)

    # Compute curvature
    from .preprocessing import compute_curvature
    curvature = compute_curvature(cloud, normals, k=min(k_neighbors, 15))

    if progress_callback:
        progress_callback(10, "Starting region growing...")

    # Step 1: Region growing
    labels = region_growing(
        xyz, normals, curvature,
        angle_thresh_deg=normal_threshold_deg,
        curv_thresh=curvature_threshold,
        min_size=min_region_size,
        max_size=max_region_size,
        k_neighbors=k_neighbors,
        progress_callback=progress_callback,
    )

    # Step 2: RANSAC on unassigned points (optional)
    if enable_ransac:
        unassigned = labels == -1
        if np.sum(unassigned) > min_region_size:
            if progress_callback:
                progress_callback(60, "RANSAC on residuals...")
            ransac_labels, _ = ransac_plane_decomposition(
                xyz[unassigned], ransac_dist_thresh, min_region_size,
            )
            # Offset RANSAC labels to avoid collision
            max_label = labels.max() + 1 if labels.max() >= 0 else 0
            valid = ransac_labels >= 0
            labels[np.where(unassigned)[0][valid]] = ransac_labels[valid] + max_label

    # Step 3: DBSCAN fallback on remaining unassigned
    if enable_dbscan_fallback:
        unassigned = labels == -1
        if np.sum(unassigned) > dbscan_min_samples:
            if progress_callback:
                progress_callback(75, "DBSCAN on residuals...")
            db_labels = dbscan_clustering(
                xyz[unassigned], eps=dbscan_eps, min_samples=dbscan_min_samples,
            )
            max_label = labels.max() + 1 if labels.max() >= 0 else 0
            valid = db_labels >= 0
            labels[np.where(unassigned)[0][valid]] = db_labels[valid] + max_label

    # Step 4: Merge over-segmented regions
    if merge_threshold < 1.0:
        if progress_callback:
            progress_callback(85, "Merging over-segmented regions...")
        labels = merge_oversegmented(labels, xyz, normals, merge_threshold)

    if progress_callback:
        n_frags = len(np.unique(labels[labels >= 0]))
        progress_callback(100, f"Geometric segmentation: {n_frags} fragments")

    return labels
