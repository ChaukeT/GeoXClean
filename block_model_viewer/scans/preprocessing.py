"""
Fragmentation Preprocessing Engine
====================================

Pure computation module for LiDAR point cloud cleaning, normal estimation,
curvature computation, RGB image correction, and LiDAR-RGB fusion.

No Qt imports. All functions accept numpy arrays and configuration parameters,
return numpy arrays and result dictionaries.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SORConfig:
    """Statistical Outlier Removal configuration."""
    k_neighbors: int = 20
    std_ratio: float = 2.0


@dataclass
class RORConfig:
    """Radius Outlier Removal configuration."""
    radius: float = 0.5  # metres
    min_points: int = 6


@dataclass
class CSFConfig:
    """Cloth Simulation Filter ground separation configuration."""
    cloth_resolution: float = 0.5
    threshold: float = 0.5
    max_iterations: int = 500
    classify_threshold: float = 0.5


@dataclass
class VoxelConfig:
    """Voxel downsampling configuration."""
    voxel_size: float = 0.05  # metres


@dataclass
class NormalConfig:
    """Normal estimation configuration."""
    k_neighbors: int = 30
    radius: Optional[float] = None  # If set, use hybrid radius+knn


@dataclass
class FusionConfig:
    """LiDAR-RGB fusion configuration."""
    camera_model_path: Optional[str] = None  # Path to camera calibration
    # If no camera model, assume co-registered ortho image
    image_resolution: float = 0.01  # metres per pixel (for ortho mapping)


@dataclass
class PreprocessingConfig:
    """Complete preprocessing pipeline configuration."""
    # Cleaning
    enable_sor: bool = True
    sor: SORConfig = field(default_factory=SORConfig)
    enable_ror: bool = False
    ror: RORConfig = field(default_factory=RORConfig)

    # Ground filtering
    enable_ground_filter: bool = False
    csf: CSFConfig = field(default_factory=CSFConfig)

    # Downsampling
    enable_voxel_downsample: bool = False
    voxel: VoxelConfig = field(default_factory=VoxelConfig)

    # Normals and curvature
    normals: NormalConfig = field(default_factory=NormalConfig)

    # Fusion
    enable_fusion: bool = False
    fusion: FusionConfig = field(default_factory=FusionConfig)


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def statistical_outlier_removal(
    cloud: np.ndarray,
    k: int = 20,
    std_ratio: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Statistical Outlier Removal filter.

    Parameters
    ----------
    cloud : (N, 3+) point cloud
    k : number of nearest neighbours for mean distance
    std_ratio : standard deviation multiplier threshold

    Returns
    -------
    cleaned : (M, 3+) cleaned point cloud
    removed_indices : (N-M,) indices of removed points
    """
    try:
        import open3d as o3d
    except ImportError:
        logger.warning("open3d not available; falling back to scipy SOR")
        return _sor_scipy(cloud, k, std_ratio)

    xyz = cloud[:, :3].astype(np.float64)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    _, inlier_idx = pcd.remove_statistical_outlier(nb_neighbors=k, std_ratio=std_ratio)
    inlier_set = set(inlier_idx)
    removed = np.array([i for i in range(len(cloud)) if i not in inlier_set], dtype=np.intp)

    return cloud[inlier_idx], removed


def _sor_scipy(
    cloud: np.ndarray, k: int, std_ratio: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Fallback SOR using scipy KDTree."""
    from scipy.spatial import cKDTree

    xyz = cloud[:, :3]
    tree = cKDTree(xyz)
    dists, _ = tree.query(xyz, k=k + 1)
    mean_dists = dists[:, 1:].mean(axis=1)
    global_mean = mean_dists.mean()
    global_std = mean_dists.std()
    threshold = global_mean + std_ratio * global_std
    mask = mean_dists <= threshold
    removed = np.where(~mask)[0]
    return cloud[mask], removed


def radius_outlier_removal(
    cloud: np.ndarray,
    radius: float = 0.5,
    min_pts: int = 6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Radius Outlier Removal filter.

    Returns
    -------
    cleaned : (M, 3+)
    removed_indices : (N-M,)
    """
    try:
        import open3d as o3d

        xyz = cloud[:, :3].astype(np.float64)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        _, inlier_idx = pcd.remove_radius_outlier(nb_points=min_pts, radius=radius)
        inlier_set = set(inlier_idx)
        removed = np.array([i for i in range(len(cloud)) if i not in inlier_set], dtype=np.intp)
        return cloud[inlier_idx], removed
    except ImportError:
        from scipy.spatial import cKDTree

        xyz = cloud[:, :3]
        tree = cKDTree(xyz)
        counts = tree.query_ball_point(xyz, r=radius, return_length=True)
        mask = counts >= min_pts
        removed = np.where(~mask)[0]
        return cloud[mask], removed


def ground_filter_csf(
    cloud: np.ndarray,
    cloth_resolution: float = 0.5,
    threshold: float = 0.5,
    max_iterations: int = 500,
    classify_threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cloth Simulation Filter for ground/non-ground separation.

    Returns
    -------
    ground : (G, 3+)
    non_ground : (NG, 3+)
    """
    try:
        import CSF

        csf = CSF.CSF()
        csf.params.bSloopSmooth = False
        csf.params.cloth_resolution = cloth_resolution
        csf.params.rigidness = 1
        csf.params.time_step = 0.65
        csf.params.class_threshold = classify_threshold
        csf.params.interations = max_iterations

        csf.setPointCloud(cloud[:, :3].astype(np.float64))
        ground_idx = CSF.VecInt()
        non_ground_idx = CSF.VecInt()
        csf.do_filtering(ground_idx, non_ground_idx)

        g_idx = np.array(ground_idx)
        ng_idx = np.array(non_ground_idx)
        return cloud[g_idx], cloud[ng_idx]
    except ImportError:
        logger.warning("CSF not installed; returning all points as non-ground")
        return np.empty((0, cloud.shape[1])), cloud


def voxel_downsample(
    cloud: np.ndarray,
    voxel_size: float = 0.05,
) -> np.ndarray:
    """
    Voxel grid downsampling.

    Returns
    -------
    downsampled : (M, 3+) with one point per occupied voxel
    """
    try:
        import open3d as o3d

        xyz = cloud[:, :3].astype(np.float64)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        down = pcd.voxel_down_sample(voxel_size=voxel_size)
        down_xyz = np.asarray(down.points)

        # Map back to original cloud to preserve extra columns
        if cloud.shape[1] > 3:
            from scipy.spatial import cKDTree
            tree = cKDTree(cloud[:, :3])
            _, idx = tree.query(down_xyz, k=1)
            return cloud[idx]
        return down_xyz
    except ImportError:
        # Simple grid-based fallback
        xyz = cloud[:, :3]
        keys = np.floor(xyz / voxel_size).astype(np.int64)
        _, unique_idx = np.unique(keys, axis=0, return_index=True)
        return cloud[unique_idx]


def estimate_normals(
    cloud: np.ndarray,
    k: int = 30,
    radius: Optional[float] = None,
) -> np.ndarray:
    """
    PCA normal estimation.

    Parameters
    ----------
    cloud : (N, 3+)

    Returns
    -------
    normals : (N, 3) unit normals
    """
    try:
        import open3d as o3d

        xyz = cloud[:, :3].astype(np.float64)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        search = o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius if radius else (k * 0.1),
            max_nn=k,
        )
        pcd.estimate_normals(search_param=search)
        pcd.orient_normals_consistent_tangent_plane(k=k)
        return np.asarray(pcd.normals)
    except ImportError:
        return _normals_scipy(cloud[:, :3], k)


def _normals_scipy(xyz: np.ndarray, k: int) -> np.ndarray:
    """Fallback PCA normals using scipy."""
    from scipy.spatial import cKDTree

    tree = cKDTree(xyz)
    _, idx = tree.query(xyz, k=k)
    normals = np.zeros_like(xyz)
    for i in range(len(xyz)):
        nbrs = xyz[idx[i]]
        cov = np.cov(nbrs.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        normals[i] = eigvecs[:, 0]  # smallest eigenvalue = normal
    # Normalise
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return normals / norms


def compute_curvature(
    cloud: np.ndarray,
    normals: np.ndarray,
    k: int = 15,
) -> np.ndarray:
    """
    Eigenvalue-based curvature estimation.

    Returns
    -------
    curvature : (N,) curvature values
    """
    from scipy.spatial import cKDTree

    xyz = cloud[:, :3]
    tree = cKDTree(xyz)
    _, idx = tree.query(xyz, k=k)

    curvature = np.zeros(len(xyz))
    for i in range(len(xyz)):
        nbrs = xyz[idx[i]]
        cov = np.cov(nbrs.T)
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.maximum(eigvals, 0)
        total = eigvals.sum()
        if total > 0:
            curvature[i] = eigvals[0] / total
    return curvature


def clahe_correction(
    image: np.ndarray,
    clip_limit: float = 2.0,
    grid_size: int = 8,
) -> np.ndarray:
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalisation) on L-channel.

    Parameters
    ----------
    image : (H, W, 3) RGB uint8

    Returns
    -------
    corrected : (H, W, 3) RGB uint8
    """
    import cv2

    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def fuse_lidar_rgb(
    cloud: np.ndarray,
    image: np.ndarray,
    image_origin: Tuple[float, float] = (0.0, 0.0),
    pixel_size: float = 0.01,
) -> np.ndarray:
    """
    Fuse LiDAR XYZ with ortho RGB by projecting points onto image plane.

    Assumes a top-down orthorectified image aligned with the cloud XY plane.

    Parameters
    ----------
    cloud : (N, 3+) — at least XYZ
    image : (H, W, 3) RGB uint8
    image_origin : (x0, y0) of image top-left corner in world coords
    pixel_size : metres per pixel

    Returns
    -------
    fused : (N, 6+) — X, Y, Z, R, G, B [, extra original columns...]
    """
    xyz = cloud[:, :3]
    h, w = image.shape[:2]

    # Project to image pixel coords
    col = ((xyz[:, 0] - image_origin[0]) / pixel_size).astype(int)
    row = ((image_origin[1] - xyz[:, 1]) / pixel_size).astype(int)  # Y inverted

    # Clip to image bounds
    col = np.clip(col, 0, w - 1)
    row = np.clip(row, 0, h - 1)

    rgb = image[row, col].astype(np.float32)  # (N, 3)

    # Build fused array: XYZ + RGB + extra columns
    if cloud.shape[1] > 3:
        fused = np.column_stack([xyz, rgb, cloud[:, 3:]])
    else:
        fused = np.column_stack([xyz, rgb])

    return fused


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------

def run_preprocessing_pipeline(
    cloud: np.ndarray,
    config: PreprocessingConfig,
    image: Optional[np.ndarray] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> Dict[str, Any]:
    """
    Run the full preprocessing pipeline.

    Parameters
    ----------
    cloud : (N, 3+) raw LiDAR points
    config : PreprocessingConfig
    image : optional (H, W, 3) RGB image for fusion
    progress_callback : (percent, message)

    Returns
    -------
    dict with keys:
        'cloud' : preprocessed point cloud
        'normals' : (N, 3) normals
        'curvature' : (N,) curvature
        'ground' : ground points (if ground filtering enabled)
        'stats' : preprocessing statistics
    """
    total_stages = sum([
        config.enable_sor,
        config.enable_ror,
        config.enable_ground_filter,
        config.enable_voxel_downsample,
        True,  # normals (always)
        True,  # curvature (always)
        config.enable_fusion and image is not None,
    ])
    stage = 0

    def _progress(msg: str):
        nonlocal stage
        stage += 1
        if progress_callback:
            pct = int(100 * stage / max(total_stages, 1))
            progress_callback(pct, msg)

    stats: Dict[str, Any] = {"input_points": len(cloud)}
    result_cloud = cloud.copy()
    ground = None

    # Stage: SOR
    if config.enable_sor:
        result_cloud, removed = statistical_outlier_removal(
            result_cloud, config.sor.k_neighbors, config.sor.std_ratio
        )
        stats["sor_removed"] = len(removed)
        _progress(f"SOR: removed {len(removed)} outliers")

    # Stage: ROR
    if config.enable_ror:
        result_cloud, removed = radius_outlier_removal(
            result_cloud, config.ror.radius, config.ror.min_points
        )
        stats["ror_removed"] = len(removed)
        _progress(f"ROR: removed {len(removed)} outliers")

    # Stage: Ground filter
    if config.enable_ground_filter:
        ground, result_cloud = ground_filter_csf(
            result_cloud,
            config.csf.cloth_resolution,
            config.csf.threshold,
            config.csf.max_iterations,
            config.csf.classify_threshold,
        )
        stats["ground_points"] = len(ground)
        stats["non_ground_points"] = len(result_cloud)
        _progress(f"Ground filter: {len(ground)} ground, {len(result_cloud)} non-ground")

    # Stage: Voxel downsample
    if config.enable_voxel_downsample:
        before = len(result_cloud)
        result_cloud = voxel_downsample(result_cloud, config.voxel.voxel_size)
        stats["voxel_before"] = before
        stats["voxel_after"] = len(result_cloud)
        _progress(f"Voxel downsample: {before} -> {len(result_cloud)}")

    # Stage: Normals
    normals = estimate_normals(result_cloud, config.normals.k_neighbors, config.normals.radius)
    stats["normals_computed"] = True
    _progress("Normal estimation complete")

    # Stage: Curvature
    curvature = compute_curvature(result_cloud, normals, k=min(15, config.normals.k_neighbors))
    stats["curvature_computed"] = True
    _progress("Curvature estimation complete")

    # Stage: Fusion
    if config.enable_fusion and image is not None:
        corrected = clahe_correction(image, clip_limit=2.0, grid_size=8)
        result_cloud = fuse_lidar_rgb(result_cloud, corrected)
        stats["fused"] = True
        _progress("LiDAR-RGB fusion complete")

    stats["output_points"] = len(result_cloud)

    return {
        "cloud": result_cloud,
        "normals": normals,
        "curvature": curvature,
        "ground": ground,
        "stats": stats,
    }
