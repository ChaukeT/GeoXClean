"""
Scan Cleaning Engine
====================

Removes outliers and estimates surface normals for fragmentation analysis.
Provides multiple cleaning strategies with configurable parameters.
"""

import logging
import numpy as np
from typing import Optional, Dict, Any, Tuple
from uuid import UUID
from datetime import datetime

from .scan_models import ScanData, CleaningReport

logger = logging.getLogger(__name__)


class ScanCleaner:
    """
    Cleans scan data by removing outliers and estimating normals.

    Supports multiple cleaning strategies optimized for fragmentation analysis.
    """

    def __init__(self):
        """Initialize cleaner with available libraries."""
        # LAZY INITIALIZATION: Don't import open3d at startup (can hang on some GPU configs)
        self._open3d_available: Optional[bool] = None
        self._scipy_available: Optional[bool] = None
        logger.info("ScanCleaner initialized (lazy dependency checks)")

    def _ensure_dependencies_checked(self):
        """Check dependencies lazily on first use."""
        if self._open3d_available is None:
            self._open3d_available = self._check_open3d()
            self._scipy_available = self._check_scipy()
            logger.info("Scan cleaner dependencies checked:")
            logger.info(f"  Open3D support: {self._open3d_available}")
            logger.info(f"  SciPy support: {self._scipy_available}")

    def _check_open3d(self) -> bool:
        """Check if Open3D is available."""
        try:
            import open3d as o3d
            return True
        except ImportError:
            return False

    def _check_scipy(self) -> bool:
        """Check if SciPy is available."""
        try:
            import scipy.spatial
            return True
        except ImportError:
            return False

    def clean_scan(self, scan_data: ScanData, scan_id: UUID,
                  outlier_method: str = "statistical",
                  normal_method: str = "auto",
                  outlier_params: Optional[Dict[str, Any]] = None,
                  normal_params: Optional[Dict[str, Any]] = None) -> Tuple[ScanData, CleaningReport]:
        """
        Clean scan data by removing outliers and estimating normals.

        Args:
            scan_data: Input scan data
            scan_id: UUID of the scan
            outlier_method: Method for outlier removal ("statistical", "radius", "none")
            normal_method: Method for normal estimation ("pca", "jet", "auto", "none")
            outlier_params: Parameters for outlier removal
            normal_params: Parameters for normal estimation

        Returns:
            Tuple of (cleaned_scan_data, cleaning_report)
        """
        # Ensure dependencies are checked (lazy init)
        self._ensure_dependencies_checked()

        if scan_data.points is None:
            raise ValueError("Cannot clean scan with no point data")

        original_point_count = len(scan_data.points)
        cleaned_points = scan_data.points.copy()
        cleaned_colors = scan_data.colors.copy() if scan_data.colors is not None else None
        cleaned_normals = scan_data.normals
        cleaned_intensities = scan_data.intensities.copy() if scan_data.intensities is not None else None

        # Track cleaning operations
        outliers_removed = 0
        duplicates_removed = 0

        # Remove duplicates first
        cleaned_points, cleaned_colors, cleaned_intensities, duplicates_removed = \
            self._remove_duplicates(cleaned_points, cleaned_colors, cleaned_intensities)

        # Remove outliers
        if outlier_method != "none":
            outlier_params = outlier_params or {}
            cleaned_points, cleaned_colors, cleaned_intensities, outliers_removed = \
                self._remove_outliers(cleaned_points, cleaned_colors, cleaned_intensities,
                                    method=outlier_method, **outlier_params)

        # Estimate normals if requested
        normals_computed = False
        normal_computation_method = None

        if normal_method != "none":
            normal_params = normal_params or {}
            cleaned_normals, normals_computed, normal_computation_method = \
                self._estimate_normals(cleaned_points, method=normal_method, **normal_params)

        # Create cleaned scan data
        cleaned_scan = ScanData(
            points=cleaned_points,
            faces=scan_data.faces,  # Faces unchanged
            normals=cleaned_normals,
            colors=cleaned_colors,
            intensities=cleaned_intensities,
            classifications=scan_data.classifications,  # Classifications unchanged
            crs=scan_data.crs,
            units=scan_data.units,
            file_format=scan_data.file_format,
            is_cleaned=True,
            has_normals=normals_computed
        )

        # Create cleaning report
        report = CleaningReport(
            scan_id=scan_id,
            timestamp=datetime.now(),
            input_point_count=original_point_count,
            output_point_count=len(cleaned_points),
            outliers_removed=outliers_removed,
            duplicates_removed=duplicates_removed,
            normals_computed=normals_computed,
            normal_method=normal_computation_method
        )

        logger.info(f"Cleaning complete for scan {scan_id}: {original_point_count} → {len(cleaned_points)} points "
                   f"(removed {outliers_removed} outliers, {duplicates_removed} duplicates)")

        return cleaned_scan, report

    def _remove_duplicates(self, points: np.ndarray,
                          colors: Optional[np.ndarray] = None,
                          intensities: Optional[np.ndarray] = None,
                          tolerance: float = 0.001) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], int]:
        """
        Remove duplicate/near-duplicate points.

        Args:
            points: Point coordinates
            colors: Optional color array
            intensities: Optional intensity array
            tolerance: Distance tolerance for duplicates

        Returns:
            Tuple of (cleaned_points, cleaned_colors, cleaned_intensities, duplicates_removed)
        """
        from scipy.spatial import cKDTree

        tree = cKDTree(points)
        to_keep = np.ones(len(points), dtype=bool)

        # Find duplicate pairs
        duplicate_pairs = tree.query_pairs(r=tolerance)

        # Mark duplicates for removal (keep first occurrence)
        removed_indices = set()
        for i, j in duplicate_pairs:
            if j not in removed_indices:
                to_keep[j] = False
                removed_indices.add(j)

        duplicates_removed = len(removed_indices)

        # Apply filtering
        cleaned_points = points[to_keep]
        cleaned_colors = colors[to_keep] if colors is not None else None
        cleaned_intensities = intensities[to_keep] if intensities is not None else None

        return cleaned_points, cleaned_colors, cleaned_intensities, duplicates_removed

    def _remove_outliers(self, points: np.ndarray,
                        colors: Optional[np.ndarray] = None,
                        intensities: Optional[np.ndarray] = None,
                        method: str = "statistical",
                        **params) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], int]:
        """
        Remove outlier points using specified method.

        Args:
            points: Point coordinates
            colors: Optional color array
            intensities: Optional intensity array
            method: Outlier removal method
            **params: Method-specific parameters

        Returns:
            Tuple of (cleaned_points, cleaned_colors, cleaned_intensities, outliers_removed)
        """
        if method == "statistical":
            return self._remove_outliers_statistical(points, colors, intensities, **params)
        elif method == "radius":
            return self._remove_outliers_radius(points, colors, intensities, **params)
        else:
            logger.warning(f"Unknown outlier removal method: {method}")
            return points, colors, intensities, 0

    def _remove_outliers_statistical(self, points: np.ndarray,
                                   colors: Optional[np.ndarray] = None,
                                   intensities: Optional[np.ndarray] = None,
                                   nb_neighbors: int = 20,
                                   std_ratio: float = 2.0) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], int]:
        """
        Remove statistical outliers based on distance to neighbors.

        Args:
            points: Point coordinates
            colors: Optional color array
            intensities: Optional intensity array
            nb_neighbors: Number of neighbors to consider
            std_ratio: Standard deviation ratio threshold

        Returns:
            Tuple of (cleaned_points, cleaned_colors, cleaned_intensities, outliers_removed)
        """
        if not self._scipy_available:
            logger.warning("SciPy not available, skipping statistical outlier removal")
            return points, colors, intensities, 0

        from scipy.spatial import cKDTree

        tree = cKDTree(points)
        distances, _ = tree.query(points, k=min(nb_neighbors + 1, len(points)))

        # Use mean distance to k nearest neighbors
        mean_distances = np.mean(distances[:, 1:], axis=1)  # Skip self (distance 0)

        # Statistical outlier detection
        mean_dist = np.mean(mean_distances)
        std_dist = np.std(mean_distances)

        threshold = mean_dist + std_ratio * std_dist
        inlier_mask = mean_distances <= threshold

        outliers_removed = np.sum(~inlier_mask)

        return (points[inlier_mask],
                colors[inlier_mask] if colors is not None else None,
                intensities[inlier_mask] if intensities is not None else None,
                outliers_removed)

    def _remove_outliers_radius(self, points: np.ndarray,
                               colors: Optional[np.ndarray] = None,
                               intensities: Optional[np.ndarray] = None,
                               nb_points: int = 16,
                               radius: float = 0.05) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], int]:
        """
        Remove outliers based on radius outlier removal.

        Args:
            points: Point coordinates
            colors: Optional color array
            intensities: Optional intensity array
            nb_points: Minimum neighbors within radius
            radius: Search radius

        Returns:
            Tuple of (cleaned_points, cleaned_colors, cleaned_intensities, outliers_removed)
        """
        if not self._scipy_available:
            logger.warning("SciPy not available, skipping radius outlier removal")
            return points, colors, intensities, 0

        from scipy.spatial import cKDTree

        tree = cKDTree(points)
        neighbors_within_radius = tree.query_ball_point(points, r=radius)

        # Count neighbors (excluding self)
        neighbor_counts = np.array([len(neighbors) - 1 for neighbors in neighbors_within_radius])

        inlier_mask = neighbor_counts >= nb_points
        outliers_removed = np.sum(~inlier_mask)

        return (points[inlier_mask],
                colors[inlier_mask] if colors is not None else None,
                intensities[inlier_mask] if intensities is not None else None,
                outliers_removed)

    def _estimate_normals(self, points: np.ndarray,
                         method: str = "auto",
                         **params) -> Tuple[Optional[np.ndarray], bool, Optional[str]]:
        """
        Estimate surface normals for point cloud.

        Args:
            points: Point coordinates
            method: Normal estimation method
            **params: Method-specific parameters

        Returns:
            Tuple of (normals, success, method_used)
        """
        if method == "auto":
            # Try Open3D first, fall back to PCA
            if self._open3d_available:
                method = "open3d"
            elif self._scipy_available:
                method = "pca"
            else:
                logger.warning("No normal estimation method available")
                return None, False, None

        if method == "open3d":
            return self._estimate_normals_open3d(points, **params)
        elif method == "pca":
            return self._estimate_normals_pca(points, **params)
        else:
            logger.warning(f"Unknown normal estimation method: {method}")
            return None, False, None

    def _estimate_normals_open3d(self, points: np.ndarray,
                                **params) -> Tuple[Optional[np.ndarray], bool, Optional[str]]:
        """
        Estimate normals using Open3D.

        Args:
            points: Point coordinates
            **params: Open3D parameters

        Returns:
            Tuple of (normals, success, method_used)
        """
        try:
            import open3d as o3d

            # Create Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)

            # Estimate normals
            search_param = o3d.geometry.KDTreeSearchParamHybrid(
                radius=params.get('radius', 0.1),
                max_nn=params.get('max_nn', 30)
            )

            pcd.estimate_normals(search_param=search_param)

            # Orient normals consistently
            pcd.orient_normals_consistent_tangent_plane(k=params.get('orientation_knn', 100))

            normals = np.asarray(pcd.normals, dtype=np.float32)

            return normals, True, "open3d"

        except Exception as e:
            logger.warning(f"Open3D normal estimation failed: {e}")
            return None, False, None

    def _estimate_normals_pca(self, points: np.ndarray,
                             k_neighbors: int = 15) -> Tuple[Optional[np.ndarray], bool, Optional[str]]:
        """
        Estimate normals using PCA on k-nearest neighbors.

        Args:
            points: Point coordinates
            k_neighbors: Number of neighbors for PCA

        Returns:
            Tuple of (normals, success, method_used)
        """
        if not self._scipy_available:
            return None, False, None

        try:
            from scipy.spatial import cKDTree

            tree = cKDTree(points)
            _, neighbor_indices = tree.query(points, k=min(k_neighbors + 1, len(points)))

            normals = np.zeros_like(points, dtype=np.float32)

            for i, neighbors in enumerate(neighbor_indices):
                # Get neighbor points (excluding self)
                neighbor_points = points[neighbors[1:]]  # Skip self

                if len(neighbor_points) < 3:
                    # Not enough neighbors for PCA
                    normals[i] = [0, 0, 1]  # Default normal
                    continue

                # Center points
                centered = neighbor_points - neighbor_points.mean(axis=0)

                # PCA via SVD
                _, _, vt = np.linalg.svd(centered)

                # Normal is the smallest eigenvector (last column of V^T)
                normal = vt[-1]

                # Ensure consistent orientation (pointing away from centroid)
                centroid = points[i]
                neighbor_centroid = neighbor_points.mean(axis=0)

                if np.dot(normal, centroid - neighbor_centroid) < 0:
                    normal = -normal

                normals[i] = normal

            return normals, True, "pca"

        except Exception as e:
            logger.warning(f"PCA normal estimation failed: {e}")
            return None, False, None


def clean_scan(scan_data: ScanData, scan_id: UUID,
              outlier_method: str = "statistical",
              normal_method: str = "auto",
              outlier_params: Optional[Dict[str, Any]] = None,
              normal_params: Optional[Dict[str, Any]] = None) -> Tuple[ScanData, CleaningReport]:
    """
    Convenience function to clean scan data.

    Args:
        scan_data: Input scan data
        scan_id: UUID of the scan
        outlier_method: Method for outlier removal
        normal_method: Method for normal estimation
        outlier_params: Parameters for outlier removal
        normal_params: Parameters for normal estimation

    Returns:
        Tuple of (cleaned_scan_data, cleaning_report)
    """
    cleaner = ScanCleaner()
    return cleaner.clean_scan(scan_data, scan_id, outlier_method, normal_method,
                            outlier_params, normal_params)
