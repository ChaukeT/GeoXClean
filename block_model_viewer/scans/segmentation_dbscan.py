"""
DBSCAN Segmentation
===================

Segments point clouds using density-based clustering (DBSCAN/HDBSCAN).
Suitable for identifying fragments based on spatial proximity and density.
"""

import logging
import numpy as np
from typing import Optional, Dict, Any
from uuid import UUID
from datetime import datetime

from .scan_models import ScanData, SegmentationResult, DBSCANParams

logger = logging.getLogger(__name__)


class DBSCANSegmenter:
    """
    Segments point clouds using DBSCAN or HDBSCAN clustering.

    DBSCAN identifies dense regions of points separated by sparse areas.
    """

    def __init__(self):
        """Initialize segmenter."""
        self._check_dependencies()

    def _check_dependencies(self):
        """Check required dependencies."""
        try:
            from sklearn.cluster import DBSCAN
            self._sklearn_available = True
        except ImportError:
            self._sklearn_available = False
            logger.warning("scikit-learn not available - DBSCAN segmentation disabled")

        try:
            import hdbscan
            self._hdbscan_available = True
        except ImportError:
            self._hdbscan_available = False
            logger.warning("hdbscan not available - HDBSCAN option disabled")

    def segment(self, scan_data: ScanData, scan_id: UUID,
               params: DBSCANParams,
               progress_callback: Optional[callable] = None) -> SegmentationResult:
        """
        Segment scan data using DBSCAN/HDBSCAN.

        Args:
            scan_data: Scan data to segment
            scan_id: UUID of the scan
            params: Segmentation parameters
            progress_callback: Optional progress callback

        Returns:
            SegmentationResult
        """
        if scan_data.points is None:
            raise ValueError("Cannot segment scan with no point data")

        # Validate parameters
        errors = params.validate()
        if errors:
            raise ValueError(f"Invalid parameters: {', '.join(errors)}")

        points = scan_data.points

        if progress_callback:
            progress_callback(0.0, "Preparing data for clustering...")

        # Check memory requirements (rough estimate)
        n_points = len(points)
        estimated_memory_mb = (n_points * 3 * 8) / (1024 * 1024)  # 3D points, 8 bytes per float

        if estimated_memory_mb > 1000:  # > 1GB
            logger.warning(f"Large dataset detected: ~{estimated_memory_mb:.1f} MB. "
                          "Consider using octree downsampling.")

        if progress_callback:
            progress_callback(0.1, f"Running {'HDBSCAN' if params.use_hdbscan else 'DBSCAN'}...")

        # Perform clustering
        try:
            if params.use_hdbscan:
                fragment_labels = self._cluster_hdbscan(points, params)
            else:
                fragment_labels = self._cluster_dbscan(points, params)
        except MemoryError:
            raise MemoryError("Dataset too large for clustering. Enable octree downsampling or reduce point count.")
        except Exception as e:
            raise RuntimeError(f"Clustering failed: {str(e)}")

        if progress_callback:
            progress_callback(0.8, "Post-processing clusters...")

        # Post-process results
        fragment_labels = self._post_process_clusters(fragment_labels, params)

        # Analyze results
        unique_labels, counts = np.unique(fragment_labels, return_counts=True)

        # Filter out noise points (-1)
        if -1 in unique_labels:
            valid_labels = unique_labels[unique_labels != -1]
            valid_counts = counts[unique_labels != -1]
            noise_points = counts[unique_labels == -1][0]
        else:
            valid_labels = unique_labels
            valid_counts = counts
            noise_points = 0

        fragment_count = len(valid_labels)

        # Quality assessment
        quality_score = self._assess_segmentation_quality(points, fragment_labels, fragment_count, valid_counts)

        # Check for segmentation failures
        warnings = []

        if fragment_count == 0:
            warnings.append("No fragments detected - check epsilon and min_points parameters")
        elif fragment_count == 1:
            if n_points > params.min_points * 2:  # More than 2x min cluster size
                warnings.append("Single fragment detected - possible under-segmentation (epsilon too large)")

        # Check for over-segmentation (too many small clusters)
        if fragment_count > 0:
            small_clusters = np.sum(valid_counts < params.min_points * 2)
            if small_clusters > fragment_count * 0.5:  # More than 50% small clusters
                warnings.append(f"High number of small clusters ({small_clusters}/{fragment_count}) - possible over-segmentation")

        result = SegmentationResult(
            scan_id=scan_id,
            timestamp=datetime.now(),
            strategy="dbscan",
            parameters=params,
            fragment_labels=fragment_labels,
            fragment_count=fragment_count,
            noise_points=noise_points,
            fragmentation_quality_score=quality_score,
            warnings=warnings
        )

        if progress_callback:
            progress_callback(1.0, f"DBSCAN segmentation complete: {fragment_count} fragments, {noise_points} noise points")

        logger.info(f"DBSCAN segmentation complete for scan {scan_id}: "
                   f"{fragment_count} fragments, {noise_points} noise points")

        return result

    def _cluster_dbscan(self, points: np.ndarray, params: DBSCANParams) -> np.ndarray:
        """
        Perform DBSCAN clustering.

        Args:
            points: Point coordinates
            params: DBSCAN parameters

        Returns:
            Cluster labels (-1 for noise)
        """
        if not self._sklearn_available:
            raise ImportError("scikit-learn required for DBSCAN clustering")

        from sklearn.cluster import DBSCAN

        # Configure DBSCAN
        dbscan_params = {
            'eps': params.epsilon,
            'min_samples': params.min_points,
            'metric': 'euclidean',
            'algorithm': 'auto',
            'n_jobs': -1  # Use all available cores
        }

        # Run DBSCAN
        dbscan = DBSCAN(**dbscan_params)
        labels = dbscan.fit_predict(points)

        return labels.astype(np.int32)

    def _cluster_hdbscan(self, points: np.ndarray, params: DBSCANParams) -> np.ndarray:
        """
        Perform HDBSCAN clustering.

        Args:
            points: Point coordinates
            params: DBSCAN parameters (adapted for HDBSCAN)

        Returns:
            Cluster labels (-1 for noise)
        """
        if not self._hdbscan_available:
            raise ImportError("hdbscan required for HDBSCAN clustering")

        import hdbscan

        # Configure HDBSCAN (adapt DBSCAN params)
        hdbscan_params = {
            'min_cluster_size': max(params.min_cluster_size, params.min_points),
            'min_samples': params.min_points,
            'cluster_selection_epsilon': params.epsilon,
            'metric': 'euclidean',
            'core_dist_n_jobs': -1
        }

        # Run HDBSCAN
        clusterer = hdbscan.HDBSCAN(**hdbscan_params)
        labels = clusterer.fit_predict(points)

        return labels.astype(np.int32)

    def _post_process_clusters(self, labels: np.ndarray, params: DBSCANParams) -> np.ndarray:
        """
        Post-process cluster labels.

        Args:
            labels: Raw cluster labels
            params: Segmentation parameters

        Returns:
            Processed cluster labels
        """
        processed_labels = labels.copy()

        # Handle noise points if requested
        if not params.filter_noise:
            # Keep noise points as-is (-1)
            pass
        else:
            # Noise points remain as -1
            pass

        # Filter small clusters
        unique_labels, counts = np.unique(processed_labels, return_counts=True)

        for label, count in zip(unique_labels, counts):
            if label != -1 and count < params.min_points:
                # Remove small clusters (mark as noise)
                processed_labels[processed_labels == label] = -1

        # Re-label clusters to be consecutive (optional)
        # This ensures cluster IDs are 0, 1, 2, ... instead of potentially sparse
        unique_labels = np.unique(processed_labels)
        if -1 in unique_labels:
            valid_labels = unique_labels[unique_labels != -1]
        else:
            valid_labels = unique_labels

        if len(valid_labels) > 0:
            # Create mapping to consecutive labels
            label_mapping = {-1: -1}
            for new_label, old_label in enumerate(valid_labels):
                label_mapping[old_label] = new_label

            processed_labels = np.array([label_mapping[label] for label in processed_labels])

        return processed_labels

    def _assess_segmentation_quality(self, points: np.ndarray, labels: np.ndarray,
                                   fragment_count: int, valid_counts: Optional[np.ndarray] = None) -> float:
        """
        Assess the quality of DBSCAN segmentation.

        Args:
            points: Point coordinates
            labels: Cluster labels
            fragment_count: Number of valid clusters
            valid_counts: Counts for valid clusters

        Returns:
            Quality score (0-1, higher is better)
        """
        if fragment_count == 0:
            return 0.0

        if valid_counts is None:
            unique_labels, counts = np.unique(labels, return_counts=True)
            valid_counts = counts[unique_labels != -1]

        # Quality metrics for DBSCAN

        # 1. Cluster size balance (prefer balanced cluster sizes)
        if len(valid_counts) > 1:
            size_cv = np.std(valid_counts) / np.mean(valid_counts)
            balance_score = max(0, 1.0 - size_cv)
        else:
            balance_score = 0.5  # Neutral for single cluster

        # 2. Noise ratio (lower noise is better)
        noise_ratio = np.sum(labels == -1) / len(labels)
        noise_score = 1.0 - noise_ratio

        # 3. Spatial coherence (simplified - could compute silhouette score)
        coherence_score = 0.8  # Placeholder

        # Combined score
        quality_score = (balance_score + noise_score + coherence_score) / 3.0

        return min(1.0, max(0.0, quality_score))


def segment_dbscan(scan_data: ScanData, scan_id: UUID,
                  params: DBSCANParams,
                  progress_callback: Optional[callable] = None) -> SegmentationResult:
    """
    Convenience function for DBSCAN segmentation.

    Args:
        scan_data: Scan data to segment
        scan_id: UUID of the scan
        params: Segmentation parameters
        progress_callback: Optional progress callback

    Returns:
        SegmentationResult
    """
    segmenter = DBSCANSegmenter()
    return segmenter.segment(scan_data, scan_id, params, progress_callback)
