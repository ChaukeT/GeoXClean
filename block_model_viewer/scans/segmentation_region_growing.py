"""
Region Growing Segmentation
===========================

Segments point clouds into fragments using region growing based on surface normals
and curvature. Suitable for identifying distinct objects in scan data.
"""

import logging
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from uuid import UUID
from datetime import datetime
from collections import deque

from .scan_models import ScanData, SegmentationResult, RegionGrowingParams

logger = logging.getLogger(__name__)


class RegionGrowingSegmenter:
    """
    Segments point clouds using region growing algorithm.

    Based on surface normal similarity and curvature constraints.
    """

    def __init__(self):
        """Initialize segmenter."""
        self._check_dependencies()

    def _check_dependencies(self):
        """Check required dependencies."""
        try:
            import scipy.spatial
            self._scipy_available = True
        except ImportError:
            self._scipy_available = False
            logger.warning("SciPy not available - region growing segmentation may be limited")

    def segment(self, scan_data: ScanData, scan_id: UUID,
               params: RegionGrowingParams,
               progress_callback: Optional[callable] = None) -> SegmentationResult:
        """
        Segment scan data using region growing.

        Args:
            scan_data: Cleaned scan data with normals
            scan_id: UUID of the scan
            params: Segmentation parameters
            progress_callback: Optional progress callback

        Returns:
            SegmentationResult
        """
        if scan_data.points is None:
            raise ValueError("Cannot segment scan with no point data")

        if not scan_data.has_normals or scan_data.normals is None:
            raise ValueError("Normals required for region growing segmentation. Run cleaning first.")

        # Validate parameters
        errors = params.validate()
        if errors:
            raise ValueError(f"Invalid parameters: {', '.join(errors)}")

        points = scan_data.points
        normals = scan_data.normals

        if progress_callback:
            progress_callback(0.0, "Computing curvature...")

        # Compute curvature for each point
        curvature = self._compute_curvature(points, normals, params.k_neighbors)

        if progress_callback:
            progress_callback(0.1, "Finding seed points...")

        # Find seed points (high curvature regions)
        seed_mask = self._find_seed_points(curvature, params.curvature_threshold)

        if progress_callback:
            progress_callback(0.2, "Building neighbor graph...")

        # Build k-nearest neighbor graph for region growing
        neighbor_graph = self._build_neighbor_graph(points, params.k_neighbors)

        if progress_callback:
            progress_callback(0.3, "Growing regions...")

        # Perform region growing
        fragment_labels = self._grow_regions(points, normals, neighbor_graph, seed_mask,
                                           params.normal_threshold_deg, params.min_region_size,
                                           progress_callback)

        if progress_callback:
            progress_callback(0.9, "Post-processing regions...")

        # Post-process regions
        fragment_labels = self._post_process_regions(fragment_labels, params)

        # Analyze results
        unique_labels = np.unique(fragment_labels)
        fragment_count = len(unique_labels) if -1 not in unique_labels else len(unique_labels) - 1
        noise_points = np.sum(fragment_labels == -1)

        # Quality assessment
        quality_score = self._assess_segmentation_quality(points, fragment_labels, fragment_count)

        # Check for segmentation failures
        warnings = []
        if fragment_count == 1 and len(points) > params.min_region_size:
            warnings.append("Single fragment detected - possible under-segmentation")
        elif fragment_count == 0:
            warnings.append("No fragments detected - check parameters and data quality")

        # Check for over-segmentation
        if fragment_count > len(points) * 0.1:  # More than 10% fragments
            warnings.append("High fragment count - possible over-segmentation")

        result = SegmentationResult(
            scan_id=scan_id,
            timestamp=datetime.now(),
            strategy="region_growing",
            parameters=params,
            fragment_labels=fragment_labels,
            fragment_count=fragment_count,
            noise_points=noise_points,
            fragmentation_quality_score=quality_score,
            warnings=warnings
        )

        if progress_callback:
            progress_callback(1.0, f"Segmentation complete: {fragment_count} fragments")

        logger.info(f"Region growing segmentation complete for scan {scan_id}: "
                   f"{fragment_count} fragments, {noise_points} noise points")

        return result

    def _compute_curvature(self, points: np.ndarray, normals: np.ndarray,
                          k_neighbors: int) -> np.ndarray:
        """
        Compute point curvature based on normal variation in neighborhood.

        Args:
            points: Point coordinates (N, 3)
            normals: Point normals (N, 3)
            k_neighbors: Number of neighbors to consider

        Returns:
            Curvature values (N,)
        """
        if not self._scipy_available:
            raise ImportError("SciPy required for curvature computation")

        from scipy.spatial import cKDTree

        tree = cKDTree(points)
        _, neighbor_indices = tree.query(points, k=min(k_neighbors + 1, len(points)))

        curvature = np.zeros(len(points))

        for i, neighbors in enumerate(neighbor_indices):
            # Get neighbor normals (excluding self)
            neighbor_normals = normals[neighbors[1:]]

            if len(neighbor_normals) == 0:
                curvature[i] = 0.0
                continue

            # Compute normal variation (1 - average dot product with reference normal)
            ref_normal = normals[i]
            dot_products = np.abs(np.dot(neighbor_normals, ref_normal))
            curvature[i] = 1.0 - np.mean(dot_products)

        return curvature

    def _find_seed_points(self, curvature: np.ndarray, threshold: float) -> np.ndarray:
        """
        Find seed points based on curvature threshold.

        Args:
            curvature: Curvature values
            threshold: Minimum curvature for seed selection

        Returns:
            Boolean mask of seed points
        """
        # Select points with curvature above threshold
        seed_mask = curvature >= threshold

        # Ensure we have at least some seeds
        if not np.any(seed_mask):
            # If no points meet threshold, select highest curvature points
            sorted_indices = np.argsort(curvature)[::-1]
            n_seeds = max(1, len(curvature) // 1000)  # At least 1 seed, or 0.1% of points
            seed_mask = np.zeros(len(curvature), dtype=bool)
            seed_mask[sorted_indices[:n_seeds]] = True

        return seed_mask

    def _build_neighbor_graph(self, points: np.ndarray, k_neighbors: int) -> List[List[int]]:
        """
        Build k-nearest neighbor graph.

        Args:
            points: Point coordinates
            k_neighbors: Number of neighbors per point

        Returns:
            List of neighbor indices for each point
        """
        if not self._scipy_available:
            raise ImportError("SciPy required for neighbor graph construction")

        from scipy.spatial import cKDTree

        tree = cKDTree(points)
        _, neighbor_indices = tree.query(points, k=min(k_neighbors + 1, len(points)))

        # Convert to list of lists (exclude self)
        neighbor_graph = []
        for neighbors in neighbor_indices:
            neighbor_graph.append(neighbors[1:].tolist())

        return neighbor_graph

    def _grow_regions(self, points: np.ndarray, normals: np.ndarray,
                     neighbor_graph: List[List[int]], seed_mask: np.ndarray,
                     normal_threshold_deg: float, min_region_size: int,
                     progress_callback: Optional[callable] = None) -> np.ndarray:
        """
        Perform region growing from seed points.

        Args:
            points: Point coordinates
            normals: Point normals
            neighbor_graph: Neighbor connectivity
            seed_mask: Boolean mask of seed points
            normal_threshold_deg: Maximum normal angle difference (degrees)
            min_region_size: Minimum points per region
            progress_callback: Optional progress callback

        Returns:
            Fragment labels (-1 for unassigned)
        """
        n_points = len(points)
        fragment_labels = np.full(n_points, -1, dtype=np.int32)

        # Convert angle threshold to radians
        normal_threshold_rad = np.deg2rad(normal_threshold_deg)

        # Find seed point indices
        seed_indices = np.where(seed_mask)[0]

        # Sort seeds by curvature (process high-curvature seeds first)
        # Assume curvature was computed earlier - for now, randomize order
        np.random.shuffle(seed_indices)

        current_label = 0

        for seed_idx in seed_indices:
            if fragment_labels[seed_idx] != -1:
                continue  # Already assigned

            # Start new region
            region_points = []
            region_queue = deque([seed_idx])

            while region_queue:
                current_idx = region_queue.popleft()

                if fragment_labels[current_idx] != -1:
                    continue

                # Check if point can be added to region
                can_add = self._can_add_to_region(current_idx, region_points,
                                                points, normals, normal_threshold_rad)

                if can_add:
                    fragment_labels[current_idx] = current_label
                    region_points.append(current_idx)

                    # Add neighbors to queue
                    for neighbor_idx in neighbor_graph[current_idx]:
                        if fragment_labels[neighbor_idx] == -1:
                            region_queue.append(neighbor_idx)

            # Check region size
            if len(region_points) < min_region_size:
                # Remove small region
                for idx in region_points:
                    fragment_labels[idx] = -1
            else:
                current_label += 1

            if progress_callback and current_label % 10 == 0:
                progress = 0.3 + 0.5 * (current_label / max(1, len(seed_indices)))
                progress_callback(progress, f"Growing region {current_label}...")

        return fragment_labels

    def _can_add_to_region(self, point_idx: int, region_points: List[int],
                          points: np.ndarray, normals: np.ndarray,
                          normal_threshold_rad: float) -> bool:
        """
        Check if a point can be added to a region based on normal similarity.

        Args:
            point_idx: Index of point to test
            region_points: Current region point indices
            points: Point coordinates
            normals: Point normals
            normal_threshold_rad: Normal angle threshold (radians)

        Returns:
            True if point can be added
        """
        if len(region_points) == 0:
            return True  # First point always allowed

        # Check normal similarity with region points
        point_normal = normals[point_idx]
        region_normals = normals[region_points]

        # Compute angle differences
        dot_products = np.abs(np.dot(region_normals, point_normal))
        angles = np.arccos(np.clip(dot_products, -1.0, 1.0))

        # Point can be added if similar to at least one region point
        min_angle = np.min(angles)

        return min_angle <= normal_threshold_rad

    def _post_process_regions(self, fragment_labels: np.ndarray,
                            params: RegionGrowingParams) -> np.ndarray:
        """
        Post-process regions: merge small regions, clean up noise.

        Args:
            fragment_labels: Current fragment labels
            params: Segmentation parameters

        Returns:
            Processed fragment labels
        """
        processed_labels = fragment_labels.copy()

        # Find small regions
        unique_labels, counts = np.unique(fragment_labels, return_counts=True)
        small_region_mask = counts < params.min_region_size

        # Remove small regions (set to noise)
        for label in unique_labels[small_region_mask]:
            if label != -1:  # Don't remove noise points
                processed_labels[processed_labels == label] = -1

        # Merge nearby small regions (simplified - could be enhanced)
        # For now, just remove them

        return processed_labels

    def _assess_segmentation_quality(self, points: np.ndarray, fragment_labels: np.ndarray,
                                   fragment_count: int) -> float:
        """
        Assess the quality of segmentation.

        Args:
            points: Point coordinates
            fragment_labels: Fragment labels
            fragment_count: Number of fragments

        Returns:
            Quality score (0-1, higher is better)
        """
        if fragment_count == 0:
            return 0.0

        # Simple quality metrics
        # 1. Fragment size distribution (prefer balanced sizes)
        unique_labels, counts = np.unique(fragment_labels, return_counts=True)
        if -1 in unique_labels:
            counts = counts[unique_labels != -1]  # Exclude noise

        if len(counts) == 0:
            return 0.0

        # Coefficient of variation of fragment sizes (lower is better)
        size_cv = np.std(counts) / np.mean(counts) if np.mean(counts) > 0 else 1.0

        # Convert to quality score (lower CV = higher quality)
        size_balance_score = max(0, 1.0 - size_cv)

        # 2. Spatial separation (fragments should be spatially coherent)
        # Simplified: check if fragments are reasonably separated
        separation_score = 0.8  # Placeholder - could compute actual spatial metrics

        # Combined score
        quality_score = (size_balance_score + separation_score) / 2.0

        return min(1.0, max(0.0, quality_score))


def segment_region_growing(scan_data: ScanData, scan_id: UUID,
                          params: RegionGrowingParams,
                          progress_callback: Optional[callable] = None) -> SegmentationResult:
    """
    Convenience function for region growing segmentation.

    Args:
        scan_data: Scan data to segment
        scan_id: UUID of the scan
        params: Segmentation parameters
        progress_callback: Optional progress callback

    Returns:
        SegmentationResult
    """
    segmenter = RegionGrowingSegmenter()
    return segmenter.segment(scan_data, scan_id, params, progress_callback)
