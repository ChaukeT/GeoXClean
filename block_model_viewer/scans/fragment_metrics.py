"""
Fragment Metrics Computation
============================

Computes geometric and statistical metrics for identified fragments.
Includes volume, shape factors, PSD computation, and quality assessment.
"""

import logging
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from uuid import UUID
from datetime import datetime
from scipy.spatial import ConvexHull

from .scan_models import ScanData, FragmentMetrics, PSDResults

logger = logging.getLogger(__name__)


class FragmentMetricsComputer:
    """
    Computes geometric and statistical metrics for fragments.

    Handles volume computation, shape analysis, and PSD generation.
    """

    def __init__(self):
        """Initialize metrics computer."""
        self._check_dependencies()

    def _check_dependencies(self):
        """Check required dependencies."""
        try:
            import scipy.spatial
            self._scipy_available = True
        except ImportError:
            self._scipy_available = False

        try:
            import trimesh
            self._trimesh_available = True
        except ImportError:
            self._trimesh_available = False

    def compute_fragment_metrics(self, scan_data: ScanData, fragment_labels: np.ndarray,
                               progress_callback: Optional[callable] = None) -> List[FragmentMetrics]:
        """
        Compute metrics for all fragments in segmented data.

        Args:
            scan_data: Original scan data
            fragment_labels: Fragment labels from segmentation (-1 for noise)
            progress_callback: Optional progress callback

        Returns:
            List of FragmentMetrics for each fragment
        """
        if scan_data.points is None:
            raise ValueError("Cannot compute metrics without point data")

        points = scan_data.points

        # Get unique fragment labels (exclude noise)
        unique_labels = np.unique(fragment_labels)
        fragment_ids = unique_labels[unique_labels != -1]

        if len(fragment_ids) == 0:
            logger.warning("No fragments found for metrics computation")
            return []

        fragment_metrics = []

        for i, fragment_id in enumerate(fragment_ids):
            if progress_callback:
                progress = (i + 1) / len(fragment_ids)
                progress_callback(progress * 0.9, f"Computing metrics for fragment {fragment_id}...")

            # Get points for this fragment
            fragment_mask = fragment_labels == fragment_id
            fragment_points = points[fragment_mask]

            try:
                metrics = self._compute_single_fragment_metrics(fragment_points, fragment_id)
                fragment_metrics.append(metrics)
            except Exception as e:
                logger.warning(f"Failed to compute metrics for fragment {fragment_id}: {e}")
                # Create minimal metrics record
                centroid = fragment_points.mean(axis=0)
                metrics = FragmentMetrics(
                    fragment_id=fragment_id,
                    point_count=len(fragment_points),
                    volume_m3=0.0,  # Invalid
                    equivalent_diameter_m=0.0,  # Invalid
                    sphericity=0.0,  # Invalid
                    elongation=1.0,  # Neutral
                    aspect_ratio=(1.0, 1.0, 1.0),  # Neutral
                    confidence_score=0.0,  # Invalid
                    centroid=(centroid[0], centroid[1], centroid[2])
                )
                fragment_metrics.append(metrics)

        if progress_callback:
            progress_callback(1.0, f"Computed metrics for {len(fragment_metrics)} fragments")

        return fragment_metrics

    def _compute_single_fragment_metrics(self, points: np.ndarray, fragment_id: int) -> FragmentMetrics:
        """
        Compute metrics for a single fragment.

        Args:
            points: Points belonging to this fragment (N, 3)
            fragment_id: Fragment identifier

        Returns:
            FragmentMetrics for this fragment
        """
        n_points = len(points)

        # Centroid
        centroid = points.mean(axis=0)

        # Volume computation
        volume_m3 = self._compute_volume(points)

        # Equivalent diameter (diameter of sphere with same volume)
        if volume_m3 > 0:
            equivalent_diameter_m = (6 * volume_m3 / np.pi) ** (1/3)
        else:
            equivalent_diameter_m = 0.0

        # Shape factors
        sphericity, elongation, aspect_ratio = self._compute_shape_factors(points)

        # Confidence score
        confidence_score = self._compute_confidence_score(points, volume_m3)

        # Surface area (optional)
        surface_area_m2 = self._compute_surface_area(points)

        # Bounding box volume
        bounds = points.min(axis=0), points.max(axis=0)
        bounding_box_volume = np.prod(bounds[1] - bounds[0])

        return FragmentMetrics(
            fragment_id=fragment_id,
            point_count=n_points,
            volume_m3=volume_m3,
            equivalent_diameter_m=equivalent_diameter_m,
            sphericity=sphericity,
            elongation=elongation,
            aspect_ratio=aspect_ratio,
            confidence_score=confidence_score,
            centroid=(centroid[0], centroid[1], centroid[2]),
            surface_area_m2=surface_area_m2,
            bounding_box_volume_m3=bounding_box_volume
        )

    def _compute_volume(self, points: np.ndarray) -> float:
        """
        Compute fragment volume using convex hull.

        Args:
            points: Fragment points

        Returns:
            Volume in cubic meters
        """
        if len(points) < 4:
            # Not enough points for volume computation
            return 0.0

        try:
            # Compute convex hull
            hull = ConvexHull(points)

            # Volume of convex hull
            volume = hull.volume

            # Validate volume (should be positive)
            if volume <= 0:
                return 0.0

            return volume

        except Exception as e:
            logger.warning(f"Convex hull volume computation failed: {e}")
            return 0.0

    def _compute_surface_area(self, points: np.ndarray) -> Optional[float]:
        """
        Compute approximate surface area.

        Args:
            points: Fragment points

        Returns:
            Surface area or None if computation fails
        """
        if len(points) < 4:
            return None

        try:
            # Use convex hull surface area as approximation
            hull = ConvexHull(points)
            return hull.area
        except Exception:
            return None

    def _compute_shape_factors(self, points: np.ndarray) -> Tuple[float, float, Tuple[float, float, float]]:
        """
        Compute shape factors: sphericity, elongation, aspect ratio.

        Args:
            points: Fragment points

        Returns:
            Tuple of (sphericity, elongation, aspect_ratio)
        """
        if len(points) < 3:
            return 0.0, 1.0, (1.0, 1.0, 1.0)

        try:
            # Compute covariance matrix
            centered = points - points.mean(axis=0)
            cov_matrix = np.cov(centered.T)

            # Eigenvalue decomposition
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

            # Sort eigenvalues (largest first)
            sorted_indices = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[sorted_indices]

            # Ensure positive eigenvalues
            eigenvalues = np.maximum(eigenvalues, 1e-10)

            # Dimensions along principal axes
            dims = np.sqrt(eigenvalues * 12)  # Approximation for extent

            # Sort dimensions (largest first)
            dims = np.sort(dims)[::-1]
            a, b, c = dims  # Length, width, height

            # Sphericity: ratio of surface area of sphere with same volume to actual surface area
            volume = (4/3) * np.pi * (a/2) * (b/2) * (c/2)  # Ellipsoid volume approximation

            if volume > 0:
                # Surface area of equivalent sphere
                equiv_radius = (3 * volume / (4 * np.pi)) ** (1/3)
                sphere_surface_area = 4 * np.pi * equiv_radius**2

                # Actual surface area (approximated)
                actual_surface_area = self._compute_surface_area(points)
                if actual_surface_area and actual_surface_area > 0:
                    sphericity = sphere_surface_area / actual_surface_area
                    sphericity = np.clip(sphericity, 0.0, 1.0)
                else:
                    # Fallback: based on aspect ratios
                    sphericity = min(a, b, c) / max(a, b, c)
            else:
                sphericity = 0.0

            # Elongation: ratio of longest to shortest dimension
            if min(a, b, c) > 0:
                elongation = max(a, b, c) / min(a, b, c)
            else:
                elongation = 1.0

            # Aspect ratio tuple
            aspect_ratio = (float(a), float(b), float(c))

            return sphericity, elongation, aspect_ratio

        except Exception as e:
            logger.warning(f"Shape factor computation failed: {e}")
            return 0.0, 1.0, (1.0, 1.0, 1.0)

    def _compute_confidence_score(self, points: np.ndarray, volume: float) -> float:
        """
        Compute confidence score based on data quality.

        Args:
            points: Fragment points
            volume: Computed volume

        Returns:
            Confidence score (0-1)
        """
        scores = []

        # 1. Point count score (more points = higher confidence)
        n_points = len(points)
        if n_points >= 1000:
            point_count_score = 1.0
        elif n_points >= 100:
            point_count_score = 0.7
        elif n_points >= 10:
            point_count_score = 0.3
        else:
            point_count_score = 0.1
        scores.append(point_count_score)

        # 2. Volume validity score
        volume_score = 1.0 if volume > 0 else 0.0
        scores.append(volume_score)

        # 3. Spatial distribution score (avoid collinear points)
        if len(points) >= 3:
            try:
                # Check if points span 3D space
                centered = points - points.mean(axis=0)
                cov_matrix = np.cov(centered.T)
                eigenvalues = np.linalg.eigvals(cov_matrix)

                # Score based on how well points fill 3D space
                condition_number = np.max(eigenvalues) / np.min(eigenvalues) if np.min(eigenvalues) > 0 else 1000
                distribution_score = min(1.0, 100.0 / condition_number)
                scores.append(distribution_score)
            except:
                scores.append(0.5)
        else:
            scores.append(0.5)

        # 4. Noise score (points should be reasonably close together)
        if len(points) >= 2:
            try:
                # Compute spread
                distances = np.linalg.norm(points - points.mean(axis=0), axis=1)
                spread_score = 1.0 / (1.0 + np.std(distances))  # Lower spread = higher score
                scores.append(spread_score)
            except:
                scores.append(0.5)
        else:
            scores.append(0.5)

        # Combine scores (geometric mean)
        if scores:
            confidence = np.prod(scores) ** (1.0 / len(scores))
        else:
            confidence = 0.0

        return min(1.0, max(0.0, confidence))

    def compute_psd(self, fragment_metrics: List[FragmentMetrics], scan_id: UUID,
                   volume_weighted: bool = False) -> PSDResults:
        """
        Compute particle size distribution from fragment metrics.

        Args:
            fragment_metrics: List of fragment metrics
            scan_id: UUID of the parent scan
            volume_weighted: Whether to compute volume-weighted PSD

        Returns:
            PSDResults with percentiles and statistics
        """
        if not fragment_metrics:
            # Return empty results
            return PSDResults(
                scan_id=scan_id,
                timestamp=datetime.now(),
                fragments=[],
                p10_m=0.0, p50_m=0.0, p80_m=0.0,
                fragment_count=0,
                mean_diameter_m=0.0,
                std_diameter_m=0.0
            )

        # Extract equivalent diameters
        diameters = np.array([fm.equivalent_diameter_m for fm in fragment_metrics])

        # Filter out invalid diameters
        valid_mask = diameters > 0
        valid_diameters = diameters[valid_mask]
        valid_metrics = [fm for fm, valid in zip(fragment_metrics, valid_mask) if valid]

        if len(valid_diameters) == 0:
            logger.warning("No valid fragment diameters for PSD computation")
            return PSDResults(
                scan_id=scan_id,
                timestamp=datetime.now(),
                fragments=fragment_metrics,
                p10_m=0.0, p50_m=0.0, p80_m=0.0,
                fragment_count=0,
                mean_diameter_m=0.0,
                std_diameter_m=0.0
            )

        # Compute percentiles explicitly (no interpolation)
        sorted_diams = np.sort(valid_diameters)

        def compute_percentile(p: float) -> float:
            """Compute percentile explicitly."""
            n = len(sorted_diams)
            index = (n - 1) * (p / 100.0)

            if index.is_integer():
                return sorted_diams[int(index)]
            else:
                # Linear interpolation between adjacent values
                lower_idx = int(np.floor(index))
                upper_idx = int(np.ceil(index))
                weight = index - lower_idx

                if upper_idx < n:
                    return sorted_diams[lower_idx] * (1 - weight) + sorted_diams[upper_idx] * weight
                else:
                    return sorted_diams[lower_idx]

        # Compute P10, P50, P80
        p10_m = compute_percentile(10.0)
        p50_m = compute_percentile(50.0)
        p80_m = compute_percentile(80.0)

        # Volume-weighted percentiles (optional)
        p10_volume_m = None
        p50_volume_m = None
        p80_volume_m = None

        if volume_weighted:
            volumes = np.array([fm.volume_m3 for fm in valid_metrics])
            total_volume = np.sum(volumes)

            if total_volume > 0:
                # Sort by diameter but weight by volume
                sort_indices = np.argsort(valid_diameters)
                sorted_volumes = volumes[sort_indices]
                cumulative_volume = np.cumsum(sorted_volumes)

                def compute_volume_percentile(target_percent: float) -> float:
                    target_volume = total_volume * (target_percent / 100.0)
                    idx = np.searchsorted(cumulative_volume, target_volume)

                    if idx == 0:
                        return valid_diameters[sort_indices[0]]
                    elif idx >= len(valid_diameters):
                        return valid_diameters[sort_indices[-1]]
                    else:
                        # Interpolate
                        v1 = cumulative_volume[idx-1]
                        v2 = cumulative_volume[idx]
                        d1 = valid_diameters[sort_indices[idx-1]]
                        d2 = valid_diameters[sort_indices[idx]]

                        weight = (target_volume - v1) / (v2 - v1)
                        return d1 + weight * (d2 - d1)

                p10_volume_m = compute_volume_percentile(10.0)
                p50_volume_m = compute_volume_percentile(50.0)
                p80_volume_m = compute_volume_percentile(80.0)

        # Compute histogram for distribution
        hist_bins = np.logspace(np.log10(max(1e-6, np.min(valid_diameters))),
                               np.log10(np.max(valid_diameters)), 50)
        histogram, bin_edges = np.histogram(valid_diameters, bins=hist_bins)

        # Summary statistics
        total_volume_m3 = sum(fm.volume_m3 for fm in valid_metrics)
        mean_diameter_m = np.mean(valid_diameters)
        std_diameter_m = np.std(valid_diameters)

        return PSDResults(
            scan_id=scan_id,
            timestamp=datetime.now(),
            fragments=fragment_metrics,  # Include all fragments, not just valid ones
            p10_m=p10_m,
            p50_m=p50_m,
            p80_m=p80_m,
            p10_volume_m=p10_volume_m,
            p50_volume_m=p50_volume_m,
            p80_volume_m=p80_volume_m,
            distribution_histogram=histogram,
            bin_edges=bin_edges,
            total_volume_m3=total_volume_m3,
            fragment_count=len(valid_metrics),
            mean_diameter_m=mean_diameter_m,
            std_diameter_m=std_diameter_m
        )


def compute_fragment_metrics(scan_data: ScanData, fragment_labels: np.ndarray,
                           progress_callback: Optional[callable] = None) -> List[FragmentMetrics]:
    """
    Convenience function to compute fragment metrics.

    Args:
        scan_data: Original scan data
        fragment_labels: Fragment labels from segmentation
        progress_callback: Optional progress callback

    Returns:
        List of FragmentMetrics
    """
    computer = FragmentMetricsComputer()
    return computer.compute_fragment_metrics(scan_data, fragment_labels, progress_callback)


def compute_psd(fragment_metrics: List[FragmentMetrics], scan_id: UUID,
               volume_weighted: bool = False) -> PSDResults:
    """
    Convenience function to compute PSD.

    Args:
        fragment_metrics: Fragment metrics
        scan_id: Parent scan ID
        volume_weighted: Whether to compute volume-weighted PSD

    Returns:
        PSDResults
    """
    computer = FragmentMetricsComputer()
    return computer.compute_psd(fragment_metrics, scan_id, volume_weighted)
