"""
Scan Data Models
================

Dataclasses for scan analysis pipeline data structures.
All models are immutable and serializable for auditability.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import UUID


class ScanProcessingMode(Enum):
    """Processing mode for scan analysis."""
    PREVIEW = "preview"  # Fast preview, downsampled data
    ANALYSIS = "analysis"  # Full analysis, all points


@dataclass
class ScanData:
    """
    Unified scan data container.

    Supports both point clouds and meshes with optional attributes.
    """
    points: Optional[np.ndarray] = None  # (N, 3) point coordinates
    faces: Optional[np.ndarray] = None  # (M, 3) or (M, 4) face indices
    normals: Optional[np.ndarray] = None  # (N, 3) per-point normals
    colors: Optional[np.ndarray] = None  # (N, 3) RGB colors
    intensities: Optional[np.ndarray] = None  # (N,) intensity values
    classifications: Optional[np.ndarray] = None  # (N,) classification codes

    # Metadata
    crs: Optional[str] = None  # CRS code (e.g., "EPSG:32633")
    units: str = "meters"  # "meters" or "feet"
    file_format: str = "UNKNOWN"  # "LAS", "PLY", "OBJ", etc.

    # Processing state
    is_cleaned: bool = False
    has_normals: bool = False

    def is_point_cloud(self) -> bool:
        """Check if this is a point cloud (no faces)."""
        return self.points is not None and self.faces is None

    def is_mesh(self) -> bool:
        """Check if this is a mesh (has faces)."""
        return self.points is not None and self.faces is not None

    def point_count(self) -> int:
        """Get number of points/vertices."""
        return len(self.points) if self.points is not None else 0

    def face_count(self) -> int:
        """Get number of faces (meshes only)."""
        return len(self.faces) if self.faces is not None else 0

    def bounds(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Get axis-aligned bounding box (min, max)."""
        if self.points is None:
            return None
        return self.points.min(axis=0), self.points.max(axis=0)

    def centroid(self) -> Optional[np.ndarray]:
        """Get centroid of the scan."""
        if self.points is None:
            return None
        return self.points.mean(axis=0)


@dataclass
class ValidationViolation:
    """Single validation violation."""
    violation_type: str  # "error" or "warning"
    field: str  # Which field/aspect failed
    message: str  # Human-readable message
    details: Optional[Dict[str, Any]] = None  # Additional technical details

    def is_error(self) -> bool:
        return self.violation_type == "error"

    def is_warning(self) -> bool:
        return self.violation_type == "warning"


@dataclass
class ValidationReport:
    """Complete validation report for a scan."""
    scan_id: UUID
    timestamp: datetime
    is_valid: bool
    violations: List[ValidationViolation]

    # Summary statistics
    total_points: int
    coordinate_range: Optional[Tuple[np.ndarray, np.ndarray]] = None
    density_estimate: Optional[float] = None  # points per cubic unit
    outlier_count: Optional[int] = None

    def error_count(self) -> int:
        """Count of error-level violations."""
        return sum(1 for v in self.violations if v.is_error())

    def warning_count(self) -> int:
        """Count of warning-level violations."""
        return sum(1 for v in self.violations if v.is_warning())

    def has_errors(self) -> bool:
        """Check if report has any errors."""
        return self.error_count() > 0

    def has_warnings(self) -> bool:
        """Check if report has any warnings."""
        return self.warning_count() > 0


@dataclass
class CleaningReport:
    """Report from cleaning/normal estimation stage."""
    scan_id: UUID
    timestamp: datetime
    input_point_count: int
    output_point_count: int

    # Cleaning statistics
    outliers_removed: int = 0
    duplicates_removed: int = 0
    normals_computed: bool = False
    normal_method: Optional[str] = None  # "pca", "jet", etc.

    # Quality metrics
    average_density: Optional[float] = None
    noise_estimate: Optional[float] = None

    def removal_rate(self) -> float:
        """Fraction of points removed during cleaning."""
        if self.input_point_count == 0:
            return 0.0
        return self.outliers_removed / self.input_point_count


@dataclass
class SegmentationParams:
    """Base class for segmentation parameters."""
    strategy: str  # "region_growing" or "dbscan"

    def validate(self) -> List[str]:
        """Validate parameters, return list of error messages."""
        return []


@dataclass
class RegionGrowingParams(SegmentationParams):
    """Parameters for region growing segmentation."""
    strategy: str = "region_growing"
    normal_threshold_deg: float = 30.0  # Maximum angle difference
    curvature_threshold: float = 0.01  # Minimum curvature for seeds
    k_neighbors: int = 15  # For normal estimation
    min_region_size: int = 100  # Minimum points per fragment
    max_region_size: int = 1_000_000  # Maximum points per fragment
    merge_distance: float = 0.1  # Merge regions within distance (meters)

    def validate(self) -> List[str]:
        errors = super().validate()
        if self.normal_threshold_deg <= 0 or self.normal_threshold_deg >= 180:
            errors.append("normal_threshold_deg must be between 0 and 180")
        if self.curvature_threshold < 0:
            errors.append("curvature_threshold must be >= 0")
        if self.k_neighbors < 3:
            errors.append("k_neighbors must be >= 3")
        if self.min_region_size < 10:
            errors.append("min_region_size must be >= 10")
        return errors


@dataclass
class DBSCANParams(SegmentationParams):
    """Parameters for DBSCAN-based segmentation."""
    strategy: str = "dbscan"
    epsilon: float = 0.05  # Distance threshold (meters)
    min_points: int = 20  # Minimum points per cluster
    use_hdbscan: bool = False  # Use HDBSCAN for variable density
    min_cluster_size: int = 50  # For HDBSCAN
    filter_noise: bool = True  # Remove noise points from fragments

    def validate(self) -> List[str]:
        errors = super().validate()
        if self.epsilon <= 0:
            errors.append("epsilon must be > 0")
        if self.min_points < 1:
            errors.append("min_points must be >= 1")
        if self.use_hdbscan and self.min_cluster_size < 2:
            errors.append("min_cluster_size must be >= 2 when using HDBSCAN")
        return errors


@dataclass
class FragmentMetrics:
    """Metrics for a single fragment."""
    fragment_id: int
    point_count: int
    volume_m3: float
    equivalent_diameter_m: float
    sphericity: float  # 0-1, perfect sphere = 1
    elongation: float  # >= 1, length/width ratio
    aspect_ratio: Tuple[float, float, float]  # (L, W, H) ratios
    confidence_score: float  # 0-1, based on data quality
    centroid: Tuple[float, float, float]  # (x, y, z)

    # Optional mesh-based metrics
    surface_area_m2: Optional[float] = None
    bounding_box_volume_m3: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "fragment_id": self.fragment_id,
            "point_count": self.point_count,
            "volume_m3": self.volume_m3,
            "equivalent_diameter_m": self.equivalent_diameter_m,
            "sphericity": self.sphericity,
            "elongation": self.elongation,
            "aspect_ratio": list(self.aspect_ratio),
            "confidence_score": self.confidence_score,
            "centroid": list(self.centroid),
            "surface_area_m2": self.surface_area_m2,
            "bounding_box_volume_m3": self.bounding_box_volume_m3
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> FragmentMetrics:
        """Create from dictionary."""
        return cls(
            fragment_id=data["fragment_id"],
            point_count=data["point_count"],
            volume_m3=data["volume_m3"],
            equivalent_diameter_m=data["equivalent_diameter_m"],
            sphericity=data["sphericity"],
            elongation=data["elongation"],
            aspect_ratio=tuple(data["aspect_ratio"]),
            confidence_score=data["confidence_score"],
            centroid=tuple(data["centroid"]),
            surface_area_m2=data.get("surface_area_m2"),
            bounding_box_volume_m3=data.get("bounding_box_volume_m3")
        )


@dataclass
class PSDResults:
    """Complete particle size distribution results."""
    scan_id: UUID
    timestamp: datetime
    fragments: List[FragmentMetrics]

    # Percentiles (explicitly computed, no interpolation)
    p10_m: float  # 10th percentile equivalent diameter
    p50_m: float  # 50th percentile (median)
    p80_m: float  # 80th percentile

    # Volume-weighted percentiles (optional)
    p10_volume_m: Optional[float] = None
    p50_volume_m: Optional[float] = None
    p80_volume_m: Optional[float] = None

    # Distribution data
    distribution_histogram: np.ndarray = field(default_factory=lambda: np.array([]))
    bin_edges: np.ndarray = field(default_factory=lambda: np.array([]))

    # Summary statistics
    total_volume_m3: float = 0.0
    fragment_count: int = 0
    mean_diameter_m: float = 0.0
    std_diameter_m: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "scan_id": str(self.scan_id),
            "timestamp": self.timestamp.isoformat(),
            "fragments": [f.to_dict() for f in self.fragments],
            "p10_m": self.p10_m,
            "p50_m": self.p50_m,
            "p80_m": self.p80_m,
            "p10_volume_m": self.p10_volume_m,
            "p50_volume_m": self.p50_volume_m,
            "p80_volume_m": self.p80_volume_m,
            "distribution_histogram": self.distribution_histogram.tolist() if len(self.distribution_histogram) > 0 else [],
            "bin_edges": self.bin_edges.tolist() if len(self.bin_edges) > 0 else [],
            "total_volume_m3": self.total_volume_m3,
            "fragment_count": self.fragment_count,
            "mean_diameter_m": self.mean_diameter_m,
            "std_diameter_m": self.std_diameter_m
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PSDResults:
        """Create from dictionary."""
        return cls(
            scan_id=UUID(data["scan_id"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            fragments=[FragmentMetrics.from_dict(f) for f in data["fragments"]],
            p10_m=data["p10_m"],
            p50_m=data["p50_m"],
            p80_m=data["p80_m"],
            p10_volume_m=data.get("p10_volume_m"),
            p50_volume_m=data.get("p50_volume_m"),
            p80_volume_m=data.get("p80_volume_m"),
            distribution_histogram=np.array(data.get("distribution_histogram", [])),
            bin_edges=np.array(data.get("bin_edges", [])),
            total_volume_m3=data.get("total_volume_m3", 0.0),
            fragment_count=data.get("fragment_count", 0),
            mean_diameter_m=data.get("mean_diameter_m", 0.0),
            std_diameter_m=data.get("std_diameter_m", 0.0)
        )


@dataclass
class SegmentationResult:
    """Result from segmentation operation."""
    scan_id: UUID
    timestamp: datetime
    strategy: str  # "region_growing" or "dbscan"
    parameters: SegmentationParams

    # Results
    fragment_labels: np.ndarray  # (N,) array, fragment ID per point (-1 for noise)
    fragment_count: int
    noise_points: int  # Points not assigned to any fragment

    # Quality metrics
    fragmentation_quality_score: float  # 0-1, higher is better
    warnings: List[str] = field(default_factory=list)

    def is_success(self) -> bool:
        """Check if segmentation was successful."""
        return self.fragment_count > 0 and len(self.warnings) == 0


@dataclass
class ScanOperationResult:
    """Generic result container for scan operations."""
    operation: str  # "ingest", "validate", "clean", "segment", "metrics"
    scan_id: UUID
    success: bool
    timestamp: datetime
    result_data: Any  # Operation-specific result
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    processing_time_seconds: float = 0.0

    def has_errors(self) -> bool:
        return len(self.errors) > 0

    def has_warnings(self) -> bool:
        return len(self.warnings) > 0
