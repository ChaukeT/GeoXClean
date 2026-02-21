"""
Structural Package - Structural data, stereonet analysis, and kinematic analysis.

Includes:
- datasets: Core measurement types (PlaneMeasurement, LineationMeasurement)
- feature_types: Structural features (FaultFeature, FoldFeature, UnconformityFeature)
- stereonet: Stereonet plotting and analysis
- rose_diagram: Rose diagram computation
- kinematic_analysis: Slope stability and wedge failure analysis
"""

from .datasets import (
    PlaneMeasurement,
    LineationMeasurement,
    StructuralDataset,
)

from .feature_types import (
    FeatureType,
    FaultDisplacementType,
    FoldStyle,
    UnconformityType,
    StructuralOrientation,
    StructuralFeature,
    FaultFeature,
    FoldFeature,
    UnconformityFeature,
    StructuralFeatureCollection,
    create_fault_from_points,
    create_fold_from_axes,
    create_unconformity_from_points,
)

from .stereonet import (
    plane_poles_to_stereonet,
    density_grid,
    cluster_planes,
    # Structural feature support
    fault_orientations_to_planes,
    unconformity_orientations_to_planes,
    fold_axes_to_lineations,
    fold_limbs_to_planes,
    collection_to_planes,
    collection_to_lineations,
    lineations_to_stereonet,
    analyze_structural_collection,
)

from .rose_diagram import (
    compute_rose_bins,
)

from .kinematic_analysis import (
    kinematic_plane_slope_feasibility,
    kinematic_wedge_feasibility,
    # Structural feature support
    fault_plane_stability,
    analyze_structural_collection_stability,
)

from .validation import (
    ValidationSeverity,
    ValidationIssue,
    ValidationResult,
    validate_structural_collection,
    log_structural_import_audit,
    write_audit_log,
)

__all__ = [
    # Datasets
    "PlaneMeasurement",
    "LineationMeasurement",
    "StructuralDataset",
    # Feature types
    "FeatureType",
    "FaultDisplacementType",
    "FoldStyle",
    "UnconformityType",
    "StructuralOrientation",
    "StructuralFeature",
    "FaultFeature",
    "FoldFeature",
    "UnconformityFeature",
    "StructuralFeatureCollection",
    "create_fault_from_points",
    "create_fold_from_axes",
    "create_unconformity_from_points",
    # Stereonet
    "plane_poles_to_stereonet",
    "density_grid",
    "cluster_planes",
    # Stereonet - structural feature support
    "fault_orientations_to_planes",
    "unconformity_orientations_to_planes",
    "fold_axes_to_lineations",
    "fold_limbs_to_planes",
    "collection_to_planes",
    "collection_to_lineations",
    "lineations_to_stereonet",
    "analyze_structural_collection",
    # Rose diagram
    "compute_rose_bins",
    # Kinematic analysis
    "kinematic_plane_slope_feasibility",
    "kinematic_wedge_feasibility",
    # Kinematic - structural feature support
    "fault_plane_stability",
    "analyze_structural_collection_stability",
    # Validation
    "ValidationSeverity",
    "ValidationIssue",
    "ValidationResult",
    "validate_structural_collection",
    "log_structural_import_audit",
    "write_audit_log",
]

