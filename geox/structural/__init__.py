"""
GeoX Structural Analysis Package.

Engine-grade stereonet, rose diagram, and kinematic feasibility analysis.

This package provides:
- Vector-space canonical orientation storage (unit normals/axes)
- Deterministic, auditable results
- No UI dependencies - pure computation

Architecture:
- geox.structural.core: Pure computational kernels (no UI imports)
- block_model_viewer.structural: UI adapters and backward-compatible wrappers
"""

__version__ = "1.0.0"

from .core import (
    # Models
    OrientationData,
    StructuralSet,
    StereonetResult,
    RoseResult,
    KinematicResult,
    AnalysisBundle,
    # Orientation math
    dip_dipdir_to_normal,
    normal_to_dip_dipdir,
    plunge_trend_to_vector,
    vector_to_plunge_trend,
    alpha_beta_to_normal,
    canonicalize_to_lower_hemisphere,
    # Stereonet
    project_schmidt,
    project_wulff,
    compute_great_circle,
    # Statistics
    fisher_mean,
    fisher_statistics,
    spherical_kde,
    # Clustering
    cluster_orientations,
    # Rose
    compute_rose_histogram,
    # Kinematics
    planar_sliding_feasibility,
    wedge_sliding_feasibility,
    toppling_feasibility,
    kinematic_analysis,
)

__all__ = [
    "__version__",
    # Models
    "OrientationData",
    "StructuralSet",
    "StereonetResult",
    "RoseResult",
    "KinematicResult",
    "AnalysisBundle",
    # Orientation math
    "dip_dipdir_to_normal",
    "normal_to_dip_dipdir",
    "plunge_trend_to_vector",
    "vector_to_plunge_trend",
    "alpha_beta_to_normal",
    "canonicalize_to_lower_hemisphere",
    # Stereonet
    "project_schmidt",
    "project_wulff",
    "compute_great_circle",
    # Statistics
    "fisher_mean",
    "fisher_statistics",
    "spherical_kde",
    # Clustering
    "cluster_orientations",
    # Rose
    "compute_rose_histogram",
    # Kinematics
    "planar_sliding_feasibility",
    "wedge_sliding_feasibility",
    "toppling_feasibility",
    "kinematic_analysis",
]

