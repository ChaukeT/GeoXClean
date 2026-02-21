"""
GeoX Structural Analysis Core - Pure Computational Kernels.

NO UI IMPORTS ALLOWED IN THIS PACKAGE.

This module provides industry-grade structural geology computations:
- Vector-based orientation math (all angles converted to unit vectors)
- Stereonet projections (Schmidt equal-area, Wulff equal-angle)
- Spherical KDE density estimation
- Clustering (HDBSCAN/DBSCAN in orientation space)
- Fisher statistics (mean direction, concentration, confidence cones)
- Rose diagram computation
- Kinematic feasibility analysis (planar, wedge, toppling)

Design Principles:
- All orientations stored as unit vectors (Nx3 arrays)
- Angles are inputs only - immediately converted to vectors
- Angles re-derived on demand for display
- Vectorized operations (no per-point Python loops)
- Deterministic and auditable (seeds, parameters logged)
"""

__version__ = "1.0.0"

from .models import (
    OrientationData,
    StructuralSet,
    StereonetResult,
    RoseResult,
    KinematicResult,
    AnalysisBundle,
    FailureMode,
    NetType,
    Hemisphere,
    WeightingMode,
)

from .orientation_math import (
    dip_dipdir_to_normal,
    normal_to_dip_dipdir,
    plunge_trend_to_vector,
    vector_to_plunge_trend,
    alpha_beta_to_normal,
    normal_to_alpha_beta,
    canonicalize_to_lower_hemisphere,
    canonicalize_to_upper_hemisphere,
    normalize_vectors,
    angle_between_vectors,
    rotation_matrix_axis_angle,
    rotate_vectors,
    strike_from_dip_direction,
    dip_direction_from_strike,
)

from .stereonet import (
    project_schmidt,
    project_wulff,
    inverse_project_schmidt,
    inverse_project_wulff,
    compute_great_circle,
    compute_small_circle,
    generate_stereonet_grid,
)

from .spherical_stats import (
    fisher_mean,
    fisher_kappa,
    fisher_statistics,
    spherical_kde,
    spherical_kde_grid,
    confidence_cone,
    spherical_variance,
    resultant_length,
)

from .clustering import (
    cluster_orientations,
    identify_sets,
    merge_sets,
    compute_set_statistics,
)

from .rose import (
    compute_rose_histogram,
    compute_rose_statistics,
    apply_declination_correction,
    bidirectional_to_axial,
)

from .kinematics import (
    planar_sliding_feasibility,
    wedge_sliding_feasibility,
    toppling_feasibility,
    kinematic_analysis,
    compute_daylight_envelope,
    compute_friction_cone,
    line_of_intersection,
    summarize_kinematic_results,
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
    "FailureMode",
    "NetType",
    "Hemisphere",
    "WeightingMode",
    # Orientation math
    "dip_dipdir_to_normal",
    "normal_to_dip_dipdir",
    "plunge_trend_to_vector",
    "vector_to_plunge_trend",
    "alpha_beta_to_normal",
    "normal_to_alpha_beta",
    "canonicalize_to_lower_hemisphere",
    "canonicalize_to_upper_hemisphere",
    "normalize_vectors",
    "angle_between_vectors",
    "rotation_matrix_axis_angle",
    "rotate_vectors",
    "strike_from_dip_direction",
    "dip_direction_from_strike",
    # Stereonet
    "project_schmidt",
    "project_wulff",
    "inverse_project_schmidt",
    "inverse_project_wulff",
    "compute_great_circle",
    "compute_small_circle",
    "generate_stereonet_grid",
    # Statistics
    "fisher_mean",
    "fisher_kappa",
    "fisher_statistics",
    "spherical_kde",
    "spherical_kde_grid",
    "confidence_cone",
    "spherical_variance",
    "resultant_length",
    # Clustering
    "cluster_orientations",
    "identify_sets",
    "merge_sets",
    "compute_set_statistics",
    # Rose
    "compute_rose_histogram",
    "compute_rose_statistics",
    "apply_declination_correction",
    "bidirectional_to_axial",
    # Kinematics
    "planar_sliding_feasibility",
    "wedge_sliding_feasibility",
    "toppling_feasibility",
    "kinematic_analysis",
    "compute_daylight_envelope",
    "compute_friction_cone",
    "line_of_intersection",
    "summarize_kinematic_results",
]

