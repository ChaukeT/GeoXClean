"""
Professional Geostatistical Result Data Structures
==================================================

This module defines comprehensive data structures for storing all professional
output attributes from estimation and simulation methods, matching the outputs
produced by industry-standard software (Leapfrog, Isatis, Datamine, Surpac).

Author: Block Model Viewer Development Team
Date: 2025
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
import numpy as np


# ============================================================================
# ESTIMATION METHOD RESULTS
# ============================================================================

@dataclass
class OrdinaryKrigingResults:
    """
    Comprehensive results from Ordinary Kriging estimation.
    
    Contains all standard attributes produced by professional geostatistical software.
    """
    # Core Estimation Outputs
    estimates: np.ndarray  # Est - the kriged grade
    status: np.ndarray  # Estimation status (OK, no samples, min samples not met, etc.)
    
    # Kriging System Attributes
    kriging_mean: np.ndarray  # KM - Local mean implied by the OK system
    kriging_variance: np.ndarray  # KV - Variance of the estimation error
    kriging_efficiency: np.ndarray  # KE = 1 - (KV / block variance)
    slope_of_regression: np.ndarray  # SoR - Measure of conditional bias (ideal = 1)
    lagrange_multiplier: np.ndarray  # LM - For OK constraint; used to assess stability
    
    # Neighbourhood / Weight Attributes
    num_samples: np.ndarray  # NS - Number of samples used
    sum_weights: np.ndarray  # SumW - Sum of weights (OK ∑w = 1)
    sum_negative_weights: np.ndarray  # SumN - Negative weights indicate instability
    min_distance: np.ndarray  # MinD - Isotropic distance to nearest sample
    avg_distance: np.ndarray  # AvgD - Average distance to samples
    nearest_sample_id: np.ndarray  # NearID - Nearest sample ID (useful for QA)
    num_duplicates_removed: np.ndarray  # ND - Number of duplicates removed
    
    # Search and Geometry Outputs
    search_pass: np.ndarray  # Pass - Pass1, Pass2, Pass3
    search_volume: np.ndarray  # SearchVol - Search volume used
    ellipsoid_rotation: Optional[np.ndarray] = None  # EllRot - Neighbourhood ellipsoid rotation
    anisotropy_x: Optional[np.ndarray] = None  # AniX - Neighbourhood anisotropy factor X
    anisotropy_y: Optional[np.ndarray] = None  # AniY - Neighbourhood anisotropy factor Y
    anisotropy_z: Optional[np.ndarray] = None  # AniZ - Neighbourhood anisotropy factor Z
    
    # Optional Audit Outputs
    weight_vectors: Optional[np.ndarray] = None  # W1...Wn - Weight vector export (n_samples, n_points)
    sample_coords_matrix: Optional[np.ndarray] = None  # Sample coordinate matrix export
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SimpleKrigingResults:
    """
    Comprehensive results from Simple Kriging estimation.
    
    Similar to OK but with SK-specific attributes.
    """
    # Core Estimation Outputs
    estimates: np.ndarray  # Est - the kriged grade
    status: np.ndarray  # Estimation status
    
    # SK-Specific
    global_mean: float  # GM - SK requires this global mean
    
    # Kriging System Attributes
    kriging_variance: np.ndarray  # KV - Variance of the estimation error
    kriging_efficiency: np.ndarray  # KE = 1 - (KV / block variance)
    # Note: SoR = 1 by definition for SK
    
    # Neighbourhood / Weight Attributes
    num_samples: np.ndarray  # NS
    sum_weights: np.ndarray  # SumW (not constrained to 1 for SK)
    sum_negative_weights: np.ndarray  # SumN
    min_distance: np.ndarray  # MinD
    avg_distance: np.ndarray  # AvgD
    nearest_sample_id: np.ndarray  # NearID
    num_duplicates_removed: np.ndarray  # ND
    
    # Search and Geometry Outputs
    search_pass: np.ndarray  # Pass
    search_volume: Optional[np.ndarray] = None
    ellipsoid_rotation: Optional[np.ndarray] = None
    anisotropy_x: Optional[np.ndarray] = None
    anisotropy_y: Optional[np.ndarray] = None
    anisotropy_z: Optional[np.ndarray] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UniversalKrigingResults:
    """
    Comprehensive results from Universal Kriging estimation.
    
    Same as OK + trend attributes.
    """
    # All OK attributes
    estimates: np.ndarray
    status: np.ndarray
    kriging_mean: np.ndarray
    kriging_variance: np.ndarray
    kriging_efficiency: np.ndarray
    slope_of_regression: np.ndarray
    lagrange_multiplier: np.ndarray
    num_samples: np.ndarray
    sum_weights: np.ndarray
    sum_negative_weights: np.ndarray
    min_distance: np.ndarray
    avg_distance: np.ndarray
    nearest_sample_id: np.ndarray
    num_duplicates_removed: np.ndarray
    search_pass: np.ndarray
    
    # UK-Specific (required fields first)
    drift_value: np.ndarray  # DriftVal - Drift value at estimation point
    residual_estimate: np.ndarray  # EstResidual - Residual estimate
    
    # Optional fields (with defaults)
    search_volume: Optional[np.ndarray] = None
    trend_coefficients: Optional[np.ndarray] = None  # β₀, β₁, β₂, ... (n_points, n_beta)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IndicatorKrigingResults:
    """
    Comprehensive results from Indicator Kriging estimation.
    
    Used for P(grade > cutoff), lithology, codes.
    """
    # IK Outputs
    indicator_probability: np.ndarray  # Prob1 - P(indicator=1)
    local_conditional_variance: np.ndarray  # LCV - p(1-p)
    num_samples: np.ndarray  # NS - Number of samples used
    indicator_kriging_variance: np.ndarray  # IKV - Kriging variance of indicator
    search_pass: np.ndarray  # Pass
    min_distance: np.ndarray  # MinD
    avg_distance: np.ndarray  # AvgD
    num_duplicates_removed: np.ndarray  # ND
    
    # Optional
    indicator_trend: Optional[np.ndarray] = None  # If soft IK used
    multiple_probabilities: Optional[np.ndarray] = None  # If multi-cutoff IK used (n_points, n_cutoffs)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CoKrigingResults:
    """
    Comprehensive results from Co-Kriging estimation.
    
    Professional audit-grade output including:
    - Primary and secondary estimates
    - Weight fractions for audit (primary vs secondary influence)
    - Correlation analysis metadata
    - Markov-1 cross-covariance contribution
    - Neighbor statistics per block
    """
    # CoK-Specific Attributes (required fields first)
    primary_estimate: np.ndarray  # Est₁ - Primary variable estimate
    cross_covariance_contribution: np.ndarray  # XCov - Cross-covariance contribution to estimate
    cokriging_variance: np.ndarray  # CoKV - CoKriging variance
    
    # Standard Attributes (required fields)
    num_samples_primary: np.ndarray  # NS per variable - number of primary neighbors used
    num_samples_secondary: np.ndarray  # Number of secondary values used (1 for collocated)
    sum_weights_primary: np.ndarray  # Σ|wp| - sum of absolute primary weights
    sum_weights_secondary: np.ndarray  # |ws| - secondary weight magnitude
    min_distance: np.ndarray  # MinD - distance to nearest neighbor
    avg_distance: np.ndarray  # AvgD - average distance to neighbors
    search_pass: np.ndarray  # Pass number (for multi-pass estimation)
    
    # Optional fields (with defaults)
    secondary_estimate: Optional[np.ndarray] = None  # Est₂ - Secondary variable estimate at targets
    primary_weight_fraction: Optional[np.ndarray] = None  # |wp| / (|wp| + |ws|) - primary influence %
    secondary_weight_fraction: Optional[np.ndarray] = None  # |ws| / (|wp| + |ws|) - secondary influence % (audit metric)
    
    # Metadata (includes correlation analysis, Markov-1 parameters, etc.)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RegressionKrigingResults:
    """
    Comprehensive results from Regression Kriging / Kriging with External Drift (KED).
    """
    # KED-Specific Outputs (required fields first)
    drift_prediction: np.ndarray  # DriftPred - Drift prediction
    residual_estimate: np.ndarray  # ResidualEst - Residual estimate
    combined_estimate: np.ndarray  # Est = Drift + Residual
    ked_variance: np.ndarray  # KED variance
    
    # Standard Outputs (required fields)
    num_samples: np.ndarray  # NS
    kriging_variance: np.ndarray  # KV
    kriging_efficiency: np.ndarray  # KE
    min_distance: np.ndarray  # MinD
    avg_distance: np.ndarray  # AvgD
    search_pass: np.ndarray  # Pass
    
    # Optional fields (with defaults)
    drift_coefficients: Optional[np.ndarray] = None  # Drift coefficients
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# SIMULATION METHOD RESULTS
# ============================================================================

@dataclass
class SGSIMResults:
    """
    Comprehensive results from Sequential Gaussian Simulation (SGSIM).
    """
    # Per-realisation outputs (required fields first)
    realizations: np.ndarray  # Rₖ - Simulated value for each realisation (nreal, nz, ny, nx)
    
    # Summary outputs (required fields)
    mean: np.ndarray  # E[Z] - Mean over realisations
    variance: np.ndarray  # Var[Z] - Variance over realisations
    std_dev: np.ndarray  # SD - Standard deviation
    coefficient_of_variation: np.ndarray  # CV - Coefficient of variation
    
    # Optional fields (with defaults)
    realization_ids: Optional[np.ndarray] = None  # Realisation ID
    num_samples_per_node: Optional[np.ndarray] = None  # NS used at each node (nreal, nz, ny, nx)
    lagrange_multiplier_per_node: Optional[np.ndarray] = None  # Optional LM per node
    probability_above_cutoff: Optional[np.ndarray] = None  # P(Grade > cutoff) - multiple cutoffs (n_cutoffs, nz, ny, nx)
    p10: Optional[np.ndarray] = None  # P10 value
    p50: Optional[np.ndarray] = None  # P50 value (median)
    p90: Optional[np.ndarray] = None  # P90 value
    normal_score_field: Optional[np.ndarray] = None  # Normal-score field
    back_transform_adjustment: Optional[np.ndarray] = None  # Back-transformation adjustment factor
    neighbourhood_scan_stats: Optional[np.ndarray] = None  # Neighbourhood scan statistics
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SISResults:
    """
    Comprehensive results from Sequential Indicator Simulation (SIS).
    """
    # Per-realisation (required fields first)
    indicator_realizations: np.ndarray  # Indicator value (0/1) for each realisation (nreal, nz, ny, nx)
    
    # Summary (required fields)
    probability_indicator_one: np.ndarray  # P(indicator=1) - probability field
    indicator_variance_field: np.ndarray  # Indicator variance field
    
    # Optional fields (with defaults)
    realization_ids: Optional[np.ndarray] = None  # Realisation ID
    num_samples_per_node: Optional[np.ndarray] = None  # NS
    indicator_conditional_variance: Optional[np.ndarray] = None  # Indicator conditional variance
    connectivity_cluster_id: Optional[np.ndarray] = None  # Connectivity cluster ID (if enabled)
    cluster_volume: Optional[np.ndarray] = None  # Cluster volume
    cluster_tonnage: Optional[np.ndarray] = None  # Cluster tonnage
    cluster_rank: Optional[np.ndarray] = None  # Cluster rank
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DBSResults:
    """
    Comprehensive results from Direct Block Simulation (DBS).
    """
    # Per-realisation (required fields first)
    dbs_grade: np.ndarray  # DBS grade (nreal, nz, ny, nx)
    
    # Summary (required fields)
    dbs_mean: np.ndarray  # DBS mean
    dbs_variance: np.ndarray  # DBS variance
    
    # Optional fields (with defaults)
    dbs_residual_component: Optional[np.ndarray] = None  # DBS residual component
    dbs_conditional_variance: Optional[np.ndarray] = None  # DBS conditional variance
    num_samples_per_node: Optional[np.ndarray] = None  # NS
    dbs_probability_above_cutoff: Optional[np.ndarray] = None  # DBS P(>cut)
    local_variance_inflation_factors: Optional[np.ndarray] = None  # Local variance inflation factors
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TurningBandsResults:
    """
    Comprehensive results from Turning Bands Simulation.
    """
    # Per-realisation (required fields first)
    simulated_field: np.ndarray  # Simulated field Rₖ (nreal, nz, ny, nx)
    
    # Summary (required fields)
    mean: np.ndarray  # Mean over realisations
    variance: np.ndarray  # Variance over realisations
    
    # Optional fields (with defaults)
    band_set_used: Optional[np.ndarray] = None  # Band set used
    num_bands: Optional[int] = None  # NBands - Number of bands
    p10: Optional[np.ndarray] = None  # P10 value
    p50: Optional[np.ndarray] = None  # P50 value
    p90: Optional[np.ndarray] = None  # P90 value
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MPSResults:
    """
    Comprehensive results from Multiple-Point Simulation (MPS).
    """
    # Per-realisation (required fields first)
    simulated_value: np.ndarray  # Simulated categorical/continuous value (nreal, nz, ny, nx)
    
    # Optional fields (with defaults)
    template_match_score: Optional[np.ndarray] = None  # Template match score
    training_image_frequency_hit: Optional[np.ndarray] = None  # Training image frequency hit
    neighbourhood_count: Optional[np.ndarray] = None  # Neighbourhood count
    ti_reproduction_score: Optional[np.ndarray] = None  # TI reproduction score
    connectivity_maps: Optional[np.ndarray] = None  # Connectivity maps
    continuity_probability_maps: Optional[np.ndarray] = None  # Continuity probability maps
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CoSimulationResults:
    """
    Comprehensive results from Co-Simulation (CoSGSIM / Co-SIS).
    """
    # Per-realisation (required fields first)
    co_simulated_primary: np.ndarray  # Co-simulated primary variable (nreal, nz, ny, nx)
    co_simulated_secondary: np.ndarray  # Co-simulated secondary variable (nreal, nz, ny, nx)
    
    # Summary (required fields)
    co_simulation_variance: np.ndarray  # Co-simulation variance
    
    # Optional fields (with defaults)
    joint_realization_id: Optional[np.ndarray] = None  # Joint realisation ID
    cross_variance_contribution: Optional[np.ndarray] = None  # Cross-variance contribution
    joint_distributions: Optional[np.ndarray] = None  # Joint distributions
    conditional_probability: Optional[np.ndarray] = None  # Conditional P(Primary > cutoff | Secondary)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

