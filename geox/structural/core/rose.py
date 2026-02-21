"""
Rose Diagram Computation - Proper Directional Statistics.

Computes rose diagrams with:
- Axial vs bidirectional handling
- Multiple weighting modes (count, length, area, persistence)
- Declination and domain rotation corrections
- Circular statistics (mean direction, variance, resultant length)
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple, Union

from .models import RoseResult, WeightingMode


# =============================================================================
# ROSE HISTOGRAM COMPUTATION
# =============================================================================

def compute_rose_histogram(
    directions: np.ndarray,
    n_bins: int = 36,
    is_axial: bool = True,
    weighting: WeightingMode = WeightingMode.COUNT,
    weights: Optional[np.ndarray] = None,
    declination: float = 0.0,
    domain_rotation: float = 0.0,
) -> RoseResult:
    """
    Compute a rose diagram histogram.
    
    Args:
        directions: Array of directions in degrees (0-360)
        n_bins: Number of bins (default 36 for 10-degree bins)
        is_axial: If True, treat data as axial (bidirectional, 0-180 symmetric)
        weighting: Weighting mode for bin counts
        weights: Optional array of weights (for length/area/persistence weighting)
        declination: Magnetic declination correction in degrees (added to directions)
        domain_rotation: Domain rotation correction in degrees (added to directions)
    
    Returns:
        RoseResult with histogram data and statistics
    """
    directions = np.atleast_1d(np.asarray(directions, dtype=np.float64))
    
    if len(directions) == 0:
        return RoseResult(
            bin_edges=np.linspace(0, 360, n_bins + 1),
            bin_centers=np.linspace(5, 355, n_bins),
            counts=np.zeros(n_bins),
            n_bins=n_bins,
            weighting=weighting,
            is_axial=is_axial,
            mean_direction=0.0,
            mean_resultant_length=0.0,
            circular_variance=1.0,
            parameters={"n_data": 0}
        )
    
    # Apply corrections
    directions = directions + declination + domain_rotation
    
    # Wrap to 0-360
    directions = directions % 360
    
    # For axial data, fold to 0-180 range
    if is_axial:
        directions = np.where(directions >= 180, directions - 180, directions)
        bin_range = (0, 180)
        bin_edges = np.linspace(0, 180, n_bins // 2 + 1)
    else:
        bin_range = (0, 360)
        bin_edges = np.linspace(0, 360, n_bins + 1)
    
    # Prepare weights
    if weights is None or weighting == WeightingMode.COUNT:
        bin_weights = None
    else:
        bin_weights = np.asarray(weights, dtype=np.float64)
        if len(bin_weights) != len(directions):
            raise ValueError("Weights must have same length as directions")
    
    # Compute histogram
    if bin_weights is not None:
        counts, _ = np.histogram(directions, bins=bin_edges, weights=bin_weights)
    else:
        counts, _ = np.histogram(directions, bins=bin_edges)
    
    # For axial data, mirror the histogram
    if is_axial:
        # Double the bins to cover 0-360
        full_counts = np.concatenate([counts, counts])
        full_bin_edges = np.linspace(0, 360, n_bins + 1)
    else:
        full_counts = counts
        full_bin_edges = bin_edges
    
    # Bin centers
    bin_centers = (full_bin_edges[:-1] + full_bin_edges[1:]) / 2
    
    # Circular statistics
    mean_dir, mean_r, circ_var = _circular_statistics(
        directions, 
        is_axial=is_axial,
        weights=bin_weights
    )
    
    # Build parameters dict for audit
    parameters = {
        "n_data": len(directions),
        "n_bins": n_bins,
        "is_axial": is_axial,
        "weighting": weighting.value,
        "declination": declination,
        "domain_rotation": domain_rotation,
    }
    
    return RoseResult(
        bin_edges=full_bin_edges,
        bin_centers=bin_centers,
        counts=full_counts,
        n_bins=len(full_counts),
        weighting=weighting,
        is_axial=is_axial,
        mean_direction=mean_dir,
        mean_resultant_length=mean_r,
        circular_variance=circ_var,
        parameters=parameters,
    )


def compute_rose_from_normals(
    normals: np.ndarray,
    use_dip_direction: bool = True,
    n_bins: int = 36,
    is_axial: bool = True,
    weighting: WeightingMode = WeightingMode.COUNT,
    weights: Optional[np.ndarray] = None,
    **kwargs
) -> RoseResult:
    """
    Compute rose diagram from plane normal vectors.
    
    Args:
        normals: Nx3 array of unit normal vectors
        use_dip_direction: If True, use dip direction; if False, use strike
        n_bins: Number of bins
        is_axial: If True, treat as axial data
        weighting: Weighting mode
        weights: Optional weights
        **kwargs: Additional arguments passed to compute_rose_histogram
    
    Returns:
        RoseResult
    """
    from .orientation_math import normal_to_dip_dipdir, strike_from_dip_direction
    
    normals = np.atleast_2d(normals)
    
    # Get dip directions
    _, dip_directions = normal_to_dip_dipdir(normals)
    
    if use_dip_direction:
        directions = dip_directions
    else:
        directions = strike_from_dip_direction(dip_directions)
    
    return compute_rose_histogram(
        directions=directions,
        n_bins=n_bins,
        is_axial=is_axial,
        weighting=weighting,
        weights=weights,
        **kwargs
    )


def compute_rose_from_lineations(
    vectors: np.ndarray,
    n_bins: int = 36,
    is_axial: bool = True,
    weighting: WeightingMode = WeightingMode.COUNT,
    weights: Optional[np.ndarray] = None,
    **kwargs
) -> RoseResult:
    """
    Compute rose diagram from lineation vectors (trend).
    
    Args:
        vectors: Nx3 array of lineation direction vectors
        n_bins: Number of bins
        is_axial: If True, treat as axial data
        weighting: Weighting mode
        weights: Optional weights
        **kwargs: Additional arguments
    
    Returns:
        RoseResult
    """
    from .orientation_math import vector_to_plunge_trend
    
    vectors = np.atleast_2d(vectors)
    
    # Get trends
    _, trends = vector_to_plunge_trend(vectors)
    
    return compute_rose_histogram(
        directions=trends,
        n_bins=n_bins,
        is_axial=is_axial,
        weighting=weighting,
        weights=weights,
        **kwargs
    )


# =============================================================================
# CIRCULAR STATISTICS
# =============================================================================

def compute_rose_statistics(
    directions: np.ndarray,
    is_axial: bool = True,
    weights: Optional[np.ndarray] = None
) -> dict:
    """
    Compute comprehensive circular statistics for directional data.
    
    Args:
        directions: Array of directions in degrees
        is_axial: If True, double angles for axial data
        weights: Optional weights
    
    Returns:
        Dictionary with circular statistics
    """
    directions = np.atleast_1d(np.asarray(directions, dtype=np.float64))
    n = len(directions)
    
    if n == 0:
        return {
            "mean_direction": 0.0,
            "mean_resultant_length": 0.0,
            "circular_variance": 1.0,
            "circular_std": float('inf'),
            "concentration": 0.0,
            "rayleigh_p": 1.0,
            "n": 0,
        }
    
    # Handle weights
    if weights is None:
        weights = np.ones(n)
    else:
        weights = np.asarray(weights, dtype=np.float64)
    
    total_weight = np.sum(weights)
    if total_weight < 1e-10:
        total_weight = 1.0
    
    # For axial data, double the angles
    if is_axial:
        angles_rad = 2 * np.radians(directions)
    else:
        angles_rad = np.radians(directions)
    
    # Weighted mean resultant vector
    C = np.sum(weights * np.cos(angles_rad)) / total_weight
    S = np.sum(weights * np.sin(angles_rad)) / total_weight
    
    # Mean resultant length
    R_bar = np.sqrt(C**2 + S**2)
    
    # Mean direction
    mean_dir = np.degrees(np.arctan2(S, C))
    if is_axial:
        mean_dir = mean_dir / 2  # Convert back
    mean_dir = mean_dir % 360
    
    # Circular variance (0 = concentrated, 1 = uniform)
    circ_var = 1 - R_bar
    
    # Circular standard deviation
    if R_bar < 1 - 1e-10:
        circ_std = np.degrees(np.sqrt(-2 * np.log(R_bar)))
        if is_axial:
            circ_std = circ_std / 2
    else:
        circ_std = 0.0
    
    # Concentration parameter (von Mises kappa approximation)
    if R_bar < 0.53:
        kappa = 2 * R_bar + R_bar**3 + 5 * R_bar**5 / 6
    elif R_bar < 0.85:
        kappa = -0.4 + 1.39 * R_bar + 0.43 / (1 - R_bar)
    else:
        kappa = 1 / (R_bar**3 - 4 * R_bar**2 + 3 * R_bar)
    
    # Rayleigh test p-value (test for uniformity)
    R = R_bar * n
    rayleigh_p = np.exp(-R**2 / n)
    
    return {
        "mean_direction": mean_dir,
        "mean_resultant_length": R_bar,
        "circular_variance": circ_var,
        "circular_std": circ_std,
        "concentration": kappa,
        "rayleigh_p": rayleigh_p,
        "n": n,
    }


def _circular_statistics(
    directions: np.ndarray,
    is_axial: bool = True,
    weights: Optional[np.ndarray] = None
) -> Tuple[float, float, float]:
    """
    Internal function to compute basic circular statistics.
    
    Returns:
        Tuple of (mean_direction, mean_resultant_length, circular_variance)
    """
    stats = compute_rose_statistics(directions, is_axial, weights)
    return (
        stats["mean_direction"],
        stats["mean_resultant_length"],
        stats["circular_variance"]
    )


# =============================================================================
# CORRECTIONS AND TRANSFORMATIONS
# =============================================================================

def apply_declination_correction(
    directions: np.ndarray,
    declination: float
) -> np.ndarray:
    """
    Apply magnetic declination correction to directions.
    
    Args:
        directions: Array of directions in degrees (magnetic north)
        declination: Declination in degrees (positive = east)
    
    Returns:
        Corrected directions (true north)
    """
    return (np.asarray(directions) + declination) % 360


def bidirectional_to_axial(directions: np.ndarray) -> np.ndarray:
    """
    Convert bidirectional (0-360) data to axial (0-180) representation.
    
    Folds directions >= 180 to their complementary direction.
    
    Args:
        directions: Array of directions in degrees (0-360)
    
    Returns:
        Array of directions in degrees (0-180)
    """
    directions = np.asarray(directions, dtype=np.float64) % 360
    return np.where(directions >= 180, directions - 180, directions)


def axial_to_bidirectional(directions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Expand axial (0-180) data to bidirectional representation.
    
    Args:
        directions: Array of axial directions in degrees (0-180)
    
    Returns:
        Tuple of (original directions, opposite directions)
    """
    directions = np.asarray(directions, dtype=np.float64) % 180
    opposite = (directions + 180) % 360
    return directions, opposite


def compute_rayleigh_test(
    directions: np.ndarray,
    is_axial: bool = True
) -> Tuple[float, float]:
    """
    Rayleigh test for circular uniformity.
    
    Tests H0: data is uniformly distributed on circle.
    
    Args:
        directions: Array of directions in degrees
        is_axial: If True, double angles
    
    Returns:
        Tuple of (test_statistic, p_value)
    """
    directions = np.atleast_1d(np.asarray(directions, dtype=np.float64))
    n = len(directions)
    
    if n < 2:
        return 0.0, 1.0
    
    if is_axial:
        angles_rad = 2 * np.radians(directions)
    else:
        angles_rad = np.radians(directions)
    
    C = np.sum(np.cos(angles_rad))
    S = np.sum(np.sin(angles_rad))
    R = np.sqrt(C**2 + S**2)
    
    # Test statistic
    Z = R**2 / n
    
    # P-value (approximation for large n)
    p_value = np.exp(-Z) * (1 + (2 * Z - Z**2) / (4 * n) - 
                            (24 * Z - 132 * Z**2 + 76 * Z**3 - 9 * Z**4) / (288 * n**2))
    
    return float(Z), float(max(0, min(1, p_value)))

