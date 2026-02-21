"""
Spherical Statistics - Fisher Distribution and Spherical KDE.

Implements vectorized spherical statistical methods for orientation data.

Fisher Distribution:
- Maximum likelihood estimation of mean direction
- Concentration parameter (kappa)
- Confidence cones
- Dispersion metrics

Spherical KDE:
- von Mises-Fisher kernel
- Vectorized density estimation
- Adaptive bandwidth
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Optional, Dict, Any


# =============================================================================
# FISHER STATISTICS
# =============================================================================

def resultant_length(normals: np.ndarray) -> float:
    """
    Compute the resultant length R of a set of unit vectors.
    
    R is the magnitude of the vector sum, ranging from 0 (uniform)
    to N (all vectors identical).
    
    Args:
        normals: Nx3 array of unit vectors
    
    Returns:
        Resultant length R
    """
    normals = np.atleast_2d(normals)
    vector_sum = np.sum(normals, axis=0)
    return float(np.linalg.norm(vector_sum))


def mean_resultant_length(normals: np.ndarray) -> float:
    """
    Compute the mean resultant length R-bar = R / N.
    
    Ranges from 0 (uniform) to 1 (perfectly concentrated).
    
    Args:
        normals: Nx3 array of unit vectors
    
    Returns:
        Mean resultant length
    """
    normals = np.atleast_2d(normals)
    n = len(normals)
    if n == 0:
        return 0.0
    return resultant_length(normals) / n


def fisher_mean(normals: np.ndarray) -> np.ndarray:
    """
    Compute the Fisher mean direction (unit vector).
    
    The mean direction is the normalized vector sum.
    
    Args:
        normals: Nx3 array of unit vectors
    
    Returns:
        3-element unit vector (mean direction)
    """
    normals = np.atleast_2d(normals)
    vector_sum = np.sum(normals, axis=0)
    norm = np.linalg.norm(vector_sum)
    
    if norm < 1e-10:
        # Uniform distribution - no preferred direction
        return np.array([0.0, 0.0, -1.0])  # Default to vertical down
    
    return vector_sum / norm


def fisher_kappa(normals: np.ndarray) -> float:
    """
    Estimate the Fisher concentration parameter kappa.
    
    Uses the maximum likelihood estimator for kappa given R-bar.
    
    Args:
        normals: Nx3 array of unit vectors
    
    Returns:
        Concentration parameter kappa (0 = uniform, large = concentrated)
    
    Note:
        For p=3 dimensions, the MLE is approximately:
        kappa ≈ (p - 2) / (1 - R_bar^2) for R_bar close to 1
        kappa ≈ R_bar * (p - R_bar^2) / (1 - R_bar^2) more generally
    """
    normals = np.atleast_2d(normals)
    n = len(normals)
    
    if n < 2:
        return 0.0
    
    R_bar = mean_resultant_length(normals)
    
    if R_bar >= 0.9999:
        # Very concentrated - use large kappa approximation
        return 1000.0
    
    if R_bar < 0.001:
        # Nearly uniform
        return 0.0
    
    # Dimension p = 3
    p = 3
    
    # MLE approximation for Fisher distribution
    # kappa = R_bar * (p - R_bar^2) / (1 - R_bar^2)
    kappa = R_bar * (p - R_bar**2) / (1 - R_bar**2)
    
    # Apply bias correction for small samples
    if n < 10:
        kappa = max(0, kappa - 2 / n)
    
    return float(kappa)


def fisher_statistics(normals: np.ndarray) -> Dict[str, Any]:
    """
    Compute comprehensive Fisher statistics for orientation data.
    
    Args:
        normals: Nx3 array of unit vectors
    
    Returns:
        Dictionary with:
            - mean_direction: Unit vector mean
            - mean_dip: Mean dip angle (degrees)
            - mean_dip_direction: Mean dip direction (degrees)
            - kappa: Concentration parameter
            - R: Resultant length
            - R_bar: Mean resultant length
            - confidence_95: 95% confidence cone half-angle (degrees)
            - confidence_99: 99% confidence cone half-angle (degrees)
            - dispersion: Angular dispersion (degrees)
            - spherical_variance: Spherical variance (0-1)
            - n: Number of measurements
    """
    normals = np.atleast_2d(normals)
    n = len(normals)
    
    if n == 0:
        return {
            "mean_direction": np.array([0, 0, -1]),
            "mean_dip": 0.0,
            "mean_dip_direction": 0.0,
            "kappa": 0.0,
            "R": 0.0,
            "R_bar": 0.0,
            "confidence_95": 180.0,
            "confidence_99": 180.0,
            "dispersion": 90.0,
            "spherical_variance": 1.0,
            "n": 0,
        }
    
    # Basic statistics
    mean_dir = fisher_mean(normals)
    R = resultant_length(normals)
    R_bar = R / n
    kappa = fisher_kappa(normals)
    
    # Convert mean to dip/dip-direction
    from .orientation_math import normal_to_dip_dipdir
    mean_dip, mean_dd = normal_to_dip_dipdir(mean_dir.reshape(1, 3))
    
    # Confidence cones
    conf_95 = confidence_cone(normals, confidence=0.95)
    conf_99 = confidence_cone(normals, confidence=0.99)
    
    # Dispersion (angular standard deviation)
    disp = spherical_dispersion(normals)
    
    # Spherical variance
    sph_var = spherical_variance(normals)
    
    return {
        "mean_direction": mean_dir,
        "mean_dip": float(mean_dip[0]),
        "mean_dip_direction": float(mean_dd[0]),
        "kappa": kappa,
        "R": R,
        "R_bar": R_bar,
        "confidence_95": conf_95,
        "confidence_99": conf_99,
        "dispersion": disp,
        "spherical_variance": sph_var,
        "n": n,
    }


def confidence_cone(
    normals: np.ndarray,
    confidence: float = 0.95
) -> float:
    """
    Compute the confidence cone half-angle for the mean direction.
    
    Uses Fisher distribution theory.
    
    Args:
        normals: Nx3 array of unit vectors
        confidence: Confidence level (e.g., 0.95 for 95%)
    
    Returns:
        Half-angle of confidence cone in degrees
    """
    normals = np.atleast_2d(normals)
    n = len(normals)
    
    if n < 2:
        return 180.0
    
    R = resultant_length(normals)
    
    if R < 1e-10:
        return 180.0
    
    # Fisher's formula for confidence cone
    # cos(alpha) = 1 - ((n - R) / R) * ((1 / (1 - p))^(1 / (n - 1)) - 1)
    # where p is the confidence level
    
    p = confidence
    
    # For small samples, use exact formula
    term = ((n - R) / R) * ((1.0 / (1.0 - p))**(1.0 / (n - 1)) - 1)
    
    cos_alpha = 1 - term
    cos_alpha = np.clip(cos_alpha, -1, 1)
    
    alpha = np.degrees(np.arccos(cos_alpha))
    
    return float(alpha)


def spherical_variance(normals: np.ndarray) -> float:
    """
    Compute spherical variance (1 - R_bar).
    
    Ranges from 0 (perfectly concentrated) to 1 (uniform).
    
    Args:
        normals: Nx3 array of unit vectors
    
    Returns:
        Spherical variance
    """
    return 1 - mean_resultant_length(normals)


def spherical_dispersion(normals: np.ndarray) -> float:
    """
    Compute angular dispersion (standard deviation analog).
    
    Uses the formula: dispersion = arccos(R_bar) in degrees.
    
    Args:
        normals: Nx3 array of unit vectors
    
    Returns:
        Angular dispersion in degrees
    """
    R_bar = mean_resultant_length(normals)
    R_bar = np.clip(R_bar, -1, 1)
    return float(np.degrees(np.arccos(R_bar)))


# =============================================================================
# SPHERICAL KDE (Kernel Density Estimation)
# =============================================================================

def spherical_kde(
    normals: np.ndarray,
    grid_vectors: np.ndarray,
    bandwidth: Optional[float] = None,
    weights: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute spherical kernel density estimation using von Mises-Fisher kernel.
    
    Uses vectorized dot products for efficiency.
    
    Args:
        normals: Nx3 array of data unit vectors
        grid_vectors: Mx3 array of grid unit vectors to evaluate density at
        bandwidth: Concentration parameter for kernel (higher = narrower)
                   If None, uses Scott's rule adapted for spherical data
        weights: Optional array of weights for each data point
    
    Returns:
        M-element array of density values at grid points
    
    Note:
        Density is computed as sum of von Mises-Fisher kernels:
        f(x) = sum_i w_i * C(kappa) * exp(kappa * (x · d_i))
        where C(kappa) is the normalization constant.
    """
    normals = np.atleast_2d(normals)
    grid_vectors = np.atleast_2d(grid_vectors)
    
    n = len(normals)
    if n == 0:
        return np.zeros(len(grid_vectors))
    
    # Default bandwidth using Scott's rule analog for spherical data
    if bandwidth is None:
        # Use concentration of data to set bandwidth
        kappa_data = fisher_kappa(normals)
        # Bandwidth roughly inversely related to data concentration
        if kappa_data > 1:
            bandwidth = np.sqrt(kappa_data) * (n ** (-1/5))
        else:
            bandwidth = 5.0 * (n ** (-1/5))  # Broader for dispersed data
        bandwidth = max(1.0, min(50.0, bandwidth))
    
    # Default weights
    if weights is None:
        weights = np.ones(n) / n
    else:
        weights = np.asarray(weights)
        weights = weights / np.sum(weights)  # Normalize
    
    # Compute density using vectorized operations
    # Dot product between all grid points and all data points
    # Result is (M, N) matrix
    dots = grid_vectors @ normals.T  # (M, N)
    
    # von Mises-Fisher kernel: exp(kappa * dot)
    # We don't need the normalization constant since we'll normalize anyway
    kernel_values = np.exp(bandwidth * dots)  # (M, N)
    
    # Weighted sum over data points
    density = kernel_values @ weights  # (M,)
    
    # Normalize to max = 1 for plotting convenience
    max_density = np.max(density)
    if max_density > 1e-10:
        density = density / max_density
    
    return density


def spherical_kde_grid(
    normals: np.ndarray,
    n_lat: int = 91,
    n_lon: int = 181,
    bandwidth: Optional[float] = None,
    weights: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute spherical KDE on a regular lat/lon grid.
    
    Args:
        normals: Nx3 array of data unit vectors (should be in lower hemisphere)
        n_lat: Number of latitude divisions (90 degrees range)
        n_lon: Number of longitude divisions (360 degrees range)
        bandwidth: KDE bandwidth parameter
        weights: Optional weights for each data point
    
    Returns:
        Tuple of:
            - lat_grid: Latitude grid in degrees (n_lat x n_lon)
            - lon_grid: Longitude grid in degrees (n_lat x n_lon)
            - density: Density values (n_lat x n_lon)
    """
    from .orientation_math import canonicalize_to_lower_hemisphere
    
    normals = np.atleast_2d(normals)
    normals = canonicalize_to_lower_hemisphere(normals)
    
    # Create grid (lower hemisphere only: lat from 0 to 90)
    lat = np.linspace(0, 90, n_lat)
    lon = np.linspace(0, 360, n_lon, endpoint=False)
    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing='ij')
    
    # Convert to unit vectors
    lat_rad = np.radians(lat_grid)
    lon_rad = np.radians(lon_grid)
    
    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    sin_lon = np.sin(lon_rad)
    cos_lon = np.cos(lon_rad)
    
    # Lower hemisphere (negative Z)
    gx = sin_lat * sin_lon
    gy = sin_lat * cos_lon
    gz = -cos_lat
    
    grid_vectors = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    
    # Compute density
    density_flat = spherical_kde(normals, grid_vectors, bandwidth, weights)
    density = density_flat.reshape(n_lat, n_lon)
    
    return lat_grid, lon_grid, density


def adaptive_bandwidth(normals: np.ndarray, method: str = "scott") -> float:
    """
    Compute adaptive bandwidth for spherical KDE.
    
    Args:
        normals: Nx3 array of unit vectors
        method: "scott" or "silverman" or "concentration"
    
    Returns:
        Bandwidth (concentration parameter)
    """
    normals = np.atleast_2d(normals)
    n = len(normals)
    
    if n < 2:
        return 10.0
    
    if method == "concentration":
        # Use data concentration
        kappa = fisher_kappa(normals)
        return max(1.0, kappa)
    
    elif method == "silverman":
        # Silverman's rule adapted
        kappa = fisher_kappa(normals)
        sigma = 1.0 / np.sqrt(kappa) if kappa > 0.1 else 3.0
        bandwidth = 1.06 * sigma * (n ** (-1/5))
        return 1.0 / (bandwidth ** 2)
    
    else:  # scott
        # Scott's rule adapted
        kappa = fisher_kappa(normals)
        sigma = 1.0 / np.sqrt(kappa) if kappa > 0.1 else 3.0
        bandwidth = sigma * (n ** (-1/5))
        return 1.0 / (bandwidth ** 2)


# =============================================================================
# CONTOUR LEVELS
# =============================================================================

def density_contour_levels(
    density: np.ndarray,
    n_levels: int = 6,
    method: str = "linear"
) -> np.ndarray:
    """
    Compute contour levels for density plot.
    
    Args:
        density: 2D array of density values
        n_levels: Number of contour levels
        method: "linear", "percentile", or "log"
    
    Returns:
        Array of contour level values
    """
    max_density = np.max(density)
    min_density = np.min(density[density > 0]) if np.any(density > 0) else 0
    
    if method == "percentile":
        # Use percentiles of non-zero values
        non_zero = density[density > 0]
        if len(non_zero) == 0:
            return np.array([0.5, 1.0])
        percentiles = np.linspace(10, 100, n_levels)
        levels = np.percentile(non_zero, percentiles)
    
    elif method == "log":
        # Logarithmic spacing
        if min_density <= 0:
            min_density = max_density / 100
        levels = np.logspace(np.log10(min_density), np.log10(max_density), n_levels)
    
    else:  # linear
        levels = np.linspace(max_density / n_levels, max_density, n_levels)
    
    return levels

