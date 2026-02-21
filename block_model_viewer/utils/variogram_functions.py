"""
Shared variogram functions for Kriging and SGSIM.

This module centralizes variogram model implementations to avoid duplication
between kriging3d.py and sgsim3d.py.
"""

import numpy as np
from typing import Callable, Tuple, Dict, Any, List, Optional


def spherical_variogram(h: np.ndarray, nugget: float, sill: float, range_: float) -> np.ndarray:
    """
    Spherical variogram model.
    
    Args:
        h: Distance array
        nugget: Nugget effect (variance at h=0)
        sill: Total sill (nugget + partial sill)
        range_: Range parameter (distance where sill is reached)
    
    Returns:
        Variogram values
    """
    gamma = np.zeros_like(h)
    
    # For h = 0
    gamma[h == 0] = 0
    
    # For 0 < h <= range
    mask = (h > 0) & (h <= range_)
    gamma[mask] = nugget + (sill - nugget) * (
        1.5 * (h[mask] / range_) - 0.5 * (h[mask] / range_) ** 3
    )
    
    # For h > range
    gamma[h > range_] = sill
    
    return gamma


def exponential_variogram(h: np.ndarray, nugget: float, sill: float, range_: float) -> np.ndarray:
    """
    Exponential variogram model.
    
    Args:
        h: Distance array
        nugget: Nugget effect
        sill: Total sill
        range_: Practical range (distance where 95% of sill is reached)
    
    Returns:
        Variogram values
    """
    gamma = np.zeros_like(h)
    
    # For h = 0
    gamma[h == 0] = 0
    
    # For h > 0
    mask = h > 0
    gamma[mask] = nugget + (sill - nugget) * (1 - np.exp(-3 * h[mask] / range_))
    
    return gamma


def gaussian_variogram(h: np.ndarray, nugget: float, sill: float, range_: float) -> np.ndarray:
    """
    Gaussian variogram model (practical range convention).
    
    Uses practical range where 95% of sill is reached at h = range_
    gamma(h) = nugget + (sill - nugget) * (1 - exp(-3 * (h/range)^2))
    
    Args:
        h: Distance array
        nugget: Nugget effect
        sill: Total sill
        range_: Practical range parameter
    
    Returns:
        Variogram values
    """
    gamma = np.zeros_like(h)
    
    # For h = 0
    gamma[h == 0] = 0
    
    # For h > 0 - uses practical range convention (factor of 3)
    mask = h > 0
    gamma[mask] = nugget + (sill - nugget) * (1 - np.exp(-3.0 * (h[mask] / range_) ** 2))
    
    return gamma


def linear_variogram(h: np.ndarray, nugget: float, slope: float) -> np.ndarray:
    """
    Linear variogram model (unbounded).
    
    Args:
        h: Distance array
        nugget: Nugget effect
        slope: Linear slope
    
    Returns:
        Variogram values
    """
    gamma = nugget + slope * h
    return gamma


def power_variogram(h: np.ndarray, nugget: float, coef: float, exponent: float) -> np.ndarray:
    """
    Power variogram model.
    
    Args:
        h: Distance array
        nugget: Nugget effect
        coef: Coefficient
        exponent: Power exponent (0 < exponent < 2)
    
    Returns:
        Variogram values
    """
    gamma = nugget + coef * h ** exponent
    return gamma


def get_variogram_function(model_type: str) -> Callable:
    """
    Get variogram function by model type name.
    
    Args:
        model_type: One of 'spherical', 'exponential', 'gaussian', 'linear', 'power'
    
    Returns:
        Variogram function
    
    Raises:
        ValueError: If model_type is not recognized
    """
    models = {
        'spherical': spherical_variogram,
        'exponential': exponential_variogram,
        'gaussian': gaussian_variogram,
        'linear': linear_variogram,
        'power': power_variogram
    }
    
    model_type_lower = model_type.lower()
    if model_type_lower not in models:
        raise ValueError(
            f"Unknown variogram model '{model_type}'. "
            f"Choose from: {', '.join(models.keys())}"
        )
    
    return models[model_type_lower]


def fit_variogram(
    distances: np.ndarray,
    semivariances: np.ndarray,
    model_type: str = 'spherical'
) -> Tuple[float, float, float]:
    """
    Fit a variogram model to experimental data using least squares.
    
    Industry-standard approach:
    - Nugget estimated from first lags or extrapolated intercept
    - Sill estimated from plateau of experimental variogram
    - Range estimated where variogram reaches ~95% of sill
    
    Args:
        distances: Array of lag distances
        semivariances: Array of semivariance values
        model_type: Variogram model type
    
    Returns:
        Tuple of (nugget, sill, range)
    """
    from scipy.optimize import curve_fit
    
    # Clean input data
    distances = np.asarray(distances, dtype=float)
    semivariances = np.asarray(semivariances, dtype=float)
    
    # Filter invalid values
    mask = np.isfinite(distances) & np.isfinite(semivariances) & (distances >= 0)
    distances = distances[mask]
    semivariances = semivariances[mask]
    
    if len(distances) < 3:
        # Not enough data - return reasonable defaults
        return 0.0, float(np.max(semivariances)) if len(semivariances) > 0 else 1.0, 100.0
    
    # Sort by distance
    sort_idx = np.argsort(distances)
    distances = distances[sort_idx]
    semivariances = semivariances[sort_idx]
    
    # Get variogram function
    variogram_func = get_variogram_function(model_type)
    
    # Industry-standard initial parameter estimates
    y_min = float(np.min(semivariances))
    y_max = float(np.max(semivariances))
    x_max = float(np.max(distances))
    x_min = float(np.min(distances[distances > 0])) if np.any(distances > 0) else 1.0
    
    # 1. Nugget estimation - critical for proper variogram
    # Check for very short-distance pairs (duplicates or tight spacing)
    very_short_mask = distances < x_min * 1.5
    
    if np.sum(very_short_mask) >= 2:
        # Use mean of shortest-distance pairs for nugget
        nugget_guess = float(np.mean(semivariances[very_short_mask]))
    elif len(distances) >= 3 and distances[0] > 0:
        # Weighted extrapolation using first 3 points
        n_pts = min(3, len(distances))
        x_early = distances[:n_pts]
        y_early = semivariances[:n_pts]
        weights = 1.0 / (x_early + 1e-6)
        weights = weights / weights.sum()
        x_mean = np.sum(weights * x_early)
        y_mean = np.sum(weights * y_early)
        slope = np.sum(weights * (x_early - x_mean) * (y_early - y_mean)) / (np.sum(weights * (x_early - x_mean)**2) + 1e-12)
        nugget_guess = max(0.0, float(y_mean - slope * x_mean))
    elif len(distances) >= 2 and distances[0] > 0:
        slope = (semivariances[1] - semivariances[0]) / (distances[1] - distances[0] + 1e-12)
        nugget_guess = max(0.0, float(semivariances[0] - slope * distances[0]))
    else:
        nugget_guess = max(0.0, y_min * 0.8)
    
    # Sanity check
    if len(semivariances) >= 5:
        nugget_guess = min(nugget_guess, y_min * 1.2)
    
    # 2. Sill: estimate from plateau (mean of outer lags)
    n_outer = max(1, len(semivariances) // 3)
    sill_guess = float(np.mean(semivariances[-n_outer:]))
    sill_guess = max(sill_guess, y_max * 0.8)
    
    # Ensure sill > nugget with margin
    if sill_guess <= nugget_guess * 1.1:
        sill_guess = nugget_guess + (y_max - y_min) * 0.5 + 0.1
    
    # 3. Range: find where variogram approaches ~80% of sill
    target_gamma = nugget_guess + 0.8 * (sill_guess - nugget_guess)
    range_guess = x_max * 0.5
    for i, yi in enumerate(semivariances):
        if yi >= target_gamma:
            range_guess = float(distances[i])
            break
    range_guess = max(range_guess, x_max * 0.2)
    
    # Bounds
    r_lo = max(float(np.min(distances[distances > 0])) if np.any(distances > 0) else 1e-3, 1e-3)
    r_hi = x_max * 2.0
    
    # Sill bounds: strictly > nugget
    sill_lo = nugget_guess * 1.05 + 0.01
    sill_hi = max(y_max * 2.0, sill_guess * 2.0)
    nug_hi = sill_guess * 0.9
    
    try:
        # Fit the model with constrained bounds
        params, _ = curve_fit(
            variogram_func,
            distances,
            semivariances,
            p0=[nugget_guess, sill_guess, range_guess],
            bounds=([0, sill_lo, r_lo], [nug_hi, sill_hi, r_hi]),
            maxfev=10000
        )
        
        nugget, sill, range_ = params
        
        # Ensure sill > nugget
        if sill <= nugget:
            sill = nugget + 0.1
        
        return float(nugget), float(sill), float(range_)
    
    except Exception as e:
        # Fallback to initial guesses if fitting fails
        return float(nugget_guess), float(sill_guess), float(range_guess)


def calculate_experimental_variogram(
    coordinates: np.ndarray,
    values: np.ndarray,
    n_lags: int = 15,
    lag_tolerance: float = 0.5,
    lag_distance: Optional[float] = None,
    normalize: bool = False,
    max_samples: int = 5000,
    random_state: Optional[int] = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate experimental variogram from point data.

    ⚠️ PERFORMANCE OPTIMIZED: Uses subsampling and KDTree for large datasets.
    For datasets > max_samples, randomly subsamples to prevent O(N²) memory explosion.

    DETERMINISM: This function is fully deterministic when random_state is set.
    All random subsampling uses the same seeded RNG.

    Args:
        coordinates: (N, 3) array of (x, y, z) coordinates
        values: (N,) array of property values
        n_lags: Number of lag bins
        lag_tolerance: Tolerance factor for lag assignment (if lag_distance not given)
                       OR absolute tolerance in meters (if lag_distance is given)
        lag_distance: Optional explicit lag distance in meters
        normalize: If True, divide semivariance by data variance
        max_samples: Maximum number of samples to use (subsample if N > max_samples)
        random_state: Random seed for reproducibility (default 42)

    Returns:
        Tuple of (lag_distances, semivariances, pair_counts)
    """
    from scipy.spatial import cKDTree

    n_points = len(coordinates)

    # Create seeded RNG for all random operations - ensures determinism
    rng = np.random.default_rng(random_state)

    # CRITICAL FIX: Subsample if dataset is too large to prevent O(N²) memory explosion
    # For 10,000 points: pdist creates 100M distance matrix (800MB RAM)
    # For 50,000 points: pdist creates 2.5B distance matrix (20GB RAM) - CRASH!
    if n_points > max_samples:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(
            f"Dataset too large ({n_points} points) for full pairwise calculation. "
            f"Subsampling to {max_samples} random points (seed={random_state})."
        )
        # Random subsample without replacement - using seeded RNG
        idx = rng.choice(n_points, max_samples, replace=False)
        coordinates = coordinates[idx]
        values = values[idx]
        n_points = max_samples

    # Calculate maximum distance for lag binning
    # Use different strategies based on dataset size
    use_full_matrix = n_points <= 1000  # Only use full matrix for small datasets

    if use_full_matrix:
        # Small dataset: calculate full distance matrix (acceptable for N <= 1000)
        from scipy.spatial.distance import pdist, squareform
        distances = squareform(pdist(coordinates))
        max_dist = np.max(distances)
    else:
        # Large dataset: use KDTree-based sampling to estimate max distance
        # Sample 1000 random pairs to estimate max distance - using seeded RNG
        sample_size = min(1000, n_points)
        sample_idx = rng.choice(n_points, sample_size, replace=False)
        sample_coords = coordinates[sample_idx]
        sample_tree = cKDTree(sample_coords)
        sample_distances, _ = sample_tree.query(sample_coords, k=min(sample_size, 100))
        max_dist = np.max(sample_distances[sample_distances < np.inf]) * 2  # Conservative estimate

        # Build KDTree for the full dataset (for lag binning)
        tree = cKDTree(coordinates)
    
    # Use explicit lag_distance if provided, otherwise derive from max_dist
    if lag_distance is not None and lag_distance > 0:
        lag_size = lag_distance
        max_lag = min(max_dist, lag_size * n_lags)
        # Adjust n_lags if needed so we don't exceed max_dist
        n_lags = min(n_lags, int(max_dist / lag_size) + 1)
        n_lags = max(5, n_lags)  # At least 5 lags
        # If lag_tolerance is given as absolute (>1), use it directly
        tol = lag_tolerance if lag_tolerance > 1 else lag_size * lag_tolerance
    else:
        lag_size = max_dist / n_lags
        max_lag = max_dist
        tol = lag_size * lag_tolerance
    
    # Initialize arrays
    lag_distances = np.zeros(n_lags)
    semivariances = np.zeros(n_lags)
    pair_counts = np.zeros(n_lags, dtype=int)
    
    # OPTIMIZED: Use different strategies based on dataset size
    if use_full_matrix:
        # Small dataset: use pre-calculated full distance matrix
        # Calculate semivariance for each lag
        for i in range(n_lags):
            lag_center = (i + 0.5) * lag_size
            lag_min = max(0, lag_center - tol)
            lag_max = lag_center + tol
            
            # Find pairs within this lag bin
            mask = (distances >= lag_min) & (distances < lag_max)
            
            if np.any(mask):
                # Get value differences for pairs in this lag (vectorized)
                # Use upper triangle only to avoid double counting
                upper_mask = np.triu(mask, k=1)
                if np.any(upper_mask):
                    # Vectorized calculation for upper triangle
                    i_upper, j_upper = np.where(upper_mask)
                    value_diffs_sq = (values[i_upper] - values[j_upper]) ** 2
                    dists_upper = distances[upper_mask]
                    
                    # Calculate average semivariance
                    valid_pairs = len(value_diffs_sq)
                    if valid_pairs > 0:
                        semivariances[i] = 0.5 * np.mean(value_diffs_sq)
                        lag_distances[i] = np.mean(dists_upper)
                        pair_counts[i] = valid_pairs
    else:
        # Large dataset: use KDTree for efficient neighbor search
        # Query all points within max_lag distance
        max_search_radius = max_lag + tol
        
        for i in range(n_lags):
            lag_center = (i + 0.5) * lag_size
            lag_min = max(0, lag_center - tol)
            lag_max = lag_center + tol
            
            # Collect pairs for this lag bin
            pair_diffs_sq = []
            pair_distances = []
            
            # For each point, find neighbors within lag_max distance
            # Use a sampling approach for very large datasets
            sample_step = max(1, n_points // 1000)  # Sample every Nth point
            sampled_indices = np.arange(0, n_points, sample_step)
            
            for j in sampled_indices:
                # Find neighbors within lag_max
                neighbor_indices = tree.query_ball_point(
                    coordinates[j], 
                    r=lag_max,
                    p=2  # Euclidean distance
                )
                
                for k in neighbor_indices:
                    if k > j:  # Only count each pair once
                        dist = np.linalg.norm(coordinates[j] - coordinates[k])
                        if lag_min <= dist < lag_max:
                            pair_distances.append(dist)
                            pair_diffs_sq.append((values[j] - values[k]) ** 2)
            
            # Calculate statistics for this lag
            if len(pair_diffs_sq) > 0:
                pair_distances = np.array(pair_distances)
                pair_diffs_sq = np.array(pair_diffs_sq)
                
                # Scale pair count to account for sampling
                actual_pair_count = len(pair_diffs_sq) * (sample_step ** 2)
                semivariances[i] = 0.5 * np.mean(pair_diffs_sq)
                lag_distances[i] = np.mean(pair_distances)
                pair_counts[i] = int(actual_pair_count)
    
    # Filter out lags with no pairs
    valid_lags = pair_counts > 0
    
    result_lags = lag_distances[valid_lags]
    result_gamma = semivariances[valid_lags]
    result_counts = pair_counts[valid_lags]
    
    # Apply normalization if requested
    if normalize and len(result_gamma) > 0:
        data_variance = np.var(values)
        if data_variance > 0:
            result_gamma = result_gamma / data_variance
    
    return result_lags, result_gamma, result_counts


def anisotropic_distance(
    dx: np.ndarray,
    dy: np.ndarray,
    dz: np.ndarray,
    range_x: float,
    range_y: float,
    range_z: float
) -> np.ndarray:
    """
    Calculate anisotropic distance using range ratios.
    
    Args:
        dx, dy, dz: Coordinate differences
        range_x, range_y, range_z: Range parameters in each direction
    
    Returns:
        Anisotropic distances
    """
    # Normalize by ranges to account for anisotropy
    scaled_dx = dx / range_x if range_x > 0 else dx
    scaled_dy = dy / range_y if range_y > 0 else dy
    scaled_dz = dz / range_z if range_z > 0 else dz
    
    # Calculate Euclidean distance in scaled space
    distance = np.sqrt(scaled_dx**2 + scaled_dy**2 + scaled_dz**2)
    
    return distance


# ============================================================================
# STEP 22: Variogram utilities for UK/CoK/IK
# ============================================================================

def compute_residuals_for_drift(
    values: np.ndarray,
    coords: np.ndarray,
    drift_model: Any
) -> np.ndarray:
    """
    Compute residuals after removing drift for Universal Kriging variogram fitting.
    
    Args:
        values: (N,) data values
        coords: (N, 3) coordinates
        drift_model: DriftModel instance or drift basis callable
    
    Returns:
        (N,) residual values
    """
    from ..geostats.universal_kriging import DriftModel
    
    if not isinstance(drift_model, DriftModel):
        drift_model = DriftModel(drift_model)
    
    # Compute design matrix
    F = drift_model.design_matrix(coords)
    
    # Fit drift coefficients using least squares
    # F * beta = values
    beta, _, _, _ = np.linalg.lstsq(F, values, rcond=None)
    
    # Compute drift
    drift = F @ beta
    
    # Residuals
    residuals = values - drift
    
    return residuals


def compute_cross_variogram(
    var1: np.ndarray,
    var2: np.ndarray,
    coords: np.ndarray,
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Compute experimental cross-variogram for Co-Kriging.

    Cross-variogram: γ_12(h) = 0.5 * E[(Z_1(x) - Z_1(x+h)) * (Z_2(x) - Z_2(x+h))]

    ⚠️ PERFORMANCE OPTIMIZED: Uses subsampling for large datasets.

    DETERMINISM: This function is fully deterministic when random_state is set
    in params. All random subsampling uses the same seeded RNG.

    Args:
        var1: (N,) primary variable values
        var2: (N,) secondary variable values
        coords: (N, 3) coordinates
        params: Parameters dict (n_lags, max_range, lag_tolerance, max_samples, random_state)

    Returns:
        Dict with experimental variogram data
    """
    from scipy.spatial import cKDTree

    n_points = len(coords)
    max_samples = params.get('max_samples', 5000)
    random_state = params.get('random_state', 42)

    # Create seeded RNG for all random operations - ensures determinism
    rng = np.random.default_rng(random_state)

    # CRITICAL FIX: Subsample if dataset is too large
    if n_points > max_samples:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(
            f"Dataset too large ({n_points} points) for cross-variogram. "
            f"Subsampling to {max_samples} random points (seed={random_state})."
        )
        idx = rng.choice(n_points, max_samples, replace=False)
        coords = coords[idx]
        var1 = var1[idx]
        var2 = var2[idx]
        n_points = max_samples

    n_lags = params.get('n_lags', 15)
    lag_tolerance = params.get('lag_tolerance', 0.5)
    max_range = params.get('max_range', None)

    # Use KDTree for efficient distance calculation
    tree = cKDTree(coords)

    # Estimate max distance - using seeded RNG
    if max_range is None:
        if n_points > 1000:
            # Sample-based estimate
            sample_size = min(1000, n_points)
            sample_idx = rng.choice(n_points, sample_size, replace=False)
            sample_coords = coords[sample_idx]
            sample_tree = cKDTree(sample_coords)
            sample_distances, _ = sample_tree.query(sample_coords, k=min(sample_size, 100))
            max_dist = np.max(sample_distances[sample_distances < np.inf]) * 2
        else:
            # Small dataset: calculate full distance matrix
            from scipy.spatial.distance import pdist, squareform
            distances = squareform(pdist(coords))
            max_dist = np.max(distances)
    else:
        max_dist = max_range
        # For small datasets, still calculate full matrix if needed
        if n_points <= 1000:
            from scipy.spatial.distance import pdist, squareform
            distances = squareform(pdist(coords))
    
    # Calculate maximum distance
    if max_range is None:
        max_dist = np.max(distances)
    else:
        max_dist = max_range
    
    lag_size = max_dist / n_lags
    tol = lag_size * lag_tolerance
    
    # Initialize arrays
    lag_distances = np.zeros(n_lags)
    cross_semivariances = np.zeros(n_lags)
    pair_counts = np.zeros(n_lags, dtype=int)
    
    # Calculate cross-semivariance for each lag
    if n_points <= 1000 and 'distances' in locals():
        # Small dataset: use pre-calculated distance matrix
        for i in range(n_lags):
            lag_min = i * lag_size
            lag_max = (i + 1) * lag_size * (1 + lag_tolerance)
            
            # Find pairs within this lag bin
            mask = (distances >= lag_min) & (distances < lag_max)
            
            if np.any(mask):
                # Calculate cross-semivariance: 0.5 * E[(Z1(x) - Z1(x+h)) * (Z2(x) - Z2(x+h))]
                cross_products = np.zeros(distances.shape)
                
                for j in range(n_points):
                    for k in range(j + 1, n_points):
                        if mask[j, k]:
                            diff1 = var1[j] - var1[k]
                            diff2 = var2[j] - var2[k]
                            cross_products[j, k] = diff1 * diff2
                
                # Calculate average cross-semivariance
                valid_pairs = mask.sum() // 2
                if valid_pairs > 0:
                    cross_semivariances[i] = 0.5 * cross_products[mask].sum() / valid_pairs
                    lag_distances[i] = distances[mask].mean()
                    pair_counts[i] = valid_pairs
    else:
        # Large dataset: use KDTree for efficient neighbor search
        max_search_radius = max_dist + tol
        
        for i in range(n_lags):
            lag_min = i * lag_size
            lag_max = (i + 1) * lag_size * (1 + lag_tolerance)
            
            # Collect pairs for this lag bin using KDTree
            cross_products_list = []
            pair_distances_list = []
            
            # Sample points for efficiency
            sample_step = max(1, n_points // 1000)
            sampled_indices = np.arange(0, n_points, sample_step)
            
            for j in sampled_indices:
                neighbor_indices = tree.query_ball_point(
                    coords[j],
                    r=lag_max,
                    p=2
                )
                
                for k in neighbor_indices:
                    if k > j:
                        dist = np.linalg.norm(coords[j] - coords[k])
                        if lag_min <= dist < lag_max:
                            diff1 = var1[j] - var1[k]
                            diff2 = var2[j] - var2[k]
                            cross_products_list.append(diff1 * diff2)
                            pair_distances_list.append(dist)
            
            if len(cross_products_list) > 0:
                # Scale pair count to account for sampling
                actual_pair_count = len(cross_products_list) * (sample_step ** 2)
                cross_semivariances[i] = 0.5 * np.mean(cross_products_list)
                lag_distances[i] = np.mean(pair_distances_list)
                pair_counts[i] = int(actual_pair_count)
    
    # Filter out lags with no pairs
    valid_lags = pair_counts > 0
    
    return {
        'lag_distances': lag_distances[valid_lags],
        'cross_semivariances': cross_semivariances[valid_lags],
        'pair_counts': pair_counts[valid_lags],
        'n_lags': n_lags,
        'max_range': max_dist
    }


def compute_indicator_variograms(
    values: np.ndarray,
    thresholds: List[float],
    coords: np.ndarray,
    params: Dict[str, Any]
) -> Dict[float, Dict[str, Any]]:
    """
    Compute experimental variograms for binary indicator variables.

    DETERMINISM: This function is fully deterministic when random_state is set
    in params. All random subsampling uses the same seeded RNG.

    Args:
        values: (N,) data values
        thresholds: List of threshold values
        coords: (N, 3) coordinates
        params: Parameters dict (n_lags, max_range, lag_tolerance, random_state)

    Returns:
        Dict mapping threshold -> experimental variogram dict
    """
    result = {}
    random_state = params.get('random_state', 42)

    for threshold in thresholds:
        # Create binary indicator
        indicator = (values <= threshold).astype(float)

        # Compute experimental variogram for indicator - with seed
        lag_distances, semivariances, pair_counts = calculate_experimental_variogram(
            coords, indicator,
            n_lags=params.get('n_lags', 15),
            lag_tolerance=params.get('lag_tolerance', 0.5),
            random_state=random_state
        )

        result[threshold] = {
            'lag_distances': lag_distances,
            'semivariances': semivariances,
            'pair_counts': pair_counts,
            'threshold': threshold,
            'indicator_mean': np.mean(indicator)
        }

    return result


# ============================================================================
# STEP 23: Variogram Assistant Helpers
# ============================================================================

def build_candidate_models(
    experimental: Dict[str, Any],
    model_families: List[str] = None,
    max_structures: int = 2
) -> List[Dict[str, Any]]:
    """
    Build initial candidate variogram models from experimental data.
    
    This is a wrapper that delegates to variogram_assistant.build_candidate_models
    but returns dicts instead of VariogramCandidateModel objects for compatibility.
    
    Args:
        experimental: Experimental variogram dict with 'lag_distances' and 'semivariances'
        model_families: List of model types to try (default: ['spherical', 'exponential', 'gaussian'])
        max_structures: Maximum number of nested structures (default: 2)
    
    Returns:
        List of candidate model dicts
    """
    from ..geostats.variogram_assistant import build_candidate_models as build_candidates
    
    candidates = build_candidates(experimental, model_families, max_structures)
    
    # Convert to dicts
    return [
        {
            'model_type': c.model_type,
            'ranges': c.ranges,
            'sills': c.sills,
            'nugget': c.nugget,
            'anisotropy': c.anisotropy,
            'metadata': c.metadata
        }
        for c in candidates
    ]


def evaluate_variogram_model(
    experimental: Dict[str, Any],
    model: Dict[str, Any]
) -> Dict[str, float]:
    """
    Evaluate a variogram model against experimental data.
    
    Args:
        experimental: Experimental variogram dict
        model: Model dict with 'model_type', 'ranges', 'sills', 'nugget'
    
    Returns:
        Dict with metrics: 'sse', 'rmse', 'mae', 'r2'
    """
    from ..geostats.variogram_assistant import (
        VariogramCandidateModel,
        evaluate_variogram_model as eval_model
    )
    
    # Convert dict to VariogramCandidateModel
    candidate = VariogramCandidateModel(
        model_type=model.get('model_type', 'spherical'),
        ranges=model.get('ranges', [100.0]),
        sills=model.get('sills', [1.0]),
        nugget=model.get('nugget', 0.0),
        anisotropy=model.get('anisotropy', {})
    )
    
    return eval_model(experimental, candidate)


def cross_validate_variogram(
    coords: np.ndarray,
    values: np.ndarray,
    model: Dict[str, Any],
    method: str = "OK",
    n_folds: int = 5
) -> Dict[str, float]:
    """
    Perform cross-validation of a variogram model using kriging.
    
    Args:
        coords: (N, 3) data coordinates
        values: (N,) data values
        model: Model dict with variogram parameters
        method: Kriging method ("OK" or "UK")
        n_folds: Number of folds for cross-validation (or -1 for LOOCV)
    
    Returns:
        Dict with 'rmse', 'mae', 'bias', 'correlation'
    """
    from ..geostats.variogram_assistant import (
        VariogramCandidateModel,
        cross_validate_variogram as cv_variogram
    )
    
    # Convert dict to VariogramCandidateModel
    candidate = VariogramCandidateModel(
        model_type=model.get('model_type', 'spherical'),
        ranges=model.get('ranges', [100.0]),
        sills=model.get('sills', [1.0]),
        nugget=model.get('nugget', 0.0),
        anisotropy=model.get('anisotropy', {})
    )
    
    return cv_variogram(coords, values, candidate, method=method, n_folds=n_folds)











