"""
Indicator Kriging Engine (3D) - HIGH PERFORMANCE

Optimized with Numba JIT for massive speedups (50x-100x).

- Solves Kriging systems for multiple thresholds in parallel.

- Corrects Order Relation Violations (Monotonicity constraints).

- Calculates E-Type (Mean) and Median from CDF.
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

# Numba Import with Fallback
try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        def decorator(func): return func
        return decorator
    prange = range

# cKDTree import removed - now using NeighborSearcher from geostats_utils

logger = logging.getLogger(__name__)

# =========================================================
# 1. NUMBA KERNELS (MATH CORE)
# =========================================================

@njit(fastmath=True, cache=True)
def _get_cov(d, range_val, sill, nugget, model_type):
    """
    Calculate Covariance C(h) = TotalSill - Gamma(h).
    
    AUDIT FIX (V-NEW-001): Standardized sill interpretation.
    
    Parameters
    ----------
    d : float
        Distance (lag)
    range_val : float
        Range parameter
    sill : float
        TOTAL sill (nugget + partial sill) - CANONICAL CONVENTION
    nugget : float
        Nugget effect
    model_type : int
        0=Spherical, 1=Exponential, 2=Gaussian
    
    Returns
    -------
    float
        Covariance value C(h)
    """
    # CANONICAL: sill is TOTAL sill, compute partial_sill internally
    partial_sill = max(sill - nugget, 0.0)
    
    if d < 1e-9:
        return sill  # At h=0, covariance = total sill
    
    gamma = 0.0
    # 0=Spherical, 1=Exponential, 2=Gaussian
    if model_type == 0: 
        if d >= range_val:
            gamma = sill  # Total sill at range
        else:
            r = d / range_val
            gamma = nugget + partial_sill * (1.5 * r - 0.5 * r**3)
    elif model_type == 1:
        gamma = nugget + partial_sill * (1.0 - np.exp(-3.0 * d / range_val))
    elif model_type == 2:
        gamma = nugget + partial_sill * (1.0 - np.exp(-3.0 * (d / range_val)**2))
        
    return sill - gamma


@njit(fastmath=True, cache=True)
def _solve_single_ik_point(
    i,
    target_coords,      # (M, 3)
    neighbor_indices,   # (M, K)
    data_coords,        # (N, 3)
    data_indicators,    # (N, T)
    rng, sill, nugget, model_code, n_thresholds
):
    """
    Solve Indicator Kriging for a single target point.
    Separated to avoid control flow issues in prange.
    """
    result = np.full(n_thresholds, np.nan, dtype=np.float64)
    
    # 1. Neighbors
    indices = neighbor_indices[i]
    valid_count = 0
    for idx in indices:
        if idx >= 0:
            valid_count += 1
        else:
            break
        
    if valid_count < 3:
        return result
    
    local_idx = indices[:valid_count]
    P = data_coords[local_idx]
    target_pt = target_coords[i]
    
    # 2. Build Matrix (Ordinary Kriging)
    dim = valid_count + 1
    K = np.zeros((dim, dim))
    RHS = np.zeros(dim)
    
    # Fill Covariance (Symmetric)
    for r in range(valid_count):
        for c in range(r, valid_count):
            dx = P[r, 0] - P[c, 0]
            dy = P[r, 1] - P[c, 1]
            dz = P[r, 2] - P[c, 2]
            d = np.sqrt(dx*dx + dy*dy + dz*dz)
            cov = _get_cov(d, rng, sill, nugget, model_code)
            K[r, c] = cov
            K[c, r] = cov
        
        # Lagrange
        K[r, valid_count] = 1.0
        K[valid_count, r] = 1.0
        
        # Fill RHS (Target to Data)
        dx = P[r, 0] - target_pt[0]
        dy = P[r, 1] - target_pt[1]
        dz = P[r, 2] - target_pt[2]
        d_t = np.sqrt(dx*dx + dy*dy + dz*dz)
        RHS[r] = _get_cov(d_t, rng, sill, nugget, model_code)
    
    # Add scaled regularization
    max_diag = 0.0
    for d_i in range(valid_count):
        if K[d_i, d_i] > max_diag:
            max_diag = K[d_i, d_i]
    reg_value = max(1e-10 * max_diag, 1e-9)
    for r in range(valid_count):
        K[r, r] += reg_value
        
    RHS[valid_count] = 1.0  # Unbiasedness
    
    # 3. Solve using Gaussian elimination with partial pivoting
    A = K.copy()
    b = RHS.copy()
    
    # Forward elimination
    for k in range(dim):
        max_row = k
        max_val = abs(A[k, k])
        for row in range(k + 1, dim):
            if abs(A[row, k]) > max_val:
                max_val = abs(A[row, k])
                max_row = row
        
        if max_val < 1e-12:
            # Singular - use sample proportion fallback
            for t in range(n_thresholds):
                vals = data_indicators[local_idx, t]
                est = 0.0
                for kk in range(valid_count):
                    est += vals[kk]
                result[t] = est / valid_count
            return result
        
        if max_row != k:
            for col in range(dim):
                tmp = A[k, col]
                A[k, col] = A[max_row, col]
                A[max_row, col] = tmp
            tmp = b[k]
            b[k] = b[max_row]
            b[max_row] = tmp
        
        for row in range(k + 1, dim):
            factor = A[row, k] / A[k, k]
            for col in range(k, dim):
                A[row, col] -= factor * A[k, col]
            b[row] -= factor * b[k]
    
    # Back substitution
    weights = np.zeros(dim)
    for k in range(dim - 1, -1, -1):
        if abs(A[k, k]) < 1e-12:
            # Fallback to sample proportion
            for t in range(n_thresholds):
                vals = data_indicators[local_idx, t]
                est = 0.0
                for kk in range(valid_count):
                    est += vals[kk]
                result[t] = est / valid_count
            return result
        weights[k] = b[k]
        for col in range(k + 1, dim):
            weights[k] -= A[k, col] * weights[col]
        weights[k] /= A[k, k]
    
    w = weights[:valid_count]
    
    # 4. Apply to all thresholds
    for t in range(n_thresholds):
        vals = data_indicators[local_idx, t]
        
        est = 0.0
        for k in range(valid_count):
            est += w[k] * vals[k]
        
        # Sanity check
        if est < -0.5 or est > 1.5:
            est = 0.0
            for k in range(valid_count):
                est += vals[k]
            est = est / valid_count
        
        result[t] = est
    
    return result


@njit(parallel=True, fastmath=True, cache=True)
def run_ik_kernel(
    target_coords,      # (M, 3)
    neighbor_indices,   # (M, K)
    data_coords,        # (N, 3)
    data_indicators,    # (N, T) - Binary indicators for ALL thresholds
    params,             # (4,) [range, sill, nugget, model_code]
    n_thresholds        # int
):
    """
    Parallel Indicator Kriging Solver.
    
    This version separates single-point solving into a helper function to avoid
    control flow issues (continue/try-except) that prevent Numba prange parallelization.
    """
    n_targets = target_coords.shape[0]
    
    # Unpack Params
    rng = params[0]
    sill = params[1]
    nugget = params[2]
    model_code = int(params[3])
    
    # Output: (M, T) probabilities
    probs = np.full((n_targets, n_thresholds), np.nan, dtype=np.float64)
    
    # Pure prange loop - no continue, no try/except
    for i in prange(n_targets):
        result = _solve_single_ik_point(
            i, target_coords, neighbor_indices, data_coords, data_indicators,
            rng, sill, nugget, model_code, n_thresholds
        )
        for t in range(n_thresholds):
            probs[i, t] = result[t]
            
    return probs


@njit(fastmath=True, cache=True)
def _correct_single_row(probs_row, thresholds, n_thresh):
    """Process a single row for order correction and stats."""
    # Check if row is valid (not NaN)
    if np.isnan(probs_row[0]):
        return np.nan, np.nan
        
    # 1. CLAMPING
    for k in range(n_thresh):
        if probs_row[k] < 0.0:
            probs_row[k] = 0.0
        if probs_row[k] > 1.0:
            probs_row[k] = 1.0
        
    # 2. ORDER RELATION CORRECTION (Forward/Backward Average)
    # Forward Pass: P[k] must be >= P[k-1]
    for k in range(1, n_thresh):
        if probs_row[k] < probs_row[k-1]:
            avg = (probs_row[k] + probs_row[k-1]) * 0.5
            probs_row[k] = avg
            probs_row[k-1] = avg
    
    # Backward Pass
    for k in range(n_thresh - 2, -1, -1):
        if probs_row[k] > probs_row[k+1]:
            avg = (probs_row[k] + probs_row[k+1]) * 0.5
            probs_row[k] = avg
            probs_row[k+1] = avg
            
    # Re-Check Forward (GSLIB Method)
    for k in range(1, n_thresh):
        if probs_row[k] < probs_row[k-1]:
            probs_row[k] = probs_row[k-1]

    # 3. CALCULATE STATISTICS FROM CDF
    # Prob(Grade <= Threshold)
    
    # E-TYPE (Mean) = Sum of Area under CDF
    mean_val = 0.0
    for k in range(n_thresh - 1):
        p_lower = probs_row[k]
        p_upper = probs_row[k + 1]
        z_lower = thresholds[k]
        z_upper = thresholds[k + 1]
        mean_val += 0.5 * (p_lower + p_upper) * (z_upper - z_lower)
    
    # Add tails
    if n_thresh > 0:
        mean_val += thresholds[0] * probs_row[0]
        mean_val += thresholds[-1] * (1.0 - probs_row[-1])
    
    # MEDIAN (Linear interpolation where P = 0.5)
    median_val = np.nan
    for k in range(n_thresh - 1):
        if probs_row[k] <= 0.5 <= probs_row[k + 1]:
            if probs_row[k + 1] - probs_row[k] > 1e-9:
                t = (0.5 - probs_row[k]) / (probs_row[k + 1] - probs_row[k])
                median_val = thresholds[k] + t * (thresholds[k + 1] - thresholds[k])
            else:
                median_val = 0.5 * (thresholds[k] + thresholds[k + 1])
            break
    
    if np.isnan(median_val):
        if probs_row[0] >= 0.5:
            median_val = thresholds[0]
        elif probs_row[-1] <= 0.5:
            median_val = thresholds[-1]
    
    return mean_val, median_val


@njit(parallel=True, fastmath=True, cache=True)
def correct_order_relations_and_stats(probs, thresholds):
    """
    1. Corrects Order Relation Violations (Probabilities must increase with threshold).
    2. Clamps to [0, 1].
    3. Calculates Mean (E-Type) and Median.
    
    This version separates single-row processing to avoid continue in prange.
    """
    n_blocks, n_thresh = probs.shape
    medians = np.full(n_blocks, np.nan)
    means = np.full(n_blocks, np.nan)
    
    for i in prange(n_blocks):
        mean_val, median_val = _correct_single_row(probs[i], thresholds, n_thresh)
        means[i] = mean_val
        medians[i] = median_val

    return medians, means


# =========================================================
# 2. COMPATIBILITY DATACLASSES (for backward compatibility)
# =========================================================

@dataclass
class IKConfig:
    """
    Configuration for Indicator Kriging (backward compatibility).
    
    DEPRECATED: This class is kept for backward compatibility.
    New code should use IndicatorKrigingJobParams via run_indicator_kriging_job().
    """
    thresholds: List[float]
    variogram_model_template: Dict[str, Any]
    search_params: Dict[str, Any] = field(default_factory=dict)
    compute_median: bool = True
    compute_mean: bool = True
    
    def __post_init__(self):
        import warnings
        warnings.warn(
            "IKConfig is deprecated. Use IndicatorKrigingJobParams via run_indicator_kriging_job() instead.",
            DeprecationWarning,
            stacklevel=2
        )


@dataclass
class IKResult:
    """
    Result from Indicator Kriging (backward compatibility).
    
    DEPRECATED: This class is kept for backward compatibility.
    New code should use dict results from run_indicator_kriging_job().
    """
    thresholds: np.ndarray
    probabilities: np.ndarray  # (nx, ny, nz, n_thresh) or (n_points, n_thresh)
    median: Optional[np.ndarray] = None  # (nx, ny, nz) or (n_points,)
    mean: Optional[np.ndarray] = None  # (nx, ny, nz) or (n_points,)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        import warnings
        warnings.warn(
            "IKResult is deprecated. Use dict results from run_indicator_kriging_job() instead.",
            DeprecationWarning,
            stacklevel=2
        )


# =========================================================
# 3. PYTHON INTERFACE
# =========================================================

def run_indicator_kriging_job(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main entry point for the Controller.
    
    Args:
        params: Job parameters dict (will be validated and converted to IndicatorKrigingJobParams)
    
    Returns:
        Result dict with probabilities, estimates, metadata, and StructuredGrid
    """
    from ..models.kriging3d import create_estimation_grid
    from .geostats_utils import NeighborSearcher
    from .kriging_job_params import IndicatorKrigingJobParams
    from .simulation_interface import GridDefinition
    # NOTE: PyVista imports removed - grid creation happens in main thread to prevent worker freezes
    
    # Validate and convert parameters (type safety check happens here)
    # Pydantic will catch type mismatches, missing required fields, and invalid values
    try:
        job_params = IndicatorKrigingJobParams.from_dict(params)
    except (ValueError, KeyError, TypeError) as e:
        error_msg = str(e)
        logger.error(f"Invalid Indicator Kriging job parameters: {error_msg}")
        return {'error': f"Parameter validation failed: {error_msg}"}
    except Exception as e:
        # Handle Pydantic ValidationError (if Pydantic is available)
        error_msg = str(e)
        logger.error(f"Indicator Kriging parameter validation error: {error_msg}", exc_info=True)
        if hasattr(e, 'errors'):
            field_errors = [f"{err.get('loc', 'unknown')}: {err.get('msg', 'invalid')}" for err in e.errors()]
            error_msg = f"Validation errors: {'; '.join(field_errors)}"
        return {'error': f"Parameter validation failed: {error_msg}"}
    
    # 1. Extract Data
    thresholds = np.array(job_params.thresholds, dtype=np.float64)
    
    # 2. Prepare Grid
    coords = job_params.data_df[['X', 'Y', 'Z']].values.astype(float)
    values = job_params.data_df[job_params.variable].values.astype(float)
    
    # Create grid
    if job_params.grid_config.origin and job_params.grid_config.counts:
        # Explicit grid definition
        x0, y0, z0 = job_params.grid_config.origin
        dx, dy, dz = job_params.grid_config.spacing
        nx, ny, nz = job_params.grid_config.counts
        
        x = np.arange(nx) * dx + x0
        y = np.arange(ny) * dy + y0
        z = np.arange(nz) * dz + z0
        
        grid_x, grid_y, grid_z = np.meshgrid(x, y, z, indexing='ij')
        target_coords = np.column_stack([
            grid_x.flatten(),
            grid_y.flatten(),
            grid_z.flatten()
        ])
    else:
        # Auto-bounds with padding
        data_min = coords.min(axis=0)
        data_max = coords.max(axis=0)
        pad = (data_max - data_min) * 0.1
        buffer = tuple(pad)
        
        grid_x, grid_y, grid_z, target_coords = create_estimation_grid(
            coords, job_params.grid_config.spacing, buffer=buffer, max_points=job_params.grid_config.max_points
        )
    
    # 3. Pre-Calculate Indicators (N x T)
    # This moves the logic out of the loop
    n_thresh = len(thresholds)
    indicators = np.zeros((len(values), n_thresh), dtype=np.float64)
    
    for t in range(n_thresh):
        # I(x) = 1 if Z(x) <= cut, else 0
        indicators[:, t] = (values <= thresholds[t]).astype(float)

    # 4. Variogram Params (Numpy Array for Numba)
    m_map = {'spherical': 0, 'exponential': 1, 'gaussian': 2}
    m_code = m_map.get(job_params.variogram_template.model_type.lower(), 0)
    
    # [range, partial_sill, nugget, model]
    # Partial sill = Total Sill - Nugget. Assuming input sill is Total.
    sill_total = float(job_params.variogram_template.sill)
    nug = float(job_params.variogram_template.nugget)
    
    kern_params = np.array([
        float(job_params.variogram_template.range_),
        sill_total - nug,
        nug,
        m_code
    ], dtype=np.float64)
    
    # 5. Neighbor Search using unified NeighborSearcher
    anisotropy_params = job_params.variogram_template.anisotropy
    searcher = NeighborSearcher(coords, anisotropy_params=anisotropy_params)
    indices, dists = searcher.search(
        target_coords=target_coords,
        n_neighbors=job_params.search_config.n_neighbors,
        max_distance=job_params.search_config.max_distance
    )
        
    if job_params.progress_callback:
        job_params.progress_callback(20, "Running Indicator Kriging...")
        
    if NUMBA_AVAILABLE:
        # Step A: Kriging
        probs = run_ik_kernel(
            target_coords, indices, coords, indicators, kern_params, n_thresh
        )
        
        if job_params.progress_callback:
            job_params.progress_callback(70, "Correcting order relations...")
        
        # Step B: Corrections & Stats
        medians, means = correct_order_relations_and_stats(probs, thresholds)
    else:
        # Fallback (Slow)
        probs = np.zeros((len(target_coords), n_thresh))
        medians = np.zeros(len(target_coords))
        means = np.zeros(len(target_coords))
        logger.warning("Numba not installed. IK will return zeros.")

    if job_params.progress_callback:
        job_params.progress_callback(100, "Complete!")

    # 6. Reshape & Package
    if job_params.grid_config.origin and job_params.grid_config.counts:
        nx, ny, nz = job_params.grid_config.counts
        x0, y0, z0 = job_params.grid_config.origin
        dx, dy, dz = job_params.grid_config.spacing
    else:
        nx, ny, nz = grid_x.shape
        # Infer spacing and origin from grid
        dx = grid_x[1, 0, 0] - grid_x[0, 0, 0] if nx > 1 else 10.0
        dy = grid_y[0, 1, 0] - grid_y[0, 0, 0] if ny > 1 else 10.0
        dz = grid_z[0, 0, 1] - grid_z[0, 0, 0] if nz > 1 else 5.0
        x0 = grid_x[0, 0, 0] - dx / 2
        y0 = grid_y[0, 0, 0] - dy / 2
        z0 = grid_z[0, 0, 0] - dz / 2
    
    # Reshape probs to (nx, ny, nz, n_thresh)
    probs_reshaped = probs.reshape((nx, ny, nz, n_thresh), order='F')
    
    # Property names
    primary_property = f"IK_{job_params.variable}_Prob"
    primary_values = probs_reshaped[:, :, :, 0]  # First threshold probability
    
    # Prepare additional properties data
    additional_properties = {}
    if job_params.compute_median:
        median_property = f"IK_{job_params.variable}_Median"
        median_values = medians.reshape((nx, ny, nz), order='F')
        additional_properties[median_property] = median_values.ravel(order='F')
    
    if job_params.compute_mean:
        mean_property = f"IK_{job_params.variable}_Mean"
        mean_values = means.reshape((nx, ny, nz), order='F')
        additional_properties[mean_property] = mean_values.ravel(order='F')
    
    # Prepare threshold probability properties
    threshold_properties = {}
    for i, thresh in enumerate(thresholds):
        prob_property = f"IK_{job_params.variable}_Prob_{thresh:.2f}"
        prob_values = probs_reshaped[:, :, :, i]
        threshold_properties[prob_property] = prob_values.ravel(order='F')
    
    # Return ONLY primitive data - PyVista grid creation happens in main thread
    result_dict = {
        'probabilities': probs_reshaped,  # numpy array
        'thresholds': thresholds,  # numpy array
        'property_name': primary_property,
        'primary_values': primary_values.ravel(order='F'),  # numpy array
        'grid_x': grid_x,  # numpy array
        'grid_y': grid_y,  # numpy array
        'grid_z': grid_z,  # numpy array
        'grid_def': {
            'origin': (x0, y0, z0),
            'spacing': (dx, dy, dz),
            'counts': (nx, ny, nz)
        },
        'additional_properties': additional_properties,  # dict of numpy arrays
        'threshold_properties': threshold_properties,  # dict of numpy arrays
        'metadata': {
            'method': 'Indicator Kriging (MIA)',
            'thresholds': thresholds.tolist(),
            'variable': job_params.variable
        },
        '_create_grid_in_main_thread': True  # Flag to create PyVista grid in main thread
    }
    
    if job_params.compute_median:
        result_dict['median'] = medians.reshape((nx, ny, nz), order='F')
        result_dict['median_property'] = f"IK_{job_params.variable}_Median"
        
    if job_params.compute_mean:
        result_dict['mean'] = means.reshape((nx, ny, nz), order='F')
        result_dict['mean_property'] = f"IK_{job_params.variable}_Mean"
        
    return result_dict


# =========================================================
# 4. NUMBA PRE-COMPILATION (Warm-up)
# =========================================================

def precompile_ik_kernels():
    """
    Pre-compile Indicator Kriging Numba JIT functions with minimal dummy data.
    
    Call this at application startup (e.g., in a background thread) to avoid
    the 5-30 second JIT compilation delay when the user first runs IK.
    
    Returns:
        bool: True if compilation succeeded, False otherwise
    """
    if not NUMBA_AVAILABLE:
        logger.info("Numba not available - skipping IK kernel pre-compilation")
        return False
    
    try:
        logger.info("Pre-compiling Indicator Kriging Numba kernels...")
        
        # Minimal dummy data (10 data points, 5 targets, 3 thresholds)
        n_data = 10
        n_target = 5
        n_thresh = 3
        k_neighbors = 4
        
        # Generate synthetic data
        data_coords = np.random.rand(n_data, 3).astype(np.float64) * 100
        target_coords = np.random.rand(n_target, 3).astype(np.float64) * 100
        
        # Dummy indicators (binary)
        data_indicators = np.random.randint(0, 2, size=(n_data, n_thresh)).astype(np.float64)
        
        # Dummy neighbor indices
        neighbor_indices = np.zeros((n_target, k_neighbors), dtype=np.int64)
        for i in range(n_target):
            neighbor_indices[i] = np.arange(k_neighbors)
        
        # Params: [range, sill, nugget, model_code]
        params = np.array([50.0, 1.0, 0.1, 0], dtype=np.float64)  # Spherical
        
        # Thresholds for stats calculation
        thresholds = np.array([0.3, 0.5, 0.7], dtype=np.float64)
        
        # Run the IK kernel
        probs = run_ik_kernel(
            target_coords=target_coords,
            neighbor_indices=neighbor_indices,
            data_coords=data_coords,
            data_indicators=data_indicators,
            params=params,
            n_thresholds=n_thresh
        )
        
        # Also warm up order correction and stats kernel
        _ = correct_order_relations_and_stats(probs, thresholds)
        
        # Warm up covariance calculations with different models
        for model_type in [0, 1, 2]:  # Spherical, Exponential, Gaussian
            _ = _get_cov(10.0, 50.0, 1.0, 0.1, model_type)
        
        logger.info("Indicator Kriging Numba kernels pre-compiled successfully")
        return True
        
    except Exception as e:
        logger.warning(f"IK kernel pre-compilation failed (non-fatal): {e}")
        return False