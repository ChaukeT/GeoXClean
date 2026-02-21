"""
Universal Kriging Engine (3D) - HIGH PERFORMANCE VERSION

Optimized with Numba JIT compilation for massive speedups (50x-100x).

Supports Constant, Linear, and Quadratic drift models.
"""

import logging
from typing import Callable, Optional, Tuple, Dict, Any
import numpy as np

# Try importing Numba
try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Dummy decorator if Numba is missing
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range

# cKDTree import removed - now using NeighborSearcher from geostats_utils

from ..models.kriging3d import (
    apply_anisotropy,
    get_variogram_function,
)
from .geostats_utils import NeighborSearcher
from .uk_validation_utils import DriftFitter, UKPostProcessor, ResidualVariogramCalculator
from ..models.geostat_results import UniversalKrigingResults

logger = logging.getLogger(__name__)

# =========================================================
# 1. NUMBA KERNELS (The Speed Engine)
# =========================================================


@njit(fastmath=True, cache=True)
def _calc_drift_row(x, y, z, drift_code):
    """
    Computes drift vector for a single point.
    drift_code: 0=Constant, 1=Linear, 2=Quadratic
    """
    if drift_code == 0:   # Constant
        return np.array([1.0])
    
    elif drift_code == 1: # Linear
        return np.array([1.0, x, y, z])
    
    elif drift_code == 2: # Quadratic
        return np.array([
            1.0, x, y, z, 
            x*x, y*y, z*z, 
            x*y, x*z, y*z
        ])
    return np.array([1.0])


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
    if model_type == 0:  # Spherical
        if d >= range_val:
            gamma = sill  # Total sill at range
        else:
            r = d / range_val
            gamma = nugget + partial_sill * (1.5 * r - 0.5 * r**3)
    elif model_type == 1:  # Exponential
        gamma = nugget + partial_sill * (1.0 - np.exp(-3.0 * d / range_val))
    elif model_type == 2:  # Gaussian
        gamma = nugget + partial_sill * (1.0 - np.exp(-3.0 * (d / range_val)**2))
        
    return sill - gamma


@njit(fastmath=True, cache=True)
def _solve_single_uk_point(
    i,
    target_coords,      # (M, 3)
    target_coords_aniso,# (M, 3)
    neighbor_indices,   # (M, K)
    data_coords,        # (N, 3)
    data_coords_aniso,  # (N, 3)
    data_values,        # (N,)
    rng, sill, nugget, model_code, drift_code, n_beta, total_sill
):
    """
    Solve Universal Kriging for a single target point.
    Separated to avoid control flow issues in prange.
    """
    # 1. Identify valid neighbors
    indices = neighbor_indices[i]
    n_neigh = 0
    for j in range(len(indices)):
        if indices[j] >= 0:
            n_neigh += 1
        else:
            break  # Assume -1 values are at the end
    
    # Must have enough points to solve drift
    if n_neigh < n_beta + 1:
        return np.nan, np.nan
        
    # 2. Extract Local Data
    local_idx = indices[:n_neigh]
    P_aniso = data_coords_aniso[local_idx]  # For covariance (distance)
    P_orig = data_coords[local_idx]          # For drift (trend)
    v = data_values[local_idx]
    
    target_pt_aniso = target_coords_aniso[i]  # For covariance
    target_pt_orig = target_coords[i]          # For drift
    
    # 3. Build System Matrix (LHS)
    mat_size = n_neigh + n_beta
    K = np.zeros((mat_size, mat_size))
    RHS = np.zeros(mat_size)
    
    # A. Fill Covariance Block (Top-Left) - using transformed coords
    for r in range(n_neigh):
        for c in range(r, n_neigh): # Symmetric
            dx = P_aniso[r,0] - P_aniso[c,0]
            dy = P_aniso[r,1] - P_aniso[c,1]
            dz = P_aniso[r,2] - P_aniso[c,2]
            dist = np.sqrt(dx*dx + dy*dy + dz*dz)
            cov = _get_cov(dist, rng, sill, nugget, model_code)
            K[r, c] = cov
            K[c, r] = cov
        
    # Add scaled regularization for numerical stability
    max_diag = 0.0
    for d_i in range(n_neigh):
        if K[d_i, d_i] > max_diag:
            max_diag = K[d_i, d_i]
    reg_value = max(1e-10 * max_diag, 1e-9)
    for r in range(n_neigh):
        K[r, r] += reg_value

    # B. Fill Drift Block (Top-Right and Bottom-Left) - using normalized coords
    for r in range(n_neigh):
        drift_vec = _calc_drift_row(P_orig[r,0], P_orig[r,1], P_orig[r,2], drift_code)
        for b in range(n_beta):
            val = drift_vec[b]
            K[r, n_neigh + b] = val
            K[n_neigh + b, r] = val

    # 4. Build RHS Vector
    for r in range(n_neigh):
        dx = P_aniso[r,0] - target_pt_aniso[0]
        dy = P_aniso[r,1] - target_pt_aniso[1]
        dz = P_aniso[r,2] - target_pt_aniso[2]
        d_target = np.sqrt(dx*dx + dy*dy + dz*dz)
        RHS[r] = _get_cov(d_target, rng, sill, nugget, model_code)
        
    target_drift = _calc_drift_row(target_pt_orig[0], target_pt_orig[1], target_pt_orig[2], drift_code)
    for b in range(n_beta):
        RHS[n_neigh + b] = target_drift[b]

    # 5. Solve using LU decomposition (more stable than solve in Numba)
    # Manual Gaussian elimination with partial pivoting
    n = mat_size
    A = K.copy()
    b = RHS.copy()
    
    # Forward elimination
    for k in range(n):
        # Partial pivoting
        max_row = k
        max_val = abs(A[k, k])
        for row in range(k + 1, n):
            if abs(A[row, k]) > max_val:
                max_val = abs(A[row, k])
                max_row = row
        
        if max_val < 1e-12:
            # Near-singular matrix
            return np.mean(v), total_sill
        
        # Swap rows
        if max_row != k:
            for col in range(n):
                tmp = A[k, col]
                A[k, col] = A[max_row, col]
                A[max_row, col] = tmp
            tmp = b[k]
            b[k] = b[max_row]
            b[max_row] = tmp
        
        # Eliminate
        for row in range(k + 1, n):
            factor = A[row, k] / A[k, k]
            for col in range(k, n):
                A[row, col] -= factor * A[k, col]
            b[row] -= factor * b[k]
    
    # Back substitution
    weights = np.zeros(n)
    for k in range(n - 1, -1, -1):
        if abs(A[k, k]) < 1e-12:
            return np.mean(v), total_sill
        weights[k] = b[k]
        for col in range(k + 1, n):
            weights[k] -= A[k, col] * weights[col]
        weights[k] /= A[k, k]
    
    # 6. Estimate
    w_krig = weights[:n_neigh]
    mu = weights[n_neigh:]
    
    est = 0.0
    for j in range(n_neigh):
        est += w_krig[j] * v[j]
    
    # Sanity check: if estimate is extreme, fall back to mean
    data_mean = np.mean(v)
    data_max = v[0]
    data_min = v[0]
    for j in range(1, n_neigh):
        if v[j] > data_max:
            data_max = v[j]
        if v[j] < data_min:
            data_min = v[j]
    data_range = data_max - data_min
    
    if data_range > 0 and abs(est - data_mean) > 10 * data_range:
        est = data_mean
    
    # 7. Variance
    term2 = 0.0
    for j in range(n_neigh):
        term2 += w_krig[j] * RHS[j]
    term3 = 0.0
    for j in range(n_beta):
        term3 += mu[j] * RHS[n_neigh + j]
    var = total_sill - term2 - term3
    
    if var < 0.0:
        var = 0.0
    
    return est, var


@njit(parallel=True, fastmath=True, cache=True)
def run_uk_kernel(
    target_coords,      # (M, 3) - Normalized coords for drift (centered at local origin)
    target_coords_aniso,# (M, 3) - Transformed coords for covariance
    neighbor_indices,   # (M, K)
    data_coords,        # (N, 3) - Normalized coords for drift (centered at local origin)
    data_coords_aniso,  # (N, 3) - Transformed coords for covariance
    data_values,        # (N,)
    params,             # [range, sill, nugget, model_code, drift_code]
    n_beta              # int (size of drift vector)
):
    """
    Parallel Universal Kriging Solver.
    
    Uses normalized coords (centered at local origin) for drift calculation to ensure
    numerical stability with large coordinate systems (e.g., UTM). Transformed coords
    are used for covariance calculation (anisotropy handling).
    
    This version separates single-point solving into a helper function to avoid
    control flow issues (continue/try-except) that prevent Numba prange parallelization.
    """
    n_targets = target_coords.shape[0]
    
    # Unpack params
    rng = params[0]
    sill = params[1]
    nugget = params[2]
    model_code = int(params[3])
    drift_code = int(params[4])
    
    total_sill = sill + nugget
    
    # Outputs
    estimates = np.full(n_targets, np.nan)
    variances = np.full(n_targets, np.nan)
    
    # Pure prange loop - no continue, no try/except
    for i in prange(n_targets):
        est, var = _solve_single_uk_point(
            i, target_coords, target_coords_aniso,
            neighbor_indices, data_coords, data_coords_aniso, data_values,
            rng, sill, nugget, model_code, drift_code, n_beta, total_sill
        )
        estimates[i] = est
        variances[i] = var
            
    return estimates, variances


# =========================================================
# 2. PYTHON WRAPPER CLASS
# =========================================================


class DriftModel:
    def __init__(self, basis: str):
        self.basis_type = basis.lower()
        if self.basis_type == 'constant':
            self.n_beta = 1
            self.code = 0
        elif self.basis_type == 'linear':
            self.n_beta = 4
            self.code = 1
        elif self.basis_type == 'quadratic':
            self.n_beta = 10
            self.code = 2
        else:
            # Default to constant if unknown
            self.n_beta = 1
            self.code = 0
    
    def design_matrix(self, coords):
        # Kept for compatibility, but Kernel does this internally now
        pass


class UniversalKriging3D:
    """Numba-accelerated Universal Kriging."""

    # Default chunk size for progress reporting (process N targets at a time)
    # Increased for better performance on large grids (less overhead from chunking)
    DEFAULT_CHUNK_SIZE = 200000  # 200k blocks per chunk (was 50k)

    def __init__(self, coords, values, variogram_model, drift_model, config):
        self.coords = np.asarray(coords, dtype=float)
        self.values = np.asarray(values, dtype=float)
        self.variogram = variogram_model
        self.drift = drift_model
        self.config = config
        
        if not NUMBA_AVAILABLE:
            logger.warning("Numba not detected! Universal Kriging will be very slow.")


    def estimate(self, target_coords, progress_callback=None):
        """
        Estimate values at target coordinates.
        
        Args:
            target_coords: (M, 3) array of target coordinates
            progress_callback: Optional callable(percent: int, message: str) for progress updates
        
        Returns:
            estimates: (M,) array of estimated values
            variances: (M,) array of kriging variances
        """
        target_coords = np.asarray(target_coords, dtype=float)
        n_targets = len(target_coords)
        
        # 0. Normalize coordinates for drift calculation (CRITICAL for numerical stability)
        # Large coordinates (e.g., UTM 6,000,000) cause precision errors in float64 drift matrices
        # Normalize to local origin (0,0,0) relative to grid center
        all_coords = np.vstack([self.coords, target_coords])
        coord_center = np.mean(all_coords, axis=0)
        
        # Normalize data and target coordinates for drift (subtract center)
        data_coords_normalized = self.coords - coord_center
        target_coords_normalized = target_coords - coord_center
        
        if progress_callback:
            progress_callback(10, f"Preparing data ({n_targets:,} targets)...")
        
        # 1. Anisotropy Transform (for covariance calculation only)
        aniso = self.variogram.get('anisotropy') if self.variogram else None
        if aniso:
            # Use NeighborSearcher for unified neighbor search with anisotropy
            anisotropy_params = {
                'azimuth': aniso.get('azimuth', 0.0),
                'dip': aniso.get('dip', 0.0),
                'major_range': aniso.get('major_range', self.variogram.get('range', 100.0) if self.variogram else 100.0),
                'minor_range': aniso.get('minor_range', self.variogram.get('range', 100.0) if self.variogram else 100.0),
                'vert_range': aniso.get('vert_range', self.variogram.get('range', 100.0) if self.variogram else 100.0)
            }
            searcher = NeighborSearcher(self.coords, anisotropy_params=anisotropy_params)
            data_trans = searcher.get_transformed_coords()
            targ_trans = apply_anisotropy(
                target_coords,
                anisotropy_params['azimuth'],
                anisotropy_params['dip'],
                anisotropy_params['major_range'],
                anisotropy_params['minor_range'],
                anisotropy_params['vert_range']
            )
            eff_range = 1.0  # Normalized space
            search_r = self.config.get('max_distance', None) if self.config else None
        else:
            searcher = NeighborSearcher(self.coords, anisotropy_params=None)
            data_trans = self.coords
            targ_trans = target_coords
            eff_range = self.variogram.get('range', 100.0) if self.variogram else 100.0
            search_r = self.config.get('max_distance', None) if self.config else None

        if progress_callback:
            progress_callback(15, f"Searching for neighbors ({n_targets:,} targets)...")
        
        # 2. Neighbor Search using unified NeighborSearcher (optimized batch query)
        indices, dists = searcher.search(
            target_coords=target_coords,
            n_neighbors=self.config.get('n_neighbors', 12) if self.config else 12,
            max_distance=search_r
        )
        
        if progress_callback:
            progress_callback(20, "Neighbor search complete, setting up kriging system...")

        # 3. Prepare Numba Params
        m_map = {'spherical': 0, 'exponential': 1, 'gaussian': 2}
        m_code = m_map.get(self.variogram.get('model_type', 'spherical').lower() if self.variogram else 'spherical', 0)
        
        params = np.array([
            eff_range,
            (self.variogram.get('sill', 1.0) if self.variogram else 1.0) - (self.variogram.get('nugget', 0.0) if self.variogram else 0.0),  # Partial sill
            self.variogram.get('nugget', 0.0) if self.variogram else 0.0,
            m_code,
            self.drift.code
        ], dtype=np.float64)

        # 4. Run Parallel Kernel (with chunking for progress updates on large grids)
        if not NUMBA_AVAILABLE:
            logger.error("Numba not installed. UK will fail or requires legacy fallback.")
            return np.zeros(n_targets), np.zeros(n_targets)
        
        # Optimize: Use larger chunks for very large grids to reduce overhead
        # Only chunk if grid is extremely large AND progress callback is needed
        chunk_size = self.config.get('chunk_size', self.DEFAULT_CHUNK_SIZE) if self.config else self.DEFAULT_CHUNK_SIZE
        
        # For grids <= chunk_size or no progress callback, process all at once (fastest)
        # For very large grids with progress, use chunking but minimize overhead
        if n_targets <= chunk_size or progress_callback is None:
            if progress_callback:
                progress_callback(25, f"Processing {n_targets:,} targets...")
            
            est, var = run_uk_kernel(
                target_coords=target_coords_normalized,
                target_coords_aniso=targ_trans,
                neighbor_indices=indices,
                data_coords=data_coords_normalized,
                data_coords_aniso=data_trans,
                data_values=self.values,
                params=params,
                n_beta=self.drift.n_beta
            )
            
            if progress_callback:
                progress_callback(95, "Finalizing...")
        else:
            # Chunked processing for progress updates on very large grids
            # Use fewer, larger chunks to minimize overhead
            n_chunks = int(np.ceil(n_targets / chunk_size))
            logger.info(f"UK: Processing {n_targets:,} targets in {n_chunks} chunks of ~{chunk_size:,}")
            
            est = np.full(n_targets, np.nan)
            var = np.full(n_targets, np.nan)
            
            # Pre-allocate arrays to avoid repeated allocation
            # Extract chunk data more efficiently (views instead of copies where possible)
            for chunk_idx in range(n_chunks):
                start_idx = chunk_idx * chunk_size
                end_idx = min(start_idx + chunk_size, n_targets)
                
                # Progress: 25% to 95% is the kriging phase
                # Update less frequently for large grids (every chunk, not every block)
                chunk_progress = 25 + int(70 * ((chunk_idx + 1) / n_chunks))
                if progress_callback and (chunk_idx == 0 or chunk_idx == n_chunks - 1 or (chunk_idx + 1) % max(1, n_chunks // 5) == 0):
                    progress_callback(
                        chunk_progress,
                        f"Kriging chunk {chunk_idx + 1}/{n_chunks} ({end_idx:,}/{n_targets:,} blocks)..."
                    )
                
                # Extract chunk data (use views where possible for efficiency)
                chunk_targets_norm = target_coords_normalized[start_idx:end_idx]
                chunk_targets_aniso = targ_trans[start_idx:end_idx]
                chunk_indices = indices[start_idx:end_idx]
                
                # Run kernel on chunk
                chunk_est, chunk_var = run_uk_kernel(
                    target_coords=chunk_targets_norm,
                    target_coords_aniso=chunk_targets_aniso,
                    neighbor_indices=chunk_indices,
                    data_coords=data_coords_normalized,
                    data_coords_aniso=data_trans,
                    data_values=self.values,
                    params=params,
                    n_beta=self.drift.n_beta
                )
                
                est[start_idx:end_idx] = chunk_est
                var[start_idx:end_idx] = chunk_var
            
            if progress_callback:
                progress_callback(95, "Finalizing...")

        return est, var


# =========================================================
# 3. COMPATIBILITY WRAPPERS (Enhanced)
# =========================================================


def universal_kriging_3d_full(
    data_coords, data_values, target_coords,
    variogram_params, drift_model, n_neighbors=12,
    max_distance=None, model_type="spherical", **kwargs
):
    """
    Wrapper that initializes the Optimized Engine and runs it.
    """
    # Initialize Engine
    uk = UniversalKriging3D(
        data_coords, data_values,
        variogram_params, drift_model,
        {'n_neighbors': n_neighbors, 'max_distance': max_distance}
    )
    
    # Run
    est, var = uk.estimate(target_coords)
    
    # Return formatted result object
    return UniversalKrigingResults(
        estimates=est,
        kriging_variance=var,
        status=np.zeros(len(est), dtype=int),  # 0=OK
        kriging_mean=est,
        kriging_efficiency=np.full_like(est, np.nan),
        slope_of_regression=np.full_like(est, np.nan),
        lagrange_multiplier=np.full_like(est, np.nan),
        num_samples=np.full(len(est), n_neighbors, dtype=int),
        sum_weights=np.full_like(est, 1.0),
        sum_negative_weights=np.zeros_like(est),
        min_distance=np.zeros_like(est),
        avg_distance=np.zeros_like(est),
        nearest_sample_id=np.full(len(est), -1, dtype=int),
        num_duplicates_removed=np.zeros_like(est, dtype=int),
        search_pass=np.ones_like(est, dtype=int),
        search_volume=np.zeros_like(est),
        trend_coefficients=np.zeros((len(est), drift_model.n_beta)),
        drift_value=np.zeros_like(est),
        residual_estimate=np.zeros_like(est),
        metadata={}
    )


def run_universal_kriging_job(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run Universal Kriging job (wrapper for controller integration).
    
    Args:
        params: Job parameters dict (will be validated and converted to UniversalKrigingJobParams)
    
    Returns:
        Result dict with estimates, variances, metadata
    """
    from ..models.kriging3d import create_estimation_grid
    from .kriging_job_params import UniversalKrigingJobParams
    from .simulation_interface import GridDefinition
    # NOTE: PyVista imports removed - grid creation happens in main thread
    
    # Validate and convert parameters (type safety check happens here)
    # Pydantic will catch type mismatches, missing required fields, and invalid values
    try:
        job_params = UniversalKrigingJobParams.from_dict(params)
    except (ValueError, KeyError, TypeError) as e:
        error_msg = str(e)
        logger.error(f"Invalid Universal Kriging job parameters: {error_msg}")
        return {'error': f"Parameter validation failed: {error_msg}"}
    except Exception as e:
        # Handle Pydantic ValidationError (if Pydantic is available)
        # This provides detailed field-level error messages
        error_msg = str(e)
        logger.error(f"Universal Kriging parameter validation error: {error_msg}", exc_info=True)
        # Extract field errors if available (Pydantic format)
        if hasattr(e, 'errors'):
            field_errors = [f"{err.get('loc', 'unknown')}: {err.get('msg', 'invalid')}" for err in e.errors()]
            error_msg = f"Validation errors: {'; '.join(field_errors)}"
        return {'error': f"Parameter validation failed: {error_msg}"}
    
    # Prepare data
    coords = job_params.data_df[['X', 'Y', 'Z']].values
    values = job_params.data_df[job_params.variable].values
    
    # Create drift model
    drift_model = DriftModel(job_params.drift_type)
    
    # Create estimation grid
    if job_params.grid_config.counts is not None:
        # Use explicit grid definition
        nx, ny, nz = job_params.grid_config.counts
        xmin, ymin, zmin = job_params.grid_config.origin or (0.0, 0.0, 0.0)
        dx, dy, dz = job_params.grid_config.spacing
        
        gx = np.arange(nx) * dx + xmin + dx / 2.0
        gy = np.arange(ny) * dy + ymin + dy / 2.0
        gz = np.arange(nz) * dz + zmin + dz / 2.0
        GX, GY, GZ = np.meshgrid(gx, gy, gz, indexing="ij")
        target_coords = np.column_stack([GX.ravel(), GY.ravel(), GZ.ravel()])
        grid_x, grid_y, grid_z = GX, GY, GZ
    else:
        # Fallback to auto-grid
        data_range = coords.max(axis=0) - coords.min(axis=0)
        padding_factor = 0.1
        buffer = tuple(data_range * padding_factor)
        grid_x, grid_y, grid_z, target_coords = create_estimation_grid(
            coords, job_params.grid_config.spacing, buffer=buffer, max_points=job_params.grid_config.max_points
        )
    
    # Convert VariogramParams to dict format expected by kriging functions
    variogram_params_dict = job_params.variogram_params.to_dict()

    # === RESIDUAL VARIOGRAM COMPUTATION ===
    # Compute complete residual variogram instead of just sill correction
    logger.info("Computing residual variogram for proper UK...")
    residual_vario_calculator = ResidualVariogramCalculator()
    drift_type_map = {0: 'constant', 1: 'linear', 2: 'quadratic'}
    drift_type = drift_type_map.get(drift_model.code, 'linear')

    residual_variogram_result = residual_vario_calculator.compute_residual_variogram(
        coords, values, drift_type, variogram_params_dict.get('model_type', 'spherical')
    )

    if 'error' in residual_variogram_result:
        logger.warning(f"Residual variogram computation failed: {residual_variogram_result['error']}")
        logger.warning("Falling back to original variogram parameters")
        corrected_variogram_params = variogram_params_dict.copy()
        residual_info = None
    else:
        # Use the full residual variogram parameters
        corrected_variogram_params = residual_variogram_result['variogram_params']
        residual_info = residual_variogram_result
        logger.info(f"Residual variogram fitted: sill={corrected_variogram_params['sill']:.3f}, "
                   f"range={corrected_variogram_params['range']:.1f}, "
                   f"nugget={corrected_variogram_params['nugget']:.3f}")

    # Log grid size for user awareness
    n_targets = len(target_coords)
    logger.info(f"Universal Kriging: Estimating {n_targets:,} blocks with {len(coords):,} data points")

    # Always use class-based method with progress callback support
    uk = UniversalKriging3D(
        coords,
        values,
        corrected_variogram_params,  # Use corrected parameters
        drift_model,
        {
            'n_neighbors': job_params.search_config.n_neighbors,
            'max_distance': job_params.search_config.max_distance
        }
    )
    
    # Pass progress callback to estimate method
    estimates_flat, variances_flat = uk.estimate(target_coords, progress_callback=job_params.progress_callback)
    
    # Reshape to grid
    nx, ny, nz = grid_x.shape
    
    # Infer grid definition
    if job_params.grid_config.origin and job_params.grid_config.counts:
        x0, y0, z0 = job_params.grid_config.origin
        dx, dy, dz = job_params.grid_config.spacing
    else:
        # Infer from grid coordinates
        dx = grid_x[1, 0, 0] - grid_x[0, 0, 0] if nx > 1 else 10.0
        dy = grid_y[0, 1, 0] - grid_y[0, 0, 0] if ny > 1 else 10.0
        dz = grid_z[0, 0, 1] - grid_z[0, 0, 0] if nz > 1 else 5.0
        x0 = grid_x[0, 0, 0] - dx / 2
        y0 = grid_y[0, 0, 0] - dy / 2
        z0 = grid_z[0, 0, 0] - dz / 2
    
    estimates = estimates_flat.reshape((nx, ny, nz), order='F')
    variances = variances_flat.reshape((nx, ny, nz), order='F')

    # === OPTIONAL POST-PROCESSING FILTERS ===
    # Apply post-processing to handle edge instabilities
    post_processing_stats = {}
    if job_params.post_processing_config:
        estimates, post_processing_stats = _apply_post_processing_filters(
            estimates, coords, values, corrected_variogram_params, drift_model,
            job_params.post_processing_config
        )

    # Property names
    property_name = f"UK_{job_params.variable}"
    variance_property = f"UK_{job_params.variable}_var"
    
    # Return ONLY primitive data - PyVista grid creation happens in main thread
    return {
        'estimates': estimates,  # numpy array
        'variances': variances,  # numpy array
        'grid_x': grid_x,  # numpy array
        'grid_y': grid_y,  # numpy array
        'grid_z': grid_z,  # numpy array
        'grid_def': {
            'origin': (x0, y0, z0),
            'spacing': (dx, dy, dz),
            'counts': (nx, ny, nz)
        },
        'property_name': property_name,
        'variance_property': variance_property,
        'metadata': {
            'method': 'Universal Kriging',
            'drift_type': job_params.drift_type,
            'variogram_params': corrected_variogram_params,
            'n_neighbors': job_params.search_config.n_neighbors,
            'residual_sill_correction': corrected_variogram_params.get('_correction_info'),
            'post_processing': post_processing_stats if post_processing_stats else None
        },
        '_create_grid_in_main_thread': True  # Flag to create PyVista grid in main thread
    }


# =========================================================
# 4. RESIDUAL SILL CORRECTION UTILITIES
# =========================================================

def _compute_residual_sill_correction(coords: np.ndarray, values: np.ndarray,
                                   original_variogram: Dict[str, Any],
                                   drift_model: DriftModel) -> Dict[str, Any]:
    """
    Compute residual sill correction for proper UK.

    Fits drift to data, computes residuals, and adjusts variogram sill to match
    residual variance rather than total variance.

    Args:
        coords: (N, 3) coordinate array
        values: (N,) value array
        original_variogram: Original variogram parameters
        drift_model: Drift model object

    Returns:
        Corrected variogram parameters
    """
    try:
        # Fit drift to data
        drift_fitter = DriftFitter()
        drift_type_map = {0: 'constant', 1: 'linear', 2: 'quadratic'}
        drift_type = drift_type_map.get(drift_model.code, 'linear')

        drift_fit = drift_fitter.fit_drift(coords, values, drift_type)

        if 'error' in drift_fit:
            logger.warning(f"Drift fitting failed: {drift_fit['error']}. Using original variogram.")
            return original_variogram

        # Compute residual variance
        residuals = drift_fit['residuals']
        residual_variance = float(np.var(residuals))

        # Get original parameters
        nugget = original_variogram.get('nugget', 0.0)
        original_sill = original_variogram.get('sill', residual_variance + nugget)

        # Corrected sill = residual variance - nugget
        # This ensures the variogram models the residual spatial variation
        corrected_sill = max(0.01, residual_variance - nugget)  # Minimum sill to avoid issues

        # Create corrected variogram
        corrected_variogram = original_variogram.copy()
        corrected_variogram['sill'] = corrected_sill

        # Log the correction
        sill_reduction = original_sill - corrected_sill
        logger.info(f"Residual sill correction: {original_sill:.3f} → {corrected_sill:.3f} "
                   f"(reduction: {sill_reduction:.3f}, residual var: {residual_variance:.3f})")

        # Store correction info for validation
        corrected_variogram['_correction_info'] = {
            'original_sill': original_sill,
            'corrected_sill': corrected_sill,
            'residual_variance': residual_variance,
            'drift_r_squared': drift_fit.get('r_squared', 0),
            'drift_rmse': drift_fit.get('rmse', 0)
        }

        return corrected_variogram

    except Exception as e:
        logger.warning(f"Residual sill correction failed: {e}. Using original variogram.")
        return original_variogram


def _apply_post_processing_filters(estimates: np.ndarray, coords: np.ndarray, values: np.ndarray,
                                 variogram_params: Dict[str, Any], drift_model: DriftModel,
                                 config: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Apply post-processing filters to UK estimates.

    Args:
        estimates: (nx, ny, nz) UK estimates
        coords: (N, 3) data coordinates
        values: (N,) data values
        variogram_params: Variogram parameters
        drift_model: Drift model
        config: Post-processing configuration

    Returns:
        Filtered estimates and statistics
    """
    try:
        post_processor = UKPostProcessor()

        # Flatten estimates for processing
        original_shape = estimates.shape
        estimates_flat = estimates.ravel()

        # Get drift predictions if needed for mean reversion
        drift_predictions = None
        if config.get('mean_reversion', False):
            drift_fitter = DriftFitter()
            drift_type_map = {0: 'constant', 1: 'linear', 2: 'quadratic'}
            drift_type = drift_type_map.get(drift_model.code, 'linear')

            # Fit drift to data
            drift_fit = drift_fitter.fit_drift(coords, values, drift_type)
            if 'error' not in drift_fit:
                # Predict drift at all grid points (simplified - would need full grid coords)
                # For now, use data locations as proxy
                drift_predictions = drift_fit['predictions']

        # Apply combined UK stabilization filters
        data_values_for_filters = values  # Pass original data for reference bounds
        filter_config = {
            'drift_contraction': True,
            'contraction_factor': 0.85,  # Shrink toward drift by 15%
            'soft_truncation': True,
            'lower_percentile': 0.5,   # Very soft lower bound
            'upper_percentile': 99.5,  # Very soft upper bound
            'positivity_constraint': True,
            'min_value': 0.0
        }

        filtered_flat, stats = post_processor.apply_combined_uk_filters(
            estimates_flat, drift_predictions, data_values_for_filters, filter_config
        )

        # Reshape back
        filtered = filtered_flat.reshape(original_shape)

        logger.info(f"Applied post-processing filters: {len(stats.get('applied_filters', []))} filters used")

        return filtered, stats

    except Exception as e:
        logger.warning(f"Post-processing failed: {e}. Returning original estimates.")
        return estimates, {'error': str(e)}


# =========================================================
# 5. NUMBA PRE-COMPILATION (Warm-up)
# =========================================================

def precompile_uk_kernels():
    """
    Pre-compile Numba JIT functions with minimal dummy data.
    
    Call this at application startup (e.g., in a background thread) to avoid
    the 5-30 second JIT compilation delay when the user first runs kriging.
    
    Returns:
        bool: True if compilation succeeded, False otherwise
    """
    if not NUMBA_AVAILABLE:
        logger.info("Numba not available - skipping UK kernel pre-compilation")
        return False
    
    try:
        logger.info("Pre-compiling Universal Kriging Numba kernels...")
        
        # Minimal dummy data (10 data points, 5 targets)
        n_data = 10
        n_target = 5
        k_neighbors = 4
        
        # Generate synthetic data
        data_coords = np.random.rand(n_data, 3).astype(np.float64) * 100
        data_values = np.random.rand(n_data).astype(np.float64)
        target_coords = np.random.rand(n_target, 3).astype(np.float64) * 100
        
        # Dummy neighbor indices (each target has k_neighbors neighbors)
        neighbor_indices = np.zeros((n_target, k_neighbors), dtype=np.int64)
        for i in range(n_target):
            neighbor_indices[i] = np.arange(k_neighbors)
        
        # Params: [range, sill, nugget, model_code, drift_code]
        params = np.array([50.0, 1.0, 0.1, 0, 0], dtype=np.float64)  # Spherical, Constant
        
        # Warm up helper functions first (dependencies of run_uk_kernel)
        _ = _calc_drift_row(0.0, 0.0, 0.0, 0)  # Constant
        _ = _calc_drift_row(1.0, 2.0, 3.0, 1)  # Linear
        _ = _calc_drift_row(1.0, 2.0, 3.0, 2)  # Quadratic
        
        # Warm up covariance calculations with different models
        for model_type in [0, 1, 2]:  # Spherical, Exponential, Gaussian
            _ = _get_cov(10.0, 50.0, 1.0, 0.1, model_type)
        
        # Warm up single point solver (critical for parallelization)
        _ = _solve_single_uk_point(
            0,
            target_coords,
            target_coords,  # Same for no anisotropy
            neighbor_indices,
            data_coords,
            data_coords,  # Same for no anisotropy
            data_values,
            50.0, 1.0, 0.1, 0, 0, 1, 1.1  # rng, sill, nugget, model_code, drift_code, n_beta, total_sill
        )
        
        # Run the full parallel kernel to trigger compilation
        _ = run_uk_kernel(
            target_coords=target_coords,
            target_coords_aniso=target_coords,  # Same for no anisotropy
            neighbor_indices=neighbor_indices,
            data_coords=data_coords,
            data_coords_aniso=data_coords,  # Same for no anisotropy
            data_values=data_values,
            params=params,
            n_beta=1  # Constant drift
        )
        
        logger.info("Universal Kriging Numba kernels pre-compiled successfully")
        return True
        
    except Exception as e:
        logger.warning(f"UK kernel pre-compilation failed (non-fatal): {e}")
        return False