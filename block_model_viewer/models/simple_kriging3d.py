"""
3D Simple Kriging Engine - HIGH PERFORMANCE

Uses the same optimized approach as Ordinary Kriging (kriging3d.py):
- NeighborSearcher for unified neighbor search
- Numba JIT kernel for parallel execution (if available)
- SciPy fallback for non-Numba environments

Solves Z* = Mean + Sum(Weights * (Data - Mean))
"""

import logging
from typing import Tuple, Optional, Dict
from dataclasses import dataclass
import numpy as np

from scipy.spatial.distance import cdist
from scipy.linalg import solve, LinAlgError

try:
    from scipy.spatial import cKDTree as KDTree
except ImportError:
    KDTree = None

# Import from kriging3d for consistency
from block_model_viewer.models.kriging3d import (
    apply_anisotropy,
    get_variogram_function,
    MODEL_GAMMA
)
from block_model_viewer.models.geostat_results import SimpleKrigingResults

# Try to import NeighborSearcher
try:
    from ..geostats.geostats_utils import NeighborSearcher
    NEIGHBOR_SEARCHER_AVAILABLE = True
except ImportError:
    NEIGHBOR_SEARCHER_AVAILABLE = False
    NeighborSearcher = None

# Try to import Numba kernel
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def jit(*args, **kwargs):
        def decorator(func): return func
        return decorator
    prange = range

logger = logging.getLogger(__name__)


@dataclass
class SKParameters:
    global_mean: float = 0.0
    variogram_type: str = "spherical"
    sill: float = 1.0
    nugget: float = 0.0
    range_major: float = 100.0
    range_minor: float = 50.0
    range_vert: float = 25.0
    azimuth: float = 0.0
    dip: float = 0.0
    ndmax: int = 12
    max_search_radius: float = 200.0
    nmin: int = 1  # Minimum samples required
    sectoring: str = "No sectoring"  # Sector search option


# =========================================================
# NUMBA JIT KERNEL (same pattern as kriging_engine.py)
# =========================================================

@jit(nopython=True, fastmath=True)
def _variogram_spherical(h, rng, sill, nugget):
    """Spherical variogram."""
    if h >= rng:
        return nugget + sill
    ratio = h / rng
    return nugget + sill * (1.5 * ratio - 0.5 * ratio * ratio * ratio)


@jit(nopython=True, fastmath=True)
def _variogram_exponential(h, rng, sill, nugget):
    """Exponential variogram."""
    return nugget + sill * (1.0 - np.exp(-3.0 * h / rng))


@jit(nopython=True, fastmath=True)
def _variogram_gaussian(h, rng, sill, nugget):
    """Gaussian variogram."""
    return nugget + sill * (1.0 - np.exp(-3.0 * (h / rng)**2))


@jit(nopython=True, fastmath=True)
def _calc_covariance(dist, model_type, rng, sill, nugget):
    """Calculate Covariance C(h) = TotalSill - Gamma(h)."""
    total_sill = sill + nugget
    
    if model_type == 0:
        gamma = _variogram_spherical(dist, rng, sill, nugget)
    elif model_type == 1:
        gamma = _variogram_exponential(dist, rng, sill, nugget)
    elif model_type == 2:
        gamma = _variogram_gaussian(dist, rng, sill, nugget)
    else:
        gamma = _variogram_spherical(dist, rng, sill, nugget)
    
    return total_sill - gamma


@jit(nopython=True, fastmath=True)
def _solve_single_sk_point(
    i,
    target_coords,      # (M, 3)
    neighbor_indices,   # (M, K)
    data_coords,        # (N, 3)
    data_values,        # (N,)
    rng, sill, nugget, model_type, global_mean, total_sill, k_neighbors, nmin
):
    """
    Solve Simple Kriging for a single target point.
    Separated to avoid control flow issues in prange.
    """
    indices = neighbor_indices[i]
    
    # Extract valid neighbors
    local_coords = np.zeros((k_neighbors, 3))
    local_values = np.zeros(k_neighbors)
    
    valid_count = 0
    for n in range(k_neighbors):
        idx = indices[n]
        if idx >= 0:
            local_coords[valid_count, 0] = data_coords[idx, 0]
            local_coords[valid_count, 1] = data_coords[idx, 1]
            local_coords[valid_count, 2] = data_coords[idx, 2]
            local_values[valid_count] = data_values[idx]
            valid_count += 1
    
    if valid_count < nmin:
        return global_mean, total_sill, 0, 0.0, 0.0, np.nan, np.nan, np.nan, 1

    # Build Covariance Matrix (optimized: symmetric, compute upper triangle only)
    mat_k = np.zeros((valid_count, valid_count))
    rhs = np.zeros(valid_count)
    
    # Pre-compute target point for RHS
    target_pt = target_coords[i]
    
    for r in range(valid_count):
        # RHS: covariance between target and neighbor r (compute once per row)
        tx = local_coords[r, 0] - target_pt[0]
        ty = local_coords[r, 1] - target_pt[1]
        tz = local_coords[r, 2] - target_pt[2]
        d_target = np.sqrt(tx*tx + ty*ty + tz*tz)
        rhs[r] = _calc_covariance(d_target, model_type, rng, sill, nugget)
        
        # LHS: symmetric matrix - compute upper triangle and mirror
        for c in range(r, valid_count):
            dx = local_coords[r, 0] - local_coords[c, 0]
            dy = local_coords[r, 1] - local_coords[c, 1]
            dz = local_coords[r, 2] - local_coords[c, 2]
            d = np.sqrt(dx*dx + dy*dy + dz*dz)
            cov_val = _calc_covariance(d, model_type, rng, sill, nugget)
            mat_k[r, c] = cov_val
            if r != c:  # Mirror to lower triangle
                mat_k[c, r] = cov_val

    # Add regularization
    max_diag = 0.0
    for d_i in range(valid_count):
        if mat_k[d_i, d_i] > max_diag:
            max_diag = mat_k[d_i, d_i]
    reg_value = max(1e-10 * max_diag, 1e-9)
    for d_i in range(valid_count):
        mat_k[d_i, d_i] += reg_value

    # Solve using Gaussian elimination with partial pivoting
    A = mat_k.copy()
    b = rhs.copy()
    dim = valid_count
    
    # Forward elimination
    for k in range(dim):
        max_row = k
        max_val = abs(A[k, k])
        for row in range(k + 1, dim):
            if abs(A[row, k]) > max_val:
                max_val = abs(A[row, k])
                max_row = row
        
        if max_val < 1e-12:
            return global_mean, total_sill, 0, 0.0, 0.0, np.nan, np.nan, np.nan, 1
        
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
            return global_mean, total_sill, 0, 0.0, 0.0, np.nan, np.nan, np.nan, 1
        weights[k] = b[k]
        for col in range(k + 1, dim):
            weights[k] -= A[k, col] * weights[col]
        weights[k] /= A[k, k]
    
    # Simple Kriging Estimate
    est = global_mean
    for w_i in range(valid_count):
        est += weights[w_i] * (local_values[w_i] - global_mean)
    
    # Sanity check
    data_min = local_values[0]
    data_max = local_values[0]
    for w_i in range(valid_count):
        if local_values[w_i] < data_min:
            data_min = local_values[w_i]
        if local_values[w_i] > data_max:
            data_max = local_values[w_i]
    data_range = data_max - data_min
    
    if data_range > 0 and abs(est - global_mean) > 10 * data_range:
        est = global_mean
    
    # SK Variance
    var = total_sill
    for w_i in range(valid_count):
        var -= weights[w_i] * rhs[w_i]

    if var < 0.0:
        var = 0.0

    # Diagnostic metrics
    sum_weights = 0.0
    n_negative = 0
    for w_i in range(valid_count):
        sum_weights += weights[w_i]
        if weights[w_i] < 0.0:
            n_negative += 1

    pct_negative = 100.0 * n_negative / max(valid_count, 1)

    # Distances to samples (already computed during RHS assembly above)
    distances = np.zeros(valid_count)
    for r in range(valid_count):
        tx = local_coords[r, 0] - target_pt[0]
        ty = local_coords[r, 1] - target_pt[1]
        tz = local_coords[r, 2] - target_pt[2]
        distances[r] = np.sqrt(tx*tx + ty*ty + tz*tz)

    min_dist = np.inf
    avg_dist = 0.0
    max_dist = 0.0
    if valid_count > 0:
        min_dist = distances[0]
        max_dist = distances[0]
        for d_i in range(valid_count):
            if distances[d_i] < min_dist:
                min_dist = distances[d_i]
            if distances[d_i] > max_dist:
                max_dist = distances[d_i]
            avg_dist += distances[d_i]
        avg_dist /= valid_count
    else:
        min_dist = np.nan
        avg_dist = np.nan

    # Solver status flag (0=success)
    solver_flag = 0

    return est, var, valid_count, sum_weights, pct_negative, min_dist, avg_dist, max_dist, solver_flag


@jit(nopython=True, parallel=True, fastmath=True)
def run_sk_kernel(
    target_coords,      # (M, 3)
    neighbor_indices,   # (M, K)
    data_coords,        # (N, 3)
    data_values,        # (N,)
    params              # [range, sill, nugget, model_code, global_mean, nmin]
):
    """
    Parallel Simple Kriging kernel (Numba JIT).
    
    This version separates single-point solving into a helper function to avoid
    control flow issues (continue/try-except) that prevent Numba prange parallelization.
    """
    n_targets = target_coords.shape[0]
    k_neighbors = neighbor_indices.shape[1]
    
    # Unpack params
    rng = params[0]
    sill = params[1]
    nugget = params[2]
    model_type = int(params[3])
    global_mean = params[4]
    total_sill = sill + nugget

    # Output arrays
    estimates = np.full(n_targets, np.nan)
    variances = np.full(n_targets, np.nan)
    neighbour_counts = np.full(n_targets, 0, dtype=int)

    # Diagnostic arrays
    sum_weights_arr = np.full(n_targets, 0.0)
    pct_neg_weights_arr = np.full(n_targets, 0.0)
    min_dist_arr = np.full(n_targets, np.nan)
    avg_dist_arr = np.full(n_targets, np.nan)
    max_dist_arr = np.full(n_targets, np.nan)
    solver_status_arr = np.full(n_targets, 0, dtype=np.int32)

    # Pure prange loop - no continue, no try/except
    for i in prange(n_targets):
        est, var, n_used, sum_w, pct_neg, min_d, avg_d, max_d, slv_flag = _solve_single_sk_point(
            i, target_coords, neighbor_indices, data_coords, data_values,
            rng, sill, nugget, model_type, global_mean, total_sill, k_neighbors, int(params[5])
        )
        estimates[i] = est
        variances[i] = var
        neighbour_counts[i] = n_used
        sum_weights_arr[i] = sum_w
        pct_neg_weights_arr[i] = pct_neg
        min_dist_arr[i] = min_d
        avg_dist_arr[i] = avg_d
        max_dist_arr[i] = max_d
        solver_status_arr[i] = slv_flag

    return (estimates, variances, neighbour_counts,
            sum_weights_arr, pct_neg_weights_arr,
            min_dist_arr, avg_dist_arr, max_dist_arr,
            solver_status_arr)


# =========================================================
# MAIN API (matches kriging3d.py structure)
# =========================================================

def simple_kriging_fast(
    data_coords: np.ndarray,
    data_values: np.ndarray,
    target_coords: np.ndarray,
    variogram_params: Dict,
    n_neighbors: int = 12,
    max_distance: Optional[float] = None,
    model_type: str = "spherical",
    global_mean: Optional[float] = None,
    nmin: int = 1,
    sectoring: str = "No sectoring",
    progress_callback=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    High-performance Simple Kriging using Numba JIT compilation.
    Same interface as ordinary_kriging_fast() for consistency.

    Parameters
    ----------
    data_coords : np.ndarray
        (N, 3) array of data coordinates
    data_values : np.ndarray
        (N,) array of data values
    target_coords : np.ndarray
        (M, 3) array of target coordinates
    variogram_params : Dict
        Variogram parameters including 'range', 'sill', 'nugget', and optional 'anisotropy'
    n_neighbors : int
        Maximum number of neighbors to use
    max_distance : float, optional
        Maximum search distance
    model_type : str
        Variogram model type ('spherical', 'exponential', 'gaussian')
    global_mean : float, optional
        Global mean for Simple Kriging (required). If None, uses data mean.
    nmin : int
        Minimum number of samples required (default: 1)
    sectoring : str
        Sector search option ('No sectoring', '4 sectors', '8 sectors')
    progress_callback : callable, optional
        Progress callback function(progress: int, message: str)

    Returns
    -------
    estimates : np.ndarray
        (M,) array of kriged estimates
    variances : np.ndarray
        (M,) array of kriging variances
    neighbour_counts : np.ndarray
        (M,) array of number of neighbors used for each estimate
    """
    if not NUMBA_AVAILABLE:
        logger.warning("Numba not available, falling back to standard implementation")
        return simple_kriging_3d_standard(
            data_coords, data_values, target_coords, variogram_params,
            n_neighbors, max_distance, model_type, global_mean, progress_callback
        )
    
    # Parse parameters
    rng = float(variogram_params["range"])
    sill_total = float(variogram_params["sill"])
    nug = float(variogram_params.get("nugget", 0.0))
    partial_sill = sill_total - nug
    
    # Global mean (required for SK)
    if global_mean is None:
        global_mean = float(np.nanmean(data_values))
        logger.info(f"Simple Kriging: Using data mean = {global_mean:.4f}")
    
    # Map model to code
    model_map = {"spherical": 0, "exponential": 1, "gaussian": 2}
    model_code = model_map.get(model_type.lower(), 0)
    
    params = np.array([rng, partial_sill, nug, model_code, global_mean, nmin], dtype=np.float64)
    
    # Handle anisotropy
    anisotropy_params = variogram_params.get("anisotropy", None)
    if anisotropy_params:
        azimuth = float(anisotropy_params.get("azimuth", 0.0))
        dip = float(anisotropy_params.get("dip", 0.0))
        major_range = float(anisotropy_params.get("major_range", rng))
        minor_range = float(anisotropy_params.get("minor_range", rng))
        vert_range = float(anisotropy_params.get("vert_range", rng))
        
        data_coords_aniso = apply_anisotropy(
            data_coords, azimuth, dip, major_range, minor_range, vert_range
        )
        target_coords_aniso = apply_anisotropy(
            target_coords, azimuth, dip, major_range, minor_range, vert_range
        )
        params[0] = 1.0  # Normalized range
        range_geometric_mean = (major_range * minor_range * vert_range) ** (1.0 / 3.0)
        
        logger.info(f"SK anisotropic: azimuth={azimuth:.1f}°, dip={dip:.1f}°")
    else:
        data_coords_aniso = data_coords
        target_coords_aniso = target_coords
        range_geometric_mean = rng
    
    # Neighbor search (optimized - batch query for large grids)
    m = target_coords.shape[0]
    
    if progress_callback:
        progress_callback(10, f"Querying neighbors for {m:,} blocks...")
    
    if NEIGHBOR_SEARCHER_AVAILABLE:
        # NeighborSearcher is optimized for batch queries
        searcher = NeighborSearcher(data_coords, anisotropy_params=anisotropy_params)
        neighbor_indices, _ = searcher.search(
            target_coords=target_coords,
            n_neighbors=n_neighbors,
            max_distance=max_distance
        )
        if anisotropy_params:
            data_coords_for_kernel = searcher.get_transformed_coords()
        else:
            data_coords_for_kernel = data_coords
        target_coords_for_kernel = target_coords_aniso if anisotropy_params else target_coords
    else:
        # Fallback to KDTree - batch query with distance filtering
        tree = KDTree(data_coords_aniso)
        # Query more neighbors than needed, then filter by distance
        k_query = min(n_neighbors * 3, len(data_coords_aniso))  # Query 3x more for distance filtering
        distances, neighbor_indices = tree.query(target_coords_aniso, k=k_query)

        # Apply distance filter consistently with NeighborSearcher
        if max_distance is not None:
            # Convert indices to array if needed
            if neighbor_indices.ndim == 1:
                neighbor_indices = neighbor_indices.reshape(-1, 1)
                distances = distances.reshape(-1, 1)

            # Filter by distance for each target point
            filtered_indices = []
            for i in range(len(target_coords_aniso)):
                valid_mask = distances[i] <= max_distance
                valid_indices = neighbor_indices[i][valid_mask][:n_neighbors]  # Cap at n_neighbors
                # Pad with -1 if needed
                if len(valid_indices) < n_neighbors:
                    padded = np.full(n_neighbors, -1, dtype=int)
                    padded[:len(valid_indices)] = valid_indices
                    filtered_indices.append(padded)
                else:
                    filtered_indices.append(valid_indices)

            neighbor_indices = np.array(filtered_indices)
        else:
            # No distance filter, just ensure proper shape
            if neighbor_indices.ndim == 1:
                neighbor_indices = neighbor_indices.reshape(-1, 1)

        data_coords_for_kernel = data_coords_aniso
        target_coords_for_kernel = target_coords_aniso
    
    if progress_callback:
        progress_callback(20, "Neighbor search complete")
    
    # Use chunked processing for progress updates (always chunk, even for small grids)
    # This allows smooth progress bar updates instead of jumping from 30% to 100%
    CHUNK_SIZE = 5000  # Process 5k blocks at a time for smooth progress updates
    
    n_chunks = max(1, int(np.ceil(m / CHUNK_SIZE)))
    
    if n_chunks == 1 and progress_callback is None:
        # Very small grid with no progress callback - process all at once (fastest)
        (estimates, variances, neighbour_counts,
         sum_weights, pct_neg_weights,
         min_dist, avg_dist, max_dist,
         solver_status) = run_sk_kernel(
            target_coords_for_kernel,
            neighbor_indices,
            data_coords_for_kernel,
            data_values,
            params
        )
    else:
        # Chunked processing with progress updates
        if n_chunks > 1:
            logger.info(f"Simple Kriging: Processing {m:,} blocks in {n_chunks} chunks")
        
        estimates = np.full(m, np.nan)
        variances = np.full(m, np.nan)
        neighbour_counts = np.full(m, 0, dtype=int)

        # Diagnostic arrays
        sum_weights = np.full(m, 0.0)
        pct_neg_weights = np.full(m, 0.0)
        min_dist = np.full(m, np.nan)
        avg_dist = np.full(m, np.nan)
        max_dist = np.full(m, np.nan)
        solver_status = np.full(m, 0, dtype=int)

        for chunk_idx in range(n_chunks):
            start_idx = chunk_idx * CHUNK_SIZE
            end_idx = min(start_idx + CHUNK_SIZE, m)

            # Progress: 30% to 95% is the kriging phase
            chunk_progress = 30 + int(65 * ((chunk_idx + 1) / n_chunks))
            if progress_callback:
                progress_callback(
                    chunk_progress,
                    f"SK: {chunk_progress}% ({end_idx:,}/{m:,} blocks)"
                )

            # Extract chunk data
            chunk_targets = target_coords_for_kernel[start_idx:end_idx]
            chunk_indices = neighbor_indices[start_idx:end_idx]

            # Run kernel on chunk
            (chunk_est, chunk_var, chunk_nn,
             chunk_sum_w, chunk_pct_neg,
             chunk_min_d, chunk_avg_d, chunk_max_d,
             chunk_slv) = run_sk_kernel(
                chunk_targets,
                chunk_indices,
                data_coords_for_kernel,
                data_values,
                params
            )

            estimates[start_idx:end_idx] = chunk_est
            variances[start_idx:end_idx] = chunk_var
            neighbour_counts[start_idx:end_idx] = chunk_nn
            sum_weights[start_idx:end_idx] = chunk_sum_w
            pct_neg_weights[start_idx:end_idx] = chunk_pct_neg
            min_dist[start_idx:end_idx] = chunk_min_d
            avg_dist[start_idx:end_idx] = chunk_avg_d
            max_dist[start_idx:end_idx] = chunk_max_d
            solver_status[start_idx:end_idx] = chunk_slv
    
    if progress_callback:
        progress_callback(100, "Simple Kriging complete")

    # Logging
    ok = ~np.isnan(estimates)
    n_ok = int(ok.sum())
    logger.info(f"Simple Kriging completed: {n_ok}/{m} valid estimates")
    if n_ok:
        vals = estimates[ok]
        nn_vals = neighbour_counts[ok]
        logger.info(f"Estimate range [{vals.min():.3f}, {vals.max():.3f}], mean {vals.mean():.3f}")
        logger.info(f"Neighbour count range [{nn_vals.min()}, {nn_vals.max()}], mean {nn_vals.mean():.1f}")

    # Package diagnostics
    diagnostics = {
        'sum_weights': sum_weights,
        'pct_negative_weights': pct_neg_weights,
        'min_distance': min_dist,
        'avg_distance': avg_dist,
        'max_distance': max_dist,
        'solver_status': solver_status
    }

    return estimates, variances, neighbour_counts, diagnostics


def simple_kriging_3d_standard(
    data_coords: np.ndarray,
    data_values: np.ndarray,
    target_coords: np.ndarray,
    variogram_params: Dict,
    n_neighbors: int = 12,
    max_distance: Optional[float] = None,
    model_type: str = "spherical",
    global_mean: Optional[float] = None,
    progress_callback=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standard Simple Kriging implementation (no Numba).
    Same structure as ordinary_kriging_3d() for consistency.
    """
    rng = float(variogram_params["range"])
    sill_total = float(variogram_params["sill"])
    nug = float(variogram_params.get("nugget", 0.0))
    
    if global_mean is None:
        global_mean = float(np.nanmean(data_values))
    
    gamma_fun = get_variogram_function(model_type)
    
    # Handle anisotropy
    anisotropy_params = variogram_params.get("anisotropy", None)
    if anisotropy_params:
        azimuth = float(anisotropy_params.get("azimuth", 0.0))
        dip = float(anisotropy_params.get("dip", 0.0))
        major_range = float(anisotropy_params.get("major_range", rng))
        minor_range = float(anisotropy_params.get("minor_range", rng))
        vert_range = float(anisotropy_params.get("vert_range", rng))
        
        data_coords_aniso = apply_anisotropy(
            data_coords, azimuth, dip, major_range, minor_range, vert_range
        )
        target_coords_aniso = apply_anisotropy(
            target_coords, azimuth, dip, major_range, minor_range, vert_range
        )
        effective_range = 1.0
        range_geometric_mean = (major_range * minor_range * vert_range) ** (1.0 / 3.0)
    else:
        data_coords_aniso = data_coords
        target_coords_aniso = target_coords
        effective_range = rng
        range_geometric_mean = rng
    
    # Build tree
    use_tree = KDTree is not None
    tree = KDTree(data_coords_aniso) if use_tree else None
    
    m = target_coords.shape[0]
    estimates = np.full(m, np.nan, dtype=float)
    variances = np.full(m, np.nan, dtype=float)
    neighbour_counts = np.full(m, 0, dtype=int)
    
    # Reduce progress callback frequency for large grids (every 10% instead of 5%)
    report_every = max(1, m // 10)
    
    for i in range(m):
        p_aniso = target_coords_aniso[i]
        
        # Neighbor search
        if use_tree:
            d, nbr_idx = tree.query(p_aniso, k=min(n_neighbors, len(data_coords)))
            if np.isscalar(d):
                d = np.array([d])
                nbr_idx = np.array([nbr_idx])
            if max_distance is not None:
                max_dist_aniso = max_distance / range_geometric_mean if anisotropy_params else max_distance
                mask = d <= max_dist_aniso
                d = d[mask]
                nbr_idx = nbr_idx[mask]
        else:
            d_all = np.linalg.norm(data_coords_aniso - p_aniso, axis=1)
            order = np.argsort(d_all)[:n_neighbors]
            nbr_idx = order
            d = d_all[order]
        
        k = len(nbr_idx)
        if k < 1:
            estimates[i] = global_mean
            variances[i] = sill_total
            neighbour_counts[i] = 0
            continue
        
        # Get neighbor coords and values
        pts = data_coords_aniso[nbr_idx]
        vals = data_values[nbr_idx]
        
        # Build covariance matrix (K x K)
        dist_mat = cdist(pts, pts)
        gamma_mat = gamma_fun(dist_mat, effective_range, sill_total - nug, nug)
        cov_mat = sill_total - gamma_mat
        
        # Add regularization
        cov_mat += np.eye(k) * max(1e-10 * np.max(np.diag(cov_mat)), 1e-9)
        
        # RHS: covariance between target and neighbors
        d_to_target = np.linalg.norm(pts - p_aniso, axis=1)
        gamma_target = gamma_fun(d_to_target, effective_range, sill_total - nug, nug)
        cov_target = sill_total - gamma_target
        
        # Solve
        try:
            weights = solve(cov_mat, cov_target, assume_a='sym')
        except LinAlgError:
            estimates[i] = global_mean
            variances[i] = sill_total
            continue
        
        # Simple Kriging estimate
        est = global_mean + np.sum(weights * (vals - global_mean))
        
        # Variance
        var = sill_total - np.dot(weights, cov_target)
        
        estimates[i] = est
        variances[i] = max(0.0, var)
        neighbour_counts[i] = k
        
        if progress_callback and (i % report_every == 0 or i == m - 1):
            pct = (i + 1) * 100 // m
            progress_callback(int(pct), f"SK: {pct}% ({i+1}/{m})")

    return estimates, variances, neighbour_counts


# =========================================================
# WRAPPER FOR OLD API (SKParameters interface)
# =========================================================

def simple_kriging_3d(
    data_coords: np.ndarray,
    data_values: np.ndarray,
    target_coords: np.ndarray,
    params: SKParameters,
    progress_callback=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Simple Kriging using SKParameters dataclass (legacy interface).
    Internally calls simple_kriging_fast() for best performance.
    
    Returns:
        estimates: Kriged estimates
        variances: Kriging variances
        neighbour_counts: Number of neighbors used per point
        diagnostics: Dict with solver/stability diagnostics
    """
    # Convert SKParameters to variogram_params dict
    variogram_params = {
        "range": params.range_major,
        "sill": params.sill,
        "nugget": params.nugget,
    }
    
    # Add anisotropy if non-isotropic
    if (params.range_major != params.range_minor or
        params.range_minor != params.range_vert or
        params.azimuth != 0 or params.dip != 0):
        variogram_params["anisotropy"] = {
            "azimuth": params.azimuth,
            "dip": params.dip,
            "major_range": params.range_major,
            "minor_range": params.range_minor,
            "vert_range": params.range_vert,
        }
    
    return simple_kriging_fast(
        data_coords=data_coords,
        data_values=data_values,
        target_coords=target_coords,
        variogram_params=variogram_params,
        n_neighbors=params.ndmax,
        max_distance=params.max_search_radius,
        model_type=params.variogram_type,
        global_mean=params.global_mean,
        progress_callback=progress_callback,
    )


# simple_kriging_3d_full removed - was never called and redundant with controller building results directly
