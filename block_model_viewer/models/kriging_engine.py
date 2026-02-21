"""
Numba-Accelerated Kriging Engine

High-performance JIT-compiled kriging kernel for large-scale 3D kriging.
Uses Numba to compile the core kriging loop into machine code and run in parallel.
"""

import numpy as np

try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Create dummy decorators if Numba not available
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range


# ==========================================
# 1. JIT-Compiled Variogram Models
# ==========================================

@jit(nopython=True, fastmath=True)
def variogram_spherical(h, rng, sill, nugget):
    """Spherical variogram model: γ(h) = nugget + sill * (1.5*(h/r) - 0.5*(h/r)³) for h <= r, else nugget + sill"""
    if h >= rng:
        return nugget + sill
    ratio = h / rng
    return nugget + sill * (1.5 * ratio - 0.5 * ratio * ratio * ratio)


@jit(nopython=True, fastmath=True)
def variogram_exponential(h, rng, sill, nugget):
    """Exponential variogram model: γ(h) = nugget + sill * (1 - exp(-3h/r))"""
    return nugget + sill * (1.0 - np.exp(-3.0 * h / rng))


@jit(nopython=True, fastmath=True)
def variogram_gaussian(h, rng, sill, nugget):
    """Gaussian variogram model: γ(h) = nugget + sill * (1 - exp(-3(h/r)²))"""
    return nugget + sill * (1.0 - np.exp(-3.0 * (h / rng)**2))


@jit(nopython=True, fastmath=True)
def calc_covariance(dist, model_type, rng, sill, nugget):
    """
    Calculates Covariance C(h) = TotalSill - Gamma(h)
    
    Parameters
    ----------
    dist : float
        Distance between points
    model_type : int
        0 = spherical, 1 = exponential, 2 = gaussian
    rng : float
        Variogram range
    sill : float
        Partial sill (total sill - nugget)
    nugget : float
        Nugget effect
        
    Returns
    -------
    float
        Covariance value
    """
    total_sill = sill + nugget
    
    # Calculate variogram gamma(h)
    gamma = 0.0
    if model_type == 0:
        gamma = variogram_spherical(dist, rng, sill, nugget)
    elif model_type == 1:
        gamma = variogram_exponential(dist, rng, sill, nugget)
    elif model_type == 2:
        gamma = variogram_gaussian(dist, rng, sill, nugget)
    
    # Covariance = TotalSill - Gamma
    return total_sill - gamma


# ==========================================
# 2. The Core Kriging Kernel (Parallelized)
# ==========================================

@jit(nopython=True, fastmath=True)
def _solve_single_ok_point(
    i,
    target_coords,      # (M, 3)
    neighbor_indices,   # (M, K)
    data_coords,        # (N, 3)
    data_values,        # (N,)
    rng, sill, nugget, model_type, total_sill, k_neighbors
):
    """
    Solve Ordinary Kriging for a single target point.
    Separated to avoid control flow issues in prange.
    """
    indices = neighbor_indices[i]
    
    # Extract coordinates and values for valid neighbors
    local_coords = np.zeros((k_neighbors, 3))
    local_values = np.zeros(k_neighbors)
    
    valid_count = 0
    for n in range(k_neighbors):
        idx = indices[n]
        if idx >= 0:
            local_coords[n, 0] = data_coords[idx, 0]
            local_coords[n, 1] = data_coords[idx, 1]
            local_coords[n, 2] = data_coords[idx, 2]
            local_values[n] = data_values[idx]
            valid_count += 1
    
    if valid_count < 3:
        return np.nan, np.nan

    # Build LHS Matrix (K) and RHS Vector
    dim = valid_count + 1
    mat_k = np.zeros((dim, dim))
    rhs = np.zeros(dim)
    
    # Fill Covariance Matrix
    for r in range(valid_count):
        for c in range(valid_count):
            dx = local_coords[r, 0] - local_coords[c, 0]
            dy = local_coords[r, 1] - local_coords[c, 1]
            dz = local_coords[r, 2] - local_coords[c, 2]
            d = np.sqrt(dx*dx + dy*dy + dz*dz)
            cov = calc_covariance(d, model_type, rng, sill, nugget)
            mat_k[r, c] = cov

        # Lagrange multipliers
        mat_k[r, valid_count] = 1.0
        mat_k[valid_count, r] = 1.0
        
        # RHS
        tx = local_coords[r, 0] - target_coords[i, 0]
        ty = local_coords[r, 1] - target_coords[i, 1]
        tz = local_coords[r, 2] - target_coords[i, 2]
        d_target = np.sqrt(tx*tx + ty*ty + tz*tz)
        rhs[r] = calc_covariance(d_target, model_type, rng, sill, nugget)

    rhs[valid_count] = 1.0

    # Add regularization
    max_diag = 0.0
    for d_i in range(valid_count):
        if mat_k[d_i, d_i] > max_diag:
            max_diag = mat_k[d_i, d_i]
    reg_value = max(1e-10 * max_diag, 1e-9)
    for d_i in range(valid_count):
        mat_k[d_i, d_i] += reg_value

    # Solve using Gaussian elimination with partial pivoting
    A = mat_k[:dim, :dim].copy()
    b = rhs[:dim].copy()
    
    # Forward elimination
    for k in range(dim):
        max_row = k
        max_val = abs(A[k, k])
        for row in range(k + 1, dim):
            if abs(A[row, k]) > max_val:
                max_val = abs(A[row, k])
                max_row = row
        
        if max_val < 1e-12:
            return np.nan, np.nan
        
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
            return np.nan, np.nan
        weights[k] = b[k]
        for col in range(k + 1, dim):
            weights[k] -= A[k, col] * weights[col]
        weights[k] /= A[k, k]
    
    # Compute Estimate
    est = 0.0
    for w_i in range(valid_count):
        est += weights[w_i] * local_values[w_i]
    
    # Sanity check
    data_min = local_values[0]
    data_max = local_values[0]
    data_sum = 0.0
    for w_i in range(valid_count):
        if local_values[w_i] < data_min:
            data_min = local_values[w_i]
        if local_values[w_i] > data_max:
            data_max = local_values[w_i]
        data_sum += local_values[w_i]
    data_mean = data_sum / valid_count
    data_range = data_max - data_min
    
    if data_range > 0 and abs(est - data_mean) > 10 * data_range:
        est = data_mean
        
    # Variance
    mu = weights[valid_count]
    sum_w_cov = 0.0
    for w_i in range(valid_count):
        sum_w_cov += weights[w_i] * rhs[w_i]
    
    var = total_sill - sum_w_cov - mu
    
    if var < 0.0:
        var = 0.0
    
    return est, var


@jit(nopython=True, parallel=True, fastmath=True)
def run_kriging_kernel(
    target_coords,      # (M, 3)
    neighbor_indices,   # (M, K) - Indices of neighbors for each block
    data_coords,        # (N, 3)
    data_values,        # (N,)
    params              # Array: [range, sill, nugget, model_code]
):
    """
    Solves the Kriging system for M blocks in parallel.
    
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
    total_sill = sill + nugget

    # Output arrays
    estimates = np.full(n_targets, np.nan)
    variances = np.full(n_targets, np.nan)
    
    # Pure prange loop - no continue, no try/except
    for i in prange(n_targets):
        est, var = _solve_single_ok_point(
            i, target_coords, neighbor_indices, data_coords, data_values,
            rng, sill, nugget, model_type, total_sill, k_neighbors
        )
        estimates[i] = est
        variances[i] = var

    return estimates, variances

