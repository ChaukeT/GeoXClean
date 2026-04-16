"""
Shared Numba kernels for all kriging methods.

SINGLE SOURCE for covariance calculation and Gaussian elimination solver.
All kriging engines (OK, SK, UK, IK, CoK) import from this module.

Convention:
    sill parameter is TOTAL sill (nugget + partial sill).
    C(0) = total_sill.
    C(h) = total_sill - gamma(h).
"""

import numpy as np

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


@njit(fastmath=True, cache=True)
def covariance_from_variogram(d, range_val, total_sill, nugget, model_type):
    """
    Calculate Covariance C(h) = TotalSill - Gamma(h).

    AUDIT FIX (V-NEW-001): Standardized sill interpretation.

    Parameters
    ----------
    d : float
        Distance (lag)
    range_val : float
        Range parameter
    total_sill : float
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
    partial_sill = max(total_sill - nugget, 0.0)

    if d < 1e-9:
        return total_sill  # At h=0, covariance = total sill

    gamma = 0.0
    if model_type == 0:  # Spherical
        if d >= range_val:
            gamma = total_sill  # Total sill at range
        else:
            r = d / range_val
            gamma = nugget + partial_sill * (1.5 * r - 0.5 * r ** 3)
    elif model_type == 1:  # Exponential
        gamma = nugget + partial_sill * (1.0 - np.exp(-3.0 * d / range_val))
    elif model_type == 2:  # Gaussian
        gamma = nugget + partial_sill * (1.0 - np.exp(-3.0 * (d / range_val) ** 2))

    return total_sill - gamma


def covariance_from_variogram_np(d, range_val, total_sill, nugget, model_type):
    """
    Calculate covariance C(h) - pure Python/numpy version.

    Same logic as covariance_from_variogram but without Numba JIT.
    Used when Numba is not available or in non-JIT contexts.

    Parameters
    ----------
    d : float
        Distance (lag)
    range_val : float
        Range parameter
    total_sill : float
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
    partial_sill = max(total_sill - nugget, 0.0)

    if d < 1e-9:
        return total_sill

    if model_type == 0:  # Spherical
        if d >= range_val:
            gamma = total_sill
        else:
            r = d / range_val
            gamma = nugget + partial_sill * (1.5 * r - 0.5 * r ** 3)
    elif model_type == 1:  # Exponential
        gamma = nugget + partial_sill * (1.0 - np.exp(-3.0 * d / range_val))
    elif model_type == 2:  # Gaussian
        gamma = nugget + partial_sill * (1.0 - np.exp(-3.0 * (d / range_val) ** 2))
    else:
        gamma = total_sill  # Fallback

    return total_sill - gamma


@njit(fastmath=True, cache=True)
def gaussian_elimination_solve(A, b, n):
    """
    Gaussian elimination with partial pivoting. SINGLE SOURCE for all kriging.

    Solves A * x = b in-place (modifies A and b).

    Parameters
    ----------
    A : np.ndarray
        (n, n) coefficient matrix (MODIFIED in-place)
    b : np.ndarray
        (n,) right-hand side vector (MODIFIED in-place)
    n : int
        System dimension

    Returns
    -------
    weights : np.ndarray
        (n,) solution vector (zeros if singular)
    success : bool
        True if solve succeeded, False if matrix is singular
    """
    # GS-15 fix: track pivot magnitudes for condition number estimation
    max_pivot = 0.0
    min_pivot = 1e300

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
            return np.zeros(n), False

        # Track pivot bounds for condition estimate
        if max_val > max_pivot:
            max_pivot = max_val
        if max_val < min_pivot:
            min_pivot = max_val

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

    # GS-15: check condition number estimate (max_pivot / min_pivot)
    # If ratio exceeds 1e10, the system is ill-conditioned and weights unreliable.
    # Return failure to force the caller to use a fallback (e.g., truncated search).
    if min_pivot > 0.0 and max_pivot / min_pivot > 1e10:
        return np.zeros(n), False

    # Back substitution
    weights = np.zeros(n)
    for k in range(n - 1, -1, -1):
        if abs(A[k, k]) < 1e-12:
            return np.zeros(n), False
        weights[k] = b[k]
        for col in range(k + 1, n):
            weights[k] -= A[k, col] * weights[col]
        weights[k] /= A[k, k]

    return weights, True
