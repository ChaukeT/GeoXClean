"""
ARBF Gaussian Process Regression — Numba-accelerated.

Posterior mean and variance via Cholesky forward-substitution for
numerically stable batch prediction (avoids explicit matrix inverse).

    v = L^{-1} k_aug(x)          (forward substitution, BLAS dtrsm)
    s^2(x) = phi(0) - ||v||^2

For augmented systems with drift (M > 0), uses LU factorisation
with batched solve instead of explicit inverse.

References
----------
- Rasmussen & Williams (2006), Gaussian Processes for Machine Learning.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

try:
    from numba import njit, prange
except ImportError:
    def njit(*args, **kwargs):
        def wrapper(fn):
            return fn
        if args and callable(args[0]):
            return args[0]
        return wrapper
    prange = range

from scipy.linalg import cho_factor, cho_solve, lu_factor, lu_solve, solve_triangular

from .kernels import evaluate_kernel, kernel_at_zero
from .utils import (
    anisotropic_distance,
    clamp_variance,
    condition_number_estimate,
    pairwise_anisotropic_distance,
    rotation_matrix,
    scale_matrix,
    stable_cholesky,
)

logger = logging.getLogger(__name__)

_GPR_VERSION = "2026-03-10-v4-cholesky"


# ---------------------------------------------------------------------------
# Numba-accelerated core operations
# ---------------------------------------------------------------------------


@njit(cache=True, parallel=True)
def _aniso_dist_and_kernel_spheroidal(
    query: np.ndarray,       # (B, 3)
    samples: np.ndarray,     # (N, 3)
    T: np.ndarray,           # (3, 3) combined transform = S @ R
    sill: float,
    alpha: float,
) -> np.ndarray:
    """Fused anisotropic distance + spheroidal kernel evaluation.

    Returns sill * phi(||T(q-s)||) as (B, N) matrix.
    Special-cases alpha=1,1.5,2 to avoid expensive pow() calls.
    """
    B = query.shape[0]
    N = samples.shape[0]
    out = np.empty((B, N), dtype=np.float64)

    alpha_is_1 = abs(alpha - 1.0) < 1e-12
    alpha_is_1_5 = abs(alpha - 1.5) < 1e-12
    alpha_is_2 = abs(alpha - 2.0) < 1e-12

    for i in prange(B):
        for j in range(N):
            dx = query[i, 0] - samples[j, 0]
            dy = query[i, 1] - samples[j, 1]
            dz = query[i, 2] - samples[j, 2]

            tx = T[0, 0] * dx + T[0, 1] * dy + T[0, 2] * dz
            ty = T[1, 0] * dx + T[1, 1] * dy + T[1, 2] * dz
            tz = T[2, 0] * dx + T[2, 1] * dy + T[2, 2] * dz

            base = 1.0 + tx * tx + ty * ty + tz * tz

            if alpha_is_1:
                out[i, j] = sill / base
            elif alpha_is_1_5:
                out[i, j] = sill / (base * np.sqrt(base))
            elif alpha_is_2:
                out[i, j] = sill / (base * base)
            else:
                out[i, j] = sill * base ** (-alpha)

    return out


# ---------------------------------------------------------------------------
# Fused predict-mean kernels (no intermediate (B,N) matrix allocation)
# ---------------------------------------------------------------------------


@njit(cache=True, parallel=True)
def _predict_mean_spheroidal_fused(
    query: np.ndarray,       # (B, 3)
    samples: np.ndarray,     # (N, 3)
    T: np.ndarray,           # (3, 3) combined transform = S @ R
    sill: float,
    alpha: float,
    weights: np.ndarray,     # (N,)
    poly_c0: float,          # constant drift coefficient
) -> np.ndarray:
    """Fused kernel evaluation + dot product for spheroidal mean prediction.

    Computes f(x) = sum_j [ sill*(1+||T(x-s_j)||^2)^(-alpha) * w_j ] + c0
    without materializing the (B, N) kernel matrix.

    Memory: O(B) output only (vs O(B*N) for materialized kernel).
    Special-cases alpha=1 and alpha=1.5 to avoid expensive pow() calls.
    """
    B = query.shape[0]
    N = samples.shape[0]
    out = np.empty(B, dtype=np.float64)

    # Detect integer/half-integer alpha for fast path
    alpha_is_1 = abs(alpha - 1.0) < 1e-12
    alpha_is_1_5 = abs(alpha - 1.5) < 1e-12
    alpha_is_2 = abs(alpha - 2.0) < 1e-12

    for i in prange(B):
        acc = poly_c0
        qx = query[i, 0]
        qy = query[i, 1]
        qz = query[i, 2]
        for j in range(N):
            dx = qx - samples[j, 0]
            dy = qy - samples[j, 1]
            dz = qz - samples[j, 2]

            tx = T[0, 0] * dx + T[0, 1] * dy + T[0, 2] * dz
            ty = T[1, 0] * dx + T[1, 1] * dy + T[1, 2] * dz
            tz = T[2, 0] * dx + T[2, 1] * dy + T[2, 2] * dz

            base = 1.0 + tx * tx + ty * ty + tz * tz

            if alpha_is_1:
                k = sill / base
            elif alpha_is_1_5:
                k = sill / (base * np.sqrt(base))
            elif alpha_is_2:
                k = sill / (base * base)
            else:
                k = sill * base ** (-alpha)

            acc += k * weights[j]
        out[i] = acc

    return out


@njit(cache=True, parallel=True)
def _predict_mean_lva_spheroidal_fused(
    query: np.ndarray,       # (B, 3)
    samples: np.ndarray,     # (N, 3)
    T_query: np.ndarray,     # (B, 3, 3) per-query transform
    sill: float,
    alpha: float,
    weights: np.ndarray,     # (N,)
    poly_c0: float,          # constant drift coefficient
) -> np.ndarray:
    """Fused LVA kernel + dot product for spheroidal mean prediction.

    Same as _predict_mean_spheroidal_fused but with per-query T matrices.
    """
    B = query.shape[0]
    N = samples.shape[0]
    out = np.empty(B, dtype=np.float64)

    alpha_is_1 = abs(alpha - 1.0) < 1e-12
    alpha_is_1_5 = abs(alpha - 1.5) < 1e-12
    alpha_is_2 = abs(alpha - 2.0) < 1e-12

    for i in prange(B):
        acc = poly_c0
        qx = query[i, 0]
        qy = query[i, 1]
        qz = query[i, 2]
        for j in range(N):
            dx = qx - samples[j, 0]
            dy = qy - samples[j, 1]
            dz = qz - samples[j, 2]

            tx = T_query[i, 0, 0] * dx + T_query[i, 0, 1] * dy + T_query[i, 0, 2] * dz
            ty = T_query[i, 1, 0] * dx + T_query[i, 1, 1] * dy + T_query[i, 1, 2] * dz
            tz = T_query[i, 2, 0] * dx + T_query[i, 2, 1] * dy + T_query[i, 2, 2] * dz

            base = 1.0 + tx * tx + ty * ty + tz * tz

            if alpha_is_1:
                k = sill / base
            elif alpha_is_1_5:
                k = sill / (base * np.sqrt(base))
            elif alpha_is_2:
                k = sill / (base * base)
            else:
                k = sill * base ** (-alpha)

            acc += k * weights[j]
        out[i] = acc

    return out


# ---------------------------------------------------------------------------
# Matrix assembly
# ---------------------------------------------------------------------------


def build_polynomial_matrix(
    coords: np.ndarray,
    drift_type: str = "constant",
) -> np.ndarray:
    """Build polynomial basis matrix P for drift terms."""
    N = coords.shape[0]
    if drift_type == "none":
        return np.zeros((N, 0), dtype=np.float64)
    elif drift_type == "constant":
        return np.ones((N, 1), dtype=np.float64)
    elif drift_type == "linear":
        return np.column_stack([np.ones(N), coords])
    else:
        raise ValueError(f"Unknown drift_type '{drift_type}'")


@njit(cache=True, parallel=True)
def _pairwise_lva_dist_numba(
    coords: np.ndarray,   # (N, 3)
    T_all: np.ndarray,    # (N, 3, 3) per-point transform T_i = S @ R_i
) -> np.ndarray:
    """Numba-jitted pairwise LVA distance (symmetric midpoint averaging).

    For each pair (i, j): d = ||T_avg @ (x_i - x_j)||
    where T_avg = 0.5 * (T_i + T_j).

    Using T_i alone (former implementation) made the covariance matrix
    depend on the data row ordering: shuffling the input array produced
    different covariance matrices and different grade estimates.  The
    averaged transform is symmetric by construction (T_avg(i,j) ==
    T_avg(j,i)) so D[i,j] == D[j,i] regardless of order.
    """
    N = coords.shape[0]
    D = np.zeros((N, N), dtype=np.float64)

    for i in prange(N):
        for j in range(i + 1, N):
            dx = coords[i, 0] - coords[j, 0]
            dy = coords[i, 1] - coords[j, 1]
            dz = coords[i, 2] - coords[j, 2]

            # Midpoint-averaged transform (symmetric — order-invariant)
            t00 = 0.5 * (T_all[i, 0, 0] + T_all[j, 0, 0])
            t01 = 0.5 * (T_all[i, 0, 1] + T_all[j, 0, 1])
            t02 = 0.5 * (T_all[i, 0, 2] + T_all[j, 0, 2])
            t10 = 0.5 * (T_all[i, 1, 0] + T_all[j, 1, 0])
            t11 = 0.5 * (T_all[i, 1, 1] + T_all[j, 1, 1])
            t12 = 0.5 * (T_all[i, 1, 2] + T_all[j, 1, 2])
            t20 = 0.5 * (T_all[i, 2, 0] + T_all[j, 2, 0])
            t21 = 0.5 * (T_all[i, 2, 1] + T_all[j, 2, 1])
            t22 = 0.5 * (T_all[i, 2, 2] + T_all[j, 2, 2])

            tx = t00 * dx + t01 * dy + t02 * dz
            ty = t10 * dx + t11 * dy + t12 * dz
            tz = t20 * dx + t21 * dy + t22 * dz

            d = np.sqrt(tx * tx + ty * ty + tz * tz)
            D[i, j] = d
            D[j, i] = d

    return D


@njit(cache=True, parallel=True)
def _query_lva_dist_and_kernel_spheroidal(
    query: np.ndarray,      # (B, 3)
    samples: np.ndarray,    # (N, 3)
    T_query: np.ndarray,    # (B, 3, 3) per-query transform
    sill: float,
    alpha: float,
) -> np.ndarray:
    """Fused LVA distance + spheroidal kernel for query points.

    Returns sill * (1 + ||T_q @ (q - s)||^2)^(-alpha) as (B, N).
    """
    B = query.shape[0]
    N = samples.shape[0]
    out = np.empty((B, N), dtype=np.float64)

    alpha_is_1 = abs(alpha - 1.0) < 1e-12
    alpha_is_1_5 = abs(alpha - 1.5) < 1e-12
    alpha_is_2 = abs(alpha - 2.0) < 1e-12

    for i in prange(B):
        for j in range(N):
            dx = query[i, 0] - samples[j, 0]
            dy = query[i, 1] - samples[j, 1]
            dz = query[i, 2] - samples[j, 2]

            tx = T_query[i, 0, 0] * dx + T_query[i, 0, 1] * dy + T_query[i, 0, 2] * dz
            ty = T_query[i, 1, 0] * dx + T_query[i, 1, 1] * dy + T_query[i, 1, 2] * dz
            tz = T_query[i, 2, 0] * dx + T_query[i, 2, 1] * dy + T_query[i, 2, 2] * dz

            base = 1.0 + tx * tx + ty * ty + tz * tz

            if alpha_is_1:
                out[i, j] = sill / base
            elif alpha_is_1_5:
                out[i, j] = sill / (base * np.sqrt(base))
            elif alpha_is_2:
                out[i, j] = sill / (base * base)
            else:
                out[i, j] = sill * base ** (-alpha)

    return out


@njit(cache=True, parallel=True)
def _query_lva_dist_numba(
    query: np.ndarray,      # (B, 3)
    samples: np.ndarray,    # (N, 3)
    T_query: np.ndarray,    # (B, 3, 3) per-query transform
) -> np.ndarray:
    """Numba-jitted LVA distance from query to sample points.

    Returns (B, N) distance matrix.
    """
    B = query.shape[0]
    N = samples.shape[0]
    out = np.empty((B, N), dtype=np.float64)

    for i in prange(B):
        for j in range(N):
            dx = query[i, 0] - samples[j, 0]
            dy = query[i, 1] - samples[j, 1]
            dz = query[i, 2] - samples[j, 2]

            tx = T_query[i, 0, 0] * dx + T_query[i, 0, 1] * dy + T_query[i, 0, 2] * dz
            ty = T_query[i, 1, 0] * dx + T_query[i, 1, 1] * dy + T_query[i, 1, 2] * dz
            tz = T_query[i, 2, 0] * dx + T_query[i, 2, 1] * dy + T_query[i, 2, 2] * dz

            out[i, j] = np.sqrt(tx * tx + ty * ty + tz * tz)

    return out


def _pairwise_lva_distance(
    coords: np.ndarray,
    orientation_field,
    S: np.ndarray,
) -> np.ndarray:
    """Pairwise anisotropic distance using locally varying rotations.

    Pre-computes all rotation matrices via batch_interpolate, then
    dispatches to a numba-jitted kernel — zero Python loops.
    """
    coords = np.ascontiguousarray(coords, dtype=np.float64)
    S = np.ascontiguousarray(S, dtype=np.float64)

    # Batch-interpolate T = S @ R at every sample point
    T_all = orientation_field.batch_interpolate_T(coords, S)
    T_all = np.ascontiguousarray(T_all, dtype=np.float64)

    return _pairwise_lva_dist_numba(coords, T_all)


def _query_lva_distance(
    query: np.ndarray,
    samples: np.ndarray,
    orientation_field,
    S: np.ndarray,
) -> np.ndarray:
    """Anisotropic distance from query to sample points using LVA.

    Pre-computes rotation matrices at all query points via
    batch_interpolate, then dispatches to numba kernel.
    """
    query = np.ascontiguousarray(query, dtype=np.float64)
    samples = np.ascontiguousarray(samples, dtype=np.float64)
    S = np.ascontiguousarray(S, dtype=np.float64)

    T_query = orientation_field.batch_interpolate_T(query, S)
    T_query = np.ascontiguousarray(T_query, dtype=np.float64)

    return _query_lva_dist_numba(query, samples, T_query)


def assemble_kernel_matrix(
    coords: np.ndarray,
    kernel_type: str = "spheroidal",
    alpha: float = 1.0,
    sill: float = 1.0,
    range_: float = 100.0,
    nugget: float = 0.0,
    accuracy: float = 1e-6,
    R: Optional[np.ndarray] = None,
    S: Optional[np.ndarray] = None,
    orientation_field: Optional[object] = None,
    drift_type: str = "constant",
    use_geodesic: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build augmented kernel matrix K_aug (Eq. 5.1).

    K_aug = [ PHI + (nugget + accuracy)*I    P  ]
            [ P^T                            0  ]

    When *orientation_field* is provided, uses locally varying
    anisotropy: R(midpoint) for each sample pair (i, j).

    When *use_geodesic* is True and *orientation_field* is set, pairs
    whose local orientations differ by more than 15° use the geodesic
    path integral (Eq. 4.4) instead of the linear midpoint approximation.
    This is required for strongly folded deposits (Bushveld, Wits Basin)
    where the straight-line distance cuts through a fold hinge.
    """
    from .utils import geodesic_distance, should_use_geodesic

    N = coords.shape[0]

    if R is None:
        R = np.eye(3, dtype=np.float64)
    if S is None:
        S = scale_matrix(range_, range_, range_)

    if orientation_field is not None and use_geodesic:
        # Build T matrices at all sample points once
        T_all = orientation_field.batch_interpolate_T(coords, S)
        T_all = np.ascontiguousarray(T_all, dtype=np.float64)
        # Linear LVA distance as the base
        D_linear = _pairwise_lva_dist_numba(coords, T_all)
        # Identify pairs with significant orientation change
        D = D_linear.copy()
        n_geo = 0
        for i in range(N):
            for j in range(i + 1, N):
                if should_use_geodesic(T_all[i], T_all[j]):
                    d_geo = geodesic_distance(
                        coords[i:i+1], coords[j:j+1],
                        orientation_field, S, n_steps=10,
                    )[0, 0]
                    D[i, j] = d_geo
                    D[j, i] = d_geo
                    n_geo += 1
        if n_geo > 0:
            logger.info(
                "Geodesic distance: %d / %d pairs used path integration "
                "(orientation change > 15°)",
                n_geo, N * (N - 1) // 2,
            )
    elif orientation_field is not None:
        # B2 fix: LVA — compute distance matrix using local rotations
        D = _pairwise_lva_distance(coords, orientation_field, S)
    else:
        D = pairwise_anisotropic_distance(coords, R, S)
    PHI = sill * evaluate_kernel(D, kernel_type=kernel_type, alpha=alpha)

    diag_add = nugget + accuracy
    np.fill_diagonal(PHI, sill + diag_add)

    P = build_polynomial_matrix(coords, drift_type)
    M = P.shape[1]

    K_aug = np.zeros((N + M, N + M), dtype=np.float64)
    K_aug[:N, :N] = PHI
    K_aug[:N, N:] = P
    K_aug[N:, :N] = P.T

    return K_aug, P


def factorise_and_solve(
    K_aug: np.ndarray,
    values: np.ndarray,
    drift_type: str = "constant",
) -> Tuple[object, np.ndarray, np.ndarray]:
    """Factorise kernel matrix and solve for weights.

    Stores the factorisation (not the explicit inverse) for numerically
    stable variance computation via forward-substitution.

    Returns
    -------
    factorisation : np.ndarray or tuple
        Cholesky factor L (ndarray, lower triangular) when M == 0,
        or (lu, piv) tuple from ``lu_factor`` when M > 0.
    weights : np.ndarray
        (N,) RBF interpolation weights.
    poly_coeffs : np.ndarray
        (M,) polynomial drift coefficients.
    """
    N = len(values)
    M = K_aug.shape[0] - N

    z_aug = np.zeros(N + M, dtype=np.float64)
    z_aug[:N] = values

    # Condition check + regularisation
    cond = condition_number_estimate(K_aug)
    if cond > 1e10:
        diag_mean = float(np.mean(np.diag(K_aug[:N, :N])))
        boost = diag_mean * 1e-6 * min(cond / 1e10, 1e4)
        K_aug[:N, :N] += boost * np.eye(N, dtype=np.float64)
        logger.info(
            "Kernel cond %.2e > 1e10. Diagonal boost %.2e applied.",
            cond, boost,
        )

    K_aug = 0.5 * (K_aug + K_aug.T)

    if M == 0:
        L = stable_cholesky(K_aug)
        y = solve_triangular(L, z_aug, lower=True)
        x = solve_triangular(L.T, y, lower=False)
        factorisation = L  # Store L only (half the memory of K_inv)
    else:
        lu_piv = lu_factor(K_aug)
        x = lu_solve(lu_piv, z_aug)
        factorisation = lu_piv  # Store (lu, piv) tuple

    weights = x[:N]
    poly_coeffs = x[N:]

    return factorisation, weights, poly_coeffs


def solve_weights_from_factor(
    factorisation,
    values: np.ndarray,
    M: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Solve for GPR weights using a pre-computed factorisation.

    Reuses an existing Cholesky or LU factorisation from a prior call to
    ``factorise_and_solve``.  Only the right-hand side (grade values) changes
    between ILR components; the kernel matrix K_aug is identical for all
    variables estimated on the same sub-domain.

    This avoids repeated O(N³) Cholesky refactorisation when iterating over
    all D-1 ILR coordinates in the multi-component pipeline.

    Parameters
    ----------
    factorisation : np.ndarray or tuple
        Cholesky factor ``L`` (lower triangular ndarray) when ``M == 0``,
        or ``(lu, piv)`` tuple from ``lu_factor`` when ``M > 0``.
    values : np.ndarray
        (N,) grade values for the new variable (e.g. ILR component j).
    M : int
        Number of drift polynomial coefficients (0 for constant drift).

    Returns
    -------
    weights : np.ndarray
        (N,) RBF interpolation weights.
    poly_coeffs : np.ndarray
        (M,) polynomial drift coefficients.
    """
    N = len(values)
    z_aug = np.zeros(N + M, dtype=np.float64)
    z_aug[:N] = values
    if M == 0:
        L = factorisation
        y = solve_triangular(L, z_aug, lower=True)
        x = solve_triangular(L.T, y, lower=False)
    else:
        lu_piv = factorisation
        x = lu_solve(lu_piv, z_aug)
    return x[:N].copy(), x[N:].copy()


# ---------------------------------------------------------------------------
# L_inv precomputation (call once at subdomain-fit time; reuse across blocks)
# ---------------------------------------------------------------------------


def compute_l_inv(factorisation) -> Optional[np.ndarray]:
    """Compute L^{-1} from a Cholesky factorisation, or None for LU path.

    Store the result on SubDomain.l_inv immediately after fitting so that
    predict_mean_and_variance / predict_variance can use a single dgemm
    call per batch instead of a fresh dtrsm solve.

    Returns None when factorisation is an LU tuple (no benefit — lu_solve
    already does a single pair of triangular solves per call).
    """
    if isinstance(factorisation, tuple):
        return None  # LU path — no precomputation benefit
    return solve_triangular(
        factorisation, np.eye(factorisation.shape[0], dtype=np.float64), lower=True
    )


# ---------------------------------------------------------------------------
# Prediction (batch, using pre-computed inverse)
# ---------------------------------------------------------------------------


def _compute_kernel_batch(
    batch: np.ndarray,
    sample_coords: np.ndarray,
    orientation_field,
    S: np.ndarray,
    T: np.ndarray,
    R: np.ndarray,
    kernel_type: str,
    alpha: float,
    sill: float,
) -> np.ndarray:
    """Compute kernel vector k(batch, samples) using best available path.

    Fuses distance+kernel for spheroidal when possible (both LVA and non-LVA).
    """
    if orientation_field is not None and kernel_type == "spheroidal":
        # Fused LVA + spheroidal: numba, no intermediate distance matrix
        batch_c = np.ascontiguousarray(batch, dtype=np.float64)
        samples_c = np.ascontiguousarray(sample_coords, dtype=np.float64)
        S_c = np.ascontiguousarray(S, dtype=np.float64)
        T_query = orientation_field.batch_interpolate_T(batch_c, S_c)
        T_query = np.ascontiguousarray(T_query, dtype=np.float64)
        return _query_lva_dist_and_kernel_spheroidal(
            batch_c, samples_c, T_query, sill, alpha,
        )
    elif orientation_field is not None:
        # LVA + non-spheroidal: compute distance, then evaluate kernel
        D = _query_lva_distance(batch, sample_coords, orientation_field, S)
        return sill * evaluate_kernel(D, kernel_type=kernel_type, alpha=alpha)
    elif kernel_type == "spheroidal":
        # Non-LVA spheroidal: existing fused numba kernel
        return _aniso_dist_and_kernel_spheroidal(
            np.ascontiguousarray(batch, dtype=np.float64),
            np.ascontiguousarray(sample_coords, dtype=np.float64),
            T, sill, alpha,
        )
    else:
        D = anisotropic_distance(batch, sample_coords, R, S)
        return sill * evaluate_kernel(D, kernel_type=kernel_type, alpha=alpha)


def predict_mean(
    query_points: np.ndarray,
    sample_coords: np.ndarray,
    weights: np.ndarray,
    poly_coeffs: np.ndarray,
    kernel_type: str = "spheroidal",
    alpha: float = 1.0,
    sill: float = 1.0,
    range_: float = 100.0,
    R: Optional[np.ndarray] = None,
    S: Optional[np.ndarray] = None,
    drift_type: str = "constant",
    orientation_field: Optional[object] = None,
) -> np.ndarray:
    """Evaluate posterior mean at query points.

    For spheroidal kernel with constant drift (the common case), uses a
    fused numba kernel that computes k(x*)^T @ w + c0 inline without
    materializing the (B, N) kernel matrix — reduces memory from O(B*N)
    to O(B) and eliminates allocation overhead.
    """
    if R is None:
        R = np.eye(3, dtype=np.float64)
    if S is None:
        S = scale_matrix(range_, range_, range_)

    query_points = np.ascontiguousarray(np.atleast_2d(query_points), dtype=np.float64)
    sample_coords = np.ascontiguousarray(sample_coords, dtype=np.float64)
    B = query_points.shape[0]
    T = np.ascontiguousarray((S @ R), dtype=np.float64)

    # Fast fused path: spheroidal kernel + constant drift
    # Avoids (B, N) kernel matrix allocation entirely.
    can_fuse = (kernel_type == "spheroidal" and drift_type == "constant"
                and len(poly_coeffs) == 1)

    if can_fuse and orientation_field is None:
        return _predict_mean_spheroidal_fused(
            query_points, sample_coords, T, sill, alpha,
            np.ascontiguousarray(weights, dtype=np.float64),
            float(poly_coeffs[0]),
        )

    if can_fuse and orientation_field is not None:
        # Pre-compute per-query T matrices, then fused LVA path
        S_c = np.ascontiguousarray(S, dtype=np.float64)
        T_query = orientation_field.batch_interpolate_T(query_points, S_c)
        T_query = np.ascontiguousarray(T_query, dtype=np.float64)
        return _predict_mean_lva_spheroidal_fused(
            query_points, sample_coords, T_query, sill, alpha,
            np.ascontiguousarray(weights, dtype=np.float64),
            float(poly_coeffs[0]),
        )

    # Fallback: materialized kernel matrix path (non-spheroidal or non-constant drift)
    BATCH = 50000  # Wider BLAS panels → better throughput (was 10000)
    estimates = np.zeros(B, dtype=np.float64)

    for start in range(0, B, BATCH):
        end = min(start + BATCH, B)
        batch = query_points[start:end]

        k_batch = _compute_kernel_batch(
            batch, sample_coords, orientation_field,
            S, T, R, kernel_type, alpha, sill,
        )

        est = k_batch @ weights
        P_batch = build_polynomial_matrix(batch, drift_type)
        if P_batch.shape[1] > 0 and len(poly_coeffs) > 0:
            est += P_batch @ poly_coeffs
        estimates[start:end] = est

    return estimates


def predict_variance(
    query_points: np.ndarray,
    sample_coords: np.ndarray,
    factorisation,
    kernel_type: str = "spheroidal",
    alpha: float = 1.0,
    sill: float = 1.0,
    range_: float = 100.0,
    nugget: float = 0.0,
    R: Optional[np.ndarray] = None,
    S: Optional[np.ndarray] = None,
    drift_type: str = "constant",
    orientation_field: Optional[object] = None,
    l_inv: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute posterior variance via forward-substitution.

    Cholesky path (M == 0):
        v = L^{-1} k_aug  (solve_triangular)
        s^2 = phi_0 - ||v||^2

    LU path (M > 0):
        v = lu_solve(fact, k_aug)
        s^2 = phi_0 - k_aug^T v
    """
    if R is None:
        R = np.eye(3, dtype=np.float64)
    if S is None:
        S = scale_matrix(range_, range_, range_)

    query_points = np.atleast_2d(query_points)
    B = query_points.shape[0]
    phi_0 = sill
    is_cholesky = not isinstance(factorisation, tuple)

    T = np.ascontiguousarray((S @ R), dtype=np.float64)
    BATCH = 50000  # Wider BLAS calls → better throughput than 5000
    variances = np.zeros(B, dtype=np.float64)

    # Use caller-supplied L^{-1} (precomputed at subdomain-fit time) when
    # available.  Fall back to computing it once here for single-domain mode.
    # Either way each variance batch uses dgemm instead of per-batch dtrsm.
    _L_inv = l_inv
    if is_cholesky and _L_inv is None:
        _L_inv = solve_triangular(
            factorisation, np.eye(factorisation.shape[0], dtype=np.float64), lower=True
        )

    for start in range(0, B, BATCH):
        end = min(start + BATCH, B)
        batch = query_points[start:end]

        k_batch = _compute_kernel_batch(
            batch, sample_coords, orientation_field,
            S, T, R, kernel_type, alpha, sill,
        )

        P_batch = build_polynomial_matrix(batch, drift_type)
        k_aug = np.hstack([k_batch, P_batch])

        if is_cholesky:
            v = _L_inv @ k_aug.T   # dgemm — fast (L_inv precomputed)
            dot_products = np.sum(v * v, axis=0)
        else:
            # LU: v = solve(K_aug, k_aug^T), s2 = phi_0 - k_aug^T v
            v = lu_solve(factorisation, k_aug.T)
            dot_products = np.sum(k_aug.T * v, axis=0)

        variances[start:end] = phi_0 - dot_products

    # B1 fix: clamp to zero (no artificial floor that hides real uncertainty)
    variances = np.maximum(variances, 0.0)
    return clamp_variance(variances)


def predict_mean_and_variance(
    query_points: np.ndarray,
    sample_coords: np.ndarray,
    weights: np.ndarray,
    poly_coeffs: np.ndarray,
    factorisation,
    kernel_type: str = "spheroidal",
    alpha: float = 1.0,
    sill: float = 1.0,
    range_: float = 100.0,
    nugget: float = 0.0,
    R: Optional[np.ndarray] = None,
    S: Optional[np.ndarray] = None,
    drift_type: str = "constant",
    orientation_field: Optional[object] = None,
    l_inv: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute both posterior mean and variance in a single pass.

    Variance uses Cholesky forward-substitution (M == 0) or batched
    LU solve (M > 0), avoiding the explicit matrix inverse.
    For spheroidal kernel, fuses distance+kernel via numba.
    """
    if R is None:
        R = np.eye(3, dtype=np.float64)
    if S is None:
        S = scale_matrix(range_, range_, range_)

    query_points = np.atleast_2d(query_points)
    B = query_points.shape[0]
    phi_0 = sill
    is_cholesky = not isinstance(factorisation, tuple)
    T = np.ascontiguousarray((S @ R), dtype=np.float64)

    BATCH = 50000
    means = np.zeros(B, dtype=np.float64)
    variances = np.zeros(B, dtype=np.float64)

    # Use caller-supplied L^{-1} (precomputed at subdomain-fit time) when
    # available.  Fall back to computing it once here for single-domain mode.
    _L_inv = l_inv
    if is_cholesky and _L_inv is None:
        _L_inv = solve_triangular(
            factorisation, np.eye(factorisation.shape[0], dtype=np.float64), lower=True
        )

    for start in range(0, B, BATCH):
        end = min(start + BATCH, B)
        batch = np.ascontiguousarray(query_points[start:end])

        k_batch = _compute_kernel_batch(
            batch, sample_coords, orientation_field,
            S, T, R, kernel_type, alpha, sill,
        )

        P_batch = build_polynomial_matrix(batch, drift_type)

        # Mean: f(x) = k(x)^T w + p(x)^T c
        est = k_batch @ weights
        if P_batch.shape[1] > 0 and len(poly_coeffs) > 0:
            est += P_batch @ poly_coeffs
        means[start:end] = est

        # Variance via dgemm (L_inv precomputed) or dtrsm (LU path)
        k_aug = np.hstack([k_batch, P_batch])
        if is_cholesky:
            v = _L_inv @ k_aug.T  # dgemm — fast (L_inv precomputed once)
            dot_products = np.sum(v * v, axis=0)
        else:
            v = lu_solve(factorisation, k_aug.T)
            dot_products = np.sum(k_aug.T * v, axis=0)
        variances[start:end] = phi_0 - dot_products

    # B1 fix: clamp to zero (no artificial floor that hides real uncertainty)
    variances = np.maximum(variances, 0.0)
    return means, clamp_variance(variances)
