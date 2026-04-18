"""
Scalar Field — Gradient-Augmented RBF Interpolation.
=====================================================

Assembles and solves the augmented kernel system (Eq. 3.2) that
combines value constraints (f(x_i) = v_i) and gradient constraints
(grad f(x_g) . n = g) in a single linear system.

For large datasets (N_v + N_g > PUM_THRESHOLD) the module automatically
switches to a Partition-of-Unity Method (PUM) that decomposes the domain
into overlapping sub-domains, solves each sub-domain independently, and
blends the results using Wendland C2 weights.  This reduces the worst-case
complexity from O(N³) to O(K × (N/K)³) ≈ O(N³/K²), allowing production
datasets with tens-of-thousands of constraints to be handled.

Reference: Wendland (2004), Scattered Data Approximation, §10.1.

Reuses ARBF kernel and utility infrastructure.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

# Reuse ARBF infrastructure
from geostats.arbf.kernels import evaluate_kernel
from geostats.arbf.utils import (
    rotation_matrix,
    scale_matrix,
    stable_cholesky,
)

from .gradient_kernels import (
    kernel_derivative_1,
    kernel_derivative_2,
    polynomial_drift,
    polynomial_drift_gradient,
)

logger = logging.getLogger(__name__)

# ── PUM configuration ──────────────────────────────────────────────────────
# Switch to PUM when total constraints exceed this threshold.
PUM_THRESHOLD: int = 1500
# Maximum number of constraints per sub-domain (controls memory + solve time).
_PUM_MAX_LOCAL: int = 350
# Minimum constraints per sub-domain (expand radius if below).
_PUM_MIN_LOCAL: int = 8
# Overlap factor: sub-domain radius = max_assigned_distance × this.
_PUM_OVERLAP: float = 1.5


# ═══════════════════════════════════════════════════════════════════
# Distance helpers
# ═══════════════════════════════════════════════════════════════════

def _compute_distances(
    coords_i: np.ndarray,
    coords_j: np.ndarray,
    range_: float,
    R: Optional[np.ndarray] = None,
    S: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Compute displacement vectors and normalised distances.

    Parameters
    ----------
    coords_i : (N, 3)
    coords_j : (M, 3)
    range_ : float
    R, S : optional rotation and scaling matrices

    Returns
    -------
    h : (N, M, 3)   raw displacement vectors (geographic space)
    r : (N, M)       normalised distances (anisotropic space)
    h_t : (N, M, 3)  displacement vectors in anisotropic space
    T : (3, 3) or None  anisotropy transform matrix (for normal transform)
    """
    # h[i, j] = coords_i[i] - coords_j[j]
    h = coords_i[:, np.newaxis, :] - coords_j[np.newaxis, :, :]  # (N, M, 3)

    T = None
    if R is not None and S is not None:
        T = S @ R  # (3, 3)
        h_t = np.einsum("ijk,lk->ijl", h, T)
        dist = np.linalg.norm(h_t, axis=-1)
    elif S is not None:
        T = S.copy()
        h_t = np.einsum("ijk,lk->ijl", h, S)
        dist = np.linalg.norm(h_t, axis=-1)
    else:
        h_t = h
        dist = np.linalg.norm(h, axis=-1)

    # Normalise by range
    r = dist / max(range_, 1e-12)

    return h, r, h_t, T


def _transform_normals(
    normals: np.ndarray,
    T: Optional[np.ndarray],
) -> np.ndarray:
    """Transform normals to anisotropic space.

    Normals transform with T (not T^{-T} as for surface normals)
    because the derivative formula d/dn phi(r) uses the chain rule:
    d/dn phi(||T h||) = phi'(r) * (T h . T n) / (r * range^2)

    Parameters
    ----------
    normals : (N, 3) or (1, N, 3)
    T : (3, 3) or None

    Returns
    -------
    transformed normals, same shape as input
    """
    if T is None:
        return normals
    if normals.ndim == 2:
        return normals @ T.T
    elif normals.ndim == 3:
        return np.einsum("ijk,lk->ijl", normals, T)
    return normals


# ═══════════════════════════════════════════════════════════════════
# Augmented Matrix Assembly
# ═══════════════════════════════════════════════════════════════════

def assemble_augmented_matrix(
    value_coords: np.ndarray,
    gradient_coords: np.ndarray,
    gradient_normals: np.ndarray,
    kernel_type: str = "spheroidal",
    alpha: float = 1.0,
    range_: float = 100.0,
    nugget: float = 0.0,
    accuracy: float = 1e-6,
    drift_type: str = "constant",
    R: Optional[np.ndarray] = None,
    S: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, int, int]:
    """Build the gradient-augmented kernel matrix (Eq. 3.2).

    Returns the full augmented system::

        [ K_vv    K_vg    P_v  ]
        [ K_gv    K_gg    P_g  ]
        [ P_v^T   P_g^T   0   ]

    Parameters
    ----------
    value_coords : (N_v, 3)
    gradient_coords : (N_g, 3)
    gradient_normals : (N_g, 3)
    kernel_type, alpha, range_, nugget, accuracy : kernel params
    drift_type : "constant" or "linear"
    R, S : rotation and scaling matrices for anisotropy

    Returns
    -------
    K_aug : (N_v + N_g + M, N_v + N_g + M)
    N_v : int
    N_g : int
    """
    N_v = value_coords.shape[0]
    N_g = gradient_coords.shape[0] if gradient_coords.size > 0 else 0

    # Drift polynomial dimensions
    M = 1 if drift_type == "constant" else 4
    N_total = N_v + N_g + M

    K_aug = np.zeros((N_total, N_total), dtype=np.float64)

    # ── K_vv block: (N_v, N_v) standard kernel ──
    if N_v > 0:
        _h_vv, r_vv, _ht_vv, _T_vv = _compute_distances(value_coords, value_coords, range_, R, S)
        K_vv = evaluate_kernel(r_vv, kernel_type, alpha)
        # Add nugget + accuracy to diagonal
        K_vv += (nugget + accuracy) * np.eye(N_v, dtype=np.float64)
        K_aug[:N_v, :N_v] = K_vv

    # ── K_vg block: (N_v, N_g) first derivative ──
    # BUG 1 FIX: Use h_t (anisotropic space) and transformed normals
    # so that h.n and r are in the same coordinate system.
    if N_v > 0 and N_g > 0:
        _h_vg, r_vg, h_t_vg, T_vg = _compute_distances(value_coords, gradient_coords, range_, R, S)
        n_t = _transform_normals(gradient_normals, T_vg)
        K_vg = kernel_derivative_1(h_t_vg, r_vg, n_t, kernel_type, alpha, range_)
        K_aug[:N_v, N_v:N_v + N_g] = K_vg

    # ── K_gv block: (N_g, N_v) = K_vg^T (by symmetry, Sec 3.3) ──
    if N_v > 0 and N_g > 0:
        K_aug[N_v:N_v + N_g, :N_v] = K_vg.T

    # ── K_gg block: (N_g, N_g) second derivative ──
    # BUG 1 FIX: Same anisotropy-consistent transform for K_gg
    if N_g > 0:
        _h_gg, r_gg, h_t_gg, T_gg = _compute_distances(gradient_coords, gradient_coords, range_, R, S)
        n_t_i = _transform_normals(gradient_normals, T_gg)
        n_t_j = n_t_i  # same set for self-interaction
        K_gg = kernel_derivative_2(
            h_t_gg, r_gg, n_t_i, n_t_j,
            kernel_type, alpha, range_,
        )
        # Add accuracy to K_gg diagonal for regularisation
        K_gg += accuracy * np.eye(N_g, dtype=np.float64)
        K_aug[N_v:N_v + N_g, N_v:N_v + N_g] = K_gg

    # ── P_v block: polynomial at value points ──
    if N_v > 0:
        P_v = polynomial_drift(value_coords, drift_type)
        K_aug[:N_v, N_v + N_g:] = P_v
        K_aug[N_v + N_g:, :N_v] = P_v.T

    # ── P_g block: polynomial gradient at gradient points ──
    if N_g > 0:
        P_g = polynomial_drift_gradient(gradient_coords, gradient_normals, drift_type)
        K_aug[N_v:N_v + N_g, N_v + N_g:] = P_g
        K_aug[N_v + N_g:, N_v:N_v + N_g] = P_g.T

    return K_aug, N_v, N_g


# ═══════════════════════════════════════════════════════════════════
# Solve
# ═══════════════════════════════════════════════════════════════════

def solve_augmented_system(
    K_aug: np.ndarray,
    values: np.ndarray,
    gradient_values: np.ndarray,
    N_v: int,
    N_g: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve the augmented system for weights.

    Parameters
    ----------
    K_aug : (N_total, N_total)
    values : (N_v,) value constraint values
    gradient_values : (N_g,) gradient constraint values (typically 1.0)
    N_v, N_g : dimensions

    Returns
    -------
    value_weights : (N_v,)
    gradient_weights : (N_g,)
    poly_coeffs : (M,)
    """
    N_total = K_aug.shape[0]
    M = N_total - N_v - N_g

    # Build RHS
    rhs = np.zeros(N_total, dtype=np.float64)
    rhs[:N_v] = values
    if N_g > 0:
        rhs[N_v:N_v + N_g] = gradient_values
    # Polynomial unbiasedness: rhs[N_v + N_g:] = 0 (already zero)

    # Solve: try Cholesky first (SPD expected), fall back to LU
    try:
        L = stable_cholesky(K_aug)
        # Forward-backward solve
        y = np.linalg.solve(L, rhs)
        x = np.linalg.solve(L.T, y)
    except (np.linalg.LinAlgError, ValueError):
        logger.warning("Cholesky failed, falling back to LU factorisation")
        x = np.linalg.solve(K_aug, rhs)

    value_weights = x[:N_v]
    gradient_weights = x[N_v:N_v + N_g] if N_g > 0 else np.array([], dtype=np.float64)
    poly_coeffs = x[N_v + N_g:]

    logger.debug(
        "Solved augmented system: N_v=%d, N_g=%d, M=%d, "
        "max|w|=%.4g, max|u|=%.4g",
        N_v, N_g, M,
        np.max(np.abs(value_weights)) if N_v > 0 else 0.0,
        np.max(np.abs(gradient_weights)) if N_g > 0 else 0.0,
    )

    return value_weights, gradient_weights, poly_coeffs


# ═══════════════════════════════════════════════════════════════════
# Evaluate
# ═══════════════════════════════════════════════════════════════════

def evaluate_scalar_field(
    query_points: np.ndarray,
    value_coords: np.ndarray,
    gradient_coords: np.ndarray,
    gradient_normals: np.ndarray,
    value_weights: np.ndarray,
    gradient_weights: np.ndarray,
    poly_coeffs: np.ndarray,
    kernel_type: str = "spheroidal",
    alpha: float = 1.0,
    range_: float = 100.0,
    R: Optional[np.ndarray] = None,
    S: Optional[np.ndarray] = None,
    drift_type: str = "constant",
) -> np.ndarray:
    """Evaluate the interpolated scalar field at query points.

    f(x) = SUM_i w_i * phi(||x - x_i||)
         + SUM_j u_j * [d/dn_j phi(||x - y_j||)]
         + p(x)

    Parameters
    ----------
    query_points : (B, 3)
    value_coords : (N_v, 3)
    gradient_coords : (N_g, 3)
    gradient_normals : (N_g, 3)
    value_weights : (N_v,)
    gradient_weights : (N_g,)
    poly_coeffs : (M,)
    kernel_type, alpha, range_ : kernel params
    R, S : anisotropy matrices
    drift_type : polynomial drift type

    Returns
    -------
    np.ndarray, shape (B,)
    """
    B = query_points.shape[0]
    N_v = value_coords.shape[0]
    N_g = gradient_coords.shape[0] if gradient_coords.size > 0 else 0

    result = np.zeros(B, dtype=np.float64)

    # ── Value contribution: SUM_i w_i * phi(||x - x_i||) ──
    if N_v > 0:
        _h_qv, r_qv, _ht_qv, _T_qv = _compute_distances(query_points, value_coords, range_, R, S)
        K_qv = evaluate_kernel(r_qv, kernel_type, alpha)  # (B, N_v)
        result += K_qv @ value_weights

    # ── Gradient contribution: SUM_j u_j * d/dn_j phi(||x - y_j||) ──
    # BUG 1 FIX: Use transformed h and normals for consistency
    if N_g > 0:
        _h_qg, r_qg, h_t_qg, T_qg = _compute_distances(query_points, gradient_coords, range_, R, S)
        n_t = _transform_normals(gradient_normals, T_qg)
        K_qg = kernel_derivative_1(h_t_qg, r_qg, n_t, kernel_type, alpha, range_)
        result += K_qg @ gradient_weights

    # ── Polynomial drift: p(x) ──
    P_q = polynomial_drift(query_points, drift_type)  # (B, M)
    result += P_q @ poly_coeffs

    return result


def make_evaluate_fn(
    value_coords: np.ndarray,
    gradient_coords: np.ndarray,
    gradient_normals: np.ndarray,
    value_weights: np.ndarray,
    gradient_weights: np.ndarray,
    poly_coeffs: np.ndarray,
    kernel_type: str = "spheroidal",
    alpha: float = 1.0,
    range_: float = 100.0,
    R: Optional[np.ndarray] = None,
    S: Optional[np.ndarray] = None,
    drift_type: str = "constant",
) -> Callable[[np.ndarray], np.ndarray]:
    """Create a closure that evaluates the scalar field.

    Returns a callable  f(query_points) -> scalar_values
    where query_points is (B, 3) and result is (B,).
    """
    def _evaluate(query_points: np.ndarray) -> np.ndarray:
        return evaluate_scalar_field(
            query_points,
            value_coords, gradient_coords, gradient_normals,
            value_weights, gradient_weights, poly_coeffs,
            kernel_type, alpha, range_, R, S, drift_type,
        )
    return _evaluate


# ═══════════════════════════════════════════════════════════════════
# Partition-of-Unity Method (PUM)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class _ImplicitSubDomain:
    """One sub-domain of a PUM scalar field decomposition."""
    index: int
    centre: np.ndarray         # (3,) geometric centre
    radius: float              # blending radius

    # Local constraint copies
    local_value_coords: np.ndarray       # (n_v, 3)
    local_gradient_coords: np.ndarray    # (n_g, 3)
    local_gradient_normals: np.ndarray   # (n_g, 3)

    # Solved local weights
    value_weights: np.ndarray            # (n_v,)
    gradient_weights: np.ndarray         # (n_g,)
    poly_coeffs: np.ndarray              # (M,)


@dataclass
class PUMScalarField:
    """Container for a fitted PUM scalar field.

    Pass to ``evaluate_scalar_field_pum`` for prediction.
    """
    subdomains: List[_ImplicitSubDomain]
    kernel_type: str
    alpha: float
    range_: float
    R: Optional[np.ndarray]
    S: Optional[np.ndarray]
    drift_type: str


def _wendland_c2_batch(distances: np.ndarray, radius: float) -> np.ndarray:
    """Vectorised Wendland C2 weight.

    w(d, R) = (1 - d/R)^4 * (1 + 4*d/R)   for d < R
    w(d, R) = 0                              for d >= R
    """
    r = distances / max(radius, 1e-12)
    w = np.where(r < 1.0, (1.0 - r) ** 4 * (1.0 + 4.0 * r), 0.0)
    return w


def _kmeans_centres(coords: np.ndarray, K: int) -> Tuple[np.ndarray, np.ndarray]:
    """K-means++ centres and per-cluster radii scaled by _PUM_OVERLAP.

    Falls back to random centres when scipy is unavailable.
    """
    try:
        from scipy.cluster.vq import kmeans2
        centres, labels = kmeans2(coords, K, minit="++", iter=50)
    except ImportError:
        # Simple random seeding fallback
        idx = np.random.choice(len(coords), K, replace=False)
        centres = coords[idx].copy()
        dists = np.linalg.norm(coords[:, np.newaxis, :] - centres[np.newaxis, :, :], axis=-1)
        labels = np.argmin(dists, axis=1)

    radii = np.zeros(K, dtype=np.float64)
    for i in range(K):
        mask = labels == i
        if np.any(mask):
            dists = np.linalg.norm(coords[mask] - centres[i], axis=1)
            radii[i] = np.max(dists) * _PUM_OVERLAP
        else:
            radii[i] = np.max(np.linalg.norm(coords - coords.mean(axis=0), axis=1)) - np.min(np.linalg.norm(coords - coords.mean(axis=0), axis=1))
    return centres, radii


def solve_augmented_system_pum(
    value_coords: np.ndarray,
    values: np.ndarray,
    gradient_coords: np.ndarray,
    gradient_normals: np.ndarray,
    gradient_values: np.ndarray,
    kernel_type: str = "spheroidal",
    alpha: float = 1.0,
    range_: float = 100.0,
    nugget: float = 0.0,
    accuracy: float = 1e-6,
    drift_type: str = "constant",
    R: Optional[np.ndarray] = None,
    S: Optional[np.ndarray] = None,
    k_subdomains: Optional[int] = None,
) -> PUMScalarField:
    """Solve the gradient-augmented system using Partition-of-Unity.

    The domain is decomposed into K overlapping sub-domains via k-means.
    Each sub-domain gets its own local augmented matrix (assembled and solved
    independently).  At query time the results are blended with Wendland C2
    weights — see ``evaluate_scalar_field_pum``.

    This reduces the O(N³) global solve to O(K × (N/K)³) ≈ O(N / K²),
    making datasets of tens-of-thousands of constraints tractable.

    Parameters
    ----------
    value_coords : (N_v, 3)
    values : (N_v,)
    gradient_coords : (N_g, 3)
    gradient_normals : (N_g, 3)
    gradient_values : (N_g,)  -- typically zeros for tangent constraints
    kernel_type, alpha, range_, nugget, accuracy, drift_type, R, S :
        Standard kernel parameters (same as ``assemble_augmented_matrix``).
    k_subdomains : int, optional
        Number of sub-domains.  Auto-computed when None:
        max(4, N_total // 150).

    Returns
    -------
    PUMScalarField
        Fitted model; pass to ``evaluate_scalar_field_pum``.
    """
    N_v = value_coords.shape[0] if value_coords.size > 0 else 0
    N_g = gradient_coords.shape[0] if gradient_coords.size > 0 else 0
    N_total = N_v + N_g

    # Build a combined coordinate array for clustering
    all_coords = np.vstack([
        value_coords if N_v > 0 else np.empty((0, 3)),
        gradient_coords if N_g > 0 else np.empty((0, 3)),
    ])  # (N_total, 3)
    # Label: first N_v rows are value constraints, rest are gradient constraints
    is_gradient = np.zeros(N_total, dtype=bool)
    if N_g > 0:
        is_gradient[N_v:] = True

    # Auto-determine K
    if k_subdomains is None:
        k_subdomains = max(4, N_total // 150)
    K = min(k_subdomains, N_total // max(_PUM_MIN_LOCAL, 1))
    K = max(K, 1)

    logger.info(
        "PUM solve: N_v=%d, N_g=%d, K=%d sub-domains", N_v, N_g, K,
    )

    centres, radii = _kmeans_centres(all_coords, K)

    # Build KD-tree for fast radius queries
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(all_coords)
        use_scipy = True
    except ImportError:
        use_scipy = False

    subdomains: List[_ImplicitSubDomain] = []

    for i in range(K):
        c = centres[i]
        r = float(radii[i])

        # Gather indices of all constraints within radius
        if use_scipy:
            local_idx = np.asarray(tree.query_ball_point(c, r=r), dtype=np.intp)
        else:
            dists = np.linalg.norm(all_coords - c, axis=1)
            local_idx = np.where(dists <= r)[0].astype(np.intp)

        # Ensure minimum sample count
        if len(local_idx) < _PUM_MIN_LOCAL:
            if use_scipy:
                k_nn = min(_PUM_MIN_LOCAL, N_total)
                _, expanded = tree.query(c.reshape(1, -1), k=k_nn)
                local_idx = expanded.ravel().astype(np.intp)
            else:
                dists = np.linalg.norm(all_coords - c, axis=1)
                local_idx = np.argsort(dists)[:_PUM_MIN_LOCAL]
            if len(local_idx) > 0:
                max_d = float(np.max(np.linalg.norm(all_coords[local_idx] - c, axis=1)))
                r = max_d * 1.1
                radii[i] = r

        # Cap maximum to keep local matrices tractable
        if len(local_idx) > _PUM_MAX_LOCAL:
            dists_local = np.linalg.norm(all_coords[local_idx] - c, axis=1)
            keep = np.argsort(dists_local)[:_PUM_MAX_LOCAL]
            local_idx = local_idx[keep]

        # Split into value vs gradient
        v_mask = local_idx[~is_gradient[local_idx]]
        g_mask = local_idx[is_gradient[local_idx]]

        local_val_coords = value_coords[v_mask] if len(v_mask) > 0 else np.empty((0, 3))
        local_vals = values[v_mask] if len(v_mask) > 0 else np.empty(0)
        local_grad_coords = gradient_coords[g_mask - N_v] if len(g_mask) > 0 else np.empty((0, 3))
        local_grad_normals = gradient_normals[g_mask - N_v] if len(g_mask) > 0 else np.empty((0, 3))
        local_grad_vals = gradient_values[g_mask - N_v] if len(g_mask) > 0 else np.empty(0)

        n_lv = len(local_val_coords)
        n_lg = len(local_grad_coords)

        if n_lv == 0 and n_lg == 0:
            continue

        try:
            K_aug, sub_N_v, sub_N_g = assemble_augmented_matrix(
                local_val_coords if n_lv > 0 else np.empty((0, 3)),
                local_grad_coords if n_lg > 0 else np.empty((0, 3)),
                local_grad_normals if n_lg > 0 else np.empty((0, 3)),
                kernel_type=kernel_type,
                alpha=alpha,
                range_=range_,
                nugget=nugget,
                accuracy=accuracy,
                drift_type=drift_type,
                R=R,
                S=S,
            )

            vw, gw, pc = solve_augmented_system(
                K_aug, local_vals, local_grad_vals, sub_N_v, sub_N_g,
            )

            subdomains.append(_ImplicitSubDomain(
                index=i,
                centre=c.copy(),
                radius=r,
                local_value_coords=local_val_coords,
                local_gradient_coords=local_grad_coords,
                local_gradient_normals=local_grad_normals,
                value_weights=vw,
                gradient_weights=gw,
                poly_coeffs=pc,
            ))

        except Exception as exc:
            logger.warning("PUM sub-domain %d solve failed: %s", i, exc)

    logger.info(
        "PUM fitted %d / %d sub-domains successfully", len(subdomains), K,
    )

    return PUMScalarField(
        subdomains=subdomains,
        kernel_type=kernel_type,
        alpha=alpha,
        range_=range_,
        R=R,
        S=S,
        drift_type=drift_type,
    )


def evaluate_scalar_field_pum(
    query_points: np.ndarray,
    pum: PUMScalarField,
    batch_size: int = 5000,
) -> np.ndarray:
    """Evaluate the PUM scalar field at query points.

    For each query point q:
      1. Find which sub-domains cover q (centre distance < radius).
      2. Evaluate each covering sub-domain's local scalar field.
      3. Compute Wendland C2 weights from distance to each centre.
      4. Return normalised weighted sum.

    Falls back to the nearest sub-domain for uncovered points.

    Parameters
    ----------
    query_points : (B, 3)
    pum : PUMScalarField
        Fitted model from ``solve_augmented_system_pum``.
    batch_size : int
        Process query points in chunks to bound peak memory.

    Returns
    -------
    np.ndarray, shape (B,)
    """
    B = query_points.shape[0]
    result = np.zeros(B, dtype=np.float64)

    if not pum.subdomains:
        logger.warning("PUM model has no sub-domains; returning zeros")
        return result

    # Precompute sub-domain centres array for batch distance computation
    sd_centres = np.array([sd.centre for sd in pum.subdomains])  # (K, 3)
    sd_radii = np.array([sd.radius for sd in pum.subdomains])    # (K,)

    kernel_type = pum.kernel_type
    alpha = pum.alpha
    range_ = pum.range_
    R = pum.R
    S = pum.S
    drift_type = pum.drift_type

    for start in range(0, B, batch_size):
        end = min(start + batch_size, B)
        qp = query_points[start:end]   # (b, 3)
        b = qp.shape[0]

        # Distances from each query point to each sub-domain centre: (b, K)
        d_to_centres = np.linalg.norm(
            qp[:, np.newaxis, :] - sd_centres[np.newaxis, :, :], axis=-1,
        )

        # Wendland C2 weights: (b, K)
        W = _wendland_c2_batch(d_to_centres, 1.0)  # placeholder — per sub-domain
        # Per sub-domain radius
        for k, sd in enumerate(pum.subdomains):
            r_k = sd.radius
            d_k = d_to_centres[:, k]
            rn = d_k / max(r_k, 1e-12)
            W[:, k] = np.where(rn < 1.0, (1.0 - rn) ** 4 * (1.0 + 4.0 * rn), 0.0)

        W_sum = W.sum(axis=1)  # (b,)
        uncovered = W_sum < 1e-14  # points outside all sub-domain radii

        # Handle uncovered points: assign weight=1 to nearest sub-domain
        if np.any(uncovered):
            nearest = np.argmin(d_to_centres[uncovered], axis=1)
            for local_i, global_i in enumerate(np.where(uncovered)[0]):
                k = int(nearest[local_i])
                W[global_i, k] = 1.0
            W_sum = W.sum(axis=1)

        W_norm = W / np.maximum(W_sum[:, np.newaxis], 1e-14)   # (b, K)

        batch_result = np.zeros(b, dtype=np.float64)

        for k, sd in enumerate(pum.subdomains):
            # Points where this sub-domain has non-zero weight
            active = W_norm[:, k] > 1e-14
            if not np.any(active):
                continue

            qp_active = qp[active]  # (b_k, 3)

            local_val = evaluate_scalar_field(
                qp_active,
                sd.local_value_coords,
                sd.local_gradient_coords,
                sd.local_gradient_normals,
                sd.value_weights,
                sd.gradient_weights,
                sd.poly_coeffs,
                kernel_type=kernel_type,
                alpha=alpha,
                range_=range_,
                R=R,
                S=S,
                drift_type=drift_type,
            )

            batch_result[active] += W_norm[active, k] * local_val

        result[start:end] = batch_result

    return result


def make_evaluate_fn_pum(pum: PUMScalarField) -> Callable[[np.ndarray], np.ndarray]:
    """Create a closure that evaluates the PUM scalar field.

    Returns a callable f(query_points) -> scalar_values
    identical in signature to ``make_evaluate_fn``.
    """
    def _evaluate(query_points: np.ndarray) -> np.ndarray:
        return evaluate_scalar_field_pum(query_points, pum)
    return _evaluate
