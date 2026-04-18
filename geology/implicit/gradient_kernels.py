"""
Gradient Kernels — Derivative Kernel Entries for the Augmented Matrix.
======================================================================

Implements Eq. 4.1 from the mathematical specification:
first and second directional derivatives of all supported radial
basis functions.  These are used to build K_vg, K_gv and K_gg blocks
of the gradient-augmented kernel matrix (Eq. 3.2).

The normalised distance is  r = ||h|| / range  where h = x_i - x_j.

All functions operate on NumPy arrays and use float64 throughout.

References
----------
- Hillier et al. (2014), Mathematical Geosciences 46(8), 931-953.
- Wendland (2004), Scattered Data Approximation, Ch. 16.
- GeoX Math Spec, Sections 3-4 and Appendix A.
"""

from __future__ import annotations

import numpy as np

# ═══════════════════════════════════════════════════════════════════
# 1.  phi'(r) and phi''(r) for each kernel
# ═══════════════════════════════════════════════════════════════════

_EPS = 1e-10  # threshold for r ≈ 0 limits


def phi_prime(r: np.ndarray, kernel_type: str, alpha: float = 1.0) -> np.ndarray:
    r"""First derivative of kernel w.r.t. normalised distance r.

    Parameters
    ----------
    r : array_like
        Normalised distances (>= 0).
    kernel_type : str
        One of ``spheroidal``, ``gaussian``, ``matern_32``, ``matern_52``.
    alpha : float
        Smoothness parameter for spheroidal kernel.

    Returns
    -------
    np.ndarray
        phi'(r) values.  phi'(0) = 0 for all kernels.
    """
    r = np.asarray(r, dtype=np.float64)
    out = np.zeros_like(r)

    if kernel_type == "spheroidal":
        # phi'(r) = -2*alpha * r * (1 + r^2)^{-(alpha+1)}   [Eq 4.2]
        mask = r > _EPS
        rm = r[mask]
        out[mask] = -2.0 * alpha * rm * (1.0 + rm * rm) ** (-(alpha + 1.0))

    elif kernel_type == "gaussian":
        # phi'(r) = -2 * r * exp(-r^2)   [Eq 4.5]
        mask = r > _EPS
        rm = r[mask]
        out[mask] = -2.0 * rm * np.exp(-(rm * rm))

    elif kernel_type == "matern_32":
        # phi'(r) = -3 * r * exp(-sqrt(3)*r)   [Eq 4.3]
        s3 = np.sqrt(3.0)
        mask = r > _EPS
        rm = r[mask]
        out[mask] = -3.0 * rm * np.exp(-s3 * rm)

    elif kernel_type == "matern_52":
        # phi'(r) = -(5/3) * r * (1 + sqrt(5)*r) * exp(-sqrt(5)*r)   [Eq 4.4]
        s5 = np.sqrt(5.0)
        mask = r > _EPS
        rm = r[mask]
        out[mask] = -(5.0 / 3.0) * rm * (1.0 + s5 * rm) * np.exp(-s5 * rm)

    else:
        raise ValueError(f"Unsupported kernel type for gradient: '{kernel_type}'")

    return out


def phi_double_prime(r: np.ndarray, kernel_type: str, alpha: float = 1.0) -> np.ndarray:
    r"""Second derivative of kernel w.r.t. normalised distance r.

    Parameters
    ----------
    r : array_like
        Normalised distances (>= 0).
    kernel_type : str
        Kernel function name.
    alpha : float
        Smoothness for spheroidal.

    Returns
    -------
    np.ndarray
        phi''(r) values.
    """
    r = np.asarray(r, dtype=np.float64)
    out = np.empty_like(r)

    if kernel_type == "spheroidal":
        # phi''(r) = -2*alpha * (1+r^2)^{-(alpha+2)} * [1 - (2*alpha+1)*r^2]  [Eq 4.2]
        # At r=0: phi''(0) = -2*alpha
        near = r < _EPS
        far = ~near
        out[near] = -2.0 * alpha
        if np.any(far):
            rm = r[far]
            r2 = rm * rm
            out[far] = -2.0 * alpha * (1.0 + r2) ** (-(alpha + 2.0)) * (1.0 - (2.0 * alpha + 1.0) * r2)

    elif kernel_type == "gaussian":
        # phi''(r) = (-2 + 4*r^2) * exp(-r^2)   [Eq 4.5]
        # At r=0: phi''(0) = -2
        near = r < _EPS
        far = ~near
        out[near] = -2.0
        if np.any(far):
            rm = r[far]
            r2 = rm * rm
            out[far] = (-2.0 + 4.0 * r2) * np.exp(-r2)

    elif kernel_type == "matern_32":
        # phi''(r) = -3 * exp(-sqrt(3)*r) * (1 - sqrt(3)*r)   [Eq 4.3]
        # At r=0: phi''(0) = -3
        s3 = np.sqrt(3.0)
        near = r < _EPS
        far = ~near
        out[near] = -3.0
        if np.any(far):
            rm = r[far]
            out[far] = -3.0 * np.exp(-s3 * rm) * (1.0 - s3 * rm)

    elif kernel_type == "matern_52":
        # phi''(r) = -(5/3) * exp(-sqrt(5)*r) * [1 + sqrt(5)*r - 5*r^2]  [Eq 4.4]
        # At r=0: phi''(0) = -5/3
        s5 = np.sqrt(5.0)
        near = r < _EPS
        far = ~near
        out[near] = -5.0 / 3.0
        if np.any(far):
            rm = r[far]
            r2 = rm * rm
            out[far] = -(5.0 / 3.0) * np.exp(-s5 * rm) * (1.0 + s5 * rm - 5.0 * r2)

    else:
        raise ValueError(f"Unsupported kernel type for gradient: '{kernel_type}'")

    return out


def phi_prime_over_r_limit(kernel_type: str, alpha: float = 1.0) -> float:
    """Return lim_{r->0} phi'(r)/r = phi''(0).

    This limit is needed for the K_gg diagonal when two gradient
    constraint points coincide (r = 0).  By L'Hopital:
    lim phi'(r)/r = phi''(0).
    """
    if kernel_type == "spheroidal":
        return -2.0 * alpha
    elif kernel_type == "gaussian":
        return -2.0
    elif kernel_type == "matern_32":
        return -3.0
    elif kernel_type == "matern_52":
        return -5.0 / 3.0
    else:
        raise ValueError(f"Unsupported kernel type: '{kernel_type}'")


# ═══════════════════════════════════════════════════════════════════
# 2.  Directional derivative kernel entries (K_vg, K_gv, K_gg)
# ═══════════════════════════════════════════════════════════════════

def kernel_derivative_1(
    h: np.ndarray,
    r: np.ndarray,
    n: np.ndarray,
    kernel_type: str,
    alpha: float,
    range_: float,
) -> np.ndarray:
    """First directional derivative kernel: d/dn phi(r).

    Implements Eq. 4.1 from the math spec::

        d/dn phi(r) = phi'(r) * (h . n) / (r * range^2)

    Used for K_vg and K_gv blocks in the augmented matrix.

    Parameters
    ----------
    h : np.ndarray, shape (N, M, 3)
        Displacement vectors x_i - x_j (value pts minus gradient pts).
    r : np.ndarray, shape (N, M)
        Normalised distances ||h|| / range.
    n : np.ndarray, shape (M, 3) or (1, M, 3)
        Unit normal directions at gradient points.
    kernel_type : str
        Kernel function name.
    alpha : float
        Smoothness parameter.
    range_ : float
        Range parameter (metres).

    Returns
    -------
    np.ndarray, shape (N, M)
        First derivative kernel values.
    """
    # Ensure n is broadcast-compatible: (1, M, 3)
    if n.ndim == 2:
        n = n[np.newaxis, :, :]  # (1, M, 3)

    # h . n  →  (N, M)
    h_dot_n = np.sum(h * n, axis=-1)

    pp = phi_prime(r, kernel_type, alpha)  # (N, M)

    # For r > eps:  result = phi'(r) * (h.n) / (r * range^2)
    # For r ≈ 0:   h → 0, so h.n → 0 and result → 0  (by symmetry, Eq 4.1)
    result = np.zeros_like(r)
    safe = r > _EPS
    # range^2 factor comes from: r = ||h|| / range, and the chain rule
    # d/dn phi(||h||/range) = phi'(r) * (1/range) * (h.n) / ||h||
    #                       = phi'(r) * (h.n) / (r * range^2)
    result[safe] = pp[safe] * h_dot_n[safe] / (r[safe] * range_ * range_)

    return result


def kernel_derivative_2(
    h: np.ndarray,
    r: np.ndarray,
    n_i: np.ndarray,
    n_j: np.ndarray,
    kernel_type: str,
    alpha: float,
    range_: float,
) -> np.ndarray:
    """Second directional derivative kernel: d²/(dn_i dn_j) phi(r).

    Implements Eq. 4.1 (second derivative) from the math spec::

        d²/(dn_i dn_j) phi(r) =
            phi''(r) * (h.n_i)(h.n_j) / (r² * range⁴)
          + phi'(r)  * [(n_i.n_j)/r - (h.n_i)(h.n_j)/r³] / range²

    Used for K_gg block in the augmented matrix.

    At r=0 (coincident points): uses L'Hopital limit
        phi'(r)/r → phi''(0),  and h → 0 so (h.n) terms vanish.
        Result = phi''(0) * (n_i . n_j) / range²

    Parameters
    ----------
    h : np.ndarray, shape (N_g1, N_g2, 3)
        Displacement vectors between gradient point sets.
    r : np.ndarray, shape (N_g1, N_g2)
        Normalised distances.
    n_i : np.ndarray, shape (N_g1, 3) or (N_g1, 1, 3)
        Normal directions at first set.
    n_j : np.ndarray, shape (N_g2, 3) or (1, N_g2, 3)
        Normal directions at second set.
    kernel_type : str
    alpha : float
    range_ : float

    Returns
    -------
    np.ndarray, shape (N_g1, N_g2)
    """
    # Broadcast normals
    if n_i.ndim == 2:
        n_i = n_i[:, np.newaxis, :]  # (N_g1, 1, 3)
    if n_j.ndim == 2:
        n_j = n_j[np.newaxis, :, :]  # (1, N_g2, 3)

    h_dot_ni = np.sum(h * n_i, axis=-1)   # (N_g1, N_g2)
    h_dot_nj = np.sum(h * n_j, axis=-1)   # (N_g1, N_g2)
    ni_dot_nj = np.sum(n_i * n_j, axis=-1)  # (N_g1, N_g2)

    pp = phi_prime(r, kernel_type, alpha)
    pp2 = phi_double_prime(r, kernel_type, alpha)

    result = np.zeros_like(r)
    range2 = range_ * range_
    range4 = range2 * range2

    # Near r=0: use L'Hopital limit
    near = r < _EPS
    far = ~near

    # Limit: phi''(0) * (n_i . n_j) / range^2
    limit_val = phi_prime_over_r_limit(kernel_type, alpha)
    result[near] = limit_val * ni_dot_nj[near] / range2

    if np.any(far):
        rm = r[far]
        r2 = rm * rm
        r3 = r2 * rm

        term1 = pp2[far] * h_dot_ni[far] * h_dot_nj[far] / (r2 * range4)
        term2 = pp[far] * (ni_dot_nj[far] / rm - h_dot_ni[far] * h_dot_nj[far] / r3) / range2

        result[far] = term1 + term2

    return result


# ═══════════════════════════════════════════════════════════════════
# 3.  Polynomial drift terms for gradient constraints
# ═══════════════════════════════════════════════════════════════════

def polynomial_drift(coords: np.ndarray, drift_type: str = "constant") -> np.ndarray:
    """Build polynomial drift matrix P for value constraints.

    Parameters
    ----------
    coords : np.ndarray, shape (N, 3)
    drift_type : str
        ``"constant"`` → P = [1, 1, ..., 1]^T  (M=1)
        ``"linear"``   → P = [1, x, y, z]       (M=4)

    Returns
    -------
    np.ndarray, shape (N, M)
    """
    N = coords.shape[0]
    if drift_type == "constant":
        return np.ones((N, 1), dtype=np.float64)
    elif drift_type == "linear":
        P = np.ones((N, 4), dtype=np.float64)
        P[:, 1:] = coords
        return P
    else:
        raise ValueError(f"Unknown drift type: '{drift_type}'")


def polynomial_drift_gradient(
    coords: np.ndarray,
    normals: np.ndarray,
    drift_type: str = "constant",
) -> np.ndarray:
    """Build polynomial gradient matrix P_g for gradient constraints.

    P_g[j, k] = grad p_k(x_j) . n_j

    For constant drift:  grad(1) = 0 → P_g = 0
    For linear drift:    grad(x) = [1,0,0], grad(y) = [0,1,0], etc.
        P_g[j, 0] = 0 (gradient of constant is zero)
        P_g[j, 1] = n_j[0]  (gradient of x term dotted with normal)
        P_g[j, 2] = n_j[1]
        P_g[j, 3] = n_j[2]

    Parameters
    ----------
    coords : np.ndarray, shape (N_g, 3)
    normals : np.ndarray, shape (N_g, 3)
    drift_type : str

    Returns
    -------
    np.ndarray, shape (N_g, M)
    """
    N_g = coords.shape[0]
    if drift_type == "constant":
        return np.zeros((N_g, 1), dtype=np.float64)
    elif drift_type == "linear":
        P_g = np.zeros((N_g, 4), dtype=np.float64)
        # grad p_0 = grad(1) = [0,0,0] → dot n = 0  (already zero)
        # grad p_1 = [1,0,0] → dot n = n_x
        P_g[:, 1] = normals[:, 0]
        P_g[:, 2] = normals[:, 1]
        P_g[:, 3] = normals[:, 2]
        return P_g
    else:
        raise ValueError(f"Unknown drift type: '{drift_type}'")
