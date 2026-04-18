"""
ARBF Kernel Functions (Covariance Form) — Numba-accelerated.

All kernels are positive-definite covariance functions satisfying:
    phi(0) = 1  (unit sill; actual sill multiplied externally)
    phi(r) >= 0  for all r >= 0
    phi(r) monotonically non-increasing

The normalised distance r = ||h|| / range is passed in.

References
----------
- Buhmann (2003), Radial Basis Functions, Cambridge University Press.
- Rasmussen & Williams (2006), Gaussian Processes for Machine Learning.
- Wendland (2004), Scattered Data Approximation, Cambridge University Press.
"""

from __future__ import annotations

import numpy as np

try:
    from numba import njit, prange
except ImportError:
    # D4: Graceful fallback when Numba is not installed.
    # Define no-op decorators so pure-numpy implementations work.
    def njit(*args, **kwargs):
        def wrapper(fn):
            return fn
        if args and callable(args[0]):
            return args[0]
        return wrapper

    prange = range

# ---------------------------------------------------------------------------
# Numba-jitted kernel implementations (operate in-place on flat arrays)
# ---------------------------------------------------------------------------


@njit(cache=True, parallel=True)
def _spheroidal_flat(r_flat: np.ndarray, alpha: float, out: np.ndarray) -> None:
    """phi(r) = (1 + r^2)^{-alpha}"""
    for i in prange(r_flat.shape[0]):
        out[i] = (1.0 + r_flat[i] * r_flat[i]) ** (-alpha)


@njit(cache=True, parallel=True)
def _gaussian_flat(r_flat: np.ndarray, out: np.ndarray) -> None:
    """phi(r) = exp(-r^2)"""
    for i in prange(r_flat.shape[0]):
        out[i] = np.exp(-(r_flat[i] * r_flat[i]))


@njit(cache=True, parallel=True)
def _matern32_flat(r_flat: np.ndarray, out: np.ndarray) -> None:
    """phi(r) = (1 + sqrt(3)*r) * exp(-sqrt(3)*r)"""
    s3 = 1.7320508075688772  # sqrt(3)
    for i in prange(r_flat.shape[0]):
        s3r = s3 * r_flat[i]
        out[i] = (1.0 + s3r) * np.exp(-s3r)


@njit(cache=True, parallel=True)
def _matern52_flat(r_flat: np.ndarray, out: np.ndarray) -> None:
    """phi(r) = (1 + sqrt(5)*r + 5/3*r^2) * exp(-sqrt(5)*r)"""
    s5 = 2.23606797749979  # sqrt(5)
    for i in prange(r_flat.shape[0]):
        ri = r_flat[i]
        s5r = s5 * ri
        out[i] = (1.0 + s5r + (5.0 / 3.0) * ri * ri) * np.exp(-s5r)


@njit(cache=True, parallel=True)
def _cubic_flat(r_flat: np.ndarray, out: np.ndarray) -> None:
    """phi(r) = max(0, 1 - 3r^2 + 2r^3) for r<1, 0 otherwise"""
    for i in prange(r_flat.shape[0]):
        ri = r_flat[i]
        if ri < 1.0:
            val = 1.0 - 3.0 * ri * ri + 2.0 * ri * ri * ri
            out[i] = max(val, 0.0)
        else:
            out[i] = 0.0


@njit(cache=True, parallel=True)
def _wendland_c2_flat(r_flat: np.ndarray, out: np.ndarray) -> None:
    """phi(r) = (1-r)^4 * (1+4r) for r<1, 0 otherwise"""
    for i in prange(r_flat.shape[0]):
        ri = r_flat[i]
        if ri < 1.0:
            t = 1.0 - ri
            out[i] = t * t * t * t * (1.0 + 4.0 * ri)
        else:
            out[i] = 0.0


@njit(cache=True, parallel=True)
def _spherical_flat(r_flat: np.ndarray, out: np.ndarray) -> None:
    """phi(r) = 1 - 1.5r + 0.5r^3  for r<1, 0 otherwise.

    True spherical covariance model (Matheron 1963) with strict finite range.
    Unlike Spheroidal (Cauchy), this reaches exactly zero at r = 1 (the range),
    so pairs separated beyond the range have zero covariance — the correct
    behaviour for deposits where grade correlation ceases at a finite distance.
    """
    for i in prange(r_flat.shape[0]):
        ri = r_flat[i]
        if ri < 1.0:
            out[i] = 1.0 - 1.5 * ri + 0.5 * ri * ri * ri
        else:
            out[i] = 0.0


# ---------------------------------------------------------------------------
# Public wrappers (preserve original API, delegate to numba)
# ---------------------------------------------------------------------------


def spheroidal(r: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Spheroidal (generalised Cauchy) kernel.

    phi(r) = (1 + r^2)^{-alpha},  alpha > 0
    """
    r = np.asarray(r, dtype=np.float64)
    shape = r.shape
    flat = r.ravel()
    out = np.empty_like(flat)
    _spheroidal_flat(flat, alpha, out)
    return out.reshape(shape)


def gaussian_kernel(r: np.ndarray) -> np.ndarray:
    """Gaussian (squared-exponential) kernel.  phi(r) = exp(-r^2)"""
    r = np.asarray(r, dtype=np.float64)
    shape = r.shape
    flat = r.ravel()
    out = np.empty_like(flat)
    _gaussian_flat(flat, out)
    return out.reshape(shape)


def matern_32(r: np.ndarray) -> np.ndarray:
    """Matern kernel with nu = 3/2."""
    r = np.asarray(r, dtype=np.float64)
    shape = r.shape
    flat = r.ravel()
    out = np.empty_like(flat)
    _matern32_flat(flat, out)
    return out.reshape(shape)


def matern_52(r: np.ndarray) -> np.ndarray:
    """Matern kernel with nu = 5/2."""
    r = np.asarray(r, dtype=np.float64)
    shape = r.shape
    flat = r.ravel()
    out = np.empty_like(flat)
    _matern52_flat(flat, out)
    return out.reshape(shape)


def cubic(r: np.ndarray) -> np.ndarray:
    """Cubic kernel with compact support."""
    r = np.asarray(r, dtype=np.float64)
    shape = r.shape
    flat = r.ravel()
    out = np.empty_like(flat)
    _cubic_flat(flat, out)
    return out.reshape(shape)


def wendland_c2(r: np.ndarray) -> np.ndarray:
    """Wendland C2 kernel with compact support."""
    r = np.asarray(r, dtype=np.float64)
    shape = r.shape
    flat = r.ravel()
    out = np.empty_like(flat)
    _wendland_c2_flat(flat, out)
    return out.reshape(shape)


def spherical_kernel(r: np.ndarray) -> np.ndarray:
    """True spherical covariance kernel with strict finite range.

    phi(r) = 1 - 1.5r + 0.5r^3  for r < 1
    phi(r) = 0                   for r >= 1

    This is the classical Matheron (1963) spherical model.  Unlike the
    Spheroidal (Cauchy) kernel which asymptotically approaches zero,
    this kernel reaches *exactly* zero at r = 1 (the practical range),
    so samples separated beyond the range contribute nothing to the
    kriging system — preventing the over-smoothing that occurs when
    distant samples retain residual correlation.
    """
    r = np.asarray(r, dtype=np.float64)
    shape = r.shape
    flat = r.ravel()
    out = np.empty_like(flat)
    _spherical_flat(flat, out)
    return out.reshape(shape)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_KERNEL_MAP = {
    "spheroidal": spheroidal,
    "gaussian": gaussian_kernel,
    "matern_32": matern_32,
    "matern_52": matern_52,
    "cubic": cubic,
    "wendland_c2": wendland_c2,
    "spherical": spherical_kernel,
}


def evaluate_kernel(
    r: np.ndarray,
    kernel_type: str = "spheroidal",
    alpha: float = 1.0,
) -> np.ndarray:
    """Evaluate a named kernel function."""
    fn = _KERNEL_MAP.get(kernel_type)
    if fn is None:
        raise ValueError(
            f"Unknown kernel type '{kernel_type}'. "
            f"Choose from {list(_KERNEL_MAP.keys())}."
        )
    if kernel_type == "spheroidal":
        return fn(r, alpha=alpha)
    return fn(r)


def kernel_at_zero(kernel_type: str = "spheroidal", alpha: float = 1.0) -> float:
    """Return phi(0) for a given kernel — always 1.0 for all supported kernels."""
    return 1.0


def supported_kernels() -> list[str]:
    """Return list of supported kernel type names."""
    return list(_KERNEL_MAP.keys())
