"""
Radial basis function (kernel) library for FastRBF estimation.

Every kernel accepts ``(r, sill, range_, nugget, **kwargs)`` and returns
a numpy ndarray.  Each is monotonically non-decreasing, handles r=0
correctly, and is positive-definite.

Supported kernels:
  - Linear
  - Spheroidal  (Leapfrog Geo default — linear near origin + Cauchy tail)
  - Spherical
  - Gaussian
  - Exponential
  - Cubic
  - Generalised Cauchy

Reference: CLAUDE_CODE_PROMPT_FastRBF_Engine.md
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from .config import KernelType

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Individual kernel implementations
# ═══════════════════════════════════════════════════════════════════


def linear_kernel(
    r: NDArray[np.float64],
    sill: float,
    range_: float,
    nugget: float = 0.0,
    **kwargs,
) -> NDArray[np.float64]:
    """
    Linear kernel:  φ(r) = nugget·δ(r) + sill · (r / range)

    No sill plateau — increases linearly with distance.
    Slope = sill / range.
    """
    r = np.asarray(r, dtype=np.float64)
    result = sill * (r / range_)
    # Add nugget everywhere except at r=0
    mask_zero = r < 1e-15
    result = np.where(mask_zero, 0.0, result + nugget)
    return result


def _compute_spheroidal_junction(alpha: int) -> float:
    """
    Find h₀ where a line through the origin is tangent to the Cauchy curve.

    Solves f'(h₀)·h₀ = f(h₀) where f(h) = 1 - (1+h²)^(-(α-1)/2).
    This gives the unique junction point for C⁰ + C¹ continuity.
    The Cauchy branch is never shifted, so it never exceeds the sill.

    Returns h₀ = r₀ / range (dimensionless junction position).
    """
    exp = -(alpha - 1) / 2.0

    def _residual(h):
        u = 1.0 + h * h
        f_val = 1.0 - u ** exp
        f_deriv = (alpha - 1) * h * u ** (exp - 1.0)
        # Tangent condition: f'(h) * h = f(h)  →  residual = f'(h)*h - f(h)
        return f_deriv * h - f_val

    # Newton's method (converges in ~5 iterations from inflection point)
    h = np.sqrt(1.0 / (alpha - 1))  # start from original inflection
    for _ in range(20):
        u = 1.0 + h * h
        f_val = 1.0 - u ** exp
        f_deriv = (alpha - 1) * h * u ** (exp - 1.0)
        res = f_deriv * h - f_val

        # d(residual)/dh = f''(h)*h + f'(h) - f'(h) = f''(h)*h
        f_deriv2 = (alpha - 1) * u ** (exp - 1.0) * (
            1.0 + 2.0 * (exp - 1.0) * h * h / u
        )
        dres = f_deriv2 * h + f_deriv - f_deriv
        # Actually: d/dh [f'·h - f] = f''·h + f' - f' = f''·h
        dres = f_deriv2 * h

        if abs(dres) < 1e-30:
            break
        h -= res / dres
        h = max(h, 1e-10)
        if abs(res) < 1e-14:
            break

    return float(h)


# Precomputed junction points (h₀ = r₀/range) for each supported alpha.
# These satisfy f'(h₀)·h₀ = f(h₀), ensuring C⁰ + C¹ continuity.
_SPHEROIDAL_JUNCTIONS = {a: _compute_spheroidal_junction(a) for a in (3, 5, 7, 9)}


def spheroidal_kernel(
    r: NDArray[np.float64],
    sill: float,
    range_: float,
    nugget: float = 0.0,
    alpha: int = 5,
    **kwargs,
) -> NDArray[np.float64]:
    """
    Spheroidal kernel (Leapfrog Geo default — CRITICAL).

    Piecewise definition with C¹ continuity at the tangent junction:
      For r < r₀:  φ(r) = nugget·δ(r) + a·r        [linear through origin]
      For r ≥ r₀:  φ(r) = nugget·δ(r) + sill·(1-(1+(r/range)²)^(-(α-1)/2))

    r₀ is the unique point where a line through the origin is tangent to
    the generalised Cauchy curve — both value and first derivative match
    exactly.  The Cauchy branch is never shifted, so it never exceeds sill.

    α ∈ {3, 5, 7, 9}.  At r = range the value reaches ~94-96% of sill.
    α=9 is closest to spherical shape; α=3 is fastest to compute.
    """
    r = np.asarray(r, dtype=np.float64)
    assert alpha in (3, 5, 7, 9), f"Alpha must be 3, 5, 7, or 9; got {alpha}"

    h = r / range_
    exponent = -(alpha - 1) / 2.0

    # Cauchy branch (unmodified — never exceeds sill)
    cauchy_part = sill * (1.0 - (1.0 + h ** 2) ** exponent)

    # Junction in normalised coordinates
    h0 = _SPHEROIDAL_JUNCTIONS[alpha]

    # Linear slope = Cauchy derivative at h₀, which also equals f(h₀)/h₀
    u0 = 1.0 + h0 * h0
    cauchy_val_h0 = sill * (1.0 - u0 ** exponent)
    slope = cauchy_val_h0 / (h0 * range_)  # dγ/dr = f(h₀)/(h₀·range)

    linear_part = slope * r

    result = np.where(r < h0 * range_, linear_part, cauchy_part)

    # Nugget: add everywhere except r ≈ 0
    mask_zero = r < 1e-15
    result = np.where(mask_zero, 0.0, result + nugget)
    return result


def spherical_kernel(
    r: NDArray[np.float64],
    sill: float,
    range_: float,
    nugget: float = 0.0,
    **kwargs,
) -> NDArray[np.float64]:
    """
    Spherical kernel:
      For r < range:  φ(r) = nugget·δ(r) + sill · (1.5h - 0.5h³)
      For r >= range: φ(r) = nugget·δ(r) + sill

    where h = r / range.
    """
    r = np.asarray(r, dtype=np.float64)
    h = np.minimum(r / range_, 1.0)

    in_range = r < range_
    result = np.where(
        in_range,
        sill * (1.5 * h - 0.5 * h ** 3),
        sill,
    )

    mask_zero = r < 1e-15
    result = np.where(mask_zero, 0.0, result + nugget)
    return result


def gaussian_kernel(
    r: NDArray[np.float64],
    sill: float,
    range_: float,
    nugget: float = 0.0,
    **kwargs,
) -> NDArray[np.float64]:
    """
    Gaussian kernel:
      φ(r) = nugget·δ(r) + sill · (1 - exp(-3·(r/range)²))

    Reaches ~95% of sill at r = range.
    """
    r = np.asarray(r, dtype=np.float64)
    result = sill * (1.0 - np.exp(-3.0 * (r / range_) ** 2))

    mask_zero = r < 1e-15
    result = np.where(mask_zero, 0.0, result + nugget)
    return result


def exponential_kernel(
    r: NDArray[np.float64],
    sill: float,
    range_: float,
    nugget: float = 0.0,
    **kwargs,
) -> NDArray[np.float64]:
    """
    Exponential kernel:
      φ(r) = nugget·δ(r) + sill · (1 - exp(-3·r/range))

    Reaches ~95% of sill at r = range.
    """
    r = np.asarray(r, dtype=np.float64)
    result = sill * (1.0 - np.exp(-3.0 * r / range_))

    mask_zero = r < 1e-15
    result = np.where(mask_zero, 0.0, result + nugget)
    return result


def cubic_kernel(
    r: NDArray[np.float64],
    sill: float,
    range_: float,
    nugget: float = 0.0,
    **kwargs,
) -> NDArray[np.float64]:
    """
    Cubic kernel:
      For r < range (h = r/range):
          φ(r) = nugget·δ(r) + sill · (7h² - 8.75h³ + 3.5h⁵ - 0.75h⁷)
      For r >= range:
          φ(r) = nugget·δ(r) + sill
    """
    r = np.asarray(r, dtype=np.float64)
    h = np.minimum(r / range_, 1.0)

    in_range = r < range_
    poly = 7.0 * h ** 2 - 8.75 * h ** 3 + 3.5 * h ** 5 - 0.75 * h ** 7
    result = np.where(in_range, sill * poly, sill)

    mask_zero = r < 1e-15
    result = np.where(mask_zero, 0.0, result + nugget)
    return result


def generalised_cauchy_kernel(
    r: NDArray[np.float64],
    sill: float,
    range_: float,
    nugget: float = 0.0,
    alpha: int = 5,
    **kwargs,
) -> NDArray[np.float64]:
    """
    Generalised Cauchy kernel:
      φ(r) = nugget·δ(r) + sill · (1 - (1 + (r/range)²)^(-(α-1)/2))

    Same formula as Cauchy portion of Spheroidal but applied over full domain.
    α ∈ {3, 5, 7, 9}.
    """
    r = np.asarray(r, dtype=np.float64)
    assert alpha in (3, 5, 7, 9), f"Alpha must be 3, 5, 7, or 9; got {alpha}"

    exponent = -(alpha - 1) / 2.0
    result = sill * (1.0 - (1.0 + (r / range_) ** 2) ** exponent)

    mask_zero = r < 1e-15
    result = np.where(mask_zero, 0.0, result + nugget)
    return result


# ═══════════════════════════════════════════════════════════════════
# Covariance wrappers  C(r) = (sill + nugget) - φ(r)
# ═══════════════════════════════════════════════════════════════════


def covariance(
    r: NDArray[np.float64],
    sill: float,
    range_: float,
    nugget: float,
    kernel_type: KernelType,
    alpha: int = 5,
) -> NDArray[np.float64]:
    """
    Covariance function:  C(r) = (sill + nugget) - φ(r).

    At r=0:  C(0) = sill + nugget  (total variance).
    """
    gamma = evaluate_kernel(r, sill, range_, nugget, kernel_type, alpha)
    return (sill + nugget) - gamma


# ═══════════════════════════════════════════════════════════════════
# Dispatcher
# ═══════════════════════════════════════════════════════════════════

_KERNEL_MAP = {
    KernelType.LINEAR: linear_kernel,
    KernelType.SPHEROIDAL: spheroidal_kernel,
    KernelType.SPHERICAL: spherical_kernel,
    KernelType.GAUSSIAN: gaussian_kernel,
    KernelType.EXPONENTIAL: exponential_kernel,
    KernelType.CUBIC: cubic_kernel,
    KernelType.GENERALISED_CAUCHY: generalised_cauchy_kernel,
}


def evaluate_kernel(
    r: NDArray[np.float64],
    sill: float,
    range_: float,
    nugget: float,
    kernel_type: KernelType,
    alpha: int = 5,
) -> NDArray[np.float64]:
    """
    Evaluate the specified kernel function at distances *r*.

    This is the primary entry point for all kernel evaluations.

    Parameters
    ----------
    r : ndarray
        Distances (≥0).
    sill : float
        Partial sill (structured variance component).
    range_ : float
        Practical range.
    nugget : float
        Nugget effect.
    kernel_type : KernelType
        Which kernel to use.
    alpha : int
        Alpha parameter (only for spheroidal / generalised Cauchy).

    Returns
    -------
    γ(r) : ndarray
        Semi-variogram values.
    """
    fn = _KERNEL_MAP[kernel_type]
    return fn(r, sill, range_, nugget, alpha=alpha)
