"""
ARBF Change-of-Support: Discrete Gaussian Model.

Corrects point-support estimates to block-support using the
affine correction (Matheron 1976).  Eq. 8.1 -- 8.3.

Samples are point-support (1-2m composites).  Blocks are
volume-support (10x10x5m).  The block-scale grade distribution
is narrower than the point-scale distribution.  Without correction,
grade-tonnage curves overstate selectivity.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .kernels import evaluate_kernel
from .utils import pairwise_anisotropic_distance, scale_matrix

logger = logging.getLogger(__name__)


@dataclass
class ChangeOfSupportResult:
    """Result of change-of-support correction."""

    corrected_estimates: np.ndarray
    support_ratio: float
    sigma_point: float
    sigma_block: float
    sigma_within_block: float
    declustered_mean: float
    method: str = "affine"


def within_block_variance(
    block_dims: np.ndarray,
    kernel_type: str = "spheroidal",
    alpha: float = 1.0,
    sill: float = 1.0,
    range_: float = 100.0,
    nugget: float = 0.0,
    n_discretisation: int = 27,
    R: Optional[np.ndarray] = None,
    S: Optional[np.ndarray] = None,
) -> float:
    """Compute within-block variance gamma_bar(V,V) (Eq. 8.2).

    sigma^2_w = (1/n^2) * SUM_i SUM_j gamma(||x_i - x_j||)

    where {x_i} are n discretisation points within the block.

    Parameters
    ----------
    block_dims : np.ndarray
        (3,) block dimensions (dx, dy, dz).
    kernel_type : str
        Kernel function name.
    alpha : float
        Kernel smoothness parameter.
    sill : float
        Variogram sill (partial sill).
    range_ : float
        Practical range.
    nugget : float
        Nugget variance.
    n_discretisation : int
        Number of interior points (8, 27, or 64).
    R, S : np.ndarray, optional
        Global anisotropy rotation and scaling matrices.  When supplied,
        the support integral is computed in the same anisotropic metric as
        the estimation kernel instead of isotropic Euclidean space.

    Returns
    -------
    float
        Within-block variance sigma^2_w.
    """
    if n_discretisation <= 1:
        return 0.0

    # Build discretisation points at origin (including block boundaries)
    n_per_dim = int(round(n_discretisation ** (1.0 / 3.0)))
    n_per_dim = max(n_per_dim, 2)
    # B6 fix: use full range [-0.5, 0.5] including boundaries
    # (was interior-only, underestimating within-block variance by ~2-5%)
    t = np.linspace(-0.5, 0.5, n_per_dim)
    gx, gy, gz = np.meshgrid(t, t, t, indexing="ij")
    points = np.column_stack([
        gx.ravel() * block_dims[0],
        gy.ravel() * block_dims[1],
        gz.ravel() * block_dims[2],
    ])
    n = len(points)

    if R is None and S is None:
        from scipy.spatial.distance import cdist
        D = cdist(points, points, metric="euclidean")
        r = D / max(range_, 1e-12)
    else:
        if R is None:
            R = np.eye(3, dtype=np.float64)
        if S is None:
            S = scale_matrix(range_, range_, range_)
        D = pairwise_anisotropic_distance(points, R, S)
        r = D

    # Variogram: gamma(h) = sill * (1 - phi(r)) + nugget * (h > 0)
    phi = evaluate_kernel(r, kernel_type=kernel_type, alpha=alpha)
    gamma = sill * (1.0 - phi) + nugget * (D > 0).astype(float)

    # Average variogram over all pairs
    sigma_w = np.mean(gamma)

    return float(sigma_w)


def affine_correction(
    raw_estimates: np.ndarray,
    declustered_mean: float,
    sigma_point: float,
    sigma_within_block: float,
    local_means: Optional[np.ndarray] = None,
) -> ChangeOfSupportResult:
    """Apply affine change-of-support correction (Eq. 8.3).

    z*_corrected = m + r * (z*_raw - m)

    where r = sigma_block / sigma_point
          sigma_block = sqrt(sigma_point^2 - sigma_within_block^2)
          m = declustered_mean (global) or local_means (per-block)

    The correction shrinks the distribution toward the mean.
    r < 1 always (block grades are less variable than point grades).

    For deposits with strong spatial trends or distinct mineralisation
    domains, passing ``local_means`` (one value per block) prevents the
    global mean from biasing high/low-grade zones.  Each block is then
    shrunk toward its own local mean rather than the single global value.

    Parameters
    ----------
    raw_estimates : np.ndarray
        (B,) raw point-support block estimates.
    declustered_mean : float
        Declustered global mean grade.  Used when ``local_means`` is None
        and also stored in the diagnostic result.
    sigma_point : float
        Point-support standard deviation (sqrt(sill + nugget)).
    sigma_within_block : float
        Within-block standard deviation (sqrt(sigma^2_w)).
    local_means : np.ndarray, optional
        (B,) per-block local mean grades.  When supplied, each block is
        corrected toward its local mean instead of the global mean.
        Useful for multi-domain or strongly trended deposits.

    Returns
    -------
    ChangeOfSupportResult
        Corrected estimates and diagnostic information.
    """
    sigma_point = max(sigma_point, 1e-12)

    # Block-support variance
    sigma_block_sq = max(sigma_point ** 2 - sigma_within_block ** 2, 0.0)
    sigma_block = np.sqrt(sigma_block_sq)

    # Support correction ratio
    r = sigma_block / sigma_point  # r < 1 always

    # Apply correction — use per-block local means when provided so that
    # blocks in high-grade domains are not over-shrunk toward the global mean.
    m = local_means if local_means is not None else declustered_mean
    corrected = m + r * (raw_estimates - m)

    logger.info(
        "Change-of-support: sigma_point=%.4f, sigma_block=%.4f, "
        "ratio=%.4f, mean=%.4f",
        sigma_point, sigma_block, r, declustered_mean,
    )

    return ChangeOfSupportResult(
        corrected_estimates=corrected,
        support_ratio=float(r),
        sigma_point=float(sigma_point),
        sigma_block=float(sigma_block),
        sigma_within_block=float(sigma_within_block),
        declustered_mean=float(declustered_mean),
        method="affine",
    )
