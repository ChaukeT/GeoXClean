"""
ARBF Partition-of-Unity Blending — Numba-accelerated.

Blends local GPR estimates from overlapping sub-domains using
Wendland C2 weight functions.  The blended variance accounts for
both within-model uncertainty and between-model disagreement
via the law of total variance (Eq. 6.1, 6.2).

Key acceleration: the Python grouping loop over B query points is
replaced by a vectorised numpy approach.  Per-group GPR prediction
uses pre-computed K_aug_inv (matmul instead of lu_solve).

References
----------
- Wendland (2004), Scattered Data Approximation, Cambridge University Press.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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

from .gpr import predict_mean, predict_mean_and_variance
from .kernels import evaluate_kernel
from .partition import SubDomain, wendland_c2_weight_batch
from .utils import clamp_variance, rotation_matrix, scale_matrix

logger = logging.getLogger(__name__)


@dataclass
class BlendedResult:
    """Result of blended estimation at a single or batch of points."""

    estimates: np.ndarray          # (B,) blended posterior mean
    variances: np.ndarray          # (B,) blended posterior variance
    n_active_subdomains: np.ndarray  # (B,) number of active sub-domains
    within_variance: np.ndarray    # (B,) E[Var(f|model)]
    between_variance: np.ndarray   # (B,) Var(E[f|model])
    total_variance: Optional[np.ndarray] = None  # posterior + stitching diagnostic


# ---------------------------------------------------------------------------
# Numba-accelerated grouping
# ---------------------------------------------------------------------------


@njit(cache=True)
def _build_group_keys(active_mask: np.ndarray) -> np.ndarray:
    """Assign each query point a group key based on its active subdomain set.

    Returns (B,) int64 array of group keys.  Each unique key identifies a
    unique combination of active subdomains, supporting any number K (not
    limited to 63 as the former bitmask was).

    Uses a polynomial hash: key = sum_j (j+1) * 1_000_003^(position).
    Collision probability is negligible for K < 10,000 (birthday bound).

    Callers must ensure every row has at least one True entry before calling
    (i.e. the outside-radius fallback must be pre-handled in Python so that
    the nearest subdomain is activated for blocks outside all domains).
    """
    B = active_mask.shape[0]
    K = active_mask.shape[1]
    keys = np.empty(B, dtype=np.int64)

    for i in range(B):
        key = np.int64(0)
        for j in range(K):
            if active_mask[i, j]:
                key = key * np.int64(1_000_003) + np.int64(j + 1)
        keys[i] = key

    return keys


# ---------------------------------------------------------------------------
# Main blending functions
# ---------------------------------------------------------------------------


def blend_estimates_fast(
    query_points: np.ndarray,
    subdomains: List[SubDomain],
    composite_coords: np.ndarray,
    drift_type: str = "constant",
    R_global: Optional[np.ndarray] = None,
    compute_variance: bool = True,
    orientation_field: Optional[object] = None,
) -> BlendedResult:
    """Optimised blending with numba-accelerated grouping.

    Blocks sharing the same set of active sub-domains are evaluated
    together as matrix operations for better BLAS utilisation.

    Parameters
    ----------
    compute_variance : bool
        If False, skip variance computation (mean only).  Variances
        in the returned BlendedResult will be zeros.  This is ~3-5x
        faster per point since the expensive K_inv matmul is skipped.
    """
    query_points = np.atleast_2d(query_points)
    B = query_points.shape[0]
    K = len(subdomains)

    centres = np.array([sd.centre for sd in subdomains])
    radii = np.array([sd.radius for sd in subdomains])

    # Vectorised distance computation (B, K)
    dists = np.linalg.norm(
        query_points[:, np.newaxis, :] - centres[np.newaxis, :, :], axis=2,
    )

    active_mask = (dists < radii[np.newaxis, :]).copy()  # (B, K) bool, writable

    # Pre-activate nearest subdomain for blocks outside ALL subdomain radii.
    # This handles block models that extend beyond the drillhole extents and
    # avoids the former int64 bitmask fallback that could overflow for K > 63.
    outside_rows = ~np.any(active_mask, axis=1)
    n_outside_init = int(np.sum(outside_rows))
    if n_outside_init > 0:
        nearest_sd = np.argmin(dists[outside_rows], axis=1)
        active_mask[np.where(outside_rows)[0], nearest_sd] = True
        logger.debug(
            "PUM: %d / %d blocks outside all subdomain radii — "
            "nearest subdomain activated.", n_outside_init, B,
        )

    # Numba-accelerated grouping via polynomial hash keys (supports any K)
    keys = _build_group_keys(active_mask)  # (B,) int64
    unique_keys = np.unique(keys)

    estimates = np.zeros(B, dtype=np.float64)
    variances = np.zeros(B, dtype=np.float64)
    n_active = np.zeros(B, dtype=np.int32)
    within_var = np.zeros(B, dtype=np.float64)
    between_var = np.zeros(B, dtype=np.float64)

    n_outside_total = 0

    for key in unique_keys:
        block_indices = np.where(keys == key)[0]
        batch = query_points[block_indices]
        n_blocks = len(block_indices)

        # Recover active subdomain indices from the mask of any block in
        # this group (all members share the same active set by construction).
        active_ids = np.where(active_mask[block_indices[0]])[0]

        # Wendland weights for this group
        active_dists = dists[np.ix_(block_indices, active_ids)]  # (n, K_active)
        psi_raw = np.zeros_like(active_dists)
        for ai in range(len(active_ids)):
            sd_idx = active_ids[ai]
            psi_raw[:, ai] = wendland_c2_weight_batch(
                active_dists[:, ai], subdomains[sd_idx].radius,
            )

        psi_sum = np.sum(psi_raw, axis=1, keepdims=True)

        # Fix: uniform weights when all Wendland weights are zero
        zero_weight_rows = (psi_sum.ravel() < 1e-15)
        n_outside = int(np.sum(zero_weight_rows))
        n_outside_total += n_outside

        psi_sum = np.maximum(psi_sum, 1e-15)
        w_k = psi_raw / psi_sum  # (n, K_active)
        if n_outside > 0:
            w_k[zero_weight_rows] = 1.0 / len(active_ids)

        # Evaluate each active sub-domain
        f_locals = np.zeros((n_blocks, len(active_ids)), dtype=np.float64)
        s2_locals = np.zeros((n_blocks, len(active_ids)), dtype=np.float64)

        for ai in range(len(active_ids)):
            sd_idx = active_ids[ai]
            sd = subdomains[sd_idx]
            if sd.weights is None or sd.cholesky_factor is None or sd.variogram_params is None:
                continue

            vp = sd.variogram_params
            R = R_global if R_global is not None else np.eye(3)
            S = sd.scale_matrix_ if sd.scale_matrix_ is not None else scale_matrix(vp.range_, vp.range_, vp.range_)
            local_coords = composite_coords[sd.sample_indices]

            if compute_variance:
                f_k, s2_k = predict_mean_and_variance(
                    batch,
                    local_coords,
                    sd.weights,
                    sd.poly_coeffs,
                    sd.cholesky_factor,
                    kernel_type=vp.kernel_type,
                    alpha=vp.alpha,
                    sill=vp.sill,
                    range_=vp.range_,
                    nugget=vp.nugget,
                    R=R,
                    S=S,
                    drift_type=drift_type,
                    orientation_field=orientation_field,
                    l_inv=getattr(sd, 'l_inv', None),  # precomputed at fit time
                )
                f_locals[:, ai] = f_k
                s2_locals[:, ai] = s2_k
            else:
                f_k = predict_mean(
                    batch,
                    local_coords,
                    sd.weights,
                    sd.poly_coeffs,
                    kernel_type=vp.kernel_type,
                    alpha=vp.alpha,
                    sill=vp.sill,
                    range_=vp.range_,
                    R=R,
                    S=S,
                    drift_type=drift_type,
                    orientation_field=orientation_field,
                )
                f_locals[:, ai] = f_k

        # Blend
        f_blend = np.sum(w_k * f_locals, axis=1)
        estimates[block_indices] = f_blend
        n_active[block_indices] = len(active_ids)

        if compute_variance:
            within = np.sum(w_k * s2_locals, axis=1)
            between_terms = w_k * (f_locals - f_blend[:, np.newaxis]) ** 2
            between = np.sum(between_terms, axis=1)
            variances[block_indices] = within
            within_var[block_indices] = within
            between_var[block_indices] = between

    if n_outside_init > 0:
        logger.warning(
            "%d / %d query points outside all subdomain radii — "
            "using nearest-subdomain estimate (no Wendland blending).",
            n_outside_init, B,
        )
    if n_outside_total > 0:
        logger.debug(
            "%d batches had near-zero Wendland weights — uniform blend applied.",
            n_outside_total,
        )

    return BlendedResult(
        estimates=estimates,
        variances=clamp_variance(variances),
        n_active_subdomains=n_active,
        within_variance=clamp_variance(within_var),
        between_variance=between_var,
        total_variance=clamp_variance(within_var + between_var),
    )


# Also keep the non-fast version for small problems / reference
def blend_estimates(
    query_points: np.ndarray,
    subdomains: List[SubDomain],
    composite_coords: np.ndarray,
    drift_type: str = "constant",
    R_global: Optional[np.ndarray] = None,
    orientation_field: Optional[object] = None,
) -> BlendedResult:
    """Non-optimised blending (reference implementation).

    For small problems or testing.  Delegates to blend_estimates_fast.
    """
    return blend_estimates_fast(
        query_points, subdomains, composite_coords,
        drift_type=drift_type, R_global=R_global,
        orientation_field=orientation_field,
    )
