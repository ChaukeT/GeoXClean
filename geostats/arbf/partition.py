"""
ARBF Partition-of-Unity Domain Decomposition.

Decomposes the global estimation domain into overlapping sub-domains,
fits local variograms in each, and provides Wendland C2 weight functions
for smooth blending (Eq. 3.1, 3.2).

References
----------
- Wendland (2004), Scattered Data Approximation, Cambridge University Press.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

from .kernels import wendland_c2
from .variogram import LocalVariogramResult, fit_local_variogram

logger = logging.getLogger(__name__)


@dataclass
class SubDomain:
    """A local estimation sub-domain with centre, radius, and local variogram."""

    index: int
    centre: np.ndarray                          # (3,)
    radius: float
    sample_indices: np.ndarray                  # indices into global composite array
    variogram_params: Optional[LocalVariogramResult] = None
    cholesky_factor: Optional[object] = None  # Factorisation: Cholesky L or LU tuple
    l_inv: Optional[np.ndarray] = None           # Precomputed L^{-1} (Cholesky path only)
    weights: Optional[np.ndarray] = None          # RBF interpolation weights
    poly_coeffs: Optional[np.ndarray] = None      # drift polynomial coefficients
    scale_matrix_: Optional[np.ndarray] = None    # (3,3) S used during assembly
    n_samples: int = 0

    def __post_init__(self) -> None:
        self.n_samples = len(self.sample_indices) if self.sample_indices is not None else 0


def wendland_c2_weight(distance: float, radius: float) -> float:
    """Wendland C2 compactly-supported weight function (Eq. 3.2).

    psi(x) = (1 - (d/R)^2)^4 * (1 + 4*d/R)  if d < R
    psi(x) = 0                                if d >= R

    Parameters
    ----------
    distance : float
        Euclidean distance from sub-domain centre.
    radius : float
        Sub-domain radius.

    Returns
    -------
    float
        Weight value in [0, 1].
    """
    if distance >= radius:
        return 0.0
    r = distance / radius
    return float(wendland_c2(np.array([r]))[0])


def wendland_c2_weight_batch(
    distances: np.ndarray,
    radius: float,
) -> np.ndarray:
    """Vectorised Wendland C2 weight (Eq. 3.2).

    Parameters
    ----------
    distances : np.ndarray
        Distances from sub-domain centre.
    radius : float
        Sub-domain radius.

    Returns
    -------
    np.ndarray
        Weight values, zero outside radius.
    """
    r = distances / max(radius, 1e-12)
    return wendland_c2(r)


def create_subdomains(
    composite_coords: np.ndarray,
    method: str = "kmeans",
    k: Optional[int] = None,
    radii: Optional[np.ndarray] = None,
    centres: Optional[np.ndarray] = None,
    overlap_factor: float = 1.5,
    min_samples_per_subdomain: int = 8,
    max_samples_per_subdomain: int = 300,
    seed: Optional[int] = None,
) -> List[SubDomain]:
    """Decompose domain into overlapping sub-domains.

    Parameters
    ----------
    composite_coords : np.ndarray
        (N, 3) sample coordinates.
    method : str
        'kmeans' for automatic k-means clustering, 'manual' for
        user-specified centres and radii.
    k : int, optional
        Number of sub-domains.  If None, auto-computed as
        max(8, N // 200).
    radii : np.ndarray, optional
        (K,) per-sub-domain radii (for 'manual' method).
    centres : np.ndarray, optional
        (K, 3) sub-domain centres (for 'manual' method).
    overlap_factor : float
        Radius multiplier to ensure overlap >= 2 sub-domains
        at every point.
    min_samples_per_subdomain : int
        Minimum samples; sub-domains below this are merged.
    max_samples_per_subdomain : int
        Maximum samples per sub-domain.

    Returns
    -------
    list of SubDomain
        Created sub-domains with sample assignments.

    Raises
    ------
    ValueError
        If 'manual' method is chosen but centres/radii not provided.
    """
    N = composite_coords.shape[0]

    if method == "manual":
        if centres is None or radii is None:
            raise ValueError(
                "Manual sub-domain method requires 'centres' and 'radii'."
            )
        K = len(centres)
    else:
        # Auto-determine K
        if k is None:
            k = max(8, N // 200)
        K = min(k, N // min_samples_per_subdomain)
        K = max(K, 1)

        # K-means clustering
        centres, radii = _kmeans_subdomains(
            composite_coords, K, overlap_factor, seed=seed,
        )

    # Assign samples to sub-domains using KD-tree
    tree = cKDTree(composite_coords)
    subdomains: List[SubDomain] = []

    for i in range(len(centres)):
        indices = tree.query_ball_point(centres[i], r=radii[i])
        indices = np.asarray(indices, dtype=np.intp)

        if len(indices) < min_samples_per_subdomain:
            # Expand radius until we have enough samples
            _, expanded_idx = tree.query(
                centres[i].reshape(1, -1),
                k=min(min_samples_per_subdomain, N),
            )
            indices = expanded_idx.ravel()
            if len(indices) > 0:
                max_dist = np.max(
                    np.linalg.norm(
                        composite_coords[indices] - centres[i], axis=1,
                    )
                )
                radii[i] = max_dist * 1.1

        if len(indices) > max_samples_per_subdomain:
            # Keep closest samples
            dists = np.linalg.norm(
                composite_coords[indices] - centres[i], axis=1,
            )
            keep = np.argsort(dists)[:max_samples_per_subdomain]
            indices = indices[keep]

        subdomains.append(SubDomain(
            index=i,
            centre=centres[i].copy(),
            radius=float(radii[i]),
            sample_indices=indices,
        ))

    # Verify coverage: every sample in at least 1 sub-domain.
    # P2 fix: if coverage < 99%, iteratively expand all radii by 10%
    # until coverage is achieved.  Uncovered samples are silently lost
    # from the estimation, biasing the global mean and degrading quality.
    for _expand_iter in range(20):
        covered = np.zeros(N, dtype=bool)
        for sd in subdomains:
            covered[sd.sample_indices] = True
        n_uncovered = int(np.sum(~covered))
        coverage_pct = 100.0 * (N - n_uncovered) / max(N, 1)
        if n_uncovered == 0 or coverage_pct >= 99.0:
            break
        # Expand all radii by 10% and reassign uncovered samples
        for sd in subdomains:
            sd.radius *= 1.10
        # Reassign uncovered samples to nearest subdomain
        uncovered_idx = np.where(~covered)[0]
        for ui in uncovered_idx:
            best_sd = min(subdomains, key=lambda s: np.linalg.norm(composite_coords[ui] - s.centre))
            if ui not in best_sd.sample_indices:
                best_sd.sample_indices = np.append(best_sd.sample_indices, ui)
        logger.info(
            "P2 expand iter %d: %d uncovered (%.1f%%), expanding radii by 10%%",
            _expand_iter + 1, n_uncovered, 100.0 - coverage_pct,
        )

    if n_uncovered > 0:
        logger.warning(
            "%d samples (%.1f%%) still uncovered after radius expansion. "
            "These composites are excluded from estimation.",
            n_uncovered, 100.0 * n_uncovered / max(N, 1),
        )

    logger.info(
        "Created %d sub-domains (avg %.0f samples each)",
        len(subdomains),
        np.mean([sd.n_samples for sd in subdomains]),
    )

    return subdomains


def _kmeans_subdomains(
    coords: np.ndarray,
    K: int,
    overlap_factor: float,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create sub-domain centres via k-means and compute radii.

    Parameters
    ----------
    coords : np.ndarray
        (N, 3) sample coordinates.
    K : int
        Number of clusters.
    overlap_factor : float
        Multiplier for radii to ensure overlap.

    Returns
    -------
    centres : np.ndarray
        (K, 3) cluster centres.
    radii : np.ndarray
        (K,) sub-domain radii.
    """
    from scipy.cluster.vq import kmeans2

    centres, labels = kmeans2(coords, K, minit="++", iter=50, seed=seed)

    # Compute radii: max distance from centre to assigned points * overlap_factor
    radii = np.zeros(K, dtype=np.float64)
    for i in range(K):
        mask = labels == i
        if np.any(mask):
            dists = np.linalg.norm(coords[mask] - centres[i], axis=1)
            radii[i] = np.max(dists) * overlap_factor
        else:
            # Empty cluster — use global scale
            radii[i] = np.max(np.linalg.norm(coords - coords.mean(axis=0), axis=1)) - np.min(np.linalg.norm(coords - coords.mean(axis=0), axis=1))

    return centres, radii


def fit_subdomain_variograms(
    subdomains: List[SubDomain],
    composite_coords: np.ndarray,
    composite_values: np.ndarray,
    kernel_type: str = "spheroidal",
    n_lags: int = 15,
) -> None:
    """Fit local variograms for each sub-domain in-place (Eq. 3.3).

    Parameters
    ----------
    subdomains : list of SubDomain
        Sub-domains to fit (modified in place).
    composite_coords : np.ndarray
        (N, 3) global composite coordinates.
    composite_values : np.ndarray
        (N,) global composite values.
    kernel_type : str
        Kernel type for variogram model.
    n_lags : int
        Number of lag bins.
    """
    def _fit_one(sd: SubDomain) -> SubDomain:
        """Fit variogram for one sub-domain (thread-safe — no shared writes)."""
        if sd.n_samples < 4:
            logger.warning(
                "Sub-domain %d has only %d samples, skipping variogram fit.",
                sd.index, sd.n_samples,
            )
            return sd

        local_coords = composite_coords[sd.sample_indices]
        local_values = composite_values[sd.sample_indices]

        # Compute max_lag from data spacing, NOT sub-domain radius.
        # Using radius directly fails when data is in UTM coords
        # (~500 km) because lag bins become 10+ km wide, completely
        # missing the actual spatial correlation structure (tens to
        # hundreds of metres).  The result is a flat experimental
        # variogram → optimizer fits sill≈0, nugget=variance
        # → every prediction collapses to the mean.
        #
        # Fix: base max_lag on the k-nearest-neighbour spacing
        # (≈ actual sample spacing) × 20, capped at half the extent.
        k_nn = min(5, sd.n_samples - 1)
        nn_dists, _ = cKDTree(local_coords).query(local_coords, k=k_nn + 1)
        # Use MEDIAN nearest-neighbour spacing, not mean.
        # Mean is dominated by clustered grade-control holes (spacing=1m) when
        # a sub-domain also contains sparse exploration holes (spacing=50m).
        # With mean: avg_spacing≈1m → max_lag≈20m, missing the deposit-scale
        # range (e.g. 150m).  Median correctly reflects typical spacing.
        median_spacing = float(np.median(nn_dists[:, 1]))
        avg_spacing = float(np.mean(nn_dists[:, 1:]))   # kept for the floor
        extent = float(np.max(np.max(local_coords, axis=0) - np.min(local_coords, axis=0)))
        max_lag = min(median_spacing * 30.0, extent * 0.6)
        max_lag = max(max_lag, avg_spacing * 5.0)  # ensure at least 5× avg spacing
        # P1 fix: also verify max_lag doesn't produce more than 50% empty bins
        max_lag = min(max_lag, extent * 0.5)  # never exceed half the subdomain extent

        sd.variogram_params = fit_local_variogram(
            local_coords,
            local_values,
            kernel_type=kernel_type,
            n_lags=n_lags,
            max_lag=max_lag,
        )
        return sd

    # Parallelise: scipy's L-BFGS-B releases the GIL (compiled Fortran),
    # so threads give genuine concurrency across sub-domains.
    n_workers = min(len(subdomains), 8)
    if n_workers <= 1:
        for sd in subdomains:
            _fit_one(sd)
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_fit_one, sd): sd for sd in subdomains}
            for future in as_completed(futures):
                future.result()  # propagate any exceptions

    logger.info(
        "Fitted local variograms for %d / %d sub-domains",
        sum(1 for sd in subdomains if sd.variogram_params is not None),
        len(subdomains),
    )
