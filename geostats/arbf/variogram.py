"""
ARBF Local Variogram Fitting.

Computes experimental variograms within sub-domains and fits
theoretical models via weighted least-squares (Cressie weights).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform

from .kernels import evaluate_kernel

logger = logging.getLogger(__name__)


@dataclass
class LocalVariogramResult:
    """Fitted local variogram parameters for a sub-domain."""

    sill: float
    nugget: float
    range_: float
    alpha: float
    kernel_type: str
    fit_residual: float
    n_pairs: int
    n_lags: int


def compute_experimental_variogram(
    coords: np.ndarray,
    values: np.ndarray,
    n_lags: int = 15,
    max_lag: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute omnidirectional experimental variogram (Matheron estimator).

    gamma_hat(h) = 1/(2*N(h)) * SUM_{(i,j) in N(h)} (z_i - z_j)^2

    Parameters
    ----------
    coords : np.ndarray
        (N, 3) sample coordinates.
    values : np.ndarray
        (N,) sample values.
    n_lags : int
        Number of lag bins.
    max_lag : float, optional
        Maximum lag distance.  Defaults to half the data extent.

    Returns
    -------
    lags : np.ndarray
        (n_lags,) lag bin centres.
    semivariance : np.ndarray
        (n_lags,) experimental semivariance at each lag.
    pair_counts : np.ndarray
        (n_lags,) number of pairs per lag bin.
    """
    coords = np.asarray(coords, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64).ravel()
    N = len(values)

    # Pairwise distances (condensed form)
    dists = pdist(coords, metric="euclidean")

    if max_lag is None:
        max_lag = np.max(dists) * 0.5

    # Pairwise squared differences
    sq_diffs = pdist(values.reshape(-1, 1), metric="sqeuclidean").ravel()

    lag_edges = np.linspace(0.0, max_lag, n_lags + 1)
    lags = 0.5 * (lag_edges[:-1] + lag_edges[1:])
    semivariance = np.zeros(n_lags, dtype=np.float64)
    pair_counts = np.zeros(n_lags, dtype=np.int64)

    for k in range(n_lags):
        mask = (dists >= lag_edges[k]) & (dists < lag_edges[k + 1])
        count = np.sum(mask)
        if count > 0:
            semivariance[k] = 0.5 * np.mean(sq_diffs[mask])
            pair_counts[k] = count

    return lags, semivariance, pair_counts


def fit_local_variogram(
    coords: np.ndarray,
    values: np.ndarray,
    kernel_type: str = "spheroidal",
    n_lags: int = 15,
    max_lag: Optional[float] = None,
    use_nugget: bool = True,
) -> LocalVariogramResult:
    """Fit a theoretical variogram model to local data (Eq. 3.3).

    Uses Cressie weights: w_l = N(h_l) / h_l^2
    (more pairs and shorter lag = more weight).

    Parameters
    ----------
    coords : np.ndarray
        (N, 3) local sample coordinates.
    values : np.ndarray
        (N,) local sample values.
    kernel_type : str
        Kernel type name.
    n_lags : int
        Number of lag bins for experimental variogram.
    max_lag : float, optional
        Maximum lag distance.
    use_nugget : bool
        Whether to fit a nugget effect.

    Returns
    -------
    LocalVariogramResult
        Fitted variogram parameters.
    """
    lags, gamma_exp, counts = compute_experimental_variogram(
        coords, values, n_lags=n_lags, max_lag=max_lag,
    )

    # Filter empty bins
    valid = counts > 0
    lags_v = lags[valid]
    gamma_v = gamma_exp[valid]
    counts_v = counts[valid].astype(np.float64)

    if len(lags_v) < 3:
        # Not enough data for fitting; return defaults
        var_data = float(np.var(values)) if len(values) > 1 else 1.0
        return LocalVariogramResult(
            sill=var_data,
            nugget=0.0,
            range_=float(np.max(np.linalg.norm(coords - coords.mean(axis=0), axis=1)) - np.min(np.linalg.norm(coords - coords.mean(axis=0), axis=1))) * 0.5,
            alpha=1.0,
            kernel_type=kernel_type,
            fit_residual=float("inf"),
            n_pairs=int(np.sum(counts)),
            n_lags=len(lags_v),
        )

    # Cressie weights (V1 fix: floor lag distances at the first non-zero
    # lag to prevent empty short-lag bins from giving zero weight while
    # the second bin gets disproportionate weight).
    lag_floor = lags_v[lags_v > 0].min() if np.any(lags_v > 0) else 1.0
    clamped_lags = np.maximum(lags_v, lag_floor)
    weights = counts_v / np.maximum(clamped_lags ** 2, 1e-12)

    # Initial guesses
    sill_init = float(np.max(gamma_v))
    range_init = float(lags_v[len(lags_v) // 2])
    nugget_init = float(gamma_v[0] * 0.1) if use_nugget else 0.0

    def _variogram_model(h: np.ndarray, sill: float, range_: float,
                         nugget: float, alpha: float) -> np.ndarray:
        """gamma(h) = sill * [1 - phi(h/range)] + nugget * (h > 0)."""
        r = h / max(range_, 1e-12)
        phi = evaluate_kernel(r, kernel_type=kernel_type, alpha=alpha)
        return sill * (1.0 - phi) + nugget * (h > 0).astype(float)

    def objective(params: np.ndarray) -> float:
        if use_nugget:
            sill, range_, nugget, alpha = params
        else:
            sill, range_, alpha = params
            nugget = 0.0
        model = _variogram_model(lags_v, sill, range_, nugget, alpha)
        # Weighted least-squares (Cressie form)
        residuals = (gamma_v / np.maximum(model, 1e-12) - 1.0) ** 2
        return float(np.sum(weights * residuals))

    # Bounds — alpha >= 0.5 to prevent ultra-flat kernels that create
    # near-singular kernel matrices (alpha=0.1 → phi(r)=(1+r²)^{-0.1}
    # barely decays → all matrix entries ≈ sill → catastrophic cancellation
    # in variance computation phi_0 - k^T K^{-1} k).
    if use_nugget:
        x0 = [sill_init, range_init, nugget_init, 1.0]
        bounds = [
            (1e-10, sill_init * 5.0),
            (1e-5, float(np.max(lags_v)) * 3.0),
            (0.0, sill_init),
            (0.5, 10.0),
        ]
    else:
        x0 = [sill_init, range_init, 1.0]
        bounds = [
            (1e-10, sill_init * 5.0),
            (1e-5, float(np.max(lags_v)) * 3.0),
            (0.5, 10.0),
        ]

    # Multi-start optimisation over alpha.  L-BFGS-B with a single start
    # at alpha=1.0 often gets trapped because the gradient w.r.t. alpha
    # is nearly flat near 1.0 relative to the scale of sill/range params.
    # Try several alpha seeds and keep the best fit.
    alpha_seeds = [0.5, 1.0, 2.0, 4.0]
    best_result = None
    best_fun = float("inf")

    for a_seed in alpha_seeds:
        x0_try = list(x0)
        x0_try[-1] = a_seed  # alpha is always the last parameter
        try:
            res = minimize(objective, x0_try, method="L-BFGS-B", bounds=bounds)
            if res.fun < best_fun:
                best_fun = res.fun
                best_result = res
        except Exception:
            continue

    if best_result is None:
        # Fallback: run once with default
        best_result = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)

    if use_nugget:
        sill_fit, range_fit, nugget_fit, alpha_fit = best_result.x
    else:
        sill_fit, range_fit, alpha_fit = best_result.x
        nugget_fit = 0.0

    logger.info(
        "Variogram fit: sill=%.4f, nugget=%.4f, range=%.1f, alpha=%.3f "
        "(residual=%.4e, n_lags=%d, n_pairs=%d)",
        sill_fit, nugget_fit, range_fit, alpha_fit,
        best_result.fun, len(lags_v), int(np.sum(counts)),
    )

    return LocalVariogramResult(
        sill=float(sill_fit),
        nugget=float(nugget_fit),
        range_=float(range_fit),
        alpha=float(alpha_fit),
        kernel_type=kernel_type,
        fit_residual=float(best_result.fun),
        n_pairs=int(np.sum(counts)),
        n_lags=len(lags_v),
    )


def fit_spheroidal_alpha(
    lags: np.ndarray,
    gamma_experimental: np.ndarray,
    sill: float,
    nugget: float,
    range_: float,
    pair_counts: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """Find the spheroidal alpha that best matches an experimental variogram.

    Given fixed sill, nugget, range (e.g. from a spherical model fit in the
    variogram panel), find the alpha that makes the spheroidal kernel
    gamma(h) = sill * (1 - (1 + (h/range)²)^{-alpha}) + nugget
    best match the experimental variogram.

    Parameters
    ----------
    lags : np.ndarray
        Lag distances.
    gamma_experimental : np.ndarray
        Experimental semivariance at each lag.
    sill, nugget, range_ : float
        Fixed variogram parameters.
    pair_counts : np.ndarray, optional
        Number of pairs per lag bin (for Cressie weighting).

    Returns
    -------
    alpha : float
        Best-fit alpha.
    residual : float
        Weighted residual of the fit.
    """
    valid = gamma_experimental > 0
    lags_v = lags[valid]
    gamma_v = gamma_experimental[valid]

    if len(lags_v) < 2:
        return 1.0, float("inf")

    if pair_counts is not None:
        counts_v = pair_counts[valid].astype(np.float64)
        weights = counts_v / np.maximum(lags_v ** 2, 1e-12)
    else:
        weights = np.ones_like(lags_v)

    def _model(alpha):
        r = lags_v / max(range_, 1e-12)
        phi = evaluate_kernel(r, kernel_type="spheroidal", alpha=alpha)
        return sill * (1.0 - phi) + nugget * (lags_v > 0).astype(float)

    def _obj(params):
        alpha = params[0]
        model = _model(alpha)
        residuals = (gamma_v / np.maximum(model, 1e-12) - 1.0) ** 2
        return float(np.sum(weights * residuals))

    best_fun = float("inf")
    best_alpha = 1.0

    for a_seed in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
        try:
            res = minimize(_obj, [a_seed], method="L-BFGS-B",
                           bounds=[(0.5, 10.0)])
            if res.fun < best_fun:
                best_fun = res.fun
                best_alpha = float(res.x[0])
        except Exception:
            continue

    logger.info(
        "Spheroidal alpha fit: alpha=%.3f (residual=%.4e) for "
        "sill=%.4f, nugget=%.4f, range=%.1f",
        best_alpha, best_fun, sill, nugget, range_,
    )

    return best_alpha, best_fun


def fit_spheroidal_full(
    lags: np.ndarray,
    gamma_experimental: np.ndarray,
    pair_counts: Optional[np.ndarray] = None,
) -> LocalVariogramResult:
    """Joint fit of all spheroidal variogram parameters.

    Fits sill, nugget, range, and alpha simultaneously using weighted
    least-squares with multi-start optimisation.

    Parameters
    ----------
    lags : np.ndarray
        Lag distances.
    gamma_experimental : np.ndarray
        Experimental semivariance at each lag.
    pair_counts : np.ndarray, optional
        Number of pairs per lag bin.

    Returns
    -------
    LocalVariogramResult
        Fully fitted spheroidal variogram.
    """
    valid = gamma_experimental > 0
    lags_v = lags[valid]
    gamma_v = gamma_experimental[valid]

    if len(lags_v) < 3:
        return LocalVariogramResult(
            sill=float(np.max(gamma_experimental)) if len(gamma_experimental) > 0 else 1.0,
            nugget=0.0,
            range_=float(np.max(lags)) * 0.3 if len(lags) > 0 else 100.0,
            alpha=1.0,
            kernel_type="spheroidal",
            fit_residual=float("inf"),
            n_pairs=int(np.sum(pair_counts)) if pair_counts is not None else 0,
            n_lags=len(lags_v),
        )

    if pair_counts is not None:
        counts_v = pair_counts[valid].astype(np.float64)
        weights = counts_v / np.maximum(lags_v ** 2, 1e-12)
    else:
        weights = np.ones_like(lags_v)

    sill_init = float(np.max(gamma_v))
    range_init = float(lags_v[len(lags_v) // 2])

    def _model(h, sill, range_, nugget, alpha):
        r = h / max(range_, 1e-12)
        phi = evaluate_kernel(r, kernel_type="spheroidal", alpha=alpha)
        return sill * (1.0 - phi) + nugget * (h > 0).astype(float)

    def _obj(params):
        sill, range_, nugget, alpha = params
        model = _model(lags_v, sill, range_, nugget, alpha)
        residuals = (gamma_v / np.maximum(model, 1e-12) - 1.0) ** 2
        return float(np.sum(weights * residuals))

    bounds = [
        (1e-10, sill_init * 5.0),
        (1e-5, float(np.max(lags_v)) * 3.0),
        (0.0, sill_init),
        (0.5, 10.0),
    ]

    best_result = None
    best_fun = float("inf")

    for a_seed in [0.5, 1.0, 2.0, 4.0]:
        x0 = [sill_init, range_init, gamma_v[0] * 0.1, a_seed]
        try:
            res = minimize(_obj, x0, method="L-BFGS-B", bounds=bounds)
            if res.fun < best_fun:
                best_fun = res.fun
                best_result = res
        except Exception:
            continue

    if best_result is None:
        x0 = [sill_init, range_init, gamma_v[0] * 0.1, 1.0]
        best_result = minimize(_obj, x0, method="L-BFGS-B", bounds=bounds)

    sill_fit, range_fit, nugget_fit, alpha_fit = best_result.x

    logger.info(
        "Full spheroidal fit: sill=%.4f, nugget=%.4f, range=%.1f, "
        "alpha=%.3f (residual=%.4e)",
        sill_fit, nugget_fit, range_fit, alpha_fit, best_result.fun,
    )

    return LocalVariogramResult(
        sill=float(sill_fit),
        nugget=float(nugget_fit),
        range_=float(range_fit),
        alpha=float(alpha_fit),
        kernel_type="spheroidal",
        fit_residual=float(best_result.fun),
        n_pairs=int(np.sum(pair_counts)) if pair_counts is not None else 0,
        n_lags=len(lags_v),
    )
