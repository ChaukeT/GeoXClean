"""
Variogram helper functions.

Consolidated, GSLIB-style model kernels plus flexible fitting. Provides
backward-compatible shims for existing callers (_pairwise_variogram,
calculate_experimental_variogram, fit_variogram) while exporting the newer
fit_variogram_model and MODEL_MAP.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np
try:
    from scipy.spatial import cKDTree
except ImportError:  # pragma: no cover
    cKDTree = None

logger = logging.getLogger(__name__)

ModelType = Literal["spherical", "exponential", "gaussian"]


# ----------------------------------------------------------------------
# Model kernels (GSLIB practical range convention)
# ----------------------------------------------------------------------
def spherical_model(h: np.ndarray, range_: float, sill: float, nugget: float) -> np.ndarray:
    """
    Spherical model.
    gamma(h) = c0 + c * (1.5(h/a) - 0.5(h/a)^3) if h < a
             = c0 + c                           otherwise
    """
    h = np.asarray(h, dtype=float)
    a = max(range_, 1e-9)
    c = max(sill - nugget, 0.0)
    hr = h / a
    gamma = np.full_like(h, nugget + c, dtype=float)
    mask = hr < 1.0
    gamma[mask] = nugget + c * (1.5 * hr[mask] - 0.5 * hr[mask] ** 3)
    return gamma


def exponential_model(h: np.ndarray, range_: float, sill: float, nugget: float) -> np.ndarray:
    """
    Exponential model (practical range: 95% sill at h = range_).
    gamma(h) = c0 + c * (1 - exp(-3h/a))
    """
    h = np.asarray(h, dtype=float)
    a = max(range_, 1e-9)
    c = max(sill - nugget, 0.0)
    return nugget + c * (1.0 - np.exp(-3.0 * h / a))


def gaussian_model(h: np.ndarray, range_: float, sill: float, nugget: float) -> np.ndarray:
    """
    Gaussian model (practical range: 95% sill at h = range_).
    gamma(h) = c0 + c * (1 - exp(-3 (h/a)^2))
    """
    h = np.asarray(h, dtype=float)
    a = max(range_, 1e-9)
    c = max(sill - nugget, 0.0)
    return nugget + c * (1.0 - np.exp(-3.0 * (h / a) ** 2))


MODEL_MAP: Dict[str, callable] = {
    "spherical": spherical_model,
    "exponential": exponential_model,
    "gaussian": gaussian_model,
}

# Legacy aliases for backwards compatibility
MODEL_FUN = {
    "spherical": spherical_model,
    "exponential": exponential_model,
    "gaussian": gaussian_model,
}


# ----------------------------------------------------------------------
# Fitting
# ----------------------------------------------------------------------
def fit_variogram_model(
    lags: np.ndarray,
    gammas: np.ndarray,
    model_type: str = "spherical",
    weights: Optional[np.ndarray] = None,
    max_lag: Optional[float] = None,
    sill_norm: bool = False,
    sill_cap: Optional[float] = None,
) -> Tuple[float, float, float]:
    """
    Fit (nugget, sill, range) using scipy.optimize.curve_fit when available,
    falling back to a coarse grid search if optimisation fails.
    
    Industry-standard approach:
    - Nugget estimated from first lag or extrapolated y-intercept
    - Sill estimated from plateau of experimental variogram
    - Range estimated from distance where variogram reaches ~95% of sill
    
    GEOSTATISTICAL CONSTRAINT:
    The sill should approximate the sample variance. When fitting directional
    variograms, providing sill_cap (typically from omnidirectional sill or sample
    variance) prevents unrealistic sill values in directions with few pairs.
    
    Parameters
    ----------
    lags : array
        Lag distances
    gammas : array
        Semivariances at each lag
    model_type : str
        Variogram model type ('spherical', 'exponential', 'gaussian')
    weights : array, optional
        Weights for each point
    max_lag : float, optional
        Maximum lag distance for range constraint
    sill_norm : bool
        Whether to normalize by sample variance
    sill_cap : float, optional
        IMPORTANT: Maximum allowable sill value. Should be set to the sample
        variance or omnidirectional sill to prevent unrealistic fits in
        directions with sparse data. This ensures geostatistical soundness.

    Returns
    -------
    nugget : float
        Nugget effect (C0) - the discontinuity at the origin
    total_sill : float
        **IMPORTANT**: This is the TOTAL sill (C0 + C), NOT partial sill!
        Total sill = nugget + partial_sill, representing the asymptotic semivariance.
        This follows GSLIB/SGeMS/PyKrige convention.
        To get partial sill (contribution): partial_sill = total_sill - nugget
    range : float
        Practical range - the distance where the model reaches ~95% of the sill

    Notes
    -----
    SILL SEMANTICS: The returned sill is TOTAL sill (C0 + C), not partial sill (C).
    This is the industry-standard convention used in GSLIB, SGeMS, and PyKrige.
    """
    lags = np.asarray(lags, float)
    gammas = np.asarray(gammas, float)
    mask = np.isfinite(lags) & np.isfinite(gammas) & (lags >= 0) & (gammas >= 0)
    x = lags[mask]
    y = gammas[mask]
    
    if x.size < 3:
        # Not enough data - return simple estimates
        sill_est = float(np.nanmax(y)) if y.size else 1.0
        if sill_cap is not None and sill_est > sill_cap * 1.2:
            sill_est = sill_cap
        range_est = float(np.nanmax(x) * 0.7) if x.size else 1.0
        return 0.0, sill_est, max(range_est, 1.0)

    func = MODEL_MAP.get(model_type, spherical_model)

    # Sort by lag for proper analysis
    sort_idx = np.argsort(x)
    x = x[sort_idx]
    y = y[sort_idx]

    # Optional sill normalization (use sample variance as reference)
    scale_var = 1.0
    if sill_norm:
        var = float(np.nanmax(y)) if np.isfinite(y).any() else 1.0
        if var > 0:
            y = y / var
            scale_var = var

    # Industry-standard initial estimates from experimental variogram
    y_min = float(np.nanmin(y))
    y_max = float(np.nanmax(y))
    x_max = float(np.nanmax(x))
    x_min = float(np.nanmin(x[x > 0])) if np.any(x > 0) else 1.0
    
    # 1. Nugget estimation - critical for proper variogram modeling
    # Best practices:
    # - Use duplicate pairs (h≈0) if available
    # - Use tight first lag bins (< 5m ideally)
    # - Extrapolate carefully to h=0
    
    # Check if we have very short-distance pairs (potential duplicates or tight spacing)
    very_short_mask = x < x_min * 1.5  # Pairs in first ~1.5x minimum spacing
    
    if np.sum(very_short_mask) >= 2:
        # Use mean of shortest-distance pairs for nugget estimate
        # This captures measurement error + micro-scale variability
        nugget0 = float(np.mean(y[very_short_mask]))
    elif len(x) >= 3 and x[0] > 0:
        # Weighted linear extrapolation to h=0 using first 3 points
        # Weight closer points more heavily
        n_pts = min(3, len(x))
        x_early = x[:n_pts]
        y_early = y[:n_pts]
        weights = 1.0 / (x_early + 1e-6)  # Higher weight for smaller distances
        weights = weights / weights.sum()
        
        # Weighted linear regression
        x_mean = np.sum(weights * x_early)
        y_mean = np.sum(weights * y_early)
        slope = np.sum(weights * (x_early - x_mean) * (y_early - y_mean)) / (np.sum(weights * (x_early - x_mean)**2) + 1e-12)
        nugget0 = max(0.0, float(y_mean - slope * x_mean))
    elif len(x) >= 2 and x[0] > 0:
        # Simple linear extrapolation
        slope = (y[1] - y[0]) / (x[1] - x[0] + 1e-12)
        nugget0 = max(0.0, float(y[0] - slope * x[0]))
    else:
        nugget0 = max(0.0, y_min * 0.8)
    
    # Sanity check: nugget shouldn't exceed the minimum observed gamma significantly
    # (unless we have very few points)
    if len(y) >= 5:
        nugget0 = min(nugget0, y_min * 1.2)
    
    # 2. Sill: estimate from the plateau (typically max gamma or mean of outer lags)
    # Use mean of last 1/3 of lags as sill estimate (plateau region)
    n_outer = max(1, len(y) // 3)
    sill0 = float(np.mean(y[-n_outer:])) if len(y) > 0 else y_max
    sill0 = max(sill0, y_max * 0.8)  # Ensure sill is reasonable
    
    # Ensure sill > nugget with margin
    if sill0 <= nugget0 * 1.1:
        sill0 = nugget0 + (y_max - y_min) * 0.5 + 0.1
    
    # 3. Range: find distance where variogram first approaches ~80% of sill
    target_gamma = nugget0 + 0.8 * (sill0 - nugget0)
    range0 = x_max * 0.5  # Default to half max distance
    for i, yi in enumerate(y):
        if yi >= target_gamma:
            range0 = float(x[i])
            break
    range0 = max(range0, x_max * 0.2)  # At least 20% of max distance

    sigma = None
    if weights is not None:
        w = np.asarray(weights, float)
        # Apply mask if weights match original length, otherwise assume already masked
        if len(w) == len(lags):
            w = w[mask]
        if len(w) == len(x):
            w = w[sort_idx]
            sigma = 1.0 / (w + 1e-6)

    # Bound range to observed lags with margin
    r_lo = max(float(np.nanmin(x[x > 0])) if np.any(x > 0) else 1e-3, 1e-3)
    r_hi = max_lag if max_lag is not None else x_max * 2.0
    r_hi = max(r_hi, r_lo * 2.0)
    
    # Ensure initial range is within bounds
    range0 = max(r_lo, min(range0, r_hi))
    
    # Sill bounds: must be strictly greater than nugget
    # GEOSTATISTICAL CONSTRAINT: Cap sill to prevent unrealistic values
    # The sill should not exceed the maximum observed gamma by much,
    # and should not exceed any provided sill_cap (e.g., omni sill or sample variance)
    sill_lo = nugget0 * 1.05 + 0.01
    
    # Default upper bound: observed max + 30% margin (not 2x which was too loose)
    sill_hi = y_max * 1.3
    
    # If sill_cap provided, it takes PRIORITY - this is the key geostatistical constraint
    # The sill_cap represents the sample variance or omnidirectional sill, which the
    # directional sill should not exceed significantly
    if sill_cap is not None:
        # Allow 20% margin above the cap for flexibility, but no more
        sill_hi = sill_cap * 1.2
        logger.debug(f"Sill capped at {sill_hi:.3f} (cap={sill_cap:.3f})")
        
        # Also constrain initial sill estimate if it's too high
        if sill0 > sill_cap * 1.2:
            sill0 = sill_cap
    
    # Ensure sill_hi is valid (at least sill_lo * 1.1)
    sill_hi = max(sill_hi, sill_lo * 1.1)
    
    # Nugget bounds: 0 to fraction of sill
    nug_hi = sill0 * 0.9

    try:
        from scipy.optimize import curve_fit

        popt, _ = curve_fit(
            func,
            x,
            y,
            p0=[range0, sill0, nugget0],
            bounds=([r_lo, sill_lo, 0.0], [r_hi, sill_hi, nug_hi]),
            sigma=sigma,
            maxfev=10000,
        )
        rng, sill, nug = popt
        
        # Ensure sill > nugget in result
        if sill <= nug:
            sill = nug + 0.1 * scale_var
        
        nug = float(nug) * scale_var
        sill = float(sill) * scale_var
        logger.debug(f"Fitted {model_type}: nugget={nug:.4f}, sill={sill:.4f}, range={rng:.2f}")
        return nug, sill, float(rng)
    except Exception as exc:
        logger.debug("curve_fit failed, falling back to grid search: %s", exc)
        # Pass sill_cap to grid search (adjust for scale_var)
        grid_sill_cap = sill_cap / scale_var if sill_cap is not None else None
        nug, sill, rng = _fit_variogram_grid(x, y, model_type, r_lo, r_hi, grid_sill_cap)
        nug *= scale_var
        sill *= scale_var
        return nug, sill, rng


def _fit_variogram_grid(dist: np.ndarray, gamma: np.ndarray, model: str, r_lo: float, r_hi: float, sill_cap: Optional[float] = None) -> Tuple[float, float, float]:
    """
    Grid search fallback for variogram fitting.
    Uses weighted least squares with emphasis on finding spatial structure.
    
    Parameters
    ----------
    sill_cap : float, optional
        Maximum allowable sill value for geostatistical soundness
    """
    dist = np.asarray(dist, float)
    gamma = np.asarray(gamma, float)
    
    # Sort by distance
    sort_idx = np.argsort(dist)
    dist = dist[sort_idx]
    gamma = gamma[sort_idx]
    
    if len(gamma) == 0:
        return 0.0, 1.0, (r_lo + r_hi) / 2
    
    gmax = float(np.nanmax(gamma))
    gmin = float(np.nanmin(gamma))
    dmax = float(np.nanmax(dist))
    
    # Estimate nugget from early lags (first 20%)
    n_early = max(1, len(gamma) // 5)
    nugget_est = float(np.mean(gamma[:n_early]))
    
    # Estimate sill from late lags (last 30%)
    n_late = max(1, len(gamma) // 3)
    sill_est = float(np.mean(gamma[-n_late:]))
    
    # Ensure sill > nugget
    if sill_est <= nugget_est:
        sill_est = gmax
        nugget_est = gmin * 0.5
    
    # Range bounds
    nug_lo, nug_hi = 0.0, min(sill_est * 0.8, gmax * 0.9)
    sill_lo = max(nugget_est * 1.1, gmin + 0.1)
    
    # GEOSTATISTICAL CONSTRAINT: Cap sill for soundness
    # If sill_cap provided, it takes PRIORITY over data-based bounds
    if sill_cap is not None:
        sill_hi = sill_cap * 1.2
        # Also constrain sill estimate
        if sill_est > sill_cap:
            sill_est = sill_cap
    else:
        sill_hi = gmax * 1.3  # More conservative than original 1.5x
    sill_hi = max(sill_hi, sill_lo * 1.1)  # Ensure valid range
    
    rng_lo, rng_hi = max(r_lo, dmax * 0.1), min(r_hi, dmax * 1.2)
    
    model_fun = MODEL_MAP.get(model, spherical_model)

    # Weights: balanced - emphasize early lags for nugget, mid for range
    weights = np.exp(-dist / (dmax * 0.5))
    weights = weights / weights.sum()

    ngrid = 15
    nug_candidates = np.linspace(nug_lo, nug_hi, ngrid)
    sill_candidates = np.linspace(sill_lo, sill_hi, ngrid)
    rng_candidates = np.linspace(rng_lo, rng_hi, ngrid)

    best = (np.inf, nugget_est, sill_est, (rng_lo + rng_hi) / 2)
    for n in nug_candidates:
        for s in sill_candidates:
            if s <= n * 1.05:  # Sill must be > nugget with margin
                continue
            for r in rng_candidates:
                pred = model_fun(dist, r, s, n)
                # Weighted MSE
                err = float(np.sum(weights * (gamma - pred) ** 2))
                if err < best[0]:
                    best = (err, n, s, r)

    _, nugget, sill, prange = best
    
    # Final sanity check
    if sill <= nugget:
        sill = nugget + (gmax - gmin) * 0.5 + 0.1
    
    logger.debug(f"Grid search {model}: nugget={nugget:.4f}, sill={sill:.4f}, range={prange:.2f}")
    return float(nugget), float(sill), float(prange)


# Backward-compatible alias
def fit_variogram(dist: np.ndarray, gamma: np.ndarray, model: ModelType = "spherical") -> Tuple[float, float, float]:
    nug, sill, rng = fit_variogram_model(dist, gamma, model_type=model)
    # Convert to nugget, partial sill, range
    return nug, max(sill - nug, 0.0), rng


def fit_nested_variogram(
    lags: np.ndarray,
    gammas: np.ndarray,
    model_type1: str = "spherical",
    model_type2: str = "exponential",
    n_structures: int = 2,
) -> Dict[str, Any]:
    """
    Fit a nested variogram model with 2 or 3 structures.
    
    Industry-standard approach:
    - Structure 1: Short-range (captures local variability)
    - Structure 2: Long-range (captures geological trend)
    - Structure 3 (optional): Regional/very long-range
    
    The fitting process:
    1. First fit single-structure model to get overall sill and nugget
    2. Then fit nested model by finding optimal split point
    3. Optimize contributions and ranges for each structure
    
    Parameters
    ----------
    lags : array
        Lag distances
    gammas : array
        Experimental semivariance values
    model_type1 : str
        Model type for first (short-range) structure
    model_type2 : str
        Model type for second (long-range) structure
    n_structures : int
        Number of structures (2 or 3)
    
    Returns
    -------
    dict with keys:
        - nugget: float
        - total_sill: float
        - structures: list of dicts with {type, contribution, range}
    """
    lags = np.asarray(lags, float)
    gammas = np.asarray(gammas, float)
    
    # Clean data
    mask = np.isfinite(lags) & np.isfinite(gammas) & (lags > 0) & (gammas >= 0)
    x = lags[mask]
    y = gammas[mask]
    
    if len(x) < 5:
        # Not enough data for nested model
        nug, sill, rng = fit_variogram_model(x, y, model_type=model_type1)
        return {
            "nugget": nug,
            "total_sill": sill,
            "structures": [{"type": model_type1, "contribution": sill - nug, "range": rng}]
        }
    
    # Sort by lag
    sort_idx = np.argsort(x)
    x = x[sort_idx]
    y = y[sort_idx]
    
    # Step 1: Fit single model to get baseline
    nug_single, sill_single, range_single = fit_variogram_model(x, y, model_type=model_type1)
    
    # Step 2: Find optimal split point for nested model
    # The split point is where short-range structure reaches its sill
    # Typically 20-40% of the total range
    
    model1_func = MODEL_MAP.get(model_type1, spherical_model)
    model2_func = MODEL_MAP.get(model_type2, exponential_model)
    
    best_fit = {"error": np.inf}
    total_contrib = sill_single - nug_single
    
    # Grid search over split ratios and range ratios
    for split_ratio in [0.2, 0.3, 0.4, 0.5]:  # Contribution to structure 1
        for range_ratio in [0.15, 0.25, 0.35, 0.5]:  # Range1 / Range2
            c1 = total_contrib * split_ratio
            c2 = total_contrib * (1 - split_ratio)
            r1 = range_single * range_ratio
            r2 = range_single
            
            # Compute model prediction
            pred = nug_single + model1_func(x, r1, c1 + nug_single, nug_single) - nug_single
            pred += model2_func(x, r2, c2 + nug_single, nug_single) - nug_single
            
            # Weighted error (emphasize short and medium lags)
            weights = np.exp(-x / (np.max(x) * 0.7))
            error = np.sum(weights * (y - pred) ** 2)
            
            if error < best_fit["error"]:
                best_fit = {
                    "error": error,
                    "nugget": nug_single,
                    "c1": c1,
                    "c2": c2,
                    "r1": r1,
                    "r2": r2,
                }
    
    # Build result
    structures = [
        {"type": model_type1, "contribution": best_fit["c1"], "range": best_fit["r1"]},
        {"type": model_type2, "contribution": best_fit["c2"], "range": best_fit["r2"]},
    ]
    
    # Add third structure if requested
    if n_structures >= 3:
        # Third structure: very long range, small contribution
        c3 = best_fit["c2"] * 0.3
        structures[1]["contribution"] = best_fit["c2"] * 0.7
        r3 = best_fit["r2"] * 2.0
        structures.append({
            "type": "exponential",
            "contribution": c3,
            "range": r3,
        })
    
    total_sill = best_fit["nugget"] + sum(s["contribution"] for s in structures)
    
    struct_info = [(s['type'], round(s['contribution'], 3), round(s['range'], 1)) for s in structures]
    logger.info(
        f"Nested variogram fit: nugget={best_fit['nugget']:.3f}, "
        f"structures: {struct_info}"
    )
    
    return {
        "nugget": best_fit["nugget"],
        "total_sill": total_sill,
        "structures": structures,
    }


# ----------------------------------------------------------------------
# Pair helpers
# ----------------------------------------------------------------------
def _sorted_pairs_array(pairs_set: set) -> np.ndarray:
    """
    Convert cKDTree.query_pairs() set to deterministically-ordered array.

    KD-tree query_pairs() returns a set with undefined iteration order.
    This function ensures consistent ordering by sorting lexicographically,
    which is essential for reproducible variogram calculations.

    Parameters
    ----------
    pairs_set : set
        Set of (i, j) tuples from cKDTree.query_pairs()

    Returns
    -------
    np.ndarray
        Array of shape (N, 2) with pairs sorted by (i, then j) for determinism
    """
    if not pairs_set:
        return np.empty((0, 2), dtype=int)

    # Convert to array and sort lexicographically for deterministic ordering
    pairs_arr = np.array(list(pairs_set), dtype=int)
    # Sort by first column, then by second column for ties
    sort_idx = np.lexsort((pairs_arr[:, 1], pairs_arr[:, 0]))
    return pairs_arr[sort_idx]


def calculate_pair_attributes(
    coords: np.ndarray,
    values: np.ndarray,
    indices_i: np.ndarray,
    indices_j: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute distances, semivariances, and connecting vectors for the selected pairs.
    """
    ci = coords[indices_i]
    cj = coords[indices_j]
    vec = cj - ci
    dists = np.linalg.norm(vec, axis=1)
    vi = values[indices_i]
    vj = values[indices_j]
    gammas = 0.5 * (vi - vj) ** 2
    return dists, gammas, vec


# ----------------------------------------------------------------------
# Experimental variogram (omni) with KDTree pairing
# ----------------------------------------------------------------------
def _pairwise_variogram(
    values: np.ndarray,
    coords: np.ndarray,
    max_pairs: Optional[int] = None,
    max_dist: Optional[float] = None,
    max_samples: int = 2000,
    random_state: Optional[int] = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute pairwise distances and semivariances. Uses cKDTree to limit pairs
    within max_dist and optionally subsamples uniformly to max_pairs.

    IMPORTANT: For large datasets, we subsample BEFORE computing pairs to avoid
    O(N²) memory/time explosion. query_pairs with large N is the bottleneck.

    DETERMINISM: This function is fully deterministic when random_state is set.
    All random operations use the same seeded RNG, and KD-tree pair ordering
    is made consistent via lexicographic sorting.

    Parameters
    ----------
    values : array
        Data values
    coords : array
        Coordinate array (N x 3)
    max_pairs : int, optional
        Maximum number of pairs to return
    max_dist : float, optional
        Maximum distance for pairs (defaults to 50% of data extent to keep reasonable)
    max_samples : int
        Maximum number of samples to use before subsampling (default 2000)
        This prevents O(N²) explosion for large datasets.
    random_state : int, optional
        Random seed for reproducibility (default 42). All subsampling operations
        use this seed to ensure deterministic results.
    """
    coords = np.asarray(coords, float)
    values = np.asarray(values, float)
    n = coords.shape[0]
    if n < 2:
        return np.array([]), np.array([])

    # Create seeded RNG for all random operations - ensures determinism
    rng = np.random.default_rng(random_state)

    # CRITICAL: Subsample data BEFORE computing pairs to avoid O(N²) explosion
    # For 50k samples, N² = 2.5 billion pairs - this freezes the app!
    if n > max_samples:
        idx = rng.choice(n, size=max_samples, replace=False)
        coords = coords[idx]
        values = values[idx]
        n = max_samples
        logger.debug(f"Subsampled to {n} points for variogram cloud (seed={random_state})")

    # Limit max_dist to reasonable fraction of data extent
    # Using full extent creates too many pairs
    extent = np.linalg.norm(coords.max(axis=0) - coords.min(axis=0))
    if max_dist is None:
        max_dist = extent * 0.5  # Use 50% of extent, not full extent
    else:
        max_dist = min(max_dist, extent)  # Cap at extent

    if cKDTree is not None:
        tree = cKDTree(coords)
        pairs = tree.query_pairs(r=max_dist)
        # Use _sorted_pairs_array for deterministic ordering
        pairs_arr = _sorted_pairs_array(pairs)
    else:
        # Fallback dense pairing (only for small N after subsampling)
        idx_i, idx_j = np.triu_indices(n, k=1)
        dists = np.linalg.norm(coords[idx_i] - coords[idx_j], axis=1)
        mask = dists <= max_dist
        pairs_arr = np.vstack((idx_i[mask], idx_j[mask])).T

    # Further subsample pairs if needed - use same seeded RNG
    if max_pairs and len(pairs_arr) > max_pairs:
        idx = rng.choice(len(pairs_arr), size=max_pairs, replace=False)
        pairs_arr = pairs_arr[idx]

    if len(pairs_arr) == 0:
        return np.array([]), np.array([])

    dists, semis, _ = calculate_pair_attributes(coords, values, pairs_arr[:, 0], pairs_arr[:, 1])
    return dists, semis


def calculate_experimental_variogram(
    coords: np.ndarray,
    values: np.ndarray,
    n_lags: int,
    max_range: float,
    pair_cap: Optional[int] = None,
    random_state: Optional[int] = 42
) -> np.ndarray:
    """
    Compute omnidirectional experimental variogram with equal-width lags up to max_range.

    Parameters
    ----------
    coords : array
        Coordinate array (N x 3)
    values : array
        Data values
    n_lags : int
        Number of lag bins
    max_range : float
        Maximum lag distance
    pair_cap : int, optional
        Maximum number of pairs to use
    random_state : int, optional
        Random seed for reproducibility (default 42)

    Returns
    -------
    np.ndarray
        Array of (distance, gamma, npairs) for each lag bin
    """
    coords = np.asarray(coords, float)
    values = np.asarray(values, float)
    dists, semivars = _pairwise_variogram(
        values, coords, max_pairs=pair_cap, max_dist=max_range, random_state=random_state
    )
    if dists.size == 0:
        return np.array([], dtype=float)

    edges = np.linspace(0.0, max_range, n_lags + 1)
    bins = np.digitize(dists, edges) - 1
    valid = (bins >= 0) & (bins < n_lags)
    if not np.any(valid):
        return np.array([], dtype=float)

    df = (
        np.vstack((bins[valid], dists[valid], semivars[valid]))
        .T
    )
    # Aggregate manually to avoid pandas dependency here
    out = []
    for b in range(n_lags):
        mask = df[:, 0] == b
        if not np.any(mask):
            continue
        dist_mean = float(np.mean(df[mask, 1]))
        gamma_mean = float(np.mean(df[mask, 2]))
        npairs = int(np.sum(mask))
        out.append((dist_mean, gamma_mean, npairs))
    return np.array(out, dtype=float)
