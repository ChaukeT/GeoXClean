"""1D variogram fitting utilities.

Standalone curve-fitting helpers that take a lag/gamma vector and
return ``(nugget, total_sill, range)`` for a single structure, or a
dict for nested 2-structure models. Lifted from
``block_model_viewer/models/variogram_functions.py`` as part of the
variogram engine consolidation (Option A) so both the legacy engine
and the v2 pipeline / assistant can share them.

Dependencies: numpy, scipy.optimize, and the canonical model kernels
from ``geostats.variogram_model``. Nothing else in GeoX.

Sill semantics
--------------
:func:`fit_variogram_model` returns the TOTAL sill (``C0 + C``),
following the GSLIB / SGeMS / PyKrige convention. Partial sill is
``total_sill - nugget``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .variogram_model import (
    MODEL_MAP,
    exponential_model,
    spherical_model,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Single-structure fit
# ---------------------------------------------------------------------------

def fit_variogram_model(
    lags: np.ndarray,
    gammas: np.ndarray,
    model_type: str = "spherical",
    weights: Optional[np.ndarray] = None,
    max_lag: Optional[float] = None,
    sill_norm: bool = False,
    sill_cap: Optional[float] = None,
    pair_counts: Optional[np.ndarray] = None,
) -> Tuple[float, float, float]:
    """Fit ``(nugget, total_sill, range)`` using ``scipy.optimize.curve_fit``
    with an industry-standard initial-guess strategy and a grid-search
    fallback when curve_fit fails.

    Returns ``(nugget, total_sill, range)`` — total sill = ``C0 + C``.

    When ``sill_cap`` is supplied (typically the sample variance or the
    omnidirectional sill) the fit is bounded to ``sill_cap * 1.2`` so
    noisy directional variograms can't produce unrealistic sills.
    """
    lags = np.asarray(lags, float)
    gammas = np.asarray(gammas, float)
    mask = np.isfinite(lags) & np.isfinite(gammas) & (lags >= 0) & (gammas >= 0)
    x = lags[mask]
    y = gammas[mask]

    if x.size < 3:
        sill_est = float(np.nanmax(y)) if y.size else 1.0
        if sill_cap is not None and sill_est > sill_cap * 1.2:
            sill_est = sill_cap
        range_est = float(np.nanmax(x) * 0.7) if x.size else 1.0
        return 0.0, sill_est, max(range_est, 1.0)

    func = MODEL_MAP.get(model_type, spherical_model)

    sort_idx = np.argsort(x)
    x = x[sort_idx]
    y = y[sort_idx]

    scale_var = 1.0
    if sill_norm:
        var = float(np.nanmax(y)) if np.isfinite(y).any() else 1.0
        if var > 0:
            y = y / var
            scale_var = var
            if sill_cap is not None:
                sill_cap = sill_cap / var

    y_min = float(np.nanmin(y))
    y_max = float(np.nanmax(y))
    x_max = float(np.nanmax(x))
    x_min = float(np.nanmin(x[x > 0])) if np.any(x > 0) else 1.0

    # Nugget estimation (duplicate pairs / tight early lags / linear extrapolation)
    very_short_mask = x < x_min * 1.5
    if np.sum(very_short_mask) >= 2:
        nugget0 = float(np.mean(y[very_short_mask]))
    elif len(x) >= 3 and x[0] > 0:
        n_pts = min(3, len(x))
        x_early = x[:n_pts]
        y_early = y[:n_pts]
        inv_dist_w = 1.0 / (x_early + 1e-6)
        if pair_counts is not None:
            pc = np.asarray(pair_counts, float)
            if len(pc) == len(lags):
                pc = pc[mask]
            if len(pc) == len(x):
                pc = pc[sort_idx]
            if len(pc) >= n_pts:
                pair_w = np.sqrt(pc[:n_pts] + 1.0)
                inv_dist_w = inv_dist_w * pair_w
        weights_nug = inv_dist_w / inv_dist_w.sum()
        x_mean = np.sum(weights_nug * x_early)
        y_mean = np.sum(weights_nug * y_early)
        slope = np.sum(weights_nug * (x_early - x_mean) * (y_early - y_mean)) / (
            np.sum(weights_nug * (x_early - x_mean) ** 2) + 1e-12
        )
        nugget0 = max(0.0, float(y_mean - slope * x_mean))
    elif len(x) >= 2 and x[0] > 0:
        slope = (y[1] - y[0]) / (x[1] - x[0] + 1e-12)
        nugget0 = max(0.0, float(y[0] - slope * x[0]))
    else:
        nugget0 = max(0.0, y_min * 0.8)

    # Non-zero nugget floor so curve_fit isn't stuck at zero
    first_lag_floor = float(y[0]) * 0.1
    if nugget0 < first_lag_floor and first_lag_floor > 0:
        nugget0 = first_lag_floor

    if len(y) >= 5:
        nugget0 = min(nugget0, y_min * 1.2)

    # Sill from plateau region
    n_outer = max(1, len(y) // 3)
    sill0 = float(np.mean(y[-n_outer:])) if len(y) > 0 else y_max
    sill0 = max(sill0, y_max * 0.8)
    if sill0 <= nugget0 * 1.1:
        sill0 = nugget0 + (y_max - y_min) * 0.5 + 0.1

    # Range from first crossing of 80% of sill
    target_gamma = nugget0 + 0.8 * (sill0 - nugget0)
    range0 = x_max * 0.5
    for i, yi in enumerate(y):
        if yi >= target_gamma:
            range0 = float(x[i])
            break
    range0 = max(range0, x_max * 0.2)

    sigma = None
    if weights is not None:
        w = np.asarray(weights, float)
        if len(w) == len(lags):
            w = w[mask]
        if len(w) == len(x):
            w = w[sort_idx]
            sigma = 1.0 / (w + 1e-6)

    r_lo = max(float(np.nanmin(x[x > 0])) if np.any(x > 0) else 1e-3, 1e-3)
    r_hi = max_lag if max_lag is not None else x_max * 1.1
    r_hi = max(r_hi, r_lo * 2.0)
    range0 = max(r_lo, min(range0, r_hi))

    sill_lo = nugget0 * 1.05 + 0.01
    sill_hi = y_max * 1.3
    if sill_cap is not None:
        sill_hi = sill_cap * 1.2
        if sill0 > sill_cap * 1.2:
            sill0 = sill_cap
    sill_hi = max(sill_hi, sill_lo * 1.1)

    nug_lo = 1e-6
    nug_hi = sill0 * 0.9

    try:
        from scipy.optimize import curve_fit

        p0 = [range0, sill0, nugget0]
        bounds = (
            [r_lo, sill_lo, nug_lo],
            [r_hi, sill_hi, nug_hi],
        )
        popt, _ = curve_fit(
            func, x, y, p0=p0, bounds=bounds, sigma=sigma, maxfev=10000
        )
        rng, sill, nug = popt
        if sill <= nug:
            sill = nug + 0.1 * scale_var
        nug = float(nug) * scale_var
        sill = float(sill) * scale_var
        logger.debug(
            "Fitted %s: nugget=%.4f, sill=%.4f, range=%.2f",
            model_type, nug, sill, rng,
        )
        return nug, sill, float(rng)
    except Exception as exc:
        logger.warning(
            "Variogram curve_fit failed: %s. Falling back to grid search — "
            "verify fitted parameters manually.", exc,
        )
        nug, sill, rng = _fit_variogram_grid(
            x, y, model_type, r_lo, r_hi, sill_cap
        )
        nug *= scale_var
        sill *= scale_var
        logger.warning(
            "Grid-search fallback: nugget=%.4f, sill=%.4f, range=%.1f",
            nug, sill, rng,
        )
        return nug, sill, rng


def _fit_variogram_grid(
    dist: np.ndarray,
    gamma: np.ndarray,
    model: str,
    r_lo: float,
    r_hi: float,
    sill_cap: Optional[float] = None,
) -> Tuple[float, float, float]:
    """Grid-search fallback for :func:`fit_variogram_model`. Used when
    curve_fit raises."""
    dist = np.asarray(dist, float)
    gamma = np.asarray(gamma, float)
    sort_idx = np.argsort(dist)
    dist = dist[sort_idx]
    gamma = gamma[sort_idx]

    if len(gamma) == 0:
        return 0.0, 1.0, (r_lo + r_hi) / 2

    gmax = float(np.nanmax(gamma))
    gmin = float(np.nanmin(gamma))
    dmax = float(np.nanmax(dist))

    n_early = max(1, len(gamma) // 5)
    nugget_est = float(np.mean(gamma[:n_early]))
    n_late = max(1, len(gamma) // 3)
    sill_est = float(np.mean(gamma[-n_late:]))

    if sill_est <= nugget_est:
        sill_est = gmax
        nugget_est = gmin * 0.5

    first_lag_floor = float(gamma[0]) * 0.1
    if nugget_est < first_lag_floor and first_lag_floor > 0:
        nugget_est = first_lag_floor

    nug_lo, nug_hi = 1e-6, min(sill_est * 0.8, gmax * 0.9)
    sill_lo = max(nugget_est * 1.1, gmin + 0.1)
    if sill_cap is not None:
        sill_hi = sill_cap * 1.2
        if sill_est > sill_cap:
            sill_est = sill_cap
    else:
        sill_hi = gmax * 1.3
    sill_hi = max(sill_hi, sill_lo * 1.1)

    rng_lo, rng_hi = max(r_lo, dmax * 0.1), min(r_hi, dmax * 1.2)
    model_fun = MODEL_MAP.get(model, spherical_model)

    weights = np.exp(-dist / (dmax * 0.5))
    w_sum = weights.sum()
    weights = (
        weights / w_sum if w_sum > 0 else np.ones_like(weights) / len(weights)
    )

    ngrid = 15
    nug_candidates = np.linspace(nug_lo, nug_hi, ngrid)
    sill_candidates = np.linspace(sill_lo, sill_hi, ngrid)
    rng_candidates = np.linspace(rng_lo, rng_hi, ngrid)

    best = (np.inf, nugget_est, sill_est, (rng_lo + rng_hi) / 2)
    for n in nug_candidates:
        for s in sill_candidates:
            if s <= n * 1.05:
                continue
            for r in rng_candidates:
                pred = model_fun(dist, r, s, n)
                err = float(np.sum(weights * (gamma - pred) ** 2))
                if err < best[0]:
                    best = (err, n, s, r)

    _, nugget, sill, prange = best
    if sill <= nugget:
        sill = nugget + (gmax - gmin) * 0.5 + 0.1

    logger.debug(
        "Grid search %s: nugget=%.4f, sill=%.4f, range=%.2f",
        model, nugget, sill, prange,
    )
    return float(nugget), float(sill), float(prange)


# ---------------------------------------------------------------------------
# Nested (multi-structure) fit
# ---------------------------------------------------------------------------

def fit_nested_variogram(
    lags: np.ndarray,
    gammas: np.ndarray,
    model_type1: str = "spherical",
    model_type2: str = "exponential",
    n_structures: int = 2,
) -> Dict[str, Any]:
    """Fit a nested variogram model with 2 or 3 structures.

    Industry-standard strategy: fit a single-structure baseline,
    then grid-search over (split_ratio, range_ratio) to place a
    short-range + long-range structure under that envelope.

    Returns a dict with ``nugget``, ``total_sill`` and a
    ``structures`` list of ``{type, contribution, range}`` entries.
    """
    lags = np.asarray(lags, float)
    gammas = np.asarray(gammas, float)
    mask = np.isfinite(lags) & np.isfinite(gammas) & (lags > 0) & (gammas >= 0)
    x = lags[mask]
    y = gammas[mask]

    if len(x) < 5:
        nug, sill, rng = fit_variogram_model(x, y, model_type=model_type1)
        return {
            "nugget": nug,
            "total_sill": sill,
            "structures": [
                {"type": model_type1, "contribution": sill - nug, "range": rng}
            ],
        }

    sort_idx = np.argsort(x)
    x = x[sort_idx]
    y = y[sort_idx]

    nug_single, sill_single, range_single = fit_variogram_model(
        x, y, model_type=model_type1
    )

    model1_func = MODEL_MAP.get(model_type1, spherical_model)
    model2_func = MODEL_MAP.get(model_type2, exponential_model)

    best_fit: Dict[str, Any] = {"error": np.inf}
    total_contrib = sill_single - nug_single

    for split_ratio in [0.2, 0.3, 0.4, 0.5]:
        for range_ratio in [0.15, 0.25, 0.35, 0.5]:
            c1 = total_contrib * split_ratio
            c2 = total_contrib * (1 - split_ratio)
            r1 = range_single * range_ratio
            r2 = range_single

            partial1 = (
                model1_func(x, r1, c1 + nug_single, nug_single) - nug_single
            )
            partial2 = (
                model2_func(x, r2, c2 + nug_single, nug_single) - nug_single
            )
            pred = nug_single + partial1 + partial2

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

    structures = [
        {"type": model_type1, "contribution": best_fit["c1"], "range": best_fit["r1"]},
        {"type": model_type2, "contribution": best_fit["c2"], "range": best_fit["r2"]},
    ]

    if n_structures >= 3:
        c3 = best_fit["c2"] * 0.3
        structures[1]["contribution"] = best_fit["c2"] * 0.7
        r3 = best_fit["r2"] * 2.0
        structures.append(
            {"type": "exponential", "contribution": c3, "range": r3}
        )

    total_sill = best_fit["nugget"] + sum(s["contribution"] for s in structures)

    struct_info = [
        (s["type"], round(s["contribution"], 3), round(s["range"], 1))
        for s in structures
    ]
    logger.info(
        "Nested variogram fit: nugget=%.3f, structures: %s",
        best_fit["nugget"], struct_info,
    )

    return {
        "nugget": best_fit["nugget"],
        "total_sill": total_sill,
        "structures": structures,
    }
