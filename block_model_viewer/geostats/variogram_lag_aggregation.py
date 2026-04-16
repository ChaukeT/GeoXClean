"""Lag-class aggregation helper for experimental variograms.

Leaf module — imports nothing from other GeoX geostats/variogram code.
Lifted from ``block_model_viewer/models/variogram3d.py`` as part of the
variogram engine consolidation (Option A) so that both the legacy engine
and the v2 pipeline can share the same deterministic aggregation logic
without either depending on the other.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def aggregate_lag_dataframe(
    distances: np.ndarray,
    semivariances: np.ndarray,
    n_lags: int,
    max_range: float,
    lag_distance: float,
    lag_tolerance: float,
    pair_weights: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Aggregate pair statistics into lag classes using explicit lag tolerance.

    Parameters
    ----------
    distances, semivariances
        Per-pair distance and semivariance arrays of equal length.
    n_lags
        Number of lag bins to produce. Bin centres are
        ``lag_distance * k`` for ``k = 1..n_lags``.
    max_range
        Upper distance cut-off; pairs beyond this are dropped.
    lag_distance
        Lag bin width / spacing.
    lag_tolerance
        Half-width of each lag bin (pairs within ``|d - centre| <= tol``
        are counted).
    pair_weights
        Optional per-pair weight (e.g. declustering). Weighted means
        are used for both distance and gamma when provided.

    Returns
    -------
    DataFrame with columns ``bin, distance, gamma, npairs`` (and
    ``pair_weight_sum`` when weights are supplied).
    """
    distances = np.asarray(distances, float)
    semivariances = np.asarray(semivariances, float)
    valid = (
        np.isfinite(distances)
        & np.isfinite(semivariances)
        & (distances >= 0.0)
        & (distances <= max_range)
    )
    if pair_weights is not None:
        pair_weights = np.asarray(pair_weights, float)
        valid &= np.isfinite(pair_weights) & (pair_weights > 0.0)

    if not np.any(valid):
        return pd.DataFrame({"distance": [], "gamma": [], "npairs": []})

    d = distances[valid]
    g = semivariances[valid]
    w = pair_weights[valid] if pair_weights is not None else None
    centers = lag_distance * np.arange(1, int(n_lags) + 1, dtype=float)

    rows: List[Dict[str, Any]] = []
    for lag_idx, center in enumerate(centers):
        in_lag = np.abs(d - center) <= lag_tolerance
        if not np.any(in_lag):
            continue

        lag_d = d[in_lag]
        lag_g = g[in_lag]
        raw_pairs = int(np.sum(in_lag))
        if w is not None:
            lag_w = w[in_lag]
            rows.append({
                "bin": lag_idx,
                "distance": float(np.average(lag_d, weights=lag_w)),
                "gamma": float(np.average(lag_g, weights=lag_w)),
                "npairs": raw_pairs,
                "pair_weight_sum": float(np.sum(lag_w)),
            })
        else:
            rows.append({
                "bin": lag_idx,
                "distance": float(np.mean(lag_d)),
                "gamma": float(np.mean(lag_g)),
                "npairs": raw_pairs,
            })

    if not rows:
        return pd.DataFrame({"distance": [], "gamma": [], "npairs": []})

    return pd.DataFrame(rows)


# Backward-compat alias matching the legacy private name so legacy
# internal call sites can re-export without touching the callers.
_aggregate_lag_dataframe = aggregate_lag_dataframe
