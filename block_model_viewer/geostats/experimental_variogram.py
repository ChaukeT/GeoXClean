"""Experimental (omnidirectional) variogram computation.

Leaf module that performs KDTree-based pair generation and lag
aggregation for point data. Lifted from
``block_model_viewer/models/variogram_functions.py`` as part of the
variogram engine consolidation (Option A). Pure numpy / scipy — no
cross-imports into other GeoX variogram modules.

Used by:
- ``models/sgsim3d.py`` for realisation QC (reproduces the conditioning
  variogram on the grid).
- Any ad-hoc caller that needs a vectorised omnidirectional experimental
  variogram without spinning up the full v2 pipeline.

The heavy v2 pipeline with direction/transform/drift lives in
``variogram_bridge_v2.run_variogram_pipeline_v2``; this module is the
lightweight path for "just give me the lag points".
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

try:
    from scipy.spatial import cKDTree
except ImportError:  # pragma: no cover
    cKDTree = None

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Deterministic pair ordering
# ---------------------------------------------------------------------------

def _sorted_pairs_array(pairs_set: set) -> np.ndarray:
    """Convert ``cKDTree.query_pairs()`` set to a deterministically-ordered
    ``(N, 2)`` array sorted lexicographically by ``(i, j)``.

    ``cKDTree.query_pairs`` returns a ``set`` whose iteration order is
    undefined — necessary to normalise so variogram outputs are
    reproducible across runs.
    """
    if not pairs_set:
        return np.empty((0, 2), dtype=int)

    pairs_arr = np.array(list(pairs_set), dtype=int)
    sort_idx = np.lexsort((pairs_arr[:, 1], pairs_arr[:, 0]))
    return pairs_arr[sort_idx]


# ---------------------------------------------------------------------------
# Pair-attribute computation
# ---------------------------------------------------------------------------

def calculate_pair_attributes(
    coords: np.ndarray,
    values: np.ndarray,
    indices_i: np.ndarray,
    indices_j: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return distances, semivariances and connecting vectors for the
    selected pairs.

    Semivariance is ``0.5 * (v_i - v_j)^2``.
    """
    ci = coords[indices_i]
    cj = coords[indices_j]
    vec = cj - ci
    dists = np.linalg.norm(vec, axis=1)
    vi = values[indices_i]
    vj = values[indices_j]
    gammas = 0.5 * (vi - vj) ** 2
    return dists, gammas, vec


# ---------------------------------------------------------------------------
# Pair generation (omnidirectional)
# ---------------------------------------------------------------------------

def pairwise_variogram(
    values: np.ndarray,
    coords: np.ndarray,
    max_pairs: Optional[int] = None,
    max_dist: Optional[float] = None,
    max_samples: int = 2000,
    random_state: Optional[int] = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute pairwise distances and semivariances.

    Uses ``cKDTree`` to limit pairs within ``max_dist`` and optionally
    sub-samples uniformly to ``max_pairs``. For large datasets we
    sub-sample BEFORE computing pairs to avoid the O(N²) explosion of
    ``query_pairs`` on tens of thousands of points.

    Determinism
    -----------
    Fully deterministic when ``random_state`` is supplied: all random
    operations use the seeded RNG, and KD-tree pair ordering is
    normalised via lexicographic sort.
    """
    coords = np.asarray(coords, float)
    values = np.asarray(values, float)
    n = coords.shape[0]
    if n < 2:
        return np.array([]), np.array([])

    rng = np.random.default_rng(random_state)

    if n > max_samples:
        idx = rng.choice(n, size=max_samples, replace=False)
        coords = coords[idx]
        values = values[idx]
        n = max_samples
        logger.debug(
            "experimental_variogram: subsampled to %d points (seed=%s)",
            n, random_state,
        )

    extent = np.linalg.norm(coords.max(axis=0) - coords.min(axis=0))
    if max_dist is None:
        max_dist = extent * 0.5
    else:
        max_dist = min(max_dist, extent)

    if cKDTree is not None:
        tree = cKDTree(coords)
        pairs = tree.query_pairs(r=max_dist)
        pairs_arr = _sorted_pairs_array(pairs)
    else:
        idx_i, idx_j = np.triu_indices(n, k=1)
        dists_all = np.linalg.norm(coords[idx_i] - coords[idx_j], axis=1)
        mask = dists_all <= max_dist
        pairs_arr = np.vstack((idx_i[mask], idx_j[mask])).T

    if max_pairs and len(pairs_arr) > max_pairs:
        idx = rng.choice(len(pairs_arr), size=max_pairs, replace=False)
        pairs_arr = pairs_arr[idx]

    if len(pairs_arr) == 0:
        return np.array([]), np.array([])

    dists, semis, _ = calculate_pair_attributes(
        coords, values, pairs_arr[:, 0], pairs_arr[:, 1]
    )
    return dists, semis


# Internal alias kept for callers that previously imported
# ``_pairwise_variogram`` from ``variogram_functions``.
_pairwise_variogram = pairwise_variogram


# ---------------------------------------------------------------------------
# Lag aggregation (array form — kept separate from the DataFrame version
# in ``variogram_lag_aggregation`` because several callers expect an
# ndarray of ``(distance, gamma, npairs)`` tuples).
# ---------------------------------------------------------------------------

def aggregate_lag_pairs(
    distances: np.ndarray,
    gammas: np.ndarray,
    n_lags: int,
    max_range: float,
    lag_distance: Optional[float] = None,
    lag_tolerance: Optional[float] = None,
) -> np.ndarray:
    """Aggregate pairwise semivariances into lag classes using explicit
    tolerance. Returns an ``(n, 3)`` array of
    ``(mean_distance, mean_gamma, npairs)`` rows (one per populated
    lag).
    """
    distances = np.asarray(distances, float)
    gammas = np.asarray(gammas, float)

    valid = (
        np.isfinite(distances)
        & np.isfinite(gammas)
        & (distances >= 0.0)
        & (distances <= max_range)
    )
    if not np.any(valid):
        return np.array([], dtype=float)

    d = distances[valid]
    g = gammas[valid]
    lag_step = (
        float(lag_distance)
        if lag_distance is not None and lag_distance > 0
        else float(max_range) / max(int(n_lags), 1)
    )
    tol = (
        float(lag_tolerance)
        if lag_tolerance is not None and lag_tolerance > 0
        else lag_step * 0.5
    )
    centers = lag_step * np.arange(1, int(n_lags) + 1, dtype=float)

    out = []
    for center in centers:
        in_lag = np.abs(d - center) <= tol
        if not np.any(in_lag):
            continue
        out.append((
            float(np.mean(d[in_lag])),
            float(np.mean(g[in_lag])),
            int(np.sum(in_lag)),
        ))

    return np.array(out, dtype=float)


_aggregate_lag_pairs = aggregate_lag_pairs  # legacy private-name alias


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def calculate_experimental_variogram(
    coords: np.ndarray,
    values: np.ndarray,
    n_lags: int,
    max_range: float,
    lag_distance: Optional[float] = None,
    lag_tolerance: Optional[float] = None,
    pair_cap: Optional[int] = None,
    max_samples: int = 2000,
    random_state: Optional[int] = 42,
) -> np.ndarray:
    """Compute an omnidirectional experimental variogram with
    equal-width lags up to ``max_range``.

    Returns an ``(n, 3)`` array of ``(distance, gamma, npairs)`` rows
    (same shape as :func:`aggregate_lag_pairs`).
    """
    coords = np.asarray(coords, float)
    values = np.asarray(values, float)
    dists, semivars = pairwise_variogram(
        values, coords,
        max_pairs=pair_cap, max_dist=max_range,
        max_samples=max_samples, random_state=random_state,
    )
    if dists.size == 0:
        return np.array([], dtype=float)

    return aggregate_lag_pairs(
        dists, semivars,
        n_lags=n_lags,
        max_range=max_range,
        lag_distance=lag_distance,
        lag_tolerance=lag_tolerance,
    )


def plot_variogram_cloud(
    data,
    xcol: str = "X",
    ycol: str = "Y",
    zcol: str = "Z",
    vcol: str = "Fe",
    max_pairs: int = 2000,
    max_dist: Optional[float] = None,
    max_samples: int = 2000,
    random_state: Optional[int] = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate variogram-cloud (per-pair ``distance`` + ``gamma``) arrays.

    Thin convenience wrapper over :func:`pairwise_variogram` for UI
    panels that want to plot the raw cloud before aggregation. Drops
    rows with NaN coordinates/values.
    """
    required_cols = [xcol, ycol, zcol, vcol]
    clean_data = data[required_cols].dropna()
    if clean_data.empty:
        raise ValueError(
            "All data contains NaN values. Cannot generate variogram cloud."
        )
    coords = clean_data[[xcol, ycol, zcol]].to_numpy(float)
    values = clean_data[vcol].to_numpy(float)
    return pairwise_variogram(
        values, coords,
        max_pairs=max_pairs,
        max_dist=max_dist,
        max_samples=max_samples,
        random_state=random_state,
    )


def calculate_experimental_variogram_from_points(
    coordinates: np.ndarray,
    values: np.ndarray,
    n_lags: int = 15,
    lag_tolerance: float = 0.5,
    lag_distance: Optional[float] = None,
    normalize: bool = False,
    max_samples: int = 5000,
    random_state: Optional[int] = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised experimental variogram for point data.

    Returns a tuple ``(lag_distances, semivariances, pair_counts)``.
    """
    coordinates = np.asarray(coordinates, dtype=float)
    values = np.asarray(values, dtype=float)

    if lag_distance is not None and lag_distance > 0:
        max_range = lag_distance * n_lags
    else:
        extent = np.linalg.norm(
            coordinates.max(axis=0) - coordinates.min(axis=0)
        )
        max_range = extent * 0.5

    result = calculate_experimental_variogram(
        coordinates, values,
        n_lags=n_lags,
        max_range=max_range,
        lag_distance=lag_distance,
        lag_tolerance=lag_tolerance,
        max_samples=max_samples,
        random_state=random_state,
    )

    if result.size == 0:
        empty = np.array([], dtype=float)
        return empty, empty, np.array([], dtype=int)

    lag_dists = result[:, 0]
    semivars = result[:, 1]
    pair_counts = result[:, 2].astype(int)

    if normalize:
        data_var = np.var(values)
        if data_var > 0:
            semivars = semivars / data_var

    return lag_dists, semivars, pair_counts
