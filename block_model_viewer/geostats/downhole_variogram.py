"""Within-hole (true downhole) variogram.

Pairs samples only from the same drillhole, using along-hole distance.
Lifted from ``block_model_viewer/models/variogram3d.py`` as part of the
variogram engine consolidation (Option A) so both the legacy engine and
the v2 pipeline can share it without a cross-dependency.

Along-hole distance is computed as the cumulative length of 3D segments
between consecutive samples in depth order. When the 3D coordinates are
degenerate (collar-only — hole not desurveyed yet) it falls back to
mid-depth differences and finally to Z-coordinate differences.

This is the CORRECT way to estimate the nugget: inter-hole vertical pair
mixing (which would happen if a v2 directional search were pointed
straight down) biases the short-lag semivariance upward and inflates the
nugget. Use this helper for every "downhole" / "within-hole" variogram.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

from .variogram_lag_aggregation import aggregate_lag_dataframe

logger = logging.getLogger(__name__)


def compute_downhole_variogram(
    coords: np.ndarray,
    values: np.ndarray,
    hole_ids: np.ndarray,
    from_depths: Optional[np.ndarray] = None,
    to_depths: Optional[np.ndarray] = None,
    sample_weights: Optional[np.ndarray] = None,
    n_lags: int = 15,
    lag_tolerance: Optional[float] = None,
    max_range: Optional[float] = None,
) -> pd.DataFrame:
    """Compute the within-hole variogram.

    Parameters
    ----------
    coords : (N, 3) ndarray
        Sample XYZ coordinates (post-desurvey).
    values : (N,) ndarray
        Sample values.
    hole_ids : (N,) ndarray
        Hole ID per sample (string or object).
    from_depths, to_depths : (N,) ndarrays, optional
        Along-hole FROM/TO depths. Used to compute sample length (which
        seeds the downhole lag) and as a fallback distance when 3D
        coordinates are degenerate.
    sample_weights : (N,) ndarray, optional
        Per-sample weights (e.g. declustering). When provided, each pair
        contributes with ``w_i * w_j``.
    n_lags : int, default 15
        Number of lag bins (clamped into [10, 25]).
    lag_tolerance : float, optional
        Half-width of each lag bin. When ``None``, defaults to 50% of
        the downhole lag spacing.
    max_range : float, optional
        Distance cut-off. When ``None``, defaults to ``downhole_lag *
        n_lags`` capped at 15 × composite length.

    Returns
    -------
    DataFrame with columns ``distance, gamma, npairs`` (and
    ``pair_weight_sum`` when weights are supplied). Empty DataFrame when
    no valid within-hole pairs were found.
    """
    coords = np.asarray(coords, dtype=float)
    values = np.asarray(values, dtype=float).ravel()
    hole_ids = np.asarray(hole_ids).ravel()
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must be an (N, 3) array")
    if values.size != coords.shape[0]:
        raise ValueError("values length must match coords")
    if hole_ids.size != coords.shape[0]:
        raise ValueError("hole_ids length must match coords")

    # Composite / sample length (if FROM/TO supplied)
    sample_length: Optional[float] = None
    if from_depths is not None and to_depths is not None:
        sample_lengths = np.abs(
            np.asarray(to_depths, float) - np.asarray(from_depths, float)
        )
        valid_lengths = sample_lengths[
            ~np.isnan(sample_lengths) & (sample_lengths > 0)
        ]
        if len(valid_lengths) > 0:
            sample_length = float(np.nanmedian(valid_lengths))

    downhole_lag = float(sample_length) if sample_length and sample_length > 0 else 2.0

    nl = int(max(10, min(25, int(n_lags))))

    if max_range is None:
        mr = downhole_lag * nl
        if sample_length is not None and sample_length > 0:
            max_reasonable = sample_length * 15
            if mr > max_reasonable:
                nl = min(nl, 15)
                mr = downhole_lag * nl
    else:
        mr = float(max_range)
        if mr <= 0:
            raise ValueError(f"max_range must be > 0, got {mr}")

    lag_tol = (
        float(lag_tolerance)
        if lag_tolerance is not None and lag_tolerance > 0
        else downhole_lag * 0.5
    )

    sample_info = f"{sample_length:.2f}m" if sample_length is not None else "unknown"
    logger.debug(
        "Downhole variogram: lag=%.2fm, n_lags=%d, max_range=%.1fm (composite=%s)",
        downhole_lag, nl, mr, sample_info,
    )

    df = pd.DataFrame(coords, columns=["X", "Y", "Z"])
    df["val"] = values
    df["hole"] = hole_ids

    use_depth_distance = False
    if from_depths is not None and to_depths is not None:
        df["FROM"] = np.asarray(from_depths, float)
        df["TO"] = np.asarray(to_depths, float)
        df["MID"] = 0.5 * (df["FROM"] + df["TO"])
        use_depth_distance = True
    elif from_depths is not None:
        df["MID"] = np.asarray(from_depths, float)
        df["FROM"] = df["MID"]
        use_depth_distance = True
    elif to_depths is not None:
        df["MID"] = np.asarray(to_depths, float)
        df["TO"] = df["MID"]
        use_depth_distance = True
    else:
        df["MID"] = df["Z"].astype(float)
        use_depth_distance = False

    dist_chunks: List[np.ndarray] = []
    semi_chunks: List[np.ndarray] = []
    pw_chunks: List[np.ndarray] = []

    total_pairs = 0
    unique_holes = df["hole"].unique()
    logger.info(
        "Downhole variogram: %d holes, use_depth_distance=%s, max_range=%.1f",
        len(unique_holes), use_depth_distance, mr,
    )
    if use_depth_distance:
        depth_range = df["MID"].max() - df["MID"].min()
        logger.info(
            "Depth range: %.1f to %.1f (span: %.1f)",
            df["MID"].min(), df["MID"].max(), depth_range,
        )
    else:
        z_range = df["Z"].max() - df["Z"].min()
        logger.info(
            "Z range: %.1f to %.1f (span: %.1f)",
            df["Z"].min(), df["Z"].max(), z_range,
        )

    holes_processed = 0
    holes_skipped = 0

    for hole_name in unique_holes:
        group = df[df["hole"] == hole_name]
        if len(group) < 2:
            holes_skipped += 1
            if holes_skipped == 1:
                logger.warning(
                    "Hole '%s' has only %d sample(s) — skipping. May indicate "
                    "missing data or incorrect HOLEID assignment.",
                    hole_name, len(group),
                )
            continue

        g_sorted = group.sort_values("MID")
        hole_orig_idx = g_sorted.index.to_numpy()
        g = g_sorted.reset_index(drop=True)
        g_vals = g["val"].to_numpy(float)
        n = len(g_vals)

        # Primary: 3D cumulative along-hole distance. Correct for
        # desurveyed / deviated holes.
        g_coords = g[["X", "Y", "Z"]].to_numpy(float)
        segment_lengths = np.linalg.norm(np.diff(g_coords, axis=0), axis=1)
        along = np.concatenate([[0.0], np.cumsum(segment_lengths)])

        if np.max(along) < 0.01:
            # 3D coords degenerate (e.g. not yet desurveyed) — fall back
            # to depth-based or Z-based along-hole distance.
            if use_depth_distance and "MID" in g.columns:
                mid_depths = g["MID"].to_numpy(float)
                along = np.abs(mid_depths - mid_depths[0])
                if holes_processed == 0:
                    logger.info(
                        "First hole '%s': %d samples, depth range %.1f to "
                        "%.1f (depth-based fallback)",
                        hole_name, n, mid_depths.min(), mid_depths.max(),
                    )
            else:
                z_vals = g["Z"].to_numpy(float)
                along = np.abs(z_vals - z_vals[0])

            if np.max(along) < 0.01:
                logger.warning(
                    "Hole %s: all %d samples at same location, skipping",
                    hole_name, n,
                )
                holes_skipped += 1
                continue
        else:
            if holes_processed == 0:
                logger.info(
                    "First hole '%s': %d samples, along-hole span %.2fm (3D coords)",
                    hole_name, n, float(np.max(along)),
                )

        holes_processed += 1

        idx_i, idx_j = np.triu_indices(n, k=1)
        dh = np.abs(along[idx_j] - along[idx_i])

        in_range = dh <= mr
        if not np.any(in_range):
            continue

        idx_i = idx_i[in_range]
        idx_j = idx_j[in_range]
        dh = dh[in_range]

        sem = 0.5 * (g_vals[idx_j] - g_vals[idx_i]) ** 2

        if sample_weights is not None:
            hole_sw = np.asarray(sample_weights, float)[hole_orig_idx]
            pw = hole_sw[idx_i] * hole_sw[idx_j]
            pw_chunks.append(pw)

        dist_chunks.append(dh)
        semi_chunks.append(sem)
        total_pairs += len(dh)

    if not dist_chunks:
        logger.warning(
            "No valid downhole pairs found. Processed %d holes, skipped %d",
            holes_processed, holes_skipped,
        )
        if holes_skipped == len(unique_holes):
            logger.warning(
                "ROOT CAUSE: All holes have only 1 sample each. This commonly "
                "occurs after lithology-based compositing when each hole "
                "intersects only one lithological unit. RECOMMENDATION: Use "
                "raw assays (not composites) for downhole variogram to "
                "estimate the nugget effect."
            )
        else:
            logger.warning(
                "Check: 1) HOLEID column detected? 2) FROM/TO columns exist? "
                "3) Multiple samples per hole?"
            )
        return pd.DataFrame({"distance": [], "gamma": [], "npairs": []})

    all_dists = np.concatenate(dist_chunks)
    all_semis = np.concatenate(semi_chunks)
    all_pair_weights = np.concatenate(pw_chunks) if pw_chunks else None

    logger.info(
        "Downhole variogram: %d pairs from %d holes (skipped %d)",
        total_pairs, holes_processed, holes_skipped,
    )
    logger.info(
        "Distance range: %.2f to %.2f m (median: %.2f)",
        float(all_dists.min()), float(all_dists.max()), float(np.median(all_dists)),
    )

    return aggregate_lag_dataframe(
        all_dists,
        all_semis,
        n_lags=nl,
        max_range=mr,
        lag_distance=downhole_lag,
        lag_tolerance=lag_tol,
        pair_weights=(
            all_pair_weights
            if all_pair_weights is not None
            and len(all_pair_weights) == len(all_dists)
            else None
        ),
    )
