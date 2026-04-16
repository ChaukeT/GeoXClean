"""Auto-detection of horizontal variogram orientation.

Lifted from ``block_model_viewer/models/variogram3d.py`` as part of the
variogram engine consolidation (Option A). Self-contained: imports only
numpy / scipy / pandas / logging. Exposes two public functions:

- :func:`estimate_default_orientation` — headline entry point; returns
  an ``(azimuth_degrees, dip_degrees, metadata)`` tuple.
- :func:`variogram_map_orientation` — the directional-semivariance scan
  used by ``estimate_default_orientation`` when values are available.

Both routines also accept ``hole_ids`` so that dense along-hole sample
clustering doesn't bias the horizontal support cloud.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.spatial import cKDTree
except ImportError:  # pragma: no cover
    cKDTree = None

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Horizontal support collapse
# ---------------------------------------------------------------------------

def collapse_horizontal_support(
    coords: np.ndarray,
    hole_ids: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Collapse dense along-hole samples to a horizontal support cloud.

    Variogram direction auto-selection should not be dominated by
    along-hole sample clustering. Prefer collar-level XY support when
    hole IDs exist; otherwise fall back to rounded unique XY sample
    positions.
    """
    xy = np.asarray(coords, float)[:, :2]
    if xy.shape[0] == 0:
        return xy

    if hole_ids is not None and len(hole_ids) == len(xy):
        try:
            df = pd.DataFrame({"hole": hole_ids, "x": xy[:, 0], "y": xy[:, 1]})
            collars = df.groupby("hole", sort=True)[["x", "y"]].median().to_numpy(float)
            if len(collars) >= 2:
                return collars
        except Exception:
            pass

    xy_rounded = np.round(xy, 0)
    unique_xy = np.unique(xy_rounded, axis=0)
    return unique_xy if len(unique_xy) >= 2 else xy


# Legacy private-name alias.
_collapse_horizontal_support = collapse_horizontal_support


# ---------------------------------------------------------------------------
# Variogram-map azimuth scan
# ---------------------------------------------------------------------------

def variogram_map_orientation(
    coords: np.ndarray,
    values: np.ndarray,
    hole_ids: Optional[np.ndarray] = None,
) -> Tuple[float, Dict[str, Any]]:
    """Estimate the major horizontal continuity direction from a
    directional semivariance scan.

    Computes semivariance at 10° azimuth increments using a coarse lag
    (half the median drillhole spacing) over the first six lags, and
    returns the azimuth of MINIMUM mean semivariance — the direction of
    maximum spatial continuity.

    Raises ``ValueError`` when the scan can't resolve a direction
    (too few pairs, flat map, etc.).
    """
    if cKDTree is None:  # pragma: no cover
        raise ValueError("scipy.spatial.cKDTree is required for variogram map")

    coords = np.asarray(coords, float)
    values = np.asarray(values, float)

    # Horizontal projection for the azimuth scan
    xy = coords[:, :2]
    n = len(xy)

    # Estimate lag from collar spacing
    support_xy = collapse_horizontal_support(coords, hole_ids=hole_ids)
    if len(support_xy) >= 2:
        tree_supp = cKDTree(support_xy)
        dd, _ = tree_supp.query(support_xy, k=min(3, len(support_xy)))
        drill_spacing = float(
            np.median(dd[:, 1] if dd.ndim == 2 else dd[1])
        )
        lag = drill_spacing * 0.5
    else:
        lag = float(np.max(xy[:, 0]) - np.min(xy[:, 0])) / 20.0

    lag = max(lag, 1.0)
    max_range = lag * 6  # scan the first 6 lags
    cone_tol = 22.5       # wider tolerance for a robust scan

    # Subsample for speed (max 2000 points)
    rng = np.random.RandomState(42)
    if n > 2000:
        idx = rng.choice(n, 2000, replace=False)
        xy_sub = xy[idx]
        vals_sub = values[idx]
    else:
        xy_sub = xy
        vals_sub = values

    tree = cKDTree(xy_sub)
    pairs = tree.query_pairs(r=max_range, output_type="ndarray")
    if len(pairs) < 50:
        raise ValueError(
            f"Too few pairs ({len(pairs)}) for variogram map orientation"
        )

    dx = xy_sub[pairs[:, 1], 0] - xy_sub[pairs[:, 0], 0]
    dy = xy_sub[pairs[:, 1], 1] - xy_sub[pairs[:, 0], 1]
    dz_val = vals_sub[pairs[:, 1]] - vals_sub[pairs[:, 0]]
    pair_gamma = 0.5 * dz_val ** 2
    pair_dist = np.sqrt(dx ** 2 + dy ** 2)
    pair_azimuth = np.rad2deg(np.arctan2(dx, dy)) % 360.0

    dist_mask = (pair_dist >= lag * 0.5) & (pair_dist <= max_range)

    test_azimuths = np.arange(0, 180, 10)  # 0–170° (symmetric)
    mean_gammas = np.full(len(test_azimuths), np.inf)

    for i, az in enumerate(test_azimuths):
        ang_diff = np.abs(pair_azimuth - az)
        ang_diff = np.minimum(ang_diff, 360.0 - ang_diff)
        ang_diff = np.minimum(ang_diff, np.abs(ang_diff - 180.0))
        az_mask = ang_diff <= cone_tol
        combined = dist_mask & az_mask
        if int(np.sum(combined)) >= 10:
            mean_gammas[i] = float(np.mean(pair_gamma[combined]))

    valid = np.isfinite(mean_gammas)
    if not np.any(valid):
        raise ValueError("No valid azimuths in variogram map scan")

    best_idx = int(np.argmin(mean_gammas[valid]))
    valid_indices = np.where(valid)[0]
    best_azimuth = float(test_azimuths[valid_indices[best_idx]])

    gamma_min = float(np.min(mean_gammas[valid]))
    gamma_max = float(np.max(mean_gammas[valid]))
    contrast = gamma_max / max(gamma_min, 1e-12)

    metadata: Dict[str, Any] = {
        "vmap_azimuth": best_azimuth,
        "vmap_contrast_ratio": contrast,
        "vmap_n_pairs": int(np.sum(dist_mask)),
        "vmap_n_test_azimuths": len(test_azimuths),
        "vmap_lag": lag,
        "vmap_max_range": max_range,
    }

    if contrast < 1.1:
        logger.info(
            "Variogram map shows weak directional contrast (ratio=%.2f). "
            "Data may be near-isotropic; PCA fallback may be equally valid.",
            contrast,
        )
        metadata["vmap_weak_contrast"] = True

    logger.info(
        "Variogram map orientation: azimuth=%.1f° (max continuity), "
        "contrast ratio=%.2f, %d pairs",
        best_azimuth, contrast, int(np.sum(dist_mask)),
    )

    return best_azimuth, metadata


# Legacy private-name alias.
_variogram_map_orientation = variogram_map_orientation


# ---------------------------------------------------------------------------
# Headline orientation estimator
# ---------------------------------------------------------------------------

def estimate_default_orientation(
    coords: np.ndarray,
    hole_ids: Optional[np.ndarray] = None,
    values: Optional[np.ndarray] = None,
) -> Tuple[float, float, Dict[str, Any]]:
    """Estimate a defensible default horizontal orientation from data.

    Uses two methods and selects the more defensible:

    1. **Variogram map** (preferred when values are provided):
       directional semivariance scan; picks direction of MINIMUM
       semivariance (maximum continuity). Reflects spatial continuity
       of the variable, not drilling geometry.

    2. **PCA of collar positions** (fallback):
       principal axis of the horizontal support cloud. Reflects
       drilling pattern geometry, which may differ from geological
       continuity.

    .. warning::
        The auto-derived orientation is labelled
        ``requires_user_confirmation`` in the metadata. A Competent
        Person must verify it against geological knowledge before
        using it in resource estimation.
    """
    support_xy = collapse_horizontal_support(coords, hole_ids=hole_ids)

    vmap_az: Optional[float] = None
    vmap_meta: Dict[str, Any] = {}
    if values is not None and len(values) == len(coords) and len(coords) >= 30:
        try:
            vmap_az, vmap_meta = variogram_map_orientation(
                coords, values, hole_ids
            )
        except Exception as e:
            logger.debug("Variogram map orientation failed: %s", e)

    pca_az: Optional[float] = None
    pca_meta: Dict[str, Any] = {}
    if support_xy.shape[0] >= 2:
        centered = support_xy - np.mean(support_xy, axis=0, keepdims=True)
        cov = np.cov(centered, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        idx = int(np.argmax(eigvals))
        principal = eigvecs[:, idx]
        if np.all(np.isfinite(principal)) and np.linalg.norm(principal) > 1e-12:
            pca_az = float(
                np.rad2deg(np.arctan2(principal[0], principal[1])) % 360.0
            )
            eigenvalues = np.sort(np.maximum(eigvals, 0.0))[::-1]
            anisotropy_ratio = (
                float(eigenvalues[0] / max(eigenvalues[1], 1e-12))
                if len(eigenvalues) > 1
                else float("inf")
            )
            pca_meta = {
                "pca_azimuth": pca_az,
                "pca_support_points": int(len(support_xy)),
                "pca_anisotropy_ratio": anisotropy_ratio,
            }

    if vmap_az is not None:
        azimuth = vmap_az
        source = "variogram_map"
        note = (
            "Estimated from directional semivariance scan (direction of "
            "maximum spatial continuity). Reflects the variable's spatial "
            "structure, NOT the drilling pattern. REQUIRES USER "
            "CONFIRMATION against geology."
        )
        metadata: Dict[str, Any] = {
            "orientation_source": source,
            "orientation_note": note,
            "requires_user_confirmation": True,
            "orientation_support_points": int(len(coords)),
            **vmap_meta,
            **pca_meta,
        }
        if pca_az is not None:
            divergence = abs(vmap_az - pca_az)
            divergence = min(divergence, 360.0 - divergence)
            metadata["vmap_pca_divergence_deg"] = float(divergence)
            if divergence > 15.0:
                logger.warning(
                    "Variogram map azimuth (%.1f°) diverges from "
                    "PCA-of-collars (%.1f°) by %.1f°. Geological continuity "
                    "may differ from drilling pattern — variogram map "
                    "preferred, user MUST verify.",
                    vmap_az, pca_az, divergence,
                )
    elif pca_az is not None:
        azimuth = pca_az
        source = "horizontal_pca_support"
        note = (
            "WARNING: Estimated from DRILLING PATTERN geometry (PCA of "
            "collar positions), NOT from spatial continuity of the "
            "variable. If the drill grid is not aligned with geological "
            "strike, this orientation may be WRONG. REQUIRES USER "
            "CONFIRMATION against geology."
        )
        metadata = {
            "orientation_source": source,
            "orientation_note": note,
            "requires_user_confirmation": True,
            "orientation_support_points": int(len(support_xy)),
            "orientation_ratio": pca_meta.get("pca_anisotropy_ratio", 1.0),
            **pca_meta,
        }
    else:
        return 0.0, 0.0, {
            "orientation_source": "fallback_default",
            "orientation_note": (
                "Insufficient data for orientation estimation; using 0 "
                "degrees."
            ),
            "requires_user_confirmation": True,
        }

    return azimuth, 0.0, metadata
