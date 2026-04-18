"""
Auto-Parameter Detection for Implicit Geological Modelling
==========================================================

Computes interpolation parameters automatically from the geometry of
contact data, eliminating the need for manual parameter tuning in most
workflows.  The resulting dict is compatible with GeologicalModelBuilder.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def auto_compute_parameters(
    contacts_df,
    orientations_df=None,
) -> Dict[str, Any]:
    """Compute interpolation parameters from data geometry.

    Parameters
    ----------
    contacts_df : pd.DataFrame
        Contact points with X, Y, Z columns.
    orientations_df : pd.DataFrame, optional
        Structural measurements with dip, azimuth (and optionally
        feature_type) columns.

    Returns
    -------
    dict
        Parameter dict compatible with GeologicalModelBuilder:
        kernel_type, alpha, range_max, range_mid, range_min,
        azimuth, dip, pitch, nugget, accuracy, drift_type,
        constraint_method.
    """
    try:
        from scipy.spatial import cKDTree
        HAS_SCIPY = True
    except ImportError:
        HAS_SCIPY = False

    # ── Locate X, Y, Z columns ────────────────────────────────────
    if contacts_df is None or contacts_df.empty:
        return _fallback_params()

    col_map = {c.lower(): c for c in contacts_df.columns}
    x_col = col_map.get("x") or col_map.get("easting")
    y_col = col_map.get("y") or col_map.get("northing")
    z_col = col_map.get("z") or col_map.get("elevation")

    if not all([x_col, y_col, z_col]):
        return _fallback_params()

    try:
        coords = contacts_df[[x_col, y_col, z_col]].dropna().values.astype(np.float64)
    except Exception:
        return _fallback_params()

    if len(coords) < 2:
        return _fallback_params()

    # ── Range: 3x nearest-neighbour distance between DRILLHOLE COLLARS ──
    #
    # BUG FIX: using all contact points gives within-hole spacing (~2m for
    # contacts along the same hole) which is far too small.  The correct
    # reference length is the spacing *between* holes (~80m on a typical grid).
    # We approximate collar positions as the shallowest (max-Z) contact per
    # hole, then run KNN on those collar coordinates only.
    hole_id_col = None
    for candidate in ("hole_id", "holeid", "bhid", "drillhole", "hole"):
        if candidate in col_map:
            hole_id_col = col_map[candidate]
            break

    spacing_source = "contact points (no hole_id column)"
    if HAS_SCIPY and hole_id_col and contacts_df[hole_id_col].nunique() >= 2:
        try:
            # One representative point per hole: shallowest contact (highest Z)
            z_col_name = z_col  # already resolved above
            collar_df = (
                contacts_df[[hole_id_col, x_col, y_col, z_col_name]]
                .dropna()
                .sort_values(z_col_name, ascending=False)
                .groupby(hole_id_col, sort=False)
                .first()
                .reset_index()
            )
            collar_coords = collar_df[[x_col, y_col, z_col_name]].values.astype(np.float64)

            if len(collar_coords) >= 2:
                k = min(2, len(collar_coords))
                tree = cKDTree(collar_coords)
                nn_dists, _ = tree.query(collar_coords, k=k)
                avg_spacing = float(np.median(nn_dists[:, k - 1]))
                range_max = max(50.0, avg_spacing * 3.0)
                spacing_source = f"collar spacing ({len(collar_coords)} holes)"
            else:
                range_max = _estimate_range_simple(coords)
        except Exception:
            range_max = _estimate_range_simple(coords)
    elif HAS_SCIPY and len(coords) >= 3:
        try:
            tree = cKDTree(coords)
            nn_dists, _ = tree.query(coords, k=2)
            avg_spacing = float(np.median(nn_dists[:, 1]))
            range_max = max(50.0, avg_spacing * 3.0)
        except Exception:
            range_max = _estimate_range_simple(coords)
    else:
        range_max = _estimate_range_simple(coords)

    logger.info(
        "auto_compute_parameters: %d contacts, spacing from %s → range_max=%.1f m",
        len(coords), spacing_source, range_max,
    )

    # ── Anisotropy defaults: ISOTROPIC ────────────────────────────
    #
    # REMOVED: PCA of contact coordinates.
    # PCA measures the drilling PATTERN geometry, not rock geometry.
    # Example: a north-south drill line over flat coal produces a
    # "vertical" ellipsoid, which is the opposite of the true geology.
    #
    # When no structural measurements are available, default to isotropic
    # (equal ranges in all directions).  The user can always override
    # azimuth/dip/ranges manually in the Advanced Settings.
    azimuth = 0.0
    dip = 0.0
    range_mid = range_max
    range_min = range_max

    # ── Override azimuth/dip from structural orientations ─────────
    if orientations_df is not None and not orientations_df.empty:
        try:
            odf = orientations_df
            o_col_map = {c.lower(): c for c in odf.columns}
            dip_col = o_col_map.get("dip")
            az_col = o_col_map.get("azimuth") or o_col_map.get("dip_direction")
            ft_col = o_col_map.get("feature_type") or o_col_map.get("type")

            # Filter to bedding only if feature_type is available
            bedding_df = odf
            if ft_col:
                mask = odf[ft_col].astype(str).str.lower().isin(
                    {"bedding", "stratigraphy", "stratiform", "s0"}
                )
                if mask.sum() >= 3:
                    bedding_df = odf[mask]

            if dip_col and az_col and len(bedding_df) >= 3:
                mean_dip = float(bedding_df[dip_col].dropna().mean())
                mean_az = float(bedding_df[az_col].dropna().mean())
                azimuth = mean_az % 360
                dip = mean_dip
                logger.info(
                    "Structural override: azimuth=%.1f°, dip=%.1f° "
                    "(from %d bedding measurements)",
                    azimuth, dip, len(bedding_df),
                )
        except Exception as exc:
            logger.debug("Structural override failed: %s", exc)

    params = {
        "kernel_type": "spheroidal",
        "alpha": 1.0,
        "range_max": round(range_max, 1),
        "range_mid": round(range_mid, 1),
        "range_min": round(range_min, 1),
        "azimuth": round(azimuth, 1),
        "dip": round(dip, 1),
        "pitch": 0.0,
        "nugget": 0.0,
        "accuracy": 1e-8,
        "drift_type": "constant",
        "constraint_method": "gradient",
    }

    logger.info(
        "Auto parameters: range=[%.0f, %.0f, %.0f] m, az=%.1f°, dip=%.1f°",
        params["range_max"], params["range_mid"], params["range_min"],
        params["azimuth"], params["dip"],
    )
    return params


def _estimate_range_simple(coords: np.ndarray) -> float:
    """Fallback range estimate without scipy: use bounding box diagonal / 3."""
    span = coords.max(axis=0) - coords.min(axis=0)
    diagonal = float(np.linalg.norm(span))
    return max(50.0, diagonal / 3.0)


def _fallback_params() -> Dict[str, Any]:
    """Return safe defaults when auto-detection is not possible."""
    return {
        "kernel_type": "spheroidal",
        "alpha": 1.0,
        "range_max": 250.0,
        "range_mid": 250.0,   # isotropic fallback — user overrides as needed
        "range_min": 250.0,   # isotropic fallback
        "azimuth": 0.0,
        "dip": 0.0,
        "pitch": 0.0,
        "nugget": 0.0,
        "accuracy": 1e-8,
        "drift_type": "constant",
        "constraint_method": "gradient",
    }
