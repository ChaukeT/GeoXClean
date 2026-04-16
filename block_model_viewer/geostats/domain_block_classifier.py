"""
Domain Block Classifier
=======================
Given an Indicator RBF domain stored in the registry, classify
block centroids as Inside / Outside using the isosurface mesh
or the probability field.

Provides two operations every estimation method needs:

1. ``get_domain_mask(centroids, registry)``
   → boolean mask: True = block is inside the domain

2. ``scatter_to_full_grid(values, mask, n_total, fill=np.nan)``
   → expand a reduced array back to the full grid

Usage in any estimation controller::

    from ..geostats.domain_block_classifier import get_domain_mask, scatter_to_full_grid

    # Before estimation: filter centroids
    domain_mask = get_domain_mask(all_centroids, registry)
    if domain_mask is not None:
        active = all_centroids[domain_mask]
    else:
        active = all_centroids
        domain_mask = np.ones(len(all_centroids), dtype=bool)

    # Run engine on active blocks only
    estimates = engine.run(active, ...)

    # After estimation: scatter back to full grid
    full_estimates = scatter_to_full_grid(estimates, domain_mask, len(all_centroids))
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def get_domain_mask(
    block_centroids: np.ndarray,
    registry,
    domain_value: str = "Inside",
    iso_value: float = 0.5,
) -> Optional[np.ndarray]:
    """Return a boolean mask (True = block belongs to *domain_value*).

    Tries, in order:
      1. Probability-field interpolation at block centroids (PRIMARY —
         numerically robust, no mesh topology dependency)
      2. Point-in-mesh test against the IRBF isosurface (FALLBACK —
         requires watertight mesh, can be unreliable at grid boundaries)

    Returns None if no Indicator RBF domain exists in the registry.
    """
    irbf = _get_irbf_data(registry)
    if irbf is None:
        return None

    iso_val = irbf.get("iso_value", iso_value)
    inside_mask = None
    _method = "none"

    # --- Method 1 (PRIMARY): probability field interpolation ---
    prob = irbf.get("probability_field")
    x = irbf.get("x", irbf.get("x_coords"))
    y = irbf.get("y", irbf.get("y_coords"))
    z = irbf.get("z", irbf.get("z_coords"))
    if prob is not None and x is not None and y is not None and z is not None:
        inside_mask = _prob_inside(block_centroids, prob, x, y, z, iso_val)
        _method = "probability_field"

    # --- Method 2 (FALLBACK): point-in-mesh ---
    if inside_mask is None:
        verts = irbf.get("iso_surface_verts")
        faces = irbf.get("iso_surface_faces")
        if verts is not None and faces is not None and len(verts) > 0:
            inside_mask = _mesh_inside(block_centroids, verts, faces)
            _method = "point_in_mesh"

    if inside_mask is None:
        return None

    if domain_value == "Inside":
        mask = inside_mask
    else:
        mask = ~inside_mask

    n_active = int(mask.sum())
    logger.info(
        "Domain filter: %d / %d blocks in '%s' (%.1f%%) — method: %s",
        n_active, len(mask), domain_value,
        100.0 * n_active / max(len(mask), 1), _method,
    )
    return mask


def scatter_to_full_grid(
    reduced: np.ndarray,
    mask: np.ndarray,
    n_total: int,
    fill: float = np.nan,
) -> np.ndarray:
    """Expand a reduced-length array back to the full grid.

    ``reduced`` has length ``mask.sum()``.
    Returns array of length ``n_total`` with ``fill`` where ``mask`` is False.
    """
    full = np.full(n_total, fill, dtype=reduced.dtype)
    full[mask] = reduced
    return full


def apply_domain_mask_to_grid(
    grid,
    params: dict,
    method_name: str = "estimation",
) -> None:
    """Apply the selected domain to a visualization grid in-place.

    Works with any PyVista grid that exposes ``cell_data`` and
    ``cell_centers()`` (StructuredGrid, ImageData, UnstructuredGrid).

    Behaviour:
      1. NaN-mask all non-domain properties outside the selected domain
      2. Add ``DOMAIN`` (0=outside, 1=inside) for user-facing visualization
      3. Add ``domain_mask`` (0=hidden, 1=visible) for renderer/tooling logic

    No-op when ``params`` does not contain a ``domain_column`` key or when
    no Indicator RBF domain is available in the registry.
    """
    domain_col = params.get("domain_column")
    if not domain_col:
        return
    if grid is None:
        return
    try:
        centroids = np.asarray(grid.cell_centers().points, dtype=float)
        dm = get_domain_mask(
            centroids,
            params.get("_registry"),
            domain_value=params.get("domain_value", "Inside"),
        )
        if dm is None:
            return
        outside = ~dm
        n_outside = int(outside.sum())
        if n_outside > 0:
            for prop_key in list(grid.cell_data.keys()):
                if prop_key in {"DOMAIN", "domain_mask"}:
                    continue  # preserve user-facing labels + renderer mask
                arr = grid.cell_data[prop_key].copy().astype(float)
                arr[outside] = np.nan
                grid.cell_data[prop_key] = arr

        # Add domain label property so users can visualise domain boundaries
        domain_value = params.get("domain_value", "Inside")
        n_cells = len(dm)
        domain_codes = np.zeros(n_cells, dtype=np.int32)
        domain_codes[dm] = 1   # inside domain
        domain_codes[outside] = 0  # outside domain
        grid.cell_data["DOMAIN"] = domain_codes
        grid.cell_data["domain_mask"] = dm.astype(np.uint8, copy=False)

        # Store label mapping so the legend can show readable names
        import json
        label_map = {0: "Outside", 1: str(domain_value)}
        try:
            grid.field_data["DOMAIN_LABEL_MAP"] = [json.dumps(label_map)]
        except Exception:
            pass  # field_data not supported on all grid types

        logger.info(
            "%s: Applied domain '%s' to %d cells (%d outside, %d inside). "
            "Added DOMAIN + domain_mask properties for visualization.",
            method_name, domain_value, n_cells, n_outside, int(dm.sum()),
        )
    except Exception as exc:
        logger.warning("%s domain masking failed: %s", method_name, exc)


# ── Internal ─────────────────────────────────────────────────────

def _get_irbf_data(registry) -> Optional[dict]:
    # Try the passed registry first, then the global singleton
    candidates = [registry]
    try:
        from ..core.data_registry import DataRegistry
        singleton = DataRegistry.instance()
        if singleton is not registry:
            candidates.append(singleton)
    except Exception:
        pass

    for reg in candidates:
        if reg is None:
            continue
        for name in ("get_indicator_rbf_domain", "get_data", "get_results"):
            fn = getattr(reg, name, None)
            if fn is None:
                continue
            try:
                d = fn("indicator_rbf_domain") if name in ("get_data", "get_results") else fn()
                if isinstance(d, dict):
                    return d
            except Exception:
                pass
    return None


def _mesh_inside(centroids, verts, faces) -> Optional[np.ndarray]:
    try:
        import pyvista as pv
        fp = np.hstack([
            np.full((len(faces), 1), 3, dtype=np.int64),
            faces.astype(np.int64),
        ]).ravel()
        surface = pv.PolyData(verts, fp)
        pts = pv.PolyData(centroids)
        sel = pts.select_enclosed_points(surface, check_surface=False)
        inside = np.asarray(sel["SelectedPoints"], dtype=bool)
        logger.info("IRBF mesh: %d/%d inside", inside.sum(), len(inside))
        return inside
    except Exception as e:
        logger.warning("Point-in-mesh failed: %s", e)
        return None


def _prob_inside(centroids, prob, x, y, z, iso_value) -> np.ndarray:
    from scipy.interpolate import RegularGridInterpolator

    # ── Coordinate alignment ──────────────────────────────────────
    # The IRBF probability field may be in a different coordinate
    # system (e.g., UTM) than the SGSIM grid centroids (local).
    # Detect this by comparing centroids of both, shift if needed.
    irbf_center = np.array([
        (x[0] + x[-1]) / 2,
        (y[0] + y[-1]) / 2,
        (z[0] + z[-1]) / 2,
    ])
    grid_center = np.mean(centroids, axis=0)
    offset = irbf_center - grid_center
    offset_mag = np.linalg.norm(offset)
    grid_diag = np.linalg.norm(centroids.max(axis=0) - centroids.min(axis=0))

    query_centroids = centroids
    if offset_mag > grid_diag * 0.5:
        logger.info(
            "IRBF domain: coordinate shift detected (offset=%.0f m, grid_diag=%.0f m). "
            "Shifting centroids to match IRBF field.",
            offset_mag, grid_diag,
        )
        query_centroids = centroids + offset

    interp = RegularGridInterpolator(
        (z, y, x), prob, method="linear",
        bounds_error=False, fill_value=0.0,
    )
    pts = np.column_stack([query_centroids[:, 2], query_centroids[:, 1], query_centroids[:, 0]])
    p = interp(pts)
    inside = p >= iso_value
    logger.info(
        "IRBF prob: %d/%d inside (iso=%.2f, offset=%.0f m)",
        inside.sum(), len(inside), iso_value, offset_mag,
    )
    return inside
