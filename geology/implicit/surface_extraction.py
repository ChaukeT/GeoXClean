"""
Surface Extraction — Marching Cubes and Mesh Utilities.
========================================================

Evaluates a scalar field on a regular 3D grid, then extracts
isosurfaces using marching cubes (scikit-image or PyVista).
"""

from __future__ import annotations

import logging
from typing import Callable, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Grid Evaluation
# ═══════════════════════════════════════════════════════════════════

def evaluate_field_on_grid(
    grid_origin: np.ndarray,
    grid_spacing: np.ndarray,
    grid_dims: Tuple[int, int, int],
    evaluate_fn: Callable,
    batch_size: int = 50_000,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> np.ndarray:
    """Evaluate a scalar field on a regular 3D grid.

    Parameters
    ----------
    grid_origin : (3,) grid min corner [x0, y0, z0].
    grid_spacing : (3,) cell size [dx, dy, dz].
    grid_dims : (nx, ny, nz) number of grid vertices per axis.
    evaluate_fn : callable (B, 3) -> (B,).
    batch_size : max points per evaluation call.
    progress_callback : optional (percent, message).

    Returns
    -------
    np.ndarray, shape (nx, ny, nz)
    """
    nx, ny, nz = grid_dims
    N = nx * ny * nz

    # Build grid coordinates
    x = grid_origin[0] + np.arange(nx) * grid_spacing[0]
    y = grid_origin[1] + np.arange(ny) * grid_spacing[1]
    z = grid_origin[2] + np.arange(nz) * grid_spacing[2]

    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    # Evaluate in batches
    values = np.empty(N, dtype=np.float64)
    n_batches = (N + batch_size - 1) // batch_size

    for b in range(n_batches):
        start = b * batch_size
        end = min(start + batch_size, N)
        values[start:end] = evaluate_fn(points[start:end])

        if progress_callback is not None:
            pct = int(100.0 * end / N)
            progress_callback(pct, f"Evaluating grid: {end}/{N} points")

    return values.reshape((nx, ny, nz))


# ═══════════════════════════════════════════════════════════════════
# Marching Cubes Surface Extraction
# ═══════════════════════════════════════════════════════════════════

def extract_isosurface(
    scalar_field: np.ndarray,
    grid_origin: np.ndarray,
    grid_spacing: np.ndarray,
    isovalue: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract isosurface using marching cubes.

    Parameters
    ----------
    scalar_field : (nx, ny, nz) evaluated scalar field values.
    grid_origin : (3,) grid min corner.
    grid_spacing : (3,) cell spacing.
    isovalue : float, the isosurface level.

    Returns
    -------
    vertices : (V, 3) vertex coordinates in world space.
    faces : (F, 3) triangle face indices.
    """
    try:
        from skimage.measure import marching_cubes
    except ImportError:
        raise ImportError(
            "scikit-image is required for marching cubes surface extraction. "
            "Install with: pip install scikit-image"
        )

    # Check that the scalar field spans the isovalue
    fmin, fmax = scalar_field.min(), scalar_field.max()
    if isovalue < fmin or isovalue > fmax:
        logger.warning(
            "Isovalue %.4f is outside scalar field range [%.4f, %.4f]. "
            "Returning empty mesh.",
            isovalue, fmin, fmax,
        )
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.int64)

    verts, faces, normals, values = marching_cubes(
        scalar_field,
        level=isovalue,
        spacing=tuple(grid_spacing),
    )

    # Transform vertices from grid-local to world coordinates
    verts = verts + grid_origin[np.newaxis, :]

    logger.info(
        "Extracted isosurface at %.4f: %d vertices, %d triangles",
        isovalue, verts.shape[0], faces.shape[0],
    )

    return verts.astype(np.float64), faces.astype(np.int64)


def cleanup_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    min_area: float = 1e-10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Remove degenerate triangles from a mesh.

    Parameters
    ----------
    vertices : (V, 3)
    faces : (F, 3)
    min_area : minimum triangle area to keep.

    Returns
    -------
    vertices : (V, 3) (unchanged)
    faces : (F', 3) cleaned faces
    """
    if faces.shape[0] == 0:
        return vertices, faces

    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    # Cross product to get area
    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)

    mask = areas > min_area
    n_removed = int(np.sum(~mask))
    if n_removed > 0:
        logger.debug("Removed %d degenerate triangles", n_removed)

    return vertices, faces[mask]


def field_to_pyvista_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    surface_name: str = "surface",
):
    """Convert vertices + faces to a PyVista PolyData mesh.

    Parameters
    ----------
    vertices : (V, 3)
    faces : (F, 3)
    surface_name : label stored in mesh field_data.

    Returns
    -------
    pyvista.PolyData
    """
    try:
        import pyvista as pv
    except ImportError:
        raise ImportError("PyVista is required for mesh conversion.")

    if vertices.shape[0] == 0 or faces.shape[0] == 0:
        mesh = pv.PolyData()
        mesh.field_data["surface_name"] = [surface_name]
        return mesh

    # PyVista expects faces as [n_verts, v0, v1, v2, ...]
    F = faces.shape[0]
    pv_faces = np.column_stack([
        np.full(F, 3, dtype=np.int64),
        faces,
    ]).ravel()

    mesh = pv.PolyData(vertices.astype(np.float64), pv_faces)
    mesh.field_data["surface_name"] = [surface_name]

    return mesh
