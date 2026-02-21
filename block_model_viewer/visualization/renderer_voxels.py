"""Voxel + surface rendering helpers for GeoX (PyVista).

This is a drop-in helper module.

Why this exists
- Rendering voxel blocks as an unstructured grid with per-cell colours is heavy.
- Using pv.UniformGrid / ImageData is faster and more reliable for categorical volumes.

Usage
In your existing Renderer class, you can:

    from block_model_viewer.visualization.renderer_voxels import (
        add_categorical_voxel_volume,
        add_surface_meshes,
        clear_layers_by_tag,
    )

and call these functions with your plotter.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import pyvista as pv
except Exception:  # pragma: no cover
    pv = None  # type: ignore


def _require_pv() -> None:
    if pv is None:
        raise RuntimeError("PyVista is not available")


def add_categorical_voxel_volume(
    plotter,
    labels: np.ndarray,
    origin: Tuple[float, float, float],
    spacing: Tuple[float, float, float],
    name: str = "Geological Model (Solid)",
    opacity: float = 0.85,
    show_edges: bool = False,
    tag: Optional[str] = None,
) -> object:
    """Render a categorical voxel model efficiently.

    Args:
        plotter: PyVista plotter
        labels: 3D array (nx, ny, nz) with categorical labels
        origin: (x0, y0, z0) origin of the grid
        spacing: (dx, dy, dz) spacing between voxels
        name: Name for the actor
        opacity: Opacity (0-1)
        show_edges: Whether to show edges
        tag: Optional tag for layer management

    Returns:
        Actor object
    """
    _require_pv()

    # Infer grid dims from labels
    if labels.ndim == 3:
        nx, ny, nz = labels.shape
        flat = labels.ravel(order="F")
    else:
        raise ValueError("labels must be 3D array (nx, ny, nz)")

    dx, dy, dz = spacing

    # UniformGrid is ImageData in newer pyvista
    grid = pv.UniformGrid(dimensions=(nx + 1, ny + 1, nz + 1), spacing=(dx, dy, dz), origin=origin)
    grid.cell_data["Formation"] = flat.astype(np.int32)

    actor = plotter.add_mesh(
        grid,
        name=name,
        scalars="Formation",
        opacity=opacity,
        show_edges=show_edges,
        categories=True,
        render=False,
    )

    # Store tag if provided
    if tag:
        actor._geox_tag = tag  # type: ignore

    return actor


# Alias for backward compatibility
add_voxel_volume = add_categorical_voxel_volume


def add_surface_meshes(
    plotter,
    surfaces: Union[List, Dict],
    tag_prefix: str = "geo_surface",
    opacity: float = 0.9,
    smooth_shading: bool = True,
) -> Dict[str, object]:
    """Add multiple surface meshes to plotter and return actor map.

    Args:
        plotter: PyVista plotter
        surfaces: Either:
            - List of SurfaceMesh objects (from ModelResult)
            - Dict[str, pv.PolyData] mapping names to meshes
        tag_prefix: Prefix for layer tags
        opacity: Opacity (0-1)
        smooth_shading: Whether to use smooth shading

    Returns:
        Dict mapping surface names to actors
    """
    _require_pv()
    actors: Dict[str, object] = {}

    # Handle List[SurfaceMesh] from ModelResult
    if isinstance(surfaces, list):
        for surface in surfaces:
            if hasattr(surface, "vertices") and hasattr(surface, "faces"):
                # Convert SurfaceMesh to PolyData
                verts = np.asarray(surface.vertices, dtype=np.float64)
                faces = np.asarray(surface.faces)
                
                # CRITICAL FIX: Convert (N, 3) face array to PyVista format [3, i, j, k, ...]
                if faces.ndim == 2:
                    n_faces = len(faces)
                    n_verts_per_face = faces.shape[1]
                    faces_pv = np.hstack([
                        np.full((n_faces, 1), n_verts_per_face, dtype=np.int64),
                        faces.astype(np.int64)
                    ]).flatten()
                else:
                    faces_pv = faces.astype(np.int64)
                
                mesh = pv.PolyData(verts, faces_pv)
                name = getattr(surface, "name", f"surface_{len(actors)}")
                actors[name] = plotter.add_mesh(
                    mesh,
                    name=name,
                    opacity=opacity,
                    smooth_shading=smooth_shading,
                    show_edges=False,
                    render=False,
                )
                # Store tag
                tag = f"{tag_prefix}_{name}"
                actors[name]._geox_tag = tag  # type: ignore
    # Handle Dict[str, PolyData]
    elif isinstance(surfaces, dict):
        for name, mesh in surfaces.items():
            if mesh is None or (hasattr(mesh, "n_points") and mesh.n_points == 0):
                continue
            actors[name] = plotter.add_mesh(
                mesh,
                name=name,
                opacity=opacity,
                smooth_shading=smooth_shading,
                show_edges=False,
                render=False,
            )
            # Store tag
            tag = f"{tag_prefix}_{name}"
            actors[name]._geox_tag = tag  # type: ignore
    else:
        raise ValueError(f"surfaces must be List[SurfaceMesh] or Dict[str, PolyData], got {type(surfaces)}")

    return actors


def clear_layers_by_tag(plotter, tag_prefix: str) -> None:
    """Remove all actors with tags matching the prefix.

    Args:
        plotter: PyVista plotter
        tag_prefix: Tag prefix to match (e.g., "geo_")
    """
    _require_pv()

    actors_to_remove = []
    for actor in plotter.renderer.GetActors():
        if hasattr(actor, "_geox_tag"):
            tag = actor._geox_tag  # type: ignore
            if tag and tag.startswith(tag_prefix):
                actors_to_remove.append(actor)

    for actor in actors_to_remove:
        plotter.remove_actor(actor, render=False)

