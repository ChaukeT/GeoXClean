"""
Geological Model Visualisation Utilities
=========================================

Converts GeologicalModelBuilder results into the renderer package format
expected by SurfaceRenderer.load_geology_package().

Also provides helpers for:
- Generating domain solid meshes from scalar fields
- Creating contact point spheres for QC overlay
- Creating structural orientation discs for QC overlay
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False


# ═══════════════════════════════════════════════════════════════════
# Package builder: builder result → renderer package
# ═══════════════════════════════════════════════════════════════════

def build_geology_package(
    build_result: Dict[str, Any],
    contacts_df=None,
    orientations_df=None,
    strat_column: Optional[List[Dict[str, str]]] = None,
    lithology_grouping: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, Any]:
    """Convert GeologicalModelBuilder.build() output to renderer package format.

    Parameters
    ----------
    build_result : dict
        Output from GeologicalModelBuilder.build() containing:
        - surfaces: {name: pyvista.PolyData} or {name: {vertices, faces}}
        - scalar_field: np.ndarray (nx, ny, nz)
        - grid_origin, grid_spacing, grid_dims
        - contact_honouring_summary, audit_record
        - isovalues, unit_names (for stratigraphic models)
    contacts_df : DataFrame, optional
        Contact points for QC overlay.
    orientations_df : DataFrame, optional
        Structural measurements for orientation disc overlay.
    strat_column : list of dict, optional
        Stratigraphic column definition.
    lithology_grouping : dict, optional
        Lithology grouping for unit names.

    Returns
    -------
    dict
        Package compatible with SurfaceRenderer.load_geology_package():
        - surfaces: list of {name, vertices, faces, formation}
        - solids: list of {unit_name, vertices, faces, volume_m3}
        - unified_mesh: dict or None
        - report: audit data
        - contact_points: list of {X, Y, Z, surface_name} (for QC overlay)
        - orientation_discs: list of {X, Y, Z, dip, azimuth} (for QC overlay)
    """
    package = {
        "surfaces": [],
        "solids": [],
        "unified_mesh": None,
        "report": build_result.get("audit_record"),
        "log": {
            "elapsed_seconds": build_result.get("elapsed_seconds", 0),
            "contact_honouring_summary": build_result.get("contact_honouring_summary", {}),
        },
        "contact_points": [],
        "orientation_discs": [],
    }

    # ── Convert surfaces ──────────────────────────────────────────
    raw_surfaces = build_result.get("surfaces", {})
    if isinstance(raw_surfaces, dict):
        for name, mesh in raw_surfaces.items():
            surface_entry = _mesh_to_surface_dict(name, mesh)
            if surface_entry:
                package["surfaces"].append(surface_entry)
    elif isinstance(raw_surfaces, list):
        # Already in list format
        package["surfaces"] = raw_surfaces

    # ── Generate domain solids from scalar field ──────────────────
    scalar_field = build_result.get("scalar_field")
    if scalar_field is not None:
        # Normalise isovalues: dict → sorted list of floats
        raw_iso = build_result.get("isovalues", [0.0])
        if isinstance(raw_iso, dict):
            iso_list = sorted(raw_iso.values())
        elif isinstance(raw_iso, (list, tuple)):
            iso_list = sorted(raw_iso)
        else:
            iso_list = [float(raw_iso)]

        unit_names = build_result.get("unit_names")

        solids = generate_domain_solids(
            scalar_field=scalar_field,
            grid_origin=build_result.get("grid_origin", [0, 0, 0]),
            grid_spacing=build_result.get("grid_spacing", [1, 1, 1]),
            isovalues=iso_list,
            unit_names=unit_names,
        )
        package["solids"] = solids

        # Build unified mesh for voxel rendering
        unified = build_unified_mesh(
            scalar_field=scalar_field,
            grid_origin=build_result.get("grid_origin", [0, 0, 0]),
            grid_spacing=build_result.get("grid_spacing", [1, 1, 1]),
            isovalues=iso_list,
            unit_names=unit_names,
        )
        if unified is not None:
            package["unified_mesh"] = unified

    # ── Contact points for QC overlay ─────────────────────────────
    if contacts_df is not None:
        package["contact_points"] = build_contact_points(contacts_df)

    # ── Structural orientation discs ──────────────────────────────
    if orientations_df is not None:
        package["orientation_discs"] = build_orientation_discs(orientations_df)

    logger.info(
        "Built geology package: %d surfaces, %d solids, unified=%s, "
        "%d contact points, %d orientation discs",
        len(package["surfaces"]),
        len(package["solids"]),
        package["unified_mesh"] is not None,
        len(package["contact_points"]),
        len(package["orientation_discs"]),
    )

    return package


def _mesh_to_surface_dict(name: str, mesh) -> Optional[Dict[str, Any]]:
    """Convert a PyVista PolyData or dict to surface dict format."""
    if mesh is None:
        return None

    # PyVista PolyData
    if hasattr(mesh, "points") and hasattr(mesh, "faces"):
        vertices = np.asarray(mesh.points, dtype=np.float64)
        # Extract triangle indices from PyVista face format [3, v0, v1, v2, ...]
        raw_faces = np.asarray(mesh.faces)
        if len(raw_faces) > 0:
            # Reshape from flat [3, a, b, c, 3, d, e, f, ...] to (M, 3)
            try:
                n_faces = mesh.n_faces
                faces = raw_faces.reshape(-1, 4)[:, 1:4] if raw_faces[0] == 3 else raw_faces
            except Exception:
                faces = raw_faces
        else:
            faces = np.empty((0, 3), dtype=np.int64)

        return {
            "name": name,
            "formation": name,
            "vertices": vertices,
            "faces": faces,
        }

    # Already a dict with vertices/faces
    if isinstance(mesh, dict):
        return {
            "name": name,
            "formation": name,
            "vertices": np.asarray(mesh.get("vertices", []), dtype=np.float64),
            "faces": np.asarray(mesh.get("faces", []), dtype=np.int64),
        }

    return None


# ═══════════════════════════════════════════════════════════════════
# Domain solid generation from scalar field
# ═══════════════════════════════════════════════════════════════════

def generate_domain_solids(
    scalar_field: np.ndarray,
    grid_origin: List[float],
    grid_spacing: List[float],
    isovalues: List[float],
    unit_names: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Generate solid volume meshes for each geological domain.

    Uses marching cubes to extract the bounding isosurfaces for each
    domain and creates closed solid meshes.

    Parameters
    ----------
    scalar_field : np.ndarray
        3D scalar field (nx, ny, nz).
    grid_origin : list of float
        [x0, y0, z0] origin of the grid.
    grid_spacing : list of float
        [dx, dy, dz] cell sizes.
    isovalues : list of float
        Sorted isovalues that separate domains.
    unit_names : list of str, optional
        Names for each domain (len = len(isovalues) + 1).

    Returns
    -------
    list of dict
        Solid entries with {unit_name, vertices, faces, volume_m3}.
    """
    if not HAS_PYVISTA:
        logger.warning("PyVista not available — cannot generate domain solids")
        return []

    if unit_names is None:
        unit_names = [f"Domain_{i}" for i in range(len(isovalues) + 1)]

    solids = []
    nx, ny, nz = scalar_field.shape
    origin = np.array(grid_origin, dtype=np.float64)
    spacing = np.array(grid_spacing, dtype=np.float64)
    cell_vol = float(np.prod(spacing))

    try:
        nx, ny, nz = scalar_field.shape
        origin = np.array(grid_origin, dtype=np.float64)
        spacing = np.array(grid_spacing, dtype=np.float64)
        sorted_iso = sorted(isovalues)
        n_domains = len(sorted_iso) + 1

        # Build a scalar-field grid (continuous float values, not integer codes)
        # clip_scalar interpolates at isovalue boundaries → smooth, non-jagged solids
        sf_grid = pv.ImageData(
            dimensions=(nx, ny, nz),
            spacing=tuple(spacing),
            origin=tuple(origin),
        )
        field_flat = scalar_field.ravel(order="F").astype(np.float64)
        sf_grid.point_data["sf"] = field_flat

        for k in range(n_domains):
            unit_name = unit_names[k] if k < len(unit_names) else f"Domain_{k}"
            try:
                # Clip the continuous scalar field between adjacent isosurfaces.
                # clip_scalar uses linear interpolation at the threshold plane →
                # smooth, analytically correct boundaries (no stairstepping).
                if k == 0:
                    domain = sf_grid.clip_scalar(scalars="sf", value=sorted_iso[0], invert=True)
                elif k == n_domains - 1:
                    domain = sf_grid.clip_scalar(scalars="sf", value=sorted_iso[-1])
                else:
                    domain = sf_grid.clip_scalar(scalars="sf", value=sorted_iso[k])
                    domain = domain.clip_scalar(scalars="sf", value=sorted_iso[k - 1], invert=True)

                if domain is None or domain.n_cells == 0:
                    continue

                surface = domain.extract_surface()
                if surface.n_points == 0:
                    continue

                # Clean up degenerate faces
                surface = surface.clean()

                verts = np.asarray(surface.points, dtype=np.float64)
                raw_faces = np.asarray(surface.faces)
                if len(raw_faces) > 0 and raw_faces[0] == 3:
                    faces = raw_faces.reshape(-1, 4)[:, 1:4]
                else:
                    faces = raw_faces

                volume_m3 = float(domain.volume) if hasattr(domain, "volume") else domain.n_cells * float(np.prod(spacing))

                solids.append({
                    "unit_name": unit_name,
                    "name": unit_name,
                    "vertices": verts,
                    "faces": faces,
                    "volume_m3": volume_m3,
                    "domain_code": k,
                })

            except Exception as exc:
                logger.warning("Failed to extract solid for %s: %s", unit_name, exc)

    except Exception as exc:
        logger.error("Domain solid generation failed: %s", exc)

    logger.info("Generated %d domain solids", len(solids))
    return solids


def build_unified_mesh(
    scalar_field: np.ndarray,
    grid_origin: List[float],
    grid_spacing: List[float],
    isovalues: List[float],
    unit_names: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """Build a unified voxel mesh with Formation_ID per cell.

    This creates a single structured grid with domain codes, preventing
    Z-fighting between overlapping surfaces.

    Returns
    -------
    dict or None
        {grid: pv.ImageData, formation_names: list, formation_ids: ndarray}
    """
    if not HAS_PYVISTA:
        return None

    if unit_names is None:
        unit_names = [f"Domain_{i}" for i in range(len(isovalues) + 1)]

    try:
        nx, ny, nz = scalar_field.shape
        origin = np.array(grid_origin, dtype=np.float64)
        spacing = np.array(grid_spacing, dtype=np.float64)

        # scalar_field shape (nx,ny,nz) is VERTEX counts → use as point dimensions
        grid = pv.ImageData(
            dimensions=(nx, ny, nz),
            spacing=tuple(spacing),
            origin=tuple(origin),
        )

        # Assign domain codes based on isovalue ranges (point data)
        field_flat = scalar_field.ravel(order="F")
        domain_ids = np.zeros(len(field_flat), dtype=np.int32)

        sorted_iso = sorted(isovalues)
        for k, iso in enumerate(sorted_iso):
            domain_ids[field_flat >= iso] = k + 1

        grid.point_data["Formation_ID"] = domain_ids

        # Map IDs to names
        formation_names = []
        for cell_id in domain_ids:
            if cell_id < len(unit_names):
                formation_names.append(unit_names[cell_id])
            else:
                formation_names.append(f"Domain_{cell_id}")

        return {
            "grid": grid,
            "formation_names": unit_names,
            "formation_ids": domain_ids,
            "vertices": np.asarray(grid.points, dtype=np.float64),
            "n_cells": grid.n_cells,
        }

    except Exception as exc:
        logger.error("Unified mesh generation failed: %s", exc)
        return None


# ═══════════════════════════════════════════════════════════════════
# Contact point spheres for QC overlay
# ═══════════════════════════════════════════════════════════════════

def build_contact_points(contacts_df) -> List[Dict[str, Any]]:
    """Build contact point data for QC sphere overlay.

    Parameters
    ----------
    contacts_df : DataFrame
        Columns: X, Y, Z, surface_name (or unit_above, unit_below).

    Returns
    -------
    list of dict
        [{X, Y, Z, surface_name, hole_id}, ...]
    """
    points = []
    if contacts_df is None or contacts_df.empty:
        return points

    col_map = {c.lower(): c for c in contacts_df.columns}
    x_col = col_map.get('x') or col_map.get('easting')
    y_col = col_map.get('y') or col_map.get('northing')
    z_col = col_map.get('z') or col_map.get('elevation')

    if not all([x_col, y_col, z_col]):
        return points

    surface_col = col_map.get('surface_name')
    hole_col = None
    for name in ('holeid', 'hole_id', 'bhid'):
        if name in col_map:
            hole_col = col_map[name]
            break

    for _, row in contacts_df.iterrows():
        try:
            entry = {
                "X": float(row[x_col]),
                "Y": float(row[y_col]),
                "Z": float(row[z_col]),
            }
            if surface_col and surface_col in row.index:
                entry["surface_name"] = str(row[surface_col])
            if hole_col and hole_col in row.index:
                entry["hole_id"] = str(row[hole_col])
            points.append(entry)
        except (ValueError, TypeError):
            continue

    return points


# ═══════════════════════════════════════════════════════════════════
# Structural orientation discs for QC overlay
# ═══════════════════════════════════════════════════════════════════

def build_orientation_discs(orientations_df) -> List[Dict[str, Any]]:
    """Build orientation disc data for structural measurement overlay.

    Parameters
    ----------
    orientations_df : DataFrame
        Columns: X, Y, Z, dip, azimuth, [feature_type].

    Returns
    -------
    list of dict
        [{X, Y, Z, dip, azimuth, feature_type}, ...]
    """
    discs = []
    if orientations_df is None or orientations_df.empty:
        return discs

    col_map = {c.lower(): c for c in orientations_df.columns}
    x_col = col_map.get('x') or col_map.get('easting')
    y_col = col_map.get('y') or col_map.get('northing')
    z_col = col_map.get('z') or col_map.get('elevation')
    dip_col = col_map.get('dip')
    az_col = col_map.get('azimuth') or col_map.get('dip_direction') or col_map.get('strike')

    if not all([x_col, y_col, z_col, dip_col, az_col]):
        return discs

    feature_col = col_map.get('feature_type') or col_map.get('type')

    for _, row in orientations_df.iterrows():
        try:
            entry = {
                "X": float(row[x_col]),
                "Y": float(row[y_col]),
                "Z": float(row[z_col]),
                "dip": float(row[dip_col]),
                "azimuth": float(row[az_col]),
            }
            if feature_col and feature_col in row.index:
                entry["feature_type"] = str(row[feature_col])
            else:
                entry["feature_type"] = "bedding"
            discs.append(entry)
        except (ValueError, TypeError):
            continue

    return discs
