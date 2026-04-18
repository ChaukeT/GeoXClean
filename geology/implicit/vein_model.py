"""
Vein Model -- Boolean Combination of HW/FW Surfaces.
=====================================================

Models narrow tabular bodies (veins, shear zones, dykes) using separate
hanging wall and footwall surfaces combined via Boolean operations.

Three methods:
  - median_thickness: interpolate median surface + thickness field (recommended)
  - separate_hw_fw: interpolate HW and FW independently, combine
  - boolean: explicit Boolean combination f_vein = max(f_hw, -f_fw)

Mathematical basis: Eq. 8.1 of the GeoX Math Specification.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .contact_data import dip_azimuth_to_normal, orientations_dataframe_to_list
from .signed_distance import construct_sdf_constraints, estimate_contact_normals
from .scalar_field import (
    assemble_augmented_matrix,
    solve_augmented_system,
    make_evaluate_fn,
    solve_augmented_system_pum,
    make_evaluate_fn_pum,
    PUM_THRESHOLD,
)
from .surface_extraction import (
    evaluate_field_on_grid,
    extract_isosurface,
    cleanup_mesh,
    field_to_pyvista_mesh,
)
from .validation import check_contact_honouring

logger = logging.getLogger(__name__)


# =====================================================================
# Data classes
# =====================================================================

@dataclass
class VeinIntersection:
    """A drillhole intersection of a vein."""
    hole_id: str
    hw_point: np.ndarray     # (3,) hanging wall contact
    fw_point: np.ndarray     # (3,) footwall contact
    midpoint: np.ndarray     # (3,) midpoint = (hw + fw) / 2
    thickness: float         # ||hw - fw||
    normal: np.ndarray       # (3,) estimated vein normal
    is_pinch_out: bool = False


# =====================================================================
# Vein intersection extraction
# =====================================================================

def extract_vein_intersections(
    drillhole_data: pd.DataFrame,
    vein_code: str,
    lithology_column: str = "lith_code",
    x_col: str = "X",
    y_col: str = "Y",
    z_col: str = "Z",
    from_col: str = "depth_from",
    to_col: str = "depth_to",
    hole_id_col: str = "hole_id",
    grouping: Optional[Dict] = None,
) -> List[VeinIntersection]:
    """Extract vein intersections from drillhole lithology logs.

    For each drillhole, finds intervals coded as the vein unit.
    The top of the interval is the HW contact, the bottom is the FW
    contact.  Computes midpoint and thickness.

    Handles multiple vein intersections per hole (stacked veins).

    Parameters
    ----------
    drillhole_data : pd.DataFrame
        Lithology log with interval data.
    vein_code : str
        Lithology code identifying the vein unit.
    grouping : dict, optional
        {group_name: [raw_codes]} -- if provided, all codes in the
        group matching vein_code are treated as vein.

    Returns
    -------
    List of VeinIntersection
    """
    # Build set of codes that count as vein
    vein_codes = {vein_code}
    if grouping:
        for group_name, codes in grouping.items():
            if vein_code in codes or group_name == vein_code:
                vein_codes.update(str(c) for c in codes)

    intersections = []

    for hid, hole_df in drillhole_data.groupby(hole_id_col):
        hole_sorted = hole_df.sort_values(from_col)

        # Find contiguous vein intervals
        in_vein = False
        hw_depth = None
        hw_point = None

        for _, row in hole_sorted.iterrows():
            code = str(row.get(lithology_column, "")).strip()
            is_vein = code in vein_codes

            if is_vein and not in_vein:
                # Entry into vein: HW contact
                in_vein = True
                hw_depth = float(row[from_col])
                hw_point = np.array([
                    float(row[x_col]), float(row[y_col]), float(row[z_col]),
                ], dtype=np.float64)

            elif not is_vein and in_vein:
                # Exit from vein: FW contact
                in_vein = False
                fw_depth = float(row[from_col])
                fw_point = np.array([
                    float(row[x_col]), float(row[y_col]), float(row[z_col]),
                ], dtype=np.float64)

                midpoint = (hw_point + fw_point) / 2.0
                thickness = np.linalg.norm(fw_point - hw_point)

                # Default normal: unit vector from HW to FW (downward)
                diff = fw_point - hw_point
                norm_mag = np.linalg.norm(diff)
                if norm_mag > 1e-10:
                    normal = diff / norm_mag
                else:
                    normal = np.array([0.0, 0.0, -1.0])

                intersections.append(VeinIntersection(
                    hole_id=str(hid),
                    hw_point=hw_point,
                    fw_point=fw_point,
                    midpoint=midpoint,
                    thickness=thickness,
                    normal=normal,
                ))

        # Handle case where vein continues to end of hole
        if in_vein and hw_point is not None:
            last_row = hole_sorted.iloc[-1]
            fw_point = np.array([
                float(last_row[x_col]), float(last_row[y_col]),
                float(last_row[z_col]),
            ], dtype=np.float64)
            midpoint = (hw_point + fw_point) / 2.0
            thickness = np.linalg.norm(fw_point - hw_point)
            diff = fw_point - hw_point
            norm_mag = np.linalg.norm(diff)
            normal = diff / norm_mag if norm_mag > 1e-10 else np.array([0.0, 0.0, -1.0])

            intersections.append(VeinIntersection(
                hole_id=str(hid),
                hw_point=hw_point,
                fw_point=fw_point,
                midpoint=midpoint,
                thickness=thickness,
                normal=normal,
            ))

    logger.info(
        "Extracted %d vein intersections from %d drillholes",
        len(intersections),
        drillhole_data[hole_id_col].nunique() if hole_id_col in drillhole_data.columns else 0,
    )

    return intersections


# ─────────────────────────────────────────────────────────────────────────────
# Helpers for barren-hole pinch-out elevation estimation
# ─────────────────────────────────────────────────────────────────────────────

def _build_preliminary_sdf(intersections: "List[VeinIntersection]") -> "Callable[[np.ndarray], np.ndarray]":
    """Fit a lightweight scalar field from the known vein midpoints.

    Uses a simple radial-basis pass (no gradient constraints) — fast enough
    for the preliminary purpose of finding zero-crossing depths in barren
    holes.  The field value is negative inside the vein (+1 at HW, -1 at FW,
    0 at midpoint).

    Returns a callable (B, 3) -> (B,) or None if the fit fails.
    """
    if not intersections:
        return None
    try:
        from .scalar_field import (
            assemble_augmented_matrix,
            solve_augmented_system,
            make_evaluate_fn,
        )
        from .signed_distance import construct_sdf_constraints

        mid_coords = np.array([vi.midpoint for vi in intersections], dtype=np.float64)
        hw_coords = np.array([vi.hw_point for vi in intersections], dtype=np.float64)
        fw_coords = np.array([vi.fw_point for vi in intersections], dtype=np.float64)

        # Value constraints: HW = +1, midpoint = 0, FW = -1
        vc = np.vstack([hw_coords, mid_coords, fw_coords])
        vv = np.concatenate([
            np.ones(len(hw_coords)),
            np.zeros(len(mid_coords)),
            -np.ones(len(fw_coords)),
        ])

        # Estimate a reasonable range from extent
        span = vc.max(axis=0) - vc.min(axis=0)
        range_ = max(50.0, float(np.linalg.norm(span)) / 3.0)

        K_aug, N_v, N_g = assemble_augmented_matrix(
            vc, np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.float64),
            kernel_type="spheroidal", alpha=1.0, range_=range_,
            nugget=1e-4, accuracy=1e-6, drift_type="constant",
        )
        vw, gw, pc = solve_augmented_system(K_aug, vv, np.empty(0), N_v, 0)
        return make_evaluate_fn(
            vc, np.empty((0, 3)), np.empty((0, 3)),
            vw, gw, pc,
            "spheroidal", 1.0, range_,
        )
    except Exception as exc:
        logger.debug("Preliminary SDF build failed (will use IDW fallback): %s", exc)
        return None


def _find_sdf_zero_crossing(
    x: float,
    y: float,
    z_top: float,
    z_bot: float,
    preliminary_evaluate_fn,
    vein_midpoints: np.ndarray,
    nearby_idx: list,
) -> "Optional[float]":
    """Find vein Z via preliminary SDF zero-crossing, or IDW fallback.

    Traces the preliminary scalar field along the vertical line (x, y, z)
    from z_top to z_bot.  Returns the depth of the first zero-crossing
    (SDF sign change from positive to negative), or None if not found.

    Falls back to IDW when ``preliminary_evaluate_fn`` is None.
    """
    if preliminary_evaluate_fn is not None:
        n_samples = 60
        z_samples = np.linspace(z_top, z_bot, n_samples)
        pts = np.column_stack([
            np.full(n_samples, x),
            np.full(n_samples, y),
            z_samples,
        ])
        try:
            sf_vals = preliminary_evaluate_fn(pts)
            sign_changes = np.where(np.diff(np.sign(sf_vals)))[0]
            if len(sign_changes) == 0:
                return None
            # First zero-crossing: linear interpolation between bracketing samples
            i0 = int(sign_changes[0])
            dv = sf_vals[i0 + 1] - sf_vals[i0]
            if abs(dv) < 1e-14:
                return float(z_samples[i0])
            t = -sf_vals[i0] / dv
            return float(z_samples[i0] + t * (z_samples[i0 + 1] - z_samples[i0]))
        except Exception:
            pass  # fall through to IDW

    # IDW fallback (geologically imprecise for dipping veins, but better than nothing)
    nearby_midpoints = vein_midpoints[nearby_idx]
    dx = nearby_midpoints[:, 0] - x
    dy = nearby_midpoints[:, 1] - y
    horiz_dists = np.sqrt(dx ** 2 + dy ** 2)
    horiz_dists = np.where(horiz_dists < 1.0, 1.0, horiz_dists)
    weights = 1.0 / horiz_dists
    weights /= weights.sum()
    return float(np.dot(weights, nearby_midpoints[:, 2]))


def extract_vein_intersections_with_barren(
    lithology_df: pd.DataFrame,
    vein_code: str,
    hole_id_col: str = "hole_id",
    lith_col: str = "lith_code",
    x_col: str = "X",
    y_col: str = "Y",
    z_col: str = "Z",
    depth_from_col: str = "depth_from",
    depth_to_col: str = "depth_to",
    collar_df: Optional[pd.DataFrame] = None,
    min_thickness: float = 0.1,
    search_radius: float = 200.0,
) -> List[VeinIntersection]:
    """Extract vein intersections AND barren-hole pinch-out constraints.

    Barren holes (holes that pass through the expected vein zone but do
    NOT intersect the vein code) are critical for constraining where the
    vein DOES NOT exist.  Without them, the interpolated vein surface
    extends to the grid boundary regardless of the evidence.

    Parameters
    ----------
    lithology_df : pd.DataFrame
        Interval log with hole_id, lith_code, X, Y, Z, depth columns.
    vein_code : str
        Lithology code identifying the vein (e.g., "VN", "VEIN").
    collar_df : pd.DataFrame, optional
        Collar survey with hole_id, X, Y, Z columns.  Used to find
        barren holes.  If None, all unique hole_ids in lithology_df
        are used.
    min_thickness : float
        Minimum vein thickness to record (metres).
    search_radius : float
        Only consider holes within this distance of vein holes as
        potential barren control points.

    Returns
    -------
    list of VeinIntersection
        Normal intersections with ``is_pinch_out=False`` plus barren
        control points with ``is_pinch_out=True`` and ``thickness=0``.
    """
    # Step 1: Get normal vein intersections
    intersections = extract_vein_intersections(
        lithology_df, vein_code,
        hole_id_col=hole_id_col,
        lith_col=lith_col,
        x_col=x_col,
        y_col=y_col,
        z_col=z_col,
        depth_from_col=depth_from_col,
        depth_to_col=depth_to_col,
        min_thickness=min_thickness,
    )

    if not intersections:
        return intersections

    # Step 2: Identify all holes and which ones hit the vein
    col_map = {c.lower(): c for c in lithology_df.columns}
    h_col = col_map.get(hole_id_col.lower(), hole_id_col)
    all_hole_ids = lithology_df[h_col].dropna().unique().tolist() if h_col in lithology_df.columns else []
    vein_hole_ids = {iv.hole_id for iv in intersections}
    barren_hole_ids = [h for h in all_hole_ids if str(h) not in vein_hole_ids]

    if not barren_hole_ids:
        return intersections

    # Step 3: Build quick spatial index of vein midpoints
    try:
        from scipy.spatial import cKDTree
        vein_midpoints = np.array([iv.midpoint for iv in intersections], dtype=np.float64)
        vein_tree = cKDTree(vein_midpoints[:, :2])  # XY only
    except (ImportError, ValueError):
        logger.debug("Barren hole detection skipped: scipy not available or no vein midpoints")
        return intersections

    # Step 4a: Build a quick preliminary SDF from known vein intersections.
    # This lets us find exact zero-crossings for barren holes, correctly
    # honouring dip and strike — far superior to IDW depth interpolation.
    _preliminary_evaluate_fn = _build_preliminary_sdf(intersections)

    # Step 4b: For each barren hole, find its representative position
    # (use collar or deepest intersection within search_radius)
    x_col_ = col_map.get(x_col.lower(), x_col)
    y_col_ = col_map.get(y_col.lower(), y_col)
    z_col_ = col_map.get(z_col.lower(), z_col)

    barren_added = 0
    for hole_id in barren_hole_ids:
        hole_rows = lithology_df[lithology_df[h_col].astype(str) == str(hole_id)]
        if hole_rows.empty:
            continue

        try:
            x = float(hole_rows[x_col_].dropna().iloc[0])
            y = float(hole_rows[y_col_].dropna().iloc[0])
            z = float(hole_rows[z_col_].dropna().iloc[0])
        except (IndexError, ValueError, KeyError):
            continue

        # Check if this hole is within search_radius of any vein hole
        dists, _ = vein_tree.query([x, y], k=1)
        if float(dists) > search_radius:
            continue

        nearby_idx = vein_tree.query_ball_point([x, y], r=search_radius)
        if not nearby_idx:
            continue

        # ── Estimate vein depth via preliminary scalar field zero-crossing ──
        # IDW is wrong for dipping veins: it averages depths geometrically
        # rather than following the actual surface normal.  Instead we
        # trace the preliminary SDF along the barren hole and find where
        # it crosses zero — that is exactly where the vein would be if
        # the hole had been long enough.
        expected_z = _find_sdf_zero_crossing(
            x, y,
            z_top=float(hole_rows[z_col_].max()) + 10.0,
            z_bot=float(hole_rows[z_col_].min()) - search_radius,
            preliminary_evaluate_fn=_preliminary_evaluate_fn,
            vein_midpoints=vein_midpoints,
            nearby_idx=nearby_idx,
        )
        if expected_z is None:
            continue

        # Add a zero-thickness pinch-out control point
        midpoint = np.array([x, y, expected_z], dtype=np.float64)
        dummy_normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        pinch = VeinIntersection(
            hole_id=str(hole_id),
            hw_point=midpoint.copy(),
            fw_point=midpoint.copy(),
            midpoint=midpoint,
            thickness=0.0,
            normal=dummy_normal,
            is_pinch_out=True,
        )
        intersections.append(pinch)
        barren_added += 1

    if barren_added > 0:
        logger.info(
            "Barren hole pinch-outs: added %d control points from %d barren holes",
            barren_added, len(barren_hole_ids),
        )

    return intersections


# =====================================================================
# Minimum thickness enforcement
# =====================================================================

def enforce_minimum_thickness(
    hw_field_values: np.ndarray,
    fw_field_values: np.ndarray,
    min_thickness: float,
    grid_spacing: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Adjust HW/FW fields to enforce minimum thickness.

    Where HW and FW surfaces are closer than min_thickness,
    push them apart symmetrically from the midpoint.

    The fields are signed distance functions:
      HW surface: f_hw = 0  (f_hw < 0 is inside vein from HW side)
      FW surface: f_fw = 0  (f_fw > 0 is inside vein from FW side)

    Vein volume = {x : f_hw <= 0 AND f_fw >= 0}
    Thickness ~ f_fw - f_hw at any point inside the vein.

    Parameters
    ----------
    hw_field_values : (N,) scalar field values from HW SDF
    fw_field_values : (N,) scalar field values from FW SDF
    min_thickness : float (metres)
    grid_spacing : (3,) not used directly but kept for API consistency

    Returns
    -------
    hw_adjusted, fw_adjusted : (N,) arrays
    """
    # Local thickness estimate: difference of SDFs
    # Where both are negative (inside from both sides), thickness ~ |f_hw| + f_fw
    local_thickness = fw_field_values - hw_field_values

    too_thin = local_thickness < min_thickness

    if not np.any(too_thin):
        return hw_field_values.copy(), fw_field_values.copy()

    hw_adj = hw_field_values.copy()
    fw_adj = fw_field_values.copy()

    # At thin points, push apart symmetrically
    deficit = min_thickness - local_thickness[too_thin]
    hw_adj[too_thin] -= deficit / 2.0
    fw_adj[too_thin] += deficit / 2.0

    n_adjusted = int(too_thin.sum())
    logger.info(
        "Minimum thickness enforcement: adjusted %d points (%.1f%%)",
        n_adjusted, 100.0 * n_adjusted / len(hw_field_values),
    )

    return hw_adj, fw_adj


# =====================================================================
# Internal: build single SDF from contact points
# =====================================================================

def _build_sdf(
    contact_coords: np.ndarray,
    contact_normals: np.ndarray,
    kernel_type: str,
    alpha: float,
    range_: float,
    nugget: float,
    accuracy: float,
    drift_type: str = "constant",
    R: Optional[np.ndarray] = None,
    S: Optional[np.ndarray] = None,
    constraint_method: str = "gradient",
    offset_distance: float = 2.0,
) -> Callable:
    """Build a signed distance field from contact points and normals.

    Returns an evaluation function (B, 3) -> (B,).
    """
    value_coords, value_data, gradient_coords, gradient_normals = \
        construct_sdf_constraints(
            contact_coords, contact_normals,
            method=constraint_method,
            offset_distance=offset_distance,
        )

    N_v_total = len(value_coords)
    N_g_total = len(gradient_coords) if gradient_coords.size > 0 else 0
    gradient_values = np.ones(N_g_total, dtype=np.float64) if N_g_total > 0 else np.empty(0)

    if (N_v_total + N_g_total) > PUM_THRESHOLD:
        logger.info(
            "Vein SDF PUM mode: %d value + %d gradient constraints",
            N_v_total, N_g_total,
        )
        pum_model = solve_augmented_system_pum(
            value_coords, value_data,
            gradient_coords, gradient_normals, gradient_values,
            kernel_type=kernel_type, alpha=alpha, range_=range_,
            nugget=nugget, accuracy=accuracy, drift_type=drift_type,
            R=R, S=S,
        )
        return make_evaluate_fn_pum(pum_model)

    K_aug, N_v, N_g = assemble_augmented_matrix(
        value_coords, gradient_coords, gradient_normals,
        kernel_type=kernel_type,
        alpha=alpha,
        range_=range_,
        nugget=nugget,
        accuracy=accuracy,
        drift_type=drift_type,
        R=R, S=S,
    )

    value_weights, gradient_weights, poly_coeffs = solve_augmented_system(
        K_aug, value_data, gradient_values, N_v, N_g,
    )

    return make_evaluate_fn(
        value_coords, gradient_coords, gradient_normals,
        value_weights, gradient_weights, poly_coeffs,
        kernel_type, alpha, range_,
        R, S, drift_type,
    )


# =====================================================================
# Internal: closed vein solid generation (caps open edges)
# =====================================================================

def _cap_vein_solid(
    hw_field: np.ndarray,
    fw_field: np.ndarray,
    grid_origin,
    grid_spacing,
    grid_dims,
) -> Optional[Dict[str, Any]]:
    """Generate a closed, capped vein solid from HW and FW scalar fields.

    The vein interior is the set of points where:

        max(f_hw(x), -f_fw(x)) < 0

    i.e., inside the HW isosurface (f_hw < 0) AND above the FW isosurface
    (f_fw > 0).  Extracting this region with ``clip_scalar`` uses linear
    interpolation at the zero-crossing — the resulting mesh is analytically
    smooth and fully closed (no open boundary edges).

    Without this, the HW and FW sheets are extracted as two infinite open
    surfaces that extend to the bounding box edges.  The solid returned here
    is the actual closed vein volume suitable for reserve calculation and
    solid export.

    Parameters
    ----------
    hw_field : np.ndarray (nx, ny, nz)
        HW scalar field — zero at hanging wall surface, negative inside vein.
    fw_field : np.ndarray (nx, ny, nz)
        FW scalar field — zero at footwall surface, positive inside vein.
    grid_origin, grid_spacing : list/array of 3 floats
    grid_dims : (nx, ny, nz) tuple

    Returns
    -------
    dict with {unit_name, name, vertices, faces, volume_m3} or None
    """
    try:
        import pyvista as pv
    except ImportError:
        logger.warning("PyVista not available — cannot generate closed vein solid")
        return None

    try:
        nx, ny, nz = grid_dims
        origin  = np.asarray(grid_origin,  dtype=np.float64)
        spacing = np.asarray(grid_spacing, dtype=np.float64)

        # Boolean vein field: negative = inside vein, positive = outside
        vein_field = np.maximum(hw_field, -fw_field)

        sf_grid = pv.ImageData(
            dimensions=(nx, ny, nz),
            spacing=tuple(spacing),
            origin=tuple(origin),
        )
        sf_grid.point_data["vein_sdf"] = vein_field.ravel(order="F").astype(np.float64)

        # Keep cells where vein_sdf < 0 (vein interior)
        vein_solid = sf_grid.clip_scalar(scalars="vein_sdf", value=0.0, invert=True)
        if vein_solid is None or vein_solid.n_cells == 0:
            logger.debug("Vein interior is empty within the grid — vein may not exist here")
            return None

        surface = vein_solid.extract_surface().clean()
        if surface.n_points == 0:
            return None

        verts = np.asarray(surface.points, dtype=np.float64)
        raw_faces = np.asarray(surface.faces)
        if len(raw_faces) > 0 and raw_faces[0] == 3:
            faces = raw_faces.reshape(-1, 4)[:, 1:4]
        else:
            faces = raw_faces

        volume_m3 = (
            float(vein_solid.volume)
            if hasattr(vein_solid, "volume")
            else vein_solid.n_cells * float(np.prod(spacing))
        )

        logger.info(
            "Vein solid: %d vertices, %d faces, volume=%.1f m³",
            len(verts), len(faces), volume_m3,
        )
        return {
            "unit_name": "vein_solid",
            "name": "vein_solid",
            "vertices": verts,
            "faces": faces,
            "volume_m3": volume_m3,
        }

    except Exception as exc:
        logger.warning("Vein solid capping failed: %s", exc)
        return None


# =====================================================================
# Vein model builder
# =====================================================================

def build_vein_model(
    intersections: List[VeinIntersection],
    kernel_type: str = "spheroidal",
    alpha: float = 1.0,
    range_: float = 100.0,
    nugget: float = 0.0,
    accuracy: float = 1e-6,
    min_thickness: float = 0.5,
    method: str = "median_thickness",
    R: Optional[np.ndarray] = None,
    S: Optional[np.ndarray] = None,
    orientations_df: Optional[pd.DataFrame] = None,
    grid_resolution: float = 10.0,
    grid_extent: Any = "auto",
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> Dict[str, Any]:
    """Build a vein model from drillhole intersections.

    Parameters
    ----------
    intersections : list of VeinIntersection
    kernel_type, alpha, range_, nugget, accuracy : interpolation params
    min_thickness : float
        Minimum vein thickness to enforce (metres).
    method : str
        "median_thickness" (recommended), "separate_hw_fw", or "boolean"
    R, S : rotation/scale matrices for anisotropy
    orientations_df : optional orientation constraints
    grid_resolution : float (metres)
    grid_extent : dict or "auto"
    progress_callback : optional

    Returns
    -------
    dict with:
        hw_surface, fw_surface : mesh
        median_surface : mesh (if median_thickness method)
        hw_evaluate_fn, fw_evaluate_fn : callable
        vein_sdf : callable (combined signed distance function)
        thickness_stats : dict
        grid_origin, grid_spacing, grid_dims
    """
    if not intersections:
        raise ValueError("No vein intersections provided")

    def _progress(pct, msg):
        if progress_callback:
            progress_callback(pct, msg)

    hw_coords = np.array([vi.hw_point for vi in intersections], dtype=np.float64)
    fw_coords = np.array([vi.fw_point for vi in intersections], dtype=np.float64)
    mid_coords = np.array([vi.midpoint for vi in intersections], dtype=np.float64)
    normals = np.array([vi.normal for vi in intersections], dtype=np.float64)
    thicknesses = np.array([vi.thickness for vi in intersections], dtype=np.float64)

    # Add orientation constraints if provided
    orient_normals = None
    orient_coords = None
    if orientations_df is not None and len(orientations_df) > 0:
        orient_list = orientations_dataframe_to_list(orientations_df)
        orient_coords = np.array(
            [[o.x, o.y, o.z] for o in orient_list], dtype=np.float64,
        )
        orient_normals = np.array(
            [o.normal for o in orient_list], dtype=np.float64,
        )

    # Compute grid extent
    all_coords = np.vstack([hw_coords, fw_coords])
    if isinstance(grid_extent, dict):
        xmin, xmax = grid_extent["xmin"], grid_extent["xmax"]
        ymin, ymax = grid_extent["ymin"], grid_extent["ymax"]
        zmin, zmax = grid_extent["zmin"], grid_extent["zmax"]
    else:
        mins = all_coords.min(axis=0)
        maxs = all_coords.max(axis=0)
        extent = maxs - mins
        padding = np.maximum(extent * 0.3, grid_resolution * 3)
        xmin, ymin, zmin = mins - padding
        xmax, ymax, zmax = maxs + padding

    res = grid_resolution
    nx = max(2, int(np.ceil((xmax - xmin) / res)) + 1)
    ny = max(2, int(np.ceil((ymax - ymin) / res)) + 1)
    nz = max(2, int(np.ceil((zmax - zmin) / res)) + 1)
    grid_origin = np.array([xmin, ymin, zmin], dtype=np.float64)
    grid_spacing = np.array([res, res, res], dtype=np.float64)
    grid_dims = (nx, ny, nz)

    if method == "median_thickness":
        return _build_median_thickness(
            intersections, hw_coords, fw_coords, mid_coords, normals,
            thicknesses, kernel_type, alpha, range_, nugget, accuracy,
            min_thickness, R, S, orient_coords, orient_normals,
            grid_origin, grid_spacing, grid_dims, _progress,
        )
    elif method in ("separate_hw_fw", "boolean"):
        return _build_separate_hw_fw(
            hw_coords, fw_coords, normals,
            kernel_type, alpha, range_, nugget, accuracy,
            min_thickness, method == "boolean", R, S,
            orient_coords, orient_normals,
            grid_origin, grid_spacing, grid_dims, _progress,
        )
    else:
        raise ValueError(f"Unknown vein method: '{method}'")


def _build_median_thickness(
    intersections, hw_coords, fw_coords, mid_coords, normals,
    thicknesses, kernel_type, alpha, range_, nugget, accuracy,
    min_thickness, R, S, orient_coords, orient_normals,
    grid_origin, grid_spacing, grid_dims, progress_fn,
) -> Dict[str, Any]:
    """Median surface + thickness field method."""

    progress_fn(5, "Building median surface SDF")

    # Step 1: Build median surface SDF from midpoints
    median_evaluate_fn = _build_sdf(
        mid_coords, normals,
        kernel_type, alpha, range_, nugget, accuracy,
        R=R, S=S,
    )

    progress_fn(25, "Building HW surface")

    # Step 2: Build HW and FW surfaces
    # HW contacts: flip normal (points into hanging wall)
    hw_normals = normals.copy()
    hw_evaluate_fn = _build_sdf(
        hw_coords, hw_normals,
        kernel_type, alpha, range_, nugget, accuracy,
        R=R, S=S,
    )

    progress_fn(40, "Building FW surface")

    fw_normals = -normals.copy()  # FW normal points opposite
    fw_evaluate_fn = _build_sdf(
        fw_coords, fw_normals,
        kernel_type, alpha, range_, nugget, accuracy,
        R=R, S=S,
    )

    progress_fn(55, "Evaluating fields on grid")

    # Step 3: Evaluate on grid
    hw_field = evaluate_field_on_grid(
        grid_origin, grid_spacing, grid_dims, hw_evaluate_fn,
    )
    fw_field = evaluate_field_on_grid(
        grid_origin, grid_spacing, grid_dims, fw_evaluate_fn,
    )

    # Step 4: Enforce minimum thickness
    if min_thickness > 0:
        hw_flat = hw_field.ravel()
        fw_flat = fw_field.ravel()
        hw_flat, fw_flat = enforce_minimum_thickness(
            hw_flat, fw_flat, min_thickness, grid_spacing,
        )
        hw_field = hw_flat.reshape(grid_dims)
        fw_field = fw_flat.reshape(grid_dims)

    progress_fn(70, "Extracting surfaces")

    # Step 5: Extract meshes
    surfaces = {}
    for name, sf in [("hw_surface", hw_field), ("fw_surface", fw_field)]:
        try:
            verts, faces = extract_isosurface(sf, grid_origin, grid_spacing, 0.0)
            verts, faces = cleanup_mesh(verts, faces)
            if verts.shape[0] > 0:
                try:
                    surfaces[name] = field_to_pyvista_mesh(verts, faces, name)
                except ImportError:
                    surfaces[name] = {"vertices": verts, "faces": faces}
        except Exception as e:
            logger.warning("Failed to extract %s: %s", name, e)

    # Median surface
    median_field = evaluate_field_on_grid(
        grid_origin, grid_spacing, grid_dims, median_evaluate_fn,
    )
    try:
        verts, faces = extract_isosurface(median_field, grid_origin, grid_spacing, 0.0)
        verts, faces = cleanup_mesh(verts, faces)
        if verts.shape[0] > 0:
            try:
                surfaces["median_surface"] = field_to_pyvista_mesh(verts, faces, "median")
            except ImportError:
                surfaces["median_surface"] = {"vertices": verts, "faces": faces}
    except Exception:
        pass

    # Step 6: Closed vein solid (capped — no open boundary edges)
    vein_solid = _cap_vein_solid(hw_field, fw_field, grid_origin, grid_spacing, grid_dims)
    if vein_solid is not None:
        surfaces["vein_solid"] = vein_solid

    # Step 7: Combined vein SDF: vein = {x : f_hw <= 0 AND f_fw >= 0}
    # As a single SDF: f_vein(x) = max(f_hw(x), -f_fw(x))
    def vein_sdf(pts):
        fhw = hw_evaluate_fn(pts)
        ffw = fw_evaluate_fn(pts)
        return np.maximum(fhw, -ffw)

    progress_fn(90, "Computing thickness statistics")

    thickness_stats = {
        "mean": float(np.mean(thicknesses)),
        "median": float(np.median(thicknesses)),
        "min": float(np.min(thicknesses)),
        "max": float(np.max(thicknesses)),
        "std": float(np.std(thicknesses)),
        "n_intersections": len(intersections),
    }

    progress_fn(100, "Vein model complete")

    return {
        **surfaces,
        "hw_evaluate_fn": hw_evaluate_fn,
        "fw_evaluate_fn": fw_evaluate_fn,
        "median_evaluate_fn": median_evaluate_fn,
        "vein_sdf": vein_sdf,
        "thickness_stats": thickness_stats,
        "grid_origin": grid_origin,
        "grid_spacing": grid_spacing,
        "grid_dims": grid_dims,
    }


def _build_separate_hw_fw(
    hw_coords, fw_coords, normals,
    kernel_type, alpha, range_, nugget, accuracy,
    min_thickness, use_boolean, R, S,
    orient_coords, orient_normals,
    grid_origin, grid_spacing, grid_dims, progress_fn,
) -> Dict[str, Any]:
    """Separate HW/FW interpolation method."""

    progress_fn(10, "Building HW surface")

    hw_evaluate_fn = _build_sdf(
        hw_coords, normals,
        kernel_type, alpha, range_, nugget, accuracy,
        R=R, S=S,
    )

    progress_fn(30, "Building FW surface")

    fw_evaluate_fn = _build_sdf(
        fw_coords, -normals,
        kernel_type, alpha, range_, nugget, accuracy,
        R=R, S=S,
    )

    progress_fn(50, "Evaluating fields on grid")

    hw_field = evaluate_field_on_grid(
        grid_origin, grid_spacing, grid_dims, hw_evaluate_fn,
    )
    fw_field = evaluate_field_on_grid(
        grid_origin, grid_spacing, grid_dims, fw_evaluate_fn,
    )

    # Check for crossing
    local_thickness = fw_field - hw_field
    n_crossing = int(np.sum(local_thickness < 0))
    if n_crossing > 0:
        pct = 100.0 * n_crossing / local_thickness.size
        logger.warning(
            "HW/FW surfaces cross at %d grid points (%.1f%%). "
            "Consider using median_thickness method.",
            n_crossing, pct,
        )

    if min_thickness > 0:
        hw_flat = hw_field.ravel()
        fw_flat = fw_field.ravel()
        hw_flat, fw_flat = enforce_minimum_thickness(
            hw_flat, fw_flat, min_thickness, grid_spacing,
        )
        hw_field = hw_flat.reshape(grid_dims)
        fw_field = fw_flat.reshape(grid_dims)

    progress_fn(70, "Extracting surfaces")

    surfaces = {}
    for name, sf in [("hw_surface", hw_field), ("fw_surface", fw_field)]:
        try:
            verts, faces = extract_isosurface(sf, grid_origin, grid_spacing, 0.0)
            verts, faces = cleanup_mesh(verts, faces)
            if verts.shape[0] > 0:
                try:
                    surfaces[name] = field_to_pyvista_mesh(verts, faces, name)
                except ImportError:
                    surfaces[name] = {"vertices": verts, "faces": faces}
        except Exception as e:
            logger.warning("Failed to extract %s: %s", name, e)

    # Closed vein solid (capped — no open boundary edges)
    vein_solid = _cap_vein_solid(hw_field, fw_field, grid_origin, grid_spacing, grid_dims)
    if vein_solid is not None:
        surfaces["vein_solid"] = vein_solid

    # Combined vein SDF
    if use_boolean:
        # Eq. 8.1: f_vein(x) = max(f_hw(x), -f_fw(x))
        # Negative inside vein, positive outside
        def vein_sdf(pts):
            fhw = hw_evaluate_fn(pts)
            ffw = fw_evaluate_fn(pts)
            return np.maximum(fhw, -ffw)
    else:
        # Non-boolean: signed distance to median surface (average of HW/FW)
        # Negative below median, positive above — preserves thickness info
        def vein_sdf(pts):
            fhw = hw_evaluate_fn(pts)
            ffw = fw_evaluate_fn(pts)
            return (fhw + ffw) / 2.0

    progress_fn(100, "Vein model complete")

    return {
        **surfaces,
        "hw_evaluate_fn": hw_evaluate_fn,
        "fw_evaluate_fn": fw_evaluate_fn,
        "vein_sdf": vein_sdf,
        "grid_origin": grid_origin,
        "grid_spacing": grid_spacing,
        "grid_dims": grid_dims,
    }
