"""
Stratigraphic Model -- Potential Field for Conformable Sequences.
================================================================

Models a conformable stratigraphic sequence as isosurfaces of a single
scalar potential field.  All surfaces within a conformable package share
the same interpolation, guaranteeing they never cross (Theorem 5.1).

Known Limitation (v1.0): Unconformities not supported.
  The single-field approach models only conformable sequences.  When
  unconformable contacts are detected, a warning is logged.  To model
  multiple geological events (e.g., flat cover over folded basement),
  build each conformable sequence as a separate model run, then combine
  the domain assignments manually.  Full multi-field chronological
  modelling is planned for v2.0.

Primary modelling mode for BIF iron ore, coal, manganese, and any
layered deposit.

Mathematical basis: Eq. 5.1 of the GeoX Math Specification.

Usage::

    column = StratigraphicModelColumn.from_contacts(contacts_df, unit_order)
    value_coords, value_data, grad_coords, grad_normals = \
        build_potential_field_constraints(contacts_df, column, orientations_df)

    # Feed into Phase 1 augmented kernel system:
    K_aug, N_v, N_g = assemble_augmented_matrix(value_coords, grad_coords, ...)
    ...

    surfaces = extract_stratigraphic_surfaces(evaluate_fn, column, ...)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .contact_data import (
    StratigraphicColumn,
    dip_azimuth_to_normal,
    orientations_dataframe_to_list,
)
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
from .validation import check_contact_honouring, contact_honouring_summary

logger = logging.getLogger(__name__)


# =====================================================================
# Data classes
# =====================================================================

@dataclass
class StratigraphicUnit:
    """One unit in the stratigraphic column with isovalue."""
    name: str
    isovalue: float          # cumulative thickness from reference surface
    colour: str = "#808080"  # hex colour for display
    contact_type: str = "conformable"
    # "conformable", "unconformable_erosion", "unconformable_onlap", "gradational"


@dataclass
class StratigraphicModelColumn:
    """Ordered sequence of geological units with computed isovalues.

    The reference surface (isovalue=0) divides the column.  Units above
    have negative isovalues (potential field decreases upward in the
    standard convention), units below have positive isovalues.
    """
    units: List[StratigraphicUnit] = field(default_factory=list)
    reference_surface: str = ""  # name of the surface assigned isovalue=0

    @property
    def n_units(self) -> int:
        return len(self.units)

    @property
    def n_surfaces(self) -> int:
        return max(0, len(self.units) - 1)

    @property
    def isovalues(self) -> List[float]:
        """Sorted isovalues for all inter-unit surfaces."""
        if self.n_surfaces == 0:
            return []
        # Surface i is between units[i] and units[i+1].
        # Its isovalue is the average of the two units' isovalues.
        isos = []
        for i in range(self.n_surfaces):
            iso = (self.units[i].isovalue + self.units[i + 1].isovalue) / 2.0
            isos.append(iso)
        return sorted(isos)

    @property
    def unit_names(self) -> List[str]:
        return [u.name for u in self.units]

    def surface_name(self, index: int) -> str:
        """Name of the i-th boundary surface."""
        if index < 0 or index >= self.n_surfaces:
            raise IndexError(f"Surface index {index} out of range")
        return f"{self.units[index].name}_{self.units[index + 1].name}"

    def isovalue_for_surface(self, surface_name: str) -> Optional[float]:
        """Look up the isovalue for a named surface."""
        for i in range(self.n_surfaces):
            if self.surface_name(i) == surface_name:
                return (self.units[i].isovalue + self.units[i + 1].isovalue) / 2.0
        return None

    @classmethod
    def from_contacts(
        cls,
        contacts_df: pd.DataFrame,
        unit_order: List[str],
        reference_index: int = 0,
        default_thickness: float = 20.0,
    ) -> "StratigraphicModelColumn":
        """Build a stratigraphic column from contact data.

        Computes average thickness of each unit from drillhole
        intersections and assigns cumulative isovalues.

        Parameters
        ----------
        contacts_df : pd.DataFrame
            Contacts with columns: hole_id, depth, X, Y, Z,
            unit_above, unit_below, surface_name.
        unit_order : list of str
            Units from top (youngest) to bottom (oldest).
        reference_index : int
            Index of the reference surface (isovalue=0).
        default_thickness : float
            Thickness to use when no data available for a unit.

        Returns
        -------
        StratigraphicModelColumn with computed isovalues.
        """
        # Compute average thickness per unit from drillhole contacts
        thicknesses = {}
        for i in range(len(unit_order) - 1):
            top_unit = unit_order[i]
            bot_unit = unit_order[i + 1]
            sname = f"{top_unit}_{bot_unit}"

            # Find contacts for this surface
            mask = contacts_df["surface_name"] == sname
            if not mask.any():
                # Try reverse naming
                sname_rev = f"{bot_unit}_{top_unit}"
                mask = contacts_df["surface_name"] == sname_rev

            if mask.any():
                # For each drillhole, compute vertical thickness
                surface_contacts = contacts_df[mask]
                hole_groups = surface_contacts.groupby("hole_id") if "hole_id" in surface_contacts.columns else [(None, surface_contacts)]

                hole_thicknesses = []
                for _, grp in (surface_contacts.groupby("hole_id") if "hole_id" in surface_contacts.columns else [(None, surface_contacts)]):
                    # Find adjacent surface contacts in same hole
                    if i > 0:
                        prev_sname = f"{unit_order[i - 1]}_{unit_order[i]}"
                        prev_mask = contacts_df["surface_name"] == prev_sname
                        if "hole_id" in contacts_df.columns and "hole_id" in grp.columns:
                            hids = grp["hole_id"].unique()
                            for hid in hids:
                                prev_in_hole = contacts_df[prev_mask & (contacts_df["hole_id"] == hid)]
                                curr_in_hole = grp[grp["hole_id"] == hid]
                                if not prev_in_hole.empty and not curr_in_hole.empty:
                                    dz = abs(
                                        prev_in_hole["Z"].iloc[0]
                                        - curr_in_hole["Z"].iloc[0]
                                    )
                                    if dz > 0:
                                        hole_thicknesses.append(dz)

                if hole_thicknesses:
                    thicknesses[top_unit] = float(np.mean(hole_thicknesses))
                else:
                    thicknesses[top_unit] = default_thickness
            else:
                thicknesses[top_unit] = default_thickness

        # Last unit has no bottom contact -- use default
        thicknesses[unit_order[-1]] = default_thickness

        # Build cumulative isovalues from reference surface
        # Convention: potential field increases downward
        # Reference surface at index reference_index has isovalue = 0
        # Units above reference: negative isovalues (cumulative thickness upward)
        # Units below reference: positive isovalues (cumulative thickness downward)
        units_out = []
        cumulative = 0.0

        for i, name in enumerate(unit_order):
            if i < reference_index:
                # Above reference: compute distance upward
                dist = 0.0
                for j in range(i, reference_index):
                    dist += thicknesses.get(unit_order[j], default_thickness)
                iso = -dist
            elif i == reference_index:
                iso = 0.0
            else:
                # Below reference: compute distance downward
                dist = 0.0
                for j in range(reference_index, i):
                    dist += thicknesses.get(unit_order[j], default_thickness)
                iso = dist

            units_out.append(StratigraphicUnit(
                name=name,
                isovalue=iso,
                contact_type="conformable",
            ))
            cumulative = iso

        ref_sname = ""
        if reference_index < len(unit_order) - 1:
            ref_sname = f"{unit_order[reference_index]}_{unit_order[reference_index + 1]}"

        return cls(units=units_out, reference_surface=ref_sname)


# =====================================================================
# Surface Chronology data model (Leapfrog-inspired)
# =====================================================================

@dataclass
class SurfaceChronologyEntry:
    """One surface in the chronological order (youngest → oldest)."""
    name: str
    surface_type: str = "deposit"
    # "deposit" | "erosion" | "intrusion" | "vein" | "fault"
    unit_younger: str = ""   # lithology on younger / upper / inside side
    unit_older: str = ""     # lithology on older / lower / outside side
    enabled: bool = True     # toggle without rebuild
    n_contacts: int = 0      # number of constraining drillhole contacts
    isovalue: float = 0.0    # computed during build


@dataclass
class SurfaceChronology:
    """Complete surface chronology for a geological model (youngest first)."""
    entries: List[SurfaceChronologyEntry] = field(default_factory=list)
    background_lithology: str = "Unknown"

    def enabled_entries(self) -> List[SurfaceChronologyEntry]:
        return [e for e in self.entries if e.enabled]

    def reorder(self, new_order: List[int]) -> None:
        self.entries = [self.entries[i] for i in new_order]


# =====================================================================
# Constraint construction
# =====================================================================

def _make_tangent_pair(n: np.ndarray):
    """Two unit tangent vectors orthogonal to unit normal n.

    Used to build scale-invariant gradient constraints ∇f × n = 0:
    instead of constraining ∂f/∂n = 1 (fixed magnitude), we constrain
    ∂f/∂t1 = 0 and ∂f/∂t2 = 0 (direction-only).  The magnitude is
    then determined entirely by the value constraints (isovalues).
    """
    ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    t1 = np.cross(n, ref)
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(n, t1)
    t2 /= np.linalg.norm(t2)
    return t1, t2


def build_potential_field_constraints(
    contacts_df: pd.DataFrame,
    strat_column: StratigraphicModelColumn,
    orientations_df: Optional[pd.DataFrame] = None,
    x_col: str = "X",
    y_col: str = "Y",
    z_col: str = "Z",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert stratigraphic contacts into potential field constraints.

    Each contact point becomes a VALUE constraint:
      f(x_contact) = isovalue of that surface

    Each orientation measurement becomes a GRADIENT constraint:
      grad f(x_orientation) . n = 1

    This feeds directly into the Phase 1 augmented kernel matrix system.

    Parameters
    ----------
    contacts_df : pd.DataFrame
        Contact points with surface_name column matching strat_column.
    strat_column : StratigraphicModelColumn
        Defines which isovalue each surface has.
    orientations_df : pd.DataFrame, optional
        Structural measurements with columns: X, Y, Z, dip, azimuth.
    x_col, y_col, z_col : str
        Coordinate column names.

    Returns
    -------
    value_coords : np.ndarray (N_v, 3)
    value_data : np.ndarray (N_v,) -- isovalues at each contact
    gradient_coords : np.ndarray (N_g, 3)
    gradient_normals : np.ndarray (N_g, 3)
    """
    value_coords_list = []
    value_data_list = []

    for i in range(strat_column.n_surfaces):
        sname = strat_column.surface_name(i)
        iso = (strat_column.units[i].isovalue + strat_column.units[i + 1].isovalue) / 2.0

        # Find contacts for this surface
        mask = contacts_df["surface_name"] == sname
        if not mask.any():
            # Try the reverse name
            parts = sname.split("_", 1)
            if len(parts) == 2:
                sname_rev = f"{parts[1]}_{parts[0]}"
                mask = contacts_df["surface_name"] == sname_rev

        if mask.any():
            coords = contacts_df.loc[mask, [x_col, y_col, z_col]].values.astype(np.float64)
            value_coords_list.append(coords)
            value_data_list.append(np.full(len(coords), iso, dtype=np.float64))

    if not value_coords_list:
        raise ValueError(
            "No contacts matched any surface in the stratigraphic column. "
            f"Column surfaces: {[strat_column.surface_name(i) for i in range(strat_column.n_surfaces)]}, "
            f"Contact surface_names: {contacts_df['surface_name'].unique().tolist() if 'surface_name' in contacts_df.columns else '(missing column)'}"
        )

    value_coords = np.vstack(value_coords_list)
    value_data = np.concatenate(value_data_list)

    # Gradient constraints from orientation data
    if orientations_df is not None and len(orientations_df) > 0:
        col_lc = {c.lower(): c for c in orientations_df.columns}
        _has_xyz = all(k in col_lc for k in ("x", "y", "z"))
        _has_nxyz = all(k in col_lc for k in ("normal_x", "normal_y", "normal_z"))
        _has_dip_az = "dip" in col_lc and ("azimuth" in col_lc or "dip_direction" in col_lc)

        if _has_xyz and _has_nxyz:
            # Import format: normal_x/y/z already computed during CSV import
            gradient_coords = orientations_df[
                [col_lc["x"], col_lc["y"], col_lc["z"]]
            ].values.astype(np.float64)
            gradient_normals = orientations_df[
                [col_lc["normal_x"], col_lc["normal_y"], col_lc["normal_z"]]
            ].values.astype(np.float64)
            # Normalise in case of floating-point drift
            norms = np.linalg.norm(gradient_normals, axis=1, keepdims=True)
            norms = np.where(norms < 1e-12, 1.0, norms)
            gradient_normals = gradient_normals / norms
            logger.info(
                "Gradient constraints from structural measurements: %d points (normal_x/y/z format)",
                len(gradient_coords),
            )
            # Expand to tangent pairs
            t1_list, t2_list = [], []
            for n_row in gradient_normals:
                t1, t2 = _make_tangent_pair(n_row)
                t1_list.append(t1)
                t2_list.append(t2)
            gradient_coords = np.vstack([gradient_coords, gradient_coords])
            gradient_normals = np.vstack([
                np.array(t1_list, dtype=np.float64),
                np.array(t2_list, dtype=np.float64),
            ])
        elif _has_xyz and _has_dip_az:
            # Standard dip/azimuth format — pass actual (possibly uppercase) column names
            az_col_name = col_lc.get("azimuth") or col_lc.get("dip_direction")
            orient_list = orientations_dataframe_to_list(
                orientations_df,
                x_col=col_lc["x"],
                y_col=col_lc["y"],
                z_col=col_lc["z"],
                dip_col=col_lc["dip"],
                azimuth_col=az_col_name,
            )
            gradient_coords = np.array(
                [[o.x, o.y, o.z] for o in orient_list], dtype=np.float64,
            )
            gradient_normals = np.array(
                [o.normal for o in orient_list], dtype=np.float64,
            )
            logger.info(
                "Gradient constraints from structural measurements: %d points (dip/azimuth format)",
                len(gradient_coords),
            )
            # Expand to tangent pairs
            t1_list, t2_list = [], []
            for n_row in gradient_normals:
                t1, t2 = _make_tangent_pair(n_row)
                t1_list.append(t1)
                t2_list.append(t2)
            gradient_coords = np.vstack([gradient_coords, gradient_coords])
            gradient_normals = np.vstack([
                np.array(t1_list, dtype=np.float64),
                np.array(t2_list, dtype=np.float64),
            ])
        else:
            logger.warning(
                "Orientations DataFrame has no usable format "
                "(need X/Y/Z + normal_x/y/z OR dip/azimuth). "
                "Columns present: %s — falling through to PCA.",
                list(orientations_df.columns),
            )
            # Fall through to PCA branch by pretending orientations_df is None
            orientations_df = None

    if orientations_df is None or len(orientations_df) == 0:
        # No structural measurements — derive gradient constraints from contacts.
        # Priority:
        #   1. Survey method: contacts have hole_dir_x/y/z from desurvey → use
        #      weighted PCA augmented with drillhole direction vectors.
        #   2. PCA fallback: use positional PCA only (less accurate for flat beds).
        _has_dirs = all(c in contacts_df.columns
                        for c in ("hole_dir_x", "hole_dir_y", "hole_dir_z"))

        if _has_dirs:
            from .signed_distance import estimate_surface_normal_from_contacts_and_surveys
            logger.info(
                "build_potential_field_constraints: using survey-augmented normals "
                "(hole_dir_x/y/z present)"
            )

        grad_coord_list: list = []
        grad_normal_list: list = []

        for i in range(strat_column.n_surfaces):
            sname = strat_column.surface_name(i)
            mask = contacts_df["surface_name"] == sname
            if not mask.any():
                parts = sname.split("_", 1)
                if len(parts) == 2:
                    mask = contacts_df["surface_name"] == f"{parts[1]}_{parts[0]}"

            if not mask.any():
                continue

            s_coords = contacts_df.loc[mask, [x_col, y_col, z_col]].values.astype(
                np.float64
            )

            if len(s_coords) < 2:
                # Single contact — vertical normal as last resort
                grad_coord_list.append(s_coords)
                grad_normal_list.append(
                    np.tile([0.0, 0.0, 1.0], (len(s_coords), 1))
                )
                continue

            if _has_dirs:
                # Survey-augmented: use drillhole direction vectors stored in contacts
                s_dirs = contacts_df.loc[
                    mask, ["hole_dir_x", "hole_dir_y", "hole_dir_z"]
                ].values.astype(np.float64)
                normal = estimate_surface_normal_from_contacts_and_surveys(
                    s_coords, s_dirs
                )
                logger.info(
                    "Surface '%s': survey-augmented normal=[%.3f, %.3f, %.3f] "
                    "from %d contacts (dip=%.1f° from horiz)",
                    sname, normal[0], normal[1], normal[2], len(s_coords),
                    float(np.degrees(np.arccos(np.clip(abs(normal[2]), 0.0, 1.0)))),
                )
            elif len(s_coords) >= 3:
                # PCA fallback — eigh returns eigenvalues in ascending order
                centroid = s_coords.mean(axis=0)
                centered = s_coords - centroid
                cov = centered.T @ centered / max(1, len(centered) - 1)
                eigenvalues, eigenvectors = np.linalg.eigh(cov)
                normal = eigenvectors[:, 0].copy()
                norm_len = np.linalg.norm(normal)
                if norm_len < 1e-12:
                    normal = np.array([0.0, 0.0, 1.0])
                else:
                    normal /= norm_len
                if normal[2] < 0.0:
                    normal = -normal
                logger.info(
                    "Surface '%s': PCA normal=[%.3f, %.3f, %.3f] from %d contacts",
                    sname, normal[0], normal[1], normal[2], len(s_coords),
                )
            else:
                normal = np.array([0.0, 0.0, 1.0])

            # Subsample to ≤20 gradient constraints per surface to keep
            # the kernel matrix from growing too large
            n_grad = min(len(s_coords), 20)
            if len(s_coords) > n_grad:
                idx = np.round(
                    np.linspace(0, len(s_coords) - 1, n_grad)
                ).astype(int)
                g_coords = s_coords[idx]
            else:
                g_coords = s_coords

            grad_coord_list.append(g_coords)
            grad_normal_list.append(np.tile(normal, (len(g_coords), 1)))

        if grad_coord_list:
            raw_coords = np.vstack(grad_coord_list)
            raw_normals = np.vstack(grad_normal_list)
            # Expand each normal into two tangent constraints (scale-invariant)
            # ∂f/∂t = 0 enforces gradient DIRECTION without fixing its magnitude.
            t1_list, t2_list = [], []
            for n_row in raw_normals:
                t1, t2 = _make_tangent_pair(n_row)
                t1_list.append(t1)
                t2_list.append(t2)
            gradient_coords = np.vstack([raw_coords, raw_coords])
            gradient_normals = np.vstack([
                np.array(t1_list, dtype=np.float64),
                np.array(t2_list, dtype=np.float64),
            ])
        else:
            gradient_coords = np.empty((0, 3), dtype=np.float64)
            gradient_normals = np.empty((0, 3), dtype=np.float64)

    logger.info(
        "Potential field constraints: %d value (across %d surfaces), %d gradient",
        len(value_coords), strat_column.n_surfaces, len(gradient_coords),
    )

    return value_coords, value_data, gradient_coords, gradient_normals


# =====================================================================
# Surface extraction
# =====================================================================

def extract_stratigraphic_surfaces(
    evaluate_fn: Callable,
    strat_column: StratigraphicModelColumn,
    grid_origin: np.ndarray,
    grid_spacing: np.ndarray,
    grid_dims: Tuple[int, int, int],
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> Dict[str, Any]:
    """Extract all surfaces from a potential field.

    Evaluates the scalar field on the grid once, then extracts isosurfaces
    at each isovalue defined in the stratigraphic column.

    Parameters
    ----------
    evaluate_fn : callable (B, 3) -> (B,)
    strat_column : StratigraphicModelColumn
    grid_origin, grid_spacing : (3,) arrays
    grid_dims : (nx, ny, nz)
    progress_callback : optional

    Returns
    -------
    dict with:
        surfaces: {surface_name: pyvista.PolyData or {vertices, faces}}
        scalar_field: (nx, ny, nz) array
        isovalues: {surface_name: float}
    """
    if progress_callback:
        progress_callback(5, "Evaluating potential field on grid")

    scalar_field = evaluate_field_on_grid(
        grid_origin, grid_spacing, grid_dims,
        evaluate_fn,
        progress_callback=lambda pct, msg: (
            progress_callback(5 + pct * 60 // 100, msg)
            if progress_callback else None
        ),
    )

    surfaces = {}
    isovalues_map = {}

    for i in range(strat_column.n_surfaces):
        sname = strat_column.surface_name(i)
        iso = (strat_column.units[i].isovalue + strat_column.units[i + 1].isovalue) / 2.0
        isovalues_map[sname] = iso

        if progress_callback:
            progress_callback(
                65 + i * 30 // max(1, strat_column.n_surfaces),
                f"Extracting surface: {sname}",
            )

        try:
            verts, faces = extract_isosurface(
                scalar_field, grid_origin, grid_spacing, iso,
            )
            verts, faces = cleanup_mesh(verts, faces)

            if verts.shape[0] > 0:
                try:
                    mesh = field_to_pyvista_mesh(verts, faces, surface_name=sname)
                    surfaces[sname] = mesh
                except ImportError:
                    surfaces[sname] = {"vertices": verts, "faces": faces}

                logger.info(
                    "Surface '%s' (isovalue=%.2f): %d vertices, %d triangles",
                    sname, iso, verts.shape[0], faces.shape[0],
                )
            else:
                logger.warning(
                    "Surface '%s' (isovalue=%.2f): no vertices extracted",
                    sname, iso,
                )
        except Exception as e:
            logger.warning("Failed to extract surface '%s': %s", sname, e)

    if progress_callback:
        progress_callback(100, "Surface extraction complete")

    return {
        "surfaces": surfaces,
        "scalar_field": scalar_field,
        "isovalues": isovalues_map,
    }


# =====================================================================
# Full stratigraphic model builder
# =====================================================================

def build_stratigraphic_model(
    contacts_df: pd.DataFrame,
    unit_order: List[str],
    kernel_type: str = "spheroidal",
    alpha: float = 1.0,
    range_max: float = 100.0,
    range_mid: float = 100.0,
    range_min: float = 100.0,
    azimuth: float = 0.0,
    dip: float = 0.0,
    pitch: float = 0.0,
    nugget: float = 0.0,
    accuracy: float = 1e-6,
    drift_type: str = "constant",
    grid_resolution: float = 10.0,
    grid_extent: Any = "auto",
    tolerance: float = 1.0,
    orientations_df: Optional[pd.DataFrame] = None,
    reference_index: int = 0,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> Dict[str, Any]:
    """Build a complete stratigraphic model using the potential field method.

    End-to-end: contacts -> stratigraphic column -> potential field
    interpolation -> surface extraction -> validation.

    Parameters
    ----------
    contacts_df : pd.DataFrame
        Contact data with columns: hole_id, depth, X, Y, Z,
        unit_above, unit_below, surface_name.
    unit_order : list of str
        Units from top (youngest) to bottom (oldest).
    kernel_type, alpha, range_max, etc. : interpolation parameters
    grid_resolution : float (metres)
    grid_extent : dict or "auto"
    tolerance : float (contact honouring tolerance)
    orientations_df : optional structural measurements
    reference_index : int, which surface gets isovalue=0
    progress_callback : optional

    Returns
    -------
    dict with:
        strat_column : StratigraphicModelColumn
        surfaces : {name: mesh}
        scalar_field : (nx, ny, nz) array
        evaluate_fn : callable
        contact_misfit : pd.DataFrame
        contact_honouring_summary : dict
        grid_origin, grid_spacing, grid_dims : grid parameters
    """
    from geostats.arbf.utils import rotation_matrix, scale_matrix

    def _progress(pct, msg):
        if progress_callback:
            progress_callback(pct, msg)

    _progress(2, "Building stratigraphic column")

    # Fix 1B: Auto-detect range from data spacing when default 100 m is used.
    # 3x average nearest-neighbour distance prevents contacts from being
    # outside the effective kernel radius.
    contact_coords_for_range = contacts_df[["X", "Y", "Z"]].dropna().values.astype(np.float64)
    if range_max <= 100.0 and len(contact_coords_for_range) >= 3:
        try:
            from scipy.spatial import cKDTree
            _tree = cKDTree(contact_coords_for_range)
            _nn, _ = _tree.query(contact_coords_for_range, k=2)
            _avg_spacing = float(np.median(_nn[:, 1]))
            if _avg_spacing > 0:
                range_max = max(range_max, _avg_spacing * 3.0)
                range_mid = max(range_mid, range_max * 0.7)
                range_min = max(range_min, range_max * 0.5)
                logger.info(
                    "Auto-range from spacing %.1f m → range_max=%.1f m",
                    _avg_spacing, range_max,
                )
        except ImportError:
            pass
        except Exception as _exc:
            logger.debug("Auto-range detection failed: %s", _exc)

    # Step 1: Build stratigraphic column with isovalues
    strat_col = StratigraphicModelColumn.from_contacts(
        contacts_df, unit_order,
        reference_index=reference_index,
    )

    # ── Unconformity detection ──────────────────────────────────────
    # contact_type is set on each StratigraphicUnit but the current
    # build uses a single global potential field (Theorem 5.1 requires
    # conformable relationships).  Warn when unconformable contacts are
    # present so the user knows the results may be unreliable there.
    _unconformable_surfaces = [
        strat_col.surface_name(i)
        for i in range(strat_col.n_surfaces)
        if strat_col.units[i + 1].contact_type not in ("conformable", "gradational")
    ]
    if _unconformable_surfaces:
        logger.warning(
            "Unconformable contacts detected: %s. "
            "The single-field potential model assumes conformable relationships. "
            "Model stratigraphy may be unreliable near these boundaries. "
            "WORKAROUND: split the column into separate conformable sequences "
            "and run each as a separate Geological Model build.",
            ", ".join(_unconformable_surfaces),
        )

    _progress(5, "Constructing potential field constraints")

    # Step 2: Build constraints
    value_coords, value_data, gradient_coords, gradient_normals = \
        build_potential_field_constraints(
            contacts_df, strat_col,
            orientations_df=orientations_df,
        )

    # Step 3: Rotation/scaling matrices
    R = rotation_matrix(azimuth, dip, pitch)
    S = scale_matrix(range_max, range_mid, range_min)

    # Scale-invariant: tangent direction constraints have value 0 (∂f/∂t = 0).
    # The gradient magnitude is determined by the value (isovalue) constraints.
    N_v_total = len(value_coords)
    N_g_total = len(gradient_coords) if gradient_coords.size > 0 else 0
    gradient_values = np.zeros(N_g_total, dtype=np.float64)

    use_pum = (N_v_total + N_g_total) > PUM_THRESHOLD

    if use_pum:
        # ── PUM path: domain decomposition, local solves, Wendland C2 blend ──
        logger.info(
            "PUM mode: %d value + %d gradient constraints exceed threshold %d",
            N_v_total, N_g_total, PUM_THRESHOLD,
        )
        _progress(10, "PUM: decomposing domain into sub-domains")

        pum_model = solve_augmented_system_pum(
            value_coords, value_data,
            gradient_coords, gradient_normals, gradient_values,
            kernel_type=kernel_type,
            alpha=alpha,
            range_=range_max,
            nugget=nugget,
            accuracy=accuracy,
            drift_type=drift_type,
            R=R,
            S=S,
        )

        _progress(35, "PUM: creating blended evaluation function")
        evaluate_fn = make_evaluate_fn_pum(pum_model)

    else:
        # ── Global path: single augmented matrix solve ──
        _progress(10, "Assembling augmented kernel matrix")

        K_aug, N_v, N_g = assemble_augmented_matrix(
            value_coords, gradient_coords, gradient_normals,
            kernel_type=kernel_type,
            alpha=alpha,
            range_=range_max,
            nugget=nugget,
            accuracy=accuracy,
            drift_type=drift_type,
            R=R, S=S,
        )

        _progress(25, "Solving augmented system")

        value_weights, gradient_weights, poly_coeffs = solve_augmented_system(
            K_aug, value_data, gradient_values, N_v, N_g,
        )

        _progress(35, "Creating evaluation function")

        evaluate_fn = make_evaluate_fn(
            value_coords, gradient_coords, gradient_normals,
            value_weights, gradient_weights, poly_coeffs,
            kernel_type, alpha, range_max,
            R, S, drift_type,
        )

    # Step 5: Compute grid
    _progress(40, "Computing grid")
    contact_coords = contacts_df[["X", "Y", "Z"]].values.astype(np.float64)

    if isinstance(grid_extent, dict):
        xmin, xmax = grid_extent["xmin"], grid_extent["xmax"]
        ymin, ymax = grid_extent["ymin"], grid_extent["ymax"]
        zmin, zmax = grid_extent["zmin"], grid_extent["zmax"]
    else:
        mins = contact_coords.min(axis=0)
        maxs = contact_coords.max(axis=0)
        extent = maxs - mins
        padding = np.maximum(extent * 0.2, grid_resolution * 2)
        xmin, ymin, zmin = mins - padding
        xmax, ymax, zmax = maxs + padding

    res = grid_resolution
    nx = max(2, int(np.ceil((xmax - xmin) / res)) + 1)
    ny = max(2, int(np.ceil((ymax - ymin) / res)) + 1)
    nz = max(2, int(np.ceil((zmax - zmin) / res)) + 1)

    grid_origin = np.array([xmin, ymin, zmin], dtype=np.float64)
    grid_spacing = np.array([res, res, res], dtype=np.float64)
    grid_dims = (nx, ny, nz)

    # Step 6: Extract surfaces
    _progress(45, "Extracting stratigraphic surfaces")

    extraction_result = extract_stratigraphic_surfaces(
        evaluate_fn, strat_col,
        grid_origin, grid_spacing, grid_dims,
        progress_callback=lambda pct, msg: _progress(45 + pct * 40 // 100, msg),
    )

    # Step 7: Validate contact honouring per surface.
    # tolerance is in metres; check_contact_honouring converts isovalue misfit
    # to spatial metres via the gradient magnitude at each contact point.
    _progress(90, "Validating contact honouring")

    all_misfit_dfs = []
    for i in range(strat_col.n_surfaces):
        sname = strat_col.surface_name(i)
        iso = (strat_col.units[i].isovalue + strat_col.units[i + 1].isovalue) / 2.0

        mask = contacts_df["surface_name"] == sname
        if not mask.any():
            parts = sname.split("_", 1)
            if len(parts) == 2:
                mask = contacts_df["surface_name"] == f"{parts[1]}_{parts[0]}"

        if mask.any():
            surface_contacts = contacts_df[mask].copy()
            misfit_df = check_contact_honouring(
                surface_contacts, evaluate_fn,
                tolerance=tolerance,   # metres — gradient conversion in validation.py
                expected_value=iso,
            )
            all_misfit_dfs.append(misfit_df)

    contact_misfit = pd.concat(all_misfit_dfs, ignore_index=True) if all_misfit_dfs else pd.DataFrame()
    honouring_summary = contact_honouring_summary(contact_misfit)

    _progress(100, "Stratigraphic model complete")

    return {
        "strat_column": strat_col,
        "surfaces": extraction_result["surfaces"],
        "scalar_field": extraction_result["scalar_field"],
        "evaluate_fn": evaluate_fn,
        "contact_misfit": contact_misfit,
        "contact_honouring_summary": honouring_summary,
        "isovalues": extraction_result["isovalues"],
        "grid_origin": grid_origin,
        "grid_spacing": grid_spacing,
        "grid_dims": grid_dims,
        "value_weights": value_weights,
        "gradient_weights": gradient_weights,
        "poly_coeffs": poly_coeffs,
    }
